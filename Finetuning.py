import torch
import numpy as np
import argparse
from Models.BERT import ELECTRA_DISCRIMINATOR
from data_related.utils import Config
import os
from torch import nn
from transformers import AutoTokenizer
import random
from data_related.Custom_dataloader import FINE_TUNE_DATASET, FINE_TUNE_COLLATOR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy


def lr_scheduling(global_lr, layer_lrs, optimizer):

    for i in range(3):
        '''For enc_emb, pos_emb, and projection layers'''
        optimizer.param_groups[i]['lr'] = global_lr

    for idx, lr in enumerate(layer_lrs):
        optimizer.param_groups[idx+3]['lr'] = lr

    '''for downstream fc layer'''
    optimizer.param_groups[-1]['lr'] = global_lr


def get_layer_decayed_lrs(lrs, pct, warmup_steps, total_steps):
    layer_lrs = []
    for lr in lrs:
        layer_lrs += [linear_warmup_and_then_decay(pct=pct, lr_max=lr, total_steps=total_steps, warmup_steps=warmup_steps)]
    return layer_lrs


def make_param_groups(model, lrs, global_lr):
    param_groups = []
    """
    self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
    self.pos_emb = nn.Embedding(self.config.n_enc_seq + 1, self.config.d_hidn)

    self.embeddings_project = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
    """
    param_groups += [{"params": model.backbone.encoder.enc_emb.parameters(), "lr": global_lr}]
    param_groups += [{"params": model.backbone.encoder.pos_emb.parameters(), "lr": global_lr}]
    param_groups += [{"params": model.backbone.encoder.embeddings_project.parameters(), "lr": global_lr}]

    for idx, lr in enumerate(lrs):
        param_groups += [{"params": model.backbone.encoder.layers[idx].parameters(), "lr": lrs[idx]}]

    param_groups += [{"params": model.fc.parameters(), "lr": global_lr}]

    return param_groups


def get_layer_lrs(lr, decay_rate, num_hidden_layers=12):
    lrs = [lr * (decay_rate ** depth) for depth in range(num_hidden_layers)]
    return list(reversed(lrs))


def linear_warmup_and_then_decay(pct, lr_max, total_steps, warmup_steps=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    step_i = round(pct * total_steps)
    if step_i <= warmup_steps:  # warm up
        return lr_max * min(1.0, step_i/warmup_steps)
    else:  # decay
        return (lr_max-end_lr) * (1 - (step_i-warmup_steps)/(total_steps-warmup_steps)) ** decay_power + end_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    #  batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # 내림차순으로 각 logit 을 정렬하여 첫번째 리턴값을 구성하고 두번째 리턴값은 logit 의 indices
    pred = pred.t()  # batchsize X maxk -> maxk X batchsize
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target : 256 -> 1, 256 -> 5, 256 (복사된 형태)
    """
    Target                              비교    pred
    2,3,7,1.... 각 배치별 정답 indices     <==>  top 1 batch 별 indices
    2,3,7,1.... 각 배치별 정답 indices     <==>  top 2 batch 별 indices
    2,3,7,1.... 각 배치별 정답 indices     <==>  top 3 batch 별 indices
    2,3,7,1.... 각 배치별 정답 indices     <==>  top 4 batch 별 indices
    2,3,7,1.... 각 배치별 정답 indices     <==>  top 5 batch 별 indices
    """

    res = []
#     print(output)
#     print(target)
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k)  # value < batch size
    return res


class Evaluation:
    def __init__(self, task, dataloader, logging_dir, device):
        self.task = task
        self.dataloader = dataloader
        self.writer = SummaryWriter(log_dir=logging_dir)
        self.device = device
        """
        MatthewCorrCoef : CoLA
        - MCC=(TP×TN−FP×FN) / (sqrt( (TP+FP)(TP+FN)(TN+FP)(TN+FN) ))
        Range -1 ~ 1 , 1일 수록 두 관측치가 유사
        
        accuracy : SST-2, MNLI, QNLI, RTE, WNLI
        F1 Score : MRPC, QQP
        PearsonCorrCoef, SpearmanCorrCoef : STS-B
        """

    def cls_evaluation(self, model, cur_epoch, topk=(1,)):
        model.eval()

        acc_groups = {f"Epoch {cur_epoch}'s top-{k}": 0.0 for k in topk}
        num_total = 0

        for idx, batch in enumerate(self.dataloader):
            sentences, labels = batch
            sentences, labels = sentences.to(self.device), labels.to(self.device)
            num_total += labels.size(0)
            preds = model(sentences)
            accs = accuracy(preds, labels, topk)

            for idx, acc_key in enumerate(acc_groups):
                acc_groups[acc_key] += accs[idx]

        for idx, acc_key in enumerate(acc_groups):
            acc_groups[acc_key] /= num_total
            acc_groups[acc_key] *= 100

        self.writer.add_scalar(tag=f"{self.task} / Accuracy (%)",
                               scalar_value=acc_groups[f"Epoch {cur_epoch}'s top-1"],
                               global_step=cur_epoch)
        acc = acc_groups[f"Epoch {cur_epoch}'s top-1"] 
        print(f"Accuracy : {acc} %")

    def f1_eval(self, model, cur_epoch):
        model.eval()
        sample_cnt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        num_total = 0

        for idx, batch in enumerate(self.dataloader):
            sentences, labels = batch
            sentences, labels = sentences.to(self.device), labels.to(self.device)
            num_total += labels.size(0)
            preds = model(sentences)
            top_values, top_indices = preds.topk(1, 1)
            mask = labels.eq(top_indices.view(-1))
            tp = (mask * preds.view(-1)).sum()
            tn = mask.sum() - tp
            fp = (~mask * preds).sum()
            fn = (~mask).sum() - fp

            sample_cnt["TP"] += tp
            sample_cnt["TN"] += tn
            sample_cnt["FP"] += fp
            sample_cnt["FN"] += fn

        F1_SCORE = sample_cnt["TP"] / (sample_cnt["TP"] + 0.5 * (sample_cnt["FP"]+sample_cnt["FN"]))
        print(f"Current epoch : {cur_epoch}, F1 SCORE : {F1_SCORE}")
        self.writer.add_scalar(tag=f"{self.task} / F1-Score (%)",
                               scalar_value=F1_SCORE,
                               global_step=cur_epoch)

    def reg_evaluation(self, model, cur_epoch):
        model.eval()
        num_total = 0
        corr_total = 0.
        for idx, batch in enumerate(self.dataloader):
            sentences, labels = batch
            sentences, labels = sentences.to(self.device), labels.to(self.device)
            num_total += labels.size(0)
            preds = model(sentences)
            r, p = scipy.stats.pearsonr(preds.squeeze(), labels)
            corr_total += r
        avg_corr = (corr_total / num_total)
        print(f"Current epoch : {cur_epoch}, Average pearson correlation coefficient : {avg_corr}")
        self.writer.add_scalar(tag=f"{self.task} / Pearson Corr",
                               scalar_value=avg_corr,
                               global_step=cur_epoch)

    def task_wise_eval(self, model, cur_epoch):
        if self.task in ["SST-2", "MNLI", "QNLI", "RTE", "WNLI", "CoLA"]:
            self.cls_evaluation(model=model, cur_epoch=cur_epoch)
        elif self.task in ["MRPC", "QQP"]:
            self.f1_eval(model=model, cur_epoch=cur_epoch)
        elif self.task == "STS-B":
            self.reg_evaluation(model=model, cur_epoch=cur_epoch)
        else:
            raise Exception("It is not valid dataset for evaluation. Please check the dataset")


class Downstream_wrapper(nn.Module):
    def __init__(self, downstream_backbone, task, config):
        super(Downstream_wrapper, self).__init__()
        self.backbone = downstream_backbone
        self.task = task

        self.drop = nn.Dropout(0.1)

        if task in ["CoLA", "SST-2", "MRPC", "QQP", "QNLI", "RTE", "WNLI"]:
            num_cls = 2
        elif task in ["STS-B"]:
            num_cls = 1
        else:
            '''MNLI'''
            num_cls = 3

        self.fc = nn.Linear(config.n_head * config.d_head, num_cls)


    def forward(self, inputs):
        """
        :param inputs:
        :return:
        """
        outputs, _ = self.backbone(inputs)
        outputs = self.drop(outputs[:, 0, :])
        outputs = self.fc(outputs)
        return outputs.squeeze()


def fine_tuner(args):
    if not os.path.exists(args.task):
        os.mkdir(args.task)

    '''MANUAL SEED ALLOCATION'''
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    '''configuration of electra discriminator'''
    cfg = Config({"n_enc_vocab": 30522,  # correct
                  "n_enc_seq": 512,  # correct
                  "n_seg_type": 2,  # correct
                  "n_layer": 12,  # correct
                  "d_hidn": 128,  # correct
                  "i_pad": 0,  # correct
                  "d_ff": 1024,  # correct
                  "n_head": 4,  # correct
                  "d_head": 64,  # correct
                  "dropout": 0.1,  # correct
                  "layer_norm_epsilon": 1e-12  # correct
                 })

    ED = ELECTRA_DISCRIMINATOR(config=cfg)
    pretrain_checkpoint = torch.load(args.pretrained_model_weight_path)
    ED.load_state_dict(pretrain_checkpoint["state_dict"])
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    downstream_Backbone = ED.bert
    model = Downstream_wrapper(downstream_backbone=downstream_Backbone, task=args.task, config=cfg)
    model = model.to(args.device)

    train_set = FINE_TUNE_DATASET(task=args.task, mode='train', root_dir=args.data_root_dir)
    test_set = FINE_TUNE_DATASET(task=args.task, mode='test', root_dir=args.data_root_dir)

    collator_fn = FINE_TUNE_COLLATOR(tokenizer=tokenizer)

    Train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collator_fn, drop_last=True)

    Test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collator_fn, drop_last=True)

    lrs = get_layer_lrs(lr=args.lr, decay_rate=0.8, num_hidden_layers=12)
    param_groups = make_param_groups(model=model, lrs=lrs, global_lr=args.lr)
    optimizer = torch.optim.Adam(param_groups)

    loss_func = nn.MSELoss() if args.task == "STS-B" else nn.CrossEntropyLoss()
    Evaluator = Evaluation(task=args.task, dataloader=Test_loader, logging_dir=args.logging_dir, device=args.device)
    train_iter = 0
    total_iter = Train_loader.__len__() * args.epochs
    total_warm_up = int(args.warmup_fraction * total_iter)
    for epoch in range(args.epochs):
        for idx, data in enumerate(Train_loader):
            sentences, labels = data  # two sentences are grafted with [SEP].
            sentences, labels = sentences.to(args.device), labels.to(args.device)
            """
            Learning rate scheduling
            """
            pct = train_iter / total_iter
            global_lr = linear_warmup_and_then_decay(pct=pct, lr_max=args.lr, total_steps=total_iter,
                                                     warmup_steps=total_warm_up, end_lr=0.0, decay_power=1)

            layer_lrs = get_layer_decayed_lrs(lrs=lrs, pct=pct,
                                              warmup_steps=total_warm_up,
                                              total_steps=total_iter)

            lr_scheduling(global_lr=global_lr, layer_lrs=layer_lrs, optimizer=optimizer)

            optimizer.zero_grad()

            outputs = model(sentences)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if train_iter % 50 ==0:
                    print(f"{epoch} / {args.epochs}, {idx+1} / {Train_loader.__len__()}, Train Loss : {loss.item()}")
            train_iter += 1

        if (epoch + 1) % args.eval_period == 0:
            Evaluator.task_wise_eval(model=model, cur_epoch=epoch)

    print("End of fine-tuning")
    Evaluator.writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_weight_path", 
                        type=str, 
                        default="/vision/7032593/NLP/ELECTRA/check_points/DISC_ITER_190000_LM_MODEL.pth")
    parser.add_argument("--task", type=str, default="CoLA",
                        choices=["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI"])
    parser.add_argument("--data_root_dir", type=str, default="/vision/7032593/NLP/GLUE-baselines/glue_data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_root_dir", type=str, default="./save")
    parser.add_argument("--warmup_fraction", type=int, default=0.1)
    parser.add_argument("--logging_dir", type=str, default="./finetune_logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_period", type=int, default=1, help="epoch unit")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    fine_tuner(args)
    """
    Hyperparameter GLUE Value
    Learning Rate 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
    Adam eps 1e-6 v
    Adam β1 0.9 v
    Adam β2 0.999 v
    Layerwise LR decay 0.8 for Base/Small, 0.9 for Large
    Learning rate decay Linear
    Warmup fraction 0.1
    Attention Dropout 0.1
    Dropout 0.1
    Weight Decay 0
    Batch Size 32
    Train Epochs 10 for RTE and STS, 2 for SQuAD, 3 for other tasks
    """