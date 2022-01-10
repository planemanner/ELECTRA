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


class Evaluation:
    def __init__(self, task, dataloader, logging_dir):
        """
        MatthewCorrCoef : CoLA
        accuracy : SST-2, MNLI, QNLI, RTE, WNLI
        F1 Score : MRPC, QQP
        PearsonCorrCoef, SpearmanCorrCoef : STS-B
        """
        self.task = task
        self.dataloader = dataloader
        self.logging_dir = logging_dir

    def evaluation(self):
        """
        --------------
        | 결과 Plotting|
        --------------
        """


class Downstream_wrapper(nn.Module):
    def __init__(self, downstream_backbone, task, config):
        super(Downstream_wrapper, self).__init__()
        self.backbone = downstream_backbone
        self.task = task

        self.drop = nn.Dropout(0.1)
        # tasks = ["CoLA", "SST-2", "MRPC", "QQP",
        #  "STS-B", "MNLI", "QNLI", "RTE", "WNLI"]
        if task in ["CoLA", "SST-2", "MRPC", "QQP", "QNLI", "RTE", "WNLI"]:
            num_cls = 2
        elif task in ["STS-B"]:
            num_cls = 1
        else:
            '''MNLI'''
            num_cls = 3

        self.fc = nn.Linear(config.d_hidn, num_cls)

    def forward(self, inputs):
        """
        :param inputs:
        :return:
        """
        outputs, _, _ = self.backbone(inputs)
        outputs = self.drop(outputs)
        outputs = self.fc(outputs)
        return outputs


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
                  "n_seg_type": 2,
                  "n_layer": 12,
                  "d_hidn": 128,  # correct
                  "i_pad": 0,
                  "d_ff": 256,
                  "n_head": 4,
                  "d_head": 16,
                  "dropout": 0.1,
                  "layer_norm_epsilon": 1e-12
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
                              shuffle=True, num_workers=args.num_workers, collate_fn=collator_fn)

    Test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, collate_fn=collator_fn)

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)

    loss_func = nn.MSELoss() if args.task == "STS-B" else nn.CrossEntropyLoss()
    Evaluator = Evaluation(task=args.task, dataloader=Test_loader)

    for epoch in range(args.epochs):
        for idx, data in enumerate(Train_loader):
            sentences, labels = data  # two sentences are grafted with [SEP].
            sentences, labels = sentences.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            outputs = model(sentences)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"{epoch} / {args.epochs}, {idx+1} / {Train_loader.__len__()}, Train Loss : {loss.item()}")

        if (epoch + 1) % args.eval_period == 0:
            Evaluator.evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_weight_path", type=str, default="")
    parser.add_argument("--task", type=str, default="CoLA",
                        choices=["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI"])
    parser.add_argument("--data_root_dir", type=str, default="")
    parser.add_argument("--pretrained_model_weight_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_root_dir", type=str, default="")
    parser.add_argument("--logging_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_period", type=int, default=1, help="epoch unit")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=12)

    args = parser.parse_args()
    fine_tuner(args)
