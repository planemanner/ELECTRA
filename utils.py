# -*- coding: utf-8 -*-
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
from scipy import stats
from enum import Enum


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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
    param_groups += [{"params": model.backbone.encoder.token_embedding.parameters(), "lr": global_lr}]
    param_groups += [{"params": model.backbone.encoder.pos_embedding.parameters(), "lr": global_lr}]
    param_groups += [{"params": model.backbone.encoder.intermediate.parameters(), "lr": global_lr}]

    for idx, lr in enumerate(lrs):
        param_groups += [{"params": model.backbone.encoder.layers[idx].parameters(), "lr": lrs[idx]}]

    param_groups += [{"params": model.fc.parameters(), "lr": global_lr * lrs[idx] * 0.8}]

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
    maxk = max(topk)
    batch_size = target.size(0)
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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))  # value < batch size
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
            bs = labels.size(0)
            num_total += bs
            preds = model(sentences)
            batch_accs = accuracy(preds, labels, topk)

            for idx, acc_key in enumerate(acc_groups):
                acc_groups[acc_key] += batch_accs[idx] * bs  #  accumulation

        for idx, acc_key in enumerate(acc_groups):
            acc_groups[acc_key] /= num_total

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
            mask = labels.eq(top_indices.view(-1)) # 정답 마스크
            tp = (mask * top_indices.view(-1)).sum()
            tn = mask.sum() - tp
            fp = (~mask * top_indices.view(-1)).sum()
            fn = (~mask).sum() - fp

            sample_cnt["TP"] += tp
            sample_cnt["TN"] += tn
            sample_cnt["FP"] += fp
            sample_cnt["FN"] += fn

        F1_SCORE = sample_cnt["TP"] / (sample_cnt["TP"] + 0.5 * (sample_cnt["FP"]+sample_cnt["FN"]))
        print(f"Current epoch : {cur_epoch}, F1 SCORE : {F1_SCORE * 100}")
        self.writer.add_scalar(tag=f"{self.task} / F1-Score (%)",
                               scalar_value=F1_SCORE,
                               global_step=cur_epoch)

    def reg_evaluation(self, model, cur_epoch):
        model.eval()
        num_total = 0
        total_pearson = []
        total_spearman = []
        for idx, batch in enumerate(self.dataloader):
            sentences, labels = batch
            sentences, labels = sentences.to(self.device), labels.to(self.device)
            preds = model(sentences)
            total_pearson += [stats.pearsonr(preds.cpu().reshape(-1), labels.reshape(-1))[0]]
            total_spearman += [stats.spearmanr(preds.cpu().reshape(-1), labels.reshape(-1))[0]]
        
        self.writer.add_scalar(tag=f"{self.task} / Pearson",
                               scalar_value=np.mean(total_pearson),
                               global_step=cur_epoch)
        
        self.writer.add_scalar(tag=f"{self.task} / Spearman",
                               scalar_value=np.mean(total_spearman),
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
        self.activation = None
        
        if task in ["CoLA", "SST-2", "MRPC", "QQP", "QNLI", "RTE", "WNLI"]:
            num_cls = 2
        elif task in ["STS-B"]:
            num_cls = 1
            self.activation = torch.nn.Sigmoid()
        else:
            '''MNLI'''
            num_cls = 3

        self.fc = nn.Linear(config.n_head * config.d_head, num_cls)
        nn.init.xavier_uniform_(self.fc.weight.data, gain=1)
        self.fc.bias.data.zero_()


    def forward(self, inputs):
        """
        :param inputs:
        :return:
        """
        outputs = self.backbone(inputs)
        outputs = self.drop(outputs.max(dim=1)[0])
        outputs = self.fc(outputs)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs.squeeze()