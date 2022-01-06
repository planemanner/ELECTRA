import torch
import numpy as np
import argparse
from Models.BERT import ELECTRA_DISCRIMINATOR
from data_related.utils import Config
import os
from torch import nn


class Downstream_wrapper(nn.Module):
    def __init__(self, downstream_backbone, task):
        super(Downstream_wrapper, self).__init__()
        self.backbone = downstream_backbone
        self.task = task


def fine_tuner(args):
    if not os.path.exists(args.task):
        os.mkdir(args.task)

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
    ED.to(args.device)

    Downstream_Backbone = ED.bert


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_weight_path", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--dataset_root_dir", type=str, default="")
    parser.add_argument("--pretrained_model_weight_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_root_dir", type=str, default="")
    parser.add_argument("--logging_dir", type=str, default="")
    args = parser.parse_args()
    fine_tuner(args)
