# -*- coding: utf-8 -*-
import os
import torch
import yaml
from Pruning.Pruning_utils import arch_generator, FP_GETTER, get_module, get_indices, stage_grouper, replace_layer, get_thin_params, accelerate
import numpy as np
from data_related.utils import Config
from data_related.Custom_dataloader import FINE_TUNE_COLLATOR, FINE_TUNE_DATASET
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from Models.BERT import BERT, ELECTRA_DISCRIMINATOR
from copy import deepcopy
from Models.BasicModules import get_attn_pad_mask
from thop import profile
from fvcore.nn import FlopCountAnalysis
import random


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NLP_Pruner:
    def __init__(self, model, forward_pass_info_path):
        self.model = model
        self.archs = arch_generator(model)
        self.layer_wise_in_scores = {}
        self.layer_wise_out_scores = {}
        self.percentiler = []
        self.scores = {}
        self.forward_pass_info_path = forward_pass_info_path
        self.pruned_out_filters = {}
        self.pruned_in_filters = {}
        FP = FP_GETTER(model=model, arch_type='BERT', yaml_path=forward_pass_info_path)
        FP.GET_FP()
        self.FP_DATA = None
        self.test_sample = None
        self.get_attn_mask = get_attn_pad_mask()
        print(count_parameters(model))
    def get_individual_scores(self, dataloader, device='cpu'):

        data = next(iter(dataloader))
        seqs, labels = data
#         attn_mask = self.get_attn_mask(seqs, seqs, 0)
        self.test_sample = seqs

#         flops = FlopCountAnalysis(self.model, (seqs, attn_mask))
#         print(flops.total())
#         base_macs, base_params = profile(self.model, inputs=(seqs, attn_mask))
#         print(f"base macs:{base_macs}, base params:{base_params}")

        output = self.model(seqs.to(device))
        loss = torch.sum(output)
        loss.backward()
        
        with open(self.forward_pass_info_path, 'r') as _f:
            self.FP_DATA = yaml.load(_f, yaml.FullLoader)
        _f.close()

        for arch_name, module_name in self.FP_DATA.items():
            tmp_module = get_module(self.model, arch_name.split('.'))

            param = list(tmp_module.parameters())[0]

            self.scores[arch_name] = torch.clone(param.grad * param).detach().abs_()
            param.grad.data.zero_()

    def layer_scoring(self):
        for arch_name in self.scores:
            cur_module = get_module(self.model, arch_name.split('.'))
            if isinstance(cur_module, torch.nn.Linear) or isinstance(cur_module, torch.nn.Conv1d):

                if isinstance(cur_module, torch.nn.Linear):
                    self.layer_wise_out_scores[arch_name] = self.scores[arch_name].sum(dim=1)
                    self.layer_wise_in_scores[arch_name] = self.scores[arch_name].sum(dim=0)
                    self.percentiler += [*self.layer_wise_out_scores[arch_name]]
                    
                elif isinstance(cur_module, torch.nn.Conv1d):
                    self.layer_wise_out_scores[arch_name] = self.scores[arch_name].sum(dim=(1, 2))
                    self.layer_wise_in_scores[arch_name] = self.scores[arch_name].sum(dim=(0, 2))
                    self.percentiler += [*self.layer_wise_out_scores[arch_name]]

            elif isinstance(cur_module, torch.nn.LayerNorm):
                self.layer_wise_in_scores[arch_name] = self.scores[arch_name]

    def block_pruner(self, input_dim, target_block, threshold, n_head=4, min_channel_ratio=0.2):

        tmp_num_filters = []
        for idx, module_name in enumerate(target_block):
            score_vector_out = self.layer_wise_out_scores[module_name]
            num_remain = (score_vector_out >= threshold).sum()
            minima = int(min_channel_ratio * score_vector_out.shape[0])

            if num_remain < minima:
                num_remain = minima
            residue = (num_remain % n_head)
            num_remain -= residue
            tmp_num_filters += [num_remain]

            if idx == 2:
                break
                
        num_decision = np.max(tmp_num_filters)
        last_input_dim = input_dim
        pos_cnt = 0
        
        for idx, module_name in enumerate(target_block):

            if idx < 3:
                score_vector_out = self.layer_wise_out_scores[module_name]
                score_vector_in = self.layer_wise_in_scores[module_name]
                self.pruned_out_filters[module_name] = get_indices(score_vector_out, num_remain=num_decision)
                self.pruned_in_filters[module_name] = get_indices(score_vector_in, num_remain=last_input_dim)

            elif idx == 3:
                score_vector_out = self.layer_wise_out_scores[module_name]
                score_vector_in = self.layer_wise_in_scores[module_name]
                self.pruned_out_filters[module_name] = get_indices(score_vector_out, num_remain=last_input_dim)
                self.pruned_in_filters[module_name] = get_indices(score_vector_in, num_remain=num_decision)
            else:
                m = get_module(self.model, module_name.split('.'))
                if isinstance(m, torch.nn.LayerNorm):
                    score_vector_in = self.layer_wise_in_scores[module_name]
                    self.pruned_in_filters[module_name] = get_indices(score_vector_in, num_remain=last_input_dim)

                else:

                    if pos_cnt == 0:
                        score_vector_out = self.layer_wise_out_scores[module_name]
                        score_vector_in = self.layer_wise_in_scores[module_name]
                        minima = int(min_channel_ratio * score_vector_out.shape[0])
                        num_remain = (score_vector_out >= threshold).sum()

                        if num_remain < minima:
                            num_remain = minima

                        self.pruned_in_filters[module_name] = get_indices(score_vector_in, num_remain=last_input_dim)
                        self.pruned_out_filters[module_name] = get_indices(score_vector_out, num_remain=num_remain)

                        intermediate_dim = num_remain
                        pos_cnt += 1

                    else:
                        score_vector_out = self.layer_wise_out_scores[module_name]
                        score_vector_in = self.layer_wise_in_scores[module_name]
                        self.pruned_in_filters[module_name] = get_indices(score_vector_in, num_remain=intermediate_dim)
                        self.pruned_out_filters[module_name] = get_indices(score_vector_out, num_remain=last_input_dim)

        return last_input_dim

    def bert_pruner(self, threshold_rank, min_channel_ratio, n_head):
        """
        threshold_rank: 0 ~ 100
        """
        '''Embedding layer pruning'''
        threshold_score = np.percentile(self.percentiler, threshold_rank)

        GROUPS = stage_grouper(self.forward_pass_info_path)

        for idx, (arch_name, module_name) in enumerate(self.FP_DATA.items()):
            m = get_module(self.model, arch_name.split('.'))
            if isinstance(m, torch.nn.Embedding):
                '''Embedding is not a target of pruning'''
                num_tokens, d_model = m.weight.shape
            else:
                if idx == 3:
                    '''until embedding projection'''
                    break
                else:
                    score_vector_in = self.layer_wise_in_scores[arch_name]
                    score_vector_out = self.layer_wise_out_scores[arch_name]
                    
                    self.pruned_in_filters[arch_name] = get_indices(score_vector_in, num_remain=score_vector_in.shape[0])
                    num_remain = (score_vector_out >= threshold_score).sum()
                    minima = int(min_channel_ratio * score_vector_out.shape[0])
                    
                    if num_remain < minima:
                        num_remain = minima
                    if num_remain % 4 == 0:
                        pass
                    else:
                        num_remain -= (num_remain % 4)                         
                    
                    self.pruned_out_filters[arch_name] = get_indices(score_vector_out, num_remain=num_remain)

        first_block_input_dim = num_remain
        '''block prune'''
        for module_list in GROUPS["Encoder_Block"]:

            first_block_input_dim = self.block_pruner(
                input_dim=first_block_input_dim, target_block=module_list,
                threshold=threshold_score, n_head=n_head, min_channel_ratio=min_channel_ratio)


    def Exec_prune(self, min_layer, threshold_rank, dataloader, device, n_head):
        self.get_individual_scores(dataloader, device)
        self.layer_scoring()
        self.bert_pruner(threshold_rank, min_layer, n_head)

    def acceleration(self, min_layer, threshold_rank, dataloader, save_dir, model_name, device, n_head):
        self.Exec_prune(min_layer, threshold_rank, dataloader, device, n_head)
        for module_name in self.archs:
            args = module_name.split('.')
            cur_module = get_module(self.model, args)

            if isinstance(cur_module, torch.nn.Conv1d) or isinstance(cur_module, torch.nn.Linear):
                new_weight, new_bias = get_thin_params(cur_module, self.pruned_out_filters[module_name])

                new_layer = replace_layer(old_layer=cur_module, rep_weight=new_weight,
                                          rep_bias=new_bias,
                                          in_channel_indices=self.pruned_in_filters[module_name])

                accelerate(self.model, args, new_layer)

            elif isinstance(cur_module, torch.nn.LayerNorm):

                new_weight, new_bias = get_thin_params(cur_module, self.pruned_in_filters[module_name])
                new_layer = replace_layer(old_layer=cur_module,
                                          rep_weight=new_weight,
                                          rep_bias=new_bias)

                accelerate(self.model, args, new_layer)

        model_path = os.path.join(save_dir, f"{model_name}.pth")
        self.model.eval()
        """
        test
        """
        seqs = self.test_sample

#         print(count_parameters(self.model))
        pruned_macs, pruned_params = profile(self.model, inputs=(seqs, ))
        print(f"pruned macs:{pruned_macs}, pruned params:{pruned_params}")
#         flops = FlopCountAnalysis(self.model, (seqs, attn_mask))
#         print(flops.total())
#         print(count_parameters(model))
        print(self.model)
    
        with torch.no_grad():
            torch.save(self.model, model_path)


# os.environ["TOKENIZERS_PARALLELISM"] = "true"
if __name__ == "__main__":
    
    cfg = Config({"n_enc_vocab": 30522,  # correct
                    "n_enc_seq": 512,  # correct
                    "n_seg_type": 2,  # correct
                    "n_layer": 12,  # correct
                    "d_model": 128,  # correct
                    "i_pad": 0,  # correct
                    "d_ff": 1024,  # correct
                    "n_head": 4,  # correct
                    "d_head": 64,  # correct
                    "dropout": 0.1,  # correct
                    "layer_norm_epsilon": 1e-12  # correct
                    })
    
    seed = 4
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    batch_size = 32
    num_workers = 8
    score_rank = 99
    
    data_root_dir = "/vision/7032593/NLP/GLUE-baselines/glue_data"
    tokenizer_path = "/vision/7032593/NLP/ELECTRA/tokenizer_files"
    weight_path = "/vision2/7032593/ELECTRA/check_points/DISC_ITER_020000_LM_MODEL.pth"
    fp_info_path = "./BERT.yaml"
    
    task = "RTE"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    with torch.no_grad():
        ED = ELECTRA_DISCRIMINATOR(cfg, device='cpu')
        ED.load_state_dict(torch.load(weight_path)["state_dict"])
    model = ED.bert
    
    if os.path.isfile(fp_info_path):
        os.remove(fp_info_path)
        
    train_set = FINE_TUNE_DATASET(task=task, mode='train', root_dir=data_root_dir)
    collator_fn = FINE_TUNE_COLLATOR(tokenizer=tokenizer)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, collate_fn=collator_fn, drop_last=True)

    PRUNER = NLP_Pruner(model=model, forward_pass_info_path=fp_info_path)
    PRUNER.acceleration(min_layer=0.1, threshold_rank=score_rank, dataloader=train_loader,
                        save_dir="./", model_name=f"PRUNED_BERT_{score_rank}_SEED_{seed}", device='cpu', n_head=cfg.n_head)
