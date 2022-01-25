import torch
import numpy as np

bs = 4
n_seq = 3
n_voca = 1000
sample_logit = torch.randn(bs, n_seq, n_voca)
sample_labels = torch.randn(bs, n_seq)
sampling_from_list_of_list = [[True, True, False], [True, False, False], [False, True, False], [True, False, False]]
output = sample_logit[sampling_from_list_of_list, :]
label_output = sample_labels[sampling_from_list_of_list, :]
print(output.shape)
print(label_output.shape)
torch.nn.CrossEntropyLoss()


def sampler(Dist, logits, device):
    Gumbel = Dist.sample(logits.shape).to(device)
    return (logits.float() + Gumbel).argmax(dim=-1)


sample_dist = torch.distributions.gumbel.Gumbel(0, 1)
device = 'cpu'

output = sampler(sample_dist, sample_logit, device)
print(output.shape)
tt = torch.tensor(sampling_from_list_of_list).long()
print(tt)
print(15/11*16)