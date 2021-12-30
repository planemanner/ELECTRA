import torch
import random
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_input = tokenizer("Hello, I'm [MASK] single sentence!")
print(tokenizer.vocab)
print(tokenizer.decode(encoded_input["input_ids"]))
# print(tokenizer.decode(0))
# print(tokenizer.decode(1))
# print(tokenizer.decode(2))
# print(tokenizer.decode(3))
# #
# def masking_seq(seq, mask_ratio=0.15):
#     len_with_pad = len(seq)
#     seq_len = len_with_pad - (seq.eq(0).sum() + 2)  # sos, eos is denoted by 2 and pad is the other
#     masking_list = []
#     mask_size = int(seq_len * mask_ratio)
#     for _ in range(mask_size):
#         masking_list += [random.randint(1, (seq_len-1))]
#     seq[masking_list] = 3
#
#     return seq
#
#
# def batch_wise_masking(tokens, mask_ratio=0.163):
#     # mask_ratio is empirically determined by examining thousand times to meet 15 % in every iteration
#     masked_outputs = []
#     for tok in tokens:
#         masked_outputs += [masking_seq(tok, mask_ratio)]  # Tensor format
#     return torch.stack(masked_outputs)
#
#
# BS, NUM_POS, NUM_VOCAB = 128, 512, 1000
#
# sample = torch.randint(low=103, high=1000, size=(BS, NUM_POS))
# masked_output = batch_wise_masking(sample)
# accumulated_mask_percentage = []
# for tmp_output in masked_output:
#     num_mask = sum(tmp_output == 3)
#     mask_ratio = num_mask/NUM_POS*100
#     accumulated_mask_percentage += [mask_ratio]
#     print(f"The number of mask is : {num_mask}. \n The ratio of masking is : {mask_ratio} %")
# print(f"Average Masking Ratio is {np.mean(accumulated_mask_percentage)} %")