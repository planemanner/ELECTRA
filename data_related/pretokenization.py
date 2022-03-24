from transformers import ElectraTokenizerFast
from tqdm import tqdm
from Custom_dataloader import LM_dataset
import json
from collections import defaultdict
import json
import os


def pre_tokenization(d_pathes, result_path, max_len):
    dataset = LM_dataset(d_pathes=d_pathes)
    dict_dataset = {'token': [], 'attention_mask': []}
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-generator")
    """
    dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    """
    for sp in tqdm(dataset.data):
        dict_info = tokenizer(sp, padding=True, truncation=True, max_length=max_len)
        token = dict_info['input_ids']
        attn_mask = dict_info['attention_mask']
        dict_dataset['token'] += [token]
        dict_dataset['attention_mask'] += [attn_mask]

    json.dump(dict_dataset, result_path, indent=4)


# root_dir = "/Users/hmc/Desktop/NLP_DATA"
# out_path = "/Users/hmc/Desktop/NLP_DATA/pretokenized_LM_dataset.json"
# d_path_1 = os.path.join(root_dir, "bookcorpus.lm.txt")
# d_path_2 = os.path.join(root_dir, "enwiki_210901.lm.txt")
#
# d_pathes = [d_path_1, d_path_2]
# pre_tokenization(d_pathes, result_path=out_path, max_len=512)
