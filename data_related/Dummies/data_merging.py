import os
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
import json


def merger(src_1, src_2, tgt):
    with open(src_1, "r") as f_1:
        data_1 = f_1.readlines()
    f_1.close()

    with open(src_2, "r") as f_2:
        data_2 = f_2.readlines()
    f_2.close()

    with open(tgt, "a") as t:
        print("First file")
        for d_1 in tqdm(data_1):
            t.write(d_1)
        print("Second file")
        for d_2 in tqdm(data_2):
            t.write(d_2)
    t.close()


root_dir = "/Users/hmc/Desktop/NLP_DATA"
output_name = "merged_data.json"
files = glob(os.path.join(root_dir, "*.txt"))

# merger(src_1=files[0], src_2=files[1], tgt=os.path.join(root_dir, "merged_lm.txt"))

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")


with open(os.path.join(root_dir, "merged_lm.txt"), 'r') as p:
    data = p.readlines()
p.close()

tokenized_data = {"token": None, "length": None}

for idx, seq in tqdm(enumerate(data)):
    tmp_dict_data = tokenizer(seq, truncation=True)
    tokenized_data[idx] = {}
    tokenized_data[idx]["token"] = tmp_dict_data["input_ids"]
    tokenized_data[idx]["length"] = len(tmp_dict_data["input_ids"])

with open(os.path.join(root_dir, output_name), 'w') as merged_file:
    json.dump(tokenized_data, merged_file, indent=4)
merged_file.close()
"""
총 74,653,391 개의 문장
"""
# _dict_type_info = tokenizer(data[:3], padding=True, truncation=True, return_tensors="pt")
# print(_dict_type_info["input_ids"])
# debug = 0