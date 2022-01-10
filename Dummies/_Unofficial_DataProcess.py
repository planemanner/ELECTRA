import torch
import random
import numpy as np
from transformers import AutoTokenizer
import csv
import pandas as pd
import sys

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

sentence = ["High and High go and way of you [SEP]",
            "Hey, man what are you doing?",
            "What the...hell!",
            "you should do that more and more and more and more.",
            "What kind of words are remained as naive?"]

tk = tokenizer(sentence, padding=False, truncation=True)

print(tk["input_ids"])


# f = open("../msr_paraphrase_train.txt", 'r', encoding='utf-8')
# data = f.readlines()
# for _d in data:
#     print(_d)
#     debug=0

# scores = []
#
# for _d in r[1:]:
#     # scores += [float(_d[-1].strip("\n"))]
#     score = _d[-1].strip("\n")
#     scores += [float(score)]
#     # print(f" Score : {score}, Length : {len(score)}")
#     # debug=0
# print(np.min(scores))




