import torch
import random
import numpy as np
from transformers import AutoTokenizer
import csv
import pandas as pd
import sys

csv.field_size_limit(sys.maxsize)

f = open('../msr_paraphrase_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
r = list(rdr)
f.close()

for _d in r:
    print(_d)
    debug = 0


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




