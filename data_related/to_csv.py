import csv
import pandas as pd
import os
import torch



def conversion(path, tgt_dir, mode):
    _rep_data = {"sentence1": [], "sentence2": [], "label": []}
    with open(path, newline='') as tsv_f:
        rows = list(csv.reader(tsv_f, delimiter='\t'))
        for row in rows:
            if len(row) != 4:
                _id, sen1, label = row
                s1, s2 = sen1.split("\t")
                _rep_data["sentence1"] += [s1]
                _rep_data["sentence2"] += [s2]
                _rep_data["label"] += [label]
            else:
                _rep_data["sentence1"] += [row[1]]
                _rep_data["sentence2"] += [row[2]]
                _rep_data["label"] += [row[3]]

    tsv_f.close()
    df = pd.DataFrame(_rep_data)
    tgt_path = os.path.join(tgt_dir, f"{mode}.csv")
    df.to_csv(tgt_path)

#
# root_dir = "/Users/hmc/Desktop/NLP_DATA"
#
# path = os.path.join(root_dir, "test.tsv")
# conversion(path,tgt_dir=root_dir, mode='test')