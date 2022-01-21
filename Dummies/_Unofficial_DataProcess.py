import csv
import pandas as pd
import os

root = "/Users/hmc/Desktop/NLP_DATA"
data_path = os.path.join(root, "train.tsv")

f = open(data_path, 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
data = list(rdr)[1:]  # 범주가 첫 행에 있으므로 해당 내용 제거
f.close()
for d in data:
    if len(d) != 4:
        print(d)
        print(len(d))
# train_path = os.path.join(root, "train.tsv")
# test_path = os.path.join(root, "test.tsv")

# f = open(train_path, "r")
# data = f.readlines()
# _rep_data = {"Quality": [], "#1 ID": [], "#2 ID": [], "#1 String": [], "#2 String": []}
# for idx, d in enumerate(data[1:]):
#
#     q, _id1, _id_2, sentence_1, sentence_2 = d.split("\t")
#     _rep_data["Quality"] += [q]
#     _rep_data["#1 ID"] += [_id1]
#     _rep_data["#2 ID"] += [_id_2]
#     _rep_data["#1 String"] += [sentence_1]
#     _rep_data["#2 String"] += [sentence_2.strip("\n")]
#
# df = pd.DataFrame(_rep_data)
# df.to_csv(root+"/train.csv")
# train_path = os.path.join(root, "train.csv")
# with open(train_path, newline='') as csv_f:
#     rows = list(csv.reader(csv_f))
#     for row in rows:
#         print(row)
#         debug=0