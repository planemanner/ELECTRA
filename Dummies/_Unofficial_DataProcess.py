import csv

data_path = "/Users/hmc/Desktop/NLP_DATA/dev.tsv"
f = open(data_path, 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
data = list(rdr)[1:]  # 범주가 첫 행에 있으므로 해당 내용 제거
print(data)

f.close()