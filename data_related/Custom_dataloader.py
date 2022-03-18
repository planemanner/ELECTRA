from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import torch
import csv
import sys
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
csv.field_size_limit(sys.maxsize)


class LM_dataset(Dataset):
    def __init__(self, d_pathes):
        super(LM_dataset, self).__init__()
        self.data = []
        for d_path in d_pathes:
            _ft = open(d_path, "r")
            self.data += _ft.readlines()
            _ft.close()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class LM_collater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        dict_info = self.tokenizer(batch, padding=True, truncation=True)

        return torch.as_tensor(data=dict_info["input_ids"], dtype=torch.long)


class FINE_TUNE_DATASET(Dataset):
    def __init__(self, task, mode, root_dir):
        super(FINE_TUNE_DATASET, self).__init__()
        mode = mode.lower()
        assert mode in ["train", "test"], "You must choose train or test as a mode"
        self.task = task
        self.data = self.load_dataset(task, mode, root_dir)
        self.mode = mode

    def load_dataset(self, task, mode, root_dir):
        if task != "MNLI" or mode != "test":
            if task == "MRPC" or task== "RTE":
                data_path = os.path.join(root_dir, task, f"{mode}.csv")
                with open(data_path, newline='') as csv_f:
                    rows = list(csv.reader(csv_f))
                    data = rows[1:]
                csv_f.close()
            else:
                data_path = os.path.join(root_dir, task, f"{mode}.tsv")

                f = open(data_path, 'r', encoding='utf-8')
                rdr = csv.reader(f, delimiter='\t')
                data = list(rdr)[1:]  # 범주가 첫 행에 있으므로 해당 내용 제거
                f.close()
            print(data_path)
        else:
            """
            In the case of MNLI, the test datasets are consisted of two types (matched, mismatched domains).
            """
            data_path_1 = os.path.join(root_dir, task, f"{mode}_mismatched.tsv")
            data_path_2 = os.path.join(root_dir, task, f"{mode}_matched.tsv")

            f_1 = open(data_path_1, 'r', encoding='utf-8')
            rdr = csv.reader(f_1, delimiter='\t')
            data_1 = list(rdr)[1:]
            f_1.close()

            f_2 = open(data_path_2, 'r', encoding='utf-8')
            rdr = csv.reader(f_2, delimiter='\t')
            data_2 = list(rdr)[1:]
            f_2.close()
            data = data_1 + data_2  # list of lists
            print(data_path_1)
            print(data_path_2)
        return data

    def __getitem__(self, idx):
        if self.task == "CoLA" :
            _, label, _, sentence = self.data[idx]
            return sentence, int(label)

        elif self.task == "SST-2":
            sentence, label = self.data[idx]  # 0 : negative, 1: positive
            return sentence, int(label)

        elif self.task == "MRPC":

            _index, label, sentence1_id, sentence2_id, sentence_1, sentence_2 = self.data[idx]
            return sentence_1 + "[SEP]" + sentence_2, int(label)

        elif self.task == "QQP":
            _, _, _, question_1, question_2, label = self.data[idx]  # 0 : not similar, 1: similar
            return question_1 + "[SEP]" + question_2, int(label)

        elif self.task == "STS-B":
            sentence_1, sentence_2, similarity = self.data[idx][-3], self.data[idx][-2], self.data[idx][-1]
            return sentence_1 + "[SEP]" + sentence_2, float(similarity)

        elif self.task == "MNLI":
            """
            Note :
            contradiction : 0
            entailment : 1
            neutral : 2
            """
            sentence_1, sentence_2, label = self.data[idx][-4], self.data[idx][-3], self.data[idx][-1]
            if label.lower() == "neutral":
                long_type_label = 0
            elif label.lower() == "entailment":
                long_type_label = 1
            else:
                long_type_label = 2
            return sentence_1 + "[SEP]" + sentence_2, long_type_label

        elif self.task == "QNLI":
            """
            Note:
            not_entailment : 0
            entailment : 1
            """
            question, answer, label = self.data[idx][1], self.data[idx][2], self.data[idx][3]
            if label.lower() == "entailment":
                long_type_label = 1
            else:
                long_type_label = 0
            return question + "[SEP]" + answer, long_type_label

        elif self.task == "RTE":
            """
            Note:
            not_entailment : 0
            entailment : 1
            """
            sample = self.data[idx]

            question, answer, label = self.data[idx][1], self.data[idx][2], self.data[idx][3]
#             print(label)
            if label.lower() == "entailment":
                long_type_label = 1
            else:
                long_type_label = 0
            return question + "[SEP]" + answer, long_type_label

        elif self.task == "WNLI":
            information, query, label = self.data[idx][1], self.data[idx][2], self.data[idx][3]
            return information + "[SEP]" + query, label
        else:
            raise Exception("Please check the dataset name")

    def __len__(self):
        return len(self.data)


class FINE_TUNE_COLLATOR:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sentences = []
        labels = []
        for data in batch:
            sentences += [data[0]]
            labels += [data[1]]
        dict_info = self.tokenizer(sentences, padding=True, truncation=True)
        return torch.as_tensor(data=dict_info["input_ids"], dtype=torch.long), torch.as_tensor(labels)




