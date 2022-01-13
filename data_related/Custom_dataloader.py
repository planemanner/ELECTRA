from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LM_dataset(Dataset):
    def __init__(self, d_path):
        super(LM_dataset, self).__init__()
        assert "txt" in d_path, "You must put in a file having txt extension"
        _ft = open(d_path, "r")
        self.data = _ft.readlines()
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


# data_path = os.path.join("/Users/hmc/Desktop/NLP_DATA", "merged_lm.txt")
# train_db = LM_dataset(d_path=data_path)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# collater = LM_collater(tokenizer=tokenizer)
# train_loder = DataLoader(dataset=train_db, batch_size=128,
#                          shuffle=True, num_workers=8, collate_fn=collater)


# for _data in train_loder:
#     tokenized_data = _data
#     print(f"data shape : {len(tokenized_data)}")
#     print(f"data sample : {tokenized_data[0]}")
#     print(f"data type : {type(tokenized_data[0])}")

#     debug = 0

