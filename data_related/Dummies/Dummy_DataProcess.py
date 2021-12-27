
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import os
from glob import glob
from transformers import ElectraTokenizerFast


def prepare_data(dataset_f, tokenizer, batch_size: int = 64, num_workers: int = 0):
    """Given an input file, prepare the train, test, validation dataloaders.
       The created datasets will be preprocessed and save to disk.
    :param dataset_f: input file
    :param tokenizer: pretrained tokenizer that will prepare the data, i.e. convert tokens into IDs
    :param batch_size: batch size for the dataloaders
    :param num_workers: number of CPU workers to use during dataloading. On Windows this must be zero
    :return: a dictionary containing train, test, validation dataloaders
    """

    def collate(batch):
        """Collates gathered items to form a batch which is then used in training, evaluation, or testing.
        batch : List[Dict[str, Tensor]]
        output shape : Dict[str, Tensor]
        :param batch: a list of samples from the dataset. Each sample is a dictionary with keys "input_ids".
        :return: the created batch with keys "input_ids"
        """

        all_input_ids = pad_sequence([item["input_ids"] for item in batch]).to(torch.long)
        print(all_input_ids)
        return {"input_ids": all_input_ids}

    def preprocess(sentences):
        """Preprocess the raw input sentences from the text file.
        :param sentences: a list of sentences (strings)
        :return: a dictionary of "input_ids"
        """
        tokens = [s.split() for s in tqdm(sentences)]

        # The sequences are not padded here. we leave that to the dataloader in collate
        # That means: a bit slower processing, but a smaller saved dataset size
        return tokenizer(tokens,
                         add_special_tokens=False,
                         is_pretokenized=True,
                         return_token_type_ids=False,
                         return_attention_mask=False)
    print("Start to preprocess")
    _data = {}
    dataset_f = glob(os.path.join(dataset_f, "*.txt"))
    print(dataset_f)
    # {"text": Path(dataset_f).read_text(encoding="utf-8").splitlines()}
    print("Step 1: getting data")
    for i, _data_path in enumerate(dataset_f):
        _data[f"text_{i}"] = Path(_data_path).read_text(encoding="utf-8").splitlines()
    print("Step 2: from Raw to dict")
    dataset = Dataset.from_dict(_data)

    # Split the dataset into train, test, dev
    # 90% (train), 10% (test + validation)
    print("Step 3: Splitting data into train and test partitions")
    train_testvalid = dataset.train_test_split(test_size=0.1)
    # 10% of total (test), 10% of total (validation)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

    dataset = DatasetDict({"train": train_testvalid["train"],
                           "test": test_valid["test"],
                           "valid": test_valid["train"]})

    print("Step 4: Tokenizing data")
    dataset = dataset.map(preprocess, input_columns=["text"], batched=True)
    dataset.set_format("torch", columns=["input_ids"])
    print("Step 5. Finally, Partitioning on memories")
    return {partition: DataLoader(ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate,
                                  num_workers=num_workers,
                                  pin_memory=True) for partition, ds in dataset.items()}

ROOT_DIR = "/Users/hmc/Desktop/NLP_DATA"
tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")
prepare_data(dataset_f=ROOT_DIR, tokenizer=tokenizer, batch_size=128, num_workers=16)


