import os
import argparse
from glob import glob
from tokenizers import BertWordPieceTokenizer
import json


def vocab_making(args):
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        pass
    print(f"Vocabulary file will be saved at {args.save_dir}.")
    FILE_PATHS = glob(os.path.join(args.root_dir, "*.txt"))

    print("--------Build Tokenizer----------")
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    print("--------Finished building Tokenizer----------")

    print("--------Start to train tokenizer----------")
    tokenizer.train(files=FILE_PATHS, vocab_size=30522, min_frequency=2,
                    limit_alphabet=6000, wordpieces_prefix='##',
                    special_tokens=['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    print("--------Ended training tokenizer----------")
    tokenizer.save_model(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/Users/hmc/Desktop/NLP_DATA")
    parser.add_argument("--save_dir", type=str, default="/Users/hmc/Desktop/projects/BERT")

    args = parser.parse_args()
    vocab_making(args)
