import sentencepiece as sp
from random import shuffle, choice, randrange, random
import pandas as pd
import os
from tqdm import tqdm
import sys
import csv
import json
import torch

csv.field_size_limit(sys.maxsize)


def weight_setup(src_module, tgt_module):
    if isinstance(src_module, torch.nn.Module) and isinstance(tgt_module, torch.nn.Module):
        tgt_module.weight = src_module.weight
    else:
        raise Exception("src or tgt is not a torch module")


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

"""
bellow code lines are paul-hyun's codes for korean  
ref : (https://paul-hyun.github.io/bert-01/)
"""
# def kor_vocab_maker(dataset_path, output_dir, corpus="kowiki.txt",
#                 prefix="kowiki", seperator=u"\u241D", voc_size=8000):
#     out_file = os.path.join(output_dir, "kowiki.txt")
#     df = pd.read_csv(dataset_path, sep=seperator, engine="python")
#     with open(out_file, "w") as f:
#         for index, row in df.iterrows():
#             f.write(row["text"])  # title 과 text를 중복 되므로 text만 저장 함
#             f.write("\n\n\n\n")  # 구분자
#
#     corpus = corpus
#     prefix = prefix
#
#     sp.SentencePieceTrainer.train(
#         f"--input={corpus} --model_prefix={prefix} --vocab_size={voc_size + 7}" +
#         " --model_type=bpe" +
#         " --max_sentence_length=999999" +  # 문장 최대 길이
#         " --pad_id=0 --pad_piece=[PAD]" +  # pad (0)
#         " --unk_id=1 --unk_piece=[UNK]" +  # unknown (1)
#         " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence (2)
#         " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence (3)
#         " --user_defined_symbols=[SEP],[CLS],[MASK]")  # 사용자 정의 토큰


# def create_pretrain_mask(tokens, mask_cnt, vocab_list):
#     '''This function make mask for MLM Task'''
#     cand_idx = []
#     for (i, token) in enumerate(tokens):
#         if token == "[CLS]" or token == "[SEP]":
#             continue
#         if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
#             cand_idx[-1].append(i)
#         else:
#             cand_idx.append([i])
#     shuffle(cand_idx)
#
#     mask_lms = []
#     for index_set in cand_idx:
#         if len(mask_lms) >= mask_cnt:
#             break
#         if len(mask_lms) + len(index_set) > mask_cnt:
#             continue
#         for index in index_set:
#             masked_token = None
#             if random() < 0.8:  # 80% replace with [MASK]
#                 masked_token = "[MASK]"
#             else:
#                 if random() < 0.5:  # 10% keep original
#                     masked_token = tokens[index]
#                 else:  # 10% random word
#                     masked_token = choice(vocab_list)
#         mask_lms.append({"index": index, "label": tokens[index]})
#         tokens[index] = masked_token
#     mask_lms = sorted(mask_lms, key=lambda x: x["index"])
#     mask_idx = [p["index"] for p in mask_lms]
#     mask_label = [p["label"] for p in mask_lms]
#
#     return tokens, mask_idx, mask_label


# def trim_tokens(tokens_a, tokens_b, max_seq):
#     '''This function trim tokens which exceed their limited length'''
#     while True:
#         total_length = len(tokens_a) + len(tokens_b)
#         if total_length <= max_seq:
#             break
#         if len(tokens_a) > len(tokens_b):
#             del tokens_a[0]
#         else:
#             tokens_b.pop()

#
# def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
#     """
#     This function do conversion for comptuable data format
#     :param docs:
#     :param doc_idx:
#     :param doc:
#     :param n_seq:
#     :param mask_prob:
#     :param vocab_list:
#     :return:
#     """
#     # for [CLS], [SEP], [SEP]
#     max_seq = n_seq - 3
#     tgt_seq = max_seq
#
#     instances = []
#     current_chunk = []
#     current_length = 0
#     for i in range(len(doc)):
#         current_chunk.append(doc[i])  # line
#         current_length += len(doc[i])
#         if i == len(doc) - 1 or current_length >= tgt_seq:
#             if 0 < len(current_chunk):
#                 a_end = 1
#                 if 1 < len(current_chunk):
#                     a_end = randrange(1, len(current_chunk))
#                 tokens_a = []
#                 for j in range(a_end):
#                     tokens_a.extend(current_chunk[j])
#
#                 tokens_b = []
#                 if len(current_chunk) == 1 or random() < 0.5:
#                     is_next = 0
#                     tokens_b_len = tgt_seq - len(tokens_a)
#                     random_doc_idx = doc_idx
#                     while doc_idx == random_doc_idx:
#                         random_doc_idx = randrange(0, len(docs))
#                     random_doc = docs[random_doc_idx]
#
#                     random_start = randrange(0, len(random_doc))
#                     for j in range(random_start, len(random_doc)):
#                         tokens_b.extend(random_doc[j])
#                 else:
#                     is_next = 1
#                     for j in range(a_end, len(current_chunk)):
#                         tokens_b.extend(current_chunk[j])
#
#                 trim_tokens(tokens_a, tokens_b, max_seq)
#                 assert 0 < len(tokens_a)
#                 assert 0 < len(tokens_b)
#
#                 tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
#                 segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
#
#                 tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob),
#                                                                     vocab_list)
#
#                 instance = {
#                     "tokens": tokens,
#                     "segment": segment,
#                     "is_next": is_next,
#                     "mask_idx": mask_idx,
#                     "mask_label": mask_label
#                 }
#                 instances.append(instance)
#
#             current_chunk = []
#             current_length = 0
#
#     return instances
#
#
# def make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob):
#     vocab_list = []
#     for id in range(vocab.get_piece_size()):
#         if not vocab.is_unknown(id):
#             vocab_list.append(vocab.id_to_piece(id))
#
#     line_cnt = 0
#     with open(in_file, "r") as in_f:
#         for line in in_f:
#             line_cnt += 1
#
#     docs = []
#     with open(in_file, "r") as f:
#         doc = []
#         with tqdm(total=line_cnt, desc=f"Loading") as pbar:
#             for i, line in enumerate(f):
#                 line = line.strip()
#                 if line == "":
#                     if 0 < len(doc):
#                         docs.append(doc)
#                         doc = []
#                 else:
#                     pieces = vocab.encode_as_pieces(line)
#                     if 0 < len(pieces):
#                         doc.append(pieces)
#                 pbar.update(1)
#         if doc:
#             docs.append(doc)
#
#     for index in range(count):
#         output = out_file.format(index)
#         if os.path.isfile(output): continue
#
#         with open(output, "w") as out_f:
#             with tqdm(total=len(docs), desc=f"Making") as pbar:
#                 for i, doc in enumerate(docs):
#                     instances = create_pretrain_instances(docs, i, doc, n_seq, mask_prob, vocab_list)
#                     for instance in instances:
#                         out_f.write(json.dumps(instance))
#                         out_f.write("\n")
#                     pbar.update(1)


def evaluator(model, dataloader, cur_epoch, total_epoch):

    print(f"Epoch : {cur_epoch + 1} / {total_epoch}, Training Loss : {None}")
    print(f"Epoch : {cur_epoch + 1} / {total_epoch}, Eval Loss : {None}")

# in_file = "./kowiki.txt"
# out_file = "./kowiki_bert.json"
# vocab_file = "./kowiki.model"
# vocab = sp.SentencePieceProcessor()
# vocab.load(vocab_file)
# count = 10
# n_seq = 256
# mask_prob = 0.15
#
# make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob)