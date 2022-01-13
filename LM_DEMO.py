from Models.BERT import BERT, ELECTRA_GENERATOR
import torch
from transformers import AutoTokenizer
import argparse
from data_related.utils import Config
from data_related.Custom_dataloader import LM_dataset, LM_collater
import os
from torch.utils.data import DataLoader
from Pretraining import mask_token_filler, batch_wise_masking


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def demo(args):

    cfg = Config({"n_enc_vocab": 30522,  # correct
                  "n_enc_seq": 512,  # correct
                  "n_seg_type": 2,
                  "n_layer": 12,
                  "d_hidn": 128,  # correct
                  "i_pad": 0,
                  "d_ff": 256,
                  "n_head": 4,
                  "d_head": 16,
                  "dropout": 0.1,
                  "layer_norm_epsilon": 1e-12
                  })

    Model = ELECTRA_GENERATOR(config=cfg)
    check_point = torch.load(args.weight_path)
    Model.load_state_dict(check_point["state_dict"])
    Model.to(args.device)
    Model.eval()
    test_dataset = LM_dataset(d_path=args.test_sequences)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collater = LM_collater(tokenizer=tokenizer)
    test_loder = DataLoader(dataset=test_dataset, batch_size=1,
                            shuffle=False, num_workers=8, collate_fn=collater)

    Gumbel_Distribution = torch.distributions.gumbel.Gumbel(0, 1)

    for seq_tokens in test_loder:
        seq_tokens = seq_tokens.to(args.device)
        input_seq = tokenizer.decode(seq_tokens)
        print(f"Input Sequence : {input_seq}")
        # (BS, n_enc_seq, n_enc_vocab)
        recon_output = Model(seq_tokens)
        recon_tokens = torch.argmax(recon_output, dim=2)
        recon_seq = tokenizer.decode(recon_tokens)
        print(f"Reconstructed sequence : {recon_seq}")
        with torch.no_grad():
            masked_tokens, masked_lists = batch_wise_masking(seq_tokens)
            Generated_tokens, Disc_labels = mask_token_filler(sampling_distribution=Gumbel_Distribution,
                                                              Generator_logits=recon_output, device=args.device,
                                                              masked_tokens=masked_tokens,
                                                              masking_indices=masked_lists, labels=seq_tokens)
        recon_and_sample = tokenizer.decode(Generated_tokens)
        print(f"Sampled Sequence : {recon_and_sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--test_sequences", type=str, default="LM_test.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    demo(args)

