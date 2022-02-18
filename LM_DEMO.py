from Models.BERT import BERT, ELECTRA_GENERATOR, mask_tokens, sampler, ELECTRA_MODEL
import torch
from transformers import AutoTokenizer
import argparse
from data_related.utils import Config
from data_related.Custom_dataloader import LM_dataset, LM_collater
import os
from torch.utils.data import DataLoader


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def demo(args):

    G_cfg = Config({"n_enc_vocab": 30522,  # correct
                    "n_enc_seq": 512,  # correct
                    "n_seg_type": 2,  # correct
                    "n_layer": 12,  # correct
                    "d_model": 128,  # correct
                    "i_pad": 0,  # correct
                    "d_ff": 256,  # correct
                    "n_head": 1,  # correct
                    "d_head": 64,  # correct
                    "dropout": 0.1,  # correct
                    "layer_norm_epsilon": 1e-12  # correct
                    })

    D_cfg = Config({"n_enc_vocab": 30522,  # correct
                    "n_enc_seq": 512,  # correct
                    "n_seg_type": 2,  # correct
                    "n_layer": 12,  # correct
                    "d_model": 128,  # correct
                    "i_pad": 0,  # correct
                    "d_ff": 1024,  # correct
                    "n_head": 4,  # correct
                    "d_head": 64,  # correct
                    "dropout": 0.1,  # correct
                    "layer_norm_epsilon": 1e-12  # correct
                    })

    model = ELECTRA_MODEL(D_cfg, G_cfg, device=args.device).to(args.device)
    
    check_point = torch.load(args.weight_path)
    
    model.generator.load_state_dict(check_point["state_dict"])
    model = model.generator.to(args.device)
    model.eval()
    
    test_dataset = LM_dataset(d_path=args.test_sequences)
    
    tokenizer_path = "/vision/7032593/NLP/ELECTRA/tokenizer_files"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    collater = LM_collater(tokenizer=tokenizer)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=8, collate_fn=collater)
    
    distribution = torch.distributions.gumbel.Gumbel(0, 1)
    
    for seq_tokens in test_loader:
        seq_tokens = seq_tokens.to(args.device)
        masked_tokens, masked_labels, replace_mask = mask_tokens(inputs=seq_tokens, mask_token_index=103, 
                                                                 vocab_size=D_cfg.n_enc_vocab,
                                                                 special_token_indices=[100, 102, 0, 101, 103])
        g_logits = model(masked_tokens)
        m_g_logits = g_logits[replace_mask, :]
        recon = tokenizer.decode(g_logits.argmax(dim=2)[0])

        with torch.no_grad():
            sampled_tokens = sampler(Dist=distribution, logits=m_g_logits, device=args.device)
            generated_tokens = masked_tokens.clone()
            generated_tokens[replace_mask] = sampled_tokens
        print(generated_tokens)
        recon_and_sample = tokenizer.decode(generated_tokens[0])
        print(f"Generated Sequence : {recon_and_sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, 
                        default="/vision2/7032593/ELECTRA/check_points/GEN_ITER_100000_LM_MODEL.pth")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--test_sequences", type=str, default="./LM_test.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    demo(args)

