from torch import nn
import torch
from BasicModules import Encoder
from data_related.utils import Config
import yaml

"""
# Configuration
  - ELECTRA-SMALL : 
  {
      "number-of-layers" : 12,
      "hidden-size" : 256,
      "sequence-length" : 128,
      "ffn-inner-hidden-size" : 1024,
      "attention-heads" : 4,
      "warmup-steps" : 10000,
      "learning-rate" : 5e-4,
      "batch-size" : 128,
      "train-steps" : 1450000,
      "save-steps" : 100000
  }
"""


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)

        # (bs, n_enc_seq, d_hidn)

    def forward(self, inputs):
        # (bs, n_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, self_attn_probs = self.encoder(inputs)
        # (bs, d_hidn)

        # (bs, n_enc_seq, n_enc_vocab), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, self_attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class ELECTRA_DISCRIMINATOR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BERT(config)
        self.projector = nn.Linear(config.d_head * config.n_head, config.d_hidn)
        self.discriminator = nn.Linear(config.d_hidn, 2, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.tanh = torch.tanh

    def forward(self, inputs):
        # (bs, n_enc_seq, d_hidn), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, attn_probs = self.bert(inputs)
        # (bs, 2)
        outputs = self.projector(outputs)
        outputs = self.tanh(outputs)
        outputs = self.dropout(outputs)
        cls_logit = self.discriminator(outputs)

        return cls_logit


class ELECTRA_GENERATOR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BERT(config)
        self.projector = nn.Linear(config.d_head * config.n_head, config.d_hidn)
        self.layer_norm = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)
        self.language_model = nn.Linear(config.d_hidn, config.n_enc_vocab, bias=True)
        """
        Example.)
        입력 Sentence
        - She is a lovely person who has powerful energy and delightful
        Step.1) Tokenization : 5  [MASK] 7 8 9 10 11 12 13 [MASK] 15
          - Mask Ratio 는 전체 길이의 15 % uniform 이라 하는데, 다른 분포가 좋을 것 같긴 함
          - 어쨌든 실험은 공평하게 해야하기에 Uniform distribution 에서 뽑아오기 
        Step.2) Replacement : 5  22 7 8 9 10 11 12 13 34 15 
          - 이 때, Generator 로부터 샘플링
        Generator 는 masked 된 곳의 token 을 Prediction (log from)
        """
    def forward(self, inputs):
        outputs, attn_probs = self.bert(inputs)
        outputs = self.projector(outputs)
        outputs = self.layer_norm(outputs)
        lm_outs = self.language_model(outputs)
        # (BS, n_enc_seq, n_enc_vocab)
        return lm_outs


def weight_sync(src_model, tgt_model):
    tgt_model.encoder.enc_emb.weight = src_model.encoder.enc_emb.weight
    tgt_model.encoder.pos_emb.weight = src_model.encoder.pos_emb.weight


class name_provider:
    def __init__(self,):
        self.call_cnt = 0

    def __call__(self, m, module_name):

        # name = f"{str(self.call_cnt)}_{module_name}"

        # self.call_cnt += 1
        name = module_name
        return name


class Hook:
    def __init__(self, module, module_name, name_provider, hint=None, yaml_path='./ResNet50.yaml', backward=False):
        """
        :param module_idx:
        :param module:
        :param hint: If hint is None, this means that it is not a weighted residual connection or a kind of similar type
        :param yaml_path:
        :param backward:
        """
        self.yaml_path = yaml_path
        self.name_provider = name_provider
        self.hint = hint
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.module_name = module_name

    def hook_fn(self, module, input, output):

        module_name = self.name_provider(module, self.module_name)
        # p = module.weight
        param = list(module.parameters())
        print(param)
        data = {module_name: module.__repr__()}
        """
        Activation 이나 Pooling 이면 Skip 하도록 하는 게 좋을까 ? Maybe Yes.
        """

        with open(self.yaml_path, "a") as f:
            yaml.dump(data, f)
        f.close()
        self.close()

    def close(self):
        self.hook.remove()


def get_module(model, args, level=1):
    wrap_cnt = 0
    total_wrap = len(args)
    module = model
    """
    level=0 : weight level
    level=1 : 가장 안쪽 모듈
    level=2 : level 1 위 모듈
    level=n : level n-1 위 모듈
    """
    while (total_wrap-level) > wrap_cnt:
        module = getattr(module, args[wrap_cnt])
        wrap_cnt += 1
    return module


cfg = Config({"n_enc_vocab": 30522,  # correct
              "n_enc_seq": 512,  # correct
              "n_seg_type": 2,  # correct
              "n_layer": 12,  # correct
              "d_hidn": 128,  # correct
              "i_pad": 0,  # correct
              "d_ff": 1024,  # correct
              "n_head": 4,  # correct
              "d_head": 64,  # correct
              "dropout": 0.1,  # correct
              "layer_norm_epsilon": 1e-12  # correct
              })


def get_archs(model):
    archs = []
    for name, value in model.named_parameters():
        archs.append(name)
    return archs


def arch_generator(model):
    params_generator = model.parameters()
    params = []
    new_archs = []
    archs = get_archs(model)
    for p, arch in zip(params_generator, archs):
        if 'bias' not in arch:
            params += [p]
            new_archs += [arch]

    return new_archs


class Hook_wrapper:
    def __init__(self, model, yaml_path='./ResNet50.yaml'):
        self.model = model
        self.yaml_path = yaml_path
        self.name_provider = name_provider()

    def __call__(self):
        archs = arch_generator(self.model)
        for idx, arch_name in enumerate(archs):
            args = arch_name.split('.')
            m = get_module(self.model, args=args)
            Hook(module=m, module_name=arch_name, name_provider=self.name_provider, yaml_path=self.yaml_path)

from thop import profile

EG = ELECTRA_GENERATOR(cfg)
# archs = arch_generator(EG)
# Hook_wrapper = Hook_wrapper(model=EG, yaml_path="./eg.yaml")
# Hook_wrapper()

sample = torch.randint(low=1, high=100, size=(1, 128))
FLOPS, PARAM = profile(model=EG, inputs=(sample,))
print(PARAM, FLOPS)
print(PARAM/1e7)
# output = EG(sample)
# loss = output.sum()
# loss.backward()
# path = './eg.yaml'
# with open(path, 'r') as _f:
#     data = yaml.load(_f, yaml.FullLoader)
# _f.close()
# for (key, val) in data.items():
#     # print(key)
#     print(val)

