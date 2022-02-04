import torch
import torch.nn as nn
import yaml


def stage_grouper(arch_info_path):
    """
    This function is hard-coded version.
    """
    with open(arch_info_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    GROUPS = {"Front_Block": [], "Encoder_Block": [], "End_Block": []}
    ENCODER_CNT = 0
    SUB_GROUPS = []
    for idx, (key, value) in enumerate(data.items()):
        if idx < 3:
            GROUPS["Front_Block"] += [key]
        elif (idx >= 3 and idx <= len(data)) and 'encoder' in key:
            splited_name = key.split('.')
            if int(splited_name[2]) == ENCODER_CNT:
                SUB_GROUPS += [key]
            else:
                GROUPS["Encoder_Block"] += [SUB_GROUPS]
                SUB_GROUPS = [key]
                ENCODER_CNT += 1
        else:
            if len(SUB_GROUPS) != 0:
                GROUPS["Encoder_Block"] += [SUB_GROUPS]
                SUB_GROUPS = []
            else:
                pass

            GROUPS["End_Block"] += [key]

    return GROUPS


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
        # param = list(module.parameters())
        # print(param)
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


class name_provider:
    def __init__(self,):
        self.call_cnt = 0

    def __call__(self, m, module_name):

        # name = f"{str(self.call_cnt)}_{module_name}"

        # self.call_cnt += 1
        name = module_name
        return name


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


class FP_GETTER:
    def __init__(self, model, arch_type, yaml_path):
        self.model = model
        self.arch_type = arch_type
        self.yaml_path = yaml_path
        """
        이 모듈이 결과적으로 제 기능을 하려면, 순회구조 형태의 자료형을 짜는 게 좋음.
        전체 Block이 있고 그 Block 안에서 돌고 각 Block 별로 Level을 나타내는 Flag 도 필요하고.
        따라서 트리 구조이면서 결과적으로 순회 시 인접 노드가 어디인지 알면 됨.

        """

    def ResNet_FP(self):
        """
        BottleNeck 이냐 아니면 BasicBlock 이냐 이걸 선정
        Residual Connection 의 참조해야하는 Layer 번호를 어떻게 잡아내어 힌트를 주는 게 좋을까 ?
        :param model: model (torch type)
        :param yaml_path:
        :return:
        """

    def EFV2_FP(self):
        pass

    def VGG(self):
        pass

    def MV2(self):
        pass

    def ViT(self):
        pass

    def BERT(self):
        """
        namespace
        enc_emb : output dim - d_hidn
        pos_emb : output dim - d_hidn
        embeddings_project : input dim - d_hidn
        embeddings_project 의 입력단은 상위 embedder 의 출력 차원이랑 동일
        layers : subname (numbering)

        Note:
            Layernorm 도 Learnable 파라미터 존재함. (따라서 Dependency check 시 사용 필요, BN 과 동일한 취급할 것)
            Multihead Attention 의 각 Layer 는 correlation (layer 끼리도, 인접 Block 사이에도 있다.
              - Multihead Attention 의 마지막 Linear 부분은 d_head 값과 n_head 값을 반드시 알고 있는 상황에서 수행
              - Layernorm 도 존재하니 이를 고려해서 Pruning (Deendency check)
            For EncoderLayer, PoswiseFeedForwardNet has transpose operation in forward pass.

        """
        Forward_Hooker = Hook_wrapper(model=self.model, yaml_path=self.yaml_path)
        Forward_Hooker()  # Get yaml file

    def GET_FP(self):
        if self.arch_type == "ResNet":
            self.ResNet_FP()
        elif self.arch_type == "EFV2":
            self.EFV2_FP()
        elif self.arch_type == "VGG":
            self.VGG()
        elif self.arch_type == "MV2":
            self.MV2()
        elif self.arch_type == "VIT":
            self.ViT()
        elif self.arch_type == "BERT":
            self.BERT()
        else:
            raise Exception("This model is not compatible now. Please leave an ask for this issue")


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


def get_indices(score_vector, num_remain):
    if num_remain == score_vector.shape[0]:
        return torch.tensor([i for i in range(score_vector.shape[0])])
    else:
        thresh = score_vector.kthvalue(len(score_vector) - num_remain)[0]
        indices = (score_vector > thresh).nonzero(as_tuple=True)[0].detach()

        return indices


def replace_layer(old_layer, rep_weight, rep_bias, in_channel_indices=None):

    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False

    if isinstance(old_layer, nn.Conv2d):
        new_groups = 1
        # in_channels = rep_weight.size(1)
        out_channels = rep_weight.size(0)

        new_layer = nn.Conv2d(len(in_channel_indices),
                              out_channels,
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag,
                              groups=new_groups)

        if len(in_channel_indices) != new_layer.weight.shape[1]:
            new_layer.weight.data.copy_(rep_weight.data.index_select(1, in_channel_indices))
        else:
            pass

        if rep_bias is not None:
            new_layer.bias.data.copy_(rep_bias.data)

    elif isinstance(old_layer, nn.Conv1d):
        new_groups = 1
        # in_channels = rep_weight.size(1)
        out_channels = rep_weight.size(0)
        new_layer = nn.Conv1d(len(in_channel_indices),
                              out_channels,
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag,
                              groups=new_groups)
        if len(in_channel_indices) != new_layer.weight.shape[1]:
            new_layer.weight.data.copy_(rep_weight.data.index_select(1, in_channel_indices))
        else:
            pass

        if rep_bias is not None:
            new_layer.bias.data.copy_(rep_bias.data)

    elif isinstance(old_layer, nn.BatchNorm2d):

        new_layer = nn.BatchNorm2d(rep_weight.size(0))
        new_layer.weight.data.copy_(rep_weight)
        new_layer.bias.data.copy_(rep_bias)

    elif isinstance(old_layer, nn.LayerNorm):

        new_layer = nn.LayerNorm(rep_weight.size(0))
        new_layer.weight.data.copy_(rep_weight)
        new_layer.bias.data.copy_(rep_bias)

    elif isinstance(old_layer, nn.Linear):
        '''old_layer, rep_weight, rep_bias, in_channel_indices=None'''
        if in_channel_indices is not None:
            new_layer = nn.Linear(len(in_channel_indices), rep_weight.size(0))
            new_layer.weight.data.copy_(rep_weight.data.index_select(1, in_channel_indices))
        else:
            new_layer = nn.Linear(rep_weight.size(1), rep_weight.size(0))
            new_layer.weight.data.copy_(rep_weight.data)
        if rep_bias is not None:
            new_layer.bias.data.copy_(rep_bias.data)

    else:

        assert False, "unsupport layer type:" + \
                      str(type(old_layer))

    return new_layer


def get_thin_params(layer, select_channels, dim=0, flag_last=False):
    """
    Get params from layers after pruning
    """
    if isinstance(layer, nn.Conv2d):

        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, nn.BatchNorm2d):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(dim, select_channels)

        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(dim, select_channels)
        else:
            thin_bias = None

        return thin_weight, thin_bias

    elif isinstance(layer, nn.Linear):

        thin_weight = layer.weight.data.index_select(dim, select_channels)

        if (layer.bias is not None) and (flag_last is False):
            thin_bias = layer.bias.data.index_select(0, select_channels)

        else:
            thin_bias = None
    elif isinstance(layer, nn.LayerNorm):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(dim, select_channels)

        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(dim, select_channels)
        else:
            thin_bias = None
    elif isinstance(layer, nn.Conv1d):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    return thin_weight, thin_bias


def get_module(model, args, level=1):
    wrap_cnt = 0
    total_wrap = len(args)
    module = model
    """
    level=0 : weight level
    level=1 : 가장 안쪽 모듈
    level=2 : level 1 위 모듈
    ...
    level=n : level n-1 위 모듈
    """
    while (total_wrap-level) > wrap_cnt:
        module = getattr(module, args[wrap_cnt])
        wrap_cnt += 1
    return module


def accelerate(model, args, new_layer):
    if len(args) == 2:
        setattr(get_module(model, args, level=2), args[0], new_layer)
    elif len(args) == 3:
        upper = getattr(model, args[0])
        setattr(upper, args[1], new_layer)
    elif len(args) == 4:
        upper = getattr(model, args[0])
        middle = getattr(upper, args[1])
        setattr(middle, args[2], new_layer)

    elif len(args) == 5:
        upper = getattr(model, args[0])
        middle = getattr(upper, args[1])
        inner = getattr(middle, args[2])
        setattr(inner, args[3], new_layer)
    elif len(args) == 6:
        upper = getattr(model, args[0])
        middle_1 = getattr(upper, args[1])
        middle_2 = getattr(middle_1, args[2])
        inner = getattr(middle_2, args[3])
        setattr(inner, args[4], new_layer)
    elif len(args) == 7:
        upper = getattr(model, args[0])
        middle_1 = getattr(upper, args[1])
        middle_2 = getattr(middle_1, args[2])
        middle_3 = getattr(middle_2, args[3])
        inner = getattr(middle_3, args[4])
        setattr(inner, args[5], new_layer)