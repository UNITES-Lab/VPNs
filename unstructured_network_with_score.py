import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Conv2d, Linear



def replace_layers(model, old_layer, new_layer, channelPrune, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_layers(module, old_layer, new_layer,channelPrune,args)

        if type(module) == old_layer:
            if isinstance(module, nn.Conv2d):
                if args.network == 'vgg':
                    layer_new = new_layer(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, False,channelPrune)
                else:
                    layer_new = new_layer(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias,channelPrune)
            elif isinstance(module, nn.Linear):
                layer_new = new_layer(module.in_features, module.out_features, module.bias,channelPrune)
            layer_new.weight.data=module.weight.data
            if args.network == 'vgg' and isinstance(module, nn.Conv2d):
                pass
            elif module.bias is not None:
                layer_new.bias.data=module.bias.data
            model._modules[name] = layer_new

    return model


def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "subnet":
        return SubnetConv, SubnetLinear
    else:
        raise ValueError("Incorrect layer type")


# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        zero = torch.tensor([0.]).cuda()
        one = torch.tensor([1.]).cuda()
        out = torch.where(out <= threshold, zero, one)
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        channel_prune='kernel'
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.channel_prune=channel_prune
        if channel_prune=='kernel':
            self.popup_scores = Parameter(torch.randn((self.weight.shape[0],self.weight.shape[1],1,1)))
        elif channel_prune=='channel':
            self.popup_scores = Parameter(torch.randn((self.weight.shape[0],1,1,1)))
        elif channel_prune=='inputchannel':
            self.popup_scores = Parameter(torch.randn((1,self.weight.shape[1],self.weight.shape[2],self.weight.shape[3])))
        else:
            self.popup_scores = Parameter(torch.randn(self.weight.shape))
        self.popup_scores.is_score=True
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        # self.weight.requires_grad = False
        # if self.bias is not None:
        #     self.bias.requires_grad = False
        self.w = 0
        self.k=False
        self.threshold=False

    def set_prune_rate(self, k):
        self.k = k

    def set_threshold(self, threshold):
        self.threshold = threshold

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        if self.threshold is False:
            adj=1.0
        else:
            adj = GetSubnet.apply(self.popup_scores.abs(), self.threshold)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.
    def __init__(self, in_features, out_features, bias=True,channel_prune='kernel'):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.channel_prune=channel_prune
        if channel_prune=='channel':
            self.popup_scores = Parameter(torch.randn((self.weight.shape[0],1)))
        elif channel_prune=='inputchannel':
            self.popup_scores = Parameter(torch.randn((1,self.weight.shape[1])))
        else:
            self.popup_scores = Parameter(torch.randn(self.weight.shape))
        self.popup_scores.is_score=True
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        # self.weight.requires_grad = False
        # self.bias.requires_grad = False
        self.w = 0
        self.k=False
        self.threshold=False

    def set_prune_rate(self, k):
        self.k = k

    def set_threshold(self, threshold):
        self.threshold = threshold

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        if self.threshold is False:
            adj=1.0
        else:
            adj = GetSubnet.apply(self.popup_scores.abs(), self.threshold)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x


def set_scored_network(network, args):
    cl, ll = get_layers('subnet')
    network=replace_layers(network,Conv2d,cl,'weight',args)
    network=replace_layers(network,Linear,ll,'weight',args)
    network.to(args.device)
    if args.hydra_scaled_init:
        print('Using scaled score initialization\n')
        initialize_scaled_score(network)
    
    return network


def set_prune_rate(network, prune_rate):
    cl, ll = get_layers('subnet')
    for name, module in network.named_modules():
        if isinstance(module,cl):
            module.set_prune_rate(prune_rate)
        if  isinstance(module,ll):
            module.set_prune_rate(prune_rate)


def set_prune_threshold(network, prune_rate):
    score_dict={}
    for name, weight in network.named_parameters():
        if hasattr(weight, 'is_score') and weight.is_score:
            score_dict[name] = torch.clone(weight.data).detach().abs_()
    global_scores = torch.cat([torch.flatten(v) for v in score_dict.values()])
    k = int((1 - prune_rate) * global_scores.numel())
    if not k < 1:
        threshold, _ = torch.kthvalue(global_scores, k)
        cl, ll = get_layers('subnet')
        for name, module in network.named_modules():
            if isinstance(module,cl):
                module.set_threshold(threshold)
            if  isinstance(module,ll):
                module.set_threshold(threshold)


def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def switch_to_prune(model):
    # print(f"#################### Pruning network ####################")
    # print(f"===>>  gradient for weights: None  | training importance scores only")

    unfreeze_vars(model, "popup_scores")
    freeze_vars(model, "weight")
    freeze_vars(model, "bias")


def switch_to_finetune(model):
    # print(f"#################### Fine-tuning network ####################")
    # print(
    #     f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
    # )
    freeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")


def switch_to_bilevel(model):
    unfreeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")
