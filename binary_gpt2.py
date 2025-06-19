import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

def binary_activation(x):
    return SignSTE.apply(x)

class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input):
        bin_weight = binary_activation(self.weight)
        if self.bias is not None:
            return F.linear(binary_activation(input), bin_weight, self.bias)
        return F.linear(binary_activation(input), bin_weight)

def convert_to_binary(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            bin_layer = BinaryLinear(child.in_features, child.out_features, bias=child.bias is not None)
            bin_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                bin_layer.bias.data = child.bias.data.clone()
            setattr(module, name, bin_layer)
        else:
            convert_to_binary(child)

class GPT2BinaryModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_binary(self)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
