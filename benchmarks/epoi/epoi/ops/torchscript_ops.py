"""The fused ops by torchscript."""
from typing import List
from functools import partial
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.function import once_differentiable
try:
    from functorch.compile import memory_efficient_fusion
except:
    memory_efficient_fusion = None


class BiasGeLUFunction(torch.autograd.Function):
    """Bias+GeLU. Copied from Megatron-LM."""

    @torch.jit.script
    def bias_gelu(bias, y):
        # print(bias.size(), y.size())
        x = bias + y
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    # gradient of tanh approximation of gelu
    # gradient of actual gelu is:
    # 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
    @torch.jit.script
    def bias_gelu_back(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
            1 + tanh_out
        )
        return ff * g

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return BiasGeLUFunction.bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = BiasGeLUFunction.bias_gelu_back(grad_output, bias, input)
        return tmp, tmp


class FusedBiasGELU(torch.nn.Module):
    def __init__(self, size, device=None, dtype=None, prev_weight=None, fused=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)

    def reset_parameters(self, prev_weight=None):
        range = (0, 1)
        if prev_weight is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *range)

    def forward(self, input):
        if self.fused:
            return BiasGeLUFunction.apply(input, self.bias)
        return F.gelu(input + self.bias, approximate="none")


def new_gelu(input):
    """New GELU activation function copied from HuggingFace transformers."""
    return (
        0.5
        * input
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def bias_new_gelu(input, bias):
    return new_gelu(input + bias)


class FusedBiasNewGELU(torch.nn.Module):
    def __init__(self, size, device=None, dtype=None, prev_weight=None, fused=True, aot=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)
        if self.fused:
            if aot and memory_efficient_fusion is not None:
                self.func = memory_efficient_fusion(bias_new_gelu)
            else:
                self.func = torch.jit.script(bias_new_gelu)
        else:
            self.func = bias_new_gelu

    def reset_parameters(self, prev_weight=None):
        range = (0, 1)
        if prev_weight is not None and len(prev_weight.shape) > 1:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *range)

    def forward(self, input):
        return self.func(input, self.bias)


class MM(torch.nn.Module):
    """
    Copied from HuggingFace transformers.
    The MM layer defined defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        torch.nn.init.normal_(w, std=0.02)
        self.weight = torch.nn.Parameter(w)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(nf))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if self.bias is not None:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        else:
            x = torch.mm(x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def fused_dropout_add_layernorm(
    input1,
    input2,
    weight,
    bias,
    dropout_prob: float,
    training: bool,
    normalized_shape: List[int],
    eps: float,
):
    """torchscript tracable dropout-add-layernorm.
    (non-tensor arguments must have type annotations)
    """
    dropout_out = F.dropout(input1, dropout_prob, training=training)
    norm_input = dropout_out + input2
    norm_output = F.layer_norm(norm_input, normalized_shape, weight, bias, eps)
    return norm_output


class FusedDropoutAddLayerNorm(torch.nn.Module):
    def __init__(self, size, dropout_prob, eps=1e-5, fused=True, aot=False):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(size, eps=eps)
        self.dropout_prob = dropout_prob
        self.fused = fused
        if fused and aot and memory_efficient_fusion is not None:
            # FIXME: it works fine in benchmark but failed with HF Trainer with
            # RuntimeError: Trying to backward through the graph a second time
            # (or directly access saved tensors after they have already been freed).
            self.func = partial(
                fused_dropout_add_layernorm,
                dropout_prob=self.dropout_prob,
                training=self.training,
                eps=self.layer_norm.eps,
                normalized_shape=self.layer_norm.normalized_shape,
            )
            self.func = memory_efficient_fusion(self.func)
        else:
            self.func = fused_dropout_add_layernorm
            if fused:
                self.func = torch.jit.script(self.func)
            self.func = partial(
                self.func,
                dropout_prob=self.dropout_prob,
                training=self.training,
                eps=self.layer_norm.eps,
                normalized_shape=self.layer_norm.normalized_shape,
            )

    def forward(self, input1, input2):
        return self.func(
            input1,
            input2,
            self.layer_norm.weight,
            self.layer_norm.bias,
        )
    

def convolution_backward(grad_out, X, weight, stride):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1), stride=stride).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input


class Conv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, stride):
        ctx.save_for_backward(X, weight)
        return F.conv2d(X, weight, stride=stride)


    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out, stride=2):
        X, weight = ctx.saved_tensors
        return convolution_backward(grad_out, X, weight, stride=stride)
    

def unsqueeze_all(t):
    # Helper function to ``unsqueeze`` all the dimensions that we reduce over
    return t[None, :, None, None]

def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # We use the formula: ``out = (X - mean(X)) / (sqrt(var(X)) + eps)``
    # in batch norm 2D forward. To simplify our derivation, we follow the
    # chain rule and compute the gradients as follows before accumulating
    # them all into a final grad_input.
    #  1) ``grad of out wrt var(X)`` * ``grad of var(X) wrt X``
    #  2) ``grad of out wrt mean(X)`` * ``grad of mean(X) wrt X``
    #  3) ``grad of out wrt X in the numerator`` * ``grad of X wrt X``
    # We then rewrite the formulas to use as few extra buffers as possible
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps)**2  # ``d_denom = -num / denom**2``
    # It is useful to delete tensors when you no longer need them with ``del``
    # For example, we could've done ``del tmp`` here because we won't use it later
    # In this case, it's not a big difference because ``tmp`` only has size of (C,)
    # The important thing is avoid allocating NCHW-sized tensors unnecessarily
    d_var = d_denom / (2 * sqrt_var)  # ``denom = torch.sqrt(var) + eps``
    # Compute ``d_mean_dx`` before allocating the final NCHW-sized grad_input buffer
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    # ``d_mean_dx`` has already been reassigned to a C-sized buffer so no need to worry

    # ``(1) unbiased_var(x) = ((X - unsqueeze_all(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)``
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    # (2) mean (see above)
    grad_input += d_mean_dx
    # (3) Add 'grad_out / <factor>' without allocating an extra buffer
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)  # ``sqrt_var + eps > 0!``
    return grad_input

class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, eps=1e-3):
        # Don't save ``keepdim`` values for backward
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.save_for_backward(X)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, = ctx.saved_tensors
        return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)


class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, stride: int=1, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # (1) Only need to save this single buffer for backward!
        ctx.save_for_backward(X, conv_weight)

        if stride < 1:
            stride = 1

        # (2) Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight, stride=stride)
        # (3) Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    def backward(ctx, grad_out, stride=1):
        X, conv_weight, = ctx.saved_tensors
        # (4) Batch norm backward
        # (5) We need to recompute conv
        X_conv_out = F.conv2d(X, conv_weight, stride=stride)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                                       ctx.N, ctx.eps)
        # (6) Conv2d backward
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight, stride=stride)
        return grad_X, grad_input, None, None, None, None, None
    

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        self.stride = stride
        # Initialize
        self.reset_parameters()

    def forward(self, X):
        # print('self.stride', self.stride)
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps, self.stride)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))


def fuse_linear_bn(linear, bn):
    w = linear.weight
    mean = bn.running_mean.cuda()
    var_sqrt = torch.sqrt(bn.running_var + bn.eps).cuda()

    beta = bn.weight.cuda()
    gamma = bn.bias.cuda()

    if linear.bias is not None:
        b = linear.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w.cuda()
    b = b.cuda()
    w = w * (beta / var_sqrt).reshape([4096, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_linear = nn.Linear(linear.in_features,
                         linear.out_features)
                                             
    fused_linear.weight = nn.Parameter(w)
    fused_linear.bias = nn.Parameter(b)
    return fused_linear