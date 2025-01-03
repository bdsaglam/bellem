"""Custom Torch layers"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/ml.layer.ipynb.

# %% auto 0
__all__ = ['reverse_grad', 'GradReverse']

# %% ../../nbs/ml.layer.ipynb 3
import torch
import torch.nn as nn
from torch.autograd import Function

# %% ../../nbs/ml.layer.ipynb 4
class reverse_grad(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return - grad_out * ctx.lambd, None

class GradReverse(nn.Module):
    def __init__(self, lambd=1.):
        super().__init__()
        self.lambd = torch.tensor(lambd, requires_grad=False)

    def forward(self, x):
        return reverse_grad.apply(x, self.lambd)
