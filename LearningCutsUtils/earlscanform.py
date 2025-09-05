import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F

from LearningCutsUtils.OneToOneLinear import OneToOneLinear


class EarlScanNetwork(torch.nn.Module):
    def __init__(self,features,eta,pt,weights=None,activationscale=2.):
        super().__init__()
        self.features = features
        self.pt = pt
        self.eta = eta
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.nets = torch.nn.ModuleList([
            torch.nn.ModuleList([
                OneToOneLinear(features, self.activation_scale_factor, self.weights)
                for j in range(len(self.pt))
            ])
            for i in range(len(self.eta))
        ])

    def forward(self, x):
        outputs = []
        for i in range(len(self.eta)):
            row = []
            for j in range(len(self.pt)):
                if x[i][j] is None:
                    raise ValueError(f"x[{i}][{j}] is None")
                if not isinstance(x[i][j], torch.Tensor):
                    raise TypeError(f"x[{i}][{j}] is type {type(x[i][j])}, expected torch.Tensor")
                row.append(self.nets[i][j](x[i][j]))
            outputs.append(row)
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)

