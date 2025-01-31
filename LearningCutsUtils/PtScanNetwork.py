import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F

from LearningCutsUtils.OneToOneLinear import OneToOneLinear


class PtScanNetwork(torch.nn.Module):
    def __init__(self,features,pt,weights=None,activationscale=2.,postroot=1.):
        super().__init__()
        self.features = features
        self.pt = pt
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.post_product_root=postroot
        self.nets = torch.nn.ModuleList([OneToOneLinear(features, self.activation_scale_factor, self.weights, self.post_product_root) for i in range(len(self.pt))])

    def forward(self, x):
        outputs=torch.cat(tuple(self.nets[i](x[i]) for i in range(len(self.pt))),  dim=0)
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)

