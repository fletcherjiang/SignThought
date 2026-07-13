import torch.nn as nn
from torch import Tensor
from .MHSA_RPE import MultiHeadedSelfAttentionModule, ContextualMultiHeadedSelfAttentionModule, GlossFreeAttentionModule, MultiHeadedCrossAttentionModule, RelativeMultiheadSelfAttentionModule

class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = module_factor

    def forward(self, inputs: Tensor, mask: Tensor = None) -> Tensor:
        if isinstance(self.module, MultiHeadedSelfAttentionModule) or isinstance(self.module, RelativeMultiheadSelfAttentionModule) or isinstance(self.module, GlossFreeAttentionModule) or isinstance(self.module, ContextualMultiHeadedSelfAttentionModule):
            return (self.module(inputs, mask=mask) * self.module_factor) + (inputs * self.input_factor)
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

