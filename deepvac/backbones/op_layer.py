from typing import List
import torch
import torch.nn as nn

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)
