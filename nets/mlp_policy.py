import torch
from torch import nn
from pfrl.policies import SoftmaxCategoricalHead
from .common import MLP


class MLPWithSoftmaxHead(nn.Module):
    def __init__(self, d_obs: int, d_hiddens: list, d_action: int):
        super().__init__()
        self.mlp = MLP([d_obs] + d_hiddens + [d_action], activation=torch.relu)
        self.head = SoftmaxCategoricalHead()

    def forward(self, x):
        logits = self.mlp(x)
        return self.head(logits)
