import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(28 * 28, 256)
        self.hidden = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden(x)
        return F.log_softmax(x, dim=1)
