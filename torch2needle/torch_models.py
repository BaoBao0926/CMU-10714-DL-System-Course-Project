import torch
import torch.nn as nn

class TorchMLP_v1(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=5, output_dim=1):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net1(x)


class TorchMLP_v2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=5, output_dim=1):
        super().__init__()
        self.net = nn.Linear(output_dim, output_dim)
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.net3 = nn.Linear(output_dim, output_dim)


    def forward(self, x):
        x = self.net(self.net1(x) + self.net2(x) + self.net1(x))
        x = x - self.net3(x)
        return x

