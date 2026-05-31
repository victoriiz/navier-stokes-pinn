import torch
import torch.nn as nn
import numpy as np


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class PINN(nn.Module):
    def __init__(self, hidden_layers=6, hidden_width=64, Re=100.0):
        super().__init__()
        self.Re = Re

        sizes = [3] + [hidden_width] * hidden_layers + [3]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(Sine())
        self.net = nn.Sequential(*layers)

        # SIREN initialisation
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            n_in = layer.weight.shape[1]
            bound = 1.0 / n_in if i == 0 else np.sqrt(6.0 / n_in)
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
