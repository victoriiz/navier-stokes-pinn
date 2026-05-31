import torch
import torch.nn as nn
import numpy as np

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

"""
PINN class for 2D incompressible Navier-Stokes equations. The network takes spatial coordinates (x, y,t) as input and outputs the velocity components (u, v) and pressure (p).

Args:
    hidden_layers (int): Number of hidden layers in the neural network.
    hidden_width (int): Number of neurons in each hidden layer.
    Re (float): Reynolds number for the fluid flow.
"""
class PINN(nn.Module):
    def __init__(
            self,
            hidden_layers: int = 6,
            hidden_width: int = 164,
            Re: float = 100.0
    ):
        super().__init__()
        self.Re = Re

        sizes = [3] + [hidden_width] * hidden_layers + [3]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(Sine())
            self.net = nn.Sequential(*layers)
            self._init_weights()
        
        def _init_weights(self):
            # initialized according to SIREN paper for sine activation
            linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
            for i, layer in enumerate(linear_layers):
                n_in = layer.weight.shape[1]
                if i == 0:
                    # first layer: use uniform(-1/n, 1/n) to keep inputs in [-pi, pi]
                    bound = 1.0/n_in
                else:
                    # hidden layers: use uniform(-sqrt(6/n), sqrt(6/n)) for sine activation per SIREN paper
                    bound = np.sqrt(6.0/n_in) 
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Tensor of shape (N, 1) representing x-coordinates.
            y (torch.Tensor): Tensor of shape (N, 1) representing y-coordinates.
            t (torch.Tensor): Tensor of shape (N, 1) representing time.

        Returns:
            tuple (u, v, p): A tuple containing three tensors each of shape (N, 1) representing the velocity components and pressure.
        """
        inputs = torch.cat([x, y, t], dim=-1)
        outputs = self.net(inputs)
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        return u, v, p