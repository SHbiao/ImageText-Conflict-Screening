"""Historical signed readouts: Linear/MLP s=-u(h), residual s=z-u(h)."""
import numpy as np
import torch


def network(kind, dimension=2560):
    d = dimension + int(kind == 'linear_hz')
    net = (torch.nn.Sequential(torch.nn.Linear(d, 1)) if kind.startswith('linear') else
           torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)))
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    return net


def score(net, h, z, scale, kind):
    x = (h - scale['mean']) / scale['std']
    if kind == 'linear_hz':
        x = torch.cat([x, ((z - scale['z_mean']) / scale['z_std'])[..., None]], -1)
    out = -net(x).squeeze(-1)
    return z + out if kind == 'residual_hz' else out


def load_readout(checkpoint, standardizer, kind, device='cpu'):
    net = network(kind).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    net.eval()
    net.requires_grad_(False)
    with np.load(standardizer) as s:
        scale = {k: torch.as_tensor(s[k], dtype=torch.float32, device=device) for k in s.files}
    return net, scale
