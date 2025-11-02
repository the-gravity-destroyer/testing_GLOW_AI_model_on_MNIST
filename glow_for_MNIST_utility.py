import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.reshape import SqueezeTransform
from nflows.transforms.normalization import ActNorm
from nflows.transforms.conv import OneByOneConvolution   # (oft hier, nicht conv.OneByOneConvolution)
from nflows.transforms.coupling import AffineCouplingTransform


# --- kleines Hilfsnetz fürs Coupling (CNN) ---
def coupling_net(in_channels, out_channels):
    # out_channels wird von nflows vorgegeben (meist 2*#transformed_channels)
    hidden = 256
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden, hidden, 1),
        nn.ReLU(),
        nn.Conv2d(hidden, out_channels, 3, padding=1),
    )

def channel_mask(Channel):
    # (1,C,1,1) 0/1-Maske – alternierende Kanäle
    m = torch.zeros(Channel)
    m[::2] = 1.0
    return m

def glow_block(Channel):
    """Ein Flow-Step nach Glow: ActNorm → 1x1Conv → AffineCoupling"""
    return CompositeTransform([
        ActNorm(features=Channel),
        OneByOneConvolution(num_channels=Channel),
        AffineCouplingTransform(
            mask=channel_mask(Channel),
            transform_net_create_fn=coupling_net
        ),
    ])

def create_simple_flow():
    # MNIST: 1×28×28 → Squeeze(factor=2) ⇒ 4×14×14
    C0, H0, W0 = 1, 28, 28
    s = 2
    C = C0 * s * s   # 4
    H = H0 // s      # 14
    W = W0 // s      # 14

    transforms = []

    # 1) Squeeze einmalig am Anfang
    transforms.append(SqueezeTransform(factor=s))

    # 2) Mehrere Glow-Steps auf (C,H,W)
    flow_steps = 4
    for _ in range(flow_steps):
        transforms.append(glow_block(C))

    transform = CompositeTransform(transforms)
    base = StandardNormal(shape=[C, H, W])  # Basisdichte im gesqueezten Raum

    return Flow(transform, base)


