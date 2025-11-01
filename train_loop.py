import torch
import torch.nn as nn

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.reshape import SqueezeTransform
from nflows.transforms.normalization import ActNorm
from nflows.transforms.permutations import InvertibleConv1x1   # (oft hier, nicht conv.OneByOneConvolution)
from nflows.transforms.coupling import AffineCouplingTransform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def channel_mask(C):
    # (1,C,1,1) 0/1-Maske – alternierende Kanäle
    m = torch.zeros(C)
    m[::2] = 1.0
    return m.view(1, C, 1, 1)

def glow_block(C):
    """Ein Flow-Step nach Glow: ActNorm → 1x1Conv → AffineCoupling"""
    return CompositeTransform([
        ActNorm(features=C),
        InvertibleConv1x1(num_channels=C),
        AffineCouplingTransform(
            mask=channel_mask(C),
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
    K = 4  # Anzahl Flow-Steps
    for _ in range(K):
        transforms.append(glow_block(C))

    transform = CompositeTransform(transforms)
    base = StandardNormal(shape=[C, H, W])  # Basisdichte im gesqueezten Raum

    return Flow(transform, base)


flow = create_simple_flow().to(device)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

for x, _ in loader:
    x = x.to(device)                                  # [B,1,28,28]
    log_px = flow.log_prob(x)                         # ruft forward()+Jacobian auf
    loss = -log_px.mean()
    opt.zero_grad(); loss.backward(); opt.step()

