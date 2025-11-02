import torch
import torch.nn as nn
from datasets.dequant import DequantizedMNIST
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.reshape import SqueezeTransform
from nflows.transforms.normalization import ActNorm
from nflows.transforms.conv import OneByOneConvolution   # (oft hier, nicht conv.OneByOneConvolution)
from nflows.transforms.coupling import AffineCouplingTransform
from torch.utils.data import DataLoader



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

def channel_mask(Channel):
    # (1,C,1,1) 0/1-Maske – alternierende Kanäle
    m = torch.zeros(Channel)
    m[::2] = 1.0
    return m.view(1, Channel, 1, 1)

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


def get_train_loader(batch_size=256):
    train_dataset = DequantizedMNIST(root="./data", train=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

def get_test_loader(batch_size=256):
    test_dataset = DequantizedMNIST(root="./data", train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)


flow = create_simple_flow().to(device)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

loader = get_train_loader()

for epoch in range(1, 11):
    flow.train()        # Trainingsmodus        
    total_loss = 0

    for batch, _ in loader:                # _ steht für Labels, die wir nicht brauchen, da unüberwachtes Lernen
        batch = batch.to(device)           # [B,1,28,28]
        log_px = flow.log_prob(batch)      # Forward + Jacobian
        loss = -log_px.mean()              # Minimiere negative Log-Wahrscheinlichkeit

        opt.zero_grad()                    # Gradienten zurücksetzen
        loss.backward()                    # Backprop
        opt.step()                         # Parameter-Update

        total_loss += loss.item()

    print(f"Epoch {epoch} | loss: {total_loss/len(loader):.4f}")


flow.eval() # Evaluation-Modus für Testen
test_loader = get_test_loader()

with torch.no_grad(): # Deaktiviert Gradientenberechnung
    negative_log_likelihood = 0
    for batch, _ in test_loader:
        batch = batch.to(device)
        negative_log_likelihood += (-flow.log_prob(batch)).mean().item() # Summe der NLL über alle Batches
    print(f"Test NLL: {negative_log_likelihood / len(test_loader):.4f}")


