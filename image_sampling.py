import torch
from glow_utility import create_simple_flow

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageSampler:
    def __init__(self, flow, device=None):
        self.flow = flow
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.flow.to(self.device).eval()

    @torch.no_grad()
    def sample(self, num_samples, clamp=True):
        # Einfacher & sicherer: die eingebaute Methode verwenden
        sampling = self.flow.sample(num_samples)  # [N,C,H,W]
        if clamp:
            sampling = sampling.clamp(0.0, 1.0)
        return sampling

    @torch.no_grad()
    def sample_with_temperature(self, num_samples, T=0.7, clamp=True):
        # Optional: Temperatur-Sampling (z aus N(0, I*T^2))
        shape = (num_samples, *self.flow._distribution._shape)
        z = torch.randn(shape, device=self.device) * T
        x, _ = self.flow._transform.inverse(z)
        if clamp:
            x = x.clamp(0.0, 1.0)
        return x

# Flow erzeugen und Gewichte laden
flow = create_simple_flow().to(device)
flow.load_state_dict(torch.load("checkpoints/flow_mnist.pt", map_location=device))
flow.eval()

sampler = ImageSampler(flow, device=device)
imgs = sampler.sample(64)  # -> Tensor [64, 1, 28, 28]
