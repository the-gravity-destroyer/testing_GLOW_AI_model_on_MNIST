import torch
from glow_for_MNIST_utility import create_simple_flow

class ImageSampler:
    def __init__(self, checkpoint_path="checkpoints/flow_mnist.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.flow = create_simple_flow().to(self.device)
        self.flow.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    @torch.no_grad()
    def sample(self, num_samples, temperature=1.0, clamp=True):
        if temperature == 1.0:
            samples = self.flow.sample(num_samples)
        else:
            shape = (num_samples, *self.flow._distribution._shape)
            z = torch.randn(shape, device=self.device) * temperature
            samples, _ = self.flow._transform.inverse(z)
        
        if clamp:
            samples = samples.clamp(0.0, 1.0)
        return samples