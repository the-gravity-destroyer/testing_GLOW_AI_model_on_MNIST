import glow_utility
import torch
from datasets import DequantizedMNIST
from torch.utils.data import DataLoader

class Evaluation:
    def __init__(self):
        self.flow = glow_utility.create_simple_flow()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.flow.to(self.device)

    def evaluate(self):
        self.flow.eval() # Evaluation-Modus für Testen
        test_loader = glow_utility.get_test_loader()

        with torch.no_grad(): # Deaktiviert Gradientenberechnung
            negative_log_likelihood = 0
            for batch, _ in test_loader:
                batch = batch.to(self.device)
                negative_log_likelihood += (-self.flow.log_prob(batch)).mean().item() # Summe der NLL über alle Batches
            print(f"Test NLL: {negative_log_likelihood / len(test_loader):.4f}")


def get_test_loader(batch_size=256):
    test_dataset = DequantizedMNIST(root="./data", train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)