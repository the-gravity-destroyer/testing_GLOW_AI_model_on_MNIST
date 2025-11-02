import glow_for_MNIST_utility
import torch
from dequant import DequantizedMNIST
from torch.utils.data import DataLoader

class Training:
    def __init__(self):
        self.weights_already_trained = False
        self.flow = glow_for_MNIST_utility.create_simple_flow()
        self.flow.to(self.device)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.opt = torch.optim.Adam(self.flow.parameters(), lr=1e-3)


    def train(self):

        if self.weights_already_trained:
            print("Weights already trained, skipping training.")
            return
        
        loader = self.get_train_loader()

        for epoch in range(1, 11):
            self.flow.train()        # Trainingsmodus        
            total_loss = 0

            for batch, _ in loader:                # _ steht für Labels, die wir nicht brauchen, da unüberwachtes Lernen
                batch = batch.to(self.device)           # [B,1,28,28]
                log_px = self.flow.log_prob(batch)      # Forward + Jacobian
                loss = -log_px.mean()              # Minimiere negative Log-Wahrscheinlichkeit

                self.opt.zero_grad()                    # Gradienten zurücksetzen
                loss.backward()                    # Backprop
                self.opt.step()                         # Parameter-Update

                total_loss += loss.item()

            print(f"Epoch {epoch} | loss: {total_loss/len(loader):.4f}")
        # Modell speichern
        torch.save(self.flow.state_dict(), "checkpoints/flow_mnist.pt")
        self.weights_already_trained = True


    def get_train_loader(batch_size=256):
        train_dataset = DequantizedMNIST(root="./data", train=True)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)