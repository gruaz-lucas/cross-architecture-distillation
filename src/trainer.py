import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, optimizer, dataloader, device="cpu", teacher=None, cfg=None, ignore_index=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.teacher = teacher
        self.cfg = cfg
        self.ignore_index = ignore_index

    def train(self, epochs, save_path):
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total = 0
            correct = 0
            for x, y in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.ignore_index)

                # distillation
                if self.teacher:
                    with torch.no_grad():
                        teacher_logits = self.teacher(x)
                    distill_loss = F.kl_div(
                        F.log_softmax(logits / self.cfg.distillation.temperature, dim=-1),
                        F.softmax(teacher_logits / self.cfg.distillation.temperature, dim=-1),
                        reduction='batchmean'
                    ) * (self.cfg.distillation.temperature ** 2)
                    loss = self.cfg.distillation.alpha * loss + (1 - self.cfg.distillation.alpha) * distill_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                masked_logits = logits[y != self.ignore_index]
                masked_y = y[y != self.ignore_index]
                correct += (masked_logits.argmax(dim=-1) == masked_y).sum().item()
                total += masked_y.numel()
            accuracy = correct / total * 100
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(self.dataloader):.4f}, Accuracy = {accuracy:.2f}%")
            # print example input, target, and prediction
            print("Example:")
            print(f"Input: {x[0].cpu().numpy()}")
            print(f"Target: {y[0].cpu().numpy()}")
            print(f"Prediction: {logits[0].argmax(dim=-1).cpu().numpy()}")



        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
