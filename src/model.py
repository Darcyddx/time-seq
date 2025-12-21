import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureToScoreLayer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.fc = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
