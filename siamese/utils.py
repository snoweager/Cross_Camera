import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dists = torch.nn.functional.pairwise_distance(out1, out2)
        loss = label * dists.pow(2) + (1 - label) * (self.margin - dists).clamp(min=0).pow(2)
        return loss.mean()
