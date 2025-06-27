import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()  # Remove classifier

        self.encoder = base
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
