import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torchvision.transforms as transforms

class PlayerPairDataset(Dataset):
    def __init__(self, data_dir, pairs_json, transform=None):
        self.data_dir = data_dir
        self.pairs = self._load_pairs(pairs_json)
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor()
        ])

    def _load_pairs(self, path):
        import json
        with open(path) as f:
            return json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, pair["img1"])).convert("RGB")
        img2 = Image.open(os.path.join(self.data_dir, pair["img2"])).convert("RGB")
        label = torch.tensor(pair["label"], dtype=torch.float32)

        return self.transform(img1), self.transform(img2), label
