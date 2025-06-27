import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from siamese.dataset import Siamese
from siamese.model import Siamese

dataset = Siamese("outputs/player_matches.json", "outputs/crops/")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Siamese().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def contrastive_loss(x1, x2, y, margin=1.0):
    dist = F.pairwise_distance(x1, x2)
    return torch.mean((1 - y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(margin - dist, min=0.0), 2))

for epoch in range(10):
    model.train()
    total_loss = 0
    for img1, img2, label in loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        out1, out2 = model(img1, img2)
        loss = contrastive_loss(out1, out2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "models/siamese_model.pth")
