import torch
from torchvision import transforms
from PIL import Image
from siamese.model import SiameseNet
import os
import json

def extract_embeddings(img_dir, output_json):
    model = SiameseNet()
    model.load_state_dict(torch.load("models/siamese_model.pt"))
    model.eval().cuda()

    transform = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor()
    ])

    embeddings = []
    for img_file in os.listdir(img_dir):
        if not img_file.endswith(".jpg"):
            continue

        path = os.path.join(img_dir, img_file)
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).cuda()

        with torch.no_grad():
            emb = model.forward_once(img_tensor).cpu().squeeze().tolist()

        embeddings.append({
            "img": img_file,
            "embedding": emb
        })

    with open(output_json, 'w') as f:
        json.dump(embeddings, f, indent=2)

    print(f"🔐 Embeddings saved to {output_json}")

# Usage:
# extract_embeddings("outputs/crops/tacticam", "outputs/tacticam_siamese_embeddings.json")
# extract_embeddings("outputs/crops/broadcast", "outputs/broadcast_siamese_embeddings.json")
