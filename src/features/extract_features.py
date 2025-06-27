import json
import os
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
import numpy as np

def extract_appearance_features(video_path, detections_path, output_path):
    # Load detections
    with open(detections_path, 'r') as f:
        detections = json.load(f)

    # Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pretrained ResNet model (remove last layer)
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    features = []
    frame_cache = {}

    for det in tqdm(detections):
        frame_id = det['frame']

        # Cache frames to avoid duplicate decoding
        if frame_id not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_id] = frame
        else:
            frame = frame_cache[frame_id]

        x1, y1, x2, y2 = det['bbox']
        cropped = frame[y1:y2, x1:x2]

        # If detection is too small, skip
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue

        # Transform and extract features
        with torch.no_grad():
            img_tensor = transform(cropped).unsqueeze(0)
            feat = model(img_tensor).squeeze().numpy()
            feat = feat / np.linalg.norm(feat)  # L2 normalization

        det['embedding'] = feat.tolist()
        features.append(det)

    # Save features
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)

    print(f"Extracted features for {len(features)} players -> {output_path}")
