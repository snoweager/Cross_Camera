import cv2
import json
import os
from ultralytics import YOLO
from tqdm import tqdm

def run_detection(video_path, model_path, output_json, camera_label):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    detections = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box in results.boxes:
            label = model.names[int(box.cls)]
            if label != 'player':
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            detections.append({
                "frame": frame_num,
                "camera": camera_label,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "label": label
            })

        frame_num += 1

    cap.release()

    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"[✅] Saved {len(detections)} detections to {output_json}")
