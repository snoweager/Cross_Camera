import cv2
import json
import os
from tqdm import tqdm

def annotate_video(video_path, features_path, id_map_path, output_path, camera_label):
    # Load player features
    with open(features_path, 'r') as f:
        features = json.load(f)

    # Load ID mapping
    with open(id_map_path, 'r') as f:
        id_map = json.load(f)

    # Prepare video
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_dict = {}
    for feat in features:
        frame_num = feat['frame']
        frame_dict.setdefault(frame_num, []).append(feat)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_num in tqdm(range(total_frames), desc=f"Annotating {camera_label}"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in frame_dict:
            for det in frame_dict[frame_num]:
                x1, y1, x2, y2 = det['bbox']
                key = f"{camera_label}_{frame_num}_{x1}_{y1}_{x2}_{y2}"

                if key in id_map:
                    pid = id_map[key]
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Label
                    cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")
