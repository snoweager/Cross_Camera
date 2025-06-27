import os
import json
import cv2

def crop_players(video_path, features_path, output_dir):
    with open(features_path, 'r') as f:
        data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    frames = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for det in data:
        frame_id = det["frame"]
        bbox = det["bbox"]  # [x1, y1, x2, y2]

        if frame_id not in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            frames[frame_id] = frame
        else:
            frame = frames[frame_id]

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        crop_filename = f"{frame_id}_{'_'.join(map(str, bbox))}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)

        if crop.size > 0:
            cv2.imwrite(crop_path, crop)

    cap.release()
    print(f"Saved {len(data)} crops to {output_dir}")

if __name__ == "__main__":
    crop_players(
        video_path="videos/tacticam.mp4",
        features_path="outputs/tacticam_features.json",
        output_dir="outputs/crops/tacticam"
    )
    crop_players(
        video_path="videos/broadcast.mp4",
        features_path="outputs/broadcast_features.json",
        output_dir="outputs/crops/broadcast"
    )
