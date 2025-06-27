from src.detection.detect_players import run_detection
import os

# Paths
model_path = "best.pt"
videos_dir = "videos"
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

# Run for both cameras
run_detection(
    video_path=os.path.join(videos_dir, "broadcast.mp4"),
    model_path=model_path,
    output_json=os.path.join(output_dir, "broadcast_detections.json"),
    camera_label="broadcast"
)

run_detection(
    video_path=os.path.join(videos_dir, "tacticam.mp4"),
    model_path=model_path,
    output_json=os.path.join(output_dir, "tacticam_detections.json"),
    camera_label="tacticam"
)
