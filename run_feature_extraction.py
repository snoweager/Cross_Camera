from src.features.extract_features import extract_appearance_features
import os

extract_appearance_features(
    video_path="videos/broadcast.mp4",
    detections_path="outputs/broadcast_detections.json",
    output_path="outputs/broadcast_features.json"
)

extract_appearance_features(
    video_path="videos/tacticam.mp4",
    detections_path="outputs/tacticam_detections.json",
    output_path="outputs/tacticam_features.json"
)
