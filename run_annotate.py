from src.utils.annotate_video import annotate_video

annotate_video(
    video_path="videos/broadcast.mp4",
    features_path="outputs/broadcast_features.json",
    id_map_path="outputs/player_id_map.json",
    output_path="outputs/broadcast_annotated.mp4",
    camera_label="broadcast"
)

annotate_video(
    video_path="videos/tacticam.mp4",
    features_path="outputs/tacticam_features.json",
    id_map_path="outputs/player_id_map.json",
    output_path="outputs/tacticam_annotated.mp4",
    camera_label="tacticam"
)
