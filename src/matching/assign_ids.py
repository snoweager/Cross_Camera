import json
import os

def assign_consistent_ids(match_file, output_file):
    with open(match_file, 'r') as f:
        matches = json.load(f)

    player_map = {}

    for m in matches:
        tacticam_frame = m["tacticam_frame"]
        broadcast_frame = m["broadcast_frame"]
        tacticam_bbox = m["tacticam_bbox"]
        broadcast_bbox = m["broadcast_bbox"]
        broadcast_id = int(m["player_id"])

        # Build keys for tacticam and broadcast
        tacticam_key = f"tacticam_{tacticam_frame}_{'_'.join(map(str, tacticam_bbox))}"
        broadcast_key = f"broadcast_{broadcast_frame}_{'_'.join(map(str, broadcast_bbox))}"

        player_map[tacticam_key] = broadcast_id
        player_map[broadcast_key] = broadcast_id

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(player_map, f, indent=2)

    print(f"Saved {len(player_map)} total mappings for both tacticam and broadcast to '{output_file}'")
