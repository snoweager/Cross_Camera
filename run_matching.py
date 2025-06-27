import os
import json
from src.matching.match_players import match_players, load_embeddings

# Load features
tacticam_feats = load_embeddings("outputs/tacticam_features.json")
broadcast_feats = load_embeddings("outputs/broadcast_features.json")

# Run matching
matched = match_players(tacticam_feats, broadcast_feats)

# Save match results to file
os.makedirs("outputs", exist_ok=True)
with open("outputs/matches.json", "w") as f:
    json.dump(matched, f, indent=2)

print(f"Matched {len(matched)} players across cameras and saved to 'outputs/matches.json'")
