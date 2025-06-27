import json
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

def load_embeddings(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_distance_matrix(tacticam_feats, broadcast_feats):
    M, N = len(tacticam_feats), len(broadcast_feats)
    matrix = np.zeros((M, N))

    for i, t in enumerate(tacticam_feats):
        for j, b in enumerate(broadcast_feats):
            d = cosine(t['embedding'], b['embedding'])  # lower is better
            matrix[i, j] = d

    return matrix

def match_players(tacticam_feats, broadcast_feats, threshold=0.4):
    cost_matrix = compute_distance_matrix(tacticam_feats, broadcast_feats)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    for t_idx, b_idx in zip(row_ind, col_ind):
        distance = cost_matrix[t_idx, b_idx]
        if distance < threshold:
            matched.append({
                "tacticam_frame": int(tacticam_feats[t_idx]['frame']),
                "broadcast_frame": int(broadcast_feats[b_idx]['frame']),
                "distance": round(float(distance), 3),
                "tacticam_bbox": [int(x) for x in tacticam_feats[t_idx]['bbox']],
                "broadcast_bbox": [int(x) for x in broadcast_feats[b_idx]['bbox']],
                "player_id": int(b_idx)
            })

    return matched
