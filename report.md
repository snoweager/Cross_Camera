# Cross-Camera Player Re-Identification Report

## Objective

The goal of this project is to **re-identify players across two camera views** (`broadcast.mp4` and `tacticam.mp4`) from the same sports match. Each player should retain a **consistent identity (ID)** even when captured from different angles or after temporary occlusions.

---

## Problem Definition

Player re-identification (Re-ID) is a key challenge in sports analytics, requiring consistent tracking of individuals across diverse views.

This task requires:
- **Object detection** (to localize players)
- **Feature extraction** (to describe each player)
- **Matching algorithm** (to compare and re-ID players)

---

## Methodology

### Player Detection
- **Model**: Ultralytics YOLOv11 (fine-tuned)
- **Input**: `broadcast.mp4` and `tacticam.mp4`
- **Output**: Bounding boxes in `*_detections.json`

### Cropping & Preprocessing
- Detected player crops are saved and resized to `128x256`
- Used in both traditional and Siamese workflows

### Feature Extraction
#### Traditional
- **Model**: ResNet50 (pretrained on ImageNet)
- **Output**: 2048-dim appearance embedding
- Normalized with L2 norm

#### Deep Re-ID (Siamese)
- **Model**: Twin CNN with contrastive loss
- **Training**: Positive = same player, Negative = different
- **Output**: Custom embeddings for fine-grained similarity

### Matching
- **Distance metric**: Cosine similarity
- **Assignment**: Hungarian algorithm
- **Threshold**: Similarity score < 0.4
- **Output**: `matches.json`

### ID Assignment
- Builds global player ID mapping (`player_id_map.json`)
- Ensures tacticam ↔ broadcast alignment
- Stored per-frame for both videos

### Annotation
- `cv2.putText` + OpenCV drawing
- Annotated videos:
  - `outputs/tacticam_annotated.mp4`
  - `outputs/broadcast_annotated.mp4`

### Visual Inspector
- Built with **Streamlit** (`app.py`)
- Top-10 match viewer with:
  - Player crops
  - Cosine distance
  - View consistency

---

## Results

### Qualitative Evaluation
- IDs are consistent between views for most players.
- Siamese model improves robustness over ResNet baseline.
- Annotated videos clearly show consistent identity labeling.

### Optional Quantitative
- With ground truth:
  - Precision/Recall
  - ID switches
  - MOTA/MOTP

---

## Files & Outputs

| File                          | Description                              |
|------------------------------|------------------------------------------|
| `matches.json`               | Matched player pairs with distances      |
| `*_features.json`            | Embedding vectors                        |
| `*_detections.json`          | Bounding box detections                  |
| `*_annotated.mp4`            | Final annotated output videos            |
| `crops/`                     | Cropped and resized player images        |
| `player_id_map.json`         | Global ID consistency mapping            |
| `siamese/`                   | Deep re-ID training and inference code   |

---

## Highlights

- Combines traditional + deep learning-based Re-ID
- Accurate player tracking across views
- Clean modular pipeline
- Visual dashboard for analysis
- Reproducible JSON & MP4 outputs

---

## Challenges Encountered

- **ID Switches in Overlapping Regions**:  
  Matching failed for players too close in proximity or overlapping in space.

- **Training Siamese Model**:  
  Creating a well-balanced dataset of positive/negative pairs was tricky and required augmentations.

- **File Management**:  
  Keeping consistent paths and JSON structure across steps required careful attention.

- **Streamlit Crashes**:  
  Errors arose when referenced crops were missing or misnamed — fixed with ID formatting.


## Bonus Features Implemented

- Siamese network-based deep visual re-ID
- Streamlit visual inspector dashboard
- Color-coded ID annotation
- Cropped player gallery

---

## Future Improvements

- Train Siamese on larger player pair dataset
- Integrate Deep SORT for tracking over time
- Add spatial/temporal features into embeddings
- Use optical flow or keypoint estimation
- Implement full evaluation metrics with labels

---

## Credits

- YOLOv11 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Siamese Network built using PyTorch
- Visualized with OpenCV and Streamlit

---

*For code and documentation, see* `README.md`  
*All videos and data saved under* `outputs/`

---
