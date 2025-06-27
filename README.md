# Cross-Camera Player Re-Identification

This project solves the problem of **player re-identification** across multiple camera angles in sports broadcasts using deep learning and computer vision techniques.

> Goal: Ensure that the same player retains the **same ID** even when captured from **different cameras** or across **video transitions**.

---

## Project Structure

```

Cross\_Camera/
├── notebooks/                # Jupyter notebooks
├── src/                     # Traditional pipeline (detection, features, matching)
│   ├── detection/           # Player detection (YOLOv11)
│   ├── features/            # Feature extraction (ResNet, etc.)
│   ├── matching/            # Matching logic
│   └── utils/               # Utilities
├── siamese/                 # Deep Re-ID using Siamese Network
├── outputs/                 # All generated outputs
│   ├── broadcast\_annotated.mp4
│   ├── tacticam\_annotated.mp4
│   ├── \*\_detections.json
│   ├── \*\_features.json
│   ├── matches.json
│   ├── player\_id\_map.json
│   ├── crops/               # Cropped player images
├── videos/                  # Input videos
├── models/                  # Trained models
├── best.pt                 # Provided YOLOv11 checkpoint
├── app.py                  # Streamlit dashboard for visual inspection
├── README.md
└── report.md

````

---

## Pipeline Overview

### Detection
- **YOLOv11** is used to detect players in both `broadcast.mp4` and `tacticam.mp4`.
- Detections are saved as JSON for reproducibility.

### Feature Extraction
- Players are cropped and resized to `128x256`.
- Two approaches are used:
  - **Baseline**: ResNet50 pre-trained on ImageNet
  - **Advanced**: Siamese Network trained on player pair similarity

### Matching Across Cameras
- Extracted embeddings are compared using:
  - Cosine similarity
  - Hungarian matching algorithm
- Matching results saved in `matches.json`.

### Assign Consistent IDs
- A mapping dictionary aligns player identities between `tacticam` and `broadcast`.

### Annotated Output
- `cv2.putText()` is used to annotate players across both videos with same consistent ID.
- Saved as:
  - `outputs/tacticam_annotated.mp4`
  - `outputs/broadcast_annotated.mp4`

### Visual Inspector (Streamlit)
```bash
streamlit run app.py
````

* View crops from both views
* Check match quality and similarity scores

---

## Deep Re-ID (Siamese Network)

### Key Features:

* Learns visual similarity between same-player pairs
* Uses contrastive loss
* Trained using:

  * `positive` = same player across cams
  * `negative` = different players

### Files:

* `siamese/model.py`: Architecture
* `siamese/train_siamese.py`: Training logic
* `siamese/embeddings_extractor.py`: Feature replacement
* `siamese/dataset.py`: Player pair dataset

---

## Evaluation

| Metric       | Description                              |
| ------------ | ---------------------------------------- |
| Visual       | Matching looks consistent frame to frame |
| Quantitative | Optional metrics like precision/recall   |
| Annotated    | Clear color-coded IDs over time          |

---

## Bonus Features

* Streamlit dashboard for top-10 visual matches
* Siamese Network integration
* Modular + well-structured pipeline
* Outputs (JSON/MP4) for reproducibility

---

## Dependencies

```bash
pip install -r requirements.txt
```

**Key Libraries:**

* `Ultralytics YOLOv11`
* `PyTorch`
* `OpenCV`
* `Streamlit`
* `SciPy`, `NumPy`, `Pillow`, etc.

---

## Author

Padma Sindhoora Ayyagari
mailto: sindhoora.ayyagari@gmail.com
---

## Final Notes

This project demonstrates a scalable, end-to-end solution to cross-camera player re-identification using both traditional and deep learning methods. The modularity allows future extensions like:

* Deep SORT tracking
* Temporal smoothing
* Real-time inference
* Evaluation metrics (MOTA/MOTP)

---

To Note: the best.pt - is given as a link here:
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

