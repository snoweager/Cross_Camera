import streamlit as st
from PIL import Image
import json

st.title("Player Re-ID Visual Inspector")

with open("outputs/matches.json") as f:
    matches = json.load(f)

def bbox_to_str(bbox):
    return "_".join(str(v) for v in bbox)

# Show top-10 matched players
for match in matches[:10]:
    col1, col2 = st.columns(2)

    tacticam_frame = match["tacticam_frame"]
    tacticam_bbox_str = bbox_to_str(match["tacticam_bbox"])
    tacticam_path = f"outputs/crops/tacticam/{tacticam_frame}_{tacticam_bbox_str}.jpg"

    broadcast_frame = match["broadcast_frame"]
    broadcast_bbox_str = bbox_to_str(match["broadcast_bbox"])
    broadcast_path = f"outputs/crops/broadcast/{broadcast_frame}_{broadcast_bbox_str}.jpg"

    with col1:
        st.image(tacticam_path, caption="Tacticam")
    with col2:
        st.image(broadcast_path, caption="Broadcast")

    st.text(f"Cosine Distance: {match['distance']}")
