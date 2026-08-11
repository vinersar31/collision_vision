"""CollisionVision — Streamlit app for interactive car-damage segmentation.

Upload a photo of a crashed car and view YOLOv8-Seg instance masks overlaid on
the image, alongside a per-instance breakdown of the detected damage.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from collision_vision.inference import DamageSegmenter
from collision_vision.visualize import MaskVisualizer

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = "models/best.pt"


@st.cache_resource(show_spinner="Loading segmentation model...")
def load_segmenter(weights: str) -> DamageSegmenter:
    """Load (and cache) the damage segmenter for a given weights path."""
    return DamageSegmenter(weights=weights)


def main() -> None:
    st.set_page_config(page_title="CollisionVision", page_icon="🚗", layout="wide")
    st.title("CollisionVision — Car Damage Segmentation")
    st.caption(
        "Instance segmentation of dents, scratches and structural damage with YOLOv8-Seg."
    )

    # --- Sidebar controls --------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        weights = st.text_input("Model weights (.pt)", value=DEFAULT_WEIGHTS)
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
        opacity = st.slider("Mask opacity", 0.1, 0.9, 0.4, 0.05)
        show_contours = st.checkbox("Draw contours", value=True)
        show_labels = st.checkbox("Draw labels", value=True)

    uploaded = st.file_uploader(
        "Upload a photo of a crashed car", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        st.info("Upload an image to run damage segmentation.")
        return

    weights_path = Path(weights).resolve()
    if not weights_path.is_relative_to(Path.cwd()):
        st.error(f"Invalid weights path '{weights}'. Path traversal is not allowed.")
        return

    if not weights_path.is_file():
        st.error(
            f"Weights not found at '{weights}'. Train a model first "
            "(`collision-vision train`) or point to a valid .pt file."
        )
        return

    # --- Inference ---------------------------------------------------------
    image_rgb = np.array(Image.open(uploaded).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        segmenter = load_segmenter(weights)
    except Exception:
        logger.exception("Failed to load segmentation model from '%s'", weights)
        st.error(
            "An unexpected error occurred while loading the model. Please check the logs."
        )
        return

    with st.spinner("Detecting damage..."):
        instances = segmenter.predict(image_bgr, conf=confidence)

    visualizer = MaskVisualizer(alpha=opacity)
    overlay_bgr = visualizer.overlay(
        image_bgr, instances, draw_contours=show_contours, draw_labels=show_labels
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # --- Results -----------------------------------------------------------
    left, right = st.columns(2)
    left.subheader("Original")
    left.image(image_rgb, use_container_width=True)
    right.subheader("Detected damage")
    right.image(overlay_rgb, use_container_width=True)

    if not instances:
        st.warning("No damage detected. Try lowering the confidence threshold.")
        return

    st.subheader(f"Detected {len(instances)} damage region(s)")
    table = [
        {
            "Damage type": inst.class_name,
            "Confidence": f"{inst.confidence:.1%}",
            "Area (px)": inst.area_px,
            "Area (% of image)": f"{inst.area_fraction(image_bgr.shape):.2%}",
        }
        for inst in instances
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
