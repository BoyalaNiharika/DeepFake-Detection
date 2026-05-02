import streamlit as st
import torch
import cv2
import tempfile
import base64
import os

from model import HybridDeepfakeModel
from utils import extract_frames, predict_video


# --------------------------
# Device Setup
# --------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --------------------------
# Page Config
# --------------------------

st.set_page_config(
    page_title="Deepfake Detection",
    layout="wide"
)

# --------------------------
# Session State
# --------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

# --------------------------
# Load Image Function
# --------------------------

def get_base64_image(image_file):

    if not os.path.exists(image_file):

        st.error(
            f"Image not found: {image_file}"
        )

        return None

    with open(image_file, "rb") as f:

        data = f.read()

    return base64.b64encode(data).decode()

# ====================================================
# 🏠 HOME PAGE
# ====================================================

if st.session_state.page == "home":

    img_base64 = get_base64_image(
        "Images/Home Page.jpeg"
    )

    if img_base64:

        # Full background image
        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image: url("data:image/jpeg;base64,{img_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}

            .title {{
                font-size: 50px;
                font-weight: bold;
                color: white;
                text-align: center;
                text-shadow: 2px 2px 10px black;
                margin-top: 120px;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

        # Title
        st.markdown(
            """
            <div class="title">
                Deepfake Video Detection Using Hybrid Deep Learning
            </div>
            """,
            unsafe_allow_html=True
        )

        # Push button to bottom
        st.markdown("<br>" * 18, unsafe_allow_html=True)

        # Bottom centered button
        col1, col2, col3 = st.columns([2,1,2])

        with col2:

            if st.button(
                "🔍 Start Detection",
                use_container_width=True
            ):

                st.session_state.page = "detect"
                st.rerun()

# ====================================================
# 🎥 DETECTION PAGE
# ====================================================

elif st.session_state.page == "detect":

    st.title("🎭 Deepfake Video Detection Using Hybrid Deep Learning")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(
            delete=False
        )

        tfile.write(uploaded_video.read())

        video_path = tfile.name

        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.video(video_path)

        seq_length = st.slider(
            "Select Sequence Length",
            5,
            30,
            10
        )

        frames = extract_frames(
            video_path,
            seq_length
        )

        st.subheader("🖼 Extracted Frames")

        cols = st.columns(5)

        for i, frame in enumerate(frames):

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            cols[i % 5].image(
                frame_rgb,
                use_container_width=True
            )

        @st.cache_resource
        def load_model():

            model = HybridDeepfakeModel(
                num_classes=2,
                seq_len=10
            ).to(device)

            model.load_state_dict(
                torch.load(
                    "Saved Models/best_model.pt",
                    map_location=device
                )
            )

            model.eval()

            return model

        model = load_model()

        if st.button("🎯 Detect Deepfake"):

            with st.spinner("Processing..."):

                pred, conf = predict_video(
                    model,
                    video_path,
                    seq_length
                )

            if pred == 1:

                st.success(
                    f"REAL VIDEO ({conf:.2f}%)"
                )

            else:

                st.error(
                    f"FAKE VIDEO ({conf:.2f}%)"
                )