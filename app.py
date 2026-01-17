import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fer import FER
import random
import cv2

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="centered"
)

# ------------------ CUSTOM STYLING ------------------
st.markdown("""
<style>
.stApp { background-color: #0f0f0f; }
.main-title { text-align:center; font-size:3rem; color:white; }
.subtitle { text-align:center; color:gray; margin-bottom:20px; }
.card {
    background: rgba(30,30,30,0.9);
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1 class='main-title'>😊 Facial Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image or use webcam to detect emotions</p>", unsafe_allow_html=True)

# ------------------ EMOTION QUOTES ------------------
QUOTES = {
    "happy": ["Happiness is a journey, not a destination."],
    "sad": ["Every storm runs out of rain."],
    "angry": ["Anger fades, calmness stays."],
    "fear": ["Courage starts with facing fear."],
    "surprise": ["Life is full of beautiful surprises."],
    "disgust": ["Trust yourself and stay positive."],
    "neutral": ["Stay calm and move forward."]
}

# ------------------ EMOTION ICONS ------------------
EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}

# ------------------ FUNCTIONS ------------------
def get_random_quote(emotion):
    return random.choice(QUOTES.get(emotion, QUOTES["neutral"]))

def map_emotion(raw_emotion):
    return raw_emotion.lower()   # ✅ Correct mapping

# ------------------ SIDEBAR ------------------
st.sidebar.title("🎯 Input Method")
option = st.sidebar.radio("Choose Input:", ["Upload Image", "Webcam"])

# ------------------ IMAGE INPUT ------------------
image = None

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

elif option == "Webcam":
    cam = st.camera_input("Take a photo")
    if cam:
        image = Image.open(cam).convert("RGB")

# ------------------ PROCESS IMAGE ------------------
if image is not None:

    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Analyzing Emotion..."):
        detector = FER(mtcnn=True)
        result = detector.detect_emotions(np.array(image))

    if not result:
        st.error("No face detected. Try again with better lighting.")
    else:
        face = max(result, key=lambda x: x["box"][2] * x["box"][3])
        emotions = face["emotions"]

        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]

        mapped = map_emotion(dominant_emotion)

        # -------- Display Result --------
        st.markdown("## 🎯 Detected Emotion")
        st.success(f"{EMOJI[mapped]} **{mapped.upper()}**  (Confidence: {confidence:.2f})")

        # -------- Emotion Chart --------
        st.markdown("### 📊 Emotion Probability")
        fig, ax = plt.subplots()
        ax.barh(list(emotions.keys()), list(emotions.values()))
        ax.set_xlabel("Confidence")
        ax.set_title("Emotion Distribution")
        st.pyplot(fig)

        # -------- Quote --------
        st.markdown("### 💬 Motivation")
        st.info(get_random_quote(mapped))

        # -------- Bounding Box --------
        x, y, w, h = face["box"]
        img = np.array(image)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, mapped.upper(), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        st.markdown("### 🧠 Face Detection")
        st.image(img, use_container_width=True)

else:
    st.info("👆 Upload an image or use webcam to begin.")

