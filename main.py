import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
import pandas as pd


# CONFIG
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 100

CONFUSION_MAP = {
    "Angry": 0.8,
    "Disgust": 0.7,
    "Fear": 0.9,
    "Sad": 0.8,
    "Neutral": 0.4,
    "Happy": 0.2,
    "Surprise": 0.3
}


# SESSION STORAGE
if "confusion_history" not in st.session_state:
    st.session_state.confusion_history = []

if "time_history" not in st.session_state:
    st.session_state.time_history = []

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()


# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()


# FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# VIDEO PROCESSOR
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.confusion_list = []
        self.num_faces = 0
        self.last_process_time = 0
        self.last_faces_data = []
        self.last_avg_conf = 0.0  # NEW

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        current_time = time.time()

        if current_time - self.last_process_time > 0.5:

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )

            self.num_faces = len(faces)
            self.confusion_list = []
            self.last_faces_data = []

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]

                try:
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                except:
                    continue

                face = face / 255.0
                face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

                try:
                    preds = model(face, training=False).numpy()[0]
                except Exception as e:
                    print("Model error:", e)
                    continue

                max_prob = np.max(preds)
                emotion = EMOTIONS[np.argmax(preds)]

                conf_score = CONFUSION_MAP[emotion] * max_prob
                self.confusion_list.append(conf_score)

                self.last_faces_data.append((x, y, w, h, emotion, conf_score))

            #Compute average confusion
            if len(self.confusion_list) > 0:
                self.last_avg_conf = sum(self.confusion_list) / len(self.confusion_list)
            else:
                self.last_avg_conf = 0.0

            self.last_process_time = current_time

        # DRAW
        for (x, y, w, h, emotion, conf_score) in self.last_faces_data:

            if conf_score > 0.6:
                color = (0, 0, 255)
            elif conf_score > 0.3:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            cv2.putText(
                img,
                f"{emotion} ({conf_score:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.putText(
            img,
            f"Faces: {self.num_faces}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# UI
st.set_page_config(page_title="Smart Classroom AI", page_icon="🎭")
st.title("🎭 Smart Classroom - Emotion Detection")

webrtc_ctx = webrtc_streamer(
    key="emotion",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# ANALYTICS 
st.subheader("📊 Classroom Analytics (Live)")

metric_placeholder = st.empty()
status_placeholder = st.empty()
graph_placeholder = st.empty()

# 🔴 CLEAR BUTTON (always visible)
if st.button("🗑️ Clear Graph"):
    st.session_state.confusion_history = []
    st.session_state.time_history = []
    st.session_state.start_time = time.time()
    st.success("Graph cleared!")


# LIVE PROCESSING
if webrtc_ctx.video_processor and webrtc_ctx.state.playing:

    while webrtc_ctx.state.playing:

        processor = webrtc_ctx.video_processor

        avg_conf = processor.last_avg_conf
        faces = processor.num_faces

        elapsed_time = time.time() - st.session_state.start_time

        # Store every 5 seconds
        if len(st.session_state.time_history) == 0 or \
           elapsed_time - st.session_state.time_history[-1] >= 5:

            st.session_state.time_history.append(elapsed_time)
            st.session_state.confusion_history.append(avg_conf)

        # Display metrics
        metric_placeholder.metric(
            "Average Confusion Score",
            f"{avg_conf:.2f}"
        )

        st.write(f"👥 Students Detected: {faces}")

        # Status
        if avg_conf > 0.6:
            status_placeholder.error("⚠️ High Confusion")
        elif avg_conf > 0.3:
            status_placeholder.warning("⚠️ Moderate Confusion")
        else:
            status_placeholder.success("✅ Students Engaged")

        # Graph update
        if len(st.session_state.time_history) > 1:
            df = pd.DataFrame({
                "Time (s)": st.session_state.time_history,
                "Confusion": st.session_state.confusion_history
            })
            graph_placeholder.line_chart(df.set_index("Time (s)"))

        time.sleep(1)

# WHEN VIDEO STOPS → SHOW LAST GRAPH
elif len(st.session_state.time_history) > 1:

    st.info("Session stopped — showing recorded analytics")

    df = pd.DataFrame({
        "Time (s)": st.session_state.time_history,
        "Confusion": st.session_state.confusion_history
    })

    graph_placeholder.line_chart(df.set_index("Time (s)"))

else:
    st.info("Start camera to begin analytics")

#venv\Scripts\activate