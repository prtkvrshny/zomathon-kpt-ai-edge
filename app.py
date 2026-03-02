import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# --- 1. Load the AI Model ---
@st.cache_resource
def load_zomato_model():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

# --- 2. Session State Initialization ---
if 'status' not in st.session_state:
    st.session_state.status = "Preparing"
if 'kpt' not in st.session_state:
    st.session_state.kpt = "--"
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = True
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# --- 3. UI Configuration ---
st.set_page_config(page_title="Zomato Smart Kitchen", layout="wide")
st.title("🔴 Zomato Partner: Zero-Touch KPT Dashboard")
st.markdown("Automated prep-time tracking powered by **Deep Learning & Edge Computer Vision.**")
st.divider()

model, class_names = load_zomato_model()

# --- 4. Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric(label="Order ID", value="ZOM-7782")

if st.session_state.status == "Preparing":
    col2.metric(label="Kitchen Status", value="👨‍🍳 Preparing (AI Monitoring)")
else:
    col2.metric(label="Kitchen Status", value="✅ Ready for Pickup")

col3.metric(label="Calculated KPT", value=st.session_state.kpt)
st.divider()

# --- 5. Cloud-Friendly AI Camera Logic ---
if st.session_state.camera_active:
    st.subheader("Cloud Demo: Scan Parcel to Verify Dispatch")
    
    # Native Streamlit camera for cloud deployment
    img_file_buffer = st.camera_input("Place parcel in frame and take snapshot")
    
    if img_file_buffer is not None:
        # Convert image for OpenCV/TensorFlow
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Deep Learning Pre-processing
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        # AI Inference
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        elapsed_seconds = int(time.time() - st.session_state.start_time)
        MIN_PREP_THRESHOLD = 30 # Anti-fraud timer

        # Trigger Logic
        if "parcel" in class_name.lower() and confidence_score > 0.85:
            if elapsed_seconds < MIN_PREP_THRESHOLD:
                st.error(f"⚠️ **WAIT: Minimum Prep Time Not Reached.** ({elapsed_seconds}s / {MIN_PREP_THRESHOLD}s). This prevents fraudulent early-clicks.")
            else:
                st.success(f"✅ **PARCEL VERIFIED** (Confidence: {confidence_score*100:.0f}%). Dispatching Rider!")
                st.session_state.status = "Ready"
                st.session_state.kpt = f"{elapsed_seconds} seconds"
                st.session_state.camera_active = False
                time.sleep(2) # Brief pause so they can read the success message
                st.rerun()
        else:
            st.warning(f"No valid Zomato parcel detected. (AI Confidence for Parcel: {prediction[0][class_names.index('parcel')] if 'parcel' in class_names else 0:.0f}%)")

# --- 6. Reset Button ---
if not st.session_state.camera_active:
    st.success(f"Order Completed! Final KPT: {st.session_state.kpt}. Data synced to Zomato Logistics.")
    st.divider()
    if st.button("🔄 Start Next Order (Demo Reset)"):
        st.session_state.status = "Preparing"
        st.session_state.kpt = "--"
        st.session_state.camera_active = True
        st.session_state.start_time = time.time()
        st.rerun()
