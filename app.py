import streamlit as st
import cv2
import numpy as np
import time
import os

# --- THE MAGIC FIX: Forces compatibility with Teachable Machine models ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tensorflow.keras.models import load_model

# --- 1. Load the AI Model (Cached for Speed) ---
@st.cache_resource
def load_teachable_machine_model():
    # Load the model
    model = load_model("keras_model.h5", compile=False)
    # Load the labels
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_teachable_machine_model()

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

# --- 4. Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric(label="Order ID", value="ZOM-7782")

if st.session_state.status == "Preparing":
    col2.metric(label="Kitchen Status", value="👨‍🍳 Preparing (Live AI Feed)")
else:
    col2.metric(label="Kitchen Status", value="✅ Ready for Pickup")
    
col3.metric(label="Calculated KPT", value=st.session_state.kpt)

st.divider()

# --- 5. The Embedded AI Camera Logic ---
if st.session_state.camera_active:
    st.subheader("Live Deep Learning Dispatch Feed")
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    frames_detected = 0
    
    # CONFIGURATION: Minimum time required for a realistic order (Seconds)
    MIN_PREP_THRESHOLD = 150 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
            
        # --- Deep Learning Pre-processing ---
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        
        # --- AI Inference ---
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        
        # Calculate current elapsed time
        elapsed_seconds = int(time.time() - st.session_state.start_time)

        # --- Trigger Logic with Sanity Check ---
        # Checks for "parcel" in your label name
        if "parcel" in class_name.lower() and confidence_score > 0.85:
            
            # IF DETECTED TOO EARLY (Potential Fraud or False Positive)
            if elapsed_seconds < MIN_PREP_THRESHOLD:
                cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 4)
                cv2.putText(frame, f"WAIT: Preparing... ({elapsed_seconds}s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                frames_detected = 0
            
            # VALID COMPLETION
            else:
                cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 255, 0), 4)
                cv2.putText(frame, f"PARCEL DETECTED: {confidence_score*100:.0f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                frames_detected += 1
                if frames_detected >= 3:
                    st.session_state.status = "Ready"
                    st.session_state.kpt = f"{elapsed_seconds} seconds"
                    st.session_state.camera_active = False 
                    cap.release()
                    st.rerun()
        else:
            # Normal Scanning State
            cv2.putText(frame, f"Scanning Counter... (Elapsed: {elapsed_seconds}s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frames_detected = 0 
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

# --- 6. Reset / New Order Button ---
if not st.session_state.camera_active:
    st.success(f"Order Completed in {st.session_state.kpt}! Data synced to Zomato Logistics.")
    st.divider()
    if st.button("🔄 Start Next Order (Demo Reset)"):
        st.session_state.status = "Preparing"
        st.session_state.kpt = "--"
        st.session_state.camera_active = True
        st.session_state.start_time = time.time()
        st.rerun()
