# Zomathon 2026: Zero-Touch KPT Pipeline

## Project Overview
This project solves the "noisy data" problem in KPT (Kitchen Prep Time) prediction. Instead of relying on error-prone manual merchant inputs, we utilize **Custom Deep Learning** to detect parcel completion via an edge-computing vision stream.

## Key Features
- **AI-Powered Input:** Replaces subjective human clicks with objective, high-confidence Deep Learning inference.
- **Real-Time Dashboard:** A unified Streamlit interface that streams live video and logs precise timestamps.
- **Zero-Touch Workflow:** The merchant doesn't change their behavior; the AI observes and acts passively.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run app.py`
3. The model will auto-load and begin scanning for your registered parcels.