import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scapy.all import rdpcap, IP
import time
import plotly.express as px

st.set_page_config(page_title="Real-Time IP Spoof IDS", layout="wide")

# --- Load Model & Scaler ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model/spoof_detector_v1.keras')
    scaler = joblib.load('model/network_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 Real-Time Behavioral Network IDS")
st.sidebar.header("Simulation Control")
speed = st.sidebar.slider("Simulation Speed (ms delay)", 10, 500, 100)
uploaded_file = st.sidebar.file_uploader("Upload Mixed Stream PCAP", type=["pcap"])

if uploaded_file:
    # Save temp
    with open("stream.pcap", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    packets = rdpcap("stream.pcap")
    st.sidebar.success(f"Loaded {len(packets)} packets.")
    
    # Dashboard Placeholders
    kpi1, kpi2, kpi3 = st.columns(3)
    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    
    # State tracking
    processed_data = []
    
    # --- REAL-TIME SIMULATION LOOP ---
    # We process in chunks of 20 packets to simulate "Flow Windows"
    step = 20
    for i in range(0, len(packets) - step, step):
        window = packets[i : i + step]
        
        # Feature Extraction for this specific window
        # (Simplified logic for the simulation demo)
        fwd = sum(1 for p in window if p[IP].dst == "192.168.1.50")
        bwd = len(window) - fwd
        total = len(window)
        
        asymmetry = abs(fwd - bwd) / total
        avg_size = np.mean([len(p) for p in window])
        # Mapping other 4 features to window constants for demo stability
        feats = np.array([[asymmetry, avg_size, avg_size, 1.0, 100.0, 64.0]])
        
        # Inference
        scaled_feats = scaler.transform(feats)
        prob = model.predict(scaled_feats, verbose=0)[0][0]
        is_spoofed = prob > 0.5
        
        # Update Data
        status = "⚠️ SPOOF" if is_spoofed else "✅ NORMAL"
        processed_data.append({
            "Time": i,
            "Asymmetry": round(asymmetry, 2),
            "Avg Size": int(avg_size),
            "Confidence": round(prob, 4),
            "Status": status
        })
        
        # Keep only last 50 for the chart
        df = pd.DataFrame(processed_data)
        
        # Update Dashboard
        kpi1.metric("Current Status", status, delta="Attack Detected" if is_spoofed else "Clear", delta_color="inverse" if is_spoofed else "normal")
        kpi2.metric("Mean Asymmetry", f"{df['Asymmetry'].mean():.2f}")
        kpi3.metric("Detection Count", len(df[df['Status'] == "⚠️ SPOOF"]))
        
        with chart_placeholder.container():
            fig = px.line(df.suffix(50), x="Time", y="Confidence", title="Live Network Probability Stream")
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Detection Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
        table_placeholder.table(df.tail(5))
        
        time.sleep(speed / 1000)

    st.balloons()
    st.success("Simulation Complete: All packets analyzed.")
