import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scapy.all import rdpcap, IP
import time
import plotly.express as px
import os

st.set_page_config(page_title="Real-Time IP Spoof IDS", layout="wide")

# --- Load Model & Scaler ---
@st.cache_resource
def load_assets():
    # Adjusted paths to match your GitHub structure
    model = tf.keras.models.load_model('model/spoof_detector_v1.keras')
    scaler = joblib.load('model/network_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 Real-Time Behavioral Network IDS")
st.markdown("This dashboard simulates a live network environment by streaming packets from a PCAP file and performing window-based behavioral analysis.")

st.sidebar.header("Simulation Control")
speed = st.sidebar.slider("Simulation Speed (ms delay)", 10, 1000, 200)
uploaded_file = st.sidebar.file_uploader("Upload Mixed Stream PCAP", type=["pcap"])

if uploaded_file:
    # Save temp file for Scapy to read
    temp_pcap = "stream_buffer.pcap"
    with open(temp_pcap, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Initializing Network Stream..."):
        packets = rdpcap(temp_pcap)
    
    st.sidebar.success(f"Stream Loaded: {len(packets)} packets")
    
    # Dashboard Layout
    kpi1, kpi2, kpi3 = st.columns(3)
    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    
    # State tracking
    processed_history = []
    
    # --- REAL-TIME SIMULATION LOOP ---
    # We process in 'Windows' of 20 packets to calculate behavior
    window_size = 20
    
    for i in range(0, len(packets) - window_size, 5): # Step by 5 for smoother sliding window
        window = packets[i : i + window_size]
        
        # 1. Feature Extraction from the current window
        pkt_sizes = [len(p) for p in window]
        avg_size = np.mean(pkt_sizes)
        
        # Calculate Asymmetry in this window (Forward vs Backward)
        # Assuming 192.168.1.50 is the internal asset being protected
        inbound_pkts = sum(1 for p in window if IP in p and p[IP].dst == "192.168.1.50")
        outbound_pkts = window_size - inbound_pkts
        asymmetry = abs(inbound_pkts - outbound_pkts) / window_size
        
        # Prepare feature vector for model (Must match the 6 features trained)
        # Features: [Asymmetry, Avg Size, Avg Fwd Seg, Inbound, Pkts/s, Fwd Min Len]
        # We derive the others or use window-specific medians
        current_features = np.array([[
            asymmetry, 
            avg_size, 
            avg_size, # Avg Fwd Seg (approximated for window)
            1.0 if inbound_pkts > 0 else 0.0, 
            window_size / 0.1, # Simulated high packets/s
            min(pkt_sizes)
        ]])
        
        # 2. Model Inference
        scaled_features = scaler.transform(current_features)
        prediction_prob = model.predict(scaled_features, verbose=0)[0][0]
        is_spoofed = prediction_prob > 0.5
        
        # 3. Update History
        status_label = "⚠️ SPOOF ATTACK" if is_spoofed else "✅ NORMAL TRAFFIC"
        processed_history.append({
            "Window": i // 5,
            "Asymmetry": round(asymmetry, 3),
            "Avg_Packet_Size": int(avg_size),
            "Inbound_Ratio": inbound_pkts / window_size,
            "Confidence": round(prediction_prob, 4),
            "Status": status_label
        })
        
        # Convert to DataFrame for visualization
        df = pd.DataFrame(processed_history)
        
        # 4. Update UI Elements
        # Using .tail(50) for the moving window chart
        display_df = df.tail(50)
        
        # KPI Update
        kpi1.metric("Current Status", status_label, 
                   delta="SPOOFING DETECTED" if is_spoofed else "SECURE", 
                   delta_color="inverse" if is_spoofed else "normal")
        
        kpi2.metric("Window Asymmetry", f"{asymmetry:.2f}")
        
        total_alerts = len(df[df['Status'] == "⚠️ SPOOF ATTACK"])
        kpi3.metric("Total Detections", total_alerts)
        
        # Plotly Graph
        with chart_placeholder.container():
            fig = px.line(display_df, x="Window", y="Confidence", 
                          title="Real-Time Probability Stream (MLP Inference)",
                          range_y=[0, 1])
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                          annotation_text="Detection Threshold (0.5)")
            # Add a colored background if attack is detected
            if is_spoofed:
                fig.update_layout(plot_bgcolor='rgba(255,0,0,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Recent Alert Table
        table_placeholder.table(df.tail(5)[['Window', 'Status', 'Confidence', 'Asymmetry']])
        
        # Control simulation speed
        time.sleep(speed / 1000)

    st.success("Network Stream Analysis Finalized.")
    os.remove(temp_pcap) # Clean up
