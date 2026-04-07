import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scapy.all import rdpcap, IP
import time
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="🔴 Live IP Spoofing IDS", layout="wide", page_icon="📡")

# ====================== LOAD MODEL & SCALER ======================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model/spoof_detector_v1.keras')
    scaler = joblib.load('model/network_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 🔴 Real-Time Behavioral IP Spoofing IDS")
st.markdown("**Simulates live network monitoring** — processes uploaded PCAP as a streaming capture with sliding-window behavioral analysis.")

# ====================== SIDEBAR CONTROLS ======================
st.sidebar.header("Simulation Controls")
speed = st.sidebar.slider("Simulation Speed (ms per window)", 10, 800, 150)
window_size = st.sidebar.slider("Analysis Window Size (packets)", 10, 50, 20)
step_size = st.sidebar.slider("Step Size (packets)", 1, 10, 5)
protected_ip = st.sidebar.text_input("Protected Asset IP", value="192.168.1.50")
uploaded_file = st.sidebar.file_uploader("Upload PCAP File", type=["pcap", "pcapng"])

if not uploaded_file:
    st.info("👆 Upload a mixed traffic PCAP to start real-time simulation")
    st.stop()

# ====================== PROCESS PCAP ======================
temp_pcap = "temp_stream.pcap"
with open(temp_pcap, "wb") as f:
    f.write(uploaded_file.getbuffer())

with st.spinner("Loading PCAP..."):
    packets = rdpcap(temp_pcap)

st.sidebar.success(f"✅ Loaded {len(packets):,} packets")

# ====================== SESSION STATE ======================
if "paused" not in st.session_state:
    st.session_state.paused = False
if "processed_history" not in st.session_state:
    st.session_state.processed_history = []

# ====================== LAYOUT ======================
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    status_kpi = st.empty()
with col2:
    st.subheader("Live Detection Stream")
with col3:
    alert_kpi = st.empty()

chart_placeholder = st.empty()
packet_log = st.empty()
table_placeholder = st.empty()
progress_bar = st.progress(0)

# Control buttons
btn_col1, btn_col2 = st.columns(2)
pause_btn = btn_col1.button("⏸️ Pause" if not st.session_state.paused else "▶️ Resume", 
                           use_container_width=True)
if pause_btn:
    st.session_state.paused = not st.session_state.paused

# ====================== REAL-TIME SIMULATION ======================
processed_history = st.session_state.processed_history
total_windows = (len(packets) - window_size) // step_size + 1

for i in range(0, len(packets) - window_size + 1, step_size):
    if st.session_state.paused:
        time.sleep(0.1)
        continue

    window = packets[i:i + window_size]

    # --- Feature Extraction (exactly matching your trained 6 features) ---
    pkt_sizes = [len(p) for p in window if IP in p]
    if not pkt_sizes:
        continue

    # Timestamps for realistic Packets/s
    try:
        start_time = float(window[0].time)
        end_time = float(window[-1].time)
        time_delta = max(end_time - start_time, 0.001)
        pkts_per_sec = window_size / time_delta
    except:
        pkts_per_sec = 0.0

    # Inbound / Outbound to protected IP
    inbound_pkts = sum(1 for p in window if IP in p and p[IP].dst == protected_ip)
    outbound_pkts = sum(1 for p in window if IP in p and p[IP].src == protected_ip)
    total_ip_pkts = inbound_pkts + outbound_pkts or window_size
    asymmetry = abs(inbound_pkts - outbound_pkts) / total_ip_pkts

    # Forward direction = packets going TO protected IP (typical in reflection attacks)
    fwd_sizes = [len(p) for p in window if IP in p and p[IP].dst == protected_ip]
    avg_fwd_seg = np.mean(fwd_sizes) if fwd_sizes else np.mean(pkt_sizes)
    fwd_min_len = min(fwd_sizes) if fwd_sizes else min(pkt_sizes)

    # The exact 6 features your model was trained on
    current_features = np.array([[
        asymmetry,                    # 0: Asymmetry
        np.mean(pkt_sizes),           # 1: Avg Packet Size
        avg_fwd_seg,                  # 2: Avg Fwd Seg Size
        1.0 if inbound_pkts > 0 else 0.0,  # 3: Inbound Flag
        pkts_per_sec,                 # 4: Packets/s (REAL)
        fwd_min_len                   # 5: Fwd Min Len
    ]])

    # Model inference
    scaled_features = scaler.transform(current_features)
    prediction_prob = model.predict(scaled_features, verbose=0)[0][0]
    is_spoofed = prediction_prob > 0.5

    # --- Record ---
    status_label = "⚠️ SPOOF ATTACK" if is_spoofed else "✅ NORMAL"
    processed_history.append({
        "Window": len(processed_history),
        "Asymmetry": round(asymmetry, 4),
        "Avg_Size": int(np.mean(pkt_sizes)),
        "Pkts_per_Sec": round(pkts_per_sec, 1),
        "Inbound_Pkts": inbound_pkts,
        "Confidence": round(prediction_prob, 4),
        "Status": status_label,
        "Timestamp": datetime.now().strftime("%H:%M:%S")
    })

    df = pd.DataFrame(processed_history)

    # ====================== UPDATE UI ======================
    # KPIs
    with status_kpi.container():
        st.metric("CURRENT STATUS", 
                  status_label,
                  delta="ATTACK DETECTED" if is_spoofed else "SECURE",
                  delta_color="inverse" if is_spoofed else "normal")

    total_alerts = len(df[df['Status'].str.contains("SPOOF")])
    with alert_kpi.container():
        st.metric("Total Spoof Alerts", total_alerts, 
                  delta=f"+{1 if is_spoofed else 0}")

    # Charts
    display_df = df.tail(50)
    with chart_placeholder.container():
        fig = px.line(display_df, x="Window", y="Confidence",
                      title="🔴 Real-Time Spoof Probability (MLP Output)",
                      labels={"Confidence": "Attack Probability"},
                      template="plotly_dark")
        fig.add_hline(y=0.5, line_dash="dash", line_color="#ff4444",
                      annotation_text="Detection Threshold")
        if is_spoofed:
            fig.update_layout(plot_bgcolor='rgba(255, 68, 68, 0.15)')
        st.plotly_chart(fig, use_container_width=True)

        # Second chart - Asymmetry
        fig2 = px.line(display_df, x="Window", y="Asymmetry",
                       title="Asymmetry Trend (Key Spoofing Signature)",
                       template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # Packet log
    with packet_log.expander("📦 Last 8 Packets (Live Log)", expanded=True):
        recent = window[-8:]
        log_data = []
        for p in recent:
            if IP in p:
                log_data.append({
                    "Src": p[IP].src,
                    "Dst": p[IP].dst,
                    "Len": len(p),
                    "Time": f"{float(p.time):.3f}"
                })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True)

    # Recent table
    table_placeholder.table(df.tail(6)[['Window', 'Timestamp', 'Status', 'Confidence', 'Asymmetry', 'Pkts_per_Sec']])

    # Progress
    progress = (i + window_size) / len(packets)
    progress_bar.progress(min(progress, 1.0))

    time.sleep(speed / 1000.0)

# ====================== CLEANUP & SUMMARY ======================
st.success("✅ Simulation Completed!")
os.remove(temp_pcap)

st.subheader("📊 Final Analysis Summary")
final_df = pd.DataFrame(processed_history)
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Windows Analyzed", len(final_df))
col_b.metric("Max Attack Confidence", f"{final_df['Confidence'].max():.1%}")
col_c.metric("Total Spoof Alerts", len(final_df[final_df['Status'].str.contains("SPOOF")]))

st.dataframe(final_df.style.highlight_max(subset=['Confidence'], color='#ff4444'), use_container_width=True)

st.caption("Developed by HUI Students")
