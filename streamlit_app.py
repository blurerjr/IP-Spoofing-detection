import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scapy.all import rdpcap, IP, TCP, UDP
import time
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI IP Spoofing Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ AI-Driven IP Spoofing Detection Dashboard")
st.markdown("""
**Thesis Project Simulation Dashboard** This tool analyzes network packet captures (.pcap) using a Deep Learning (MLP) model to detect IP Spoofing signatures based on flow asymmetry and packet behaviors.
""")

# --- 2. LOAD DEPLOYMENT ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('model/spoof_detector_v1.keras')
        scaler = joblib.load('model/network_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models. Please ensure the 'model' folder exists in the repo. Details: {e}")
        return None, None

model, scaler = load_artifacts()

# --- 3. FEATURE EXTRACTION ENGINE (SCAPY) ---
def extract_flow_features(pcap_path, target_ip):
    """
    Simulates CICFlowMeter logic to extract the 6 required features.
    """
    flows = {}
    packets = rdpcap(pcap_path)
    
    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            
            # Define flow key (bidirectional)
            flow_key = tuple(sorted([src_ip, dst_ip]))
            
            if flow_key not in flows:
                flows[flow_key] = {
                    'fwd_pkts': 0, 'bwd_pkts': 0,
                    'fwd_bytes': 0, 'bwd_bytes': 0,
                    'start_time': pkt.time, 'end_time': pkt.time,
                    'fwd_min_len': float('inf'),
                    'src_ip': src_ip, 'dst_ip': dst_ip
                }
            
            flow = flows[flow_key]
            flow['end_time'] = max(flow['end_time'], pkt.time)
            pkt_len = len(pkt)
            
            # Determine direction based on the first packet seen
            if src_ip == flow['src_ip']:
                flow['fwd_pkts'] += 1
                flow['fwd_bytes'] += pkt_len
                flow['fwd_min_len'] = min(flow['fwd_min_len'], pkt_len)
            else:
                flow['bwd_pkts'] += 1
                flow['bwd_bytes'] += pkt_len

    # Calculate final features for the model
    features_list = []
    flow_records = []
    
    for key, data in flows.items():
        total_pkts = data['fwd_pkts'] + data['bwd_pkts']
        duration = max(float(data['end_time'] - data['start_time']), 0.0001)
        
        # 1. Asymmetry Ratio
        asymmetry_ratio = abs(data['fwd_pkts'] - data['bwd_pkts']) / total_pkts if total_pkts > 0 else 0
        # 2. Average Packet Size
        avg_pkt_size = (data['fwd_bytes'] + data['bwd_bytes']) / total_pkts if total_pkts > 0 else 0
        # 3. Avg Fwd Segment Size
        avg_fwd_seg_size = data['fwd_bytes'] / data['fwd_pkts'] if data['fwd_pkts'] > 0 else 0
        # 4. Inbound (1 if targeting our defined victim IP, 0 otherwise)
        inbound = 1.0 if (data['dst_ip'] == target_ip or data['src_ip'] == target_ip) else 0.0
        # 5. Flow Packets/s
        flow_pkts_s = total_pkts / duration
        # 6. Fwd Packet Length Min
        fwd_pkt_len_min = data['fwd_min_len'] if data['fwd_min_len'] != float('inf') else 0

        features = [asymmetry_ratio, avg_pkt_size, avg_fwd_seg_size, inbound, flow_pkts_s, fwd_pkt_len_min]
        features_list.append(features)
        
        flow_records.append({
            'Endpoint 1': key[0],
            'Endpoint 2': key[1],
            'Duration (s)': round(duration, 4),
            'Total Packets': total_pkts
        })
        
    return pd.DataFrame(features_list), pd.DataFrame(flow_records)

# --- 4. STREAMLIT UI & LOGIC ---
st.sidebar.header("Simulation Settings")
target_ip = st.sidebar.text_input("Defending IP (Target)", value="192.168.1.50", help="The IP address acting as the victim in the PCAP.")

uploaded_file = st.sidebar.file_uploader("Upload a .pcap file", type=["pcap"])

if uploaded_file is not None and model is not None:
    with st.spinner('Parsing packets and extracting behavioral features...'):
        # Save uploaded file temporarily for Scapy
        temp_path = "temp_capture.pcap"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract Features
        X_raw, df_display = extract_flow_features(temp_path, target_ip)
        os.remove(temp_path) # Clean up
        
        if len(X_raw) > 0:
            st.success(f"Successfully extracted {len(X_raw)} network flows!")
            
            # Inference Pipeline
            X_scaled = scaler.transform(X_raw)
            predictions_prob = model.predict(X_scaled)
            predictions = (predictions_prob > 0.5).astype(int).flatten()
            
            # Append results to display dataframe
            df_display['Spoof Probability'] = predictions_prob.flatten()
            df_display['Status'] = ["⚠️ SPOOFED" if p == 1 else "✅ BENIGN" for p in predictions]
            
            # --- DASHBOARD METRICS ---
            st.markdown("### Analysis Results")
            col1, col2, col3 = st.columns(3)
            total_spoofed = sum(predictions)
            
            col1.metric("Total Network Flows", len(df_display))
            col2.metric("Spoofed Flows Detected", total_spoofed, delta_color="inverse")
            col3.metric("Network Health", f"{100 - (total_spoofed/len(df_display)*100):.1f}%", delta_color="normal")
            
            # --- DETAILED DATA TABLE ---
            st.markdown("### Flow Analysis Details")
            # Style the dataframe to highlight spoofed rows
            def highlight_spoofed(val):
                color = '#ff4b4b' if val == '⚠️ SPOOFED' else '#00cc66'
                return f'color: {color}'
            
            st.dataframe(df_display.style.map(highlight_spoofed, subset=['Status']), use_container_width=True)
            
        else:
            st.warning("No valid IP flows found in the uploaded PCAP.")
else:
    st.info("👈 Please upload a .pcap file in the sidebar to begin analysis.")
