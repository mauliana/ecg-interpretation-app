import streamlit as st
import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
from tools import Tools

# Create a Tools instance
tool = Tools()

st.set_page_config(layout="wide")
st.title("ECG Interpretation app")

info_df = pd.DataFrame(
    {
        "Parameter": ["Heart Rate (HR)", "PR Interval", "QRS Duration", "QT Interval", "QTc (Corrected QT)"],
        "Normal Range": ["60 – 100", "120 – 200", "≤ 120", "~350 – 450 (men), ~360 – 460 (women)", "≤ 440 (men), ≤ 460 (women)"],
        "Unit": ["bpm", "ms", "ms", "ms", "ms"],
        "Description": [
            "Number of heartbeats per minute (<60 = bradycardia, >100 = tachycardia)", 
            "Time from atrial depolarization to ventricular depolarization",
            "Time of ventricular depolarization (Q to S)",
            "Total time for ventricular depolarization + repolarization (Q to T)",
            "Heart-rate corrected QT interval (Bazett's Formula)"
            ]
    }
)

# read the sample data
base_path = '/Users/mauliana/Documents/Work/GAIA/code/ecg/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
normal_data = base_path+'/records100/08000/08874_lr'
hyp_data = base_path+"/records100/00000/00750_lr"
abnormal_data = base_path+"/records100/13000/13625_lr" # ['STTC', 'HYP', 'MI', 'CD']
mi_data = base_path+"/records100/00000/00063_lr"
brady_sample = base_path+"/records100/00000/00002_lr"
svt_sample = base_path+"/records100/00000/00017_lr"
tachy_sample = base_path+"/records100/04000/04761_lr"


record = wfdb.rdrecord(svt_sample, sampfrom=0, sampto=1000)  # 10 seconds @ 100Hz
sampling_rate = 100

signals = record.p_signal
n_leads = signals.shape[1]
lead_names = record.sig_name
time = np.linspace(0, signals.shape[0] / sampling_rate, signals.shape[0])

# with st.container():
with st.expander("Overview of All 12 ECG Leads"):
    view_all_signal = tool.plot_12_leads(signals, n_leads, lead_names, time)
    st.plotly_chart(view_all_signal, use_container_width=True)

# ----------------- Select a Lead to Explore in Detail -----------------
st.subheader("Inspect a Specific Lead with Detected Peaks")
selected_lead = st.selectbox("Select Lead to Inspect", lead_names)

selected_index = lead_names.index(selected_lead)
signal = signals[:, selected_index]
ecg_signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)


# # ---- Plotly ECG plot with peaks ----
col1, col2 = st.columns([3, 1])

with col1:
    st.write(f"This plot shows P, Q, R, S, T peaks detected from Lead {selected_lead} ECG signal.")
    plot = tool.plot_ecg_with_peaks(signal, sampling_rate, info, lead_name=selected_lead)
    st.plotly_chart(plot, use_container_width=True)

    with st.expander("See explanation of signal reading"):
        st.dataframe(info_df, hide_index=True)

with col2:
    st.subheader("Signal Reading:")
    avg_hr = tool.avg_heart_rate(ecg_signals)
    hr = f"Average Heart Rate: {avg_hr:.2f} bpm"
    st.write(hr)

    pr, qrs, qt, qtc, rr = tool.calculate_interval(ecg_signals, info, sampling_rate)
    st.write(pr)
    st.write(qrs)
    st.write(qt)
    st.write(qtc)
    st.write(rr)

    st.divider()
    st.markdown("**ECG Analysis Based on Lead II:** ")
    signal_lead2 = signals[:, 1]

    rythm_type = tool.rythm_analysis(signal_lead2, sampling_rate)
    st.write(f"**Rythm type**: {rythm_type}")

    st.divider()
    
    st.markdown("**ECG Diagnostic Class:** :blue-background[Normal] / :red-background[Hypertrophy] / :green-background[STTC] / :grey-background[CD]")
    






