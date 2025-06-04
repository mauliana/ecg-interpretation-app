import streamlit as st
import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import tempfile
import os
from collections import defaultdict
from tools import Tools
from mi import MI
from arrhytmia import Arrhytmia

# Create a object instance
tool = Tools()
mi = MI()
arr = Arrhytmia()

st.set_page_config(layout="wide")
st.title("ECG Interpretation app")

info_df = tool.load_table_info()

# Load patient metadata (a sample of recorded patient data saved in csv)
metadata = tool.load_metadata()

# --------- Upload a file ---------------
st.subheader("Upload WFDB files (.dat, .hea)")

uploaded_files = st.file_uploader(
    "Upload related WFDB files (example: .hea and .dat file)", 
    type=["hea", "dat"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Save uploaded files to a temporary directory
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Detect the record name (by removing extension)
    record_names = list(set([os.path.splitext(f.name)[0] for f in uploaded_files]))

    if len(record_names) == 1:
        record_name = record_names[0]
        
        # Read the record
        # record = wfdb.rdrecord(os.path.join(temp_dir, record_name), sampfrom=0, sampto=1000)
        record = wfdb.rdrecord(os.path.join(temp_dir, record_name))
        # sampling_rate = record.fs  # safer to take fs from record metadata
        # Try to get sampling rate from file if possible
        try:
            sampling_rate_record = int(record.fs)
            st.success(f"Detected sampling rate from file: {sampling_rate_record} Hz")
            sampling_rate = sampling_rate_record
        except:
            sampling_rate = st.selectbox(
                "Select Sampling Rate (Hz)",
                options=[100, 500],
                index=0  # default to 100 Hz
            )
            st.warning(f"Failed to detect sampling rate from file, select sampling rate")

        st.subheader("Patient Data")
        
        # get patient data
        patient_data = tool.search_metadata(metadata,record_name)

        st.write(f"**Patient Id:** {patient_data.index[0]}")
        st.write(f"**Age:** {patient_data["age"].iloc[0]}")
        st.write(f"**Gender:** {patient_data["sex"].iloc[0]}")
        st.write(f"**Recording data:** {patient_data["recording_date"].iloc[0]}")
        st.divider()

        signals = record.p_signal
        n_leads = signals.shape[1]
        lead_names = record.sig_name
        time = np.linspace(0, signals.shape[0] / sampling_rate, signals.shape[0])

        with st.expander("Overview of All 12 ECG Leads"):
            view_all_signal = tool.plot_12_leads(signals, n_leads, lead_names, time)
            st.plotly_chart(view_all_signal, use_container_width=True)

        # ----------------- Select a Lead to Explore in Detail -----------------
        st.subheader("Inspect a Specific Lead with Detected Peaks")

        # Analyze once
        df_status, lead_plot_data = mi.analyze_all_leads(patient_data, signals, lead_names, sampling_rate)

        selected_lead = st.selectbox("Select Lead to Inspect", lead_names)

        # Get data for plot
        ecg_cleaned, time, waves, rpeaks, lead_status, v_threshold = lead_plot_data[selected_lead]
        
        selected_index = lead_names.index(selected_lead)
        signal = signals[:, selected_index]

        ecg_signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)

        # plot = tool.plot_ecg_with_peaks(signal, sampling_rate, info, lead_name=selected_lead)
        plot = tool.plot_lead(ecg_cleaned, time, waves, rpeaks, selected_lead, sampling_rate, lead_status, v_threshold)
        st.plotly_chart(plot, use_container_width=True)
        
        col1, col2 = st.columns([3, 2])

        with col1:
            subcol1, subcol2 = st.columns([3,2])
            with subcol1:
                st.markdown("**Signal Reading Based on Selected Signal:**")
                interval = arr.calculate_interval(ecg_signals, info, sampling_rate, patient_data["sex"].iloc[0])
                st.dataframe(interval, hide_index=True)
            with subcol2:
                st.markdown("**ECG Analysis Based on Lead II:** ")
                signal_lead2 = signals[:, 1]

                rythm_type = arr.rhythm_analysis(signal_lead2, sampling_rate)
                st.write(f"**Rythm type**: {rythm_type}")

                st.divider()
                
                st.markdown(f"**ECG Diagnostic Class:** :blue-background[{patient_data["diagnostic_superclass"].values}]")

            with st.expander("See explanation of signal reading"):
                st.dataframe(info_df, hide_index=True)
            

        with col2:
            st.markdown("**MI Diagnose:**")
            # Check for MI regions
            contiguous_results = mi.identify_contiguous_regions(df_status)

            # Group detected leads by region
            grouped_results = defaultdict(list)
            for region, leads in contiguous_results:
                grouped_results[region].append(leads)

            if grouped_results:
                for region, leads_list in grouped_results.items():
                    if region == "Anterior":
                        st.write(f"**{region}** Left Anterior Descending (LAD)")
                    elif region == "Anteroseptal":
                        st.write(f"**{region}** LAD (proximal)")
                    elif region == "Anterolateral":
                        st.write(f"**{region}** LAD or Left Circumflex")
                    elif region == "Lateral":
                        st.write(f"**{region}** Left Circumflex")
                    elif region == "High Lateral":
                        st.write(f"**{region}** Left Circumflex (high branch)")
                    elif region == "Inferior":
                        st.write("Right Coronary Artery (RCA) or Left Circumflex")
                    elif region == "Posterior":
                        st.write(f"**{region}** Posterior descending (RCA or Circumflex)")
                    
                    # Print all lead combinations for this region
                    for leads in leads_list:
                        st.markdown(f"- {', '.join(sorted(leads))}")
                    # st.markdown(leads_list)
            else:
                st.info("No ST elevation or depression in defined contiguous lead groups.")
            
            # Show full result table
            # st.markdown("**Detail information:**")
            # st.dataframe(df_status)
    else:
        st.error("Please upload the matching .dat and .hea files for a single record.")

else:
    st.info("Please upload your WFDB files (.dat and .hea) to start.")
    

        






