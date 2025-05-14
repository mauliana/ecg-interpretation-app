import numpy as np
import pandas as pd
import neurokit2 as nk
import plotly.graph_objects as go
import plotly.subplots as sp
import ast

class Tools:
    def __init__(self):
        self.data = None
    
    def load_table_info(self):
        info = pd.DataFrame(
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
        return info

    def load_metadata(self):
        metadata = pd.read_csv('clean_metadata.csv', index_col='patient_id')
        return metadata
    
    def search_metadata(self, df, filename):
        rate_type = filename[-2:]
        column_name = "filename_"+rate_type

        # search data
        # data = data.loc[data[column_name] == filename]
        data = df[df[column_name].str.contains(filename, case=False, na=False)]
        result = data.copy()
        result['sex'] = result['sex'].map({0: 'Male', 1: 'Female'})
        
        # get the class
        diag_class = ast.literal_eval(result['diagnostic_superclass'].iloc[0])

        result = result.drop(columns=['filename_lr', 'filename_hr', 'strat_fold','diagnostic_superclass', 'scp_codes'])
        return diag_class[0], result

    def plot_ecg_with_peaks(self, signal, sampling_rate, info, lead_name="Lead II", title=None):
        """
        Plots ECG signal and highlights detected peaks (P, Q, R, S, T) 

        Parameters:
        - signal: 1D numpy array of ECG signal
        - sampling_rate: Sampling frequency (Hz)
        - info: Dictionary output from nk.ecg_process(), containing peaks
        - lead_name: Optional, name of the ECG lead
        - title: Optional, custom title for the plot
        """
        time = np.linspace(0, len(signal) / sampling_rate, len(signal))

        fig = go.Figure()

        # ECG signal trace
        fig.add_trace(go.Scatter(
            x=time,
            y=signal,
            mode='lines',
            name='ECG Signal',
            line=dict(color='black', width=1)
        ))

        peak_types = {
            "ECG_P_Peaks": "purple",
            "ECG_Q_Peaks": "blue",
            "ECG_R_Peaks": "red",
            "ECG_S_Peaks": "orange",
            "ECG_T_Peaks": "green"
        }

        for peak_label, color in peak_types.items():
            peaks = info.get(peak_label)
            if peaks is not None:
                peaks = self.clean(peaks)
                fig.add_trace(go.Scatter(
                    x=time[peaks],
                    y=signal[peaks],
                    mode='markers',
                    name=peak_label.replace("ECG_", "").replace("_Peaks", ""),
                    marker=dict(color=color, size=6, symbol="circle")
                ))

        fig.update_layout(
            title=title or f"ECG Signal ({lead_name}) with Detected P, Q, R, S, T Peaks",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            legend_title="Peak Types",
            template="plotly_white",
            height=500
        )

        return fig

    def plot_12_leads(self, signals, n_leads, lead_names, time):
        fig_all = sp.make_subplots(rows=6, cols=2, shared_xaxes=True, vertical_spacing=0.03,
                               subplot_titles=lead_names[:12])  # adjust if fewer/more than 12

        for i in range(min(12, n_leads)):
            row = i // 2 + 1
            col = i % 2 + 1
            fig_all.add_trace(go.Scatter(
                x=time,
                y=signals[:, i],
                mode='lines',
                name=lead_names[i],
                line=dict(color='black', width=1)  
            ), row=row, col=col)

        fig_all.update_layout(
            height=1000,
            showlegend=False,
            template="plotly_white",
            margin=dict(t=50, b=30)
        )
        return fig_all

    def avg_heart_rate(self, ecg_signals):
        return ecg_signals["ECG_Rate"].mean()   

    def clean(self, peaks):
        return np.array(peaks)[~np.isnan(peaks)].astype(int)

    def calculate_interval(self, ecg_signals, info, sampling_rate):
        p_peaks = self.clean(info["ECG_P_Peaks"])
        r_peaks = self.clean(info["ECG_R_Peaks"])
        q_peaks = self.clean(info["ECG_Q_Peaks"])
        s_peaks = self.clean(info["ECG_S_Peaks"])
        t_peaks = self.clean(info["ECG_T_Peaks"])

        # Align arrays (just match each P to its following R)
        min_len = min(len(p_peaks), len(r_peaks), len(t_peaks), len(q_peaks))
        pr_interval = (r_peaks[:min_len] - p_peaks[:min_len]) / sampling_rate * 1000  # in ms
        qrs_interval = (s_peaks[:min_len] - q_peaks[:min_len]) / sampling_rate * 1000
        qt_interval = (t_peaks[:min_len] - q_peaks[:min_len]) / sampling_rate * 1000

        QT_sec = (t_peaks[:min_len] - q_peaks[:min_len]) / sampling_rate
        RR_sec = np.diff(r_peaks[:min_len+1]) / sampling_rate
        RR_ms = RR_sec * 1000

        QTc = QT_sec[:len(RR_sec)] / np.sqrt(RR_sec)  # Still in seconds
        QTc_ms = QTc * 1000  # Convert to ms if needed

        pr = f"PR interval: {np.mean(pr_interval):.2f} ms"
        qrs = f"QRS interval: {np.mean(qrs_interval):.2f} ms"
        qt = f"QT interval: {np.mean(qt_interval):.2f} ms"
        rr = f"RR interval: {np.mean(RR_ms):.2f} ms"
        qtc = f"QTc interval: {np.mean(QTc_ms):.2f} ms"
        
        return pr, qrs, qt, qtc, rr
    
    def compute_rr_intervals(self, r_peaks, sampling_rate):
        rr_intervals_ms = (np.diff(r_peaks) / sampling_rate) * 1000  # ms
        rr_mean = np.mean(rr_intervals_ms)
        rr_variation = np.max(rr_intervals_ms) - np.min(rr_intervals_ms)
        return rr_intervals_ms, rr_mean, rr_variation

    def detect_rhythm_type(self, hr, rr_mean, rr_variation):
        if hr < 60 and rr_mean > 1000:
            return f"Bradycardia \n\nHR: {hr:.2f} bpm (< 60), RR: {int(rr_mean)} ms (> 1000)"
        elif hr > 250:
            return f"Atrial Flutter \n\nHR: {hr:.2f} bpm (> 250)"
        elif hr > 150 and rr_mean < 400:
            return f"Supraventricular Tachycardia (SVT) \n\nHR: {hr:.2f} bpm (> 150), RR: {int(rr_mean)} ms (< 400)"
        elif hr > 100 and rr_mean < 600:
            return f"Tachycardia \n\nHR: {hr:.2f} bpm (> 100), RR: {int(rr_mean)} ms (< 600)"
        elif 60 <= hr <= 100 and rr_variation > 120:
            return f"Sinus Arrhythmia \n\nHR: {hr:.2f} bpm (normal), RR variation: {int(rr_variation)} ms (> 120)"
        else:
            return f"Normal \n\nHR: {hr:.2f} bpm (60–100), RR: {int(rr_mean)} ms, variation: {int(rr_variation)} ms"

    def rhythm_analysis(self, signal, sampling_rate):
        ecg_signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
        hr = self.avg_heart_rate(ecg_signals)
        r_peaks = self.clean(info["ECG_R_Peaks"])

        rr_intervals, rr_mean, rr_variation = self.compute_rr_intervals(r_peaks, sampling_rate)
        rhythm_type = self.detect_rhythm_type(hr, rr_mean, rr_variation)

        return rhythm_type
    
    def st_segment_analysis(self, signals, lead_names, age, gender, sampling_rate):
        v_threshold = 0.2 if gender == "Male" and age >= 40 else 0.25 if gender == "Male" else 0.15
        elev_threshold = 0.1
        dep_threshold = 0.05

        for index, lead_name in enumerate(lead_names):
            signal = signals[:, index]
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt")

            # Time axis
            time = np.arange(len(ecg_cleaned)) / sampling_rate * 1000

            threshold = v_threshold if lead_name in ["V2", "V3"] else elev_threshold

            baseline_points = []
            duration = []

            for i in range(len(rpeaks['ECG_R_Peaks'])):
                try:
                    q_onset = int(waves["ECG_Q_Peaks"][i]) if not np.isnan(waves["ECG_Q_Peaks"][i]) else None
                    r_offset = int(waves["ECG_R_Offsets"][i]) if not np.isnan(waves["ECG_R_Offsets"][i]) else None
                    p_offset = int(waves["ECG_P_Offsets"][i]) if not np.isnan(waves["ECG_P_Offsets"][i]) else None
                    t_onset = int(waves["ECG_T_Onsets"][i]) if not np.isnan(waves["ECG_T_Onsets"][i]) else None

                    if None in (q_onset, r_offset, t_onset) or r_offset >= t_onset:
                        continue

                    # Baseline: 40 ms before Q onset (typical PR segment)
                    baseline_start = max(0, q_onset - int(0.04 * sampling_rate))
                    baseline_end = q_onset
                    baseline_mean = np.mean(ecg_cleaned[baseline_start:baseline_end])
                    baseline_points.append(baseline_mean)

                    # ST segment: from J (R offset) to T onset
                    st_segment = ecg_cleaned[r_offset:t_onset]
                    st_mean = np.mean(st_segment)

                    # Compare ST to baseline
                    deviation = st_mean - baseline_mean

                    
                except Exception:
                    continue


        return 0
    


        
