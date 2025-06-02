import numpy as np
import pandas as pd
import neurokit2 as nk
import plotly.graph_objects as go
import plotly.subplots as sp
import ast
from mi import MI

class Tools:
    def __init__(self):
        self.data = None
        self.mi = MI()
    
    def load_table_info(self):
        info = pd.DataFrame(
            {
                "Parameter": ["Heart Rate (HR)", "PR Interval", "RR Interval", "QRS Duration", "QT Interval", "QTc (Corrected QT)"],
                "Normal Range": ["60 – 100", "120 – 200", "600 - 1000", "≤ 120", "~350 – 450 (men), ~360 – 460 (women)", "≤ 440 (men), ≤ 460 (women)"],
                "Unit": ["bpm", "ms", "ms", "ms", "ms", "ms"],
                "Description": [
                    "Number of heartbeats per minute (<60 = bradycardia, >100 = tachycardia)", 
                    "Time from atrial depolarization to ventricular depolarization",
                    "Time measure between two consecutive R waves",
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
        # diag_class = ast.literal_eval(result['diagnostic_superclass'])

        result = result.drop(columns=['filename_lr', 'filename_hr', 'strat_fold', 'scp_codes'])
        return result

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

    def clean(self, peaks):
        return np.array(peaks)[~np.isnan(peaks)].astype(int)
    
    def plot_lead(self, ecg_cleaned, time, waves, rpeaks, lead_name, sampling_rate, lead_status, v_threshold):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=ecg_cleaned, mode='lines', name='ECG Signal'))

        # --------- Global baseline ---------------
        g_baseline = self.mi.global_baseline(ecg_cleaned, sampling_rate, waves, rpeaks)
        if g_baseline != 0:
            fig.add_trace(go.Scatter(
                x=time,
                y=[g_baseline] * len(time),
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                name="Global Baseline (PR segment)",
                showlegend=True
            ))

        elev_threshold = v_threshold if lead_name in ["V2", "V3"] else 0.1
        depression_threshold = 0.05
        added_legend = {"elevation": False, "depression": False, "jpoint": False, "st-point": False,
                        "qpeak": False, "rpeak": False, "speak": False, "tpeak":False,"ppeak": False}

        # print(f"plot threshold {lead_name}: {elev_threshold}")
        for i in range(len(rpeaks["ECG_R_Peaks"])):
            try:
                r_idx = int(rpeaks["ECG_R_Peaks"][i])
                if r_idx >= len(ecg_cleaned):
                    continue

                # --- Q Peak ---
                q_peak = int(waves["ECG_Q_Peaks"][i]) if not np.isnan(waves["ECG_Q_Peaks"][i]) else None
                if q_peak is not None and q_peak < len(ecg_cleaned):
                    fig.add_trace(go.Scatter(
                        x=[time[q_peak]],
                        y=[ecg_cleaned[q_peak]],
                        mode='markers+text',
                        marker=dict(color='blue', size=7, symbol='triangle-up'),
                        text=["Q"],
                        textposition="top center",
                        name="Q peak" if not added_legend["qpeak"] else "",
                        showlegend=not added_legend["qpeak"]
                    ))
                    added_legend["qpeak"] = True

                # --- R Peak ---
                fig.add_trace(go.Scatter(
                    x=[time[r_idx]],
                    y=[ecg_cleaned[r_idx]],
                    mode='markers+text',
                    marker=dict(color='purple', size=7, symbol='diamond'),
                    text=["R"],
                    textposition="top center",
                    name="R peak" if not added_legend["rpeak"] else "",
                    showlegend=not added_legend["rpeak"]
                ))
                added_legend["rpeak"] = True

                # --- Prominent R ---
                prom_threshold = 0.7 if lead_name in ["V1", "V2", "V3"] else 1.5
                if ecg_cleaned[r_idx] > prom_threshold:
                    fig.add_trace(go.Scatter(
                        x=[time[r_idx]],
                        y=[ecg_cleaned[r_idx]],
                        mode='markers+text',
                        marker=dict(color='yellow', size=10, symbol='star'),
                        text=["Prominent R"],
                        textposition="bottom center",
                        name=f"Prominent R ({lead_name})" if not added_legend.get("prominent_r_" + lead_name, False) else "",
                        showlegend=not added_legend.get("prominent_r_" + lead_name, False)
                    ))
                    added_legend["prominent_r_" + lead_name] = True

                # --- S Peak ---
                s_peak = int(waves["ECG_S_Peaks"][i]) if not np.isnan(waves["ECG_S_Peaks"][i]) else None
                if s_peak is not None and s_peak < len(ecg_cleaned):
                    fig.add_trace(go.Scatter(
                        x=[time[s_peak]],
                        y=[ecg_cleaned[s_peak]],
                        mode='markers+text',
                        marker=dict(color='orange', size=7, symbol='triangle-down'),
                        text=["S"],
                        textposition="bottom center",
                        name="S peak" if not added_legend["speak"] else "",
                        showlegend=not added_legend["speak"]
                    ))
                    added_legend["speak"] = True

                # --- T Peak ---
                t_peak = int(waves["ECG_T_Peaks"][i]) if not np.isnan(waves["ECG_T_Peaks"][i]) else None
                if t_peak is not None and t_peak < len(ecg_cleaned):
                    fig.add_trace(go.Scatter(
                        x=[time[t_peak]],
                        y=[ecg_cleaned[t_peak]],
                        mode='markers+text',
                        marker=dict(color='green', size=7),
                        text=["T"],
                        textposition="top center",
                        name="T peak" if not added_legend["tpeak"] else "",
                        showlegend=not added_legend["tpeak"]
                    ))
                    added_legend["tpeak"] = True

                # --- P Peak ---
                p_peak = int(waves["ECG_P_Peaks"][i]) if not np.isnan(waves["ECG_P_Peaks"][i]) else None
                if p_peak is not None and p_peak < len(ecg_cleaned):
                    fig.add_trace(go.Scatter(
                        x=[time[p_peak]],
                        y=[ecg_cleaned[p_peak]],
                        mode='markers+text',
                        marker=dict(color='pink', size=7),
                        text=["P"],
                        textposition="top center",
                        name="P peak" if not added_legend["ppeak"] else "",
                        showlegend=not added_legend["ppeak"]
                    ))
                    added_legend["ppeak"] = True

                # --- J point ---
                j_point = int(waves["ECG_R_Offsets"][i]) if not np.isnan(waves["ECG_R_Offsets"][i]) else None
                # j_point = s_peak
                if j_point is None or j_point >= len(ecg_cleaned):
                    continue

                fig.add_trace(go.Scatter(
                    x=[time[j_point]],
                    y=[ecg_cleaned[j_point]],
                    mode='markers+text',
                    marker=dict(color='gray', size=7),
                    text=["J"],
                    textposition="top center",
                    name="J point" if not added_legend["jpoint"] else "",
                    showlegend=not added_legend["jpoint"]
                ))
                added_legend["jpoint"] = True

                # --- Baseline from PR segment ---
                p_offset = int(waves["ECG_P_Offsets"][i]) if not np.isnan(waves["ECG_P_Offsets"][i]) else None
                q_peak = int(waves["ECG_Q_Peaks"][i]) if not np.isnan(waves["ECG_Q_Peaks"][i]) else None
                if q_peak is None or q_peak >= len(ecg_cleaned):
                    continue

                baseline_start = p_offset if (p_offset is not None and p_offset < q_peak) else max(0, q_peak - int(0.04 * sampling_rate))
                baseline_end = q_peak

                baseline_window = ecg_cleaned[baseline_start:baseline_end]
                if len(baseline_window) == 0:
                    continue

                baseline_mean = np.mean(baseline_window)

                # --- ST value at 80ms after J-point ---
                st_eval_offset = j_point + int(0.08 * sampling_rate)
                if st_eval_offset >= len(ecg_cleaned):
                    continue

                st_value = ecg_cleaned[st_eval_offset]
                deviation = st_value - baseline_mean
                # deviation = st_value - g_baseline

                # --- Optional: Mark ST eval point at 80ms ---
                fig.add_trace(go.Scatter(
                    x=[time[st_eval_offset]],
                    y=[st_value],
                    mode='markers+text',
                    marker=dict(color='blue', size=6),
                    # text=["ST@80ms"],
                    textposition="bottom center",
                    name="ST eval point (80 ms)" if not added_legend["st-point"] else "",
                    showlegend=not added_legend["st-point"]
                ))
                added_legend["st-point"] = True

                if deviation >= elev_threshold:
                    color, fill, name = 'red', 'rgba(255,0,0,0.2)', "ST Elevation"
                    show_legend = not added_legend["elevation"]
                    added_legend["elevation"] = True
                elif deviation <= -depression_threshold:
                    color, fill, name = 'green', 'rgba(0,255,0,0.2)', "ST Depression"
                    show_legend = not added_legend["depression"]
                    added_legend["depression"] = True
                else:
                    continue  # No elevation/depression

                # --- Always visualize 80ms of ST segment after J-point ---
                st_duration_samples = int(0.08 * sampling_rate)
                t_onset = min(j_point + st_duration_samples, len(ecg_cleaned) - 1)

                x_vals = list(time[j_point:t_onset])
                y_vals = list(ecg_cleaned[j_point:t_onset])

                x_poly = x_vals + x_vals[::-1]
                y_poly = y_vals + [baseline_mean] * len(y_vals)

                fig.add_trace(go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    mode='lines',
                    line=dict(color=color),
                    fill='toself',
                    fillcolor=fill,
                    name=name,
                    showlegend=show_legend
                ))

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color=color, width=1),
                    name=name,
                    showlegend=False
                ))

            except Exception as e:
                print(f"Error in beat {i}: {e}")
                continue

        # fig.update_layout(
        #     title=f"Lead {lead_name}",
        #     xaxis_title="Time (ms)",
        #     yaxis_title="Amplitude (mV)",
        #     template="plotly_white",
        #     legend=dict(orientation="h", y=-0.2)
        # )

        y_margin = 0.1
        y_min = min(ecg_cleaned) - y_margin
        y_max = max(ecg_cleaned) + y_margin

        fig.update_layout(
            title=f"Lead {lead_name}",
            xaxis_title="Time (ms)",
            yaxis_title="Amplitude (mV)",
            legend=dict(orientation="h", y=-0.2),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,0,0,0.1)',
                dtick=40,  # One small box in ECG paper 
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,0,0,0.1)',
                dtick=0.1,  # One small box for 0.1 mV
                range=[y_min, y_max]
            ),
            plot_bgcolor='white'
        )

        # Get x and y axis ranges from the data
        x_start = min(time)
        x_end = max(time)
        y_start = min(ecg_cleaned)
        y_end = max(ecg_cleaned)

        # Bold vertical lines every 200 ms (5 × 40 ms)
        for x in range(int(x_start), int(x_end) + 1, 200):
            fig.add_shape(
                type="line",
                x0=x, x1=x,
                y0=y_start, y1=y_end,
                line=dict(color='rgba(255,0,0,0.5)', width=1.5),
                layer="below"
            )

        # Bold horizontal lines every 0.5 mV (5 × 0.1 mV)
        y_current = round(y_start - (y_start % 0.5), 2)
        while y_current <= y_end:
            fig.add_shape(
                type="line",
                x0=x_start, x1=x_end,
                y0=y_current, y1=y_current,
                line=dict(color='rgba(255,0,0,0.5)', width=1.5),
                layer="below"
            )
            y_current = round(y_current + 0.5, 2)
       
        return fig
