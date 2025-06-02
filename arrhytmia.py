import numpy as np
import pandas as pd
import neurokit2 as nk
import ast

class Arrhytmia:
    def __init__(self):
        self.data = None

    def clean(self, peaks):
        return np.array(peaks)[~np.isnan(peaks)].astype(int)
    
    def avg_heart_rate(self, ecg_signals):
        return ecg_signals["ECG_Rate"].mean()  

    def calculate_interval(self, ecg_signals, info, sampling_rate, sex):
        p_peaks = self.clean(info["ECG_P_Peaks"])
        r_peaks = self.clean(info["ECG_R_Peaks"])
        q_peaks = self.clean(info["ECG_Q_Peaks"])
        s_peaks = self.clean(info["ECG_S_Peaks"])
        t_peaks = self.clean(info["ECG_T_Peaks"])

        status = []

        # Average heart rate
        avg_hr = round(self.avg_heart_rate(ecg_signals), 2)
        if 60 < avg_hr < 100:
            status.append("Normal")
        else: status.append("Abnormal")

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

        pr = round(np.mean(pr_interval), 2)
        if 120 < pr < 200: status.append("Normal")
        else: status.append("Abnormal")

        rr = round(np.mean(RR_ms), 2)
        
        if 600 < rr < 1000: status.append("Normal") 
        else: status.append("Abnormal")

        qrs = round(np.mean(qrs_interval), 2)
        if qrs <= 120: status.append("Normal")
        else: status.append("Abnormal")

        qt = round(np.mean(qt_interval), 2)
        if sex == "Male":
            if 350 <= qt <= 450: status.append("Normal")
            else: status.append("Abnormal")
        elif sex == "Female":
            if 360 <= qt <= 460: status.append("Normal")
            else: status.append("Abnormal")
        
        qtc = round(np.mean(QTc_ms), 2)
        if sex == "Male":
            if qtc <= 440: status.append("Normal")
            else: status.append("Abnormal")
        elif sex == "Female":
            if qtc <= 460: status.append("Normal")
            else: status.append("Abnormal")

        # print(len(status))
        df_result = pd.DataFrame(
            {
                "Parameter": ["Average Heart Rate (HR)", "PR Interval", "RR Interval", "QRS Duration", "QT Interval","QTc (Corrected QT)"],
                "Value": [avg_hr, pr, rr, qrs, qt, qtc],
                "Unit": ["bpm", "ms", "ms", "ms", "ms", "ms"],
                "Status": status
            }
        )
        
        return df_result
     
    
    def is_rr_regular(self, rr_intervals):
        # Define regularity threshold = 6% for clinical use
        threshold = 0.06 * np.mean(rr_intervals)
        rr_diff = np.abs(np.diff(rr_intervals))      
        is_rr_regular = np.all(rr_diff < threshold)

        return True if is_rr_regular else False
    
    def is_hr_stable(self, heart_rate):
        # Threshold: 
        # < 5 bpm: Stable heart rate (low HRV)
        # 5 - 10 bpm: Mild variation (normal HRV)
        # > 10 bpm: Possibly irregular rhythm

        is_heart_rate_stable = np.std(heart_rate) < 5  # bpm variability
        return True if is_heart_rate_stable else False
        
    def detect_short_rr(self, rr_intervals, threshold=0.4):  # threshold in seconds
        return np.any(rr_intervals < threshold)

    def rhythm_analysis(self, signal, sampling_rate):
        ecg_signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
        r_peaks = self.clean(info["ECG_R_Peaks"])
        rr_intervals = np.diff(r_peaks) / sampling_rate
        reg_rr = self.is_rr_regular(rr_intervals)
        rr = round(np.mean(rr_intervals), 2) * 1000

        diffs = np.abs(np.diff(rr_intervals))
        variability = np.std(rr_intervals)
        arr_threshold = 0.08
        is_high_variability = variability > arr_threshold
        has_large_rr_change = np.any(diffs > arr_threshold)

        has_short_rr = self.detect_short_rr(rr_intervals)
        # print(rr)
        
        heart_rate = ecg_signals["ECG_Rate"]
        hr = self.avg_heart_rate(ecg_signals)
        reg_rhythm = self.is_hr_stable(heart_rate)

        p_peaks = self.clean(info["ECG_P_Peaks"])
        min_len = min(len(p_peaks), len(r_peaks))
        pr_interval = (r_peaks[:min_len] - p_peaks[:min_len]) / sampling_rate 
        pr = round(np.mean(pr_interval), 2) * 1000

        type = ""
        if hr < 60 and rr > 1000 and reg_rhythm:
            type = "Bradycardia"
        elif 60 <= hr <= 100:
            if 600 <= rr <= 1000 and 120 <= pr <= 200 and reg_rhythm:
                type = "Normal Sinus"
            elif is_high_variability and has_large_rr_change and reg_rr == False:
                type = "Arrhytmia"
            elif reg_rhythm == False and rr < 600:
                type = "Atrial Premature Complex (APC)"
            elif pr > 300 and reg_rhythm:
                type = "First Degree AV Block"
        elif hr > 100:
            if rr < 600 and 120 <= pr <= 200 and reg_rhythm:
                type = "Tachycardia"
            elif hr > 150 and has_short_rr:
                type = "Paroxysmal Supraventricular Tachycardia (PSVT)"
            elif hr > 150 and rr < 400 and reg_rhythm:
                type = "Supraventricular Tachycardia (SVT)"
            elif hr < 175 and reg_rr == False:
                type = "Atrial Fibrillation (AF)"
            elif hr > 300 and reg_rr == False:
                type = "Ventricular Fibrillation (VF)"
        
        if type == "":
            type = "Undefined Rhythm"
        # rr_intervals, rr_mean, rr_std, rr_variation, rr_cv = self.compute_rr_intervals(r_peaks, sampling_rate)

        # rhythm_type = self.detect_rhythm_type(hr, rr_mean, rr_std, rr_variation, rr_cv)
        

        return type

    # def compute_rr_intervals(self, r_peaks, sampling_rate):
    #     rr_intervals_ms = (np.diff(r_peaks) / sampling_rate) * 1000  # ms
    #     rr_mean = np.mean(rr_intervals_ms)
    #     rr_variation = np.max(rr_intervals_ms) - np.min(rr_intervals_ms)
    #     return rr_intervals_ms, rr_mean, rr_variation