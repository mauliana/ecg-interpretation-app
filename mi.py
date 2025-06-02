import numpy as np
import pandas as pd
import neurokit2 as nk
import ast

class MI:
    def __init__(self):
        self.data = None

    def global_baseline(self, ecg_cleaned, sampling_rate, waves, rpeaks):
        baseline_points = []
        for i in range(len(rpeaks["ECG_R_Peaks"])):

            q_peaks = waves["ECG_Q_Peaks"][i]
            p_offset = waves["ECG_P_Offsets"][i]

            if np.isnan(q_peaks) or np.isnan(p_offset):
                continue

            q_peaks = int(q_peaks)
            p_offset = int(p_offset)

            # Baseline: use PR segment if p_offset is valid, else fallback to Q-peak - 40ms
            if p_offset is not None and p_offset < q_peaks:
                baseline_start = p_offset
            else:
                baseline_start = max(0, q_peaks - int(0.04 * sampling_rate))
            
            baseline_end = q_peaks
            baseline_mean = np.mean(ecg_cleaned[baseline_start:baseline_end])
            baseline_points.append(baseline_mean)
        
        if baseline_points:
            overall_baseline = np.mean(baseline_points)
            return overall_baseline
        else:
            return 0
        
    # def analyze_ecg_lead(self, signal, lead_name, sampling_rate, v_threshold):
    #     ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)

    #     try:
    #         _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    #     except Exception as e:
    #         print(f"[Error] ECG peak detection failed: {e}")
    #         return "invalid", None, ecg_cleaned, None, None, None, None, False

    #     try:
    #         _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt")
    #     except Exception as e:
    #         print(f"[Error] ECG delineation failed: {e}")
    #         return "invalid", None, ecg_cleaned, None, None, rpeaks, None, False

    #     time = np.arange(len(ecg_cleaned)) / sampling_rate * 1000
    #     threshold = v_threshold if lead_name in ["V2", "V3"] else 0.1
    #     depression_threshold = 0.05
    #     t_inversion_threshold = -0.05
    #     prominent_r_threshold = 0.7 if lead_name in ["V1", "V2", "V3"] else 1.5

    #     baseline_points = []
    #     duration = []
    #     findings = []
    #     has_t_inversion = False
    #     has_prominent_r = False
    #     rs_ratios = []

    #     for i in range(len(rpeaks['ECG_R_Peaks'])):
    #         try:
    #             def safe_get(wave_name):
    #                 val = waves.get(wave_name, [None] * len(rpeaks['ECG_R_Peaks']))[i]
    #                 return int(val) if val is not None and not np.isnan(val) else None

    #             q_onset = safe_get("ECG_Q_Peaks")
    #             r_offset = safe_get("ECG_R_Offsets")
    #             t_onset = safe_get("ECG_T_Onsets")
    #             t_peak = safe_get("ECG_T_Peaks")
    #             s_peak = safe_get("ECG_S_Peaks")
    #             p_offset = safe_get("ECG_P_Offsets")

    #             r_peak = int(rpeaks['ECG_R_Peaks'][i])

    #             if None in (q_onset, r_offset, t_onset, t_peak) or r_offset >= t_onset:
    #                 continue

    #             # Baseline: use PR segment if p_offset is valid, else fallback to Q-onset - 40ms
    #             if p_offset is not None and p_offset < q_onset:
    #                 baseline_start = p_offset
    #             else:
    #                 baseline_start = max(0, q_onset - int(0.04 * sampling_rate))

    #             baseline_end = q_onset
    #             baseline_window = ecg_cleaned[baseline_start:baseline_end]
    #             if len(baseline_window) == 0:
    #                 continue

    #             baseline_mean = np.mean(baseline_window)
    #             baseline_points.append(baseline_mean)

    #             # R-wave prominence
    #             r_amplitude = ecg_cleaned[r_peak] - baseline_mean
    #             if r_amplitude >= prominent_r_threshold:
    #                 findings.append("prominent_r")
    #                 has_prominent_r = True

    #             # ST-segment deviation: use value 80ms after J-point (r_offset)
    #             st_offset = int(0.08 * sampling_rate)
    #             st_index = r_offset + st_offset

    #             if st_index < len(ecg_cleaned):
    #                 st_value = ecg_cleaned[st_index]
    #                 deviation = st_value - baseline_mean

    #                 if deviation >= threshold:
    #                     duration.append(t_onset - r_offset)
    #                     findings.append("elevation")
    #                 elif deviation <= -depression_threshold:
    #                     duration.append(t_onset - r_offset)
    #                     findings.append("depression")

    #             # T-wave inversion
    #             t_amplitude = ecg_cleaned[t_peak]
    #             if t_amplitude < (baseline_mean + t_inversion_threshold):
    #                 has_t_inversion = True
    #                 findings.append("t_inversion")

    #             # R/S ratio
    #             if s_peak is not None:
    #                 s_amplitude = ecg_cleaned[s_peak]
    #                 if np.abs(s_amplitude) > 1e-3:
    #                     rs_ratio = r_amplitude / np.abs(s_amplitude)
    #                     rs_ratios.append(rs_ratio)

    #         except Exception as e:
    #             print(f"[Warning] Skipping beat {i} due to error: {e}")
    #             continue

    #     lead_status = "normal" if not findings else "_and_".join(sorted(set(findings)))
    #     mean_rs_ratio = np.mean(rs_ratios) if rs_ratios else None

    #     return lead_status, duration if duration else None, ecg_cleaned, time, waves, rpeaks, mean_rs_ratio, has_prominent_r
    
    def analyze_ecg_lead(self, signal, lead_name, sampling_rate, v_threshold):
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)

        try:
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        except Exception as e:
            print(f"[Error] ECG peak detection failed: {e}")
            return "invalid", None, ecg_cleaned, None, None, None, None, False

        try:
            _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt")
        except Exception as e:
            print(f"[Error] ECG delineation failed: {e}")
            return "invalid", None, ecg_cleaned, None, None, rpeaks, None, False

        time = np.arange(len(ecg_cleaned)) / sampling_rate * 1000
        threshold = v_threshold if lead_name in ["V2", "V3"] else 0.1
        depression_threshold = 0.05
        t_inversion_threshold = -0.05
        prominent_r_threshold = 0.7 if lead_name in ["V1", "V2", "V3"] else 1.5

        global_baseline = self.global_baseline(ecg_cleaned, sampling_rate, waves, rpeaks)

        baseline_points = []
        duration = []
        findings = []
        has_t_inversion = False
        has_prominent_r = False
        rs_ratios = []

        for i in range(len(rpeaks['ECG_R_Peaks'])):
            try:
                # Skip this beat if required delineation points are missing or invalid
                required_keys = [
                    "ECG_Q_Peaks", "ECG_R_Offsets", "ECG_T_Onsets",
                    "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Offsets"
                ]
                if any(k not in waves or i >= len(waves[k]) or np.isnan(waves[k][i]) for k in required_keys):
                    continue

                q_peak = int(waves["ECG_Q_Peaks"][i])
                r_offset = int(waves["ECG_R_Offsets"][i])
                t_onset = int(waves["ECG_T_Onsets"][i])
                s_peak = int(waves["ECG_S_Peaks"][i])
                t_peak = int(waves["ECG_T_Peaks"][i])
                p_offset = int(waves["ECG_P_Offsets"][i])
                r_peak = int(rpeaks['ECG_R_Peaks'][i])

                if r_offset >= t_onset:
                    continue

                # Baseline: use PR segment if p_offset is valid, else fallback to Q-peak - 40ms
                baseline_start = p_offset if (p_offset < q_peak) else max(0, q_peak - int(0.04 * sampling_rate))
                baseline_end = q_peak

                baseline_window = ecg_cleaned[baseline_start:baseline_end]
                if len(baseline_window) == 0:
                    continue

                baseline_mean = np.mean(baseline_window)
                baseline_points.append(baseline_mean)

                # R-wave amplitude from baseline
                r_amplitude = ecg_cleaned[r_peak] - baseline_mean
                if r_amplitude >= prominent_r_threshold:
                    findings.append("prominent_r")
                    has_prominent_r = True

                # ST-segment deviation
                st_offset = r_offset + int(0.08 * sampling_rate)
                if st_offset >= len(ecg_cleaned):
                    continue

                st_value = ecg_cleaned[st_offset]
                deviation = st_value - baseline_mean
                if deviation >= threshold:
                    duration.append(t_onset - r_offset)
                    findings.append("elevation")
                elif deviation <= -depression_threshold:
                    duration.append(t_onset - r_offset)
                    findings.append("depression")

                # T-wave inversion
                if t_peak < t_onset:
                    t_diff = t_onset - t_peak
                    if t_diff >= t_inversion_threshold:
                        has_t_inversion = True
                        findings.append("t_inversion")

                # R/S ratio using (R - Q) / (R - S)
                qr_amp_diff = ecg_cleaned[r_peak] - ecg_cleaned[q_peak]
                rs_amp_diff = ecg_cleaned[r_peak] - ecg_cleaned[s_peak]

                if rs_amp_diff != 0:
                    rs_ratio = qr_amp_diff / abs(rs_amp_diff)
                    rs_ratios.append(rs_ratio)

            except Exception as e:
                print(f"[Warning] Skipping beat {i} due to error: {e}")
                continue

        lead_status = "normal" if not findings else "_and_".join(sorted(set(findings)))
        mean_rs_ratio = np.mean(rs_ratios) if rs_ratios else None

        return lead_status, duration if duration else None, ecg_cleaned, time, waves, rpeaks, mean_rs_ratio, has_prominent_r


    def analyze_all_leads(self, data, signals, lead_names, sampling_rate):
        if data["sex"].iloc[0] == "Male" and data["age"].iloc[0] >= 40:
            v_threshold = 0.2
        elif data["sex"].iloc[0] == "Male" and data["age"].iloc[0] < 40:
            v_threshold = 0.25
        else:
            v_threshold = 0.15

        status_records = {}
        detail_dict = {}

        for i, lead_name in enumerate(lead_names):
            signal = signals[:, i]
            status, duration, ecg_cleaned, time, waves, rpeaks, mean_rs_ratio, has_prominent_r = self.analyze_ecg_lead(signal, lead_name, sampling_rate, v_threshold)
            status_records[lead_name] = {
                "status": status, 
                "duration": duration, 
                "r/s ratio": mean_rs_ratio,
                # "prominent_r": has_prominent_r
                }
            detail_dict[lead_name] = (ecg_cleaned, time, waves, rpeaks, status, v_threshold)

        df_status = pd.DataFrame.from_dict(status_records, orient='index').reset_index().rename(columns={"index": "lead"})
        return df_status, detail_dict
    
    def identify_contiguous_regions(self, df_status):
        # Step 1: Get leads with ST elevation
        elevated_leads = df_status[df_status['status'].str.contains('elevation', case=False, na=False)]['lead'].tolist()
        elevated_set = set(elevated_leads)

        # Step 2: Get leads with ST depression (used only for Posterior)
        depressed_leads = df_status[df_status['status'].str.contains('depression', case=False, na=False)]['lead'].tolist()
        depressed_set = set(depressed_leads)

        # Step 3: Define contiguous lead patterns for elevation
        contiguous_patterns = {
            "Anterior": [
                {"V1", "V2"}, {"V1", "V3"}, {"V1", "V4"},
                {"V2", "V3"}, {"V2", "V4"}, {"V3", "V4"},
                {"V1", "V2", "V3"}, {"V1", "V2", "V4"},
                {"V1", "V3", "V4"}, {"V2", "V3", "V4"},
                {"V1", "V2", "V3", "V4"}
            ],
            "Anteroseptal": [
                {"V1", "V2"}
            ],
            "Anterolateral": [
                {"V3", "V4"}, {"V3", "V5"}, {"V3", "V6"},
                {"V4", "V5"}, {"V4", "V6"}, {"V5", "V6"},
                {"V3", "V4", "V5"}, {"V3", "V4", "V6"},
                {"V3", "V5", "V6"}, {"V4", "V5", "V6"},
                {"V3", "V4", "V5", "V6"}
            ],
            "Inferior": [
                {"II", "III"}, {"II", "AVF"}, {"III", "AVF"},
                {"II", "III", "AVF"}
            ],
            "Lateral": [
                {"I", "AVL"}, {"I", "V5"}, {"I", "V6"},
                {"AVL", "V5"}, {"AVL", "V6"}, {"V5", "V6"},
                {"I", "AVL", "V5"}, {"I", "AVL", "V6"},
                {"AVL", "V5", "V6"}, {"I", "AVL", "V5", "V6"}
            ],
            "High Lateral":[
                {"I", "AVL"}
            ]
        }

        # Step 4: Match elevated leads to patterns
        detected_regions = []

        for region, patterns in contiguous_patterns.items():
            for pattern in patterns:
                if pattern.issubset(elevated_set):
                    detected_regions.append((region, sorted(list(pattern))))

        # Step 5: Posterior MI logic using depression in V1 - V3
        posterior_patterns = [{"V1", "V2"}, {"V1", "V3"}, {"V2", "V3"}]

        for pattern in posterior_patterns:
            if pattern.issubset(depressed_set):
                df_subset = df_status[df_status['lead'].isin(pattern)]
                rs_ratios = df_subset['r/s ratio']
                r_flags = df_subset['prominent_r']
                if (rs_ratios > 1.0).any() and r_flags.any():
                    detected_regions.append(("Posterior", [f"{lead} (depression + R/S + prominent R)" for lead in sorted(pattern)]))
                    break
                else:
                    detected_regions.append(("Posterior", [f"{lead})" for lead in sorted(pattern)]))
                    break

        return detected_regions
    

