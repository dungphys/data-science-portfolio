"""
ECG Signal Processing Pipeline — Streamlit App
----------------------------------------------
Interactive front-end:
preprocessing -> QRS detection -> beat segmentation -> HRV feature extraction.
 
Run with:
    streamlit run app.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
_real_use = matplotlib.use
matplotlib_use= lambda *a, **k: None
import ecg_pipeline as ep
matplotlib.use = _real_use

TICK_FONT_SIZE = 12  # x/y tick label font size for all Plotly charts below
# set page configuration
st.set_page_config(page_title="ECG Pipeline", page_icon="🫀", layout="wide")
st.title("🫀 ECG Signal Processing")
st.caption("Bandpass + notch filtering → Pan-Tompkins QRS detection → beat segmentation "
    "→ time & frequency-domain HRV features")


# =========================================================================== #
# 1. SIGNAL SOURCE - Sidebar
# =========================================================================== #

st.sidebar.header("ECG Signal Source")
source = st.sidebar.radio(
    "Select ECG Input",
    ["Generate synthetic data", "Upload WFDB (.hea + .mat/.dat)", "Upload CSV"]
)

@st.cache_data(show_spinner="Simulating 12-lead ECG signal...")
def gen_synthetic(duration_s, sampling_rate, heart_rate, noise_level, 
                  powerline_hz, baseline_wander, version, random_state):
    return ep.generate_synthetic_12lead_ecg(
        duration_s=duration_s,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        noise_level=noise_level,
        powerline_hz=powerline_hz,
        powerline_amplitude=0.015,
        baseline_wander_amplitude=baseline_wander,
        random_state=int(random_state),
        version=version,
    )

@st.cache_data(show_spinner="Reading WFDB record...")
def load_wfdb_bytes(hea_bytes, data_bytes, hea_name, data_name):
    """
    hea_bytes: raw binary contents of the uploaded .hea file
    data_bytes: raw binary contents of the uploaded .mat/.dat file
    hea_name: original filename of the header file
    data_name: original filename of the data file
    """
    # Creates a brand-new, empty temporary directory on disk and stores its path in tmpdir
    tmpdir = tempfile.mkdtemp()
    # split A0001.hea -> "A0001" and ".hea" -> keep only "A0001"  
    record_name = os.path.splitext(hea_name)[0]
    with open(os.path.join(tmpdir, hea_name), "wb") as f:
        f.write(hea_bytes)
    with open(os.path.join(tmpdir, data_name), "wb") as f:
        f.write(data_bytes)
    df, sampling_rate = ep.load_ecg_signal(os.path.join(tmpdir, record_name))
    return df, sampling_rate

df = None
time_col = None
lead = None
sampling_rate = 360

if source == "Generate synthetic data": # Synthetic data
    st.sidebar.subheader("Simulation Settings")
    duration_s = st.sidebar.slider("Duration (s)", 5, 60, 10)
    sampling_rate = st.sidebar.select_slider(
        "Sampling rate (Hz)", options=[125, 200, 250, 360, 500], value=360
    )
    heart_rate = st.sidebar.slider("Heart rate (bpm)", 40, 180, 70)
    noise_level = st.sidebar.slider("Noise level", 0.0, 0.2, 0.03, 0.01)
    powerline_hz = st.sidebar.selectbox("Powerline frequency (Hz)", [50.0, 60.0, "None"])
    powerline_hz = None if powerline_hz == "None" else powerline_hz
    baseline_wander = st.sidebar.slider("Baseline wander amplitude", 0.0, 0.5, 0.10, 0.01)
    version = st.sidebar.radio(
        "Lead generation method", ["v1", "v2"],
        help="v1: 8 physiologically independent leads + 4 exactly-derived leads (Einthoven/Goldberger)"
            + "  \n " + "v2: neurokit2's built-in 12-lead simulator (all leads independent, not constrained)",
    )
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)
    if ep.nk is None:
        st.sidebar.error("neurokit2 is not installed. Run: pip install neurokit2")
    else:
        df = gen_synthetic(duration_s, sampling_rate, heart_rate, noise_level,
                            powerline_hz, baseline_wander, version, random_state)
        time_col = "time"
        lead = st.sidebar.selectbox("Lead to analyze", ep.ALL_LEADS, index=ep.ALL_LEADS.index("II"))
elif source == "Upload WFDB (.hea + .mat/.dat)":
    st.sidebar.subheader("WFDB Record")
    hea_file = st.sidebar.file_uploader("Header file (.hea)", type=["hea"])
    data_file = st.sidebar.file_uploader("Data file (.mat or .dat)", type=["mat", "dat"])
    if hea_file and data_file:
        try:
            df, sampling_rate = load_wfdb_bytes(hea_file.getvalue(), data_file.getvalue(),
                                  hea_file.name, data_file.name)
            time_col = "time_s"
            available_leads = [c for c in df.columns if c != time_col]
            lead = st.sidebar.selectbox("Lead to analyze", available_leads)
            st.sidebar.success(f"Loaded record — Sampling rate: {sampling_rate} Hz")
        except Exception as e:
            st.sidebar.error(f"Could not read record: {e}")
    else:
        st.info("Upload both the .hea header file and its matching .mat/.dat data file to continue.")
else:  # Upload CSV
    st.sidebar.subheader("CSV Upload")
    csv_file = st.sidebar.file_uploader("CSV file", type=["csv"])
    if csv_file:
        raw_df = pd.read_csv(csv_file)
        cols = raw_df.columns.tolist()

        time_choice = st.sidebar.selectbox("Time column (if any)", ["None"] + cols)
        lead_options = [c for c in cols if c != time_choice]
        lead = st.sidebar.selectbox("Column to analyze", lead_options)

        if time_choice != "None":
            # derive sampling rate from the chosen time column
            dt = raw_df[time_choice].diff().median()
            inferred_rate = round(1.0 / dt) if dt and dt > 0 else 250
            sampling_rate = st.sidebar.number_input(
                "Sampling rate (Hz)", value=int(inferred_rate), step=1,
                help="Inferred from the median spacing of the time column. Override if needed.",
            )
            time_col = time_choice
        else:
            sampling_rate = st.sidebar.number_input("Sampling rate (Hz)", value=250, step=1)
            raw_df["time"] = np.arange(len(raw_df)) / sampling_rate
            time_col = "time"
        df = raw_df.copy()
    else:
        st.info("Upload a CSV file to continue.")

if (df is None) or (lead is None):
    st.stop()

raw_ecg = df[lead].to_numpy(dtype=float)
t = df[time_col].to_numpy(dtype=float)

# =========================================================================== #
# 2. PIPELINE PARAMETERS - Sidebar
# =========================================================================== #

st.sidebar.header("Pipeline parameters")

with st.sidebar.expander("Filtering", expanded=False):
    st.caption(
            "Bandpass and notch filters"
        )
    low_hz = st.slider("Bandpass low cutoff (Hz)", 0.1, 2.0, 0.5, 0.1)
    high_hz = st.slider("Bandpass high cutoff (Hz)", 10.0, 100.0, 40.0, 1.0)
    order = st.slider("Bandpass filter order", 1, 8, 4)
    freq_hz = st.selectbox("Notch frequency (Hz)", [50.0, 60.0])
    quality_factor = st.slider("Notch quality factor", 5.0, 60.0, 30.0, 1.0)

with st.sidebar.expander("QRS Detection (Pan-Tompkins)", expanded=False):
    tuned_win_size = st.slider("Integration window size (s)", 0.05, 0.4, 0.15, 0.01)
    tuned_min_distance = st.slider("Min. distance between peaks / refractory (s)", 0.1, 1.0, 0.2, 0.01)
    tuned_search_radius = st.slider("Peak refinement search radius (s)", 0.1, 0.9, 0.4, 0.01)

with st.sidebar.expander("Beat Segmentation", expanded=False):
    pre = st.slider("Window before R-peak (s)", 0.05, 0.5, 0.25, 0.01)
    post = st.slider("Window after R-peak (s)", 0.1, 0.8, 0.4, 0.01)

with st.sidebar.expander("HRV & Quality Control", expanded=False):
    freq_interp = st.slider("RR resampling rate for LF/HF (Hz)", 2.0, 8.0, 4.0, 0.5)
    min_bpm = st.slider("Min. plausible heart rate (bpm)", 20, 60, 30)
    max_bpm = st.slider("Max. plausible heart rate (bpm)", 150, 360, 220)

with st.sidebar.expander("Wave delineation (P/Q/S/T)", expanded=False):
    st.caption(
        "Heuristic window-search delineator  \n"
        "Treat PR/QRS/QT/QTc/ST/P/T values as approximate."
    )
    q_search_s = st.slider("Q search window before R (s)", 0.02, 0.10, 0.05, 0.01)
    s_search_s = st.slider("S search window after R (s)", 0.02, 0.12, 0.06, 0.01)
    p_far, p_near = st.slider(
        "P search window before R (s)", 0.05, 0.40, (0.10, 0.28), 0.01,
        help="(near, far) bounds before R to search for the P wave.",
    )
    t_near, t_far = st.slider(
        "T search window after R (s)", 0.05, 0.60, (0.15, 0.40), 0.01,
        help="(near, far) bounds after R to search for the T wave.",
    )
    st_offset_s = st.slider("ST measurement point past S / J-point (s)", 0.02, 0.12, 0.06, 0.01)
 


# =========================================================================== #
# 3. RUN PIPELINE
# =========================================================================== #
pipeline = ep.ECGPipeline(
    sampling_rate=sampling_rate,
    low_hz=low_hz, high_hz=high_hz, order=order,
    freq_hz=freq_hz, quality_factor=quality_factor,
    tuned_win_size=tuned_win_size, tuned_min_distance=tuned_min_distance,
    tuned_search_radius=tuned_search_radius,
    pre=pre, post=post,
    freq_interp=freq_interp,
    min_bpm=min_bpm, max_bpm=max_bpm,
    q_search_s=q_search_s, s_search_s=s_search_s,
    p_search_s=(p_far, p_near), t_search_s=(t_near, t_far),
    st_offset_s=st_offset_s,
)
result = pipeline.run(raw_ecg=raw_ecg)
clean_ecg = result["filtered_ecg"]
r_peaks = result["r_peaks"]
beats = result["beats"]
features = result["features"]
waves = result["waves"]
intervals = result["intervals"]

# =========================================================================== #
# 4. HEADLINE METRICS
# =========================================================================== #
st.subheader(f"Results — Lead {lead}")
cols = st.columns(7)
cols[0].metric("Heart rate", f"{features.heart_rate_bpm:.1f} bpm" if np.isfinite(features.heart_rate_bpm) else "—")
cols[1].metric("Beats detected", features.n_beats)
cols[2].metric("Rejected (QC)", features.rejected_beats)
cols[3].metric("SDNN", f"{features.sdnn_ms:.1f} ms" if np.isfinite(features.sdnn_ms) else "—")
cols[4].metric("RMSSD", f"{features.rmssd_ms:.1f} ms" if np.isfinite(features.rmssd_ms) else "—")
cols[5].metric("pNN50", f"{features.pnn50:.2f} %" if np.isfinite(features.pnn50) else "—")
cols[6].metric("LF/HF", f"{features.lf_hf_ratio:.2f}" if np.isfinite(features.lf_hf_ratio) else "—")

st.caption(
    "PR / QRS / QT / QTc / ST / P & T amplitude below come from a heuristic "
    "window-search wave delineator (median across beats) — approximate, not "
    "a clinically-validated measurement."
)
ccols = st.columns(7)
ccols[0].metric("PR interval", f"{features.pr_interval_ms:.0f} ms" if np.isfinite(features.pr_interval_ms) else "—")
ccols[1].metric("QRS duration", f"{features.qrs_duration_ms:.0f} ms" if np.isfinite(features.qrs_duration_ms) else "—")
ccols[2].metric("QT interval", f"{features.qt_interval_ms:.0f} ms" if np.isfinite(features.qt_interval_ms) else "—")
ccols[3].metric("QTc (Bazett)", f"{features.qtc_ms:.0f} ms" if np.isfinite(features.qtc_ms) else "—")
ccols[4].metric("ST elevation", f"{features.st_elevation_mv:+.3f} mV" if np.isfinite(features.st_elevation_mv) else "—")
ccols[5].metric("P amplitude", f"{features.p_amplitude_mv:.3f} mV" if np.isfinite(features.p_amplitude_mv) else "—")
ccols[6].metric("T amplitude", f"{features.t_amplitude_mv:.3f} mV" if np.isfinite(features.t_amplitude_mv) else "—")
 

# =========================================================================== #
# 5. RAW vs FILTERED ECG WITH R-PEAKS
# =========================================================================== #
st.markdown("### Signal & R-peak detection")

t_min, t_max = float(t[0]), float(t[-1])
show_raw = st.checkbox("Overlay raw (unfiltered) signal", value=False)
show_waves = st.checkbox("Overlay P/Q/S/T wave markers", value=False)

if t_max > t_min:
    default_window = min(10.0, t_max - t_min)
    window_start, window_end = st.slider(
        "Time range to display (s)",
        min_value=t_min, max_value=t_max,
        value=(t_min, t_min + default_window),
        step=max((t_max - t_min) / 500, 0.01),
        help="Coarse windowing here; drag/zoom or use the range slider below the chart for fine control.",
    )
else:
    window_start, window_end = t_min, t_max

mask = (t >= window_start) & (t <= window_end)
t_view = t[mask]
raw_view = raw_ecg[mask]
clean_view = clean_ecg[mask]
peaks_view = r_peaks[(t[r_peaks] >= window_start) & (t[r_peaks] <= window_end)] if len(r_peaks) > 0 else r_peaks


fig = go.Figure()
if show_raw:
    fig.add_trace(go.Scatter(x=t_view, y=raw_view, name="Raw", line=dict(color="lightgray", width=1)))
fig.add_trace(go.Scatter(x=t_view, y=clean_view, name="Filtered", line=dict(color="#1f77b4", width=1.3)))
if len(peaks_view) > 0:
    fig.add_trace(go.Scatter(
        x=t[peaks_view], y=clean_ecg[peaks_view], mode="markers", name="R-peaks",
        marker=dict(color="red", size=8, symbol="circle"),
    ))
if show_waves and waves:
    def _wave_trace(attr, color, name, symbol="circle"):
        idxs = [int(round(getattr(w, attr))) for w in waves
                if not np.isnan(getattr(w, attr)) and window_start <= t[int(round(getattr(w, attr)))] <= window_end]
        if idxs:
            fig.add_trace(go.Scatter(
                x=t[idxs], y=clean_ecg[idxs], mode="markers", name=name,
                marker=dict(color=color, size=7, symbol=symbol),
            ))
    _wave_trace("q", "blue", "Q", symbol="triangle-down")
    _wave_trace("s", "green", "S", symbol="triangle-up")
    _wave_trace("p_peak", "purple", "P")
    _wave_trace("t_peak", "orange", "T")
fig.update_layout(
    height=420, xaxis_title="Time (s)", yaxis_title="Amplitude",
    margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.05),
    xaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
    yaxis=dict(tickfont=dict(size=TICK_FONT_SIZE))
)
fig.update_xaxes(rangeslider=dict(visible=True), rangeslider_thickness=0.08)
st.plotly_chart(fig, width="stretch")

# =========================================================================== #
# 6. RR TACHOGRAM + POINCARÉ PLOT
# =========================================================================== #
if len(r_peaks) >= 2:
    rr_ms = ep.rr_intervals_ms(r_peaks, sampling_rate)
    valid = ep.quality_control(rr_ms, min_bpm=min_bpm, max_bpm=max_bpm)

    st.markdown("### Heart-rate variability")
    c1, c2 = st.columns(2)

    with c1:
        tacho = go.Figure()
        tacho.add_trace(go.Scatter(
            y=rr_ms, mode="lines+markers", name="RR interval",
            marker=dict(color=np.where(valid, "#1f77b4", "crimson"), size=6),
            line=dict(color="#1f77b4", width=1),
        ))
        tacho.update_layout(
            title="RR tachogram (red = rejected by QC)",
            xaxis_title="Beat index", yaxis_title="RR interval (ms)",
            height=350, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
            yaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
        )
        st.plotly_chart(tacho, width="stretch")

    with c2:
        if len(rr_ms) >= 2:
            poincare = go.Figure()
            poincare.add_trace(go.Scatter(
                x=rr_ms[:-1], y=rr_ms[1:], mode="markers",
                marker=dict(color="#2ca02c", size=6, opacity=0.7),
            ))
            lo, hi = np.min(rr_ms), np.max(rr_ms)
            poincare.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                           line=dict(color="gray", dash="dash"), showlegend=False))
            poincare.update_layout(
                title=r"Poincaré plot: RR-n vs RR-(n+1)",
                xaxis_title="RRₙ (ms)", yaxis_title="RRₙ₊₁ (ms)",
                height=350, margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
                yaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
            )
            st.plotly_chart(poincare, width='stretch')
else:
    st.info("Fewer than 2 R-peaks detected — not enough data for HRV analysis.")

# =========================================================================== #
# 7. SEGMENTED BEATS OVERLAY
# =========================================================================== #
st.markdown("### Segmented beats")
if beats:
    beat_t = (np.arange(len(beats[0])) - int(pre * sampling_rate)) / sampling_rate
    beats_fig = go.Figure()
    max_overlay = min(len(beats), 150)
    for b in beats[:max_overlay]:
        beats_fig.add_trace(go.Scatter(
            x=beat_t, y=b, mode="lines", line=dict(color="lightsteelblue", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    mean_beat = np.mean(np.vstack(beats), axis=0)
    beats_fig.add_trace(go.Scatter(
        x=beat_t, y=mean_beat, mode="lines", name="Mean beat",
        line=dict(color="#d62728", width=2.5),
    ))
    beats_fig.update_layout(
        height=380, xaxis_title="Time relative to R-peak (s)", yaxis_title="Amplitude",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=TICK_FONT_SIZE)),
    )
    st.plotly_chart(beats_fig, width='stretch')
    st.caption(f"Showing {min(len(beats), max_overlay)} of {len(beats)} segmented beats, plus the average beat.")
else:
    st.info("No beats could be segmented (R-peaks too close to signal edges, or none detected).")
# =========================================================================== #
# 8. DOWNLOADS
# =========================================================================== #
st.markdown("### Export")
d1, d2, d3 = st.columns(3)

features_df = pd.DataFrame([{
    "lead": lead,
    "sampling_rate_hz": sampling_rate,
    "heart_rate_bpm": features.heart_rate_bpm,
    "mean_rr_ms": features.mean_rr_ms,
    "sdnn_ms": features.sdnn_ms,
    "rmssd_ms": features.rmssd_ms,
    "pnn50_pct": features.pnn50,
    "lf_hf_ratio": features.lf_hf_ratio,
    "n_beats": features.n_beats,
    "rejected_beats": features.rejected_beats,
    "pr_interval_ms": features.pr_interval_ms,
    "qrs_duration_ms": features.qrs_duration_ms,
    "qt_interval_ms": features.qt_interval_ms,
    "qtc_ms": features.qtc_ms,
    "st_elevation_mv": features.st_elevation_mv,
    "p_amplitude_mv": features.p_amplitude_mv,
    "t_amplitude_mv": features.t_amplitude_mv,
}])
d1.download_button(
    "Download features (CSV)",
    features_df.to_csv(index=False).encode("utf-8"),
    file_name="ecg_features.csv",
    mime="text/csv",
)

signal_df = pd.DataFrame({"time_s": t, "raw": raw_ecg, "filtered": clean_ecg})
signal_df["r_peak"] = False
if len(r_peaks) > 0:
    signal_df.loc[r_peaks, "r_peak"] = True
d2.download_button(
    "Download filtered signal + R-peaks (CSV)",
    signal_df.to_csv(index=False).encode("utf-8"),
    file_name="ecg_filtered_signal.csv",
    mime="text/csv",
)

if waves:
    beats_df = pd.DataFrame([
        {
            "beat_index": i,
            "r_sample": w.r,
            "q_sample": w.q, "s_sample": w.s,
            "p_onset_sample": w.p_onset, "p_peak_sample": w.p_peak,
            "t_peak_sample": w.t_peak, "t_offset_sample": w.t_offset,
            "pr_interval_ms": ci.pr_interval_ms,
            "qrs_duration_ms": ci.qrs_duration_ms,
            "qt_interval_ms": ci.qt_interval_ms,
            "qtc_ms": ci.qtc_ms,
            "st_elevation_mv": ci.st_elevation_mv,
            "p_amplitude_mv": ci.p_amplitude_mv,
            "t_amplitude_mv": ci.t_amplitude_mv,
        }
        for i, (w, ci) in enumerate(zip(waves, intervals))
    ])
    d3.download_button(
        "Download per-beat intervals (CSV)",
        beats_df.to_csv(index=False).encode("utf-8"),
        file_name="ecg_beat_intervals.csv",
        mime="text/csv",
    )