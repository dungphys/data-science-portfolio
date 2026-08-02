"""
ecg_pipeline.py
- ECG signal generation using neurokit2
- Loading single-lead ECG signals from files
- Filter
- QRS detector
- Beat segmentation
- Feature extraction

Author: Anh Dung Le
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import wfdb
import matplotlib.pyplot as plt


try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None

INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
DERIVED_LEADS = ["III", "aVR", "aVL", "aVF"]
ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Relative amplitude weighting applied to each independent lead's noise components. 
_LEAD_NOISE_SCALE = {
    "I": 1.0, "II": 1.0,
    "V1": 0.8, "V2": 0.9, "V3": 1.0, "V4": 1.1, "V5": 1.0, "V6": 0.9,
    "III": 1.0, "aVR": 1.0, "aVL": 1.0, "aVF": 1.0,
}

# --------------------------------------------------------------------------- #
# 1. DATA ACQUISITION
# --------------------------------------------------------------------------- #

# --------------- Signal generation --------------- #
"""
Only 8 of the 12 standard leads are physiologically independent
(I, II, V1-V6); the remaining 4 (III, aVR, aVL, aVF) are exact linear
combinations of I and II, a consequence of all frontal-plane leads
viewing the same single cardiac dipole (Einthoven's triangle).

neurokit2's built-in ``method="multileads"`` simulator generates all 12
leads independently and does NOT enforce these constraints exactly.

We define two generating codes:
- v1: Uses neurokit2's built-in ``method="multileads"`` simulator to generate 
all 12 leads.
- v2: Treats neurokit2's I, II, and V1-V6 outputs as the 8 physiologically 
independent leads, and derives III, aVR, aVL, aVF from I and II.

"""
def compute_derived_leads(lead_I: np.ndarray, lead_II: np.ndarray) -> dict[str, np.ndarray]:
    """Exact derivation of III, aVR, aVL, aVF from I and II (Einthoven + Goldberger).
    Parameters
    ----------
        lead_I: lead-I signal
        lead_II: lead-II signal
    Returns
    ----------
        dictionary containing derived lead signals 
    """
    lead_III = lead_II - lead_I
    lead_aVR = -(lead_I + lead_II) / 2
    lead_aVL = lead_I - lead_II / 2
    lead_aVF = lead_II - lead_I / 2
    return {"III": lead_III, "aVR": lead_aVR, "aVL": lead_aVL, "aVF": lead_aVF}

def generate_synthetic_12lead_ecg(
    duration_s: int = 10,
    sampling_rate: int = 250,
    heart_rate: int = 70,
    noise_level: float = 0.03,
    powerline_hz: float | None = 50.0,
    powerline_amplitude: float = 0.015,
    baseline_wander_amplitude: float = 0.10,
    random_state: int = 42,
    version: str = "v1"
) -> pd.DataFrame | None:
    """Generate a synthetic 12-lead ECG.
    Parameters
    ----------
        duration_s: recording length in second (int)
        sampling rate: samples per second
        heart_rate: target heart rate in beat-per-min (bpm).
        noise_level: white-noise standard deviation (in the same amplitude
            units as neurokit2's clean output, roughly mV-scale)
        powerline_hz : mains frequency to inject (50, 60, or None to disable).
        powerline_amplitude : amplitude of the shared powerline interference.
        baseline_wander_amplitude : amplitude of the low-frequency drift.
        random_state : seed for reproducibility. 
        version: either "v1" (8 independent + 4 derived) or 
            "v2" (12 independent)
    Returns
    ----------
        Dataframe storing the gererated signal or None 
    """ 
    if nk is None:
        raise ImportError("neurokit2 is required for 12-lead ECG generation.")

    rng = np.random.default_rng(random_state) # random number generator

    # generate a clean multilead signal
    clean_multilead = nk.ecg_simulate(
        duration=duration_s,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        method="multileads",
        random_state=random_state,
    )
    n = len(clean_multilead)
    t = np.arange(n) / sampling_rate # time 

    # shared powerline interference (same phase across the montage)
    powerline = np.zeros(n)
    if powerline_hz:
        powerline = powerline_amplitude * np.sin(2 * np.pi * powerline_hz * t)

    if version=="v1":
        # takes 8 physiologically independent leads from neurokit2
        leads: dict[str, np.ndarray] = {
            lead: clean_multilead[lead].to_numpy(dtype=float) for lead in INDEPENDENT_LEADS
        }
        # per-lead independent noise + baseline wander
        for lead in INDEPENDENT_LEADS:
            scale = _LEAD_NOISE_SCALE[lead]
            white_noise = rng.normal(0, noise_level * scale, size=n)
            wander = baseline_wander_amplitude * scale * np.sin(
                2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi)
            )
            leads[lead] = leads[lead] + white_noise + wander + powerline
        # derived leads
        derived = compute_derived_leads(leads["I"], leads["II"])
        leads.update(derived)
    elif version=="v2":
        # takes all 12 physiologically independent leads from neurokit2
        leads: dict[str, np.ndarray] = {
            lead: clean_multilead[lead].to_numpy(dtype=float) for lead in ALL_LEADS
        }
        # per-lead independent noise + baseline wander
        for lead in ALL_LEADS:
            scale = _LEAD_NOISE_SCALE[lead]
            white_noise = rng.normal(0, noise_level * scale, size=n)
            wander = baseline_wander_amplitude * scale * np.sin(
                2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi)
            )
            leads[lead] = leads[lead] + white_noise + wander + powerline
    else:
        print("version can only be v1 or v2")
        return None 

    # create dataframe
    df = pd.DataFrame({"time": t})
    for lead in ALL_LEADS:
        df[lead] = leads[lead]
 
    return df

# --------------- Signal loading from files --------------- #
def load_ecg_signal(record: list[str]) -> list[pd.DataFrame]:
    """Extract ecg signals from the given list of ecg files.
    Parameters
    ----------
        record: list of filename (without 'hea' and 'mat')
    Returns
    ----------
        Dataframe 
    """
    
    record = wfdb.rdrecord(record) # basically read two ecg files: <name>.hea and <name>.mat  
    df = pd.DataFrame(
        record.p_signal,           # signal 
        columns=record.sig_name    # signal lead (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
    )
    df.insert(0, 'time_s', np.arange(len(df)) / record.fs) # time from frequency
    
    return df, record.fs


# --------------------------------------------------------------------------- #
# 2. PREPROCESSING
# --------------------------------------------------------------------------- #
"""
Preprocessing chain: 2 options 
1. notch filter -> bandpass filter 
2. bandpass filter -> notch filter 

Selection: Option 2 (not a hard rule, just a practical choice)
    - big baseline swings can cause ringing/transients in a narrow filter; bandpassing first calms the signal down for it.
    - bandpass already knocks down most 50/60 Hz energy via its high cutoff; notch just mops up the remainder.
      (less work for notch)
    - doing the broad cut first, then the narrow cut, avoids stacking filtfilt boundary distortion on an unshaped signal.
      (fewer edge artifacts)
"""

# --------------- Bandpass filter --------------- #
def bandpass_filter(
    ecg: np.ndarray,
    sampling_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (standard diagnostic ECG band).
        ( Low-pass + high-pass included )
    Parameters
    ----------
        ecg: signal array
        sampling_rate: samples per second (freq)
        low_hz: below cutoff in Hz
        high_hz: above cutoff in Hz
        order: sets how sharp the roll-off is (higher = steeper cutoff, but more risk of ringing artifacts) 
    Returns
    ----------
        filtered signal array 
    """
    # Nyquist frequency: the maximum frequency that the sampled signal can represent at all.
    nyquist = sampling_rate / 2  
    # low- and high-cutoff freq fractions in term of Nyquist freq
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99) # safety clam (0.99) for butter
    # Butterworth bandpass filter
    # passing only frequencies between low and high, using the two normalized cutoffs
    b, a = sp_signal.butter(order, [low, high], btype="bandpass")
    # padlen = 500# min(len(ecg) - 1, int(3 * sampling_rate / low_hz))
    return sp_signal.filtfilt(b, a, ecg) # apply filter to signal

# --------------- Notch filter --------------- #
def notch_filter(
    ecg: np.ndarray,
    sampling_rate: float,
    freq_hz: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """IIR notch filter to remove powerline interference (50 or 60 Hz).
    Parameters
    ----------
        ecg: signal array
        sampling_rate: samples per second
        freq_hz: frequency to notch out in Hz
        quality_factor: controls how narrow the blocked region is (higher: narrower notch)  
    Returns
    ----------
        filtered signal array
    """
    nyquist = sampling_rate / 2 # Nyquist frequency
    if not 0 < freq_hz < nyquist:
        raise ValueError(
            f"freq_hz={freq_hz} must be less than the Nyquist frequency ({nyquist} Hz) "
            f"for sampling_rate={sampling_rate}."
        )
    # notch filter: blocks a narrow range around w0 and passes everything else
    b, a = sp_signal.iirnotch(freq_hz, quality_factor, sampling_rate)
    return sp_signal.filtfilt(b, a, ecg) # apply the filter to signal

# --------------- Full preprocessing pipeline --------------- #
def preprocess_pipeline(
    ecg: np.ndarray, 
    sampling_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
    freq_hz: float = 50.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """Full processing chain.
    Parameters
    ----------
        ecg: signal array
        sampling_rate: samples per second
        low_hz: below cutoff,
        high_hz: above cutoff,
        order: int = 4,
        freq_hz: frequency to notch out in Hz
        quality_factor: controls how narrow the blocked region is (higher: narrower notch)
    Returns
    ----------
        filtered signal array
    """
    z = bandpass_filter(
        ecg=ecg, sampling_rate=sampling_rate, 
        low_hz=low_hz, high_hz=high_hz, order=order,
    )
    z = notch_filter(
        ecg=z, sampling_rate=sampling_rate,
        freq_hz=freq_hz, quality_factor=quality_factor
    )
    return z

# --------------------------------------------------------------------------- #
# 3. QRS DETECTION (Pan-Tompkins)
# --------------------------------------------------------------------------- #
"""
QRS complex: representing the electrical activity as the heart's ventricles contract.
    - Q: first small downward deflection
    - R: the tall upward spike -> QRS detector looks for this
    - S: downward deflection right after R
QRS detection: finding each R-peak that represents ventricular depolarization in the ECG signal.
"""
def detect_qrs(
    ecg: np.ndarray, 
    sampling_rate: float,
    tuned_win_size: float=0.15,
    tuned_min_distance: float=0.2,
    tuned_search_radius: float=0.4
) -> np.ndarray:
    """Classic Pan-Tompkins QRS detector -> find R-peak. 
    Parameters
    ----------
        ecg: signal array
        sampling_rate: samples per second
        tuned_win_size: parameter to tune the window size
        tuned_min_distance: parameter to tune the min distance
        tuned_search_radius: parameter to tune the search radius
    Returns
    ----------
        indices of detected R-peaks (in samples)
    """
    # bandpass 5-15 Hz filter
    # QRS complexes concentrate most of their spectral energy in roughly the 5–15 Hz range 
    # (sometimes ~10-25 Hz)
    nyquist = sampling_rate / 2 # Nyquist frequency
    order = 2
    b, a = sp_signal.butter(order, [5/nyquist, 15/nyquist], btype='bandpass')
    filtered = sp_signal.filtfilt(b, a, ecg)

    # derivative -> compute slope at each point
    # QRS complexes have steep slopes -> emphasizes them and suppresses slower-changing waves
    derivative = np.diff(filtered, prepend=filtered[0])

    # squaring -> makes everthing positive and nonlinearly amplifies values more than smaller ones
    squared = derivative**2

    # moving window integration (~150ms window: classic Pan-Tompkins choice)
    #   - Wide enough to merge the whole QRS into one smooth bump
    #   - Narrow enough not to merge separate beats
    # -> smooth the squared signal into a single energy burst per beat
    # giving one clear bump per QRS complex
    win_size = int(tuned_win_size*sampling_rate) # can be tuned
    integrated = np.convolve(squared, np.ones(win_size)/win_size, mode="same")

    # adaptive thresholding + refactory period -> find peaks
    #   - too short refactory period: risks double-detecting a single QRS complex as two peaks
    #   - too long refactory period: incorrectly reject genuine consecutive beats in patients with fast heart rate  
    min_distance = int(tuned_min_distance*sampling_rate) # minimum distance between detected peaks (refactory period)
    threshold = 0.5 * np.max(integrated)
    peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=min_distance)

    def parabolic_refine(y, i):
            if i <= 0 or i >= len(y)-1:
                return float(i)
            denom = (y[i-1] - 2*y[i] + y[i+1])
            if denom == 0:
                return float(i)
            delta = 0.5 * (y[i-1] - y[i+1]) / denom
            return i + delta
    # peak refinement: snap each detected peak to the true local max in the raw signal
    search_radius = int(tuned_search_radius*sampling_rate) # able to tune
    refined_peaks = []
    for p in peaks:
        lo, hi = p - search_radius, p + search_radius
        if lo < 0 or hi > len(ecg):
            refined_peaks.append(p)  # can't get a full symmetric window, don't refine
            continue
        #lo, hi = max(0, p-search_radius), min(len(ecg), p+search_radius)
        window = ecg[lo:hi]
        raw_idx = np.argmax(window)
        sub_idx = parabolic_refine(window, raw_idx)        
        refined_peaks.append(int(round(lo+sub_idx)))
    refined_peaks = np.array(sorted(set(refined_peaks)))

    # RR-interval sanity check: catch any remaining too-close pairs
    # (e.g. genuine double detections that refinement alone can't fix)
    
    if len(refined_peaks) > 1:
        rr = np.diff(refined_peaks)
        drop = []
        i = 0
        while i < len(rr):
            if rr[i] < min_distance:
                # keep whichever of the pair has higher raw-signal amplitude
                a, b_idx = refined_peaks[i], refined_peaks[i+1]
                drop.append(i if ecg[a] < ecg[b_idx] else i+1)
            i += 1
        if drop:
            refined_peaks = np.delete(refined_peaks, drop)
    return np.array(sorted(set(refined_peaks)))

# --------------------------------------------------------------------------- #
# 4. BEAT SEGMENTATION
# --------------------------------------------------------------------------- #

def segment_beats(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    pre: float=0.25,
    post: float=0.4
) -> list[np.ndarray]:
    """Extract fixed-length windows around each R-peak.
    Parameters
    ----------
        ecg: signal array (filtered)
        r_peaks: array of sample indices where R-peaks were detected
        sampling_rate: samples per second (in Hz) 
        pre: how many seconds before each R-peak to include in the window (in second) 
        post: how many seconds after each R-peak to include in the window (in second)
    Returns
    ----------
        list of beat arrays (may vary near edges)
    """
    # convert pre/post time durations into sample counts
    pre_s, post_s = int(pre*sampling_rate), int(post*sampling_rate)
    beats = [] # to collect each segmented beat (each beat is one array slice of ecg)
    for r in r_peaks:
        # compute the start and end indices around each r-peak
        lo, hi = r - pre_s, r + post_s
        if lo >=0 and hi <= len(ecg): # boundary check -> skip if out of bound
            beats.append(ecg[lo:hi])
    return beats

# --------------------------------------------------------------------------- #
# 5. WAVE DELINEATION (P, Q, S, T) AND CLINICAL INTERVALS
# --------------------------------------------------------------------------- #
"""
Extends R-peak detection with a lightweight, heuristic delineation of the P,
Q, S, and T waves for each beat, then derives standard clinical intervals
from them (PR interval, QRS duration, QT/QTc, ST elevation, P/T amplitude).
 
This is a window-search + threshold-crossing delineator, NOT a wavelet-based
method like neurokit2's ecg_delineate. It is deliberately simple and tunable,
in the same spirit as the Pan-Tompkins QRS detector above, and is good enough
for a clean-ish, bandpass-filtered single-lead signal on a portfolio-scale
project. It is not a clinically-validated delineator and should not be used
to make real diagnostic claims, especially on noisy or pathological signals.
 
Simplifications made (documented, not hidden):
    - QRS onset/offset are approximated by the Q and S local minima
      themselves, rather than a separate onset/offset search.
    - PR interval uses a threshold-crossing P-onset detector rather than the
      more robust tangent method.
    - QT interval end (T-offset) uses the same threshold-crossing method,
      not the tangent method typically used clinically.
    - QTc uses Bazett's formula (QTc = QT / sqrt(RR)), which is the most
      common convention but is known to over-correct at fast heart rates.
    - ST elevation is read at a single point (J-point + a fixed offset)
      relative to an isoelectric baseline sampled from the PR segment,
      rather than fitting the ST segment shape.
"""
 
@dataclass
class BeatWaves:
    """Per-beat delineation result. All wave locations are sample indices
    into the full (filtered) ECG array; NaN means "not found" for that beat.
    """
    r: int
    q: float = np.nan
    s: float = np.nan
    p_onset: float = np.nan
    p_peak: float = np.nan
    t_peak: float = np.nan
    t_offset: float = np.nan
    baseline: float = np.nan  # isoelectric amplitude near this beat (not an index)
 
 
def _find_local_extremum(ecg: np.ndarray, lo: int, hi: int, kind: str = "min") -> float:
    """Index (in the full-signal frame) of the local min/max of ecg[lo:hi].
    Returns NaN if the window is empty or out of bounds.
    """
    lo, hi = max(0, lo), min(len(ecg), hi)
    if hi - lo < 2:
        return np.nan
    window = ecg[lo:hi]
    idx = int(np.argmin(window)) if kind == "min" else int(np.argmax(window))
    return float(lo + idx)
 
 
def _find_onset(ecg: np.ndarray, peak_idx: float, sampling_rate: float,
                 max_back_s: float, frac: float = 0.5) -> float:
    """Walk backward from `peak_idx` until the signal crosses `frac` of the
    way from the peak amplitude back toward the pre-peak baseline sample.
    Simple threshold-crossing onset detector (not a tangent/derivative method).
    """
    if np.isnan(peak_idx):
        return np.nan
    peak_i = int(round(peak_idx))
    lo = max(0, peak_i - int(max_back_s * sampling_rate))
    if peak_i <= lo:
        return float(lo)
    baseline, peak_val = ecg[lo], ecg[peak_i]
    threshold = baseline + frac * (peak_val - baseline)
    rising = peak_val >= baseline
    for i in range(peak_i, lo, -1):
        if (rising and ecg[i] <= threshold) or (not rising and ecg[i] >= threshold):
            return float(i)
    return float(lo)
 
 
def _find_offset(ecg: np.ndarray, peak_idx: float, sampling_rate: float,
                  max_fwd_s: float, frac: float = 0.5) -> float:
    """Mirror of `_find_onset`, walking forward from the peak."""
    if np.isnan(peak_idx):
        return np.nan
    peak_i = int(round(peak_idx))
    hi = min(len(ecg) - 1, peak_i + int(max_fwd_s * sampling_rate))
    if peak_i >= hi:
        return float(hi)
    baseline, peak_val = ecg[hi], ecg[peak_i]
    threshold = baseline + frac * (peak_val - baseline)
    rising = peak_val >= baseline
    for i in range(peak_i, hi):
        if (rising and ecg[i] <= threshold) or (not rising and ecg[i] >= threshold):
            return float(i)
    return float(hi)
 
 
def delineate_beat(
    ecg: np.ndarray,
    r: int,
    sampling_rate: float,
    prev_r: Optional[int] = None,
    next_r: Optional[int] = None,
    q_search_s: float = 0.05,
    s_search_s: float = 0.06,
    p_search_s: tuple[float, float] = (0.28, 0.10),  # (far, near) before R
    t_search_s: tuple[float, float] = (0.15, 0.40),  # (near, far) after R
    onset_offset_frac: float = 0.5,
) -> BeatWaves:
    """Delineate P, Q, S, T waves for a single beat around R-peak `r`.
    Parameters
    ----------
        ecg: filtered signal array
        r: sample index of this beat's R-peak
        sampling_rate: samples per second
        prev_r, next_r: neighboring R-peak indices (if any), used to keep the
            P/T search windows from spilling into the adjacent beat
        q_search_s, s_search_s: how far before/after R to search for Q/S (s)
        p_search_s: (far, near) bounds before R to search for the P wave (s)
        t_search_s: (near, far) bounds after R to search for the T wave (s)
        onset_offset_frac: threshold fraction used by the onset/offset search
    Returns
    ----------
        BeatWaves with sample-index locations for this beat (NaN if not found)
    """
    waves = BeatWaves(r=r)
 
    # --- Q: local minimum just before R
    waves.q = _find_local_extremum(ecg, r - int(q_search_s * sampling_rate), r, kind="min")
 
    # --- S: local minimum just after R
    waves.s = _find_local_extremum(ecg, r, r + int(s_search_s * sampling_rate), kind="min")
 
    # --- isoelectric baseline: sampled from the PR segment, just before the P wave window
    baseline_lo = max(0, r - int(p_search_s[1] * sampling_rate))
    baseline_hi = max(0, r - int(p_search_s[0] * sampling_rate))
    if baseline_hi > baseline_lo:
        waves.baseline = float(np.median(ecg[baseline_lo:baseline_hi]))
    else:
        waves.baseline = float(ecg[max(0, r - int(0.24 * sampling_rate))])
 
    # --- P: local maximum before QRS, bounded by the previous beat's T wave if known
    p_far = r - int(p_search_s[1] * sampling_rate)
    p_near = r - int(p_search_s[0] * sampling_rate)
    if prev_r is not None:
        p_far = max(p_far, prev_r + int(0.10 * sampling_rate))
    waves.p_peak = _find_local_extremum(ecg, p_far, p_near, kind="max")
    waves.p_onset = _find_onset(ecg, waves.p_peak, sampling_rate, max_back_s=0.08, frac=onset_offset_frac)
 
    # --- T: local maximum after QRS, bounded by the next beat's P wave if known
    t_near = r + int(t_search_s[0] * sampling_rate)
    t_far = r + int(t_search_s[1] * sampling_rate)
    if next_r is not None:
        t_far = min(t_far, next_r - int(0.05 * sampling_rate))
    waves.t_peak = _find_local_extremum(ecg, t_near, t_far, kind="max")
    waves.t_offset = _find_offset(ecg, waves.t_peak, sampling_rate, max_fwd_s=0.15, frac=onset_offset_frac)
 
    return waves
 
 
def delineate_all_beats(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    **delineate_kwargs,
) -> list[BeatWaves]:
    """Run `delineate_beat` across every detected R-peak, giving each beat
    its neighbors so the P/T search windows don't cross into adjacent beats.
    """
    waves_list = []
    for i, r in enumerate(r_peaks):
        prev_r = int(r_peaks[i - 1]) if i > 0 else None
        next_r = int(r_peaks[i + 1]) if i < len(r_peaks) - 1 else None
        waves_list.append(delineate_beat(ecg, int(r), sampling_rate, prev_r=prev_r, next_r=next_r, **delineate_kwargs))
    return waves_list
 
 
@dataclass
class ClinicalIntervals:
    """Per-beat clinical intervals/amplitudes derived from `BeatWaves`."""
    pr_interval_ms: float = np.nan
    qrs_duration_ms: float = np.nan
    qt_interval_ms: float = np.nan
    qtc_ms: float = np.nan          # Bazett-corrected QT
    st_elevation_mv: float = np.nan
    p_amplitude_mv: float = np.nan
    t_amplitude_mv: float = np.nan
 
 
def compute_beat_intervals(
    waves: BeatWaves,
    ecg: np.ndarray,
    sampling_rate: float,
    rr_s: Optional[float],
    st_offset_s: float = 0.06,
    pr_bounds_ms: tuple[float, float] = (80.0, 300.0),
    qrs_bounds_ms: tuple[float, float] = (40.0, 200.0),
    qtc_bounds_ms: tuple[float, float] = (300.0, 600.0),
) -> ClinicalIntervals:
    """Compute clinical intervals for one beat from its delineated waves.
 
    A window-search delineator will occasionally lock onto the wrong local
    extremum (e.g. baseline wander instead of a true P wave), which silently
    produces a physiologically impossible interval rather than an obviously
    wrong one. To keep a handful of mis-delineated beats from skewing the
    aggregate, PR/QRS/QTc are discarded (set to NaN) when they fall outside
    generous physiological bounds -- the same philosophy already used by
    `quality_control()` for RR intervals above. Bounds are intentionally wide
    (they span normal *and* clinically abnormal-but-real values, e.g. first-
    degree AV block or bundle branch block) so real pathology is not filtered
    out; they only catch delineation artifacts.
    Parameters
    ----------
        waves: this beat's BeatWaves
        ecg: filtered signal array (for reading amplitudes)
        sampling_rate: samples per second
        rr_s: the RR interval (in seconds) associated with this beat, used
            for Bazett QTc correction; None/NaN/<=0 skips QTc for this beat
        st_offset_s: how far past the S wave (J-point) to sample ST amplitude
        pr_bounds_ms, qrs_bounds_ms, qtc_bounds_ms: plausibility bounds; a
            computed value outside these is treated as a failed delineation
            (NaN) rather than reported
    Returns
    ----------
        ClinicalIntervals for this beat (NaN for anything that couldn't be
        computed or that failed its plausibility check)
    """
    ci = ClinicalIntervals()
    qrs_onset = waves.q if not np.isnan(waves.q) else waves.r  # simplified: Q approximates QRS onset
 
    # PR interval: P onset -> QRS onset
    if not np.isnan(waves.p_onset):
        pr = (qrs_onset - waves.p_onset) / sampling_rate * 1000.0
        if pr_bounds_ms[0] <= pr <= pr_bounds_ms[1]:
            ci.pr_interval_ms = pr
 
    # QRS duration: Q -> S (simplified onset/offset)
    if not np.isnan(waves.q) and not np.isnan(waves.s):
        qrs_d = (waves.s - waves.q) / sampling_rate * 1000.0
        if qrs_bounds_ms[0] <= qrs_d <= qrs_bounds_ms[1]:
            ci.qrs_duration_ms = qrs_d
 
    # QT interval: QRS onset -> T offset, plus Bazett-corrected QTc.
    # QT itself is reported as computed (it naturally varies with heart rate);
    # only the rate-corrected QTc is bounds-checked, since that's the value
    # that's supposed to be roughly heart-rate independent.
    if not np.isnan(waves.t_offset):
        qt = (waves.t_offset - qrs_onset) / sampling_rate * 1000.0
        ci.qt_interval_ms = qt
        if rr_s and rr_s > 0 and np.isfinite(qt):
            qtc = qt / np.sqrt(rr_s)
            if qtc_bounds_ms[0] <= qtc <= qtc_bounds_ms[1]:
                ci.qtc_ms = qtc
 
    # ST elevation: amplitude at J-point + offset, relative to isoelectric baseline
    if not np.isnan(waves.s) and not np.isnan(waves.baseline):
        j_point = min(int(round(waves.s)) + int(st_offset_s * sampling_rate), len(ecg) - 1)
        ci.st_elevation_mv = float(ecg[j_point] - waves.baseline)
 
    # P / T amplitude relative to the same isoelectric baseline
    if not np.isnan(waves.p_peak) and not np.isnan(waves.baseline):
        ci.p_amplitude_mv = float(ecg[int(round(waves.p_peak))] - waves.baseline)
    if not np.isnan(waves.t_peak) and not np.isnan(waves.baseline):
        ci.t_amplitude_mv = float(ecg[int(round(waves.t_peak))] - waves.baseline)
 
    return ci
 
 
def compute_all_intervals(
    waves_list: list[BeatWaves],
    ecg: np.ndarray,
    sampling_rate: float,
    r_peaks: np.ndarray,
    st_offset_s: float = 0.06,
) -> list[ClinicalIntervals]:
    """Compute per-beat ClinicalIntervals for a full list of delineated beats.
    Each beat's QTc uses its *preceding* RR interval (standard convention,
    since that's the cycle length that determined repolarization time); the
    first beat falls back to the following RR interval if there is one.
    """
    intervals = []
    for i, waves in enumerate(waves_list):
        if i > 0:
            rr_s = (r_peaks[i] - r_peaks[i - 1]) / sampling_rate
        elif len(r_peaks) > 1:
            rr_s = (r_peaks[1] - r_peaks[0]) / sampling_rate
        else:
            rr_s = None
        intervals.append(compute_beat_intervals(waves, ecg, sampling_rate, rr_s, st_offset_s=st_offset_s))
    return intervals
 
 
def aggregate_intervals(intervals: list[ClinicalIntervals]) -> ClinicalIntervals:
    """Aggregate per-beat ClinicalIntervals into a single summary via the
    median across beats (robust to occasional mis-delineated beats).
    """
    agg = ClinicalIntervals()
    if not intervals:
        return agg
    for field_name in agg.__dataclass_fields__:
        values = np.array([getattr(ci, field_name) for ci in intervals], dtype=float)
        valid = values[np.isfinite(values)]
        setattr(agg, field_name, float(np.median(valid)) if len(valid) else np.nan)
    return agg
 

# --------------------------------------------------------------------------- #
# 6. FEATURE EXTRACTION
# --------------------------------------------------------------------------- #        

# --------------- Container for the ECG features --------------- #
"""
heart_rate_bpm: average heart rate in bpm, derived from mean RR interval.
mean_rr_ms: average interval between consecutive R-peaks in milliseconds.
sdnn_ms: standard deviation of normal-to-normal (NN) intervals, 
        a time-domain HRV metric measuring overall RR interval variability across the recording in ms. 
        Higher = more variability (generally associated with healthy autonomic function; 
        very low SDNN can indicate reduced heart rate variability.
rmssd_ms: Root Mean Square of Successive Differences, another time-domain HRV metric; 
        measures beat-to-beat variability specifically in ms.
pnn50: percentage of successive RR interval differences greater than 50 ms; 
        another short-term HRV indicator, expressed as a percentage (0–100).
lf_hf_ratio: Low Frequency / High Frequency power ratio;
        a frequency-domain HRV metric reflecting the balance between sympathetic 
        and parasympathetic (autonomic nervous system) influence on heart rate.
n_beats: total number of R-peaks detected in the recording (integer count).
rejected_beats: number of beats flagged and excluded by quality control.
pr_interval_ms, qrs_duration_ms, qt_interval_ms, qtc_ms, st_elevation_mv,
p_amplitude_mv, t_amplitude_mv: median-aggregated clinical wave-delineation
        metrics across all beats (see ClinicalIntervals / compute_beat_intervals
        docstrings above for definitions and caveats).
"""
@dataclass
class ECGFeatures:
    heart_rate_bpm: float=np.nan
    mean_rr_ms: float=np.nan
    sdnn_ms: float=np.nan
    rmssd_ms: float=np.nan
    pnn50: float=np.nan
    lf_hf_ratio: float=np.nan
    n_beats: int=0
    rejected_beats: int=0
    pr_interval_ms: float=np.nan
    qrs_duration_ms: float=np.nan
    qt_interval_ms: float=np.nan
    qtc_ms: float=np.nan
    st_elevation_mv: float=np.nan
    p_amplitude_mv: float=np.nan
    t_amplitude_mv: float=np.nan


# --------------- consecutive R-to-R intervals --------------- #
def rr_intervals_ms(
    r_peaks: np.ndarray, 
    sampling_rate: float
) -> np.ndarray:
    """Compute consecutive R-to-R intervals.
    Parameters
    ----------
        r_peaks: array of r-peak indices
        sampling: samples per second
    Returns
    ----------
        an array of RR intervals in milliseconds
    """
    return np.diff(r_peaks) / sampling_rate * 1000.0 # 1000: s -> ms

# --------------- time-domain HRV metrics --------------- #
def hrv_time_domain(rr_ms: np.ndarray) -> tuple[float]:
    """ Compute time-domain HRV metrics
        - sdnn: overall RR variability across the whole recording
        - rmssd: beat-to-beat variability
        - pnn50: percentage of large beat-to-beat jumps (>50ms)
    Parameters
    ----------
        rr_ms: consecutive rr intervals    
    Returns
    ----------
        three time-domain HRV metrics (sdnn, rmssd, pnn50)
    """
    # standard deviation of nn intervals:
    # the spread of rr intervals around their mean (in ms)
    # ddof=1: divide by N-1 instead of N (unbiased estimator)
    sdnn = np.std(rr_ms, ddof=1)
    # computes the difference between consecutive RR intervals themselves
    diffs = np.diff(rr_ms)
    # Root Mean Square of Successive Differences;
    # a widely used time-domain HRV metric 
    # reflecting short-term, parasympathetic (vagal) influence on heart rate
    rmssd = np.sqrt(np.mean(diffs ** 2))
    # percentage of successive RR interval differences whose absolute change exceeds 50 ms
    pnn50 = 100.0 * np.sum(np.abs(diffs) > 50) / len(diffs) if len(diffs) else np.nan

    return sdnn, rmssd, pnn50

# --------------- frequency-domain HRV metrics --------------- #
def hrv_freq_domain(rr_ms: np.ndarray, freq_interp: float=4.0) -> float:
    """Compute frequency-domain HRV metrics.
    Resample RR series onto a uniform time grid (required for PSD),
    then compute LF (0.04-0.15 Hz) / HF (0.15-0.4 Hz) power ratio via Welch.
    Parameters:
    ---------- 
        rr_ms: consecutive rr intervals in ms
        freq_interp: the interpolation sampling rate (in Hz) to resample the rr series 
    Returns
        LF/HF ratio
    ----------
    """
    # check: if fewer than 4 rr intervals -> not enough data to build a meaningful spectrum
    if len(rr_ms) < 4:
        return np.nan
    # compute cummulative sum of rr intervals in s -> irregular time axis
    t_rr = np.cumsum(rr_ms)/1000.0
    # build a new, evenly spaced time axis with spacing 1/freq_interp
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1/freq_interp)
    # linear interpolation
    rr_interp = np.interp(t_uniform, t_rr, rr_ms)
    rr_interp = rr_interp - np.mean(rr_interp) # remove the DC offset -> variation around mean value
    # Welch's method: estimates the power spectral density of rr_interp
    # returns array of fred bin (f, in Hz) and power at each corresponding freq bin (pxx)
    f, pxx = sp_signal.welch(rr_interp, 
                             fs=freq_interp, 
                             nperseg =min(256, len(rr_interp))) 
    # low-freq and high-freq boolean masks (standard bands used)
    lf_mask = (f>=0.04) & (f<0.15)
    hf_mask = (f>=0.15) & (f<0.4)
    # numerical integrates the power spectrum over the selected freq ranges using trapezoid rule
    lf_power = np.trapezoid(pxx[lf_mask], f[lf_mask]) if lf_mask.any() else np.nan
    hf_power = np.trapezoid(pxx[hf_mask], f[hf_mask]) if hf_mask.any() else np.nan
    # return lf/hf ratio
    return lf_power/hf_power if hf_power and np.isfinite(hf_power) else np.nan

def quality_control(
    rr_ms: np.ndarray, 
    min_bpm:float=30, 
    max_bpm: float=220
) -> np.ndarray:
    """Flag rr intervals outside phisiological bounds).
    (a simple sanity check on detected beats)
    Parameters
    ----------
        rr_ms: arrays of rr intervals in ms
        [min_bmp, max_bmp]: phisiologically plausible heart rate range 
    Returns
    ----------
        boolean array, element-wise: true for within range, false otherwise
    """
    # convert bpm bounds into the corresp. rr intervals bound in ms
    lo_ms, hi_ms = 60000 / max_bpm, 60000 / min_bpm
    # boolean flag array for within-range values
    valid = (rr_ms >= lo_ms) & (rr_ms <= hi_ms)
    return valid

# --------------- pipeline to extract ecg features --------------- #
def extract_features(
    sampling_rate: float, 
    r_peaks: np.ndarray,
    freq_interp: float=4.0, # for hrv_freq_domain
    min_bpm: float=30, max_bpm: float=220, # for quality control
    intervals: Optional[list["ClinicalIntervals"]] = None, # pre-computed per-beat clinical intervals
) -> ECGFeatures:
    """Extract ECG features.
    Parameters
    ----------
        ecg: ecg signal arrays
        sampling_rate: sampling rate in Hz
        r_peaks: array of detected r-peak sample indices  
        freq_interp: the interpolation sampling rate (in Hz) to resample the rr series 
        [min_bmp, max_bmp]: phisiologically plausible heart rate range
        intervals: per-beat ClinicalIntervals (from compute_all_intervals), if
            available -- their median is folded into the returned features
    Returns
    ----------
        ECGFeatures dataclass instance
    """
    # create an ECGFeatures object, setting n_beats to the total number of detected r-peaks
    features = ECGFeatures(n_beats=len(r_peaks))
    # check: need at least 2 r-peaks (np.diff need >=2 points)
    if len(r_peaks)<2:
        return features
    # computes the rr intervals series in ms from r-peak indices
    rr = rr_intervals_ms(r_peaks, sampling_rate)
    # run the quality control check
    valid = quality_control(rr, min_bpm=min_bpm, max_bpm=max_bpm)
    # count no. of invalid flag
    features.rejected_beats = int(np.sum(~valid))
    # keep only intervals passing the quality control check
    rr_clean = rr[valid]
    # check if there are enough valid intervals left to compute meaningful statistics
    if len(rr_clean) < 2:
        return features
    # Derive ECG features
    features.mean_rr_ms = float(np.mean(rr_clean))
    features.heart_rate_bpm = float(60000.0 / features.mean_rr_ms)
    features.sdnn_ms, features.rmssd_ms, features.pnn50 = map(float, hrv_time_domain(rr_clean))
    features.lf_hf_ratio = float(hrv_freq_domain(rr_clean, freq_interp = freq_interp))

    # fold in median-aggregated clinical wave-delineation metrics, 
    # if provided
    if intervals:
        agg = aggregate_intervals(intervals)
        features.pr_interval_ms = agg.pr_interval_ms
        features.qrs_duration_ms = agg.qrs_duration_ms
        features.qt_interval_ms = agg.qt_interval_ms
        features.qtc_ms = agg.qtc_ms
        features.st_elevation_mv = agg.st_elevation_mv
        features.p_amplitude_mv = agg.p_amplitude_mv
        features.t_amplitude_mv = agg.t_amplitude_mv

    return features

# --------------------------------------------------------------------------- #
# 6. FULL PIPELINE WRAPPER 
# --------------------------------------------------------------------------- #

class ECGPipeline:
    def __init__(self, 
            sampling_rate,  
            low_hz: float = 0.5, high_hz: float = 40.0, order: int = 3, # for bandpass filter
            freq_hz: float = 50.0, quality_factor: float = 30.0, # for notch filter
            tuned_win_size: float=0.15, tuned_min_distance: float=0.2, tuned_search_radius: float=0.4, # for qrs detector
            pre: float=0.25, post: float=0.4, # for beat segmentation
            freq_interp: float=4.0, # for hrv_freq_domain
            min_bpm: float=30, max_bpm: float=220, # for quality control
            # --- wave delineation / clinical interval params ---
            q_search_s: float=0.05, s_search_s: float=0.06,
            p_search_s: tuple[float, float]=(0.28, 0.10),
            t_search_s: tuple[float, float]=(0.15, 0.40),
            onset_offset_frac: float=0.5,
            st_offset_s: float=0.06,
    ):
        self.sampling_rate = sampling_rate
        self.low_hz = low_hz 
        self.high_hz = high_hz
        self.order = order
        self.freq_hz = freq_hz
        self.quality_factor = quality_factor
        self.pre = pre
        self.post = post
        self.freq_interp = freq_interp
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.tuned_win_size = tuned_win_size
        self.tuned_min_distance = tuned_min_distance
        self.tuned_search_radius = tuned_search_radius
        self.q_search_s = q_search_s
        self.s_search_s = s_search_s
        self.p_search_s = p_search_s
        self.t_search_s = t_search_s
        self.onset_offset_frac = onset_offset_frac
        self.st_offset_s = st_offset_s

    # --- processing pipeline run
    def run(self, raw_ecg):
        clean_ecg = preprocess_pipeline(
            ecg=raw_ecg, 
            sampling_rate=self.sampling_rate,
            low_hz=self.low_hz,
            high_hz=self.high_hz,
            order=self.order ,
            freq_hz=self.freq_hz,
            quality_factor = self.quality_factor
        )
        r_peaks = detect_qrs( 
            ecg=clean_ecg, 
            sampling_rate=self.sampling_rate,
            tuned_win_size=self.tuned_win_size,
            tuned_min_distance=self.tuned_min_distance,
            tuned_search_radius=self.tuned_search_radius
        )
        beats = segment_beats(
            ecg=clean_ecg,
            r_peaks=r_peaks,
            sampling_rate=self.sampling_rate,
            pre=self.pre,
            post=self.post
        )

         # --- P/Q/S/T delineation + clinical intervals (needs >=1 R-peak) ---
        waves_list: list[BeatWaves] = []
        intervals: list[ClinicalIntervals] = []
        if len(r_peaks) >= 1:
            waves_list = delineate_all_beats(
                ecg=clean_ecg, r_peaks=r_peaks, sampling_rate=self.sampling_rate,
                q_search_s=self.q_search_s, s_search_s=self.s_search_s,
                p_search_s=self.p_search_s, t_search_s=self.t_search_s,
                onset_offset_frac=self.onset_offset_frac,
            )
            intervals = compute_all_intervals(
                waves_list=waves_list, ecg=clean_ecg, sampling_rate=self.sampling_rate,
                r_peaks=r_peaks, st_offset_s=self.st_offset_s,
            )


        features = extract_features(
            sampling_rate=self.sampling_rate,
            r_peaks = r_peaks,
            freq_interp=self.freq_interp,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            intervals=intervals
        )

        return {
            "filtered_ecg": clean_ecg,
            "r_peaks": r_peaks,
            "beats": beats,
            "features": features,
            "waves": waves_list,    # list[BeatWaves], per-beat P/Q/S/T locations
            "intervals": intervals, # list[ClinicalIntervals], per-beat clinical metrics
        }



if __name__ == "__main__": 
    datapath = "data/Training_WFDB"
    i = 1
    record = f"{datapath}/A{i:04d}"
    df, sampling_rate = load_ecg_signal(record)
    raw_ecg = np.array(df["II"].values)
    t = np.array(df["time_s"].values)
    print(len(df))

    # tuning parameters
    tuned_win_size = 0.25
    tuned_min_distance = 0.4 
    tuned_search_radius = 0.6

    pipeline = ECGPipeline(
        sampling_rate=sampling_rate, 
        tuned_win_size=tuned_win_size,
        tuned_min_distance=tuned_min_distance,
        tuned_search_radius=tuned_search_radius,
    )
    result = pipeline.run(raw_ecg=raw_ecg)

    f = result["features"]
    print(f"Detected {f.n_beats} beats ({f.rejected_beats} rejected by QC)")
    print(f"Heart rate     : {f.heart_rate_bpm:.1f} bpm")
    print(f"Mean RR        : {f.mean_rr_ms:.1f} ms")
    print(f"SDNN           : {f.sdnn_ms:.1f} ms")
    print(f"RMSSD          : {f.rmssd_ms:.1f} ms")
    print(f"pNN50          : {f.pnn50:.1f} %")
    print(f"LF/HF ratio    : {f.lf_hf_ratio:.2f}")
 
    plt.figure(figsize=(12, 4))
    #plt.plot(t, raw_ecg, label="Raw ECG")
    plt.plot(t, result["filtered_ecg"], label="Filtered ECG")
    plt.plot(t[result["r_peaks"]], result["filtered_ecg"][result["r_peaks"]],
                "ro", markersize=8, label="R-peaks")
    plt.xlabel("Time (s)")
    #plt.xlim(0,15)
    plt.minorticks_on()
    plt.legend()
    plt.title("ECG Pipeline Output")
    plt.tight_layout()
    #plt.savefig("/mnt/user-data/outputs/ecg_pipeline_demo.png", dpi=120)
    #print("\nSaved plot to ecg_pipeline_demo.png")
     
    plt.show()

