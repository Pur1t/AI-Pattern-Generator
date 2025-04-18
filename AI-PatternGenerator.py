# explicitly set the paths (via AudioSegment.converter and os.environ["FFPROBE_PATH"]) and added your ffmpeg_bin folder to PATH, your executable is using your bundled ffmpeg/ffprobe correctly.
# so we can suppress this RuntimeError
import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", module="pydub")

#!/usr/bin/env python3
import numpy as np
import pyfftw
from scipy.signal import find_peaks, firwin, lfilter
from pydub import AudioSegment
import argparse
import sys, os, random, concurrent.futures, shutil
from itertools import combinations
from mutagen import File as MutagenFile

##############################################
# FFmpeg Path Setup
##############################################
def setup_ffmpeg_paths():
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    ffmpeg_path = os.path.join(base_path, "ffmpeg_bin", "ffmpeg")
    ffprobe_path = os.path.join(base_path, "ffmpeg_bin", "ffprobe")
    print("FFmpeg path:", ffmpeg_path)
    print("FFprobe path:", ffprobe_path)
    if not os.path.exists(ffmpeg_path):
        print("ERROR: ffmpeg not found!")
    if not os.path.exists(ffprobe_path):
        print("ERROR: ffprobe not found!")
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
    os.environ["FFPROBE_PATH"] = ffprobe_path
    os.environ["PATH"] = os.path.join(base_path, "ffmpeg_bin") + os.pathsep + os.environ.get("PATH", "")
    print("PATH:", os.environ["PATH"])

##############################################
# Metadata Retrieval using Mutagen
##############################################
def retrieve_audio_metadata(audio_path):
    metadata = {}
    if not os.path.exists(audio_path):
        print("Audio file does not exist!")
        return metadata
    audio = MutagenFile(audio_path)
    if audio is None:
        print("Could not read audio metadata!")
        return metadata
    if 'TIT2' in audio:
        try:
            metadata['song_title'] = audio['TIT2'].text[0]
        except Exception as e:
            print("Error retrieving song title:", e)
    if 'TPE1' in audio:
        try:
            metadata['artist'] = audio['TPE1'].text[0]
        except Exception as e:
            print("Error retrieving artist:", e)
    return metadata

##############################################
# Adaptive Thresholding, Smoothing, and Section Density Functions
##############################################
def moving_average(arr, window_size=20):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='same')

def smooth_event_times(events, grid_interval, tolerance=5):
    smoothed = []
    for t, pattern in events:
        remainder = t % grid_interval
        if remainder < tolerance:
            t = t - remainder
        elif remainder > grid_interval - tolerance:
            t = t + (grid_interval - remainder)
        smoothed.append((t, pattern))
    return sorted(smoothed, key=lambda x: x[0])

def get_section_density(filtered_flux, section_duration_ms, flux_sample_duration):
    section_length = int(section_duration_ms / flux_sample_duration)
    densities = []
    for i in range(0, len(filtered_flux), section_length):
        section = filtered_flux[i:i+section_length]
        densities.append(np.mean(section))
    densities = np.array(densities)
    if densities.max() - densities.min() < 1e-6:
        return np.ones_like(densities)
    norm = (densities - densities.min()) / (densities.max() - densities.min())
    return 0.5 + norm  # maps to [0.5, 1.5]

##############################################
# BPM & Offset Detection Functions
##############################################
def segment_audio(audio, frame_size=1024, hop_size=128):
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())
    frames = np.lib.stride_tricks.sliding_window_view(samples, window_shape=frame_size)[::hop_size]
    return frames, samples

def apply_hamming_window(frames):
    return frames * np.hamming(frames.shape[1])

def compute_magnitude_spectrum(frames):
    fft_size = frames.shape[1] // 2 + 1
    magnitude_spectrum = np.zeros((frames.shape[0], fft_size))
    fft_obj = pyfftw.builders.rfft(frames[0], threads=4)
    for i, frame in enumerate(frames):
        fft_result = fft_obj(frame)
        magnitude_spectrum[i] = np.abs(fft_result)
    return np.log1p(1000.0 * magnitude_spectrum)

def compute_spectral_flux(magnitude_spectrum):
    flux = np.maximum(0, np.diff(magnitude_spectrum, axis=0))
    return np.sum(flux, axis=1)

def apply_low_pass_filter(signal, sample_rate=44100//128, cutoff_freq=7, filter_order=14):
    nyquist = sample_rate / 2.0
    cutoff_norm = cutoff_freq / nyquist
    fir_coeff = firwin(filter_order + 1, cutoff_norm, window="hamming")
    return lfilter(fir_coeff, 1.0, signal)

def generalized_autocorrelation(oss_signal, c=0.5):
    length = len(oss_signal)
    padded_length = 2 * length
    padded = np.zeros(padded_length)
    padded[:length] = oss_signal
    fft_obj = pyfftw.builders.fft(padded, threads=4)
    ifft_obj = pyfftw.builders.ifft(np.empty_like(padded), threads=4)
    dft_result = fft_obj()
    mag = np.abs(dft_result) ** c
    ac = np.real(ifft_obj(mag))
    return ac[:length]

def enhance_harmonics(ac):
    enhanced = ac.copy()
    length = len(ac)
    for factor in [2, 3, 4]:
        stretched = np.zeros_like(ac)
        min_len = min(length // factor, len(ac[::factor]))
        stretched[:min_len] = ac[::factor][:min_len]
        enhanced += stretched * (1 / factor)
    return enhanced

def find_peak_candidates(ac, fs_oss, min_bpm=60, max_bpm=200):
    min_lag = int((60 / max_bpm) * fs_oss)
    max_lag = int((60 / min_bpm) * fs_oss)
    segment = ac[min_lag:max_lag]
    peaks, _ = find_peaks(segment, height=0.01 * np.max(ac))
    bpm_candidates = [60 / ((p + min_lag) / fs_oss) for p in peaks]
    return bpm_candidates

def generate_pulse_train(length, period, phase=0):
    pulse = np.zeros(length)
    for i in range(phase, length, period):
        pulse[i] = 1.0
    return pulse

def evaluate_pulse_train(oss, period, fs_oss, phase_candidates=None):
    if phase_candidates is None:
        phase_candidates = np.arange(0, period)
    best_score = -np.inf
    best_phase = 0
    for phase in phase_candidates:
        pulse = generate_pulse_train(len(oss), period, phase)
        score = np.correlate(oss, pulse)[0]
        if score > best_score:
            best_score = score
            best_phase = phase
    return best_score, best_phase

def evaluate_candidates_with_pulse_train(oss, bpm_candidates, fs_oss):
    candidate_scores = []
    for bpm in bpm_candidates:
        period = int((60 * fs_oss) / bpm)
        score, phase = evaluate_pulse_train(oss, period, fs_oss)
        candidate_scores.append((bpm, score, phase))
    candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
    return candidate_scores

def cluster_window_bpms(bpm_estimates, bin_width=5):
    bpm_estimates = np.array(bpm_estimates)
    if bpm_estimates.size == 0:
        return None
    if bpm_estimates.size == 1:
        return bpm_estimates[0]
    min_bpm = np.min(bpm_estimates)
    max_bpm = np.max(bpm_estimates)
    bins = np.arange(min_bpm, max_bpm + bin_width, bin_width)
    hist, bin_edges = np.histogram(bpm_estimates, bins=bins)
    max_bin_index = np.argmax(hist)
    lower_bound = bin_edges[max_bin_index]
    upper_bound = bin_edges[max_bin_index + 1]
    cluster = bpm_estimates[(bpm_estimates >= lower_bound) & (bpm_estimates < upper_bound)]
    if cluster.size == 0:
        return np.median(bpm_estimates)
    return np.median(cluster)

def accumulate_bpm_estimates(filtered_flux, fs_oss, window_length, window_hop):
    bpm_estimates = []
    for start in range(0, len(filtered_flux) - window_length + 1, window_hop):
        window_oss = filtered_flux[start:start+window_length]
        ac = generalized_autocorrelation(window_oss)
        enhanced = enhance_harmonics(ac)
        bpm_candidates = find_peak_candidates(enhanced, fs_oss, min_bpm=60, max_bpm=200)
        candidate_scores = evaluate_candidates_with_pulse_train(window_oss, bpm_candidates, fs_oss)
        if candidate_scores:
            best_bpm, score, phase = candidate_scores[0]
            bpm_estimates.append(best_bpm)
    return bpm_estimates

def accumulate_and_overall_estimate(filtered_flux, fs_oss, window_length, window_hop):
    bpm_estimates = accumulate_bpm_estimates(filtered_flux, fs_oss, window_length, window_hop)
    overall_bpm = cluster_window_bpms(bpm_estimates, bin_width=5)
    return overall_bpm, bpm_estimates

def select_final_bpm(overall_bpm, oss, fs_oss):
    candidate1 = overall_bpm
    period1 = int((60 * fs_oss) / candidate1)
    score1, phase1 = evaluate_pulse_train(oss, period1, fs_oss)
    
    candidate2 = overall_bpm / 2
    period2 = int((60 * fs_oss) / candidate2)
    score2, phase2 = evaluate_pulse_train(oss, period2, fs_oss)
    
    candidate3 = overall_bpm * 2
    period3 = int((60 * fs_oss) / candidate3)
    score3, phase3 = evaluate_pulse_train(oss, period3, fs_oss)
    
    scores = [(score1, candidate1, phase1), (score2, candidate2, phase2), (score3, candidate3, phase3)]
    best_score, best_bpm, best_phase = max(scores, key=lambda x: x[0])
    return best_bpm, best_score, best_phase

def correct_bpm_candidate(bpm):
    if bpm < 100:
        return bpm * 2
    elif bpm > 160:
        return bpm / 2
    else:
        return bpm

def process_audio(song_path):
    audio = AudioSegment.from_file(song_path)
    frames, samples = segment_audio(audio)
    windowed_frames = apply_hamming_window(frames)
    magnitude_spectrum = compute_magnitude_spectrum(windowed_frames)
    spectral_flux = compute_spectral_flux(magnitude_spectrum)
    filtered_flux = apply_low_pass_filter(spectral_flux)
    
    fs_oss = 44100 / 128
    window_length = 2048
    window_hop = 1024
    overall_bpm, window_bpms = accumulate_and_overall_estimate(filtered_flux, fs_oss, window_length, window_hop)
    print("Window BPM estimates:", window_bpms)
    print("Overall BPM Estimate (via clustering):", overall_bpm)
    
    candidate_ac = generalized_autocorrelation(filtered_flux)
    enhanced_ac = enhance_harmonics(candidate_ac)
    bpm_candidates = find_peak_candidates(enhanced_ac, fs_oss, min_bpm=60, max_bpm=200)
    candidate_scores = evaluate_candidates_with_pulse_train(filtered_flux, bpm_candidates, fs_oss)
    if candidate_scores:
        best_bpm, best_score, best_phase = candidate_scores[0]
        print(f"Final BPM (from entire OSS): {best_bpm:.2f} BPM (score: {best_score:.2f}, phase: {best_phase})")
    else:
        best_bpm, best_phase = overall_bpm, 0
        print("No BPM candidate survived pulse train evaluation on entire OSS.")
    
    final_bpm = overall_bpm
    final_bpm, final_score, final_phase = select_final_bpm(final_bpm, filtered_flux, fs_oss)
    final_bpm = correct_bpm_candidate(final_bpm)
    print(f"Corrected Final BPM Estimate: {final_bpm:.2f} BPM")
    
    offset_ms = detect_offset(filtered_flux, hop_size=128, sample_rate=44100, threshold_ratio=0.3)
    return final_bpm, offset_ms, samples, magnitude_spectrum, spectral_flux, filtered_flux, enhanced_ac, best_phase, fs_oss

def detect_offset(filtered_flux, hop_size=128, sample_rate=44100, threshold_ratio=0.3):
    threshold = threshold_ratio * np.max(filtered_flux)
    for i, value in enumerate(filtered_flux):
        if value >= threshold:
            offset_sec = i * hop_size / sample_rate
            return offset_sec * 1000
    return 0.0

##############################################
# AI Pattern Generation (Note Placement)
##############################################
def get_two_note_chord_patterns():
    patterns = []
    for combo in combinations(range(4), 2):
        pattern = ["_" for _ in range(4)]
        for pos in combo:
            pattern[pos] = "O"
        patterns.append(" ".join(pattern))
    return patterns

def extract_lane(pattern):
    """Return the lane index for a single-note pattern (assumes only one 'O')."""
    tokens = pattern.split()
    for i, token in enumerate(tokens):
        if token == "O":
            return i
    return None

def enforce_consecutive_lane_constraint(events, max_consecutive=2):
    """
    Prevent more than max_consecutive single-note events in the same lane.
    If more than max_consecutive occur, force a lane change.
    """
    filtered = []
    for event in events:
        t, pattern = event
        # Check if it's a single-note event.
        if pattern.count("O") == 1:
            current_lane = extract_lane(pattern)
            # Count consecutive single-note events in the same lane in filtered list.
            consecutive = 0
            for prev_event in reversed(filtered):
                pt, ppattern = prev_event
                # Only consider if it's within a reasonable time gap (for consecutive notes)
                if t - pt > 500:  # 500ms limit for what counts as consecutive
                    break
                if ppattern.count("O") == 1 and extract_lane(ppattern) == current_lane:
                    consecutive += 1
                else:
                    break
            if consecutive >= max_consecutive:
                # Force change: choose a different lane
                available_lanes = [i for i in range(4) if i != current_lane]
                new_lane = random.choice(available_lanes)
                tokens = ["_" for _ in range(4)]
                tokens[new_lane] = "O"
                pattern = " ".join(tokens)
        filtered.append((t, pattern))
    return filtered

def enforce_minimum_gap(events, min_gap_ms=70):
    """
    Remove events that occur too close together (within min_gap_ms).
    """
    if not events:
        return events
    filtered = [events[0]]
    last_time = events[0][0]
    for t, pattern in events[1:]:
        if t - last_time >= min_gap_ms:
            filtered.append((t, pattern))
            last_time = t
        # Otherwise, skip this event.
    return filtered

def generate_pattern(final_bpm, offset_ms, samples, filtered_flux, snap_division=4, 
                     note_threshold=0.15, chord_threshold=0.6):
    """
    Generate hit objects based on a fixed beat grid and intensity analysis.
    
    - Beat grid: beat_interval = 60000 / BPM.
    - Only the first subdivision (m=0) of each beat is eligible for chord generation.
      If flux at that grid time >= chord_threshold * max(filtered_flux), generate a two-note chord.
      Otherwise, generate a single note.
    - For subdivisions m > 0, generate single note events if flux >= note_threshold.
    - No interpolation is performed.
    - Beats occurring within the first second (1000ms) after the offset are skipped.
    - Additionally, thresholds are adapted using a density factor computed over sections (16 beats per section).
    
    Returns a sorted list of (timestamp in ms, lane pattern) tuples.
    """
    beat_interval = 60000 / final_bpm  # in ms
    max_flux = np.max(filtered_flux)
    song_duration_ms = len(samples) / 44100 * 1000
    events = []
    
    flux_sample_duration = (128 / 44100) * 1000  # ms per flux sample
    two_note_patterns = get_two_note_chord_patterns()
    
    # Define section duration based on 16 beats.
    section_duration_ms = (60 / final_bpm) * 16 * 1000
    density_factors = get_section_density(filtered_flux, section_duration_ms, flux_sample_duration)
    
    n = 0
    while offset_ms + n * beat_interval <= song_duration_ms:
        beat_time = offset_ms + n * beat_interval
        # Skip the first second after offset to avoid initial flux artifacts.
        if beat_time < offset_ms + 1000:
            n += 1
            continue

        section_idx = int((beat_time - offset_ms) // section_duration_ms)
        density_factor = density_factors[section_idx] if section_idx < len(density_factors) else 1.0
        local_note_threshold = note_threshold * density_factor
        local_chord_threshold = chord_threshold * density_factor
        
        for m in range(snap_division):
            t = beat_time + m * (beat_interval / snap_division)
            index = int(round(t / flux_sample_duration))
            if index >= len(filtered_flux):
                continue
            intensity = filtered_flux[index] / max_flux
            if intensity < local_note_threshold:
                continue
            if m == 0 and intensity >= local_chord_threshold:
                pattern = random.choice(two_note_patterns)
            else:
                lane = random.choice(range(4))
                tokens = ["_" for _ in range(4)]
                tokens[lane] = "O"
                pattern = " ".join(tokens)
            events.append((int(round(t)), pattern))
        n += 1
    events.sort(key=lambda x: x[0])
    events = smooth_event_times(events, grid_interval=beat_interval)
    return events

##############################################
# Output File Assembly
##############################################
def write_chart_file(filename, metadata, timing_points, hit_objects):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("[Metadata]\n")
        for key in ["Artist", "Song", "Diff", "Chart Creator", "Source"]:
            f.write(f"{key}: {metadata.get(key, '')}\n")
        f.write("\n[TimingPoints]\n")
        for tp in timing_points:
            f.write(f"{tp[0]},{tp[1]:.6f}\n")
        f.write("\n[HitObjects]\n")
        for ho in hit_objects:
            f.write(f"{ho[0]},{ho[1]}\n")
    print(f"Chart file written to: {filename}")

##############################################
# Command-Line Interface and Main Execution
##############################################
def main():
    parser = argparse.ArgumentParser(description="AI Pattern Generator for VSRG Chart Editor")
    parser.add_argument("-c", "--config", nargs="*", default=["", "", "", "", "2"],
                        help="Configuration: Song Title, Diffname, Chart Creator, Source, Snap. Defaults if omitted.")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="Absolute path to the audio file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Absolute output path where the output folder will be created")
    args = parser.parse_args()
    
    if args.config and len(args.config) >= 5:
        song_title, diffname, chart_creator, source, snap_val = args.config
    else:
        song_title, diffname, chart_creator, source, snap_val = "", "", "", "", "2"
    
    try:
        snap_division = int(snap_val)
    except ValueError:
        snap_division = 2
    
    meta_from_audio = retrieve_audio_metadata(args.path)
    if meta_from_audio.get("song_title"):
        song_title = meta_from_audio["song_title"]
    if meta_from_audio.get("artist"):
        artist = meta_from_audio["artist"]
        if not source:
            source = artist
    else:
        artist = "Unknown"
    if not song_title:
        song_title = "No Title"
    
    print("Final Configuration:")
    print("Song Title:", song_title)
    print("Diffname:", diffname)
    print("Chart Creator:", chart_creator)
    print("Source:", source)
    print("Snap Value:", snap_division)
    print("Audio File Path:", args.path)
    print("Output Path:", args.output)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_audio, args.path)
        (final_bpm, offset_ms, samples, mag_spec, spec_flux, 
         filt_flux, enhanced_ac, best_phase, fs_oss) = future.result()
    
    print(f"\nFinal BPM: {final_bpm:.2f} BPM")
    print(f"Estimated Offset: {offset_ms:.2f} ms")
    
    beat_length = 60000 / final_bpm
    timing_points = [(int(round(offset_ms)), beat_length)]
    
    hit_objects = generate_pattern(final_bpm, offset_ms, samples, filt_flux, 
                                   snap_division=snap_division, 
                                   note_threshold=0.15, chord_threshold=0.6)
    
    base_folder = f"{artist} - {song_title}"
    output_folder = os.path.join(args.output, base_folder)
    counter = 1
    while os.path.exists(output_folder):
        output_folder = os.path.join(args.output, f"{base_folder} ({counter})")
        counter += 1
    os.makedirs(output_folder, exist_ok=True)
    
    dest_audio_path = os.path.join(output_folder, "audio.mp3")
    shutil.copy(args.path, dest_audio_path)
    
    chart_filename = os.path.join(output_folder, "chart.txt")
    metadata = {
        "Artist": artist,
        "Song": song_title,
        "Diff": diffname,
        "Chart Creator": chart_creator,
        "Source": source
    }
    write_chart_file(chart_filename, metadata, timing_points, hit_objects)
    
    score_filename = os.path.join(output_folder, "score.txt")
    with open(score_filename, "w", encoding="utf-8") as f:
        f.write("[Score]\n")
        f.write("Score: 0\n")
        f.write("Accuracy: 00.00\n")
        f.write("Rank: X\n")
    print(f"Score file written to: {score_filename}")

if __name__ == "__main__":
    setup_ffmpeg_paths()
    main()