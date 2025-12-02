"""
Step 6: Voice Activity Detection (VAD)
=======================================
Learn to detect speech vs silence/noise:
- Energy-based VAD: Simple threshold on signal energy
- Zero-Crossing Rate (ZCR): Distinguish voiced/unvoiced
- Spectral features: More robust to noise
- WebRTC VAD: Production-ready algorithm
- Deep learning VAD: State-of-the-art

Key Concepts:
- VAD critical for:
  * Reducing ASR computational load
  * Detecting turn-taking in conversations
  * Triggering processing only when needed
  * Barge-in detection (user interrupting agent)
- Frame-level detection (typically 10-30ms frames)
- Balance between false positives and false negatives
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
import webrtcvad
import struct

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_speech_with_silence(duration=5.0, sample_rate=16000):
    """
    Generate speech-like signal with silent sections.

    Returns:
        signal, speech_labels, sample_rate
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    signal = np.zeros(num_samples)
    speech_labels = np.zeros(num_samples, dtype=bool)

    # Define speech and silence segments (alternating)
    segments = [
        (0.0, 0.5, False),    # Silence
        (0.5, 1.5, True),     # Speech
        (1.5, 2.0, False),    # Silence
        (2.0, 3.5, True),     # Speech
        (3.5, 4.0, False),    # Silence
        (4.0, 5.0, True),     # Speech
    ]

    for start_t, end_t, is_speech in segments:
        start_idx = int(start_t * sample_rate)
        end_idx = int(end_t * sample_rate)

        if is_speech:
            # Generate speech-like signal
            seg_len = end_idx - start_idx
            seg_t = np.arange(seg_len) / sample_rate

            # Varying pitch
            f0 = 120 + 40 * np.sin(2 * np.pi * 2 * seg_t)
            speech_sig = 0.5 * np.sin(2 * np.pi * f0 * seg_t)

            # Add formants
            for fc in [700, 1200, 2500]:
                sos = scipy_signal.butter(4, [fc-100, fc+100], btype='band',
                                         fs=sample_rate, output='sos')
                speech_sig += 0.2 * scipy_signal.sosfilt(sos, speech_sig)

            # Add some noise
            speech_sig += 0.05 * np.random.randn(seg_len)

            signal[start_idx:end_idx] = speech_sig
            speech_labels[start_idx:end_idx] = True
        else:
            # Background noise only
            noise_level = 0.02
            signal[start_idx:end_idx] = noise_level * np.random.randn(end_idx - start_idx)
            speech_labels[start_idx:end_idx] = False

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    return signal, speech_labels, sample_rate


def energy_vad(signal, sample_rate, frame_ms=20, threshold_db=-40):
    """
    Simple energy-based VAD.

    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        frame_ms: Frame size in ms
        threshold_db: Energy threshold in dB

    Returns:
        voice_activity: Boolean array (True = speech, False = silence)
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(signal) // frame_size

    voice_activity = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        if end > len(signal):
            break

        frame = signal[start:end]

        # Calculate frame energy in dB
        energy = np.sum(frame ** 2) / len(frame)
        energy_db = 10 * np.log10(energy + 1e-10)

        # Threshold decision
        voice_activity[i] = energy_db > threshold_db

    return voice_activity, frame_size


def zero_crossing_rate(frame):
    """Calculate Zero-Crossing Rate for a frame."""
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))


def zcr_vad(signal, sample_rate, frame_ms=20, zcr_threshold=0.3):
    """
    Zero-Crossing Rate based VAD.

    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        frame_ms: Frame size in ms
        zcr_threshold: ZCR threshold

    Returns:
        voice_activity: Boolean array
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(signal) // frame_size

    voice_activity = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        if end > len(signal):
            break

        frame = signal[start:end]

        # Calculate ZCR
        zcr = zero_crossing_rate(frame)

        # Threshold decision (low ZCR = voiced speech)
        voice_activity[i] = zcr < zcr_threshold

    return voice_activity, frame_size


def spectral_vad(signal, sample_rate, frame_ms=20, freq_threshold=0.5):
    """
    Spectral energy based VAD.

    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        frame_ms: Frame size in ms
        freq_threshold: Threshold for spectral energy ratio

    Returns:
        voice_activity: Boolean array
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(signal) // frame_size

    voice_activity = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        if end > len(signal):
            break

        frame = signal[start:end]

        # Compute FFT
        fft = np.fft.rfft(frame * np.hanning(len(frame)))
        power = np.abs(fft) ** 2

        # Calculate spectral centroid
        freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)

        # Energy in speech band (300-3400 Hz) vs total
        speech_band = (freqs >= 300) & (freqs <= 3400)
        speech_energy = np.sum(power[speech_band])
        total_energy = np.sum(power)

        ratio = speech_energy / (total_energy + 1e-10)

        voice_activity[i] = ratio > freq_threshold

    return voice_activity, frame_size


def webrtc_vad_detect(signal, sample_rate, frame_ms=30, aggressiveness=3):
    """
    WebRTC VAD (production-ready).

    Args:
        signal: Audio signal (must be 16kHz, 16-bit PCM)
        sample_rate: Sample rate (8000, 16000, 32000, or 48000 Hz)
        frame_ms: Frame duration (10, 20, or 30 ms)
        aggressiveness: 0-3 (0=least aggressive, 3=most aggressive)

    Returns:
        voice_activity: Boolean array
    """
    # WebRTC VAD only supports specific configurations
    if sample_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError("WebRTC VAD requires sample rate of 8000, 16000, 32000, or 48000 Hz")

    if frame_ms not in [10, 20, 30]:
        raise ValueError("WebRTC VAD requires frame duration of 10, 20, or 30 ms")

    vad = webrtcvad.Vad(aggressiveness)

    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(signal) // frame_size

    voice_activity = np.zeros(num_frames, dtype=bool)

    # Convert to 16-bit PCM
    signal_int16 = np.int16(signal * 32767)

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        if end > len(signal):
            break

        frame = signal_int16[start:end]

        # Convert to bytes
        frame_bytes = struct.pack(f'{len(frame)}h', *frame)

        # VAD detection
        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
            voice_activity[i] = is_speech
        except Exception as e:
            voice_activity[i] = False

    return voice_activity, frame_size


def smooth_vad_decisions(voice_activity, min_speech_frames=3, min_silence_frames=3):
    """
    Smooth VAD decisions to reduce false triggers.

    Args:
        voice_activity: Raw VAD decisions
        min_speech_frames: Minimum consecutive speech frames
        min_silence_frames: Minimum consecutive silence frames

    Returns:
        Smoothed VAD decisions
    """
    smoothed = voice_activity.copy()

    # Remove short speech bursts
    speech_regions = []
    in_speech = False
    start = 0

    for i, is_speech in enumerate(smoothed):
        if is_speech and not in_speech:
            start = i
            in_speech = True
        elif not is_speech and in_speech:
            if i - start < min_speech_frames:
                smoothed[start:i] = False
            in_speech = False

    # Remove short silence gaps
    silence_regions = []
    in_silence = False
    start = 0

    for i, is_speech in enumerate(smoothed):
        if not is_speech and not in_silence:
            start = i
            in_silence = True
        elif is_speech and in_silence:
            if i - start < min_silence_frames:
                smoothed[start:i] = True
            in_silence = False

    return smoothed


def calculate_vad_metrics(predicted, ground_truth):
    """Calculate VAD performance metrics."""
    tp = np.sum(predicted & ground_truth)
    tn = np.sum(~predicted & ~ground_truth)
    fp = np.sum(predicted & ~ground_truth)
    fn = np.sum(~predicted & ground_truth)

    accuracy = (tp + tn) / len(predicted)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def demonstrate_vad_comparison():
    """Compare different VAD algorithms."""
    print("\n" + "="*60)
    print("VOICE ACTIVITY DETECTION (VAD) COMPARISON")
    print("="*60)

    # Generate test signal
    signal, ground_truth_samples, sr = generate_speech_with_silence(duration=5.0, sample_rate=16000)

    frame_ms = 20

    print(f"\nTest Signal:")
    print(f"  Duration: {len(signal)/sr:.1f} seconds")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Frame Size: {frame_ms} ms")

    # Apply different VAD methods
    energy_va, frame_size = energy_vad(signal, sr, frame_ms, threshold_db=-40)
    zcr_va, _ = zcr_vad(signal, sr, frame_ms, zcr_threshold=0.3)
    spectral_va, _ = spectral_vad(signal, sr, frame_ms, freq_threshold=0.5)
    webrtc_va, _ = webrtc_vad_detect(signal, sr, frame_ms=30, aggressiveness=3)

    # Downsample ground truth to frame level
    num_frames = len(energy_va)
    ground_truth = np.zeros(num_frames, dtype=bool)
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        # Frame is speech if >50% of samples are speech
        ground_truth[i] = np.mean(ground_truth_samples[start:end]) > 0.5

    # Calculate metrics
    print("\n" + "="*60)
    print("VAD PERFORMANCE METRICS")
    print("="*60)

    methods = {
        'Energy-based': energy_va,
        'Zero-Crossing Rate': zcr_va,
        'Spectral': spectral_va,
        'WebRTC': webrtc_va[:len(ground_truth)]  # WebRTC uses 30ms frames
    }

    print(f"\n{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1'}")
    print("-" * 70)

    for name, va in methods.items():
        if len(va) != len(ground_truth):
            # Adjust for different frame sizes
            va = va[:len(ground_truth)]

        metrics = calculate_vad_metrics(va, ground_truth)
        print(f"{name:<20} {metrics['accuracy']:.3f}       {metrics['precision']:.3f}       "
              f"{metrics['recall']:.3f}       {metrics['f1']:.3f}")

    # Visualize
    fig, axes = plt.subplots(6, 1, figsize=(14, 14))

    t = np.arange(len(signal)) / sr

    # Original signal
    axes[0].plot(t, signal, linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Speech Signal with Silence Sections')
    axes[0].grid(True, alpha=0.3)

    # Ground truth
    t_frames = np.arange(len(ground_truth)) * frame_size / sr
    axes[1].fill_between(t_frames, 0, ground_truth, step='post', alpha=0.7, color='green')
    axes[1].set_ylabel('Speech')
    axes[1].set_title('Ground Truth (Actual Speech Activity)')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    # Energy-based
    axes[2].fill_between(t_frames, 0, energy_va, step='post', alpha=0.7, color='blue')
    axes[2].set_ylabel('Speech')
    axes[2].set_title(f'Energy-based VAD (Acc: {calculate_vad_metrics(energy_va, ground_truth)["accuracy"]:.3f})')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    # ZCR
    axes[3].fill_between(t_frames, 0, zcr_va, step='post', alpha=0.7, color='orange')
    axes[3].set_ylabel('Speech')
    axes[3].set_title(f'Zero-Crossing Rate VAD (Acc: {calculate_vad_metrics(zcr_va, ground_truth)["accuracy"]:.3f})')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True, alpha=0.3)

    # Spectral
    axes[4].fill_between(t_frames, 0, spectral_va, step='post', alpha=0.7, color='purple')
    axes[4].set_ylabel('Speech')
    axes[4].set_title(f'Spectral VAD (Acc: {calculate_vad_metrics(spectral_va, ground_truth)["accuracy"]:.3f})')
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].grid(True, alpha=0.3)

    # WebRTC
    webrtc_adjusted = webrtc_va[:len(ground_truth)]
    axes[5].fill_between(t_frames[:len(webrtc_adjusted)], 0, webrtc_adjusted, step='post', alpha=0.7, color='red')
    axes[5].set_xlabel('Time (seconds)')
    axes[5].set_ylabel('Speech')
    axes[5].set_title(f'WebRTC VAD (Acc: {calculate_vad_metrics(webrtc_adjusted, ground_truth)["accuracy"]:.3f})')
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vad_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'vad_comparison.png'}")


def demonstrate_vad_smoothing():
    """Demonstrate VAD smoothing."""
    print("\n" + "="*60)
    print("VAD SMOOTHING (REDUCE FALSE TRIGGERS)")
    print("="*60)

    signal, ground_truth, sr = generate_speech_with_silence(duration=5.0, sample_rate=16000)

    # Add some short noise bursts to create false triggers
    for i in range(5):
        start = np.random.randint(0, len(signal) - 1000)
        signal[start:start+500] += 0.3 * np.random.randn(500)

    # Apply VAD
    vad_raw, frame_size = energy_vad(signal, sr, frame_ms=20, threshold_db=-40)

    # Apply smoothing
    vad_smoothed = smooth_vad_decisions(vad_raw, min_speech_frames=5, min_silence_frames=5)

    print(f"\nSmoothing Configuration:")
    print(f"  Minimum Speech Frames: 5 ({5*20} ms)")
    print(f"  Minimum Silence Frames: 5 ({5*20} ms)")

    # Count transitions
    raw_transitions = np.sum(np.abs(np.diff(vad_raw.astype(int))))
    smoothed_transitions = np.sum(np.abs(np.diff(vad_smoothed.astype(int))))

    print(f"\nResults:")
    print(f"  Raw VAD Transitions: {raw_transitions}")
    print(f"  Smoothed VAD Transitions: {smoothed_transitions}")
    print(f"  Reduction: {100*(1-smoothed_transitions/raw_transitions):.1f}%")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    t = np.arange(len(signal)) / sr
    t_frames = np.arange(len(vad_raw)) * frame_size / sr

    # Signal
    axes[0].plot(t, signal, linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Signal with Noise Bursts')
    axes[0].grid(True, alpha=0.3)

    # Raw VAD
    axes[1].fill_between(t_frames, 0, vad_raw, step='post', alpha=0.7, color='orange')
    axes[1].set_ylabel('Speech')
    axes[1].set_title(f'Raw VAD ({raw_transitions} transitions)')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    # Smoothed VAD
    axes[2].fill_between(t_frames, 0, vad_smoothed, step='post', alpha=0.7, color='green')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Speech')
    axes[2].set_title(f'Smoothed VAD ({smoothed_transitions} transitions)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vad_smoothing.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'vad_smoothing.png'}")


def vad_best_practices():
    """VAD best practices for voice engines."""
    print("\n" + "="*60)
    print("VAD BEST PRACTICES FOR VOICE ENGINES")
    print("="*60)

    print("""
1. Algorithm Selection:
   Simple Applications:
   - Energy-based: Very fast, works in quiet environments
   - Good for: Wake word detection, simple apps

   Moderate Noise:
   - WebRTC VAD: Good balance, production-ready
   - Good for: Most voice applications

   High Noise:
   - Silero VAD: Deep learning, very robust
   - Pyannote-audio: State-of-the-art
   - Good for: Noisy environments, mission-critical

2. Frame Size Selection:
   - 10ms: Very responsive, may be jittery
   - 20ms: Good balance (RECOMMENDED)
   - 30ms: More stable, slightly higher latency

3. Smoothing:
   - Always apply smoothing to reduce false triggers
   - Minimum speech duration: 100-200ms
   - Minimum silence duration: 100-200ms

4. Thresholds:
   Energy-based:
   - Quiet environment: -40 dB
   - Moderate noise: -35 dB
   - Noisy: -30 dB

   WebRTC Aggressiveness:
   - 0: Least aggressive (more false positives)
   - 1: Moderate-low
   - 2: Moderate-high (RECOMMENDED)
   - 3: Most aggressive (may cut off speech)

5. Use Cases in Voice Engines:

   a) Turn Detection:
      - Detect when user starts/stops speaking
      - Trigger ASR processing
      - Critical for conversation flow

   b) Barge-in Detection:
      - Detect user interrupting agent
      - Stop TTS playback
      - Switch to listening mode

   c) Computational Efficiency:
      - Only run ASR on speech segments
      - Save 50-80% compute vs always-on ASR

   d) Endpoint Detection:
      - Determine when utterance is complete
      - Typically: 500-1000ms silence = end of turn

6. Advanced Techniques:
   - Dual-threshold: Low threshold to start, high to continue
   - Adaptive thresholds: Adjust based on noise level
   - Look-ahead: Buffer audio before VAD trigger
   - Look-back: Include audio after VAD ends

7. Integration with ASR:
   Some modern ASR models have built-in VAD:
   - Whisper: Has implicit VAD
   - Streaming ASR: Often includes VAD
   - May not need separate VAD!

8. Testing:
   - Test in target environment
   - Measure false positive rate (speech when silent)
   - Measure false negative rate (silent when speaking)
   - Balance based on application needs
    """)

    print("\n" + "="*60)
    print("RECOMMENDED VAD LIBRARIES")
    print("="*60)
    print("""
1. WebRTC VAD (py-webrtcvad):
   - Fast, lightweight
   - Production-proven
   - C++ backend
   pip install webrtcvad

2. Silero VAD:
   - Deep learning (ONNX)
   - Very robust to noise
   - Still lightweight
   pip install silero-vad

3. Pyannote-audio:
   - State-of-the-art quality
   - Heavier (requires PyTorch)
   - Best for accuracy
   pip install pyannote.audio

4. RNNoise:
   - Combined denoising + VAD
   - Real-time capable
   - Mozilla project

For production voice engines: Start with WebRTC VAD or Silero VAD
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 1, STEP 6: VOICE ACTIVITY DETECTION (VAD)")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding VAD and its importance")
    print("2. Energy-based VAD (simple threshold)")
    print("3. Spectral-based VAD")
    print("4. WebRTC VAD (production-ready)")
    print("5. Smoothing and post-processing")
    print("6. Integration with voice engines")

    # Run demonstrations
    demonstrate_vad_comparison()
    demonstrate_vad_smoothing()
    vad_best_practices()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. VAD is Critical for Voice Engines:
   - Reduces computational load (50-80%)
   - Enables turn-taking
   - Detects barge-in
   - Improves user experience

2. Multiple Approaches:
   - Energy-based: Fast, simple (good for quiet)
   - Spectral: More robust
   - WebRTC: Production-ready, balanced
   - Deep Learning: Best quality (Silero, Pyannote)

3. Key Parameters:
   - Frame size: 20ms recommended
   - Smoothing: Always apply (min 100ms)
   - Thresholds: Environment-dependent

4. Trade-offs:
   - False Positives: Process noise as speech (wasted compute)
   - False Negatives: Miss speech (bad UX, cut-off words)
   - Balance based on application

5. For Voice Engines:
   - Use WebRTC VAD (good default)
   - Or Silero VAD (better quality)
   - 20ms frames, aggressiveness=2
   - Apply smoothing (min 5 frames)
   - Integrate with turn-taking logic

6. Modern Considerations:
   - Some ASR models have built-in VAD
   - End-to-end models may not need separate VAD
   - Test with your specific ASR model

7. Endpoint Detection:
   - 500-1000ms silence = end of turn
   - Critical for conversational AI
   - Balance responsiveness vs cutting off speech
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)
    print("\n🎉 CONGRATULATIONS! You've completed Stage 1: Foundations")
    print("\nYou now understand:")
    print("  ✓ Digital audio fundamentals")
    print("  ✓ Audio framing and windowing")
    print("  ✓ Frequency analysis (FFT/STFT)")
    print("  ✓ Perceptual features (Mel spectrograms)")
    print("  ✓ Noise reduction techniques")
    print("  ✓ Voice activity detection")
    print("\nYou're ready to move on to Stage 2: Speech-to-Text (ASR)!")
    print("="*60)


if __name__ == "__main__":
    main()
