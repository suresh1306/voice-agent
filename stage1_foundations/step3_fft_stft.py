"""
Step 3: FFT and STFT (Short-Time Fourier Transform)
====================================================
Learn frequency analysis of audio signals:
- FFT (Fast Fourier Transform): Convert time → frequency domain
- STFT: Analyze how frequencies change over time
- Spectrograms: Visual representation of frequency content
- Phase and magnitude

Key Concepts:
- FFT converts time-domain signal to frequency domain
- STFT = FFT applied to overlapping frames (time-frequency representation)
- Spectrogram = |STFT|² (magnitude squared)
- Frequency resolution = sample_rate / FFT_size
- Time resolution = frame_size / sample_rate
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq

OUTPUT_DIR = Path("outputs/step3")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_complex_signal(duration=2.0, sample_rate=16000):
    """Generate a signal with time-varying frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create signal with frequency that changes over time
    signal = np.zeros_like(t)

    # Segment 1: Low frequency (0-0.5s)
    mask1 = t < 0.5
    signal[mask1] = np.sin(2 * np.pi * 200 * t[mask1])

    # Segment 2: Mid frequency (0.5-1.0s)
    mask2 = (t >= 0.5) & (t < 1.0)
    signal[mask2] = np.sin(2 * np.pi * 600 * t[mask2])

    # Segment 3: High frequency (1.0-1.5s)
    mask3 = (t >= 1.0) & (t < 1.5)
    signal[mask3] = np.sin(2 * np.pi * 1200 * t[mask3])

    # Segment 4: Chord (multiple frequencies) (1.5-2.0s)
    mask4 = t >= 1.5
    signal[mask4] = (
        0.5 * np.sin(2 * np.pi * 300 * t[mask4]) +
        0.3 * np.sin(2 * np.pi * 600 * t[mask4]) +
        0.2 * np.sin(2 * np.pi * 900 * t[mask4])
    )

    return signal, t, sample_rate


def demonstrate_fft_basics():
    """Demonstrate basic FFT analysis."""
    print("\n" + "="*60)
    print("FFT (FAST FOURIER TRANSFORM) BASICS")
    print("="*60)

    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a signal with known frequencies
    freq1, freq2, freq3 = 440, 880, 1320  # A4, A5, E6
    signal = (
        0.5 * np.sin(2 * np.pi * freq1 * t) +
        0.3 * np.sin(2 * np.pi * freq2 * t) +
        0.2 * np.sin(2 * np.pi * freq3 * t)
    )

    # Compute FFT
    fft_result = rfft(signal)
    frequencies = rfftfreq(len(signal), 1/sample_rate)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)

    # Find peaks
    peak_indices = np.where(magnitude > 0.1 * np.max(magnitude))[0]
    detected_freqs = frequencies[peak_indices]

    print(f"\nTest Signal Contains:")
    print(f"  {freq1} Hz (amplitude 0.5)")
    print(f"  {freq2} Hz (amplitude 0.3)")
    print(f"  {freq3} Hz (amplitude 0.2)")

    print(f"\nFFT Analysis:")
    print(f"  FFT Size: {len(signal)} samples")
    print(f"  Frequency Resolution: {sample_rate / len(signal):.2f} Hz")
    print(f"  Frequency Range: 0 to {sample_rate / 2} Hz (Nyquist)")

    print(f"\nDetected Frequencies:")
    for idx, freq in zip(peak_indices[:10], detected_freqs[:10]):
        mag = magnitude[idx]
        print(f"  {freq:.1f} Hz (magnitude: {mag:.1f})")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Time domain
    axes[0].plot(t[:1000], signal[:1000], linewidth=1)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain: Multi-Tone Signal')
    axes[0].grid(True, alpha=0.3)

    # Frequency domain (magnitude)
    axes[1].plot(frequencies, magnitude, linewidth=1)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Domain: Magnitude Spectrum')
    axes[1].set_xlim(0, 2000)
    axes[1].grid(True, alpha=0.3)

    # Mark detected peaks
    for freq in detected_freqs[:10]:
        if freq < 2000:
            axes[1].axvline(freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

    # Phase
    axes[2].plot(frequencies, phase, linewidth=1)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Phase (radians)')
    axes[2].set_title('Frequency Domain: Phase Spectrum')
    axes[2].set_xlim(0, 2000)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fft_basics.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'fft_basics.png'}")


def demonstrate_fft_resolution():
    """Demonstrate FFT frequency resolution."""
    print("\n" + "="*60)
    print("FFT FREQUENCY RESOLUTION")
    print("="*60)
    print("\nFrequency Resolution = Sample Rate / FFT Size")

    sample_rate = 16000
    duration_options = [0.1, 0.5, 1.0]  # Different durations = different FFT sizes

    # Two close frequencies
    freq1, freq2 = 440, 460  # Only 20 Hz apart

    fig, axes = plt.subplots(len(duration_options), 1, figsize=(14, 10))

    for idx, duration in enumerate(duration_options):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)

        # Compute FFT
        fft_result = rfft(signal)
        frequencies = rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft_result)

        freq_resolution = sample_rate / len(signal)

        print(f"\nDuration: {duration}s ({len(signal)} samples)")
        print(f"  Frequency Resolution: {freq_resolution:.2f} Hz")
        print(f"  Can resolve {freq1} and {freq2} Hz? {freq_resolution < 20}")

        # Plot
        ax = axes[idx]
        ax.plot(frequencies, magnitude, linewidth=1)
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Duration: {duration}s, Resolution: {freq_resolution:.2f} Hz/bin')
        ax.set_xlim(400, 500)
        ax.axvline(freq1, color='r', linestyle='--', alpha=0.5, label=f'{freq1} Hz')
        ax.axvline(freq2, color='g', linestyle='--', alpha=0.5, label=f'{freq2} Hz')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if idx == len(duration_options) - 1:
            ax.set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fft_resolution.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'fft_resolution.png'}")


def compute_stft(signal, sample_rate, frame_size=512, hop_size=256, window='hann'):
    """
    Compute Short-Time Fourier Transform.

    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        frame_size: FFT size
        hop_size: Hop size in samples
        window: Window function

    Returns:
        frequencies, times, stft_matrix
    """
    # Use scipy's STFT
    f, t, Zxx = scipy_signal.stft(
        signal,
        fs=sample_rate,
        window=window,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        nfft=frame_size
    )

    return f, t, Zxx


def demonstrate_stft():
    """Demonstrate STFT analysis."""
    print("\n" + "="*60)
    print("STFT (SHORT-TIME FOURIER TRANSFORM)")
    print("="*60)

    # Generate signal with time-varying frequency
    signal, t, sample_rate = generate_complex_signal(duration=2.0, sample_rate=16000)

    # STFT parameters
    frame_size = 512
    hop_size = 256

    print(f"\nSTFT Parameters:")
    print(f"  Frame Size: {frame_size} samples ({frame_size/sample_rate*1000:.1f} ms)")
    print(f"  Hop Size: {hop_size} samples ({hop_size/sample_rate*1000:.1f} ms)")
    print(f"  Overlap: {100*(1-hop_size/frame_size):.0f}%")
    print(f"  Frequency Resolution: {sample_rate/frame_size:.2f} Hz")
    print(f"  Time Resolution: {hop_size/sample_rate*1000:.1f} ms")

    # Compute STFT
    frequencies, times, Zxx = compute_stft(signal, sample_rate, frame_size, hop_size)

    # Magnitude spectrogram
    magnitude = np.abs(Zxx)

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Time domain
    axes[0].plot(t, signal, linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Signal (Frequency Changes Over Time)')
    axes[0].grid(True, alpha=0.3)

    # Spectrogram
    im = axes[1].pcolormesh(times, frequencies, magnitude, shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_title('STFT Spectrogram (Time-Frequency Representation)')
    axes[1].set_ylim(0, 2000)
    plt.colorbar(im, ax=axes[1], label='Magnitude')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stft_basic.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'stft_basic.png'}")


def compare_stft_parameters():
    """Compare different STFT parameters."""
    print("\n" + "="*60)
    print("STFT PARAMETER COMPARISON")
    print("="*60)

    signal, t, sample_rate = generate_complex_signal(duration=2.0, sample_rate=16000)

    # Different configurations
    configs = [
        {"frame_size": 256, "hop_size": 128, "name": "Short Frame (High Time Resolution)"},
        {"frame_size": 512, "hop_size": 256, "name": "Medium Frame (Balanced)"},
        {"frame_size": 1024, "hop_size": 512, "name": "Long Frame (High Frequency Resolution)"},
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 12))

    for idx, config in enumerate(configs):
        frame_size = config['frame_size']
        hop_size = config['hop_size']
        name = config['name']

        # Compute STFT
        frequencies, times, Zxx = compute_stft(signal, sample_rate, frame_size, hop_size)
        magnitude = np.abs(Zxx)

        freq_res = sample_rate / frame_size
        time_res = hop_size / sample_rate * 1000

        print(f"\n{name}:")
        print(f"  Frame Size: {frame_size} samples")
        print(f"  Frequency Resolution: {freq_res:.2f} Hz")
        print(f"  Time Resolution: {time_res:.1f} ms")

        # Plot
        ax = axes[idx]
        im = ax.pcolormesh(times, frequencies, magnitude, shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'{name} (Freq Res: {freq_res:.1f} Hz, Time Res: {time_res:.1f} ms)')
        ax.set_ylim(0, 1500)
        plt.colorbar(im, ax=ax, label='Magnitude')

        if idx == len(configs) - 1:
            ax.set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stft_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'stft_comparison.png'}")


def demonstrate_power_spectrogram():
    """Demonstrate power spectrogram (magnitude squared)."""
    print("\n" + "="*60)
    print("POWER SPECTROGRAM")
    print("="*60)

    signal, t, sample_rate = generate_complex_signal(duration=2.0, sample_rate=16000)

    # Add some noise
    signal += 0.1 * np.random.randn(len(signal))

    frame_size = 512
    hop_size = 256

    # Compute STFT
    frequencies, times, Zxx = compute_stft(signal, sample_rate, frame_size, hop_size)

    # Different representations
    magnitude = np.abs(Zxx)
    power = magnitude ** 2
    log_power = 10 * np.log10(power + 1e-10)  # dB scale

    print(f"\nSpectrogram Representations:")
    print(f"  1. Magnitude: |STFT|")
    print(f"  2. Power: |STFT|²")
    print(f"  3. Log Power (dB): 10 * log10(|STFT|²)")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Magnitude
    im1 = axes[0].pcolormesh(times, frequencies, magnitude, shading='gouraud', cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Magnitude Spectrogram: |STFT|')
    axes[0].set_ylim(0, 1500)
    plt.colorbar(im1, ax=axes[0], label='Magnitude')

    # Power
    im2 = axes[1].pcolormesh(times, frequencies, power, shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Power Spectrogram: |STFT|²')
    axes[1].set_ylim(0, 1500)
    plt.colorbar(im2, ax=axes[1], label='Power')

    # Log Power (dB)
    im3 = axes[2].pcolormesh(times, frequencies, log_power, shading='gouraud', cmap='viridis')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Log Power Spectrogram (dB): 10*log₁₀(|STFT|²)')
    axes[2].set_ylim(0, 1500)
    plt.colorbar(im3, ax=axes[2], label='Power (dB)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_spectrogram.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'power_spectrogram.png'}")


def demonstrate_inverse_stft():
    """Demonstrate STFT reconstruction using inverse STFT."""
    print("\n" + "="*60)
    print("INVERSE STFT (Signal Reconstruction)")
    print("="*60)

    signal, t, sample_rate = generate_complex_signal(duration=1.0, sample_rate=16000)

    frame_size = 512
    hop_size = 256

    # Compute STFT
    frequencies, times, Zxx = compute_stft(signal, sample_rate, frame_size, hop_size)

    # Reconstruct signal using inverse STFT
    _, reconstructed = scipy_signal.istft(
        Zxx,
        fs=sample_rate,
        window='hann',
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        nfft=frame_size
    )

    # Trim to original length
    reconstructed = reconstructed[:len(signal)]

    # Calculate error
    error = np.mean(np.abs(signal - reconstructed))
    max_error = np.max(np.abs(signal - reconstructed))

    print(f"\nReconstruction Quality:")
    print(f"  Mean Absolute Error: {error:.8f}")
    print(f"  Max Absolute Error: {max_error:.8f}")
    print(f"  Correlation: {np.corrcoef(signal, reconstructed)[0, 1]:.8f}")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original
    axes[0].plot(t[:2000], signal[:2000], 'b-', linewidth=1, label='Original')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Reconstructed
    t_rec = np.linspace(0, len(reconstructed)/sample_rate, len(reconstructed))
    axes[1].plot(t_rec[:2000], reconstructed[:2000], 'r-', linewidth=1, label='Reconstructed')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Reconstructed from STFT (Inverse STFT)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Error
    error_signal = signal - reconstructed
    axes[2].plot(t[:2000], error_signal[:2000], 'g-', linewidth=1, label='Error')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Reconstruction Error (Mean: {error:.8f})')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inverse_stft.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'inverse_stft.png'}")


def stft_best_practices():
    """Provide STFT best practices."""
    print("\n" + "="*60)
    print("STFT BEST PRACTICES")
    print("="*60)

    sample_rate = 16000

    print(f"\nFor {sample_rate} Hz Audio (Voice):\n")

    recommendations = {
        "Voice Activity Detection": {
            "frame_size": 512,
            "hop_size": 160,
            "freq_res": 31.25,
            "time_res": 10,
        },
        "Speech Recognition": {
            "frame_size": 512,
            "hop_size": 256,
            "freq_res": 31.25,
            "time_res": 16,
        },
        "Music Analysis": {
            "frame_size": 2048,
            "hop_size": 512,
            "freq_res": 7.8,
            "time_res": 32,
        },
        "Real-time Processing": {
            "frame_size": 256,
            "hop_size": 128,
            "freq_res": 62.5,
            "time_res": 8,
        },
    }

    print(f"{'Application':<25} {'Frame':<10} {'Hop':<10} {'Freq Res':<12} {'Time Res'}")
    print("-" * 75)

    for app, params in recommendations.items():
        print(f"{app:<25} {params['frame_size']:<10} {params['hop_size']:<10} "
              f"{params['freq_res']:.1f} Hz    {params['time_res']:.1f} ms")

    print("\n" + "="*60)
    print("KEY GUIDELINES")
    print("="*60)
    print("""
1. Trade-off: Time vs Frequency Resolution
   - Longer frames → better frequency resolution
   - Shorter frames → better time resolution
   - Cannot optimize both simultaneously (uncertainty principle)

2. Power-of-2 FFT sizes (256, 512, 1024, 2048):
   - Efficient FFT computation
   - Standard in audio processing

3. Overlap (50-75%):
   - 50% (hop = frame/2): Most common, good balance
   - 75% (hop = frame/4): Smoother, more redundant
   - 0% (hop = frame): No redundancy, less smooth

4. Window Functions:
   - Hann: General purpose, good sidelobe suppression
   - Hamming: Better frequency resolution
   - Always use windowing (never rectangular for STFT)

5. For Voice Engines:
   - Use 512-sample frames (32ms at 16kHz)
   - 50% overlap (hop_size = 256)
   - Hann window
   - Convert to dB scale for better visualization
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 1, STEP 3: FFT AND STFT")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding FFT (time → frequency conversion)")
    print("2. Frequency resolution and trade-offs")
    print("3. STFT for time-frequency analysis")
    print("4. Power spectrograms")
    print("5. Inverse STFT reconstruction")

    # Run demonstrations
    demonstrate_fft_basics()
    demonstrate_fft_resolution()
    demonstrate_stft()
    compare_stft_parameters()
    demonstrate_power_spectrogram()
    demonstrate_inverse_stft()
    stft_best_practices()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. FFT transforms time-domain signal to frequency domain
   - Shows which frequencies are present
   - Frequency resolution = sample_rate / FFT_size

2. STFT analyzes how frequencies change over time
   - Apply FFT to overlapping frames
   - Creates a spectrogram (time-frequency representation)

3. Spectrogram Types:
   - Magnitude: |STFT|
   - Power: |STFT|²
   - Log Power (dB): 10*log₁₀(|STFT|²) ← Most common

4. Time-Frequency Trade-off:
   - Long frames: good frequency, poor time resolution
   - Short frames: good time, poor frequency resolution
   - Cannot optimize both (Heisenberg uncertainty principle)

5. For Voice Processing:
   - 512-sample frames at 16kHz (32ms)
   - 50% overlap (hop = 256 samples)
   - Hann window
   - dB-scale spectrogram

6. STFT is invertible:
   - Can reconstruct original signal perfectly
   - Essential for audio modification
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
