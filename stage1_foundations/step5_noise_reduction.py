"""
Step 5: Noise Reduction
========================
Learn basic noise reduction techniques:
- Spectral Subtraction: Remove noise spectrum from signal
- Wiener Filter: Optimal filter in MSE sense
- Signal-to-Noise Ratio (SNR)
- Power spectrum estimation
- Real-time noise reduction

Key Concepts:
- Noise profile: Estimate from silent sections
- Spectral subtraction: Subtract noise power spectrum
- Over-subtraction: Remove more noise (may distort)
- Minimum statistics: Track noise floor adaptively
- Critical for voice engines (clean input → better ASR)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.fft import rfft, irfft, rfftfreq
import soundfile as sf

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_noisy_speech(duration=3.0, sample_rate=16000, snr_db=5):
    """
    Generate synthetic speech with noise.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        snr_db: Signal-to-Noise Ratio in dB

    Returns:
        clean_speech, noisy_speech, noise, sample_rate
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate clean speech-like signal (varying pitch with formants)
    clean_speech = np.zeros_like(t)

    # Create segments of speech
    num_segments = 6
    segment_duration = duration / num_segments

    for i in range(num_segments):
        start_idx = int(i * segment_duration * sample_rate)
        end_idx = int((i + 1) * segment_duration * sample_rate)

        if i % 2 == 0:  # Voiced segments
            # Varying fundamental frequency
            f0 = 120 + 40 * np.sin(2 * np.pi * 2 * t[start_idx:end_idx])
            segment = np.sin(2 * np.pi * f0 * np.arange(len(f0)) / sample_rate)

            # Add formants
            for fc in [800, 1200, 2500]:
                sos = scipy_signal.butter(4, [fc-100, fc+100], btype='band',
                                         fs=sample_rate, output='sos')
                segment += 0.2 * scipy_signal.sosfilt(sos, segment)

            clean_speech[start_idx:end_idx] = segment
        else:  # Unvoiced/silent segments
            clean_speech[start_idx:end_idx] = 0.2 * np.random.randn(end_idx - start_idx)

    # Normalize
    clean_speech = clean_speech / np.max(np.abs(clean_speech)) * 0.8

    # Generate different types of noise
    # White noise
    white_noise = np.random.randn(len(t))

    # Pink noise (1/f noise) - more realistic
    white_fft = rfft(white_noise)
    freqs = rfftfreq(len(t), 1/sample_rate)
    pink_filter = 1 / np.sqrt(freqs + 1)  # 1/f characteristic
    pink_fft = white_fft * pink_filter
    noise = irfft(pink_fft, n=len(t))

    # Add some environmental noise (low frequency rumble)
    rumble = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    noise += 0.3 * rumble

    # Normalize noise
    noise = noise / np.std(noise)

    # Calculate signal power
    signal_power = np.mean(clean_speech ** 2)

    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = noise * np.sqrt(noise_power)

    # Create noisy speech
    noisy_speech = clean_speech + noise

    return clean_speech, noisy_speech, noise, sample_rate


def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return snr_db


def estimate_noise_profile(noisy_signal, sample_rate, noise_duration=0.5, n_fft=None):
    """
    Estimate noise profile from initial silent period.

    Args:
        noisy_signal: Noisy audio signal
        sample_rate: Sample rate in Hz
        noise_duration: Duration of noise-only section (seconds)
        n_fft: FFT size (defaults to signal length for matching dimensions)

    Returns:
        Noise power spectrum
    """
    # Extract noise-only section (beginning)
    noise_samples = int(noise_duration * sample_rate)
    noise_section = noisy_signal[:noise_samples]

    # Use signal length as FFT size if not specified
    if n_fft is None:
        n_fft = len(noisy_signal)

    # Compute FFT with specified size (zero-padding if needed)
    noise_fft = rfft(noise_section, n=n_fft)
    noise_power = np.abs(noise_fft) ** 2

    return noise_power


def spectral_subtraction(noisy_signal, noise_power, sample_rate, alpha=2.0, beta=0.01):
    """
    Apply spectral subtraction noise reduction.

    Args:
        noisy_signal: Noisy audio signal
        noise_power: Estimated noise power spectrum
        sample_rate: Sample rate in Hz
        alpha: Over-subtraction factor (>1 removes more noise)
        beta: Spectral floor (prevents negative values)

    Returns:
        Denoised signal
    """
    # Compute FFT of noisy signal
    noisy_fft = rfft(noisy_signal)
    noisy_magnitude = np.abs(noisy_fft)
    noisy_phase = np.angle(noisy_fft)

    # Ensure noise power matches signal FFT size
    if len(noise_power) != len(noisy_magnitude):
        raise ValueError(f"Noise power spectrum size ({len(noise_power)}) must match "
                        f"signal FFT size ({len(noisy_magnitude)}). Use n_fft parameter "
                        f"in estimate_noise_profile().")

    # Spectral subtraction
    clean_magnitude = noisy_magnitude ** 2 - alpha * noise_power

    # Apply spectral floor (minimum magnitude)
    spectral_floor = beta * noisy_magnitude ** 2
    clean_magnitude = np.maximum(clean_magnitude, spectral_floor)

    # Convert back to magnitude
    clean_magnitude = np.sqrt(clean_magnitude)

    # Reconstruct complex spectrum
    clean_fft = clean_magnitude * np.exp(1j * noisy_phase)

    # Inverse FFT
    clean_signal = irfft(clean_fft, n=len(noisy_signal))

    return clean_signal


def demonstrate_noise_reduction():
    """Demonstrate spectral subtraction noise reduction."""
    print("\n" + "="*60)
    print("SPECTRAL SUBTRACTION NOISE REDUCTION")
    print("="*60)

    # Generate noisy speech
    snr_db = 5
    clean, noisy, noise, sr = generate_noisy_speech(duration=3.0, sample_rate=16000,
                                                     snr_db=snr_db)

    print(f"\nTest Signal:")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Duration: {len(clean)/sr:.1f} seconds")
    print(f"  Target SNR: {snr_db} dB")
    print(f"  Actual SNR: {calculate_snr(clean, noise):.1f} dB")

    # Estimate noise from first 0.5 seconds
    noise_power = estimate_noise_profile(noisy, sr, noise_duration=0.5)

    # Apply spectral subtraction with different parameters
    denoised_mild = spectral_subtraction(noisy, noise_power, sr, alpha=1.0, beta=0.01)
    denoised_moderate = spectral_subtraction(noisy, noise_power, sr, alpha=2.0, beta=0.01)
    denoised_aggressive = spectral_subtraction(noisy, noise_power, sr, alpha=3.0, beta=0.01)

    # Calculate SNR improvements
    snr_noisy = calculate_snr(clean, noisy - clean)
    snr_mild = calculate_snr(clean, denoised_mild - clean)
    snr_moderate = calculate_snr(clean, denoised_moderate - clean)
    snr_aggressive = calculate_snr(clean, denoised_aggressive - clean)

    print(f"\nNoise Reduction Results:")
    print(f"  Noisy Signal SNR: {snr_noisy:.1f} dB")
    print(f"  Mild (α=1.0): {snr_mild:.1f} dB (improvement: {snr_mild-snr_noisy:+.1f} dB)")
    print(f"  Moderate (α=2.0): {snr_moderate:.1f} dB (improvement: {snr_moderate-snr_noisy:+.1f} dB)")
    print(f"  Aggressive (α=3.0): {snr_aggressive:.1f} dB (improvement: {snr_aggressive-snr_noisy:+.1f} dB)")

    # Save audio files
    sf.write(OUTPUT_DIR / "clean_speech.wav", clean, sr)
    sf.write(OUTPUT_DIR / "noisy_speech.wav", noisy, sr)
    sf.write(OUTPUT_DIR / "denoised_moderate.wav", denoised_moderate, sr)
    print(f"\n✓ Saved audio files to {OUTPUT_DIR}/")

    # Visualize
    fig, axes = plt.subplots(5, 1, figsize=(14, 14))

    t = np.arange(len(clean)) / sr
    plot_duration = 1.0  # Show first second
    plot_samples = int(plot_duration * sr)

    # Clean speech
    axes[0].plot(t[:plot_samples], clean[:plot_samples], linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Clean Speech')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1, 1)

    # Noisy speech
    axes[1].plot(t[:plot_samples], noisy[:plot_samples], linewidth=0.5, color='orange')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Noisy Speech (SNR: {snr_noisy:.1f} dB)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1, 1)

    # Denoised (mild)
    axes[2].plot(t[:plot_samples], denoised_mild[:plot_samples], linewidth=0.5, color='green')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Denoised - Mild (α=1.0, SNR: {snr_mild:.1f} dB)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1, 1)

    # Denoised (moderate)
    axes[3].plot(t[:plot_samples], denoised_moderate[:plot_samples], linewidth=0.5, color='purple')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_title(f'Denoised - Moderate (α=2.0, SNR: {snr_moderate:.1f} dB)')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-1, 1)

    # Denoised (aggressive)
    axes[4].plot(t[:plot_samples], denoised_aggressive[:plot_samples], linewidth=0.5, color='red')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].set_ylabel('Amplitude')
    axes[4].set_title(f'Denoised - Aggressive (α=3.0, SNR: {snr_aggressive:.1f} dB)')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_reduction_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'noise_reduction_comparison.png'}")


def demonstrate_spectral_analysis():
    """Show spectral analysis of noise reduction."""
    print("\n" + "="*60)
    print("SPECTRAL ANALYSIS")
    print("="*60)

    # Generate noisy speech
    clean, noisy, noise, sr = generate_noisy_speech(duration=2.0, sample_rate=16000, snr_db=5)

    # Estimate noise and denoise
    noise_power = estimate_noise_profile(noisy, sr)
    denoised = spectral_subtraction(noisy, noise_power, sr, alpha=2.0)

    # Compute spectrograms
    nperseg = 512
    noverlap = 384

    f_clean, t_clean, Sxx_clean = scipy_signal.spectrogram(clean, sr, nperseg=nperseg,
                                                           noverlap=noverlap)
    f_noisy, t_noisy, Sxx_noisy = scipy_signal.spectrogram(noisy, sr, nperseg=nperseg,
                                                           noverlap=noverlap)
    f_denoised, t_denoised, Sxx_denoised = scipy_signal.spectrogram(denoised, sr, nperseg=nperseg,
                                                                     noverlap=noverlap)

    # Convert to dB
    Sxx_clean_db = 10 * np.log10(Sxx_clean + 1e-10)
    Sxx_noisy_db = 10 * np.log10(Sxx_noisy + 1e-10)
    Sxx_denoised_db = 10 * np.log10(Sxx_denoised + 1e-10)

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    vmin, vmax = -60, 0

    # Clean
    im1 = axes[0].pcolormesh(t_clean, f_clean, Sxx_clean_db, shading='gouraud',
                            cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Clean Speech Spectrogram')
    axes[0].set_ylim(0, 4000)
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    # Noisy
    im2 = axes[1].pcolormesh(t_noisy, f_noisy, Sxx_noisy_db, shading='gouraud',
                            cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Noisy Speech Spectrogram')
    axes[1].set_ylim(0, 4000)
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    # Denoised
    im3 = axes[2].pcolormesh(t_denoised, f_denoised, Sxx_denoised_db, shading='gouraud',
                            cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title('Denoised Speech Spectrogram')
    axes[2].set_ylim(0, 4000)
    plt.colorbar(im3, ax=axes[2], label='Power (dB)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_reduction_spectrograms.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'noise_reduction_spectrograms.png'}")


def frame_based_noise_reduction(noisy_signal, sample_rate, frame_size=512, hop_size=256):
    """
    Frame-based noise reduction (suitable for real-time).

    Args:
        noisy_signal: Noisy audio signal
        sample_rate: Sample rate in Hz
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Denoised signal
    """
    # Estimate noise from first few frames
    noise_estimation_frames = 10
    noise_estimation_samples = noise_estimation_frames * hop_size
    noise_power = estimate_noise_profile(noisy_signal[:noise_estimation_samples],
                                        sample_rate, noise_estimation_samples/sample_rate,
                                        n_fft=frame_size)

    # Process frame by frame
    num_frames = 1 + (len(noisy_signal) - frame_size) // hop_size
    output = np.zeros(len(noisy_signal))
    window = np.hanning(frame_size)

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        if end > len(noisy_signal):
            break

        # Extract frame
        frame = noisy_signal[start:end] * window

        # Apply spectral subtraction
        denoised_frame = spectral_subtraction(frame, noise_power, sample_rate,
                                             alpha=2.0, beta=0.01)

        # Overlap-add
        output[start:end] += denoised_frame * window

    # Normalize overlap
    norm = np.zeros(len(noisy_signal))
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end <= len(noisy_signal):
            norm[start:end] += window ** 2

    norm[norm < 1e-10] = 1
    output = output / norm

    return output


def demonstrate_realtime_noise_reduction():
    """Demonstrate frame-based (real-time) noise reduction."""
    print("\n" + "="*60)
    print("REAL-TIME (FRAME-BASED) NOISE REDUCTION")
    print("="*60)

    clean, noisy, noise, sr = generate_noisy_speech(duration=2.0, sample_rate=16000, snr_db=5)

    frame_ms = 32
    frame_size = int(sr * frame_ms / 1000)
    hop_size = frame_size // 2

    print(f"\nReal-time Processing:")
    print(f"  Frame Size: {frame_ms} ms ({frame_size} samples)")
    print(f"  Hop Size: {hop_size} samples")
    print(f"  Latency: {frame_ms} ms")

    # Apply frame-based denoising
    denoised = frame_based_noise_reduction(noisy, sr, frame_size, hop_size)

    snr_before = calculate_snr(clean, noisy - clean)
    snr_after = calculate_snr(clean, denoised - clean)

    print(f"\nResults:")
    print(f"  SNR Before: {snr_before:.1f} dB")
    print(f"  SNR After: {snr_after:.1f} dB")
    print(f"  Improvement: {snr_after - snr_before:+.1f} dB")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    t = np.arange(len(clean)) / sr

    axes[0].plot(t, clean, linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Clean Speech')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, noisy, linewidth=0.5, color='orange')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Noisy Speech (SNR: {snr_before:.1f} dB)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, denoised, linewidth=0.5, color='green')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Real-time Denoised (SNR: {snr_after:.1f} dB, Latency: {frame_ms}ms)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "realtime_noise_reduction.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'realtime_noise_reduction.png'}")


def noise_reduction_best_practices():
    """Provide noise reduction best practices."""
    print("\n" + "="*60)
    print("NOISE REDUCTION BEST PRACTICES")
    print("="*60)

    print("""
1. Noise Profile Estimation:
   - Capture noise-only section at beginning
   - Or use adaptive noise estimation (minimum statistics)
   - Update noise profile periodically for non-stationary noise

2. Spectral Subtraction Parameters:
   - Alpha (over-subtraction): 1.0-3.0
     * 1.0: Mild, preserves speech quality
     * 2.0: Moderate, good balance
     * 3.0: Aggressive, may introduce distortion
   - Beta (spectral floor): 0.01-0.1
     * Prevents musical noise (random tones)

3. Trade-offs:
   - More aggressive → more noise removed, but more distortion
   - Less aggressive → cleaner speech, but residual noise
   - Musical noise: Common artifact (use spectral floor)

4. Advanced Techniques:
   - Wiener Filtering: Optimal in MSE sense
   - Deep Learning: RNNoise, DTLN, FullSubNet
   - Multi-band processing: Different treatment per frequency band

5. For Voice Engines:
   - Use moderate settings (α=2.0, β=0.01)
   - Frame-based for real-time (32ms frames)
   - Consider pre-trained models (RNNoise, Krisp)
   - Always test with ASR model (some ASR robust to noise)

6. When NOT to use:
   - Very clean recordings (may introduce artifacts)
   - When ASR model is noise-robust (Whisper is fairly robust)
   - High-quality studio recordings
    """)

    print("\n" + "="*60)
    print("MODERN ALTERNATIVES")
    print("="*60)
    print("""
Deep Learning Noise Reduction (Better than Spectral Subtraction):

1. RNNoise (Mozilla):
   - Real-time, lightweight
   - Recurrent Neural Network
   - Very effective for voice

2. DTLN (Dual-Signal Transformation LSTM Network):
   - State-of-the-art quality
   - Heavier computation

3. FullSubNet:
   - Excellent quality
   - Sub-band processing

4. Commercial:
   - Krisp (very popular)
   - Dolby Voice
   - NVIDIA Maxine

For production voice engines, consider using pre-trained models!
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 1, STEP 5: NOISE REDUCTION")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding noise in audio signals")
    print("2. Spectral subtraction technique")
    print("3. SNR calculation and improvement")
    print("4. Real-time frame-based processing")
    print("5. Trade-offs and best practices")

    # Run demonstrations
    demonstrate_noise_reduction()
    demonstrate_spectral_analysis()
    demonstrate_realtime_noise_reduction()
    noise_reduction_best_practices()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. Noise Reduction is Critical for Voice Engines:
   - Improves ASR accuracy
   - Better user experience
   - Reduces processing load

2. Spectral Subtraction:
   - Simple, fast, effective
   - Estimate noise spectrum → subtract from signal
   - Works in frequency domain

3. Parameters Matter:
   - Over-subtraction factor (α): controls aggressiveness
   - Spectral floor (β): prevents musical noise
   - Balance between noise removal and distortion

4. Real-time Processing:
   - Use frame-based processing (10-30ms frames)
   - Overlap-add for smooth reconstruction
   - Acceptable latency for voice apps

5. Modern Approaches:
   - Deep learning (RNNoise, DTLN) better than classical
   - Consider pre-trained models for production
   - Balance quality vs computational cost

6. For Voice Engines:
   - Test with your ASR model (Whisper is noise-robust)
   - Don't over-process clean audio
   - Consider environmental noise characteristics
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
