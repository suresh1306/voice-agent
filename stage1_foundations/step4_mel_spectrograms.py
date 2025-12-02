"""
Step 4: Mel Spectrograms
=========================
Learn perceptual frequency representation:
- Mel Scale: Mimics human hearing (logarithmic perception)
- Mel Filterbanks: Group frequencies like human ear
- Mel Spectrograms: Input for speech recognition models
- MFCC (Mel-Frequency Cepstral Coefficients)

Key Concepts:
- Humans perceive frequency logarithmically (not linearly)
- Mel scale: mel = 2595 * log10(1 + f/700)
- Mel filterbanks: Overlapping triangular filters
- MFCCs: Compact representation (typically 13-40 coefficients)
- Critical for Whisper, Wav2Vec2, and other ASR models
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.fft import fft, rfft, rfftfreq
import librosa
import librosa.display

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def hz_to_mel(freq_hz):
    """Convert frequency from Hz to Mel scale."""
    return 2595 * np.log10(1 + freq_hz / 700)


def mel_to_hz(mel):
    """Convert frequency from Mel scale to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def demonstrate_mel_scale():
    """Demonstrate the Mel scale transformation."""
    print("\n" + "="*60)
    print("MEL SCALE: PERCEPTUAL FREQUENCY")
    print("="*60)
    print("\nHumans perceive frequency logarithmically:")
    print("- Difference between 100Hz and 200Hz sounds large")
    print("- Difference between 1000Hz and 1100Hz sounds small")
    print("- Mel scale matches this perception")

    # Frequency range
    hz_frequencies = np.linspace(0, 8000, 1000)
    mel_frequencies = hz_to_mel(hz_frequencies)

    # Some key frequencies
    key_freqs_hz = [100, 200, 500, 1000, 2000, 4000, 8000]
    key_freqs_mel = [hz_to_mel(f) for f in key_freqs_hz]

    print("\n" + "="*60)
    print("Frequency Conversion Examples:")
    print("="*60)
    print(f"{'Hz':<10} {'Mel':<10} {'Difference (Mel)'}")
    print("-" * 40)

    for i, (hz, mel) in enumerate(zip(key_freqs_hz, key_freqs_mel)):
        if i > 0:
            mel_diff = mel - key_freqs_mel[i-1]
            print(f"{hz:<10} {mel:<10.1f} {mel_diff:.1f}")
        else:
            print(f"{hz:<10} {mel:<10.1f} -")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Hz scale (linear)
    axes[0].plot(hz_frequencies, hz_frequencies, 'b-', linewidth=2)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Linear Scale (Hz)')
    axes[0].set_title('Linear Frequency Scale (How we measure)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 8000)

    # Mel scale (perceptual)
    axes[1].plot(hz_frequencies, mel_frequencies, 'r-', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Mel Scale')
    axes[1].set_title('Mel Frequency Scale (How we hear)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 8000)

    # Mark key frequencies
    for hz, mel in zip(key_freqs_hz, key_freqs_mel):
        axes[1].plot(hz, mel, 'ko', markersize=8)
        axes[1].annotate(f'{hz} Hz', (hz, mel), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mel_scale.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'mel_scale.png'}")


def create_mel_filterbank(n_filters, fft_size, sample_rate, fmin=0, fmax=None):
    """
    Create Mel filterbank.

    Args:
        n_filters: Number of Mel filters
        fft_size: FFT size
        sample_rate: Sample rate in Hz
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz), defaults to Nyquist

    Returns:
        Mel filterbank matrix (n_filters x fft_bins)
    """
    if fmax is None:
        fmax = sample_rate / 2

    # Convert to Mel scale
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    # Create evenly spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)

    # Convert back to Hz
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bin numbers
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    filterbank = np.zeros((n_filters, fft_size // 2 + 1))

    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            filterbank[i, j] = (right - j) / (right - center)

    return filterbank, hz_points


def demonstrate_mel_filterbank():
    """Demonstrate Mel filterbank construction."""
    print("\n" + "="*60)
    print("MEL FILTERBANK")
    print("="*60)
    print("\nMel filterbanks group frequencies like the human ear:")
    print("- More filters at low frequencies (where we're sensitive)")
    print("- Fewer filters at high frequencies")
    print("- Triangular, overlapping filters")

    n_filters = 40
    fft_size = 512
    sample_rate = 16000

    # Create filterbank
    filterbank, hz_points = create_mel_filterbank(n_filters, fft_size, sample_rate)

    print(f"\nFilterbank Configuration:")
    print(f"  Number of Filters: {n_filters}")
    print(f"  FFT Size: {fft_size}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Frequency Range: 0 - {sample_rate/2} Hz")
    print(f"  Filterbank Shape: {filterbank.shape}")

    # Visualize all filters
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Frequency axis for plotting
    freqs = np.linspace(0, sample_rate/2, fft_size // 2 + 1)

    # Plot all filters
    for i in range(n_filters):
        axes[0].plot(freqs, filterbank[i], alpha=0.6, linewidth=1)

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Filter Response')
    axes[0].set_title(f'Mel Filterbank ({n_filters} filters)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, sample_rate/2)

    # Plot first 10 filters in detail
    for i in range(min(10, n_filters)):
        axes[1].plot(freqs, filterbank[i], linewidth=2, label=f'Filter {i+1}')

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Filter Response')
    axes[1].set_title('First 10 Mel Filters (Detail)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].set_xlim(0, 2000)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mel_filterbank.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'mel_filterbank.png'}")

    # Show filter center frequencies
    print(f"\nFirst 10 Filter Center Frequencies:")
    for i in range(min(10, n_filters)):
        print(f"  Filter {i+1:2d}: {hz_points[i+1]:6.1f} Hz")


def generate_speech_like_signal(duration=2.0, sample_rate=16000):
    """Generate a speech-like signal with formants."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Fundamental frequency (pitch) varies over time
    f0 = 150 + 50 * np.sin(2 * np.pi * 3 * t)  # Varying pitch

    # Create voiced segments
    signal = np.sin(2 * np.pi * f0 * t)

    # Add formants (resonances typical of vowels)
    # Formant frequencies for /a/ sound
    formants = [700, 1220, 2600]
    bandwidths = [130, 70, 160]

    for fc, bw in zip(formants, bandwidths):
        # Create formant using bandpass filter
        sos = scipy_signal.butter(4, [fc - bw/2, fc + bw/2], btype='band',
                                   fs=sample_rate, output='sos')
        formant_signal = scipy_signal.sosfilt(sos, signal)
        signal += 0.3 * formant_signal

    # Add some unvoiced segments (noise)
    noise_mask = (t % 0.5 < 0.1)  # Periodic noise bursts
    signal[noise_mask] += 0.5 * np.random.randn(np.sum(noise_mask))

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    return signal, sample_rate


def compute_mel_spectrogram(signal, sample_rate, n_fft=512, hop_length=256,
                           n_mels=40, fmin=0, fmax=None):
    """Compute Mel spectrogram using librosa."""
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def demonstrate_mel_spectrogram():
    """Demonstrate Mel spectrogram computation."""
    print("\n" + "="*60)
    print("MEL SPECTROGRAM")
    print("="*60)

    # Generate speech-like signal
    signal, sample_rate = generate_speech_like_signal(duration=2.0, sample_rate=16000)

    # Parameters
    n_fft = 512
    hop_length = 256
    n_mels = 80

    print(f"\nMel Spectrogram Parameters:")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  FFT Size: {n_fft}")
    print(f"  Hop Length: {hop_length}")
    print(f"  Number of Mel Bands: {n_mels}")

    # Compute regular spectrogram
    f, t, Zxx = scipy_signal.stft(signal, fs=sample_rate, nperseg=n_fft,
                                   noverlap=n_fft-hop_length)
    regular_spec = np.abs(Zxx)
    regular_spec_db = 10 * np.log10(regular_spec**2 + 1e-10)

    # Compute Mel spectrogram
    mel_spec_db = compute_mel_spectrogram(signal, sample_rate, n_fft, hop_length, n_mels)

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Time-domain signal
    t_signal = np.arange(len(signal)) / sample_rate
    axes[0].plot(t_signal, signal, linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Speech-like Signal (Time Domain)')
    axes[0].grid(True, alpha=0.3)

    # Regular spectrogram
    im1 = axes[1].pcolormesh(t, f, regular_spec_db, shading='gouraud',
                             cmap='viridis', vmin=-80, vmax=0)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Regular Spectrogram (Linear Frequency Scale)')
    axes[1].set_ylim(0, 4000)
    plt.colorbar(im1, ax=axes[1], label='Power (dB)')

    # Mel spectrogram
    img = librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='mel', ax=axes[2],
                                   cmap='viridis', fmax=sample_rate/2)
    axes[2].set_ylabel('Mel Frequency')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Mel Spectrogram (Perceptual Frequency Scale)')
    plt.colorbar(img, ax=axes[2], label='Power (dB)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mel_spectrogram_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'mel_spectrogram_comparison.png'}")


def compare_mel_resolutions():
    """Compare different Mel spectrogram resolutions."""
    print("\n" + "="*60)
    print("MEL SPECTROGRAM RESOLUTIONS")
    print("="*60)

    signal, sample_rate = generate_speech_like_signal(duration=2.0, sample_rate=16000)

    # Different Mel resolutions
    n_mels_options = [20, 40, 80, 128]

    fig, axes = plt.subplots(len(n_mels_options), 1, figsize=(14, 12))

    for idx, n_mels in enumerate(n_mels_options):
        # Compute Mel spectrogram
        mel_spec_db = compute_mel_spectrogram(signal, sample_rate, n_mels=n_mels)

        print(f"\n{n_mels} Mel Bands:")
        print(f"  Shape: {mel_spec_db.shape}")
        print(f"  Frequency Resolution: ~{sample_rate/(2*n_mels):.1f} Hz per band (approx)")

        # Plot
        img = librosa.display.specshow(mel_spec_db, sr=sample_rate,
                                       x_axis='time', y_axis='mel', ax=axes[idx],
                                       cmap='viridis', fmax=sample_rate/2)
        axes[idx].set_ylabel('Mel Freq')
        axes[idx].set_title(f'{n_mels} Mel Bands')
        plt.colorbar(img, ax=axes[idx], label='dB')

        if idx == len(n_mels_options) - 1:
            axes[idx].set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mel_spectrogram_resolutions.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'mel_spectrogram_resolutions.png'}")


def compute_mfcc(signal, sample_rate, n_mfcc=13, n_fft=512, hop_length=256):
    """Compute MFCCs (Mel-Frequency Cepstral Coefficients)."""
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfccs


def demonstrate_mfcc():
    """Demonstrate MFCC computation."""
    print("\n" + "="*60)
    print("MFCC (MEL-FREQUENCY CEPSTRAL COEFFICIENTS)")
    print("="*60)
    print("\nMFCCs are a compact representation of the Mel spectrogram:")
    print("- Apply DCT (Discrete Cosine Transform) to Mel spectrogram")
    print("- Keep first 13-40 coefficients")
    print("- Decorrelates features, reduces dimensionality")
    print("- Widely used in classical ASR systems")

    signal, sample_rate = generate_speech_like_signal(duration=2.0, sample_rate=16000)

    n_mfcc = 13
    n_fft = 512
    hop_length = 256

    # Compute MFCCs
    mfccs = compute_mfcc(signal, sample_rate, n_mfcc, n_fft, hop_length)

    # Compute Mel spectrogram for comparison
    mel_spec_db = compute_mel_spectrogram(signal, sample_rate, n_fft, hop_length, n_mels=40)

    print(f"\nMFCC Configuration:")
    print(f"  Number of MFCCs: {n_mfcc}")
    print(f"  MFCC Shape: {mfccs.shape}")
    print(f"  Mel Spectrogram Shape: {mel_spec_db.shape}")
    print(f"  Compression Ratio: {mel_spec_db.shape[0] / mfccs.shape[0]:.1f}x")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Time-domain signal
    t_signal = np.arange(len(signal)) / sample_rate
    axes[0].plot(t_signal, signal, linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Speech Signal')
    axes[0].grid(True, alpha=0.3)

    # Mel spectrogram
    img1 = librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=hop_length,
                                    x_axis='time', y_axis='mel', ax=axes[1],
                                    cmap='viridis')
    axes[1].set_ylabel('Mel Frequency')
    axes[1].set_title('Mel Spectrogram (40 bands)')
    plt.colorbar(img1, ax=axes[1], label='dB')

    # MFCCs
    img2 = librosa.display.specshow(mfccs, sr=sample_rate, hop_length=hop_length,
                                    x_axis='time', ax=axes[2], cmap='coolwarm')
    axes[2].set_ylabel('MFCC Coefficient')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title(f'MFCCs ({n_mfcc} coefficients) - Compact Representation')
    plt.colorbar(img2, ax=axes[2], label='Value')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mfcc_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'mfcc_analysis.png'}")


def mel_spectrogram_for_asr():
    """Show Mel spectrogram configuration for modern ASR."""
    print("\n" + "="*60)
    print("MEL SPECTROGRAMS FOR MODERN ASR")
    print("="*60)

    configs = {
        "Whisper (OpenAI)": {
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 80,
            "sample_rate": 16000,
            "notes": "10ms hop, 80 Mel bands"
        },
        "Wav2Vec2 (Meta)": {
            "n_fft": 400,
            "hop_length": 320,
            "n_mels": 80,
            "sample_rate": 16000,
            "notes": "20ms hop, direct waveform input"
        },
        "DeepSpeech": {
            "n_fft": 512,
            "hop_length": 256,
            "n_mels": 26,
            "sample_rate": 16000,
            "notes": "Classical MFCC-based"
        },
        "Conformer ASR": {
            "n_fft": 512,
            "hop_length": 256,
            "n_mels": 80,
            "sample_rate": 16000,
            "notes": "High resolution"
        },
    }

    print("\nCommon ASR Model Configurations:\n")
    print(f"{'Model':<20} {'FFT':<8} {'Hop':<8} {'Mels':<8} {'Notes'}")
    print("-" * 75)

    for model, config in configs.items():
        print(f"{model:<20} {config['n_fft']:<8} {config['hop_length']:<8} "
              f"{config['n_mels']:<8} {config['notes']}")

    print("\n" + "="*60)
    print("KEY CONFIGURATIONS")
    print("="*60)
    print("""
1. Whisper (Most Popular):
   - 80 Mel bands
   - 10ms hop length (low latency)
   - 16kHz sample rate
   - Log-Mel spectrogram input

2. Modern Trends:
   - 80 Mel bands (standard)
   - 10-20ms hop length
   - 16kHz sample rate
   - Sometimes direct waveform input

3. Classical ASR:
   - 13-26 MFCCs
   - 10-25ms frame, 10ms hop
   - Delta and delta-delta features

4. For Voice Engines:
   - Use 80 Mel bands (Whisper compatible)
   - 10ms hop for real-time
   - Convert to dB scale
   - Normalize to [-1, 1] or [0, 1]
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 1, STEP 4: MEL SPECTROGRAMS")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding perceptual frequency (Mel scale)")
    print("2. Mel filterbanks and their construction")
    print("3. Computing Mel spectrograms")
    print("4. MFCCs as compact representations")
    print("5. Configurations for modern ASR models")

    # Run demonstrations
    demonstrate_mel_scale()
    demonstrate_mel_filterbank()
    demonstrate_mel_spectrogram()
    compare_mel_resolutions()
    demonstrate_mfcc()
    mel_spectrogram_for_asr()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. Mel Scale matches human perception:
   - Linear at low frequencies (<1kHz)
   - Logarithmic at high frequencies (>1kHz)
   - Formula: mel = 2595 * log10(1 + f/700)

2. Mel Filterbanks:
   - Triangular, overlapping filters
   - More filters at low frequencies
   - Fewer filters at high frequencies
   - Typical: 40-128 filters

3. Mel Spectrogram Pipeline:
   - Compute STFT
   - Apply Mel filterbank
   - Convert to power/dB scale
   - Input for modern ASR models

4. MFCCs (Classical):
   - Apply DCT to Mel spectrogram
   - Keep 13-40 coefficients
   - Used in older ASR systems

5. Modern ASR (Whisper, Wav2Vec2):
   - Use Mel spectrograms directly (80 bands)
   - No MFCCs needed
   - End-to-end neural networks

6. For Voice Engines:
   - 80 Mel bands (Whisper standard)
   - Log-Mel spectrogram (dB scale)
   - 10-20ms hop length
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
