"""
Step 1: Audio Basics and Formats
==================================
Learn fundamental concepts of digital audio:
- Sample rate (Hz): Number of samples per second
- Bit depth (bits): Number of bits per sample (dynamic range)
- Channels: Mono (1) vs Stereo (2)
- Audio formats: PCM, WAV

Key Concepts:
- Sample Rate: Common rates are 8kHz (phone), 16kHz (voice), 44.1kHz (CD), 48kHz (video)
- Bit Depth: 8-bit (256 levels), 16-bit (65,536 levels), 24-bit (16.7M levels)
- Duration = Number of Samples / Sample Rate
- File Size = Sample Rate × Bit Depth × Channels × Duration / 8 (bytes)
"""

import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.5):
    """
    Generate a sine wave tone.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Samples per second
        amplitude: Wave amplitude (0.0 to 1.0)

    Returns:
        numpy array of audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave_data


def save_wav_file(filename, audio_data, sample_rate, bit_depth=16):
    """
    Save audio data as WAV file.

    Args:
        filename: Output file path
        audio_data: Audio samples (numpy array, values between -1 and 1)
        sample_rate: Sample rate in Hz
        bit_depth: Bits per sample (8 or 16)
    """
    # Normalize and convert to appropriate integer type
    if bit_depth == 16:
        # Convert to 16-bit PCM
        audio_data = np.int16(audio_data * 32767)
        sample_width = 2
    elif bit_depth == 8:
        # Convert to 8-bit PCM (unsigned)
        audio_data = np.uint8((audio_data + 1) * 127.5)
        sample_width = 1
    else:
        raise ValueError("Bit depth must be 8 or 16")

    # Write WAV file
    with wave.open(str(filename), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    print(f"✓ Saved: {filename}")


def read_wav_file(filename):
    """
    Read and parse WAV file.

    Returns:
        tuple: (audio_data, sample_rate, bit_depth, channels, duration)
    """
    with wave.open(str(filename), 'r') as wav_file:
        # Get WAV file parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Read audio data
        audio_bytes = wav_file.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 2:  # 16-bit
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            bit_depth = 16
        elif sample_width == 1:  # 8-bit
            audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
            audio_data = (audio_data.astype(np.float32) / 127.5) - 1.0
            bit_depth = 8
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        duration = n_frames / sample_rate

        return audio_data, sample_rate, bit_depth, channels, duration


def visualize_audio(audio_data, sample_rate, title="Audio Waveform"):
    """Visualize audio waveform."""
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio_data, linewidth=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    return plt


def compare_bit_depths():
    """Compare different bit depths."""
    frequency = 440  # A4 note
    duration = 0.5
    sample_rate = 16000

    # Generate sine wave
    audio = generate_sine_wave(frequency, duration, sample_rate)

    # Save in different bit depths
    save_wav_file(OUTPUT_DIR / "test_16bit.wav", audio, sample_rate, bit_depth=16)
    save_wav_file(OUTPUT_DIR / "test_8bit.wav", audio, sample_rate, bit_depth=8)

    # Read and compare
    audio_16bit, sr_16, bd_16, ch_16, dur_16 = read_wav_file(OUTPUT_DIR / "test_16bit.wav")
    audio_8bit, sr_8, bd_8, ch_8, dur_8 = read_wav_file(OUTPUT_DIR / "test_8bit.wav")

    print("\n" + "="*60)
    print("BIT DEPTH COMPARISON")
    print("="*60)
    print(f"\n16-bit Audio:")
    print(f"  - Bit Depth: {bd_16} bits")
    print(f"  - Dynamic Range: {20 * np.log10(2**bd_16):.1f} dB")
    print(f"  - Quantization Levels: {2**bd_16:,}")
    print(f"  - Sample Range: {audio_16bit.min():.6f} to {audio_16bit.max():.6f}")

    print(f"\n8-bit Audio:")
    print(f"  - Bit Depth: {bd_8} bits")
    print(f"  - Dynamic Range: {20 * np.log10(2**bd_8):.1f} dB")
    print(f"  - Quantization Levels: {2**bd_8:,}")
    print(f"  - Sample Range: {audio_8bit.min():.6f} to {audio_8bit.max():.6f}")

    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    time_16 = np.linspace(0, dur_16, len(audio_16bit))
    time_8 = np.linspace(0, dur_8, len(audio_8bit))

    ax1.plot(time_16[:1000], audio_16bit[:1000], linewidth=1, label='16-bit')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('16-bit Audio (High Quality)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(time_8[:1000], audio_8bit[:1000], linewidth=1, label='8-bit', color='orange')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('8-bit Audio (Lower Quality - Notice Quantization)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bit_depth_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {OUTPUT_DIR / 'bit_depth_comparison.png'}")


def compare_sample_rates():
    """Compare different sample rates."""
    frequency = 440
    duration = 1.0
    sample_rates = [8000, 16000, 44100]

    print("\n" + "="*60)
    print("SAMPLE RATE COMPARISON")
    print("="*60)

    fig, axes = plt.subplots(len(sample_rates), 1, figsize=(12, 10))

    for idx, sr in enumerate(sample_rates):
        # Generate audio
        audio = generate_sine_wave(frequency, duration, sr)
        filename = OUTPUT_DIR / f"test_{sr}Hz.wav"
        save_wav_file(filename, audio, sr, bit_depth=16)

        # Read back
        audio_read, sr_read, bd, ch, dur = read_wav_file(filename)

        # Calculate file size
        import os
        file_size = os.path.getsize(filename)

        print(f"\n{sr} Hz Audio:")
        print(f"  - Sample Rate: {sr_read:,} Hz")
        print(f"  - Number of Samples: {len(audio_read):,}")
        print(f"  - Duration: {dur:.3f} seconds")
        print(f"  - File Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        print(f"  - Nyquist Frequency: {sr_read/2:,} Hz (max reproducible frequency)")

        # Visualize
        time = np.linspace(0, dur, len(audio_read))
        axes[idx].plot(time[:500], audio_read[:500], linewidth=1)
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'{sr} Hz Sample Rate ({len(audio_read):,} samples)')
        axes[idx].grid(True, alpha=0.3)

        if idx == len(sample_rates) - 1:
            axes[idx].set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_rate_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {OUTPUT_DIR / 'sample_rate_comparison.png'}")


def demonstrate_nyquist_theorem():
    """Demonstrate Nyquist-Shannon sampling theorem."""
    print("\n" + "="*60)
    print("NYQUIST-SHANNON SAMPLING THEOREM")
    print("="*60)
    print("\nTheorem: To perfectly reconstruct a signal, the sample rate must be")
    print("at least 2× the highest frequency in the signal (Nyquist rate).")

    duration = 0.1
    high_freq = 4000  # 4 kHz tone

    # Good sampling: 16 kHz (2x Nyquist)
    sr_good = 16000
    audio_good = generate_sine_wave(high_freq, duration, sr_good)

    # Marginal sampling: 8 kHz (exactly Nyquist)
    sr_marginal = 8000
    audio_marginal = generate_sine_wave(high_freq, duration, sr_marginal)

    # Bad sampling: 6 kHz (below Nyquist - aliasing occurs)
    sr_bad = 6000
    audio_bad = generate_sine_wave(high_freq, duration, sr_bad)

    print(f"\nTest Signal: {high_freq} Hz sine wave")
    print(f"Nyquist Rate: {2 * high_freq} Hz (minimum)")
    print(f"\n✓ Good: {sr_good} Hz sample rate (2× Nyquist)")
    print(f"⚠ Marginal: {sr_marginal} Hz sample rate (exactly Nyquist)")
    print(f"✗ Bad: {sr_bad} Hz sample rate (below Nyquist - ALIASING!)")

    # Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    t_good = np.linspace(0, duration, len(audio_good))
    t_marginal = np.linspace(0, duration, len(audio_marginal))
    t_bad = np.linspace(0, duration, len(audio_bad))

    ax1.plot(t_good[:200], audio_good[:200], 'b.-', markersize=3, label=f'{sr_good} Hz')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'✓ Good Sampling: {sr_good} Hz (Clean Signal)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t_marginal[:100], audio_marginal[:100], 'g.-', markersize=4, label=f'{sr_marginal} Hz')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'⚠ Marginal: {sr_marginal} Hz (Exactly Nyquist)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(t_bad[:80], audio_bad[:80], 'r.-', markersize=5, label=f'{sr_bad} Hz')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title(f'✗ Bad Sampling: {sr_bad} Hz (Aliasing - Signal Distorted!)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nyquist_theorem.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {OUTPUT_DIR / 'nyquist_theorem.png'}")


def analyze_audio_properties():
    """Analyze properties of digital audio."""
    print("\n" + "="*60)
    print("AUDIO FILE SIZE CALCULATION")
    print("="*60)

    sample_rates = [8000, 16000, 44100, 48000]
    bit_depths = [8, 16]
    channels_list = [1, 2]
    duration = 60  # 1 minute

    print(f"\nFor {duration} seconds of audio:")
    print(f"\n{'Sample Rate':<12} {'Bit Depth':<12} {'Channels':<10} {'File Size':<15} {'Quality'}")
    print("-" * 70)

    for sr in sample_rates:
        for bd in bit_depths:
            for ch in channels_list:
                # File size = sample_rate × bit_depth × channels × duration / 8
                file_size_bytes = sr * (bd / 8) * ch * duration
                file_size_kb = file_size_bytes / 1024
                file_size_mb = file_size_kb / 1024

                channel_str = "Mono" if ch == 1 else "Stereo"

                # Determine quality
                if sr >= 44100 and bd == 16:
                    quality = "CD Quality"
                elif sr >= 16000 and bd == 16:
                    quality = "Voice (Good)"
                elif sr == 8000:
                    quality = "Telephone"
                else:
                    quality = "Low"

                print(f"{sr:<12} {bd:<12} {channel_str:<10} {file_size_mb:>6.2f} MB      {quality}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 1, STEP 1: AUDIO BASICS AND FORMATS")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding sample rate, bit depth, and channels")
    print("2. Working with PCM and WAV formats")
    print("3. Nyquist-Shannon sampling theorem")
    print("4. Audio file size calculations")

    # Run demonstrations
    compare_bit_depths()
    compare_sample_rates()
    demonstrate_nyquist_theorem()
    analyze_audio_properties()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. Sample Rate determines the highest frequency you can capture
   - Higher sample rate = better quality but larger files
   - Must be at least 2× highest frequency (Nyquist theorem)

2. Bit Depth determines dynamic range (difference between loudest and quietest)
   - 16-bit = 96 dB dynamic range (standard for voice)
   - 8-bit = 48 dB dynamic range (lower quality)

3. Channels: Mono (1) for voice, Stereo (2) for music

4. Common Configurations:
   - Telephone: 8 kHz, 8-bit, mono
   - Voice AI: 16 kHz, 16-bit, mono
   - CD Audio: 44.1 kHz, 16-bit, stereo
   - Professional: 48 kHz, 24-bit, stereo

5. WAV format = Header + PCM data (uncompressed)
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
