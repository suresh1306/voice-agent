"""
Step 2: Audio Frames, Chunks, and Windows
==========================================
Learn how to process audio in segments:
- Frames: Small chunks of audio samples
- Overlapping frames for smooth processing
- Window functions to reduce spectral leakage
- Real-time audio streaming concepts

Key Concepts:
- Frame Size: Typically 10-30ms for voice (160-480 samples at 16kHz)
- Hop Size (Step): Distance between frames (often 50% of frame size for overlap)
- Window Functions: Hann, Hamming, Blackman to smooth frame edges
- Overlap-Add (OLA): Reconstruct signal from overlapping frames
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import wave

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_test_signal(duration=2.0, sample_rate=16000):
    """Generate a test signal with multiple frequency components."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a signal with varying frequencies
    signal = (
        0.5 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.3 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
        0.2 * np.sin(2 * np.pi * 1500 * t)   # High frequency
    )

    return signal, sample_rate


def frame_audio(audio, frame_size, hop_size):
    """
    Split audio into overlapping frames.

    Args:
        audio: Audio signal (numpy array)
        frame_size: Number of samples per frame
        hop_size: Number of samples to move between frames

    Returns:
        2D array where each row is a frame
    """
    num_frames = 1 + (len(audio) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end <= len(audio):
            frames[i] = audio[start:end]

    return frames


def apply_window(frames, window_type='hann'):
    """
    Apply window function to frames.

    Args:
        frames: 2D array of audio frames
        window_type: 'hann', 'hamming', 'blackman', 'rectangular'

    Returns:
        Windowed frames
    """
    frame_size = frames.shape[1]

    # Create window function
    if window_type == 'hann':
        window = np.hanning(frame_size)
    elif window_type == 'hamming':
        window = np.hamming(frame_size)
    elif window_type == 'blackman':
        window = np.blackman(frame_size)
    elif window_type == 'rectangular':
        window = np.ones(frame_size)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    # Apply window to each frame
    windowed_frames = frames * window[np.newaxis, :]

    return windowed_frames, window


def overlap_add(frames, hop_size):
    """
    Reconstruct signal from overlapping frames using overlap-add method.

    Args:
        frames: 2D array of audio frames
        hop_size: Hop size used during framing

    Returns:
        Reconstructed audio signal
    """
    num_frames, frame_size = frames.shape
    output_length = (num_frames - 1) * hop_size + frame_size
    output = np.zeros(output_length)

    for i, frame in enumerate(frames):
        start = i * hop_size
        output[start:start + frame_size] += frame

    return output


def demonstrate_framing():
    """Demonstrate audio framing with different parameters."""
    print("\n" + "="*60)
    print("AUDIO FRAMING")
    print("="*60)

    # Generate test signal
    duration = 0.5
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)

    # Different frame configurations
    configs = [
        {"frame_ms": 20, "overlap_percent": 0},
        {"frame_ms": 20, "overlap_percent": 50},
        {"frame_ms": 30, "overlap_percent": 50},
    ]

    fig, axes = plt.subplots(len(configs) + 1, 1, figsize=(14, 10))

    # Plot original signal
    axes[0].plot(t[:800], signal[:800], 'b-', linewidth=1)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal (440 Hz sine wave)')
    axes[0].grid(True, alpha=0.3)

    for idx, config in enumerate(configs):
        frame_ms = config['frame_ms']
        overlap_percent = config['overlap_percent']

        frame_size = int(sample_rate * frame_ms / 1000)
        hop_size = int(frame_size * (1 - overlap_percent / 100))

        frames = frame_audio(signal, frame_size, hop_size)

        print(f"\nConfiguration {idx + 1}:")
        print(f"  Frame Size: {frame_ms} ms ({frame_size} samples)")
        print(f"  Overlap: {overlap_percent}%")
        print(f"  Hop Size: {hop_size} samples")
        print(f"  Number of Frames: {len(frames)}")

        # Visualize frames
        ax = axes[idx + 1]
        time_axis = np.arange(len(signal)) / sample_rate

        for i, frame in enumerate(frames[:10]):  # Show first 10 frames
            start = i * hop_size
            end = start + frame_size
            if end <= len(signal):
                frame_time = time_axis[start:end]
                ax.plot(frame_time, frame, alpha=0.7, linewidth=0.8)

        ax.set_ylabel('Amplitude')
        ax.set_title(f'Frames: {frame_ms}ms, {overlap_percent}% overlap')
        ax.grid(True, alpha=0.3)

        if idx == len(configs) - 1:
            ax.set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "audio_framing.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'audio_framing.png'}")


def demonstrate_window_functions():
    """Demonstrate different window functions."""
    print("\n" + "="*60)
    print("WINDOW FUNCTIONS")
    print("="*60)
    print("\nWindow functions smooth the edges of frames to reduce")
    print("spectral leakage in frequency analysis.")

    frame_size = 512
    window_types = ['rectangular', 'hann', 'hamming', 'blackman']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, window_type in enumerate(window_types):
        # Create window
        if window_type == 'rectangular':
            window = np.ones(frame_size)
        elif window_type == 'hann':
            window = np.hanning(frame_size)
        elif window_type == 'hamming':
            window = np.hamming(frame_size)
        elif window_type == 'blackman':
            window = np.blackman(frame_size)

        # Calculate properties
        coherent_gain = np.mean(window)
        equivalent_noise_bw = frame_size * np.sum(window**2) / np.sum(window)**2

        print(f"\n{window_type.capitalize()} Window:")
        print(f"  Coherent Gain: {coherent_gain:.4f}")
        print(f"  Equivalent Noise Bandwidth: {equivalent_noise_bw:.4f} bins")

        # Plot time domain
        ax = axes[idx]
        ax.plot(window, linewidth=2)
        ax.set_title(f'{window_type.capitalize()} Window')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "window_functions.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'window_functions.png'}")


def demonstrate_window_effect():
    """Show the effect of windowing on a signal."""
    print("\n" + "="*60)
    print("WINDOWING EFFECT ON SIGNAL")
    print("="*60)

    # Create a signal
    frame_size = 512
    t = np.linspace(0, 1, frame_size, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    # Apply different windows
    windows = {
        'Original': np.ones(frame_size),
        'Hann': np.hanning(frame_size),
        'Hamming': np.hamming(frame_size),
        'Blackman': np.blackman(frame_size)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, window) in enumerate(windows.items()):
        windowed_signal = signal * window

        ax = axes[idx]
        ax.plot(t, signal, 'b--', alpha=0.5, label='Original Signal', linewidth=1)
        ax.plot(t, windowed_signal, 'r-', label='Windowed Signal', linewidth=2)
        ax.plot(t, window, 'g:', alpha=0.7, label='Window', linewidth=1.5)

        ax.set_title(f'{name}')
        ax.set_xlabel('Time (normalized)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "windowing_effect.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'windowing_effect.png'}")


def demonstrate_overlap_add():
    """Demonstrate overlap-add reconstruction."""
    print("\n" + "="*60)
    print("OVERLAP-ADD RECONSTRUCTION")
    print("="*60)

    # Generate test signal
    signal, sample_rate = generate_test_signal(duration=1.0, sample_rate=16000)

    # Frame parameters
    frame_ms = 25
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = frame_size // 2  # 50% overlap

    print(f"\nFrame Size: {frame_ms} ms ({frame_size} samples)")
    print(f"Hop Size: {hop_size} samples (50% overlap)")

    # Create frames and apply window
    frames = frame_audio(signal, frame_size, hop_size)
    windowed_frames, window = apply_window(frames, 'hann')

    # Reconstruct using overlap-add
    reconstructed = overlap_add(windowed_frames, hop_size)

    # Normalize (compensate for window overlap)
    # For 50% overlap with Hann window, the sum of windows = 1.0
    reconstructed = reconstructed[:len(signal)]

    # Calculate reconstruction error
    error = np.mean(np.abs(signal - reconstructed))
    max_error = np.max(np.abs(signal - reconstructed))

    print(f"\nReconstruction Quality:")
    print(f"  Mean Absolute Error: {error:.6f}")
    print(f"  Max Absolute Error: {max_error:.6f}")
    print(f"  Number of Frames: {len(frames)}")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original signal
    t = np.arange(len(signal)) / sample_rate
    axes[0].plot(t[:2000], signal[:2000], 'b-', linewidth=1, label='Original')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Reconstructed signal
    t_rec = np.arange(len(reconstructed)) / sample_rate
    axes[1].plot(t_rec[:2000], reconstructed[:2000], 'r-', linewidth=1, label='Reconstructed')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Reconstructed Signal (Overlap-Add)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Error
    error_signal = signal - reconstructed
    axes[2].plot(t[:2000], error_signal[:2000], 'g-', linewidth=1, label='Error')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Reconstruction Error (Mean: {error:.6f})')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overlap_add.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'overlap_add.png'}")


def streaming_frame_processing():
    """Simulate real-time streaming frame processing."""
    print("\n" + "="*60)
    print("STREAMING FRAME PROCESSING")
    print("="*60)
    print("\nSimulating real-time audio processing...")

    # Generate signal
    signal, sample_rate = generate_test_signal(duration=2.0, sample_rate=16000)

    # Frame parameters (typical for voice)
    frame_ms = 20
    frame_size = int(sample_rate * frame_ms / 1000)  # 320 samples at 16kHz
    hop_size = frame_size  # No overlap for real-time streaming

    print(f"\nStreaming Configuration:")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Frame Duration: {frame_ms} ms")
    print(f"  Frame Size: {frame_size} samples")
    print(f"  Latency per frame: {frame_ms} ms")

    # Simulate streaming
    num_frames = len(signal) // frame_size
    processed_frames = []

    print(f"\nProcessing {num_frames} frames:")

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        if end <= len(signal):
            frame = signal[start:end]

            # Simulate processing (e.g., feature extraction, VAD, etc.)
            frame_energy = np.mean(frame ** 2)
            frame_max = np.max(np.abs(frame))

            processed_frames.append(frame)

            if i < 5 or i >= num_frames - 5:  # Show first and last 5 frames
                print(f"  Frame {i:3d}: Energy={frame_energy:.6f}, Peak={frame_max:.4f}")
            elif i == 5:
                print(f"  ... (processing {num_frames - 10} more frames) ...")

    total_duration = len(signal) / sample_rate
    print(f"\nProcessed {num_frames} frames in {total_duration:.2f} seconds")
    print(f"Average latency: {frame_ms} ms per frame")
    print(f"Total samples processed: {num_frames * frame_size}")


def frame_size_recommendations():
    """Provide recommendations for frame sizes."""
    print("\n" + "="*60)
    print("FRAME SIZE RECOMMENDATIONS")
    print("="*60)

    sample_rate = 16000

    recommendations = [
        {"name": "Voice Activity Detection", "frame_ms": [10, 20, 30], "overlap": 0},
        {"name": "Speech Recognition", "frame_ms": [20, 25, 30], "overlap": 50},
        {"name": "Frequency Analysis", "frame_ms": [20, 25, 32], "overlap": 50},
        {"name": "Real-time Streaming", "frame_ms": [10, 20], "overlap": 0},
        {"name": "Audio Effects", "frame_ms": [10, 20], "overlap": 50},
    ]

    print(f"\nFor {sample_rate} Hz sample rate:\n")
    print(f"{'Application':<25} {'Frame Size':<20} {'Samples':<15} {'Overlap'}")
    print("-" * 75)

    for rec in recommendations:
        name = rec['name']
        frame_sizes = rec['frame_ms']
        overlap = rec['overlap']

        frame_str = "/".join([f"{f}ms" for f in frame_sizes])
        samples_str = "/".join([str(int(sample_rate * f / 1000)) for f in frame_sizes])

        print(f"{name:<25} {frame_str:<20} {samples_str:<15} {overlap}%")

    print("\n" + "="*60)
    print("KEY GUIDELINES")
    print("="*60)
    print("""
1. Shorter frames (10-20ms):
   - Lower latency
   - Better for real-time applications
   - Less frequency resolution

2. Longer frames (25-32ms):
   - Higher frequency resolution
   - Better for frequency analysis
   - Slightly higher latency

3. Overlap:
   - 0%: Real-time streaming, VAD
   - 50%: Most frequency analysis
   - 75%: Maximum smoothness (rarely used)

4. Power-of-2 sizes (256, 512, 1024):
   - Efficient FFT computation
   - Common: 512 samples ≈ 32ms at 16kHz
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 1, STEP 2: AUDIO FRAMES AND WINDOWS")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding audio framing for processing")
    print("2. Overlapping frames and hop size")
    print("3. Window functions and their effects")
    print("4. Overlap-add reconstruction")
    print("5. Real-time streaming concepts")

    # Run demonstrations
    demonstrate_framing()
    demonstrate_window_functions()
    demonstrate_window_effect()
    demonstrate_overlap_add()
    streaming_frame_processing()
    frame_size_recommendations()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. Framing breaks audio into processable chunks
   - Frame size: typically 10-30ms for voice
   - Hop size: determines overlap between frames

2. Window Functions reduce spectral leakage
   - Hann: Good general purpose window
   - Hamming: Better frequency resolution
   - Blackman: Best sidelobe suppression

3. Overlap-Add reconstructs signal from frames
   - 50% overlap is most common
   - Window normalization is crucial

4. For Voice Engines:
   - ASR: 20-25ms frames, 50% overlap
   - VAD: 10-20ms frames, 0% overlap
   - Streaming: minimize overlap for lower latency

5. Frame size trade-offs:
   - Smaller = lower latency, less freq resolution
   - Larger = higher freq resolution, more latency
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
