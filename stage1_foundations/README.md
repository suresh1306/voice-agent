# Stage 1: Foundations (Audio + DSP)

Welcome to Stage 1 of your Voice Engine learning journey! This stage covers the fundamental concepts of digital audio processing and Digital Signal Processing (DSP) that are essential for building any voice-powered application.

## 🎯 Learning Objectives

By the end of this stage, you will understand:

1. **Digital Audio Fundamentals**
   - Sample rate, bit depth, and channels
   - PCM and WAV file formats
   - Nyquist-Shannon sampling theorem
   - Audio file size calculations

2. **Audio Framing and Windowing**
   - How to split audio into processable frames
   - Overlapping frames and hop size
   - Window functions (Hann, Hamming, Blackman)
   - Overlap-add reconstruction

3. **Frequency Analysis**
   - FFT (Fast Fourier Transform)
   - STFT (Short-Time Fourier Transform)
   - Power spectrograms
   - Time-frequency trade-offs

4. **Perceptual Audio Features**
   - Mel scale (perceptual frequency)
   - Mel filterbanks
   - Mel spectrograms
   - MFCCs (Mel-Frequency Cepstral Coefficients)

5. **Noise Reduction**
   - Spectral subtraction
   - Signal-to-Noise Ratio (SNR)
   - Real-time noise reduction
   - Trade-offs and best practices

6. **Voice Activity Detection (VAD)**
   - Energy-based VAD
   - Spectral VAD
   - WebRTC VAD
   - Smoothing and post-processing

## 📚 Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with NumPy (helpful but not required)

## 🚀 Setup

### 1. Install Dependencies

```bash
cd stage1_foundations
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import numpy, scipy, matplotlib, librosa, webrtcvad, soundfile; print('All dependencies installed successfully!')"
```

## 📖 Step-by-Step Guide

### Step 1: Audio Basics and Formats

**File:** `step1_audio_basics.py`

**What You'll Learn:**
- How digital audio works (sample rate, bit depth, channels)
- Different audio formats (PCM, WAV)
- Nyquist-Shannon sampling theorem
- Audio file size calculations

**Run:**
```bash
python step1_audio_basics.py
```

**Expected Outputs:**
- `outputs/test_16bit.wav` - 16-bit audio sample
- `outputs/test_8bit.wav` - 8-bit audio sample
- `outputs/bit_depth_comparison.png` - Visual comparison of bit depths
- `outputs/sample_rate_comparison.png` - Visual comparison of sample rates
- `outputs/nyquist_theorem.png` - Demonstration of Nyquist theorem

**Key Concepts:**
- Sample Rate = Number of samples per second (Hz)
- Bit Depth = Number of bits per sample (dynamic range)
- Nyquist Rate = 2× highest frequency (minimum for perfect reconstruction)
- Common configurations: 16kHz/16-bit for voice, 44.1kHz/16-bit for music

---

### Step 2: Audio Frames and Windows

**File:** `step2_audio_frames_windows.py`

**What You'll Learn:**
- How to split audio into frames for processing
- Overlapping frames and hop size
- Window functions to reduce spectral leakage
- Overlap-add reconstruction

**Run:**
```bash
python step2_audio_frames_windows.py
```

**Expected Outputs:**
- `outputs/audio_framing.png` - Visualization of different framing strategies
- `outputs/window_functions.png` - Comparison of window functions
- `outputs/windowing_effect.png` - Effect of windowing on signals
- `outputs/overlap_add.png` - Overlap-add reconstruction demonstration

**Key Concepts:**
- Frame Size: Typically 10-30ms for voice (160-480 samples at 16kHz)
- Hop Size: Distance between frames (often 50% of frame size)
- Window Functions: Smooth frame edges (Hann, Hamming, Blackman)
- Overlap-Add: Reconstruct signal from overlapping frames

---

### Step 3: FFT and STFT

**File:** `step3_fft_stft.py`

**What You'll Learn:**
- FFT (Fast Fourier Transform) - time to frequency domain
- STFT (Short-Time Fourier Transform) - time-frequency analysis
- Power spectrograms
- Frequency resolution vs time resolution trade-offs

**Run:**
```bash
python step3_fft_stft.py
```

**Expected Outputs:**
- `outputs/fft_basics.png` - Basic FFT analysis
- `outputs/fft_resolution.png` - Frequency resolution demonstration
- `outputs/stft_basic.png` - STFT spectrogram
- `outputs/stft_comparison.png` - Different STFT parameters
- `outputs/power_spectrogram.png` - Power spectrogram representations
- `outputs/inverse_stft.png` - STFT reconstruction

**Key Concepts:**
- FFT converts time-domain signal to frequency domain
- STFT = FFT applied to overlapping frames
- Spectrogram = Visual representation of time-frequency content
- Frequency Resolution = Sample Rate / FFT Size
- Trade-off: Longer frames = better frequency resolution, poorer time resolution

---

### Step 4: Mel Spectrograms

**File:** `step4_mel_spectrograms.py`

**What You'll Learn:**
- Mel scale (perceptual frequency representation)
- Mel filterbanks construction
- Computing Mel spectrograms
- MFCCs (compact representation)
- Configurations for modern ASR models

**Run:**
```bash
python step4_mel_spectrograms.py
```

**Expected Outputs:**
- `outputs/mel_scale.png` - Mel scale transformation
- `outputs/mel_filterbank.png` - Mel filterbank visualization
- `outputs/mel_spectrogram_comparison.png` - Mel vs regular spectrogram
- `outputs/mel_spectrogram_resolutions.png` - Different Mel band resolutions
- `outputs/mfcc_analysis.png` - MFCC visualization

**Key Concepts:**
- Mel Scale: Perceptual frequency scale (mel = 2595 × log₁₀(1 + f/700))
- Mel Filterbanks: Triangular, overlapping filters that mimic human hearing
- Mel Spectrograms: Input for modern ASR models (Whisper, Wav2Vec2)
- 80 Mel bands is standard for Whisper
- MFCCs: Classical compressed representation (13-40 coefficients)

---

### Step 5: Noise Reduction

**File:** `step5_noise_reduction.py`

**What You'll Learn:**
- Spectral subtraction technique
- Signal-to-Noise Ratio (SNR) calculation
- Real-time frame-based noise reduction
- Trade-offs between noise removal and distortion

**Run:**
```bash
python step5_noise_reduction.py
```

**Expected Outputs:**
- `outputs/clean_speech.wav` - Clean reference audio
- `outputs/noisy_speech.wav` - Noisy audio sample
- `outputs/denoised_moderate.wav` - Denoised audio
- `outputs/noise_reduction_comparison.png` - Waveform comparison
- `outputs/noise_reduction_spectrograms.png` - Spectral analysis
- `outputs/realtime_noise_reduction.png` - Real-time processing demo

**Key Concepts:**
- Spectral Subtraction: Estimate noise spectrum, subtract from signal
- Over-subtraction factor (α): Controls aggressiveness (1.0-3.0)
- Spectral floor (β): Prevents musical noise artifacts
- Real-time processing: Frame-based with 10-30ms latency
- Modern alternatives: RNNoise, DTLN (deep learning based)

---

### Step 6: Voice Activity Detection (VAD)

**File:** `step6_voice_activity_detection.py`

**What You'll Learn:**
- Energy-based VAD (simple threshold)
- Zero-Crossing Rate (ZCR)
- Spectral-based VAD
- WebRTC VAD (production-ready)
- Smoothing techniques
- Integration with voice engines

**Run:**
```bash
python step6_voice_activity_detection.py
```

**Expected Outputs:**
- `outputs/vad_comparison.png` - Comparison of VAD algorithms
- `outputs/vad_smoothing.png` - Effect of smoothing on VAD decisions

**Key Concepts:**
- VAD detects speech vs silence/noise
- Critical for: Turn-taking, barge-in, computational efficiency
- Energy-based: Fast, works in quiet environments
- WebRTC VAD: Production-ready, good balance
- Deep Learning: Silero VAD, Pyannote-audio (best quality)
- Smoothing essential to reduce false triggers

---

## 🎓 Learning Path

**Recommended Order:**
1. Start with Step 1 (Audio Basics) - Foundation
2. Progress through Steps 2-6 sequentially
3. Experiment with parameters in each script
4. Review visualizations in `outputs/` folder
5. Read the console output carefully - it explains results

**Time Estimate:**
- Each step: 30-60 minutes
- Total for Stage 1: 4-6 hours

**Tips:**
- Run each script completely before moving to the next
- Examine the generated visualizations carefully
- Try modifying parameters to see their effects
- Read the "Key Takeaways" at the end of each script

## 📊 Output Files

All scripts save their outputs to the `outputs/` directory:

```
outputs/
├── *.png              # Visualizations
├── *.wav              # Audio samples
└── (generated by scripts)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error: No module named 'xyz'**
   ```bash
   pip install -r requirements.txt
   ```

2. **Matplotlib display issues**
   - Figures are automatically saved to `outputs/`
   - If display doesn't work, check the saved PNG files

3. **WebRTC VAD errors**
   - Ensure audio is 16kHz, 16-bit PCM
   - Frame duration must be 10, 20, or 30ms

4. **Memory issues with large audio files**
   - Scripts use synthetic audio, should be lightweight
   - If issues persist, reduce duration parameters

## 📝 Key Takeaways

After completing Stage 1, you should understand:

### Audio Fundamentals
- ✅ Sample rate determines frequency range (Nyquist theorem)
- ✅ Bit depth determines dynamic range
- ✅ 16kHz/16-bit is standard for voice applications

### Frame Processing
- ✅ Audio is processed in frames (10-30ms typical)
- ✅ Overlapping frames with window functions reduce artifacts
- ✅ Overlap-add reconstructs the original signal

### Frequency Analysis
- ✅ FFT converts time → frequency domain
- ✅ STFT provides time-frequency representation
- ✅ Trade-off between time and frequency resolution

### Perceptual Features
- ✅ Mel scale matches human hearing (logarithmic)
- ✅ Mel spectrograms are input for modern ASR
- ✅ 80 Mel bands is standard (Whisper configuration)

### Preprocessing
- ✅ Noise reduction improves ASR accuracy
- ✅ VAD reduces computational load by 50-80%
- ✅ Both critical for production voice engines

## 🎯 Practice Exercises

1. **Modify Parameters:**
   - Change frame sizes in Step 2, observe effects
   - Adjust FFT size in Step 3, see frequency resolution changes
   - Try different noise reduction aggressiveness in Step 5

2. **Real Audio:**
   - Record your own voice (1-2 seconds)
   - Save as WAV file
   - Process with the scripts from Steps 3-6

3. **Combine Techniques:**
   - Load audio → Apply noise reduction → Run VAD → Extract Mel spectrogram
   - This is the typical preprocessing pipeline!

## 📖 Additional Resources

### Books
- "Understanding Digital Signal Processing" by Richard Lyons
- "Fundamentals of Speech Recognition" by Lawrence Rabiner

### Papers
- "Mel Frequency Cepstral Coefficients for Music Modeling" (Beth Logan, 2000)
- "Spectral Subtraction Based on Minimum Statistics" (Rainer Martin, 2001)

### Online Resources
- [librosa documentation](https://librosa.org/doc/latest/index.html)
- [Distill.pub: Audio Processing Intro](https://distill.pub)
- [WebRTC VAD implementation](https://github.com/wiseman/py-webrtcvad)

## ✅ Completion Checklist

Before moving to Stage 2, ensure you can:

- [ ] Explain sample rate, bit depth, and Nyquist theorem
- [ ] Frame audio with overlapping windows
- [ ] Compute and interpret FFT and STFT
- [ ] Generate Mel spectrograms
- [ ] Apply basic noise reduction
- [ ] Implement VAD for speech detection
- [ ] Understand the full preprocessing pipeline

## 🚀 Next Steps

Once you've completed Stage 1:

1. **Review** - Go through the visualizations one more time
2. **Experiment** - Try the practice exercises above
3. **Move to Stage 2** - Speech-to-Text (ASR)

**Stage 2 Preview:**
- RNN/CNN/Transformer architectures for audio
- CTC (Connectionist Temporal Classification)
- Whisper architecture deep-dive
- Real-time streaming ASR
- Building your first ASR system

---

## 📧 Questions or Issues?

If you encounter any problems or have questions:
1. Check the troubleshooting section above
2. Review the console output from the scripts
3. Examine the generated visualizations
4. Experiment with different parameters

---

## 🎉 Congratulations!

You're building a strong foundation in audio processing! This knowledge is essential for:
- Speech Recognition (ASR)
- Text-to-Speech (TTS)
- Voice Activity Detection
- Audio Classification
- Any Voice AI application

**Keep going! Stage 2 awaits! 🚀**
