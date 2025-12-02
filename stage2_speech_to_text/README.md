# Stage 2: Speech-to-Text (ASR)

Welcome to Stage 2! Now that you understand audio fundamentals from Stage 1, you'll learn how modern Automatic Speech Recognition (ASR) systems work and build practical applications.

## 🎯 Learning Objectives

By the end of this stage, you will understand:

1. **ASR Evolution** - From HMM-GMM to Transformers
2. **CTC** - How to handle unknown alignments
3. **Sequence-to-Sequence** - Attention mechanisms
4. **Wav2Vec2** - Self-supervised learning
5. **Whisper** - State-of-the-art ASR
6. **Streaming ASR** - Real-time transcription

## 📚 Prerequisites

- Completed Stage 1 (Audio + DSP fundamentals)
- Understanding of:
  - Mel spectrograms
  - Audio framing
  - Basic neural networks (helpful)

## 🚀 Setup

### 1. Install Dependencies

```bash
cd stage2_speech_to_text
pip install -r requirements.txt
```

### 2. (Optional) GPU Support

For faster model inference:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📖 Step-by-Step Guide

### Step 1: ASR Fundamentals and Architectures

**File:** `step1_asr_fundamentals.py`

**What You'll Learn:**
- ASR evolution from 1952 to 2022
- Comparing architectures (HMM-GMM, RNN, CNN, Transformer, Wav2Vec2, Whisper)
- Classical vs Modern ASR pipelines
- Evaluation metrics (WER, CER, RTF)

**Run:**
```bash
python step1_asr_fundamentals.py
```

**Expected Outputs:**
- `outputs/asr_evolution.png` - Timeline visualization

**Key Concepts:**
- WER (Word Error Rate): Primary metric
- HMM-GMM: Classical approach (70-80% accuracy)
- Transformers: Modern approach (95-99% accuracy)
- End-to-end vs component-based systems

---

### Step 2: CTC (Connectionist Temporal Classification)

**File:** `step2_ctc_classification.py`

**What You'll Learn:**
- The alignment problem in ASR
- How CTC solves it with blank tokens
- CTC collapse function
- Greedy vs beam search decoding
- When to use CTC

**Run:**
```bash
python step2_ctc_classification.py
```

**Expected Outputs:**
- `outputs/ctc_alignment_problem.png` - Visualizing the problem
- `outputs/ctc_paths.png` - Multiple valid CTC paths
- `outputs/ctc_greedy_decoding.png` - Decoding demonstration

**Key Concepts:**
- Blank token (-) for handling alignment
- Many-to-one mapping (multiple paths → same text)
- CTC Loss: Sum over all valid alignments
- Used in DeepSpeech, Wav2Vec2

---

### Step 3: Sequence-to-Sequence with Attention *(Coming Soon)*

**File:** `step3_seq2seq_attention.py`

**What You'll Learn:**
- Encoder-Decoder architecture
- Attention mechanisms (Bahdanau, Luong)
- Teacher forcing
- Inference strategies

---

### Step 4: Wav2Vec2 Self-Supervised Learning *(Coming Soon)*

**File:** `step4_wav2vec2.py`

**What You'll Learn:**
- Self-supervised pre-training
- Using HuggingFace Transformers
- Fine-tuning on custom data
- Practical ASR implementation

---

### Step 5: Whisper Architecture Deep Dive *(Coming Soon)*

**File:** `step5_whisper.py`

**What You'll Learn:**
- Whisper architecture analysis
- Multi-task learning (transcription, translation, detection)
- Using OpenAI Whisper
- Language detection and translation

---

### Step 6: Real-time Streaming ASR *(Coming Soon)*

**File:** `step6_streaming_asr.py`

**What You'll Learn:**
- Streaming vs batch processing
- VAD integration
- Buffer management
- Low-latency optimization
- Building production ASR

---

## 🎓 Learning Path

**Recommended Order:**
1. Complete all Stage 1 steps first
2. Start with Step 1 (ASR Fundamentals)
3. Progress through Steps 2-6 sequentially
4. Hands-on practice with Steps 4-6 (practical implementations)

**Time Estimate:**
- Step 1: 1 hour (theory)
- Step 2: 1-2 hours (CTC understanding)
- Step 3: 1-2 hours (attention mechanisms)
- Step 4: 2-3 hours (Wav2Vec2 hands-on)
- Step 5: 2-3 hours (Whisper hands-on)
- Step 6: 3-4 hours (streaming implementation)
- **Total: 10-15 hours**

**Tips:**
- Steps 1-3 are foundational theory
- Steps 4-6 are hands-on practical
- Run each script completely before moving to the next
- Experiment with different models and parameters

## 📊 Output Files

All scripts save their outputs to:

```
outputs/
├── *.png              # Visualizations
├── *.wav              # Audio samples (if generated)
└── *.txt              # Transcriptions
```

Models will be downloaded to:
```
models/
├── wav2vec2/          # Wav2Vec2 models
├── whisper/           # Whisper models
└── custom/            # Fine-tuned models
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error: No module named 'transformers'**
   ```bash
   pip install transformers
   ```

2. **CUDA out of memory**
   - Use smaller model size
   - Reduce batch size
   - Use CPU instead: `device='cpu'`

3. **Slow inference**
   - Use GPU if available
   - Try `faster-whisper` for Whisper
   - Use quantized models

4. **Model download fails**
   - Check internet connection
   - Manually download from HuggingFace
   - Set `cache_dir` parameter

## 📝 Key Takeaways

After completing Stage 2, you should understand:

### ASR Evolution
- ✅ HMM-GMM → RNN → CNN → Transformer progression
- ✅ Why modern systems are end-to-end
- ✅ Trade-offs between architectures

### Core Concepts
- ✅ CTC for handling alignment
- ✅ Attention mechanisms
- ✅ Self-supervised learning (Wav2Vec2)
- ✅ Multi-task learning (Whisper)

### Practical Skills
- ✅ Using HuggingFace Transformers
- ✅ Running Whisper for transcription
- ✅ Fine-tuning models on custom data
- ✅ Building streaming ASR systems

### Evaluation
- ✅ Calculating WER/CER
- ✅ Measuring latency (RTF)
- ✅ Benchmarking models

## 🎯 Practice Exercises

1. **Compare Models:**
   - Run the same audio through Wav2Vec2 and Whisper
   - Compare WER, speed, and quality

2. **Fine-tune Wav2Vec2:**
   - Collect 10-20 audio samples in your domain
   - Fine-tune Wav2Vec2
   - Measure improvement

3. **Build a Voice Recorder:**
   - Combine Stage 1 VAD with Stage 2 ASR
   - Record → Detect speech → Transcribe
   - Display real-time transcription

4. **Multi-language ASR:**
   - Test Whisper on different languages
   - Compare accuracy across languages
   - Try language detection

## 📖 Additional Resources

### Papers
- **CTC:** "Connectionist Temporal Classification" (Graves et al., 2006)
- **Attention:** "Attention is All You Need" (Vaswani et al., 2017)
- **Wav2Vec2:** "wav2vec 2.0" (Baevski et al., 2020)
- **Whisper:** "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

### Code & Models
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Wav2Vec2 Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads)

### Tutorials
- [HuggingFace ASR Course](https://huggingface.co/learn/audio-course)
- [Whisper Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [PyTorch Speech Recognition](https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html)

## ✅ Completion Checklist

Before moving to Stage 3, ensure you can:

- [ ] Explain the evolution of ASR architectures
- [ ] Describe how CTC solves the alignment problem
- [ ] Understand attention mechanisms
- [ ] Use HuggingFace Transformers for ASR
- [ ] Run Whisper for transcription
- [ ] Calculate WER for model evaluation
- [ ] Implement basic streaming ASR
- [ ] Choose appropriate models for different use cases

## 🚀 Next Steps

Once you've completed Stage 2:

1. **Review** - Go through visualizations and outputs
2. **Experiment** - Try different models and datasets
3. **Move to Stage 3** - Text-to-Speech (TTS)

**Stage 3 Preview:**
- Mel spectrogram to waveform generation
- Vocoders (WaveGlow, HiFi-GAN)
- Modern TTS (Tacotron 2, FastSpeech2)
- Voice cloning (XTTS, Bark)
- Building TTS services

---

## 📧 Questions or Issues?

If you encounter problems:
1. Check the troubleshooting section
2. Review console output from scripts
3. Examine generated visualizations
4. Try with smaller models first

---

## 🎉 Congratulations!

You're building a comprehensive understanding of modern ASR! This knowledge is essential for:
- Building voice assistants
- Transcription services
- Voice search
- Accessibility tools
- Any voice-powered application

**Keep going! Stage 3 awaits! 🚀**
