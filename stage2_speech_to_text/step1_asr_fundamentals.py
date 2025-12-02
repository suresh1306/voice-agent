"""
Step 1: ASR Fundamentals and Architectures
===========================================
Learn the basics of Automatic Speech Recognition:
- What is ASR and how does it work
- Evolution: HMM → RNN → CNN → Transformers
- Key components of ASR systems
- Acoustic vs Language Models
- Evaluation metrics (WER, CER)

Key Concepts:
- ASR = Audio → Text transcription
- Acoustic Model: Audio features → phonemes/characters
- Language Model: Improves predictions using linguistic context
- End-to-end models: Direct audio → text (no intermediate representations)
- WER (Word Error Rate): Primary evaluation metric
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def demonstrate_asr_history():
    """Demonstrate the evolution of ASR systems."""
    print("\n" + "="*60)
    print("ASR EVOLUTION: FROM HMM TO TRANSFORMERS")
    print("="*60)

    timeline = {
        "1952": {
            "name": "Bell Labs Audrey",
            "description": "Recognized digits 0-9 for single speaker",
            "accuracy": "~70%",
            "technology": "Pattern matching"
        },
        "1970s": {
            "name": "DARPA SUR",
            "description": "1000-word vocabulary",
            "accuracy": "~60%",
            "technology": "Template matching"
        },
        "1980s-2000s": {
            "name": "HMM-GMM Era",
            "description": "Hidden Markov Models + Gaussian Mixture Models",
            "accuracy": "~70-80%",
            "technology": "Statistical models"
        },
        "2010s": {
            "name": "Deep Learning Revolution",
            "description": "RNNs, LSTMs, CNNs",
            "accuracy": "~85-95%",
            "technology": "Neural networks"
        },
        "2017": {
            "name": "Transformer Era Begins",
            "description": "Attention is All You Need paper",
            "accuracy": "~95-98%",
            "technology": "Self-attention"
        },
        "2020": {
            "name": "Wav2Vec2",
            "description": "Self-supervised learning from unlabeled audio",
            "accuracy": "~95-98%",
            "technology": "Contrastive learning"
        },
        "2022": {
            "name": "Whisper",
            "description": "Large-scale multi-task training",
            "accuracy": "~96-99%",
            "technology": "Transformers + massive data"
        },
    }

    print("\nKey Milestones:\n")
    for year, info in timeline.items():
        print(f"{year}: {info['name']}")
        print(f"  • {info['description']}")
        print(f"  • Accuracy: {info['accuracy']}")
        print(f"  • Technology: {info['technology']}\n")

    # Visualize accuracy evolution
    years = [1952, 1975, 1990, 2012, 2017, 2020, 2022]
    accuracies = [70, 60, 75, 90, 96, 97, 98]

    plt.figure(figsize=(12, 6))
    plt.plot(years, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Year')
    plt.ylabel('Approximate Accuracy (%)')
    plt.title('ASR Accuracy Evolution Over Time')
    plt.grid(True, alpha=0.3)
    plt.ylim(50, 100)

    # Add annotations
    annotations = [
        (1952, 70, "Bell Labs\nAudrey"),
        (1990, 75, "HMM-GMM"),
        (2012, 90, "Deep\nLearning"),
        (2022, 98, "Whisper")
    ]

    for x, y, label in annotations:
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "asr_evolution.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'asr_evolution.png'}")


def compare_asr_architectures():
    """Compare different ASR architectures."""
    print("\n" + "="*60)
    print("ASR ARCHITECTURES COMPARISON")
    print("="*60)

    architectures = {
        "HMM-GMM (Classical)": {
            "components": ["Acoustic Model", "Language Model", "Decoder"],
            "pros": ["Interpretable", "Works with small data", "Well-understood"],
            "cons": ["Manual feature engineering", "Separate components", "Limited accuracy"],
            "accuracy": "70-80%",
            "speed": "Real-time",
            "data_needed": "~100 hours"
        },
        "RNN/LSTM": {
            "components": ["Feature Extractor", "RNN Encoder", "Decoder"],
            "pros": ["End-to-end", "Captures temporal dependencies", "Better accuracy"],
            "cons": ["Sequential (slow)", "Vanishing gradients", "Hard to parallelize"],
            "accuracy": "85-92%",
            "speed": "Slow (sequential)",
            "data_needed": "~1000 hours"
        },
        "CNN-based": {
            "components": ["Conv layers", "Pooling", "Dense layers"],
            "pros": ["Parallelizable", "Captures local patterns", "Fast training"],
            "cons": ["Limited context window", "Not inherently sequential"],
            "accuracy": "88-94%",
            "speed": "Fast",
            "data_needed": "~1000 hours"
        },
        "Transformer": {
            "components": ["Self-Attention", "Feed-Forward", "Positional Encoding"],
            "pros": ["Parallel training", "Long-range dependencies", "SOTA accuracy"],
            "cons": ["Data hungry", "Computationally expensive", "Large model size"],
            "accuracy": "95-99%",
            "speed": "Fast (parallel)",
            "data_needed": "~10,000+ hours"
        },
        "Wav2Vec2": {
            "components": ["CNN Encoder", "Transformer", "Quantization"],
            "pros": ["Self-supervised pre-training", "Works with less labeled data", "High accuracy"],
            "cons": ["Complex training", "Large model size"],
            "accuracy": "95-98%",
            "speed": "Fast",
            "data_needed": "~10 hours labeled (after pre-training)"
        },
        "Whisper": {
            "components": ["Encoder-Decoder Transformer", "Multi-task head"],
            "pros": ["Robust to noise", "Multi-lingual", "Zero-shot capable"],
            "cons": ["Large model", "Slow for real-time", "Requires GPU"],
            "accuracy": "96-99%",
            "speed": "Medium",
            "data_needed": "680,000 hours (pre-trained)"
        },
    }

    print("\nArchitecture Details:\n")
    for arch, details in architectures.items():
        print(f"{arch}:")
        print(f"  Components: {', '.join(details['components'])}")
        print(f"  Accuracy: {details['accuracy']}")
        print(f"  Speed: {details['speed']}")
        print(f"  Training Data: {details['data_needed']}")
        print(f"  Pros: {', '.join(details['pros'][:2])}")
        print(f"  Cons: {', '.join(details['cons'][:2])}")
        print()


def explain_asr_pipeline():
    """Explain the ASR pipeline stages."""
    print("\n" + "="*60)
    print("ASR PIPELINE: AUDIO → TEXT")
    print("="*60)

    print("\nClassical ASR Pipeline (HMM-GMM Era):\n")
    classical_stages = [
        ("1. Audio Input", "Raw waveform (16kHz, 16-bit PCM)"),
        ("2. Preprocessing", "Noise reduction, normalization"),
        ("3. Feature Extraction", "MFCC, filterbanks (13-40 features)"),
        ("4. Acoustic Model", "HMM-GMM: Audio features → phonemes"),
        ("5. Pronunciation Lexicon", "Phonemes → words"),
        ("6. Language Model", "N-gram: word sequences probabilities"),
        ("7. Decoder", "Beam search to find best word sequence"),
        ("8. Post-processing", "Punctuation, capitalization"),
        ("9. Output", "Text transcription")
    ]

    for stage, description in classical_stages:
        print(f"{stage:<25} {description}")

    print("\n" + "-"*60)
    print("Modern End-to-End ASR (Transformer Era):\n")
    modern_stages = [
        ("1. Audio Input", "Raw waveform (16kHz, 16-bit PCM)"),
        ("2. Preprocessing", "Optional: VAD, noise reduction"),
        ("3. Feature Extraction", "Mel spectrogram (80 bands) OR raw waveform"),
        ("4. Encoder", "Transformer/CNN: Audio features → embeddings"),
        ("5. Decoder", "Transformer: Embeddings → tokens"),
        ("6. Token Decoder", "Tokens → text (BPE/character-level)"),
        ("7. Post-processing", "Optional: punctuation restoration"),
        ("8. Output", "Text transcription")
    ]

    for stage, description in modern_stages:
        print(f"{stage:<25} {description}")

    print("\n" + "="*60)
    print("KEY DIFFERENCES:")
    print("="*60)
    print("""
Classical (HMM-GMM):
  • Separate components (acoustic, language, lexicon)
  • Manual feature engineering (MFCC)
  • Rule-based phoneme-to-word mapping
  • Complex multi-stage training

Modern (End-to-End):
  • Single neural network (encoder-decoder)
  • Learned features (or raw audio)
  • Direct audio → text mapping
  • Simple end-to-end training
  • Implicit language modeling
    """)


def calculate_wer_example():
    """Demonstrate WER (Word Error Rate) calculation."""
    print("\n" + "="*60)
    print("WER (WORD ERROR RATE) - PRIMARY ASR METRIC")
    print("="*60)

    print("\nWER Formula:")
    print("  WER = (S + D + I) / N × 100%")
    print("  Where:")
    print("    S = Substitutions (wrong words)")
    print("    D = Deletions (missing words)")
    print("    I = Insertions (extra words)")
    print("    N = Total words in reference")

    # Example calculations
    examples = [
        {
            "reference": "the quick brown fox jumps over the lazy dog",
            "hypothesis": "the quick brown fox jumps over the lazy dog",
            "description": "Perfect transcription"
        },
        {
            "reference": "the quick brown fox jumps over the lazy dog",
            "hypothesis": "the quick brown fox dumps over the lazy dog",
            "description": "1 substitution (jumps → dumps)"
        },
        {
            "reference": "the quick brown fox jumps over the lazy dog",
            "hypothesis": "the quick brown fox over the lazy dog",
            "description": "1 deletion (missing 'jumps')"
        },
        {
            "reference": "the quick brown fox jumps over the lazy dog",
            "hypothesis": "the quick brown and fox jumps over the lazy dog",
            "description": "1 insertion (extra 'and')"
        },
        {
            "reference": "the quick brown fox jumps over the lazy dog",
            "hypothesis": "the kwik braun foks jumps",
            "description": "Multiple errors"
        },
    ]

    print("\n" + "="*60)
    print("Example Calculations:")
    print("="*60 + "\n")

    for i, ex in enumerate(examples, 1):
        ref_words = ex["reference"].split()
        hyp_words = ex["hypothesis"].split()

        # Simple WER calculation (approximation for demonstration)
        n = len(ref_words)

        # Count differences (simplified)
        if ref_words == hyp_words:
            s, d, i = 0, 0, 0
        else:
            # Simplified counting
            s = sum(1 for r, h in zip(ref_words, hyp_words) if r != h)
            d = max(0, len(ref_words) - len(hyp_words))
            i = max(0, len(hyp_words) - len(ref_words))

        wer = (s + d + i) / n * 100

        print(f"Example {i}: {ex['description']}")
        print(f"  Reference:  '{ex['reference']}'")
        print(f"  Hypothesis: '{ex['hypothesis']}'")
        print(f"  Errors: S={s}, D={d}, I={i}")
        print(f"  WER: {wer:.1f}%")
        print()


def asr_metrics_explained():
    """Explain ASR evaluation metrics."""
    print("\n" + "="*60)
    print("ASR EVALUATION METRICS")
    print("="*60)

    metrics = {
        "WER (Word Error Rate)": {
            "formula": "(Substitutions + Deletions + Insertions) / Total Words",
            "range": "0-100% (lower is better)",
            "use_case": "Standard metric for ASR",
            "benchmark": "<5% = Excellent, 5-10% = Good, >20% = Poor"
        },
        "CER (Character Error Rate)": {
            "formula": "Same as WER but at character level",
            "range": "0-100% (lower is better)",
            "use_case": "Languages without clear word boundaries (Chinese, Japanese)",
            "benchmark": "<2% = Excellent"
        },
        "Real-Time Factor (RTF)": {
            "formula": "Processing Time / Audio Duration",
            "range": "0-∞ (lower is better)",
            "use_case": "Measuring inference speed",
            "benchmark": "<0.1 = Very fast, <1.0 = Real-time, >1.0 = Slower than real-time"
        },
        "Accuracy": {
            "formula": "1 - WER",
            "range": "0-100% (higher is better)",
            "use_case": "Alternative to WER",
            "benchmark": ">95% = Excellent"
        }
    }

    print("\nKey Metrics:\n")
    for metric, details in metrics.items():
        print(f"{metric}:")
        print(f"  Formula: {details['formula']}")
        print(f"  Range: {details['range']}")
        print(f"  Use Case: {details['use_case']}")
        print(f"  Benchmark: {details['benchmark']}")
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 2, STEP 1: ASR FUNDAMENTALS")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding ASR evolution (HMM → Transformers)")
    print("2. Comparing different architectures")
    print("3. Classical vs Modern ASR pipelines")
    print("4. Evaluation metrics (WER, CER, RTF)")

    # Run demonstrations
    demonstrate_asr_history()
    compare_asr_architectures()
    explain_asr_pipeline()
    calculate_wer_example()
    asr_metrics_explained()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. ASR has evolved dramatically:
   - 1980s-2000s: HMM-GMM (70-80% accuracy)
   - 2010s: Deep Learning/RNN (85-95%)
   - 2020s: Transformers (95-99%)

2. Classical ASR (HMM-GMM):
   - Multiple separate components
   - Manual feature engineering
   - Works with limited data
   - Complex training pipeline

3. Modern ASR (Transformers):
   - End-to-end architecture
   - Learned features
   - Requires lots of data
   - Simpler training, better accuracy

4. Key Architectures:
   - RNN/LSTM: Sequential, captures time dependencies
   - CNN: Parallel, captures local patterns
   - Transformer: Self-attention, SOTA results
   - Wav2Vec2: Self-supervised pre-training
   - Whisper: Large-scale multi-task training

5. Evaluation:
   - WER (Word Error Rate): Primary metric
   - Lower WER = Better performance
   - <5% WER = Excellent
   - 5-10% WER = Good
   - >20% WER = Needs improvement

6. For Voice Engines:
   - Use pre-trained models (Whisper, Wav2Vec2)
   - Fine-tune on your domain if needed
   - Consider latency vs accuracy trade-offs
   - Real-time: faster-whisper, Vosk
   - Batch: Whisper large models
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
