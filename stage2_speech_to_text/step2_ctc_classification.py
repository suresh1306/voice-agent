"""
Step 2: CTC (Connectionist Temporal Classification)
====================================================
Learn about CTC - the breakthrough that enabled end-to-end ASR:
- The alignment problem in speech recognition
- CTC loss function
- Blank token and many-to-one mapping
- Greedy vs beam search decoding
- Implementing basic CTC decoder

Key Concepts:
- Problem: Audio frames >> text characters (alignment unknown)
- Solution: CTC introduces blank token, allows many-to-one mappings
- CTC Loss: Sum over all possible alignments
- Decoding: Collapse repeated characters, remove blanks
- Used in: DeepSpeech, Wav2Vec2 (with modifications)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def explain_alignment_problem():
    """Explain the fundamental alignment problem in ASR."""
    print("\n" + "="*60)
    print("THE ALIGNMENT PROBLEM IN SPEECH RECOGNITION")
    print("="*60)

    print("""
The Challenge:
--------------
When converting speech to text, we face a fundamental problem:

Audio frames:  [f₁][f₂][f₃][f₄][f₅][f₆][f₇][f₈][f₉][f₁₀]...
Text:          [  h  ][  e  ][  l  ][  l  ][  o  ]

Questions:
1. Which audio frames correspond to which characters?
2. Some characters are longer (held) than others
3. Silence between words
4. Variable speaking rates

Example: Saying "hello"
- Frame 1-3: "h" sound
- Frame 4-5: "e" sound
- Frame 6-8: "l" sound (held longer)
- Frame 9: "l" sound
- Frame 10-12: "o" sound

We DON'T know this alignment in advance!
    """)

    # Visualize the problem
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Audio frames
    frames = np.arange(20)
    audio_amplitude = np.random.rand(20) * 0.5 + 0.5  # Random amplitudes

    axes[0].bar(frames, audio_amplitude, width=0.8, alpha=0.7, color='blue')
    axes[0].set_xlabel('Audio Frames')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Audio Frames (20 frames total)')
    axes[0].grid(True, alpha=0.3)

    # Text alignment (unknown)
    text = "hello"
    axes[1].text(0.5, 0.5, '  ?  →  ?  →  ?  →  ?  →  ?',
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].text(0.5, 0.3, f'Text: "{text}"',
                ha='center', va='center', fontsize=14)
    axes[1].text(0.5, 0.7, 'Unknown: Which frames → which characters?',
                ha='center', va='center', fontsize=12, style='italic')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Alignment Unknown!')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ctc_alignment_problem.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'ctc_alignment_problem.png'}")


def explain_ctc_solution():
    """Explain how CTC solves the alignment problem."""
    print("\n" + "="*60)
    print("CTC SOLUTION: BLANK TOKEN")
    print("="*60)

    print("""
CTC Innovation:
--------------
1. Introduce a BLANK token (ε or -)
2. Allow model to output any character at each frame
3. Post-processing rules:
   a) Collapse consecutive identical characters
   b) Remove blank tokens

Example: "hello"
-----------------
Possible CTC paths:

Path 1: [h][h][-][e][l][l][l][-][o][o]
        → hh-elll-oo
        → Collapse: h-el-o
        → Remove blanks: helo  ✗ (wrong)

Path 2: [h][-][e][-][l][-][l][-][o][-]
        → h-e-l-l-o-
        → Remove blanks: hello  ✓ (correct)

Path 3: [h][h][e][e][l][l][l][l][o][o]
        → hheellllo
        → Collapse: hello  ✓ (correct)

Key Insight: Many paths can lead to the same text!
CTC loss: Sum probability over ALL valid paths.
    """)

    # Visualize CTC paths
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    frames = list(range(10))

    # Path 1 (wrong)
    path1 = ['h', 'h', '-', 'e', 'l', 'l', 'l', '-', 'o', 'o']
    axes[0].bar(frames, [1]*10, width=0.8, alpha=0.6, color='red')
    for i, char in enumerate(path1):
        axes[0].text(i, 0.5, char, ha='center', va='center',
                    fontsize=14, fontweight='bold')
    axes[0].set_title('Path 1: hh-elll-oo → helo ✗ (Missing "l")', fontsize=12)
    axes[0].set_ylabel('Frame')
    axes[0].set_ylim(0, 1.2)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Path 2 (correct)
    path2 = ['h', '-', 'e', '-', 'l', '-', 'l', '-', 'o', '-']
    axes[1].bar(frames, [1]*10, width=0.8, alpha=0.6, color='green')
    for i, char in enumerate(path2):
        axes[1].text(i, 0.5, char, ha='center', va='center',
                    fontsize=14, fontweight='bold')
    axes[1].set_title('Path 2: h-e-l-l-o- → hello ✓ (Correct!)', fontsize=12)
    axes[1].set_ylabel('Frame')
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Path 3 (correct)
    path3 = ['h', 'h', 'e', 'e', 'l', 'l', 'l', 'l', 'o', 'o']
    axes[2].bar(frames, [1]*10, width=0.8, alpha=0.6, color='green')
    for i, char in enumerate(path3):
        axes[2].text(i, 0.5, char, ha='center', va='center',
                    fontsize=14, fontweight='bold')
    axes[2].set_title('Path 3: hheellllo → hello ✓ (Correct after collapse)', fontsize=12)
    axes[2].set_xlabel('Time Frame')
    axes[2].set_ylabel('Frame')
    axes[2].set_ylim(0, 1.2)
    axes[2].set_xticks(frames)
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ctc_paths.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'ctc_paths.png'}")


def ctc_collapse_function(path):
    """
    CTC collapse function: remove consecutive duplicates and blanks.

    Args:
        path: List of characters (including blanks '-')

    Returns:
        Collapsed string
    """
    # Step 1: Remove consecutive duplicates
    collapsed = []
    prev = None
    for char in path:
        if char != prev:
            collapsed.append(char)
            prev = char

    # Step 2: Remove blanks
    result = ''.join([c for c in collapsed if c != '-'])

    return result


def demonstrate_ctc_collapse():
    """Demonstrate CTC collapsing with examples."""
    print("\n" + "="*60)
    print("CTC COLLAPSE FUNCTION")
    print("="*60)

    test_cases = [
        (['h', 'h', 'e', 'l', 'l', 'o'], "Basic collapse"),
        (['h', '-', 'e', '-', 'l', 'l', 'o'], "With blanks"),
        (['h', 'h', '-', '-', 'e', 'l', 'l', '-', 'o', 'o'], "Mixed"),
        (['-', 'h', 'e', 'l', 'l', 'o', '-'], "Leading/trailing blanks"),
        (['h', 'e', 'l', 'l', '-', 'l', 'o'], "Blank between same char"),
    ]

    print("\nTest Cases:\n")
    print(f"{'Input Path':<35} {'Output':<10} {'Description'}")
    print("-" * 70)

    for path, description in test_cases:
        result = ctc_collapse_function(path)
        path_str = ''.join(path)
        print(f"{path_str:<35} {result:<10} {description}")

    print("""
\nKey Rules:
----------
1. Collapse consecutive identical characters:
   "hheelllo" → "helo"

2. Remove blank tokens (-):
   "h-e-l-l-o" → "hello"

3. Order matters:
   "hello" ≠ "hlelo"

4. Blank separates identical characters:
   "hel-lo" → "hello" (blank between two 'l's)
   "helllo" → "helo" (no blank, collapse to one 'l')
    """)


def greedy_ctc_decode(logits, vocab):
    """
    Greedy CTC decoding: pick most likely character at each timestep.

    Args:
        logits: (time_steps, vocab_size) probability distribution
        vocab: List of characters (including blank at index 0)

    Returns:
        Decoded text
    """
    # Get most likely character at each timestep
    best_path = []
    for t in range(logits.shape[0]):
        best_idx = np.argmax(logits[t])
        best_path.append(vocab[best_idx])

    # Apply CTC collapse
    decoded = ctc_collapse_function(best_path)

    return decoded, best_path


def demonstrate_greedy_decoding():
    """Demonstrate greedy CTC decoding."""
    print("\n" + "="*60)
    print("GREEDY CTC DECODING")
    print("="*60)

    # Define vocabulary (blank first)
    vocab = ['-', 'h', 'e', 'l', 'o']

    # Simulate CTC outputs (probability distributions)
    # Shape: (time_steps, vocab_size)
    np.random.seed(42)
    time_steps = 10

    # Create plausible probabilities for "hello"
    logits = np.zeros((time_steps, len(vocab)))

    # Frame 0-1: 'h'
    logits[0, 1] = 0.8; logits[0, 0] = 0.2
    logits[1, 1] = 0.7; logits[1, 0] = 0.3

    # Frame 2: blank
    logits[2, 0] = 0.9; logits[2, 1] = 0.1

    # Frame 3-4: 'e'
    logits[3, 2] = 0.8; logits[3, 0] = 0.2
    logits[4, 2] = 0.7; logits[4, 0] = 0.3

    # Frame 5-7: 'l'
    logits[5, 3] = 0.8; logits[5, 0] = 0.2
    logits[6, 3] = 0.9; logits[6, 0] = 0.1
    logits[7, 3] = 0.8; logits[7, 0] = 0.2

    # Frame 8: blank
    logits[8, 0] = 0.7; logits[8, 3] = 0.3

    # Frame 9: 'o'
    logits[9, 4] = 0.9; logits[9, 0] = 0.1

    # Decode
    decoded, best_path = greedy_ctc_decode(logits, vocab)

    print("\nGreedy Decoding Process:")
    print(f"\nVocabulary: {vocab}")
    print(f"\nTime Steps: {time_steps}")
    print(f"\nBest Path: {' '.join(best_path)}")
    print(f"After CTC Collapse: '{decoded}'")

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap of probabilities
    im = ax.imshow(logits.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Character', fontsize=12)
    ax.set_title('CTC Output Probabilities (Greedy Decoding)', fontsize=14)

    ax.set_xticks(range(time_steps))
    ax.set_yticks(range(len(vocab)))
    ax.set_yticklabels(vocab)

    # Add probability values
    for t in range(time_steps):
        for c in range(len(vocab)):
            text = ax.text(t, c, f'{logits[t, c]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    # Highlight best path
    for t in range(time_steps):
        best_c = np.argmax(logits[t])
        ax.add_patch(plt.Rectangle((t-0.4, best_c-0.4), 0.8, 0.8,
                                    fill=False, edgecolor='blue', linewidth=3))

    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ctc_greedy_decoding.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'ctc_greedy_decoding.png'}")


def explain_beam_search():
    """Explain beam search decoding for CTC."""
    print("\n" + "="*60)
    print("BEAM SEARCH CTC DECODING")
    print("="*60)

    print("""
Limitation of Greedy Decoding:
-------------------------------
Greedy picks the best character at each step independently.
This is NOT always optimal!

Example:
Frame 1: 'h' (0.6), '-' (0.4)   → Pick 'h'
Frame 2: 'e' (0.5), 'h' (0.5)   → Pick 'e'
Greedy path: [h, e] → "he"

But what if:
Path A: [h, e] (0.6 × 0.5 = 0.30) → "he"
Path B: [-, h, e] (0.4 × ? × ?) → could be higher!

Beam Search Solution:
---------------------
Keep track of top-k most likely paths (beam width = k)

Beam Width = 3:
Step 1: Keep 3 best paths
Step 2: Expand each, keep 3 best overall
Step 3: Continue...

Benefits:
- Explores alternative paths
- Finds globally better solutions
- Beam width trade-off: accuracy vs speed

Common beam widths:
- Greedy: width = 1 (fastest, less accurate)
- Small: width = 5-10 (good balance)
- Large: width = 100+ (slow, most accurate)
    """)


def ctc_best_practices():
    """Provide CTC best practices."""
    print("\n" + "="*60)
    print("CTC BEST PRACTICES")
    print("="*60)

    print("""
When to Use CTC:
---------------
✓ Sequence-to-sequence problems with unknown alignment
✓ Output length ≤ input length (can use blank tokens)
✓ Want simpler training (no attention mechanism needed)
✓ Need streaming/online decoding

Limitations:
-----------
✗ Cannot output sequences longer than input
✗ Conditional independence assumption (each frame independent)
✗ No implicit language model (often need external LM)
✗ Struggles with learning from scratch (need lots of data)

Architecture Tips:
-----------------
1. Input: Audio features (Mel spectrogram, MFCC)
   - Typical: 80 Mel bands, 10ms stride

2. Encoder:
   - CNN layers: Extract local patterns
   - RNN/LSTM/GRU: Model temporal dependencies
   - Transformer: Self-attention (more recent)

3. Output: Softmax over vocabulary
   - Characters: 26 letters + blank + space (28-30 tokens)
   - Or subword units (BPE): 500-5000 tokens

4. Training:
   - CTC Loss: Sum over all alignments
   - Optimizer: Adam (lr ~1e-4)
   - Batch size: 16-32 utterances

5. Decoding:
   - Greedy: Fast, good for real-time
   - Beam search: Better accuracy
   - With LM: Best accuracy (but slower)

Modern Usage:
------------
- DeepSpeech (Mozilla): Pure CTC
- Wav2Vec2: CTC on top of self-supervised encoder
- Hybrid systems: CTC + attention (best of both)
- Streaming: CTC preferred (attention needs full context)

For Voice Engines:
-----------------
✓ Use pre-trained models (Wav2Vec2, DeepSpeech)
✓ Fine-tune with CTC for your domain
✓ Greedy decode for real-time applications
✓ Beam search for offline/batch processing
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("STAGE 2, STEP 2: CTC (CONNECTIONIST TEMPORAL CLASSIFICATION)")
    print("="*60)
    print("\nLearning Objectives:")
    print("1. Understanding the alignment problem in ASR")
    print("2. How CTC solves it with blank tokens")
    print("3. CTC collapse function")
    print("4. Greedy vs beam search decoding")
    print("5. When to use CTC")

    # Run demonstrations
    explain_alignment_problem()
    explain_ctc_solution()
    demonstrate_ctc_collapse()
    demonstrate_greedy_decoding()
    explain_beam_search()
    ctc_best_practices()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. The Alignment Problem:
   - Audio has many frames, text has few characters
   - Don't know which frames → which characters
   - Traditional ASR used forced alignment

2. CTC Solution:
   - Introduce blank token (-)
   - Allow any output at each timestep
   - Post-process: collapse + remove blanks
   - Many paths can produce same text

3. CTC Collapse Rules:
   a) Remove consecutive duplicates: "hheelllo" → "helo"
   b) Remove blanks: "h-e-l-l-o" → "hello"
   c) Blank separates identical chars: "hel-lo" → "hello"

4. Decoding Strategies:
   - Greedy: Fast, pick best at each step
   - Beam Search: Better, explore multiple paths
   - With LM: Best, incorporate language model

5. CTC in Modern ASR:
   - Used in DeepSpeech, Wav2Vec2
   - Good for streaming (frame-by-frame)
   - Needs large datasets
   - Often combined with attention

6. For Voice Engines:
   - Use pre-trained CTC models
   - Greedy for real-time
   - Beam search for accuracy
   - Consider hybrid CTC-attention
    """)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
