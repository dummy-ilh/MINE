# Chapter 9: BERT Pre-Training

You now have a complete architecture — embeddings, attention, FFN, residuals, LayerNorm, 12 blocks. But it's all random weights. A BERT with random weights is useless.

Pre-training is how BERT goes from random noise to a model that understands language. And the remarkable thing is: **it learns from raw text with no human labels whatsoever.**

---

## 9.1 The Core Idea: Self-Supervised Learning

Labeled data is expensive. Getting humans to label sentiment, named entities, or question-answer pairs takes enormous time and money.

But **raw text is free**. The internet has trillions of words. The insight behind BERT:

> Can we design a task where the labels are hidden inside the text itself?

Yes. Two tasks:

```
Task 1: Masked Language Model (MLM)
  → Hide some words, predict them from context

Task 2: Next Sentence Prediction (NSP)
  → Given two sentences, did they appear together?
```

Both tasks generate infinite labeled examples automatically from raw text. No human annotation needed.

---

## 9.2 The Training Corpus

```
BooksCorpus:        800M words   (11,038 unpublished books)
English Wikipedia:  2,500M words (text only, no markup)
─────────────────────────────────
Total:              3,300M words (3.3 billion words)
```

Why books + Wikipedia?

```
Wikipedia:  factual, structured, encyclopedic knowledge
Books:      long-range narrative, complex sentence structure,
            diverse vocabulary, coherent multi-sentence reasoning
```

Together they give BERT exposure to both factual world knowledge and complex linguistic structure.

---

## 9.3 Task 1: Masked Language Model (MLM)

### The Basic Idea

Take a sentence. Randomly mask 15% of tokens. Train BERT to predict the original tokens at masked positions.

```
Original:  "the cat sat on the mat"
Masked:    "the [MASK] sat on the mat"
Target:    predict "cat" at position 2
```

### Why This Forces Bidirectional Learning

This is the key insight that separates BERT from GPT.

GPT predicts the next word — it only needs left context. So it's trained to be unidirectional.

BERT predicts a **middle word** — it needs **both** left and right context to do this well.

```
"the [MASK] sat on the mat"

Left context:  "the"           → could be many things
Right context: "sat on the mat" → something that sits on mats

Combined: animal that sits on mats → "cat"
```

The model is forced to build deep bidirectional representations to solve this task well.

### The 15% Rule — And Why It's Broken Down Further

Of all tokens in the corpus, 15% are selected for prediction. But they are NOT all replaced with [MASK]:

```
Of the selected 15%:
  80% → replaced with [MASK]
  10% → replaced with a random token
  10% → kept as the original token
```

**Example for the word "cat":**

```
80% case:  "the [MASK] sat on the mat"   → model sees [MASK]
10% case:  "the dog sat on the mat"      → model sees wrong word
10% case:  "the cat sat on the mat"      → model sees correct word
```

In all three cases, the model must **predict "cat"** at that position.

### Why Not Just Always Use [MASK]?

This is one of the most important design decisions in BERT.

**Problem 1: Train-test mismatch**

During fine-tuning and inference, [MASK] tokens **never appear**. If BERT only ever sees [MASK] during pre-training, it develops a dependency on seeing that special token to make predictions. Real sentences don't have [MASK] in them.

```
Pre-training sees:   [MASK] constantly
Fine-tuning sees:    never
→ Distribution mismatch → worse fine-tuning performance
```

**Problem 2: The model cheats**

If every masked position has [MASK], the model learns a shortcut:

```
"I need to make a prediction here ONLY when I see [MASK]"
```

It stops building good representations for non-masked tokens — why bother, they'll never be predicted?

### How the 80/10/10 Split Fixes Both Problems

**The 10% random token case:**

```
"the dog sat on the mat"   (dog replacing cat)
```

The model sees "dog" but must predict "cat." This forces it to:
- Not blindly trust the token it sees
- Use surrounding context heavily
- Build strong representations for ALL tokens, not just masked ones

**The 10% unchanged case:**

```
"the cat sat on the mat"   (cat unchanged)
```

The model sees "cat" but still must predict "cat." This forces it to:
- Build good representations even for unmasked tokens
- The model doesn't know which tokens will be asked about
- So it must represent ALL tokens well, just in case

**The result:** BERT builds rich representations for **every token in every position**, not just masked ones. This is what makes its contextual embeddings so powerful.

---

## 9.4 MLM Numerical Example — Full Forward Pass

Let's trace exactly what happens computationally during MLM training.

**Input sentence:** "the cat sat on the mat"

### Step 1: Tokenize

```
["[CLS]", "the", "cat", "sat", "on", "the", "mat", "[SEP]"]
IDs: [101, 1996, 4937, 2938, 2006, 1996, 13523, 102]
```

### Step 2: Select 15% for Masking

8 tokens total (including special tokens). 15% of 8 = 1.2 → mask 1 token.

Special tokens [CLS] and [SEP] are **never masked** — they're structural tokens.

From remaining 6 tokens, randomly select "cat" (position 2).

Apply the 80/10/10 split — say we're in the 80% case:

```
Input IDs: [101, 1996, 103, 2938, 2006, 1996, 13523, 102]
                        ↑
                   103 = [MASK] token ID
```

### Step 3: Forward Pass Through BERT

```
Embedding layer → 12 Transformer blocks
→ Final hidden states: [8 × 768]
```

The hidden state at position 2 (the [MASK] position) is a 768-d vector that has attended to all other tokens across 12 layers.

```
h_mask = hidden_state[position_2]   shape: [768]
```

### Step 4: Prediction Head

A small neural network on top of BERT:

```
h_mask → Linear(768 → 768) → GELU → LayerNorm → Linear(768 → 30522)
                                                          ↓
                                               logits over full vocabulary
                                               shape: [30522]
```

The second linear layer projects to vocabulary size — one score per possible word.

### Step 5: Softmax + Loss

```
Softmax over 30,522 logits → probability distribution over vocabulary

P("cat")  = 0.72   ← high, model is fairly sure
P("dog")  = 0.08
P("rat")  = 0.05
P("bird") = 0.03
...all others sum to 0.12

Cross-entropy loss = -log(P("cat")) = -log(0.72) = 0.329
```

### Step 6: Backpropagation

The loss 0.329 flows back through:
```
Prediction head → Block 12 → Block 11 → ... → Block 1 → Embeddings
```

All weights update slightly to make P("cat") higher next time.

**Crucially:** only position 2 contributes to the loss here. The other 7 positions are not predicted (in this example). But all 8 positions contributed to producing h_mask through attention — so their representations still get gradient signal indirectly.

---

## 9.5 Task 2: Next Sentence Prediction (NSP)

### The Motivation

MLM teaches word-level understanding. But many tasks require **sentence-level understanding**:

```
Question Answering:    Does this passage answer this question?
Natural Language Inf:  Does this hypothesis follow from this premise?
Sentence similarity:   Are these two sentences paraphrases?
```

For these, you need the model to understand relationships **between sentences**, not just within them.

NSP trains exactly this.

### The Task

Take pairs of sentences A and B from the corpus.

```
50% of the time — IsNext:
  A: "The cat sat on the mat."
  B: "It seemed very comfortable there."
  Label: IsNext (B actually followed A in the original text)

50% of the time — NotNext:
  A: "The cat sat on the mat."
  B: "The stock market fell 3% on Tuesday."
  Label: NotNext (B is a random sentence from elsewhere)
```

### The Input Format

```
[CLS] the cat sat on the mat [SEP] it seemed very comfortable there [SEP]
  ↑                            ↑                                     ↑
sentence start            sentence boundary                    sequence end

Segment IDs:
  0     0   0   0   0   0   0    0    1   1       1    1          1      1
  ↑ all sentence A ↑              ↑ all sentence B ↑
```

### The Prediction

The [CLS] token's final hidden state (after 12 layers) goes through a binary classifier:

```
h_CLS → Linear(768 → 2) → Softmax → [P(IsNext), P(NotNext)]
```

**Loss:** Cross-entropy on IsNext vs NotNext label.

### Numerical Example

**IsNext pair:**
```
A: "The cat sat on the mat."
B: "It seemed very comfortable there."

After 12 layers, h_CLS = [0.23, -0.41, 0.88, ..., 0.12]  (768-d)

Linear(768→2): [2.1, -0.8]
Softmax:       [P(IsNext)=0.94, P(NotNext)=0.06]
Label:         IsNext
Loss:          -log(0.94) = 0.062   ← low loss, correct prediction
```

**NotNext pair:**
```
A: "The cat sat on the mat."
B: "The stock market fell 3% on Tuesday."

After 12 layers, h_CLS captures the topic mismatch:
Linear(768→2): [-1.3, 1.9]
Softmax:       [P(IsNext)=0.05, P(NotNext)=0.95]
Label:         NotNext
Loss:          -log(0.95) = 0.051   ← low loss, correct prediction
```

---

## 9.6 The Combined Training Loss

Both losses are computed simultaneously and added:

```
Total Loss = MLM Loss + NSP Loss
```

In every training step:
- A batch of sentence pairs is sampled
- 15% of tokens are masked per sequence
- Both MLM and NSP losses are computed
- Gradients from both flow back together
- All weights update once

This means BERT is simultaneously learning:
```
MLM:  "understand individual words in context"
NSP:  "understand relationships between sentences"
```

---

## 9.7 The Training Procedure

### Sequence Length Strategy

BERT uses a clever two-phase approach:

```
Phase 1 (90% of training steps):
  Max sequence length: 128 tokens
  Batch size: 256
  Faster — shorter sequences fit more in memory
  Learns most linguistic patterns

Phase 2 (10% of training steps):
  Max sequence length: 512 tokens
  Batch size: 32
  Slower — but learns long-range dependencies
  Trains positional embeddings for positions 128-511
```

Why this split? **Self-attention is O(n²)**. Training with length 512 is 16× more expensive than length 128. Do most learning cheaply, then extend to full length.

### Optimizer and Schedule

```
Optimizer:    Adam
  β1 = 0.9, β2 = 0.999

Learning rate: warmup then linear decay
  Warmup:  0 → 1e-4 over first 10,000 steps
  Decay:   1e-4 → 0 over remaining steps

Warmup prevents:
  Early large gradient updates destroying random initialization
  Lets model find a stable region first
```

### Hardware and Time

```
BERT-base:
  Hardware:  4 Cloud TPUs (v2)
  Time:      4 days
  Steps:     1,000,000
  Batch:     256 sequences

BERT-large:
  Hardware:  16 Cloud TPUs (v2)
  Time:      4 days
  Steps:     1,000,000
  Batch:     256 sequences
```

This is why you don't train BERT from scratch yourself. The compute cost is enormous. You use pre-trained weights and fine-tune.

---

## 9.8 What BERT Actually Learns From These Two Tasks

This is the deep insight. Nobody told BERT about grammar, facts, or meaning. Yet after pre-training:

**From MLM alone, BERT learned:**

```
Syntax:
  "the [MASK] runs fast" → must be a noun (subject)
  "she [MASK] the ball"  → must be a verb
  BERT learned POS tagging without any POS labels

Semantics:
  "Paris is the [MASK] of France" → capital
  "water boils at 100 [MASK]"    → degrees
  BERT learned world facts

Co-reference:
  "The trophy didn't fit because [MASK] was too big"
  → BERT learned to track entities across sentences

Negation:
  "the food was not [MASK]" → bad/terrible/great? (not great)
  BERT learned negation affects meaning
```

**From NSP, BERT learned:**

```
Topic coherence:     consecutive sentences share topics
Discourse structure: how ideas flow between sentences
Logical entailment:  some sentence pairs are related, others aren't
```

All of this emerges from predicting masked words and sentence order. **The task is the teacher.**

---

## 9.9 The Controversy: Does NSP Actually Help?

In 2019, Facebook AI released **RoBERTa** — a robustly optimized BERT. They ran ablation experiments:

```
Experiment: Remove NSP, train with MLM only

Result: Performance IMPROVED on most benchmarks

MNLI:  BERT 84.3  →  RoBERTa (no NSP) 86.4   (+2.1)
SQuAD: BERT 88.5  →  RoBERTa (no NSP) 91.9   (+3.4)
```

**Why does removing NSP help?**

The NSP task forces BERT to use short sentence pairs (A + B must fit in 512 tokens). This artificially limits context length.

Without NSP, you can train on **single long sequences** up to 512 tokens — giving the model much longer context to learn from in every step. That richer context more than compensates for losing the sentence-pair signal.

Additionally, NSP turned out to be **too easy** — BERT was solving it by detecting topic shift, not understanding true sentence relationships. It wasn't teaching what Google hoped.

```
BERT:    MLM + NSP
RoBERTa: MLM only + more data + longer training + larger batches
         → Better on almost everything
```

This is important for interviews — knowing why a design decision was later revised shows depth of understanding.

---

## 9.10 Dynamic vs Static Masking

**BERT (static masking):**
```
Masking is done once before training.
Every time the model sees a sentence, the same tokens are masked.
After 40 epochs, BERT has seen each mask pattern ~40 times.
```

**RoBERTa (dynamic masking):**
```
Masking is generated fresh every time a sequence is sampled.
Each epoch, different tokens get masked.
Model sees more diverse training signal.
Better generalization.
```

Again — a seemingly small engineering choice that measurably improves performance.

---

## Chapter 9 Summary

### MLM
```
Mask 15% of tokens (80% [MASK], 10% random, 10% unchanged)
Predict original token at masked positions
Forces bidirectional context understanding
Teaches syntax, semantics, world knowledge — all from raw text
```

### NSP
```
50% real next sentences, 50% random
Binary classification using [CLS] vector
Teaches sentence-level relationship understanding
Later shown to be less useful than hoped (RoBERTa removed it)
```

### Combined
```
Total Loss = MLM Loss + NSP Loss
Trained on 3.3B words, 1M steps, 4-16 TPUs
All linguistic knowledge emerges from these two simple tasks
```

### The Evolution
```
BERT:     MLM + NSP, static masking, 3.3B words
RoBERTa:  MLM only, dynamic masking, 10× more data, larger batches
          → Better performance on almost all benchmarks
```

---

BERT now has weights that encode deep language understanding. But pre-trained BERT can't do sentiment analysis, answer questions, or tag entities — it can only predict masked words.

**Chapter 10** is where pre-trained BERT becomes useful for your actual task. Fine-tuning takes everything BERT learned in pre-training and adapts it to your specific problem — with sometimes as few as a few thousand labeled examples.

Ready?
