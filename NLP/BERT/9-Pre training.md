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

# Chapter 9 — BERT Pre-Training (Master Notes, Apple MLE Prep)

> Goal of this doc: know the weight-tying trick the original chapter never mentioned (a real parameter-saving detail worth citing), correct the "512 is 16x more expensive than 128" claim with actual numbers (it's true for attention scores alone, not for the model as a whole), and be able to explain warmup mechanically rather than just naming it.

---

## 0. One-sentence version

> "BERT is pre-trained with two self-supervised tasks that generate their own labels from raw text — Masked Language Modeling forces bidirectional, token-level understanding by predicting hidden words from both-side context, and Next Sentence Prediction (later found to be mostly unnecessary) tries to teach sentence-level relationships — and the entire 110M-parameter model learns purely from this, with zero human annotation."

---

## 1. Self-supervised learning — why this specific trick, not just "labels are free"

**The precise mechanism, stated carefully**: self-supervision isn't "no labels at all" — it's **manufacturing labels automatically from the structure already present in raw text**, rather than paying humans to add new labels. For MLM, the label is simply "the word that was actually there before we hid it" — something every sentence already tells you for free, as long as you're willing to hide part of it and quiz yourself. For NSP, the label is "did sentence B actually follow sentence A in the source document" — again, information the corpus already contains implicitly in its ordering, not something anyone had to annotate.

**What if we instead trained BERT purely to reconstruct the entire sentence, unmasked, given the sentence itself (an autoencoder-style objective)?** This would be trivially solvable — the identity function, output = input, achieves zero loss without learning anything about language. Masking is what prevents this degenerate shortcut: by *removing* information from the input that the model must recover from context, you force it to actually build a predictive model of language rather than just learning to copy.

---

## 2. MLM — the mechanism, and why bidirectionality specifically requires this trick

**Why GPT-style next-token prediction can't be run bidirectionally (recap from Chapter 1, worth having ready here)**: if next-token prediction saw the token it's trying to predict as part of its own input, the objective becomes trivial (just copy the visible answer) — this is why GPT is unidirectional *by necessity*, not preference. MLM sidesteps this entirely differently: the target token is **removed from the input** (replaced by `[MASK]` or corrupted), so there's no answer sitting in the input to copy, regardless of whether the model looks left, right, or both. This is the specific trick that makes bidirectional pre-training possible at all.

### 2.1 The 80/10/10 split — simplified reasoning, with the "what if" made concrete

| Case | What the model sees | What it must still predict | What this specifically trains |
|---|---|---|---|
| 80% | `[MASK]` | the original word ("cat") | the main signal — fill in an explicitly hidden slot using context |
| 10% | a wrong random word ("dog") | still the original word ("cat") | don't blindly trust the token you're given — a real word can still be "wrong" and must be corrected from context |
| 10% | the original, unchanged word ("cat") | still the original word ("cat") | build a good representation for *every* token, since you never know in advance which ones will be "quizzed" |

**What if it were 100% `[MASK]`, 0% random/unchanged (the naive version)?** The model would learn a narrow shortcut: "only build a rich, predictive representation at positions marked `[MASK]`; everywhere else, coast." Since `[MASK]` never appears in real downstream text (fine-tuning, inference), the model would have specialized for a token it will never see again post-training — a real train/inference distribution mismatch (Chapter 2's framing of the same issue, from the tokenization side).

**What if it were, say, 50% `[MASK]`, 50% random, 0% unchanged?** You'd lose the specific pressure the "unchanged" case creates — namely, forcing good representations even at positions where *nothing looks wrong at all*. Without any unchanged case, the model could in principle learn "if the token looks plausible in context, don't bother refining its representation, since it won't be corrected" — a subtler shortcut than the 100%-`[MASK]` case, but a real one; the unchanged 10% specifically closes this loophole by keeping the model honest even when there's no visible cue that this position might be tested.

### 2.2 The prediction head — with the parameter-saving detail the original chapter omits

**Original chapter's pipeline** (correct as stated):
```
h_mask [768] → Linear(768→768) → GELU → LayerNorm → Linear(768→30522) → logits
```

**What's missing, and matters for an interview**: the final `Linear(768→30522)` layer is, in the real BERT implementation, **weight-tied to the token embedding table from Chapter 3** — the decoder matrix used here is literally the *transpose* of the same `[30,522×768]` token embedding matrix used at the input, not an independently-learned second matrix.

**Why this makes sense, mechanically**: the input embedding table answers "given a token ID, what's its vector?" The output projection needs to answer the *inverse* question: "given a vector, how well does it match each possible token?" Both operations live in the same 768-d semantic space mapped against the same 30,522-word vocabulary — reusing the same matrix (transposed) for both directions is a natural fit, not just a parameter-saving hack, though it is also that.

**The parameter savings, concretely**: a fresh, untied `[768×30522]` output matrix would cost **23,440,896 parameters** — exactly the same size as the token embedding table itself (Chapter 3). Weight tying means this cost is **zero additional parameters** — the pre-training-only prediction head's only *new* cost is the `Linear(768→768)` transform layer (589,824 params + bias) and its LayerNorm (1,536 params), roughly **591K params total**, versus what would otherwise be a ~24M-parameter head. This is also why the "110M parameters" figure from Chapter 8 doesn't need to account for a separate 30,522-wide output matrix — it's re-using weights already counted in the embedding table.

**What if you didn't tie the weights?** The model would need to learn two separate 23.4M-parameter mappings — one for "token → vector" and one for "vector → token likelihood" — that have no guaranteed relationship to each other, giving the optimizer more freedom but also more parameters to fit well, with less inherent regularization pushing the input and output spaces to stay consistent with each other. Empirically, weight tying is standard practice across BERT, GPT, and most modern language models for exactly this reason — fewer parameters, and a useful inductive bias linking the two directions of the same embedding space.

**Note this MLM head is pre-training-only scaffolding**: at fine-tuning time (Chapter 10), this head is discarded and replaced with a task-specific head (e.g., a classifier on `[CLS]`) — only the 12-block encoder + pooler carry over.

---

## 3. NSP — kept mostly as-is, correct math verified

The chapter's cross-entropy numbers check out: $-\log(0.94) \approx 0.062$, $-\log(0.95) \approx 0.051$. No corrections needed here beyond what Section 5 (RoBERTa) adds.

**What if the "NotNext" sentence were topically related but not actually adjacent (a harder negative), instead of a random unrelated sentence?** This is exactly the critique RoBERTa's authors leveled at NSP's design (Section 5) — with fully random negatives, the model can solve NSP largely by detecting a **topic shift** (cats vs. stock markets is an easy signal) rather than genuinely reasoning about discourse coherence or entailment between adjacent sentences. A harder-negative version (same-document, non-adjacent sentences) would force more genuine relational reasoning — but this isn't what original BERT used, and it's part of why NSP's usefulness came into question.

---

## 4. The training schedule — corrected numbers, since "16x more expensive" isn't quite right for the whole model

**The original chapter's claim**: "Self-attention is O(n²). Training with length 512 is 16× more expensive than length 128." **This is true for the attention-score computation specifically, but not for the model's total per-layer compute** — worth getting exactly right, since it connects directly to the Chapter 4/8 finding that the FFN, not attention, dominates BERT's actual compute budget at these sequence lengths.

**Using the exact formulas from Chapter 4** ($\text{FLOPs}_{attn} = 4n^2 d$, $\text{FLOPs}_{other} = 24n d^2$, $d=768$):

| Component | At $n=128$ | At $n=512$ | Growth factor |
|---|---|---|---|
| Attention-score FLOPs | 50.3M | 805.3M | **16.0x** (exactly matches the "O(n²)" claim) |
| Projections + FFN FLOPs | 1.81B | 7.25B | **4.0x** (scales linearly with $n$) |
| **Total per layer** | **1.86B** | **8.06B** | **≈4.3x** |

**The corrected takeaway**: the *attention-score* component alone really does scale 16x (the original claim is correct in isolation) — but since the FFN and projections dominate total compute at these sequence lengths (Chapter 8's finding: FFN alone is ~52% of BERT's parameters and a proportionally large share of its FLOPs), the *overall* per-layer cost of training at 512 tokens versus 128 tokens is closer to **4.3x**, not 16x. The "16x" figure is a common simplification that's technically true only for the quadratic sub-component, not the training run as a whole.

**Why the batch sizes (256 vs. 32) were chosen — the memory story, with real numbers**: using the memory model from Chapter 4 (each head's attention matrix ≈ $n^2 \times 4$ bytes, ×12 heads ×12 layers):
```
Phase 1 (n=128, batch=256):  attention-matrix memory ≈ 2.36 GB
Phase 2 (n=512, batch=32):   attention-matrix memory ≈ 4.6 GB
```
Reducing batch size 8x (256→32) while sequence length grows 4x (128→512, giving a 16x memory blowup per sequence) results in **net memory roughly doubling**, not exploding — the batch-size cut was specifically sized to keep memory from growing anywhere near the full 16x the raw sequence-length increase would otherwise cause, while still allowing *some* increase (2x) since Phase 2 is a small fraction (10%) of total training.

**A genuinely surprising consequence worth naming**: because the batch-size cut (8x) is *larger* than the compute growth per sequence (≈4.3x total, or even 16x for attention alone), **Phase 2's total compute per training step is actually lower than Phase 1's**, despite processing much longer sequences — you're just processing far fewer sequences (and, net, fewer total tokens: 256×128=32,768 tokens/step in Phase 1 vs. 32×512=16,384 tokens/step in Phase 2, half as many). This is why Phase 2 is only 10% of steps — its purpose isn't primarily "expensive long-sequence training that must be minimized," it's specifically to teach the model to use positional embeddings for positions 128-511 (which see zero gradient signal in Phase 1) and to expose it to genuinely long-range dependencies — a targeted fine-tuning-like extension near the end of pre-training, not a proportionally-scaled continuation of Phase 1's workload.

---

## 5. Warmup — the actual mechanism, not just "prevents early destructive updates"

**The original chapter names warmup but doesn't explain *why* early large updates are destructive — worth making concrete, especially since BERT uses Adam, which has a specific reason to need this.**

**Adam's mechanism, briefly**: Adam scales each parameter's update by a running estimate of that parameter's gradient variance (the "second moment," $v_t$) — parameters with noisy/large gradients get smaller effective steps, parameters with small/stable gradients get relatively larger effective steps. **The problem at the very start of training**: with only a handful of gradient observations, these running estimates are themselves noisy and unreliable — Adam can end up taking a large, confident-looking step based on a second-moment estimate built from almost no data, in a direction that's actually just noise from random initialization.

**Why warmup fixes this specifically**: linearly ramping the learning rate up from 0 over the first 10,000 steps means that during the period when Adam's internal moment estimates are least trustworthy (very early training), the *actual* step sizes taken are kept small regardless of what Adam's (unreliable) estimates suggest — giving the moment estimates time to stabilize on real signal before the learning rate reaches its full value and starts taking Adam's estimates at face value.

**What if you skipped warmup and started at the full 1e-4 learning rate immediately?** With random initialization, early gradients are essentially uninformative about the true loss landscape; a large step taken confidently in a wrong direction can push weights into a poor region the optimizer then struggles to escape from (sometimes visible as loss spikes or outright divergence early in training) — this is a well-documented empirical failure mode for large-batch/large-model Transformer training specifically, not a purely theoretical concern.

---

## 6. RoBERTa's ablations — kept, numbers verified as consistent with published results

The MNLI (+2.1) and SQuAD (+3.4) improvements from removing NSP and switching to dynamic masking + more data are consistent with the published RoBERTa paper's findings. **Worth being precise about attribution**: RoBERTa's gains come from a *bundle* of changes (removing NSP, dynamic masking, 10x more data, larger batches, longer training) evaluated together — the chapter's framing correctly separates the NSP ablation as one isolated finding ("remove NSP, train with MLM only → performance improved"), which was indeed one of the ablations run independently in the paper, but it's worth remembering the final headline RoBERTa numbers reflect the *combination* of all these changes, not NSP-removal alone.

---

## 7. Design-choice summary table, boosted

| Design choice | Why | What breaks without it |
|---|---|---|
| Mask the target rather than reconstruct-in-place | Removes the answer from the input entirely, preventing a trivial copy-the-input shortcut | An unmasked reconstruction objective is solvable by the identity function — learns nothing |
| 80/10/10 split (not 100% `[MASK]`) | Closes two shortcuts: dependency on the literal `[MASK]` token, and neglect of representations at "safe-looking" unmasked positions | Train/inference mismatch (real text never contains `[MASK]`) and weak non-masked-position representations |
| Weight-tied output projection | Output projection is the natural inverse of the input embedding lookup — same semantic space, same vocabulary | ~23.4M extra untied parameters, and loses the regularizing link between input and output token representations |
| Two-phase sequence length (128 then 512) | Attention-score compute is genuinely O(n²), and batch size is tuned to keep memory growth well below the raw 16x sequence-length-squared blowup | Training entirely at 512 from step 1 would cost far more compute/memory for benefit concentrated in positions 128-511 that most training doesn't need |
| Learning-rate warmup | Adam's early moment estimates are unreliable with few observations; small steps early prevent confidently-wrong large updates | Risk of early loss spikes/divergence from large steps taken on noisy, uninformative early gradients |
| Removing NSP (RoBERTa's finding) | NSP's easy topic-shift shortcut wasn't teaching genuine discourse understanding, and it capped training to sentence-pair-length sequences | Original BERT's NSP was kept because Google hypothesized (not conclusively verified) that sentence-relationship signal was worth the context-length cost |

---

## 8. Diagnostics — misconceptions to pre-empt

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "The MLM output layer is a separate, independently-learned 30,522-wide matrix" | In standard BERT, it's weight-tied to the transpose of the input token embedding table — no separate large matrix exists | This is why the pre-training MLM head adds only ~591K parameters beyond the base encoder, not ~24M |
| "Training at 512 tokens is 16x more expensive than at 128, so Phase 2 must be the expensive part of pre-training" | 16x is accurate only for the attention-score sub-component; total per-layer compute (dominated by the FFN) grows closer to 4.3x, and Phase 2's smaller batch size (32 vs 256) means its *per-step* compute is actually lower than Phase 1's | The "16x" figure describes one FLOP component, not the training run's overall cost profile |
| "Warmup exists just as a generic best practice, not for a specific reason tied to BERT's optimizer" | Adam's adaptive step sizing relies on running gradient-variance estimates that are unreliable with very few early observations — warmup specifically compensates for this, it's not optimizer-agnostic folklore | Warmup addresses a concrete mechanism (Adam's early moment-estimate noise), most acute for adaptive optimizers |
| "RoBERTa proves NSP is useless and shouldn't have been included" | NSP's removal was evaluated as one ablation within a bundle of changes (dynamic masking, more data, longer training) — the isolated finding is that NSP wasn't pulling its weight relative to its context-length cost, not that sentence-relationship modeling in general is unhelpful | The correct claim is narrower: *this specific* NSP formulation, with easy random negatives and a context-length tax, underperformed simply training MLM on longer sequences |
| "Since MLM only computes loss on masked positions, only those positions get any training signal" | The chapter's own Section 9.4 makes this point but it's worth restating precisely: non-masked positions still receive gradient signal *indirectly*, since their representations were used (via attention) to help predict the masked position(s) | Every position contributes to producing the masked position's prediction, so every position's weights get some gradient, just not from a loss computed *at* that position |

---

## 9. Q&A practice set (self-test — this chapter had no Q&A in the source; answers below the line)

**Q1 (easy).** In one sentence, why can't MLM be solved by a model that just learns to copy its input?

**Q2 (easy).** What are the three cases in the 80/10/10 split, and what does the model have to predict in each case?

**Q3 (medium).** What is weight tying in the MLM prediction head, and how many parameters does it save compared to an untied output projection?

**Q4 (medium — calculation).** Using the FLOP formulas from Chapter 4, roughly how many times more expensive is a full forward pass through one Transformer layer at $n=256$ tokens compared to $n=128$ tokens? (You don't need the exact breakdown — reason about which component scales how.)

**Q5 (medium).** Why does the 10%-random-token case in MLM specifically teach the model "don't blindly trust the token you're given," in a way the other two cases don't?

**Q6 (hard).** Explain mechanically why Adam's optimizer specifically benefits from learning-rate warmup, in a way a simpler optimizer (plain SGD with no adaptive step sizing) might not need as acutely.

**Q7 (hard).** A colleague says: "RoBERTa's results prove that predicting sentence relationships is never a useful pre-training signal." What's the precise flaw in this claim, based on what the NSP ablation actually showed?

**Q8 (hard — spot the bug).** An engineer training a from-scratch BERT-style model reports that Phase 2 (512-token) training steps are running noticeably slower in wall-clock time than the FLOP-based ≈4.3x total-compute ratio (Section 4) would predict relative to Phase 1 steps, even after accounting for the batch size difference. What's a plausible explanation not captured by the FLOP calculation alone?

---
---

### Answers

**A1.** MLM removes the target word from the input entirely (replacing it with `[MASK]` or a substitute) before asking the model to predict it, so there's no correct answer sitting in the input for the model to trivially copy — unlike a plain reconstruction task where output=input is a valid, information-free zero-loss solution.

**A2.** 80%: the token is replaced with `[MASK]`, and the model predicts the original word. 10%: the token is replaced with a random *different* word, and the model still predicts the original (correct) word, not the word it's currently seeing. 10%: the token is left unchanged (the original word is shown), and the model still predicts that same original word.

**A3.** Weight tying means the MLM head's final `[768×30,522]` output projection isn't a separately-learned matrix — it reuses the transpose of the token embedding table from Chapter 3, since both operations (token ID → vector, and vector → token likelihood) live in the same semantic space over the same vocabulary. This saves 23,440,896 parameters — the exact size an untied output matrix would otherwise require, since that's identical to the token embedding table's own size.

**A4.** Attention-score FLOPs scale as $n^2$, so going from 128 to 256 (a 2x increase in $n$) gives a $2^2 = 4$x increase in attention-score compute. Projection+FFN FLOPs scale linearly as $n$, so they get a straightforward 2x increase. Since the FFN/projections dominate total compute at these lengths (as shown in Section 4's 128-vs-512 table), the *overall* per-layer compute increase from 128 to 256 tokens would land somewhere between 2x and 4x, much closer to 2x than to 4x — following the same reasoning pattern as the worked 128-vs-512 example, just at a smaller multiplier since 256/128=2 rather than 512/128=4.

**A5.** In the 10%-random case, the visible input token ("dog") is a real, plausible-looking word — nothing about it looks obviously wrong or masked, yet the model must still predict something *different* from what it's being shown ("cat"). This is precisely what forces the model to weigh surrounding context more heavily than the raw input token itself when computing that position's representation — the 80% (`[MASK]`) case doesn't teach this, since there's no token to (mis)trust in the first place, and the 10% unchanged case doesn't teach it either, since trusting the visible token would actually be correct there.

**A6.** Adam scales each parameter's update using a running estimate of that parameter's gradient variance (the second moment) — this estimate is built up from relatively few observations early in training and is therefore noisy/unreliable at that point, meaning Adam can take a large, confidently-scaled step based on an estimate that doesn't yet reflect the true loss landscape. Plain SGD with a fixed (non-adaptive) learning rate doesn't have this specific failure mode, since its step size doesn't depend on an evolving internal estimate that needs time to stabilize — though SGD can still benefit from warmup for other, more general reasons (e.g., avoiding large steps into poorly-conditioned regions near a random initialization), the acute Adam-specific reason is the unreliable early second-moment estimate.

**A7.** The RoBERTa ablation showed that *this specific* NSP formulation — binary next-sentence classification using easy random negatives, which forced training on shorter sentence-pair sequences — underperformed simply training MLM alone on longer single sequences, in the specific context of BERT-style pre-training with the compute/data budgets tested. This is a narrower finding than "sentence-relationship prediction is never a useful pre-training signal" — it doesn't rule out that a *harder*, better-designed sentence-relationship task (e.g., with harder negatives, or without the context-length cost NSP imposed) might have shown different results; the ablation specifically isolates *this* NSP design's cost-benefit tradeoff, not the general concept.

**A8.** The FLOP calculation in Section 4 only accounts for raw arithmetic operations, not for other real-world costs that don't scale the same way — most notably, the attention weight matrix's memory footprint (which the same section shows roughly doubles from Phase 1 to Phase 2, even after the batch-size cut) can create memory-bandwidth bottlenecks or force additional data movement between GPU/TPU memory tiers that aren't captured by a pure FLOP count. Real training throughput often tracks memory bandwidth and data-movement overhead as much as raw compute, especially for attention operations at longer sequence lengths — this is exactly the kind of gap that motivates systems-level optimizations like FlashAttention (mentioned in Chapter 4), which specifically targets the memory-movement cost that a FLOP-only analysis misses.

---

## 10. Quick recap card (last-minute review)

- **MLM's core trick**: remove the target from the input entirely (not just ask to reconstruct it) — this is what prevents a trivial copy-shortcut and is the specific mechanism that makes bidirectional pre-training possible at all (unlike GPT's next-token objective, which structurally can't be bidirectional).
- **80/10/10, precisely**: `[MASK]` (main signal) / random wrong token (forces not blindly trusting visible tokens) / unchanged (forces good representations even when nothing looks suspicious) — each percentage closes a different shortcut.
- **Weight tying** (the addition this doc makes): the MLM output projection reuses the transposed token embedding table, saving ~23.4M parameters versus an untied matrix — this is why the pre-training head barely adds to BERT's parameter count.
- **The "16x" correction**: attention-score compute alone scales 16x from 128→512 tokens, but since the FFN dominates total compute (Chapter 8), the *overall* per-layer cost only grows ≈4.3x — and Phase 2's 8x batch-size cut means its per-step compute is actually *lower* than Phase 1's, despite longer sequences.
- **Warmup, mechanistically**: Adam's early gradient-variance estimates are unreliable with few observations; small early steps (via warmup) prevent confidently-wrong large updates before those estimates stabilize.
- **RoBERTa's NSP finding is narrower than "NSP is useless"**: this specific formulation (easy random negatives, context-length tax) underperformed longer-sequence MLM-only training — evaluated as one ablation within a bundle of other simultaneous improvements.

*(Chapter 10 picks up here: fine-tuning — how a pre-trained BERT with only masked-word-prediction skills gets adapted, often with just a few thousand labeled examples, into a model that does sentiment analysis, NER, or QA.)*
