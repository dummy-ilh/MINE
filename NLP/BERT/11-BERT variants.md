# Chapter 11: BERT Variants

BERT was published October 2018. Within 18 months, a dozen variants appeared — each fixing a specific weakness. Understanding these variants, and *why* each change was made, is exactly what Google interviewers probe.

The question is never just "name some BERT variants." It's always: **"why does this variant exist, and when would you use it?"**

---

## 11.1 The Problems With BERT That Needed Fixing

Before looking at variants, understand what was broken:

```
Problem 1: Training was suboptimal
  → NSP was too easy, wasted capacity
  → Static masking saw same masks repeatedly
  → Not enough training data/compute

Problem 2: BERT-large is too slow for production
  → 340M parameters
  → 12-24 layers of sequential computation
  → Latency too high for real-time applications

Problem 3: Memory inefficiency
  → 110M-340M params even for simple tasks
  → Can't run on edge devices or small servers

Problem 4: Domain mismatch
  → Pre-trained on Wikipedia + Books
  → Medical, legal, scientific text is very different
  → Vocabulary is wrong for specialized domains

Problem 5: 512 token limit
  → Documents, legal contracts, research papers are much longer
  → Truncation loses critical information
  → Quadratic attention makes longer sequences expensive
```

Each variant below fixes one or more of these.

---

## 11.2 RoBERTa — Fixing the Training

**Robustly Optimized BERT Approach**
Facebook AI, 2019

### What Changed

```
1. Removed NSP entirely
   → Train on single long sequences (up to 512 tokens)
   → Richer context per training step

2. Dynamic masking
   → Generate new mask pattern every epoch
   → Model sees 40 different mask patterns vs BERT's 1
   → More diverse training signal

3. Much more training data
   → BERT:     3.3B words
   → RoBERTa:  160B words (+ CommonCrawl, News, etc.)

4. Larger batch sizes
   → BERT:     256 sequences
   → RoBERTa:  8,000 sequences
   → More stable gradient estimates

5. Longer training
   → BERT:     1M steps
   → RoBERTa:  500K steps but with 8000 batch = much more data seen

6. Larger byte-pair encoding vocabulary
   → BERT:     30,522 WordPiece tokens
   → RoBERTa:  50,265 BPE tokens
```

### Results

```
Task          BERT-large    RoBERTa
─────────────────────────────────────
MNLI          86.6          90.2    (+3.6)
SQuAD 2.0     83.1          86.8    (+3.7)
RACE          72.0          83.2    (+11.2)
```

Massive gains — just from better training, **zero architecture changes**.

### The Key Lesson

BERT was significantly undertrained. The architecture was fine. Google just didn't push it hard enough. RoBERTa proved that **engineering and data matter as much as architecture.**

### When to Use RoBERTa

```
Use RoBERTa when:
  → You want maximum accuracy on NLU tasks
  → You have GPU budget for a large model
  → Your task is general English NLP

Don't use when:
  → You need a very small model
  → You need multilingual support
  → Your domain is highly specialized
```

---

## 11.3 DistilBERT — Fixing the Size

**Distilled BERT**
Hugging Face, 2019

### The Problem

BERT-base has 110M parameters and 12 layers. For production:
- Inference latency: ~100ms per sequence on CPU
- Memory: 440MB just for weights
- Too slow for real-time APIs

### Knowledge Distillation

DistilBERT uses **knowledge distillation** — a technique where a small "student" model learns to mimic a large "teacher" model.

```
Teacher: BERT-base (12 layers, 110M params)
Student: DistilBERT (6 layers, 66M params)
```

### How Distillation Works

Standard training: student learns from hard labels (0 or 1).

```
Label:  [0, 0, 1, 0]   (one-hot, no information about wrong classes)
Loss:   cross-entropy(predictions, hard_labels)
```

Distillation: student also learns from teacher's **soft probabilities**:

```
Teacher output: [0.02, 0.01, 0.94, 0.03]
```

These soft probabilities are far more informative than hard labels:

```
Hard label says:  "the answer is class 3"
Teacher says:     "the answer is class 3 (94% sure),
                   but class 4 is somewhat plausible (3%),
                   and classes 1,2 are nearly impossible"
```

The student learns **relative similarities between classes** — information that's invisible in hard labels.

### The Distillation Loss

```
Total Loss = α × CrossEntropy(student_output, hard_labels)
           + β × KLDivergence(student_output, teacher_output)
           + γ × MSE(student_hidden_states, teacher_hidden_states)
```

Three terms:
- Learn from true labels
- Match teacher's output distribution
- Match teacher's internal hidden state representations

The third term is crucial — it forces DistilBERT's 6 layers to produce hidden states similar to specific layers of BERT's 12 layers.

### Layer Mapping

```
DistilBERT Layer 1  ←  BERT Layer 2
DistilBERT Layer 2  ←  BERT Layer 4
DistilBERT Layer 3  ←  BERT Layer 6
DistilBERT Layer 4  ←  BERT Layer 8
DistilBERT Layer 5  ←  BERT Layer 10
DistilBERT Layer 6  ←  BERT Layer 12
```

Every other BERT layer is distilled into one DistilBERT layer.

### Results

```
                  BERT-base    DistilBERT    Change
──────────────────────────────────────────────────
Parameters:       110M         66M           -40%
Inference speed:  1×           1.6×faster    +60%
GLUE score:       79.5         77.0          -2.5
SST-2 (sentiment):93.5         91.3          -2.2
```

**97% of BERT's performance at 60% the size and 1.6× the speed.**

For most production use cases, that 2-3% accuracy drop is completely acceptable in exchange for the speed and memory savings.

### When to Use DistilBERT

```
Use DistilBERT when:
  → Latency is critical (real-time APIs, mobile)
  → Memory is constrained (edge devices, small servers)
  → Accuracy difference is acceptable for your task
  → Cost matters (smaller model = cheaper inference)

Don't use when:
  → Maximum accuracy is required
  → Task is complex (multi-hop reasoning, complex QA)
```

---

## 11.4 ALBERT — Fixing the Memory

**A Lite BERT**
Google, 2019

### Two Key Innovations

ALBERT targets a different efficiency angle than DistilBERT. Instead of fewer layers, it reduces **parameters per layer** through two techniques.

### Innovation 1: Factorized Embedding Decomposition

**The problem:**
```
BERT token embeddings: [30,522 × 768] = 23.4M parameters

The embedding table is huge because vocab_size (30,522) × hidden_dim (768)
are multiplied directly.
```

**ALBERT's insight:**

The token embedding only needs to capture basic token identity — a simpler representation. The full 768-d complexity isn't needed at the embedding level.

Split the embedding into two smaller matrices:

```
BERT:    Vocab → 768    [30522 × 768]  = 23.4M params

ALBERT:  Vocab → 128    [30522 × 128]  = 3.9M params
         128  → 768     [128   × 768]  = 0.1M params
                                         ─────────────
                                         4.0M params   (-83%)
```

The token first maps to a small 128-d "pure identity" space, then projects up to 768-d for the Transformer. The 128-d embedding captures what the token IS. The 128→768 projection learns how to express that identity in the model's working space.

### Innovation 2: Cross-Layer Parameter Sharing

**The problem:**
```
BERT has 12 separate Transformer blocks.
Each block has its own W_Q, W_K, W_V, W_O, W1, W2.
Most of this is redundant — adjacent layers learn similar transformations.
```

**ALBERT's solution:** Share parameters across all 12 layers.

```
BERT:    Block 1 weights ≠ Block 2 weights ≠ ... ≠ Block 12 weights
         12 × 7M params = 84M params in Transformer blocks

ALBERT:  Block 1 weights = Block 2 weights = ... = Block 12 weights
         1 × 7M params = 7M params in Transformer blocks
         (same weights used 12 times)
```

**What does this mean computationally?**

The same attention + FFN weights are applied 12 times in sequence. Each application is a different "view" through the same transformation.

Think of it like **iterative refinement** — running the same algorithm multiple times on progressively refined inputs — rather than 12 different algorithms.

### Innovation 3: Replaced NSP with SOP

ALBERT also found NSP too easy (same as RoBERTa). But instead of removing it, they replaced it with **Sentence Order Prediction (SOP)**:

```
NSP:  Is sentence B the actual next sentence, or random?
      → Too easy. Model detects topic shift, not coherence.

SOP:  Are sentences A and B in the correct order, or swapped?
      → Harder. Both sentences are from the same document.
      → Forces model to understand local coherence and discourse flow.

Example:
  Correct:  "The cat sat down. It then fell asleep."
  Swapped:  "It then fell asleep. The cat sat down."
```

### Results

```
                  BERT-large    ALBERT-xxlarge    Change
─────────────────────────────────────────────────────────
Parameters:       340M          235M              -31%
MNLI:             86.6          91.3              +4.7
SQuAD 2.0:        83.1          90.9              +7.8
Training speed:   1×            1.7× faster
```

Better accuracy, fewer parameters, faster training.

**But:** Inference speed doesn't improve much. Shared weights means 12 forward passes through the same layer — still 12 matrix multiplications. ALBERT is parameter-efficient, not compute-efficient.

### When to Use ALBERT

```
Use ALBERT when:
  → GPU memory is the bottleneck (fewer params to store)
  → You want state-of-the-art accuracy
  → Training budget matters more than inference speed

Don't use when:
  → Inference latency is critical (still 12 layers of compute)
  → You need the simplest possible deployment
```

---

## 11.5 DeBERTa — The Current State of the Art

**Decoding-Enhanced BERT with Disentangled Attention**
Microsoft, 2020/2021

### The Core Idea

In standard BERT, each token's representation mixes content and position together:

```
BERT token vector = f(what the token IS + where it IS)
Content and position are entangled in one vector
```

DeBERTa **disentangles** them:

```
Each token gets TWO vectors:
  Content vector:  what the token is
  Position vector: where the token is

Attention computed using BOTH separately
```

### Disentangled Attention

Standard attention score between tokens i and j:

```
BERT:    score(i,j) = dot(Q_i, K_j)
                    = f(content_i, content_j)
```

DeBERTa computes four interaction terms:

```
DeBERTa: score(i,j) = content_i → content_j    (what to what)
                    + content_i → position_j    (what to where)
                    + position_i → content_j    (where to what)
                    (position_i → position_j dropped — less useful)
```

**Why does this help?**

```
"the cat sat on the mat"

Standard BERT on "sat":
  One vector encodes both the verb meaning AND position 3
  Hard to separately attend based on each

DeBERTa on "sat":
  Content: "this is a sitting verb"
  Position: "this is at position 3"
  
  Attending to "cat":
    Content→Content: verb attends to its subject → high
    Content→Position: verb at pos 3, subject at pos 2 → nearby boost
  
  Richer, more precise attention
```

### Enhanced Mask Decoder

DeBERTa also adds absolute position information back in just before the MLM prediction head — giving the model positional context for the final prediction without entangling it in the attention mechanism.

### Results

```
Task              BERT-large    RoBERTa-large    DeBERTa-large
──────────────────────────────────────────────────────────────
MNLI:             86.6          90.2             91.4
SQuAD 2.0:        83.1          86.8             90.7
SuperGLUE:        —             88.4             90.3
```

DeBERTa consistently outperforms RoBERTa with the same model size.

### When to Use DeBERTa

```
Use DeBERTa when:
  → You need the absolute best NLU performance
  → You're competing on benchmarks
  → Task requires precise understanding of positional relationships

In practice:
  → Most production systems use RoBERTa (simpler, well-supported)
  → DeBERTa for research / competition settings
```

---

## 11.6 Domain-Specific BERTs

Sometimes the architecture isn't the problem — the **pre-training domain** is.

### Why Domain Matters

```
BERT pre-trained on Wikipedia + Books
Fine-tuned on medical records

Problem:
  "The patient presented with dyspnea and tachycardia"
  
  dyspnea    → ["dy", "##sp", "##nea"]     ← 3 tokens, no meaning
  tachycardia → ["ta", "##chy", "##card", "##ia"]  ← 4 tokens, no meaning
  
  BERT has no strong representation for these — they were rare in Wikipedia
```

### BioBERT

Pre-trained on:
```
PubMed abstracts:    4.5B words
PubMed full texts:   13.5B words
+ Original BERT pre-training corpus
```

Result: medical terms are whole tokens with rich embeddings:
```
"dyspnea"    → one token with strong clinical meaning
"tachycardia" → one token pointing toward cardiac concepts
```

```
Task                 BERT    BioBERT    Improvement
────────────────────────────────────────────────────
Drug NER:            84.2    89.7       +5.5
Disease NER:         82.8    88.2       +5.4
Medical QA:          38.8    46.1       +7.3
```

### SciBERT

Pre-trained on:
```
Semantic Scholar corpus:  1.14M scientific papers
  18% computer science
  82% biomedical
```

New scientific vocabulary: 30,000 tokens from scientific text (vs BERT's Wikipedia vocabulary).

### LegalBERT

Pre-trained on:
```
EU legislation, court cases, contracts, legal journals
12GB of legal text
```

Legal terms stay whole. Clause structure is better understood.

### The Decision Tree

```
Your task domain?
  ├── General English      → RoBERTa or BERT-base
  ├── Medical / Clinical   → BioBERT or PubMedBERT
  ├── Scientific           → SciBERT
  ├── Legal                → LegalBERT
  ├── Financial            → FinBERT
  └── Code                 → CodeBERT
```

---

## 11.7 Long Document BERTs

BERT's 512 token limit is a hard wall. These models break it.

### Longformer

**The problem:** Full attention is O(n²). For 4096 tokens:
```
Full attention:  4096 × 4096 = 16.7M attention computations
Too slow and too much memory
```

**Longformer's solution:** Sparse attention patterns

```
Local attention:
  Each token attends to w/2 tokens on each side (sliding window)
  Complexity: O(n × w)  where w is window size
  
Global attention:
  Special tokens ([CLS], question tokens) attend to ALL tokens
  All tokens attend back to these special tokens
  Complexity: O(n × g)  where g = number of global tokens
  
Total: O(n × (w + g))  ← linear in sequence length!
```

```
Supports up to 4,096 tokens (vs BERT's 512)
8× longer sequences at roughly same compute
```

### BigBird

Similar to Longformer but adds **random attention** on top of local + global:

```
BigBird attention = Local (sliding window)
                  + Global (special tokens)
                  + Random (each token attends to r random tokens)
```

The random connections create shortcuts across the sequence — mathematically proven to preserve the expressive power of full attention.

Supports up to **4,096 tokens**, with theoretical justification that sparse random attention approximates full attention.

### When to Use Long-Document Models

```
Use Longformer/BigBird when:
  → Documents > 512 tokens (papers, contracts, books)
  → Answer may be anywhere in a long passage
  → Full document context is needed

Examples:
  → Legal contract analysis
  → Research paper summarization
  → Book-length question answering
```

---

## 11.8 The Complete Variant Comparison

```
Model         Params   Speed    Accuracy   Best For
──────────────────────────────────────────────────────────────────
BERT-base     110M     1×       Good       Baseline, learning
BERT-large    340M     0.4×     Better     High-accuracy, enough GPU
RoBERTa       125M     1×       Best gen.  General NLU tasks
DistilBERT    66M      1.6×     -2-3%      Production, latency-critical
ALBERT        12-235M  varies   Best*      Memory-constrained training
DeBERTa       140M     0.9×     Best NLU   Benchmarks, competitions
BioBERT       110M     1×       Best bio   Medical NLP
SciBERT       110M     1×       Best sci   Scientific NLP
Longformer    149M     ~1×†     Good       Long documents (4096 tokens)
```

*ALBERT accuracy depends on size variant (base/large/xlarge/xxlarge)
†Longformer is faster than BERT for sequences > 512 tokens

---

## 11.9 The Meta-Lesson

Looking across all variants, a pattern emerges:

```
BERT's architecture was mostly right.
What was wrong:
  1. Training was insufficient (RoBERTa)
  2. Model was too big for production (DistilBERT)
  3. Parameters were inefficient (ALBERT)
  4. Position and content were entangled (DeBERTa)
  5. Sequence length was too short (Longformer, BigBird)
  6. Domain mismatch (BioBERT, SciBERT, etc.)
```

Each variant is a focused fix for one of these issues. When an interviewer asks "which model would you use for X?", your answer should trace back to which of these problems X triggers.

---

## Chapter 11 Summary

| Fix Needed | Model | Key Technique |
|---|---|---|
| Better training | RoBERTa | No NSP, dynamic masking, more data |
| Smaller/faster | DistilBERT | Knowledge distillation, 6 layers |
| Fewer parameters | ALBERT | Factorized embeddings, shared weights |
| Better accuracy | DeBERTa | Disentangled content/position attention |
| Medical domain | BioBERT | Pre-trained on PubMed |
| Long documents | Longformer | Sparse local + global attention |

---

One chapter left. Chapter 12 is pure interview prep — the exact questions Google asks, how to structure your answers, the tradeoffs you need to articulate fluently, and the system design questions around deploying BERT in production.

Ready?
