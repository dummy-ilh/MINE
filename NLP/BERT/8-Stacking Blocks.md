# Chapter 8: Stacking Transformer Blocks

You now know everything that happens inside one block. Let's zoom out and ask: **what happens when you run 12 of them in sequence?**

---

## 8.1 The Full Architecture — Top to Bottom

```
Input: "the cat sat"

         [CLS]  the   cat   sat  [SEP]
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Embedding Layer             │
    │  Token + Position + Segment     │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Transformer Block 1         │
    │  MultiHeadAttn → FFN            │
    │  + Residuals + LayerNorm        │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Transformer Block 2         │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓
          ...   ...   ...   ...   ...
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Transformer Block 12        │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓

    Final hidden states: one 768-d vector per token
```

**The shape never changes:**
```
After embedding:   [5 × 768]
After block 1:     [5 × 768]
After block 2:     [5 × 768]
...
After block 12:    [5 × 768]
```

Every block takes a `[seq_len × 768]` matrix and outputs a `[seq_len × 768]` matrix. What changes is what those 768 numbers **mean** — they get progressively richer.

---

## 8.2 What's Actually Shared vs Separate Across Blocks

A common confusion. Let's be explicit:

```
SEPARATE per block (each block has its own):
  W_Q, W_K, W_V for all 12 heads    → 12 × 3 × (768×64)
  W_O                                → 768×768
  W1, b1 (FFN expand)                → 768×3072
  W2, b2 (FFN compress)              → 3072×768
  γ, β for both LayerNorms           → 4 × 768

SHARED across all blocks:
  Token Embedding Table              → 30522×768
  Position Embedding Table           → 512×768
  Segment Embedding Table            → 2×768
```

This means every block is learning **different transformations** of the same representational space. Block 1's W_Q has nothing to do with Block 7's W_Q.

---

## 8.3 How Information Flows Through 12 Blocks

Let's trace the word **"bank"** through all 12 blocks in:

```
"I went to the river bank to fish"
```

### Block 1 — Raw Signal, Local Patterns

At the embedding layer, "bank" starts with its default token embedding — a static vector that's an average of all its meanings (financial, river, etc.).

Block 1 attention patterns tend to focus on **local neighbors** and **syntactic structure**. "bank" looks at "river" next to it, and "the" before it.

```
"bank" after Block 1:
  Still mostly its default embedding
  Slight pull toward "river" (adjacent)
  Basic syntactic position noted
```

### Blocks 2-4 — Syntax Solidifies

These layers handle grammatical structure:
- Subject-verb relationships
- Noun phrase boundaries
- Prepositional phrase attachments

```
"bank" after Block 4:
  Understands it's a noun
  Knows it's the object of "to"
  Knows "river" modifies it
  Starts pulling away from financial meaning
```

### Blocks 5-8 — Semantics Emerge

Longer range dependencies. "fish" at the end of the sentence is 4 positions away — attention heads in these layers reach further.

```
"bank" after Block 8:
  "fish" has now influenced it strongly
  "river" + "fish" together = strong nature/water signal
  Financial meaning is largely suppressed
  Co-reference relationships resolved
```

### Blocks 9-12 — Task-Specific Refinement

The highest layers produce the most abstract, task-relevant representations.

```
"bank" after Block 12:
  Fully contextualized toward river/nature meaning
  Its 768-d vector is now close in embedding space to
  "shore", "riverbank", "stream" — not "finance", "deposit"
  Ready for downstream task
```

---

## 8.4 Numerical Trace — One Token Across 3 Blocks

Let's track "cat" numerically through 3 simplified blocks (d=4).

**Starting vector after embedding:**
```
cat_0 = [0.70, 0.33, -0.31, 0.73]
```

We'll use simplified transformations to show the progression.

### After Block 1:

Attention output adds context from "the" and "sat":
```
Attention(cat_0) = [0.451, 0.170, 0.160, 0.300]   (from Chapter 4)

Residual:  cat_0 + Attention = [1.151, 0.500, -0.150, 1.030]
LayerNorm: cat_1 = [1.008, -0.259, -1.524, 0.772]  (from Chapter 7)

FFN transforms:
FFN(cat_1) = [0.291, 0.070, 0.394, -0.006]   (from Chapter 6)

Residual:  cat_1 + FFN = [1.299, -0.189, -1.130, 0.766]
LayerNorm: cat_1_final = [1.201, -0.301, -1.187, 0.701]
```

### After Block 2:

Block 2 has completely different W_Q, W_K, W_V, W1, W2. It sees cat_1_final as input and runs the same process with different weights.

```
cat_1_final = [1.201, -0.301, -1.187, 0.701]

(Block 2 attention + FFN + residuals + layernorm)

cat_2_final = [0.890, 0.124, -0.956, 1.102]   ← shifted again
```

### After Block 3:

```
cat_2_final = [0.890, 0.124, -0.956, 1.102]

(Block 3 attention + FFN + residuals + layernorm)

cat_3_final = [0.654, 0.441, -0.723, 0.987]   ← shifted again
```

**The progression:**
```
Block 0 (embedding): [0.700,  0.330, -0.310,  0.730]
Block 1:             [1.201, -0.301, -1.187,  0.701]
Block 2:             [0.890,  0.124, -0.956,  1.102]
Block 3:             [0.654,  0.441, -0.723,  0.987]
```

Each block reshapes the vector. The numbers aren't random drift — each transformation is learned to make the final layer 12 output maximally useful for predicting masked tokens during pre-training.

---

## 8.5 The [CLS] Token's Journey — Most Important Token

[CLS] is the most interesting token to trace because it has a special job: **by layer 12, it must represent the entire sentence.**

How does it achieve this? Through attention.

### Layer 1:
```
[CLS] attends to all tokens, but weights are fairly uniform
[CLS]_1 = weak blend of all token embeddings
```

### Layers 2-6:
```
[CLS] starts attending more selectively
It learns which tokens carry the most sentence-level information
[CLS]_6 = richer blend, starting to encode sentence meaning
```

### Layers 7-11:
```
[CLS] heavily attends to semantically important tokens
For "the cat sat on the mat":
  [CLS] → "cat" (0.31), "sat" (0.28), "mat" (0.22), others lower
[CLS]_11 = strong sentence-level representation
```

### Layer 12:
```
[CLS]_12 = final sentence embedding
Used directly for classification tasks
Its 768 dimensions encode the full meaning of the input
```

**This is why classification tasks use [CLS]** — it's the only token specifically designed, through 12 layers of attention over all other tokens, to aggregate global sentence meaning.

---

## 8.6 Probing Experiments — What Each Layer Actually Learns

Researchers have run **probing classifiers** on BERT — training a simple linear classifier on each layer's hidden states to test what linguistic information is encoded.

Results:

```
Layer  | Best captures
───────|──────────────────────────────────────────
1-2    | Basic token features, subword information
3-4    | POS tags (noun, verb, adjective)
5-6    | Syntactic chunking (noun phrases, verb phrases)
7-8    | Syntactic dependencies (subject, object)
9-10   | Semantic roles (who did what to whom)
11-12  | Coreference, long-range semantic relations
```

This layered hierarchy mirrors how linguists describe language:

```
Surface form → POS → Phrase structure → Dependencies → Semantics
     ↓           ↓          ↓                ↓             ↓
  Layers 1-2   3-4        5-6             7-8           9-12
```

BERT **rediscovered the structure of linguistics** purely from predicting masked words. Nobody told it that layer 4 should learn POS tags.

---

## 8.7 Why 12 Layers? Why Not 6 or 24?

BERT was released in two sizes:

```
BERT-base:  12 layers, 768 hidden, 12 heads → 110M params
BERT-large: 24 layers, 1024 hidden, 16 heads → 340M params
```

**BERT-base vs BERT-large on SQuAD (QA benchmark):**
```
BERT-base:  88.5 F1
BERT-large: 90.9 F1
```

Large wins, but costs 3× the compute and memory. The tradeoff:

```
More layers → captures more abstract relationships → better accuracy
More layers → more parameters → more data needed, more compute
```

6 layers (DistilBERT) gets ~97% of BERT-base performance at 40% the size — because many of BERT's 12 layers are partially redundant.

---

## 8.8 Parameter Count — Full BERT-base

Let's add everything up:

```
EMBEDDING LAYERS:
  Token embeddings:    30,522 × 768    =  23,440,896
  Position embeddings: 512 × 768       =     393,216
  Segment embeddings:  2 × 768         =       1,536
  LayerNorm (emb):     2 × 768         =       1,536
                                         ───────────
  Embedding total:                       23,837,184

PER TRANSFORMER BLOCK (× 12):
  Attention:
    W_Q, W_K, W_V:    3 × (768×768)   =   1,769,472
    W_O:               768×768         =     589,824
    Attention biases:  3×768 + 768     =       3,072
  LayerNorm 1:         2×768           =       1,536
  FFN:
    W1:                768×3072        =   2,359,296
    b1:                3072            =       3,072
    W2:                3072×768        =   2,359,296
    b2:                768             =         768
  LayerNorm 2:         2×768           =       1,536
                                         ───────────
  Per block:                             7,087,872
  × 12 blocks:                          85,054,464

POOLER (for [CLS] output):
  Linear:              768×768         =     589,824
  Bias:                768             =         768
                                         ───────────
  Pooler total:                            590,592

─────────────────────────────────────────────────────
TOTAL:                                   109,482,240
                                       ≈ 110M parameters
```

---

## 8.9 Receptive Field — How Far Can Each Layer See?

In CNNs, a layer can only "see" a local window. In Transformers, **every layer can see every token** — but the effective receptive field still grows with depth.

Here's why:

```
Block 1:  "bank" directly attends to "river" → direct connection

Block 2:  "bank" attends to Block 1's output of "river"
          Block 1's "river" already incorporated "I went to the"
          So "bank" now indirectly sees "I went to the river"

Block 3:  "bank" attends to Block 2 representations
          Which already have 2 layers of context baked in
          Effectively seeing the full sentence multiple times over
```

By block 12, every token has integrated context from every other token **through 12 rounds of attention**. The representations are extraordinarily rich.

---

## 8.10 The Complete Forward Pass — Everything Together

Let's write out the complete forward pass of BERT from raw text to final hidden states:

```
INPUT: "the cat sat"

Step 1 — Tokenize:
  ["[CLS]", "the", "cat", "sat", "[SEP]"]
  IDs: [101, 1996, 4937, 2938, 102]

Step 2 — Three Embeddings + Sum + LayerNorm:
  X_0 = LayerNorm(TokenEmb + PosEmb + SegEmb)
  Shape: [5 × 768]

Step 3 — Block 1:
  A_1 = LayerNorm(X_0 + MultiHeadAttention(X_0))
  X_1 = LayerNorm(A_1 + FFN(A_1))
  Shape: [5 × 768]

Step 4 — Block 2:
  A_2 = LayerNorm(X_1 + MultiHeadAttention(X_1))
  X_2 = LayerNorm(A_2 + FFN(A_2))
  Shape: [5 × 768]

  ... (repeat for blocks 3-11)

Step 5 — Block 12:
  A_12 = LayerNorm(X_11 + MultiHeadAttention(X_11))
  X_12 = LayerNorm(A_12 + FFN(A_12))
  Shape: [5 × 768]

OUTPUT:
  X_12[0] = [CLS] vector  → sentence-level representation
  X_12[1] = "the" vector  → contextual embedding of "the"
  X_12[2] = "cat" vector  → contextual embedding of "cat"
  X_12[3] = "sat" vector  → contextual embedding of "sat"
  X_12[4] = [SEP] vector  → end marker
```

---

## Chapter 8 Summary

```
12 blocks, each identical in structure, different in weights
Shape stays [seq_len × 768] throughout — only content changes

Layers 1-4:   syntax, POS, local patterns
Layers 5-8:   semantics, longer range
Layers 9-12:  abstract, task-relevant representations

[CLS] aggregates full sentence meaning across all 12 layers
Each block adds one round of context refinement
Residuals ensure gradients flow all the way back to layer 1
```

| | BERT-base | BERT-large |
|---|---|---|
| Blocks | 12 | 24 |
| Hidden dim | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |

---

Now you understand BERT's full architecture from tokens to final representations. But we haven't answered: **how did BERT learn all of this?** Nobody labeled 3.3 billion words with syntax trees or semantic roles. BERT learned everything from raw text alone.

# Chapter 8 — Stacking Transformer Blocks (Master Notes, Apple MLE Prep)

> Goal of this doc: know precisely where BERT's 110M parameters actually live (it's not where most people guess), correctly distinguish "receptive field" from what actually grows with depth in a Transformer (a common conflation with CNNs), and be able to defend the layer-by-layer linguistic story as an empirical *tendency*, not an architectural guarantee.

---

## 0. One-sentence version

> "Stacking 12 identical-in-structure, different-in-weights Transformer blocks doesn't change the shape of the representation (`[seq_len × 768]` throughout) — it repeatedly refines what those 768 numbers *mean*, with each block getting a full round of attention-then-refinement over an input that already carries everything the previous blocks discovered."

---

## 1. Shared vs. separate — kept, this was already correct

**Separate per block**: all Q/K/V/O projections, both FFN weight matrices, both LayerNorms' γ/β — every block learns its own independent transformation.
**Shared across all blocks**: the three embedding tables (token, position, segment) — these are only ever looked up once, at the very bottom, before block 1.

**Why this matters for an interview**: "Block 1's $W_Q$ has nothing to do with Block 7's $W_Q$" is worth stating explicitly because it corrects a natural (wrong) assumption that a "stack of identical layers" means *literally identical, weight-tied* layers, the way some other architectures (e.g. certain RNN variants, or weight-tied recurrent Transformers like Universal Transformer) do. BERT's blocks are identical only in *structure* (same operations, same shapes) — every one of the 12 blocks has its own independently-learned 7,087,872 parameters.

---

## 2. Parameter count — where BERT's 110M actually lives (the number most people get wrong)

The chapter's own arithmetic is correct (verified: 23,837,184 + 85,054,464 + 590,592 = 109,482,240). What's missing is the **category breakdown**, which reveals something most people guess wrong when asked "where are most of BERT's parameters?"

*(see the chart above)*

| Category | Params | % of total |
|---|---|---|
| Embeddings (token+position+segment+LN) | 23,837,184 | 21.8% |
| Attention (all 12 blocks: Q/K/V/O/biases/LN) | 28,348,416 | 25.9% |
| **Feed-forward network (all 12 blocks)** | **56,669,184** | **51.8%** |
| Pooler | 590,592 | 0.5% |

**The number worth memorizing for an interview**: the **FFN holds more than half of all of BERT's parameters** — more than attention, despite attention being the mechanism nearly every paper, blog post, and diagram (including most of this chapter series) spends the most explanatory effort on. Within a single block, FFN (4,722,432 params) outweighs attention (2,362,368 params) by roughly **2-to-1**. This is a genuinely common interview trap: "explain where BERT's parameters live" is often answered "mostly attention" by people who've absorbed the *conceptual* emphasis on attention from how the architecture is usually taught, without checking the actual parameter arithmetic.

**Why the FFN is so much bigger, mechanically**: the FFN's hidden expansion is `768 → 3072 → 768` (a 4x width increase in the middle), giving two large `[768×3072]`-class matrices per block, versus attention's `[768×768]`-class matrices. This directly connects back to the Chapter 4 discussion of where BERT's FLOPs go at $n=512$: the FFN (and other non-attention-score compute) dominates both parameter count *and* FLOPs at BERT's actual operating sequence length — attention-score computation is the smaller cost on both axes at $n=512$, only becoming FLOP-dominant well past $n \approx 4{,}600$ tokens (Chapter 4).

---

## 3. What actually "grows" with depth — fixing the receptive-field conflation

**The original chapter's Section 8.9 borrows "receptive field" language from CNNs, and this needs a precise correction, since it's a genuinely common point of confusion.**

**In a CNN**, a layer's receptive field is about literal *reachability* — a neuron in layer 1 can only see pixels in a small local window; a neuron in layer 5 can see a larger window, built up by stacking local windows. Depth is required just to let information from far-apart pixels ever interact at all.

**In a Transformer, this is categorically different**: because self-attention computes $QK^T$ between *every* pair of tokens (Chapter 4), **every single token can already directly attend to every other token starting at layer 1** — reachability is total from the very first block, not something depth builds up. There's no analogue of "layer 1 can't see far-away tokens yet."

**So what does actually grow with depth, if not reachability?** Two real things:

1. **Compositional richness of what's being attended to.** At layer 1, when "bank" attends to "river," it's attending to *river's raw embedding* (basically its context-free identity). At layer 2, when "bank" attends to "river" again, it's now attending to *river's layer-1 output* — which itself already incorporated information from "I," "went," "to," "the" via layer 1's attention. So indirect, multi-hop information genuinely does accumulate with depth — not because direct access requires more hops to *reach* it, but because each hop's *content* is progressively more processed and context-laden.
2. **Increasing abstraction of the function being computed.** Layer 1 can only compute simple functions of the raw embeddings (one round of attention + one FFN); layer 12 can compute a function of layer 11's already-heavily-processed output — allowing far more complex, composed transformations, similar in spirit to how deeper feed-forward networks generally can represent more complex functions than shallow ones, independent of any "reachability" argument.

**What if a task genuinely only needed local, single-hop information — would fewer layers work just as well?** Yes, and this is exactly the empirical finding behind DistilBERT-style results (Section 5 below) — tasks that don't need deep compositional/abstract reasoning don't benefit as much from extra layers, which is part of why 6 layers can retain ~97% of 12-layer performance on many benchmarks: the *marginal* value of additional depth depends on how much genuinely deep compositional structure the task requires, not on unlocking access to tokens that were otherwise unreachable.

---

## 4. The layer-by-layer linguistic story — an empirical tendency, not a guarantee

**Section 8.6's probing-classifier table (layers 1-2 → subword, 3-4 → POS, etc.) is a real empirical finding, worth stating carefully rather than as a strict architectural law.**

**What "probing classifier" actually means, precisely**: researchers freeze a pre-trained BERT, extract hidden states at each layer, and train a small separate linear classifier on top of *just that layer's* frozen representations to predict some linguistic label (POS tag, dependency relation, coreference link, etc.). If a simple linear classifier can predict the label well from layer $k$'s hidden states, that's evidence layer $k$'s representations *linearly encode* that information — not proof the model "intends" to compute that specific linguistic category, and not proof no other layer contains related information.

**Why this needs a caveat, specifically**: 
- The boundaries are fuzzy and overlapping, not the clean discrete bands the table suggests — a linear classifier can typically extract *some* signal about POS tags from almost every layer, just with varying accuracy; the table shows where accuracy peaks, not where the information exclusively lives.
- Different attention *heads within the same layer* specialize differently (Chapter 5) — a single layer isn't monolithically "the syntax layer," it's 12 heads that may individually specialize in quite different sub-patterns, whose *aggregate* linear-probe accuracy happens to peak on syntax-adjacent tasks at that depth.
- This hierarchy emerged **purely from the MLM pre-training objective**, with no explicit supervision pushing toward this specific linguistic decomposition — it's a genuinely interesting empirical discovery, but it's a property of *what turned out to help predict masked words well*, not a designed-in curriculum. A model trained on a different objective, or a different language with different structure, isn't guaranteed to reproduce the identical layer-to-task mapping.

**What if you needed a specific linguistic feature (say, dependency parses) for your downstream task — should you always use layer 7-8's representations specifically, per this table?** This is a real, useful heuristic that some practitioners use (extracting mid-layer representations for syntax-heavy tasks rather than always using layer 12), but it should be validated empirically on your specific model/task/domain rather than assumed to hold exactly — the original probing studies were run on specific BERT checkpoints and benchmarks, and the precise layer boundaries can shift somewhat across model sizes, domains, and even different pre-training runs of "the same" architecture.

---

## 5. Why 12 layers, and why DistilBERT's result isn't just "smaller is fine"

**The chapter's DistilBERT claim (6 layers, ~97% of BERT-base performance, 40% of the size) deserves a mechanism, not just the headline number** — this is a real, common interview follow-up ("if 6 layers gets 97%, why did they bother with 12?").

**Why simply training a 6-layer model from scratch on masked-LM doesn't get you the same result**: DistilBERT isn't just "BERT with fewer layers, trained the normal way." It uses **knowledge distillation**: the 6-layer "student" is trained not just on the standard MLM objective, but also to match the *full probability distribution* the 12-layer "teacher" model produces over the vocabulary for each masked position — not just the single correct answer. 

**Why matching the teacher's full distribution helps, simplified**: a hard label only tells the student "the correct word was 'cat'." The teacher's full softmax output additionally tells the student something like "and 'cat' was 0.6 likely, but 'dog' was a reasonable second guess at 0.25, while 'car' was essentially ruled out at 0.001" — this "soft label" carries much richer information about *how the teacher generalizes* (which mistakes are more vs. less reasonable) than a hard one-hot label ever could. This extra signal is what lets a smaller model recover a disproportionate amount of a larger model's performance — it's learning from the bigger model's *reasoning*, not just re-deriving everything from raw text and hard labels alone, the way the original 12-layer model had to.

**What if you skipped distillation and just trained a 6-layer BERT from scratch on the same masked-LM data?** You'd typically get meaningfully worse than DistilBERT's ~97% figure — you'd be asking a smaller-capacity model to independently rediscover everything a 12-layer model rediscovered, from the same weak (hard-label) training signal, without benefiting from the teacher's already-compressed, more informative view of the problem. This is why "distilled 6-layer" and "trained-from-scratch 6-layer" are genuinely different things with different expected performance, despite identical final architectures.

---

## 6. Caveat on Section 8.4's numerical trace — asserted, not computed

**Worth flagging explicitly**: the original chapter's `cat_2_final = [0.890, 0.124, -0.956, 1.102] ← shifted again` and `cat_3_final = [...] ← shifted again` are presented as if computed, but no actual $W_Q, W_K, W_V, W_1, W_2$ matrices for blocks 2 and 3 are given anywhere in the source material — unlike Block 1's numbers, which trace all the way back to real (if simplified/toy) weight matrices from Chapters 4, 6, and 7. **These specific numbers for blocks 2-3 should be treated as illustrative placeholders showing "the vector keeps changing," not as a genuine worked example you could reproduce by hand.** If asked to actually demonstrate block 2's transformation in an interview, you'd need to either invent explicit toy weight matrices for block 2 (as Chapters 4-7 did for block 1) or be clear you're speaking qualitatively about the *pattern* (continued transformation, same shape) rather than defending these particular numbers as derived.

---

## 7. Design-choice summary table, boosted

| Design choice | Why | What breaks without it |
|---|---|---|
| Identical structure, independent weights per block | Each block can specialize its own transformation while keeping the architecture simple and uniform to implement/scale | Weight-tying across blocks (as some other architectures do) would force every block to compute the *same* function, unable to specialize by depth |
| Shape stays `[seq_len × 768]` at every block boundary | Lets any number of blocks be stacked without needing shape-adapting layers in between | A changing shape between blocks would require extra projection layers just to make dimensions match, adding parameters with no representational benefit |
| 12 blocks for BERT-base (not fewer/more) | Empirically tuned tradeoff between representational depth/abstraction and compute/data/parameter cost | Too few → insufficient depth for compositional/abstract reasoning tasks; too many → diminishing returns relative to cost (see DistilBERT's ~97%-at-6-layers result) |
| FFN's 4x hidden expansion (768→3072→768) | Gives the FFN (Chapter 6) enough intermediate capacity to compute rich per-token nonlinear transformations | This expansion is also *why* the FFN, not attention, ends up holding the majority of BERT's parameters — a direct consequence of this width choice |

---

## 8. Diagnostics — misconceptions to pre-empt

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "Most of BERT's parameters are in the attention mechanism" | The FFN holds 51.8% of total parameters vs. attention's 25.9% — roughly 2-to-1 within each block | Attention gets the conceptual spotlight in how Transformers are usually explained, but the FFN is where most of the actual weight matrices (and their storage/compute cost) live |
| "Deeper layers can 'see' tokens that were unreachable in earlier layers, like a CNN's receptive field" | Self-attention gives every token direct access to every other token starting at layer 1 — there's no reachability gap depth needs to close | What grows with depth is compositional richness of what's attended to and abstraction of the function computed, not literal token-to-token reachability |
| "Layer 4 is *the* POS-tagging layer, layer 8 is *the* dependency layer, etc." | Probing-classifier peaks are empirical tendencies with fuzzy, overlapping boundaries — not a clean architectural partition, and different heads within a layer specialize differently | The layer-to-linguistic-task table describes where linear-probe accuracy peaks for a specific studied model, not a guaranteed universal decomposition |
| "DistilBERT proves you can just always use a smaller model with barely any cost" | DistilBERT's ~97% figure specifically relies on knowledge distillation from a trained 12-layer teacher's soft output distribution — a 6-layer model trained from scratch on hard labels alone typically underperforms this | The performance retention is a property of the distillation *process*, not simply a property of "6 layers being basically enough" |
| "Block 2 and 3's numbers in the numerical trace were actually computed the way Block 1's were" | No explicit weight matrices for blocks 2-3 are given anywhere in the source material — those numbers illustrate the *pattern* of continued transformation, not a genuine reproducible calculation | Only Block 1's numbers trace back to real (toy) weight matrices from earlier chapters; treat blocks 2-3's numbers as placeholders for "and this keeps happening," not as derived values |

---

## 9. Q&A practice set (self-test — this chapter had no Q&A in the source; answers below the line)

**Q1 (easy).** What stays exactly the same shape from the embedding layer all the way through block 12, and what changes instead?

**Q2 (easy).** Name one thing that is shared across all 12 blocks, and one thing that is separate per block.

**Q3 (medium).** In one sentence, why does the FFN hold more parameters than attention within a single BERT block?

**Q4 (medium — calculation).** If BERT-large uses 24 blocks instead of 12, and each block's parameter count scales up somewhat due to the larger 1024 hidden dimension, would you expect the FFN-vs-attention parameter ratio (roughly 2:1 in BERT-base) to stay about the same, increase, or decrease? (Reason from the shapes, not memorized BERT-large numbers.)

**Q5 (medium).** Why is "receptive field" a somewhat misleading term to borrow from CNNs when describing what grows with depth in a Transformer?

**Q6 (hard).** A colleague claims: "Since probing studies show layer 4 encodes POS tags best, we should always extract layer 4's hidden states specifically whenever we need POS-tag-relevant features from BERT." What's the issue with treating this as a firm rule?

**Q7 (hard).** Explain, mechanically, why a 6-layer model trained via knowledge distillation from a 12-layer teacher tends to outperform an *identical* 6-layer model trained from scratch on the same raw masked-LM data.

**Q8 (hard — spot the bug).** Someone asks you to explain how BERT's "receptive field grows with depth, letting later layers see tokens earlier layers couldn't reach." Correct this claim precisely — what's actually true, and what specifically is wrong about the framing?

---
---

### Answers

**A1.** The shape `[seq_len × 768]` stays identical at every stage — after embedding, after block 1, after block 12, all the same `[5×768]` for a 5-token input. What changes is the *content*: what those 768 numbers per token actually represent gets progressively more contextually and compositionally refined as it passes through each block's independently-learned transformation.

**A2.** Shared across all 12 blocks: the token/position/segment embedding tables (looked up once, at the bottom, before any block runs). Separate per block: every block's own Q/K/V/O projection matrices, FFN weight matrices, and LayerNorm γ/β parameters — no weights are reused between blocks.

**A3.** The FFN's hidden layer expands to 4x the model width (768→3072) before compressing back down, producing two large `[768×3072]`-class weight matrices per block, whereas attention's largest matrices are `[768×768]`-class — the FFN's wider intermediate dimension is what gives it roughly double attention's parameter count per block.

**A4.** You'd expect the ratio to shift somewhat toward FFN dominating even more, or at least stay similarly FFN-heavy — because FFN parameter count scales with $d_{model}^2$ (from the two `[d × 4d]`-class matrices) while attention's Q/K/V/O parameter count also scales with $d_{model}^2$ (from `[d×d]`-class matrices, following the same "same total params regardless of head count" property from Chapter 5) — both scale quadratically with $d_{model}$, so the *ratio between them* (FFN's ~4x-wider matrices vs. attention's ~1x-wide matrices) is actually independent of $d_{model}$ and depends only on the FFN's fixed 4x expansion factor. Reasoning from the shapes: FFN ≈ $2 \times d \times 4d = 8d^2$, attention (Q+K+V+O) ≈ $4 \times d \times d = 4d^2$ — a 2:1 ratio regardless of whether $d=768$ or $d=1024$, so you'd expect the ratio to stay roughly the same across BERT-base and BERT-large.

**A5.** "Receptive field" in a CNN describes literal reachability — which pixels a neuron can possibly be influenced by, which genuinely does require more layers to expand. In a Transformer, self-attention already gives every token direct access to every other token starting at layer 1 (there is no reachability limit to expand in the first place), so nothing analogous to "unreachable until deeper" exists. What actually changes with depth is the richness/abstraction of the information being attended to (since later layers attend over increasingly processed, context-laden representations), not which tokens can be reached at all.

**A6.** Probing-classifier results describe where a linear classifier's accuracy *peaks* on a specific studied model/dataset — the boundaries are fuzzy and overlapping (other layers usually carry *some* extractable POS signal too, just less strongly), individual attention heads within that layer may specialize quite differently from each other, and the exact peak layer isn't guaranteed to transfer identically across different model sizes, domains, or even different training runs of "the same" architecture. Treating layer 4 as *the* canonical POS-tag layer as a hard rule risks either underperforming (if this specific model/checkpoint's actual peak differs) or missing complementary signal available in nearby layers — it's a reasonable empirically-motivated starting point to test, not a guarantee to assume without verification.

**A7.** A hard label only tells the student model the single correct answer for each masked position (e.g., "the answer was 'cat'"), giving a sparse training signal (right or wrong, nothing in between). The teacher's full softmax output over the vocabulary additionally encodes *how confidently and toward which alternatives* the teacher would have guessed (e.g., "cat" 0.6, "dog" 0.25, everything else negligible) — this "soft label" carries substantially more information per training example about the *shape* of the correct generalization (which errors are reasonable vs. unreasonable) than a one-hot hard label does. Training the smaller student to match this richer target lets it benefit from the larger teacher's already-refined understanding of the problem, rather than having to independently rediscover everything from the sparser raw signal a from-scratch 6-layer model would be limited to.

**A8.** What's actually true: self-attention gives every token in every layer, starting at layer 1, direct access to every other token's representation — full pairwise reachability exists from the very first block, with no "unreachable until deeper" tokens the way there would be in a CNN. What's wrong about the framing: describing this as a "growing receptive field" borrows a CNN concept (limited local reachability that expands with depth) that doesn't apply to Transformers at all — nothing about *which* tokens can be attended to changes with depth. What legitimately does change with depth is the *content* being attended to (later layers attend over progressively more processed, context-rich representations of those same always-reachable tokens) and the *complexity of the function* being computed (deeper layers can compose more elaborate transformations on top of already-refined inputs) — richness and abstraction grow, not reachability.

---

## 10. Quick recap card (last-minute review)

- **Shape invariant, content refined**: `[seq_len × 768]` unchanged from embedding through block 12; only what those numbers mean changes.
- **Shared vs. separate**: only the 3 embedding tables are shared across blocks; every block's Q/K/V/O/FFN/LayerNorm weights are independently learned.
- **Where BERT's 110M parameters actually live** (memorize this): Embeddings 21.8%, Attention 25.9%, **FFN 51.8%** (the majority, ~2:1 over attention within a block), Pooler 0.5% — a common interview trap is assuming attention dominates.
- **"Receptive field" doesn't really apply**: every token can attend to every other token starting at layer 1 (unlike CNNs) — what grows with depth is compositional richness and functional abstraction, not reachability.
- **The layer-to-linguistics probing table is an empirical tendency**, with fuzzy/overlapping boundaries and head-level variation within a layer — not a strict architectural partition.
- **DistilBERT's ~97%-at-6-layers relies on knowledge distillation** (matching the teacher's full soft output distribution, not just hard labels) — a from-scratch 6-layer model on hard labels alone typically underperforms this.
- **Caveat on the chapter's own numeric trace**: only Block 1's numbers in Section 8.4 are genuinely traceable to real toy weight matrices from earlier chapters; Blocks 2-3's numbers are illustrative placeholders, not reproducible calculations.

*(Chapter 9 picks up here: the two self-supervised pre-training objectives — Masked Language Modeling and Next Sentence Prediction — that taught this entire 12-block stack everything it knows, with no human-labeled syntax or semantics anywhere in the process.)*
