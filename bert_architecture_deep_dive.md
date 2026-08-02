# BERT Architecture Deep Dive — Layers, Norm, Residuals, Bottlenecks

Companion to the phases doc. This one zooms into **one encoder block** and asks, for every component: what bottleneck does this solve, and what's the alternative that was rejected.

## One encoder block, end to end

```
Input x (seq_len x 768)
   │
   ├────────────────┐
   ▼                │
Multi-Head          │ (residual)
Self-Attention      │
   ▼                │
  Add  ◄─────────────┘
   ▼
LayerNorm
   │
   ├────────────────┐
   ▼                │
Feed-Forward         │ (residual)
(768→3072→768, GELU) │
   ▼                │
  Add  ◄─────────────┘
   ▼
LayerNorm
   ▼
Output (seq_len x 768)
```

This block is repeated 12 times (BERT-base) or 24 times (BERT-large), stacked directly — output of block *n* is input of block *n+1*. Every component here exists because of a specific failure mode that shows up without it.

---

## 1. Multi-Head Self-Attention — the bottleneck it replaces

**The bottleneck it solves:** Before Transformers, RNNs processed sequences step by step, so information from token 1 had to survive 500 sequential updates to influence token 500 — gradients and information both decay over that many steps (the classic long-range dependency problem). CNNs fixed this partially with wider receptive fields, but still needed many stacked layers to connect distant tokens. Self-attention lets **any token attend directly to any other token in one step**, regardless of distance — O(1) path length between any two positions instead of O(n).

**Why multiple heads, not one:** A single attention head produces one weighted-average view of the sequence per token. But language has multiple simultaneous relationship types happening at once (subject-verb agreement, coreference, local syntax, topic-level relevance). One softmax distribution can't cleanly represent several unrelated relational patterns at once — it would have to blend them into a single blurry average. Splitting into `h` heads (12 heads × 64 dims = 768 for BERT-base) lets each head learn a narrower, specialized attention pattern; concatenating and projecting back combines these specialized views. Empirically, later analysis (e.g., attention-visualization papers) shows different heads really do specialize — some track syntactic dependency, some track positional/local patterns, some are close to a no-op.

**Why not just make one head 768-dimensional instead of 12x64?** You'd get more capacity per head but no forced specialization — the same "blurred average" issue reappears, and you lose the multi-view redundancy that makes training more robust (ablating a single head barely hurts, because others compensate).

**The bottleneck it introduces (trade-off):** Self-attention is O(n²·d) in time and memory — every token attends to every other token. This is *the* main scaling limit of the architecture. See the dedicated section near the end.

---

## 2. Residual (skip) Connections — the bottleneck they solve

**What:** `output = Sublayer(x) + x`, applied around both the attention sublayer and the FFN sublayer.

**The bottleneck it solves:** Very deep networks suffer from the **degradation problem** — beyond a certain depth, adding more layers makes training *harder*, not just slower, even with proper initialization (this is separate from vanishing gradients, and was the empirical finding behind ResNet). Gradients have to flow backward through every layer's nonlinearity and matrix multiply; without a shortcut, deep stacks become numerically unstable and hard to optimize.

**Why addition, specifically:** A residual connection gives the gradient a direct, unobstructed path back to earlier layers (`d(output)/d(x)` always has an identity term, so it can never fully vanish through that block). It also reframes what each sublayer needs to learn: instead of learning the *entire* transformation from scratch, a sublayer only needs to learn the **residual** — the adjustment/delta to make to its input. This is a much easier optimization target, especially early in training when a sublayer's ideal behavior might be close to "pass the input through mostly unchanged."

**Why not without it:** Empirically, transformers deeper than a few layers fail to train at all without residuals — this is a documented ablation in the original Transformer paper, not a theoretical nicety.

---

## 3. LayerNorm — the bottleneck it solves, and why not BatchNorm

**What:** Normalizes across the feature dimension (per token, per sample) — mean 0, variance 1, then a learned scale/shift.

**The bottleneck it solves:** Deep stacks accumulate activations that drift in scale layer over layer (internal covariate shift), which slows training and makes learning rates hard to tune. Normalizing keeps each layer's input distribution stable, letting you train faster and with a higher learning rate.

**Why LayerNorm and not BatchNorm (this is a very common interview question):**
- BatchNorm normalizes across the **batch dimension**, per feature — it needs a reasonably large, consistent batch to estimate stable statistics, and its behavior differs between train (batch stats) and inference (running stats).
- NLP sequences have **variable length**, and padding tokens contaminate batch statistics if you're not careful. Small or variable batch sizes (common with long sequences due to memory limits) make BatchNorm's batch statistics noisy and unreliable.
- LayerNorm normalizes each token's own feature vector independently of every other token/example in the batch — no cross-example dependency, no train/inference discrepancy, robust to variable batch size and sequence length. This is a much better fit for sequence models.

**Placement — post-norm (BERT/original Transformer) vs pre-norm (GPT-2 and later):** BERT applies norm *after* the residual add (`LayerNorm(x + Sublayer(x))`). This works but empirically makes very deep models harder to train — gradients can still grow unstably through many post-norm blocks. Later architectures moved to **pre-norm** (`x + Sublayer(LayerNorm(x))`), which keeps the residual stream numerically cleaner across many more layers and enables much deeper, more stable training (part of why post-2019 large LMs almost all use pre-norm). Knowing this distinction — and that BERT specifically made the *older* choice — is a good signal in an interview that you understand this wasn't a static design, it evolved.

---

## 4. Feed-Forward Sublayer — the bottleneck it solves

**What:** Two linear layers per position with a GELU nonlinearity in between, expanding 768 → 3072 → 768 (4x expansion).

**The bottleneck it solves:** Self-attention is fundamentally a **weighted average operation** — every output is a linear combination of value vectors. Stack attention layers alone (no FFN) and you're still limited to compositions of linear mixing; there's no per-token nonlinear transformation adding new representational capacity. The FFN sublayer is where the model does **per-token nonlinear feature transformation** — attention decides *what to look at*, the FFN decides *what to do with what you found*. Interestingly, follow-up interpretability work (e.g. on "transformer feed-forward layers as key-value memories") suggests FFN layers behave a lot like associative memory lookups, storing learned facts/patterns.

**Why 4x expansion specifically:** Widening then narrowing gives the nonlinearity more room to represent complex functions before compressing back to the residual stream's dimension — a bottleneck-then-expand-then-compress pattern common across deep learning (similar spirit to inverted residuals in vision models). 4x is largely an empirical sweet spot from the original paper, balancing capacity against the fact that FFN parameters dominate the parameter budget (see table below).

**Why GELU instead of ReLU:** ReLU has a hard, non-smooth cutoff at 0 (zero gradient for all negative inputs — "dying ReLU"). GELU is a smooth approximation that weights inputs by their percentile under a Gaussian, giving a smoother gradient landscape and empirically better convergence for Transformer-scale models. It became close to a de facto standard for Transformers after BERT.

---

## 5. Depth — why 12 layers (base) / 24 (large), and why not more or fewer

**What depth buys you:** Layers compose hierarchically. Empirical probing studies on BERT found rough patterns like: lower layers capture more **surface/syntactic** features (POS, local structure), middle layers capture **syntactic dependencies**, upper layers capture more **semantic/task-relevant** abstractions. Each additional layer lets the model build a more abstract representation on top of the previous layer's output — similar to how deeper CNNs build higher-level visual features from edges → textures → parts → objects.

**Why not fewer:** Too shallow, and representations stay closer to lexical/local patterns — the model can't build the compositional/semantic abstractions many downstream tasks need (this is roughly why the original paper reports BERT-large beating BERT-base by a consistent margin across benchmarks, especially harder ones).

**Why not much deeper:** Three compounding costs:
1. **Diminishing returns** — the jump from 12→24 layers helps meaningfully; naively going much further has historically shown shrinking gains relative to the added compute, unless paired with other changes (better norm placement, more data, architecture tweaks) — this is part of why post-BERT scaling work changed more than just layer count.
2. **Optimization difficulty** — even with residuals and LayerNorm, post-norm architectures get harder to train stably as depth grows (see the pre-norm discussion above).
3. **Compute/latency** — every extra layer adds a fixed compute and memory cost, multiplied by every inference call in production — a real bottleneck for latency-sensitive serving, not just training.

**Parameter budget (rough, BERT-base, 12 layers, d=768, heads=12, ffn=3072):**

| Component | Approx. params | Share |
|---|---|---|
| Token + position + segment embeddings | ~23M | ~21% |
| Self-attention (Q,K,V,O projections) × 12 layers | ~28M | ~26% |
| Feed-forward × 12 layers | ~56M | ~51% |
| LayerNorm params | negligible | ~0% |
| **Total** | **~110M** | 100% |

Takeaway most people miss in interviews: **the FFN sublayers, not attention, hold the majority of BERT's parameters.** Attention gets the conceptual spotlight because it's architecturally novel, but the FFN is where most of the model's capacity actually lives.

---

## 6. The real scaling bottleneck: O(n²) attention, and why it caps sequence length at 512

**The bottleneck:** For a sequence of length n, self-attention computes an n×n attention matrix (every token scores every other token) — O(n²·d) time and O(n²) memory. Double the sequence length and you 4x the attention compute/memory, not 2x.

**Why BERT caps at 512 tokens:** Partly a direct consequence of this quadratic cost (longer sequences become expensive fast), and partly because learned position embeddings (Phase 2 in the companion doc) have a fixed-size lookup table trained only up to position 512 — the model has literally never seen or learned an embedding for position 513, so it can't generalize past that length without retraining or interpolation tricks.

**Why not fixed in BERT itself:** At BERT's release (2018), this trade-off was considered acceptable for its target tasks (sentence/paragraph-level classification, QA on short passages) — most NLU benchmarks fit comfortably under 512 tokens. Fixing the quadratic bottleneck (sparse attention, linear attention, sliding-window attention, retrieval-augmented approaches) was left to later architectures (Longformer, BigBird, Reformer, etc.) specifically built for long-document tasks — a good example of "solve the problem you actually have first," and a good talking point if asked "what would you change about BERT."

---

## Why-vs-why-not, summary table

| Component | Why it's there (bottleneck solved) | Why not the alternative |
|---|---|---|
| Multi-head self-attention | O(1) path length between any two tokens vs. O(n) in RNNs; multiple relation types in parallel | Single head → blurred, unspecialized attention pattern |
| Residual connections | Fixes degradation problem in deep stacks; gives gradients a direct path | Without it, stacks >4-6 layers become very hard to train |
| LayerNorm (not BatchNorm) | Per-token normalization independent of batch/seq-length variability | BatchNorm needs stable batch statistics — unreliable with variable-length, padded NLP batches |
| Post-norm placement (BERT specifically) | Simpler, matches original Transformer, worked fine at 12-24 layers | Pre-norm scales more stably to much deeper stacks — BERT just didn't need to go that deep |
| Feed-forward sublayer | Adds per-token nonlinear capacity attention alone can't provide | Attention-only stack stays a linear-mixing operation, capped expressiveness |
| GELU activation | Smooth gradient, no dead-neuron cutoff | ReLU's hard zero cutoff loses gradient signal for negative inputs |
| 12 / 24 layers | Hierarchical feature composition (syntax → semantics) | Deeper = diminishing returns + harder optimization + serving latency cost |
| 512 token cap | Bounds the O(n²) attention cost to something tractable in 2018 hardware | Removing the cap needs a different attention mechanism entirely (sparse/linear attention) — out of scope for BERT's original design goals |

---

## Rapid-fire interview Q&A

**Q: Why LayerNorm instead of BatchNorm in Transformers?**
A: LayerNorm normalizes per-token across features, independent of batch composition — robust to variable sequence length and small/uneven batches, which BatchNorm's cross-example batch statistics are not.

**Q: What's the actual bottleneck that limits BERT's max sequence length?**
A: Quadratic O(n²) memory/compute in self-attention, compounded by learned (non-extrapolating) position embeddings capped at 512.

**Q: Where do most of BERT's parameters live?**
A: The feed-forward sublayers (~half of total params), not the attention projections — a commonly missed fact.

**Q: What would you change if redesigning BERT today?**
A: Pre-norm instead of post-norm for more stable deep training, drop NSP (per RoBERTa's findings), and use a sparse/linear attention variant to remove the 512-token ceiling.

**Q: Why do residual connections help even when LayerNorm is also stabilizing activations?**
A: They solve different problems — LayerNorm stabilizes activation *scale*, residuals fix the *gradient flow / optimization difficulty* of stacking many nonlinear transformations; you need both, not either/or.
