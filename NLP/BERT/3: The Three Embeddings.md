# Chapter 3 — The Three Embeddings (Master Notes, Apple MLE Prep)

> Goal of this doc: explain from memory why BERT needs three separate embedding tables instead of one, do the LayerNorm and sinusoidal-encoding math by hand, defend "sum not concatenate" with numbers, and correctly describe what's frozen vs. trainable at each stage of the model's life.

---

## 0. One-sentence version

> "A token ID alone tells the model *what* word this is but nothing about *where* it sits or *which sentence* it belongs to, so BERT looks up three separate learned vectors per token — identity, position, and segment — and sums them into one 768-d vector before any Transformer layer runs."

---

## 1. Why three tables and not one

**The core issue**: a token embedding table is a lookup by *identity only* — row 4937 for "cat" is the same row no matter where "cat" appears in the sequence or which sentence it's in. But meaning depends on more than identity: "cat" at position 2 is a different thing to reason about than "cat" at position 200, and "cat" in the question vs. "cat" in the passage matter differently for QA. Rather than building one enormous table indexed by (word, position, segment) — which would be gigantic and mostly empty (most words never appear at most positions) — BERT factors the problem into three small, independent tables and **adds** their contributions together.

**What if we just used one combined table, indexed by (token, position) pairs?** Size explodes: `30,522 tokens × 512 positions = ~15.6M rows` versus `30,522 + 512 = 31,034` rows split across two separate tables — roughly 500x more parameters for a table that's mostly redundant (the vast majority of (word, position) combinations are never seen in training, so most rows would stay near-randomly-initialized forever). Factoring into separate additive tables lets each one learn a **general-purpose** signal (this position tends to mean X, regardless of which word; this word tends to mean Y, regardless of where it sits) that composes across *any* combination, including combinations never literally seen in training.

---

## 2. Token embeddings — identity

**Table shape**: `[30,522 × 768]` → **~23.4M parameters**, the single largest weight matrix in BERT-base (bigger than any individual Transformer block's weights).

**What it captures, precisely**: this is the model's *prior* on a token's meaning, before any context is applied — the "default sense" a word gets averaged over all its training occurrences. This is worth noticing: **token embeddings alone have exactly the same context-blindness problem as Word2Vec** (Chapter 1) — one fixed row per token ID. The entire reason BERT still ends up context-sensitive is that this static starting vector then gets *transformed* by 12 layers of self-attention, which is where the real contextual work happens (see Section 7 below). The embedding table is the starting point, not the destination.

---

## 3. Positional embeddings — order

### 3.1 The problem, restated precisely

Self-attention computes $QK^T$ between all pairs of tokens (Chapter 1) — this operation is **permutation-invariant**: shuffle the input tokens and you get the exact same set of pairwise scores, just relabeled. Nothing about the attention *mechanism itself* encodes sequence order. Without an explicit position signal injected somewhere, "the cat sat" and "sat cat the" would be mathematically identical to the model — you'd have accidentally rebuilt Bag-of-Words (Chapter 1) with far more compute.

### 3.2 Two ways to inject position — simplified equations

**Original Transformer (2017): fixed sinusoidal encoding.**

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Plain-language read of every term:
- $pos$ — the token's position in the sequence (0, 1, 2, ...).
- $i$ — which pair of dimensions in the embedding vector you're computing (dimensions come in sin/cos pairs).
- $d$ — total embedding dimension (768 for BERT-base).
- $10000^{2i/d}$ — a **wavelength control**. For small $i$ (early dimension pairs), this term is small, so the sine/cosine oscillates *fast* as position increases — sensitive to small position changes. For large $i$ (later dimension pairs), this term is huge, so the wave oscillates *very slowly* — only sensitive to large position changes. Stacking many different wavelengths side by side across dimensions means the resulting vector encodes position at multiple resolutions simultaneously, roughly like binary digits encode a number at different scales, except smooth and continuous.

**One tiny worked numeric example** ($d=4$, so 2 sin/cos pairs, at $pos=2$):
```
i=0: PE(2,0) = sin(2 / 10000^(0/4)) = sin(2/1)   = sin(2)   ≈  0.909
     PE(2,1) = cos(2 / 10000^(0/4)) = cos(2/1)   = cos(2)   ≈ -0.416
i=1: PE(2,2) = sin(2 / 10000^(2/4)) = sin(2/100) = sin(0.02) ≈  0.020
     PE(2,3) = cos(2 / 10000^(2/4)) = cos(2/100) = cos(0.02) ≈  0.9998
```
→ position vector for pos=2 ≈ `[0.909, -0.416, 0.020, 0.9998]`. Notice dimension pair 0 (i=0) already swung through most of a full cycle by position 2, while dimension pair 1 (i=1) has barely moved — that's the multi-resolution property in action.

**BERT: fully learned positional embeddings.**

Table shape `[512 × 768]` — **~393K parameters**. Position 0 gets its own row, randomly initialized, updated by backprop exactly like any other weight — no formula, no hand-crafted structure. This is a much smaller table than the token embedding table (512 rows vs. 30,522).

### 3.3 Why BERT chose learned over the sinusoidal formula — and the real tradeoff

**The sinusoidal design's whole selling point** was letting a Transformer *extrapolate* to sequence lengths longer than anything seen in training — since the formula is defined for any $pos$, not just ones seen during training, a model could in principle handle position 1000 even if trained only up to 500. **BERT doesn't need that property**, because it has a hard, fixed 512-position ceiling by architectural design anyway (Section 3.4) — there's no "beyond training length" case to handle gracefully. Given that the extrapolation benefit is moot, BERT trades it away for a real benefit: learned embeddings can shape themselves to whatever positional patterns actually help the *specific* pre-training data and objective, rather than being locked into a fixed mathematical form decided in advance.

**What if you tried to fine-tune a BERT-style model on sequences longer than 512?** You cannot, without surgery — there's no row 512 to look up (see below). You'd need to either truncate, use a sliding-window approach (chunk + aggregate), or replace/extend the positional embedding table and re-train it (some later models, e.g. via "position interpolation," do stretch a learned table to cover new lengths, but that's a deliberate architectural intervention, not something you get for free the way sinusoidal extrapolation offers).

### 3.4 The 512 limit — mechanically, not just "the rule"

The positional embedding table literally has 512 rows: `[512 × 768]`. Token position 512 has no corresponding row to look up — this isn't a soft configuration value, it's the fixed shape of a trained weight matrix. Feeding in a longer sequence either throws an index error or (if a library silently truncates for you) discards content without necessarily telling you.

---

## 4. Segment embeddings — which sentence

**Table shape**: `[2 × 768]` — only **1,536 parameters**, tiny compared to the other two tables. Just two rows: Segment A, Segment B.

**Why the model can't infer this from token or position embeddings alone**: token embeddings only know word identity; positional embeddings only know a token's absolute index (0-511) in the *whole concatenated* input — position 8 is position 8 whether it's still inside sentence A or already into sentence B, so on its own it says nothing about which sentence that is. Only the segment embedding directly answers "which of the two segments is this token part of," and pairing it with `[SEP]` gives the model a redundant, explicit double-signal for the boundary (Chapter 2, Section 4).

**What if there were 3+ segments instead of 2?** You'd need a bigger segment table — BERT's pre-training objectives (masked LM + Next Sentence Prediction) and most of its downstream tasks are fundamentally two-sequence problems (question+passage, premise+hypothesis, sentence pair classification), so two rows was sufficient for what the model was built to do. Some later models drop the segment table entirely (see Section 8).

**For single-sentence tasks**: every token simply gets the Segment A row; Segment B's row exists in the table but is unused for that input.

---

## 5. Combining the three — sum, not concatenate

### 5.1 Why sum wins on parameter count, with actual numbers

**If you concatenated instead**: `768 (token) + 768 (position) + 768 (segment) = 2,304`-dimensional vector per token. Every downstream weight matrix in every Transformer layer — the $Q$, $K$, $V$ projections, the feed-forward layers, everything — would need its input dimension to match 2,304 instead of 768, roughly **tripling** the parameter count of every single layer in the network (Transformer layers are stacked 12 times in BERT-base — this cost compounds).

**Summing keeps the width at exactly 768** throughout the whole network. The three embeddings are added element-wise into the *same* 768-dimensional space, and different dimensions within that space are free to specialize — some ending up more identity-sensitive, some more position-sensitive, some more segment-sensitive — learned automatically via backprop, with no explicit partitioning enforced by the architecture.

**What if summing caused destructive interference — the three signals canceling each other out?** In principle a sum could lose information (two very different signals landing on the same coordinate could partially cancel). In practice this doesn't meaningfully happen because (a) the embedding space is high-dimensional (768-d) — there's plenty of room for the three tables to settle into largely non-overlapping directions during training, and (b) backprop actively *optimizes against* destructive interference, since any interference that hurt the downstream task's loss would be penalized and trained away. Empirically, summed embeddings work about as well as concatenation at a fraction of the cost — this is why virtually every Transformer variant since (RoBERTa, GPT family, T5, etc.) also sums rather than concatenates.

### 5.2 Worked numerical example — the full sum, from the chapter (kept, it's correct)

For `[CLS]` at position 0, segment A:
```
Token    [ 0.12, -0.08,  0.31,  0.05]
Position [ 0.01,  0.03, -0.02,  0.04]
Segment  [ 0.00,  0.00,  0.00,  0.00]
──────────────────────────────────────
Sum      [ 0.13, -0.05,  0.29,  0.09]
```
This is genuinely just element-wise addition — no learned mixing weights, no gating. The "mixing" all happens implicitly, later, via the shared 768-d space and what backprop does with it.

---

## 6. LayerNorm — the step the original chapter mentioned but didn't compute

**Simplified formula:**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Plain-language read of every term:
- $x$ — the raw summed vector for one token (e.g. `[0.13, -0.05, 0.29, 0.09]` above).
- $\mu$, $\sigma^2$ — the **mean and variance computed across that single token's own dimensions** (not across the batch, not across other tokens — this is what distinguishes LayerNorm from BatchNorm). Subtracting the mean and dividing by the standard deviation rescales this one vector to have mean 0 and variance 1.
- $\epsilon$ — a tiny constant (like 1e-5) purely to prevent division by zero if variance happens to be extremely small. Not conceptually important, just numerical safety.
- $\gamma$, $\beta$ — **learned** per-dimension scale and shift parameters, applied *after* normalization. This matters: LayerNorm doesn't just flatten everything to a fixed 0-mean/1-variance shape and stop — $\gamma$ and $\beta$ let the network learn to re-expand or re-shift specific dimensions if that's actually useful, so normalization doesn't destroy information the model needs, it just stabilizes the *scale* the model operates at.

**One fully worked numeric example** (using the `[CLS]` sum vector above, and for clarity assuming $\gamma=1, \beta=0$ — i.e. before any learned rescaling, so you can see the pure normalization step):
```
x = [0.13, -0.05, 0.29, 0.09]

mean (μ) = (0.13 + -0.05 + 0.29 + 0.09) / 4 = 0.46 / 4 = 0.115

deviations from mean:
  0.13  - 0.115 =  0.015
 -0.05  - 0.115 = -0.165
  0.29  - 0.115 =  0.175
  0.09  - 0.115 = -0.025

squared deviations: 0.000225, 0.027225, 0.030625, 0.000625
variance (σ²) = sum / 4 = 0.0587 / 4 = 0.014675
std (σ) = √0.014675 ≈ 0.1211

normalized (x - μ) / σ:
  0.015  / 0.1211 ≈  0.124
 -0.165  / 0.1211 ≈ -1.362
  0.175  / 0.1211 ≈  1.445
 -0.025  / 0.1211 ≈ -0.206

LayerNorm output (γ=1, β=0) ≈ [0.124, -1.362, 1.445, -0.206]
```
**Why this matters mechanically**: notice the output no longer resembles the raw sum's scale at all — the largest-magnitude raw value (0.29) became the largest normalized value (1.445), but everything is now on a standardized scale. This is the point: whatever the raw magnitude of token+position+segment happened to sum to (which could vary a lot token to token, or drift across many stacked layers), LayerNorm resets every token's vector to a consistent, well-behaved scale before it's used further — this is what keeps deep stacks of Transformer layers numerically stable during training instead of the scale exploding or vanishing layer over layer.

**What if we skipped LayerNorm entirely?** In deep networks, raw activation magnitudes tend to drift — growing or shrinking — as they pass through many stacked layers, since each layer's output becomes the next layer's input with no reset in between. Without normalization, this drift compounds across BERT's 12 layers, making gradients unstable during training (too large → divergence; too small → vanishing, similar in spirit to the RNN vanishing-gradient problem from Chapter 1, but here it's a magnitude-drift problem across depth rather than across sequence position). LayerNorm is one of the standard tools (alongside residual connections, covered in a later chapter) that makes training very deep Transformer stacks tractable at all.

---

## 7. The embedding lifecycle — pretrain / fine-tune / inference

| Stage | What happens to the three tables | Who does it |
|---|---|---|
| Pre-training | All three tables randomly initialized, updated by backprop jointly with every Transformer weight, over the full pre-training corpus (~3.3B words) | Google (or whoever pre-trains the base model) |
| Fine-tuning | You start from the pre-trained values. Either continue updating all tables on your task data, or freeze the lower layers (often including embeddings) and only update upper layers — the latter is preferred with small task datasets to avoid overfitting/forgetting | You |
| Inference | Everything frozen. Token ID 4937 always maps to the exact same row; position 2 always maps to the exact same row | Production system |

**The subtlety worth stating precisely in an interview**: the *embedding tables themselves* never become context-sensitive — token ID 4937's row is identical every single time you look it up, at every stage. What makes BERT's *output* context-sensitive is that this static starting vector for "cat" gets **transformed** by 12 layers of self-attention using the other tokens present in that specific sentence. So: **pre-training learns good starting points; attention (at inference time, using whatever sentence is actually given) does the contextual transformation on top.** This is the resolution to a common point of confusion: "if the token embedding is fixed, how is BERT context-sensitive at all?" — the fixed embedding is only the *input* to layer 1, not the *output* of the model.

**What if you froze all three embedding tables during fine-tuning?** You keep the general-purpose word/position/segment representations exactly as pre-trained, and let only the Transformer layers adapt to your task. This tends to help when your fine-tuning dataset is small (less risk of the embedding table overfitting/drifting toward your narrow dataset and losing its broad pre-trained generality) but can underperform full fine-tuning when you have ample task data and your domain's vocabulary usage genuinely differs from the pre-training distribution (recall the medical-tokenization discussion from Chapter 2 — frozen general-English token embeddings might just be a worse starting point for "myocardial"'s fragments than the model could otherwise learn to make them, given enough in-domain data).

---

## 8. Design-choice summary table, boosted

| Design choice | Why | What breaks without it |
|---|---|---|
| Separate token/position/segment tables (not one joint table) | Additive factorization avoids a combinatorially huge, mostly-empty joint table; each table generalizes independently | A joint (token, position) table would need ~15.6M rows vs. ~31K split across separate tables |
| Learned (not sinusoidal) positional embeddings | BERT has a fixed 512 ceiling anyway, so sinusoidal's extrapolation benefit is moot; learned embeddings fit the actual data better | You'd give up data-adaptive positional patterns for an extrapolation property BERT structurally can't use |
| Sum (not concatenate) the three embeddings | Keeps hidden dim at 768 throughout; concatenating would ~triple every downstream layer's parameter count | Model size and compute roughly triple for no proven accuracy gain |
| LayerNorm after summing | Keeps each token's vector on a consistent scale, preventing magnitude drift compounding across 12 stacked layers | Training instability — gradients that explode or vanish with depth |
| 2-row segment table | BERT's core pre-training objective (MLM + NSP) and most downstream tasks are two-sequence problems | Model has no explicit signal for "which sentence" beyond `[SEP]`'s boundary marker |

---

## 9. Diagnostics — misconceptions to pre-empt

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "The token embedding for 'cat' already captures context by the time you get to Transformer Block 1" | The token embedding table is a static per-ID lookup, identical regardless of sentence — it has exactly Word2Vec's context-blindness at this stage | Context-sensitivity comes entirely from self-attention transforming this static starting vector across 12 layers, not from the embedding table itself |
| "Positional embeddings and segment embeddings are somehow multiplied or gated with the token embedding" | BERT's actual combination is plain element-wise addition — no learned mixing function | Sum, then LayerNorm; the "mixing" is emergent from where in the shared 768-d space each table's outputs land, not from an explicit gate |
| "Sinusoidal encodings are strictly better since they're more principled/mathematical" | They're better *for the property BERT doesn't need* (extrapolation past training length); learned embeddings can better fit whatever positional structure actually helps loss on the training data given a fixed max length | The "better" choice depends on whether the model needs to generalize past its training length — BERT structurally doesn't |
| "LayerNorm normalizes across the batch, like BatchNorm" | LayerNorm computes mean/variance *within a single token's own feature vector*, independent of other tokens or other examples in the batch | This is precisely why LayerNorm works well with variable batch sizes and even batch size 1, unlike BatchNorm |
| "Freezing the embedding tables during fine-tuning is always the safe/better default" | It trades away the ability to adapt embeddings to a domain whose vocabulary usage differs meaningfully from pre-training data | The right choice depends on fine-tuning dataset size and domain shift — not a universal rule |

---

## 10. Q&A practice set (self-test — answers below the line)

**Q1 (easy).** Why can't the model just infer word order from the token embeddings alone, without a separate positional embedding table?

**Q2 (easy).** What is the total row count of the segment embedding table, and why is it that small?

**Q3 (medium).** Concatenating the three embeddings instead of summing would triple what, specifically, and why does that matter for a 12-layer stack?

**Q4 (medium — calculation).** A token's raw summed embedding (before LayerNorm) is `[0.4, 0.0, -0.2, 0.2]`. Compute the mean, and state (without doing the full normalization) whether the LayerNorm output for the first coordinate (0.4) will be positive or negative relative to the mean-centered value. 

**Q5 (medium).** Why did BERT choose learned positional embeddings over the original Transformer's sinusoidal formula, given the sinusoidal version was designed to generalize better to unseen lengths?

**Q6 (hard).** Explain precisely why "cat" can end up with different final vectors in different sentences, even though its token embedding row is always identical.

**Q7 (hard).** Why does LayerNorm compute its statistics per-token rather than per-batch (i.e., why not use BatchNorm)?

**Q8 (hard — spot the bug).** An engineer fine-tunes BERT on a small (500-example) domain-specific dataset and updates all three embedding tables along with every Transformer layer. Validation performance is much worse than a baseline that only fine-tuned the top 2 Transformer layers. What's the likely mechanism, and what's the fix?

---
---

### Answers

**A1.** Token embeddings are looked up purely by identity — token ID 4937 always retrieves the exact same row regardless of where it appears in the sequence, so nothing in the token embedding itself encodes position. Combined with the fact that self-attention's core operation ($QK^T$) is permutation-invariant, there is no path in the architecture for order information to enter unless it's injected explicitly, which is exactly the positional embedding table's job.

**A2.** Two rows — one for Segment A, one for Segment B. It's that small because BERT's core objectives and most downstream tasks (MLM+NSP pre-training, QA, NLI, sentence-pair classification) only ever involve at most two concatenated sequences at once; there was never a need to distinguish among more than two segments.

**A3.** It would triple the hidden dimension from 768 to 2,304 (768×3), which means every weight matrix in every one of the 12 stacked Transformer blocks (the $Q$/$K$/$V$ projections, feed-forward layers, etc.) would need roughly 3x more parameters to match that wider input, since parameter count in a linear layer scales with its input dimension. This compounds across all 12 layers rather than being a one-time cost.

**A4.** Mean = (0.4 + 0.0 + (-0.2) + 0.2) / 4 = 0.4/4 = 0.1. The first coordinate's mean-centered value is 0.4 − 0.1 = 0.3, which is positive — so after dividing by the (always-positive) standard deviation, the normalized value for that coordinate will still be positive. You don't need the full variance calculation to answer the sign question: the sign of the LayerNorm output (before applying γ/β) is always the same as the sign of (x − mean), since dividing by a positive standard deviation never flips sign.

**A5.** The sinusoidal formula's main advantage is letting a model generalize to sequence lengths longer than it was trained on. BERT has a hard, fixed 512-token limit built into its architecture regardless of which positional scheme it uses, so that extrapolation benefit is never actually usable — there's no scenario where BERT needs to handle position 600. Given that the main advantage of sinusoidal encoding is moot for BERT, the tradeoff favors learned embeddings, which can shape themselves specifically to whatever positional patterns are most useful for the actual pre-training data and objective, rather than being locked to a predetermined mathematical form.

**A6.** The token embedding row for "cat" is indeed always identical at input time — that part never changes. But that identical starting vector is then passed through 12 layers of self-attention, and at each layer, "cat"'s vector is updated based on a weighted combination of the *other tokens actually present in that specific sentence* (via the $QK^T$ attention weights from Chapter 1). Different sentences put different tokens around "cat," so the attention-weighted combination — and therefore "cat"'s vector after layer 1, and increasingly after layers 2 through 12 — ends up different each time, even though everyone started from the exact same row 4937.

**A7.** LayerNorm needs to work correctly at any batch size, including batch size 1 at inference time (a single request in production), and needs to behave consistently between training and inference. BatchNorm's statistics are computed across the batch dimension, which means its behavior depends on which other examples happen to be in the same batch, requires maintaining running statistics for use at inference (since inference often isn't batched the same way as training), and can behave poorly with small or variable batch sizes. Computing mean/variance per-token (within that single token's own feature vector) makes LayerNorm's behavior identical regardless of batch size or what other examples are present, which is a better fit for sequence models processing variable-length, often small-batch or single-example inputs.

**A8.** With only 500 examples, updating the full ~23.4M-parameter token embedding table (plus every Transformer layer) gives the optimizer far more capacity to overfit than the dataset can meaningfully constrain — the embedding table in particular can drift away from its broad, pre-trained general-English representations toward idiosyncrasies of the small fine-tuning set, effectively discarding some of the value that pre-training provided in the first place. Freezing the embedding tables (and lower Transformer layers) and only updating the top layers restricts the optimizer to a much smaller effective parameter budget better matched to 500 examples, while still letting the model adapt its task-specific decision-making on top of the (preserved) general-purpose representations. The fix is exactly what the baseline already does: freeze lower layers/embeddings, fine-tune only the top layers, when the dataset is small — and reconsider full fine-tuning only if more data becomes available or the domain shift from pre-training is severe enough to be worth the overfitting risk.

---

## 11. Quick recap card (last-minute review)

- **Three separate tables**: token `[30,522×768]` (identity, largest table, ~23.4M params), position `[512×768]` (order, learned not sinusoidal), segment `[2×768]` (which sentence, tiny).
- **Why separate, not joint**: additive factorization avoids a combinatorially huge, mostly-empty joint table; each generalizes independently.
- **Sum, not concatenate**: keeps hidden dim at 768 throughout, avoiding ~3x parameter blowup across all 12 Transformer layers.
- **Learned > sinusoidal for BERT specifically**: sinusoidal's key benefit (extrapolating past training length) is moot given BERT's hard 512-token architectural ceiling.
- **LayerNorm**: per-token (not per-batch) mean/variance normalization + learned γ/β rescale — keeps activation scale stable across 12 stacked layers; different from BatchNorm precisely because it doesn't depend on other examples in the batch.
- **The embedding table itself is never context-sensitive** — token ID 4937 always retrieves the same row. Context-sensitivity is entirely a product of self-attention transforming that static starting vector using whichever other tokens are actually present in that sentence.
- **Lifecycle**: pre-training learns all three tables from scratch via backprop; fine-tuning either continues updating them or freezes them (freezing helps with small datasets, avoids overfitting/forgetting); inference uses fixed, frozen lookups only.

*(Chapter 4 picks up here: how those context-blind, summed-and-normalized vectors become context-aware through self-attention — the mechanism only briefly previewed in Chapter 1.)*
