# Practical BERT Training Workflow — Data Cleaning & Layer-Unfreezing

Sixth companion doc. This one is pure workflow — the actual steps you'd walk through before and during fine-tuning, in order, with the decision logic spelled out (not just "clean your data," but what to check and why it breaks BERT specifically if you don't).

---

## PART 1 — Data Cleaning & Preparation, Step by Step

### Step 1: Inspect raw label quality first, before touching the text

**What to do:** Pull a random sample (100-200 rows) and manually check: do the labels actually match the text? Compute label distribution counts.

**Why first, before any text cleaning:** Text cleaning bugs are annoying but recoverable. Label noise is the single most damaging thing you can feed a model — BERT will happily learn to fit noisy labels (it has more than enough capacity), and you won't be able to tell from training metrics alone that anything's wrong. Catching this early saves you from debugging a "model that won't learn" that's actually a "dataset with 15% wrong labels" problem.

### Step 2: Deduplicate

**What to do:** Remove exact duplicates; flag and inspect near-duplicates (same text with a different label is a red flag — contradictory signal from identical input).

**Why it matters for BERT specifically:** Duplicates that land on both sides of your train/val split are **data leakage** — the model "generalizes well" on validation only because it memorized the identical example at train time. This is one of the most common causes of a model that looks great in validation and falls apart in production.

### Step 3: Check text length distribution

**What to do:** Tokenize a sample and plot token-count distribution. Find the 95th/99th percentile length.

```
Example distribution:
  median length: 42 tokens
  95th percentile: 118 tokens
  99th percentile: 210 tokens
  max: 1,850 tokens (a clear outlier)
```

**Why it matters:** This tells you your `max_length` setting — pick something around the 95th/99th percentile, not BERT's max of 512 by default. Padding every example to 512 when your typical text is 40 tokens wastes enormous compute (attention cost is quadratic in length, per the architecture doc) for zero benefit. It also surfaces outliers (a 1,850-token row is probably scraped garbage, not a real example) worth inspecting individually.

### Step 4: Decide truncation strategy for over-length text

**What to do:** If some examples exceed your chosen max_length, decide: truncate from the end (default), truncate from the start, or truncate from the middle (keep head + tail, drop middle) depending on where the task-relevant signal usually sits.

**Why this is a real decision, not a default:** For sentiment on reviews, the conclusion/summary is often at the end — truncating from the end can cut off the most label-relevant part of the text. For document classification where the topic is established early, end-truncation is usually fine. Get this wrong and you're silently discarding signal.

### Step 5: Minimal text normalization — less than you'd think

**What to do:** Fix encoding issues (mangled UTF-8, HTML entities like `&amp;`), strip control characters, standardize whitespace. That's usually it.

**What NOT to do, and why:** Don't lowercase, strip punctuation, or remove stopwords the way you would for classic NLP (TF-IDF, bag-of-words). BERT's WordPiece tokenizer and pretrained embeddings were trained on natural, punctuated, mixed-case text (if using a cased model) — punctuation and case carry real signal the model already knows how to use ("Apple" the company vs "apple" the fruit; a question mark changes meaning). Aggressively "cleaning" text the classic-NLP way actively throws away information BERT is equipped to use, and creates a mismatch with what it saw during pretraining.

### Step 6: Handle missing / malformed rows

**What to do:** Drop or flag rows with null text, null labels, or labels outside the expected set (e.g. a stray `"2"` in a binary 0/1 label column — usually a data entry bug).

**Why:** These silently crash training or, worse, don't crash but corrupt a batch's loss computation in a way that's hard to trace back.

### Step 7: Check and address class imbalance (see also the fine-tuning Q&A doc)

**What to do:** Compute class counts. If imbalance is significant (e.g. beyond roughly 3:1), decide upfront whether you'll handle it via class-weighted loss, resampling, or both — and note it now so your train/val split (next step) preserves proportions correctly.

### Step 8: Split train / validation / test — stratified, and leakage-checked

**What to do:** Stratified split (preserves class proportions in each split) rather than a naive random split when classes are imbalanced. Explicitly check for near-duplicate rows crossing the split boundary (tie back to Step 2).

**Why stratified matters:** With a 90:10 imbalance, a small validation set from a plain random split can end up with very few minority-class examples by chance, making your validation F1 for that class noisy and unreliable — stratification guarantees proportional representation.

### Step 9: Tokenization sanity check

**What to do:** Before launching a full training run, decode a few tokenized examples back to text (`tokenizer.decode(...)`) and visually confirm they match the original input, and check that the tokenizer's special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are being inserted where expected.

**Why:** This catches silent bugs — wrong tokenizer for the model checkpoint, wrong padding side, truncation happening at the wrong end — before they burn a full training run's worth of compute.

### Step 10: Baseline sanity check before real training

**What to do:** Train for a tiny number of steps (or one epoch) on a small subset, confirm loss actually decreases from its random-init value and doesn't NaN.

**Why:** Cheap, fast way to catch data pipeline bugs, learning rate that's wildly too high (loss explodes to NaN), or a labels/logits shape mismatch — before committing to the full run.

---

## PART 2 — How Many Layers to Unfreeze, and How to Decide

### The core trade-off, restated concretely

More unfrozen layers = more capacity to adapt to your task, but more risk of overfitting (especially on small data) and catastrophic forgetting of useful pretrained knowledge. Fewer unfrozen layers = safer/more regularized, but may underfit if your task genuinely needs deep representation changes.

### Decision framework, in the order you'd actually apply it

**1. Start from dataset size — the single strongest signal.**

| Labeled examples | Recommended starting point |
|---|---|
| < 1,000 | Freeze entire encoder; train only the classification head (+ maybe last 1-2 layers) |
| 1,000 – 10,000 | Unfreeze the top 4-6 layers (out of 12); freeze the bottom 6-8 |
| 10,000 – 100,000 | Full fine-tuning (all 12 layers), with discriminative learning rates (smaller LR on lower layers) |
| > 100,000 | Full fine-tuning, often with a uniform LR across layers — enough data to safely adapt everything |

**Why dataset size is the primary driver:** Every unfrozen layer adds trainable parameters that can overfit a small dataset. BERT-base's top layers alone (say, layers 9-12 + head) already have tens of millions of parameters — plenty of capacity to memorize a 500-example dataset in a couple epochs if you let it, especially since lower layers already encode generically useful features that don't need to move for most tasks.

**2. Use which layers, not just how many — lower layers are more generic, upper layers are more task-specific.**

Recall from the phases doc: lower layers tend to encode syntax/surface features, upper layers encode more abstract/semantic features. This is *why* the standard practice is to freeze from the bottom and unfreeze from the top, not the reverse — you want to keep the generic foundation stable and let the task-specific top adapt.

**3. If unsure, run progressive (gradual) unfreezing rather than guessing a fixed number upfront.**

This is a concrete, testable procedure instead of a one-shot guess:

```
Epoch 1: only the classification head is trainable, everything else frozen.
Epoch 2: unfreeze the top 2 encoder layers (11, 12), keep the rest frozen.
Epoch 3: unfreeze the next 2 (9, 10).
Epoch 4+: continue unfreezing 2 more layers every epoch, monitoring validation
          loss after each unfreeze step.
Stop unfreezing further layers as soon as validation loss stops improving
(or starts degrading) after an unfreeze step.
```

**Why this works better than a fixed guess:** It turns "how many layers to unfreeze" from a hyperparameter you guess once into something you observe directly — the point where unfreezing further stops helping validation performance *is* your answer, discovered empirically rather than assumed from a rule of thumb.

**4. Pair unfreezing with discriminative learning rates, not a uniform one, once more than the head is unfrozen.**

```
layer 12 (top, closest to output): LR x 1.0
layer 11:                          LR x 0.9
layer 10:                          LR x 0.9^2 ≈ 0.81
...
layer 1 (bottom, closest to input): LR x 0.9^11 ≈ 0.31
```

**Why:** Even when you've decided to unfreeze a layer, you don't necessarily want it to move as fast as the head — lower unfrozen layers still benefit from smaller, gentler updates than the newly-initialized head or the top layers, which need to adapt the most.

**5. Validate the choice, don't just trust the heuristic.**

Run 2-3 configurations if compute allows (e.g. head-only, top-4-unfrozen, full fine-tune) and compare validation F1 directly — the table above is a reasonable prior, not a guarantee, and datasets vary in how much task-specific adaptation they actually need.

### Signs you unfroze too much (overfitting from excess capacity)

- Train accuracy near 100% within 1 epoch, validation accuracy notably lower and not improving.
- Validation loss rising while train loss keeps falling.
- **Fix:** freeze more layers, reduce epochs, add weight decay/dropout, or get more data — in roughly that order of effort.

### Signs you unfroze too little (underfitting)

- Both train and validation accuracy plateau early, well below what a reasonable baseline (e.g. TF-IDF + logistic regression) achieves — a sign the frozen representation isn't adapting enough to the task's specific decision boundary.
- **Fix:** unfreeze more layers, or increase learning rate slightly on the unfrozen ones.

---

## The full practical pipeline, top to bottom

1. Inspect label quality on a sample
2. Deduplicate + check near-duplicates
3. Check text length distribution → set max_length
4. Decide truncation strategy
5. Minimal normalization (fix encoding, don't over-clean)
6. Drop/flag malformed rows
7. Check class imbalance → decide weighting strategy
8. Stratified train/val/test split, leakage-checked
9. Tokenization sanity check (decode and eyeball)
10. Tiny-scale training sanity check (loss goes down, no NaN)
11. Pick starting unfreeze depth from the dataset-size table
12. Fine-tune — progressive unfreezing if unsure, discriminative LRs if more than the head is unfrozen
13. Monitor validation loss/F1 per class, not just aggregate accuracy
14. Stop on validation signal (early stopping), not on a fixed epoch count if it's clearly overfitting sooner
