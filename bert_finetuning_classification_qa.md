# Fine-Tuning BERT for Classification — Practical Q&A

Fourth companion doc. The first three covered *architecture* (what BERT is made of, why). This one covers the *practical training mechanics* interviewers probe once they know you understand the model — the "you've actually fine-tuned one of these, right?" questions.

---

### Q: What learning rate do you use, and why so small?

**A:** Typically `2e-5` to `5e-5` — roughly 10-100x smaller than what you'd use training a classifier from scratch (often `1e-3` to `1e-4`).

**Why:** The pre-trained weights already encode a huge amount of useful structure learned from far more data than your fine-tuning set will ever have. A large learning rate would take large steps that overwrite this general-purpose representation before the model has a chance to gently adapt it — this is **catastrophic forgetting**. You want a small nudge, not a rewrite. Too small (e.g. `1e-6`), and you're barely adapting at all — the pre-trained representation may not be a perfect fit for your specific task and needs a real, if gentle, adjustment.

---

### Q: What's the warmup schedule, and why does it matter?

**A:** Linearly ramp the learning rate up from 0 to its peak value over the first ~10% of training steps, then linearly (or cosine) decay it back down to 0 over the rest.

Worked example — peak LR `3e-5`, 1000 total steps, 100 warmup steps:

| Step | LR | Phase |
|---|---|---|
| 0 | 0.0 | start |
| 25 | 7.5e-6 | warming up |
| 50 | 1.5e-5 | warming up |
| 100 | 3.0e-5 | peak reached |
| 300 | 2.33e-5 | decaying |
| 600 | 1.33e-5 | decaying |
| 900 | 3.3e-6 | near end |
| 999 | ~0 | done |

**Why warmup specifically:** At step 0, the new classification head (randomly initialized) is producing garbage gradients — if you immediately hit it with the full learning rate, those noisy, large gradients get backpropagated through the pretrained encoder too, potentially damaging good representations before the head has stabilized. Warmup lets the optimizer (usually Adam, which has running variance estimates that are unreliable in the first few steps) settle into a sensible update direction before applying full-strength updates.

**Why decay at the end:** Standard fine-tuning wisdom — smaller steps late in training let the model settle into a sharper minimum rather than oscillating around one.

---

### Q: How many epochs? Why not train longer if loss is still going down?

**A:** Typically **2-4 epochs** for fine-tuning, sometimes as few as 1 for large datasets.

**Why so few (surprising if you're used to training from scratch, where you might run 50+ epochs):** Fine-tuning datasets are usually small relative to the model's capacity (110M+ params for BERT-base). The pretrained weights already provide most of the "knowledge" — you're doing light adaptation, not learning from zero. Training many epochs on a small labeled set causes the model to **memorize** the fine-tuning set rather than generalize (classic overfitting), especially since BERT can easily achieve near-100% train accuracy on small datasets within a couple epochs. Watch validation loss, not train loss — if val loss starts rising while train loss keeps falling, that's your stopping signal (early stopping), not "keep training because train loss looks fine."

---

### Q: `[CLS]` token vs. mean-pooling all token embeddings — which do you use for classification, and why?

**A:** `[CLS]` is the standard default and what BERT was pretrained to use this way (via NSP), but mean-pooling over all token embeddings is a common, often competitive alternative.

**Why `[CLS]` works:** During pretraining, `[CLS]`'s final hidden state was explicitly trained (via NSP) to summarize the whole sequence for a downstream decision — so fine-tuning a classifier head on top of it is asking the model to do a variant of what it already practiced.

**Why mean-pooling sometimes works better in practice:** `[CLS]`'s representation quality depends on how well NSP transferred to your task — for some tasks (especially sentence-similarity-style tasks), mean-pooling all token vectors captures more distributed information rather than relying on one token to have aggregated everything well. This is well documented in the Sentence-BERT line of work, which found pooling strategies matter a lot for embedding quality specifically.

**Practical takeaway:** Start with `[CLS]` — it's the standard, simplest option, and works well for most single-sentence and sentence-pair classification tasks. Try mean-pooling as an ablation if you're not hitting expected performance.

---

### Q: How do you handle class imbalance?

**A:** Most common approach: **class-weighted loss**, weighting each class inversely proportional to its frequency.

Worked example — 900 negative examples, 100 positive examples:

```
w_neg = total / (2 × n_neg) = 1000 / 1800 = 0.556
w_pos = total / (2 × n_pos) = 1000 / 200  = 5.000

pos weight is 9x the neg weight
```

In PyTorch this plugs straight into `CrossEntropyLoss(weight=...)` or, for binary classification with a single logit, `BCEWithLogitsLoss(pos_weight=...)`.

**Why this works:** Without weighting, a 900:100 imbalanced dataset lets the model achieve 90% accuracy by just always predicting the majority class — the loss signal from minority-class errors is drowned out by the sheer volume of majority-class examples. Weighting rescales each example's loss contribution so the rarer class's mistakes matter proportionally more, forcing the model to actually learn to distinguish it rather than ignore it.

**Alternatives, and when to reach for them instead:**
- **Oversampling the minority class / undersampling the majority** — simpler to reason about than loss weighting, but risks overfitting to duplicated minority examples (oversampling) or throwing away useful data (undersampling).
- **Focal loss** — down-weights *easy* examples (regardless of class) and focuses gradient on hard-to-classify ones; more useful when the imbalance is extreme (e.g. >50:1) or when even weighted CE still gets dominated by a flood of easy majority examples.
- Report **F1 / precision-recall**, not accuracy, on imbalanced data regardless of which fix you use — see the metrics question below.

---

### Q: Do you freeze any layers, or fine-tune the whole model?

**A:** Full fine-tuning (all layers unfrozen) is standard and usually performs best, given enough labeled data (typically low thousands of examples or more).

**When you'd freeze lower layers instead:**
- **Very small labeled datasets** (a few hundred examples) — full fine-tuning risks overfitting badly; freezing the lower layers (which tend to encode more generic syntactic features) and only fine-tuning the top few layers + head reduces the number of trainable parameters and acts as a regularizer.
- **Compute-constrained settings** — freezing most of the network and training only the head/top layers is much cheaper (fewer gradients to compute and store).

**A middle-ground technique — discriminative (layer-wise) learning rates:** Use a smaller learning rate for earlier layers and a larger one for later layers / the new head, e.g. multiply the base LR by `0.9^(depth from top)` per layer. Rationale: lower layers encode more general, less task-specific features that shouldn't move much; upper layers and the head need to adapt more aggressively to the new task. This is a common technique from the original ULMFiT paper, carried over into BERT fine-tuning practice.

---

### Q: What batch size and sequence length do you actually pick, and why?

**A:** Batch size: as large as fits in memory, commonly 16-32 for BERT-base on a single GPU; use **gradient accumulation** to simulate a larger effective batch size when memory-constrained.

```
effective_batch_size = per_device_batch_size × grad_accum_steps × n_gpus
                      = 16 × 4 × 1 = 64
```

**Why gradient accumulation works:** You compute gradients on several small "micro-batches" without stepping the optimizer, summing (accumulating) the gradients across them, then step once — mathematically equivalent to having run one large batch through, at the cost of extra compute time but not extra memory, since only one micro-batch's activations are held at once.

**Sequence length:** Pick the smallest max length that covers most of your data (e.g. the 95th-percentile token length of your training set) rather than always maxing out at 512. **Why:** attention cost is quadratic in sequence length (see architecture doc) — padding everything to 512 when your typical example is 60 tokens wastes enormous compute and memory on padding tokens that carry no signal.

---

### Q: What metrics do you report, and why not just accuracy?

**A:** For balanced binary/multi-class classification, accuracy is fine as a headline number, but always also report **precision, recall, F1** (macro-averaged for multi-class), and ideally a **confusion matrix**.

**Why not accuracy alone:** On imbalanced data, accuracy can look great while the model completely fails on the minority class (the "always predict majority" trap described above). F1 (harmonic mean of precision and recall) penalizes models that only do well on one of the two, giving a much more honest single-number summary. Macro-averaging (average F1 per class, unweighted) specifically checks that minority classes aren't being ignored, versus micro/weighted averaging which can still be dominated by majority-class performance.

---

### Q: How do you know if the model is overfitting, and what do you do about it?

**A:** Classic signal: training loss keeps decreasing while validation loss plateaus or starts increasing.

**BERT-specific fixes, roughly in order of what to try first:**
1. **Reduce epochs / early stopping** — usually the single biggest lever, since BERT overfits fast on small data.
2. **Weight decay** (typically `0.01`) on the AdamW optimizer — penalizes large weights, standard regularization.
3. **Dropout** — already built into BERT (default 0.1 on attention and hidden layers); rarely needs manual tuning, but increasing it slightly can help on very small datasets.
4. **Freeze more layers / use discriminative LRs** (see above) — fewer trainable parameters = less capacity to overfit.
5. **Data augmentation** (back-translation, synonym replacement, or simple techniques like random token masking) if the dataset is genuinely too small — a last resort, since it can introduce noisy labels if done carelessly.

---

### Q: What are the most common practical bugs when fine-tuning BERT for classification?

**A:**
- **Tokenizer/model mismatch** — using a tokenizer from a different checkpoint than the model weights; vocab indices won't line up and training will silently produce garbage.
- **Inconsistent padding/truncation between train and inference** — if you truncate training examples at 128 tokens but don't truncate at inference, you'll evaluate on out-of-distribution input lengths.
- **Forgetting attention masks** — padding tokens must be masked out of attention, or the model wastes capacity attending to meaningless `[PAD]` tokens (and in bad cases, can be affected by their random embeddings early in training).
- **Data leakage across train/val splits** — especially in datasets with near-duplicate or templated examples (common in scraped data); the model appears to generalize well but is actually just recognizing near-duplicates it saw at train time.
- **Not shuffling data with class-correlated ordering** — if the data loader isn't shuffled and examples are grouped by class, batches become effectively single-class, destabilizing training.
- **Reusing an optimizer/scheduler state across separate fine-tuning runs** — leftover Adam moment estimates from a previous run can produce misleadingly fast (or slow) initial convergence; always reinitialize the optimizer for a new fine-tuning run.

---

### Rapid-fire

**Q: Why AdamW instead of plain Adam or SGD?**
A: AdamW decouples weight decay from the gradient-based update (plain Adam's weight decay interacts poorly with its adaptive learning rates); it's the de facto standard optimizer for Transformer fine-tuning.

**Q: Should you use the same LR for the pretrained encoder and the new classification head?**
A: Not required to, but it's the common default; discriminative LRs (smaller for encoder, larger for head) often help, especially on small datasets.

**Q: If validation F1 is high but the model fails badly on a specific subgroup of your data, what's the likely cause?**
A: Class or subgroup imbalance not reflected in the aggregate metric — check per-class/per-subgroup metrics, not just the overall macro number, since a small subgroup's failures can be invisible in an aggregate score.
