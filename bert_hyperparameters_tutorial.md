# BERT Fine-Tuning Hyperparameters — What to Use, and What Not To

Seventh companion doc. A hyperparameter-by-hyperparameter guide: the recommended value/range, *why* that range, and specifically what people reach for out of habit (usually from training-from-scratch intuition) that actively hurts fine-tuning.

**The one meta-rule underlying almost every choice below:** BERT fine-tuning is *light adaptation of an already-good model*, not training from scratch. Nearly every "don't do this" in this doc is a value that made sense for from-scratch training but is too aggressive for fine-tuning a pretrained model.

---

## Quick-reference table

| Hyperparameter | Use | Don't use | Why |
|---|---|---|---|
| Learning rate | `2e-5` – `5e-5` | `1e-3` – `1e-4` (from-scratch-sized) | Large steps overwrite pretrained knowledge (catastrophic forgetting) |
| Optimizer | AdamW | Adam, plain SGD | AdamW decouples weight decay correctly; SGD converges too slowly for this regime |
| LR schedule | Linear warmup + linear/cosine decay | Constant LR | Early noisy gradients from the random head damage the pretrained encoder without warmup |
| Warmup ratio | 6% – 10% of total steps | 0% or >20% | Too little = early instability; too much = wastes most of a short fine-tuning run at low LR |
| Batch size | 16 – 32 (effective, via grad accumulation if needed) | 1 – 4, or >128 without adjustment | Too small = noisy gradients; too large without LR scaling = poor convergence |
| Epochs | 2 – 4 | 10+ | BERT overfits small fine-tuning sets fast; more epochs ≠ better past a point |
| Weight decay | 0.01 | 0 (no regularization) or 0.1+ (from-scratch-sized) | Some decay helps generalization; too much fights the small, careful updates you want |
| Dropout | 0.1 (BERT's default, usually leave as-is) | 0.5+ | BERT's default is already tuned; aggressive dropout on a light fine-tune step mostly just slows convergence |
| Max sequence length | 95th/99th percentile of your data's token length | Always 512 | Wastes compute quadratically on padding; attention cost scales as length² |
| Gradient clipping | max norm 1.0 | No clipping | Occasional large gradients (esp. early / from imbalanced batches) can destabilize AdamW's updates |
| Class weights | Inverse-frequency weighting if imbalance > ~3:1 | Ignoring imbalance | Model defaults to predicting the majority class if minority-class loss is drowned out |
| Layers unfrozen | Dataset-size dependent (see freezing doc) | Always full fine-tune regardless of data size | Small data + full fine-tune = severe overfitting risk |
| Random seed | Fixed, and reported | Left unset / different every run | Fine-tuning is genuinely seed-sensitive on small datasets — results can swing several F1 points |

---

## Learning rate — the highest-leverage choice

**Use:** `2e-5` to `5e-5`. If you must pick one number with no time to sweep, `2e-5` or `3e-5` are the safest defaults from the original BERT paper's own recommendations.

**Don't use:** anything in the `1e-3`–`1e-4` range typical of training a model from random initialization. This is the single most common mistake for people coming from a "train a classifier from scratch" background.

**Why it matters this much:** the pretrained weights already sit in a good region of the loss landscape. A learning rate sized for random initialization takes steps large enough to knock the model out of that region before it has a chance to gently specialize — you'll often see faster initial loss drop (looks promising!) followed by a validation performance that's *worse* than a properly-tuned smaller LR, because you actually damaged useful pretrained structure.

**If your loss explodes to NaN in the first few steps:** it's almost always the learning rate being too high combined with insufficient (or no) warmup — try both fixes together, not just one.

---

## Optimizer — AdamW, not Adam or SGD

**Use:** AdamW.

**Don't use:** plain `Adam` (applies weight decay by folding it into the gradient update, which interacts badly with Adam's adaptive per-parameter learning rates and effectively under- or over-regularizes different parameters inconsistently), or SGD (converges far too slowly for the handful of epochs you're running — SGD's strength is stable long-run convergence over many epochs, which isn't the regime you're in here).

**Why AdamW specifically:** it decouples weight decay from the gradient-based update (subtracts it directly from the weights, not folded into the moment estimates), which was shown to fix exactly the inconsistent-regularization issue that plain Adam has. It's the standard for essentially all Transformer fine-tuning.

---

## Learning rate schedule — warmup + decay, not constant

**Use:** linear warmup for the first 6-10% of steps, then linear decay to 0 (or cosine decay — both work, linear is the original BERT-paper default and marginally simpler).

**Don't use:** a constant learning rate throughout training.

**Why:** at step 0, your classification head is randomly initialized and produces near-random, high-magnitude gradients. If the full learning rate hits the pretrained encoder immediately, those noisy early gradients can meaningfully perturb good pretrained weights before the head has had any chance to settle into something sensible. Warmup ramps the LR up gradually so by the time it reaches full strength, gradients flowing back through the encoder are more meaningful signal, not head-initialization noise. The decay at the end lets the model settle into a sharper minimum rather than oscillating near convergence.

**Common mistake on warmup ratio specifically:** setting it too high (e.g. 30-50%) on an already-short fine-tuning run (2-4 epochs) — you end up spending most of your limited training budget at a suppressed learning rate and never really reach peak LR long enough to make meaningful progress. 6-10% is the standard range for a reason.

---

## Batch size — bigger isn't automatically better here

**Use:** 16-32 as a per-device batch size; use gradient accumulation to reach a larger *effective* batch size if memory-constrained, rather than a tiny raw batch size.

**Don't use:** batch sizes of 1-4 without accumulation (gradient estimates become too noisy, especially with class-imbalanced data where a small batch might contain zero minority-class examples by chance), or very large batches (128+) without adjusting the learning rate upward to compensate — large-batch training with an unchanged small LR tends to converge to flatter, sometimes worse optima in the limited number of steps a short fine-tuning run allows.

**Why moderate batch sizes work well specifically for fine-tuning:** you're not trying to squeeze out marginal gains over hundreds of thousands of steps the way large-batch pretraining does — you want each of your limited number of gradient updates (few epochs, small dataset) to be a reasonably clean, representative signal without being so large that you're taking very few, very smooth (and possibly less-well-generalizing) steps.

---

## Number of epochs — resist the urge to train longer

**Use:** 2-4 epochs for most classification fine-tuning tasks. Sometimes 1 epoch is enough on genuinely large fine-tuning datasets (100k+ examples).

**Don't use:** 10+ epochs as a default, "just to be safe" — the instinct that more training time is generally better is calibrated to training from scratch, where you're building representations up from nothing. Here you're doing narrow adaptation on top of an already-strong representation.

**Why:** BERT-base has ~110M parameters and can memorize a small fine-tuning dataset (a few thousand examples) within very few epochs, at which point additional epochs push train accuracy toward 100% while validation performance stalls or degrades — textbook overfitting, and it happens *fast* relative to from-scratch training curves. Watch validation loss and use early stopping instead of committing to a fixed large epoch count upfront.

---

## Weight decay — some, not none, not too much

**Use:** `0.01` (AdamW default in most fine-tuning setups, including the original BERT paper).

**Don't use:** `0` (no regularization at all invites overfitting on small fine-tuning sets), or values in the `0.1`+ range sometimes used for training vision models or LLMs from scratch — too aggressive here fights against the small, careful updates you're trying to make to an already-good representation.

**One detail worth knowing:** weight decay should typically be **excluded** from bias terms and LayerNorm parameters (this is standard practice, and the reference fine-tuning script in the previous doc does this via a `no_decay` list) — decaying LayerNorm's scale/shift parameters toward zero doesn't serve the same regularization purpose it does for weight matrices and can actively hurt normalization behavior.

---

## Dropout — usually leave it alone

**Use:** BERT's built-in default of `0.1` on attention probabilities and hidden layers — in most fine-tuning setups you don't need to touch this at all.

**Don't use:** cranking dropout up significantly (0.3-0.5) as a first response to overfitting.

**Why leave it as-is:** dropout was already tuned during BERT's pretraining; pushing it much higher during a short fine-tuning run mostly just slows convergence within your limited epoch budget rather than meaningfully improving generalization. If you're seeing real overfitting, the higher-leverage fixes are epoch count, layer freezing depth, and dataset size — not dropout.

---

## Max sequence length — fit the data, don't default to 512

**Use:** the 95th or 99th percentile of your training set's actual token length (checked via the tokenizer, per the data-cleaning doc), rounded up to a convenient number (e.g. 64, 128, 256).

**Don't use:** BERT's max of 512 by default "just to be safe," when your real text is much shorter.

**Why:** attention cost scales as (sequence length)² — padding every example to 512 when the typical example is 40 tokens means you're paying roughly `(512/128)² ≈ 16x` more compute per batch than necessary if 128 would have covered 99% of your data, for zero benefit (padding tokens carry no signal and are masked out anyway).

---

## Gradient clipping — cheap insurance, keep it on

**Use:** clip gradient norm to `1.0` (the near-universal default for Transformer fine-tuning).

**Don't use:** no clipping at all — an occasional unusually large gradient (from an outlier example, an imbalanced batch, or just early-training noise) can otherwise cause a disproportionately large parameter update, potentially destabilizing an otherwise well-behaved training run.

**Why it's low-risk to just always include:** clipping only activates when a gradient actually exceeds the norm threshold — well-behaved training is essentially unaffected, so there's little downside to leaving it on as a safety net.

---

## Class weights — don't skip this on imbalanced data

**Use:** inverse-frequency class weights (or focal loss for more extreme imbalance) whenever class imbalance exceeds roughly 3:1 — see the fine-tuning Q&A doc for the exact weighting formula.

**Don't use:** unweighted cross-entropy on meaningfully imbalanced data and then evaluate with accuracy — this combination will look fine on the metric you're checking while the model has effectively learned to ignore the minority class.

---

## Layer-freezing depth — tie it to dataset size, not habit

**Use:** the dataset-size-driven heuristic from the layer-freezing doc (freeze more on small data, fine-tune fully on large data), ideally validated empirically with progressive unfreezing rather than guessed once.

**Don't use:** a fixed habit of "always full fine-tune" or "always freeze everything but the head" regardless of dataset size — the right answer here is genuinely data-dependent, and picking the same default every time is picking the wrong answer for some fraction of your projects.

---

## Random seed — fix it, and don't over-read a single run

**Use:** a fixed seed for reproducibility, and ideally average results (or at least note the spread) across 3-5 different seeds if you're reporting a result that matters (e.g. comparing two hyperparameter settings).

**Don't use:** a single unseeded run as the basis for a real conclusion about which configuration is better.

**Why this matters more than people expect:** fine-tuning on small-to-medium datasets is genuinely seed-sensitive — different random initializations of the classification head, combined with different data shuffling order, can swing validation F1 by a few points on their own. A single run's result being 1-2 points better than another single run's result is often within noise, not a real effect — this is a common source of false conclusions in ablation studies.

---

## Mixed precision (fp16/bf16) — free speedup, minimal downside

**Use:** enable mixed precision training (fp16 on older GPUs, bf16 on newer hardware that supports it) essentially by default — most training frameworks (e.g. Hugging Face `Trainer`) make this a single flag.

**Don't use:** fp16 without gradient scaling (most frameworks handle this automatically, but if writing a custom loop, don't skip it) — fp16's limited numeric range can cause gradient underflow without it.

**Why it's close to a free win:** roughly 1.5-3x training speedup and reduced memory usage on supported hardware, with negligible accuracy impact for fine-tuning-scale training — there's rarely a reason to leave it off other than hardware that doesn't support it.

---

## The one-sentence summary

**When in doubt, use the smaller, gentler version of every knob** — smaller learning rate, fewer epochs, less unfrozen capacity than your from-scratch instincts suggest — because fine-tuning's whole premise is that most of the hard work is already done by pretraining, and your job is careful adaptation, not re-learning.
