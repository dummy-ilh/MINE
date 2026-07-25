# Module 3 — Scaling Laws & Emergent Abilities (Master Notes, Expanded)

## 0. Why scaling laws matter — the practical question they answer

Given a fixed compute budget (say, $1M of GPU time), you have to decide: build a **bigger model** with less data, or a **smaller model** trained on **more data**? Scaling laws are empirical formulas (fit by training many models of different sizes and measuring loss) that answer exactly this — they tell you how loss changes as you scale model size (N = parameters), dataset size (D = tokens), and compute (C), so you can predict performance *before* spending millions of dollars on a training run, and so you can allocate a fixed budget optimally.

---

## 1. The general power-law form (applies to both Kaplan and Chinchilla)

### The formula, explained term by term
```
L(N) = (N_c / N) ^ α
```
- `L(N)` = test loss as a function of model size N (parameters), holding data/compute effectively unconstrained.
- `N_c` = a constant (empirically fit) — sets the scale.
- `α` (alpha) = a small positive exponent (empirically found to be around 0.076 for the Kaplan paper's parameter scaling) — controls how fast loss drops as N grows.

**In plain words**: loss decreases as a *power law* (not linearly, not exponentially) as you increase model size — meaning **diminishing returns**: to cut loss in half, you don't need 2x the parameters, you typically need an order of magnitude more, because the exponent α is small. This is the single most important shape to remember: **smooth, predictable, but ever-slowing improvement** — no sudden cliffs in the loss curve itself (this becomes important later when contrasted with "emergent abilities," which *do* look like cliffs on task-accuracy curves, just not on the loss curve).

Similar power laws hold separately for **data size D** and **compute C**:
```
L(D) = (D_c / D) ^ β
L(C) = (C_c / C) ^ γ
```
Same interpretation — more data or more compute, alone, also reduces loss following a power law with its own small exponent.

---

## 2. Kaplan et al. (2020) — the original OpenAI scaling law paper

### Core finding
Given a fixed compute budget, **loss depends primarily on model size N**, and data size matters much less — their practical recommendation was: **make the model as large as your compute budget allows, and don't worry too much about needing proportionally more data.** Their fitted guidance suggested something like: a 10x increase in compute should go mostly toward a ~5.5x increase in model size and only ~1.8x increase in data (rough ratios from their paper) — i.e., **heavily prioritize parameters over tokens.**

### The practical consequence (and the mistake this caused)
This finding directly shaped GPT-3 (175B parameters) and an entire generation of models that followed the recipe "go as big as possible, use whatever data you can get relatively quickly" — leading to models that were **enormous but comparatively undertrained** on data relative to their size.

---

## 3. Chinchilla (Hoffmann et al., 2022) — the correction

### Core finding
Chinchilla re-ran the scaling-law experiments more carefully (controlling for a subtle methodological issue in how Kaplan's team handled learning-rate schedules across different training run lengths) and found the opposite emphasis: **for a fixed compute budget, model size (N) and training tokens (D) should scale roughly equally** — not "make N huge and don't worry about D."

### The concrete numerical result (memorize this)
Chinchilla's compute-optimal ratio: **approximately 20 tokens of training data per model parameter.**

**Worked example**: Chinchilla itself was 70B parameters, trained on ~1.4 trillion tokens.
```
1.4 trillion tokens / 70 billion parameters = 20 tokens/parameter
```
This matches the ~20:1 ratio. Compare this to GPT-3 (175B parameters, trained on ~300B tokens):
```
300 billion tokens / 175 billion parameters ≈ 1.7 tokens/parameter
```
GPT-3 was trained at roughly **1.7 tokens/parameter — over 10x below the Chinchilla-optimal ratio** — meaning GPT-3 was substantially *undertrained* relative to its size. Chinchilla (70B, much smaller than GPT-3's 175B) **outperformed GPT-3 on downstream benchmarks** despite having less than half the parameters, purely because it saw proportionally far more training tokens — this is the single result that made the whole field recompute their training recipes.

### Why this happened — the actual mechanism (good for a "derive it" interview question)
Given a fixed compute budget `C`, and the well-known approximation that compute for training a transformer is roughly:
```
C ≈ 6 × N × D
```
(6 comes from: 2 FLOPs per parameter per token for the forward pass, doubled to ~4 for backward pass, plus additional smaller terms — the commonly cited approximation is 6ND FLOPs total for training).

**In plain words**: if you fix `C` (your budget), then `N` and `D` are inversely related — spend the same total compute either on a bigger model with less data, or a smaller model with more data. Chinchilla's contribution was finding the loss-minimizing *split* of that fixed product — and empirically, the optimal split sets N and D to grow at roughly the **same rate** as C grows, i.e., if you 10x your compute budget, you should roughly √10 the model size AND √10 the data size (not 10x one while leaving the other flat).

### Numerical example: what "compute-optimal" allocation means at two different budgets
Say Model A uses compute budget `C_A` and is compute-optimal at N=10B params, D=200B tokens (ratio 20:1, check: 10B×20=200B ✓).

If you get **4x more compute** (`C_B = 4×C_A`), Chinchilla's finding says scale both N and D by roughly `√4 = 2x` each (since C ≈ 6ND, and N,D grow at the same rate, C grows as the square of that shared growth factor):
```
N_B = 10B × 2 = 20B params
D_B = 200B × 2 = 400B tokens
Check: N_B × D_B ratio = 400B/20B = 20:1 ✓ (still compute-optimal ratio)
Check: compute scales by (2×2) = 4x ✓ (matches the 4x compute budget)
```
This square-root scaling of both N and D together (rather than dumping all extra compute into N alone, as Kaplan's recipe implied) is the practical takeaway to be able to reproduce on a whiteboard.

### Interview one-liner
"Kaplan said 'compute-optimal means make the model bigger'; Chinchilla corrected this by showing that for a fixed compute budget, you get lower loss by training a smaller model on proportionally more data — the compute-optimal ratio is about 20 tokens per parameter, and most large models before Chinchilla (like GPT-3) were substantially undertrained relative to their size."

### Where this played out in practice
Post-Chinchilla, model releases shifted strategy: **Llama** (Meta) is the clearest example — Llama's original paper explicitly cites Chinchilla and deliberately trains *smaller* models (7B, 13B, 65B) on *far more* tokens (1-1.4 trillion) than Chinchilla-optimal would even strictly require, because inference cost also matters in practice (a smaller model trained longer is cheaper to *serve* forever after, even if slightly compute-suboptimal at training time) — a nuance worth mentioning: Chinchilla optimizes purely for training-compute-optimal loss, not for total lifetime cost including inference, which is why some modern recipes deliberately "overtrain" small models beyond the Chinchilla point.

---

## 4. Emergent Abilities

### The core claim (Wei et al., 2022, "Emergent Abilities of Large Language Models")
Some capabilities (e.g., multi-step arithmetic, certain few-shot reasoning tasks) show **near-zero performance** on smaller models, then **sharply jump to well-above-random performance** once model scale crosses some threshold — described as "emergent" because the capability wasn't predictable by smoothly extrapolating smaller models' performance, unlike the smooth power-law curves seen for loss itself.

**Practical example cited in the literature**: a task like 3-digit multiplication might show ~0% accuracy for models under some parameter threshold, then jump to 20-30%+ accuracy once a model crosses that threshold — looking like a step function/phase transition on a plot of accuracy vs. model scale, rather than a smooth curve.

### The counter-argument (Schaefer, Miller, Steinhardt et al., 2023, "Are Emergent Abilities a Mirage?") — the critical interview-level nuance
This paper argued that **many "emergent" jumps are artifacts of the *metric* chosen, not the underlying model capability.** Their key point:

- Many emergent-ability benchmarks use a **discontinuous/nonlinear metric**, like "exact match accuracy" on multi-step problems (e.g., you must get *every* digit of a multi-digit multiplication exactly right to score a point at all — partial credit is zero).
- If the model's **per-token or per-step error rate is actually improving smoothly and continuously** with scale (following the normal power-law loss curve from Module 3's earlier formula), but the task requires getting *many* steps right simultaneously (e.g., 5 correct digits in a row), then the **probability of getting the *whole sequence* right** is roughly `(per-step accuracy)^(number of steps)` — and this compounding creates an apparent sharp jump in the *all-or-nothing* metric even though the underlying per-step capability was improving smoothly the whole time.

### Numerical example proving this compounding effect
Suppose per-digit accuracy improves smoothly with model scale:
- Small model: 50% per-digit accuracy
- Medium model: 70% per-digit accuracy
- Large model: 90% per-digit accuracy

For a task requiring **5 digits correct in a row** (exact-match metric), overall accuracy = `(per-digit accuracy)^5`:
```
Small:  0.5^5  = 0.03125  → 3.1%
Medium: 0.7^5  = 0.16807  → 16.8%
Large:  0.9^5  = 0.59049  → 59.0%
```
Plotted, this looks like a dramatic, accelerating jump (3% → 17% → 59%) — a seemingly "emergent" curve — even though the *underlying* per-digit accuracy (50% → 70% → 90%) improved perfectly smoothly and linearly-ish the entire time. **This is the exact mechanism Schaefer et al. propose as the main driver of many reported "emergent abilities."** Switch to a smoother/partial-credit metric (e.g., "fraction of digits correct" instead of "all digits correct"), and the same underlying model-scale data often shows a smooth curve, not a sharp jump.

### The interview-ready synthesis (say this if asked "are emergent abilities real?")
"There's real debate here worth presenting both sides of: Wei et al. documented genuine sharp capability jumps on many benchmarks as models scale. Schaefer et al. showed that at least some (not necessarily all) of these jumps are explainable as a measurement artifact — nonlinear, all-or-nothing metrics can turn smooth underlying improvement into apparent phase transitions. The honest current position is: some emergent behavior may be real and some may be metric-driven, and the two explanations aren't mutually exclusive — the underlying loss curve *is* smooth and predictable (per Chinchilla/Kaplan), but that doesn't guarantee every downstream *task metric* built on top of that loss will also look smooth."

---

## 5. Loss vs. downstream task performance — the connecting thread

This module closes the loop opened in Module 2's perplexity discussion: **scaling laws are fit on the smooth, well-behaved loss curve** — but businesses/users care about downstream task performance (accuracy, reasoning, usefulness), which is a *derived, often nonlinear* function of that underlying loss. This is precisely why:
- Two models can have similar loss/perplexity but different task performance (Module 2's point).
- "Emergent abilities" can appear on task metrics even when the loss curve underneath is boringly smooth (this module's point).

**One sentence to have ready**: "Loss scales predictably and smoothly with compute — that part is essentially solved science. What's still debated is how that smooth loss improvement translates into discrete, real-world task capabilities, because the translation function (the metric) is often nonlinear."

---

## 6. Side-by-side summary table (memorize this cold)

| | Kaplan et al. (2020) | Chinchilla (Hoffmann et al., 2022) |
|---|---|---|
| Main claim | Prioritize model size (N) over data (D) | N and D should scale together, ~equally |
| Compute-optimal ratio | Implicitly favored much larger N relative to D | ~20 tokens per parameter |
| Real-world example | GPT-3 (175B params, ~300B tokens, ~1.7 tok/param — undertrained) | Chinchilla (70B params, ~1.4T tokens, ~20 tok/param) |
| Result | Led to huge, undertrained models | Chinchilla (smaller) beat GPT-3 (bigger) on benchmarks |
| Later refinement | — | Llama-style models "overtrain" small models beyond Chinchilla-optimal for cheaper inference |

---

## 7. Quick-fire Q&A (self-test)

**Q: State the general power-law scaling formula and explain what the exponent tells you.**
A: `L(N) = (N_c/N)^α` — loss falls as a power law with model size; a small α means diminishing returns, so cutting loss significantly requires order-of-magnitude increases in N, not just doubling it.

**Q: What was Kaplan's practical recommendation, and why did it turn out to be wrong?**
A: Kaplan recommended prioritizing model size over data for a fixed compute budget. It was later shown (Chinchilla) to be based on a methodological artifact in how learning-rate schedules were handled across training runs, leading to systematically undertrained large models like GPT-3.

**Q: What is Chinchilla's compute-optimal token-to-parameter ratio, and what's the formula connecting compute, params, and data?**
A: ~20 tokens per parameter; `C ≈ 6ND` (compute ≈ 6 × parameters × training tokens).

**Q: If you 4x your compute budget, how should Chinchilla-optimal N and D each change?**
A: Each should scale by roughly √4 = 2x (both N and D grow at the same rate, since C ≈ 6ND and the optimal split keeps their ratio fixed).

**Q: Why did Chinchilla (70B) outperform GPT-3 (175B) despite having fewer than half the parameters?**
A: Chinchilla was trained at a near-compute-optimal ~20 tokens/parameter ratio, while GPT-3 was trained at only ~1.7 tokens/parameter — GPT-3 was severely undertrained relative to its size, so Chinchilla's better data-to-param balance won out on downstream benchmarks.

**Q: Why might a company deliberately train a model beyond the Chinchilla-optimal point?**
A: Chinchilla optimizes only for training-compute efficiency; it ignores inference cost. A smaller model "overtrained" on more tokens than strictly compute-optimal can still be cheaper to serve at scale over its lifetime, even if training itself was slightly compute-suboptimal — this is the reasoning behind Llama-style recipes.

**Q: Explain the Schaefer et al. critique of emergent abilities using the compounding-accuracy mechanism.**
A: If per-step accuracy improves smoothly with scale, but the evaluation metric requires all steps correct simultaneously (exact match), then overall accuracy = (per-step accuracy)^(number of steps) — this compounding can turn smooth underlying improvement into an apparent sharp jump on the all-or-nothing metric, without any real discontinuity in the model's actual capability.

**Q: Does the Schaefer et al. paper prove emergent abilities aren't real?**
A: No — it shows the effect is at least partly a metric artifact for many benchmarks, not that every reported emergent ability is fake; the honest answer is nuanced, presenting both the original findings and the critique.

---
*End of Module 3 (expanded). Next: Module 4 — Fine-tuning vs Prompting vs In-Context Learning (LoRA math, when to fine-tune vs prompt vs RAG).*
