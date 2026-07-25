# Model Calibration — Interview Notes (End-to-End, From Scratch)

## 1. What Calibration Actually Means (the intuition first)

Every classifier that outputs a probability (logistic regression, gradient boosting, neural nets) is really making two separate claims:
1. **Ranking claim:** "this example is more likely positive than that one" — this is what AUC-ROC/PR measure (see AUC notes). A model can be *excellent* at this while being completely wrong about calibration.
2. **Calibration claim:** "when I say 70%, I really mean there's a 70% chance." — this is a completely separate property, and most models are NOT calibrated by default.

**The one-sentence definition:** A model is **well-calibrated** if, among all the examples it assigns a predicted probability of $p$, approximately $p$ fraction of them are actually positive.

```
Take every example where the model said "70% chance of fraud":

  WELL-CALIBRATED model:     WELL-CALIBRATED model:
  ┌─────────────────────┐    Out of 100 such examples,
  │ ●●●●●●●○○○           │    ~70 really are fraud.
  └─────────────────────┘

  POORLY-CALIBRATED model (overconfident):
  ┌─────────────────────┐    Out of 100 such examples,
  │ ●●●○○○○○○○           │    only ~30 really are fraud —
  └─────────────────────┘    the model said "70%" when
                              reality was closer to 30%.
```

**Why you should care, concretely:** if a model's "0.9 probability of default" outputs are used directly to set loan interest rates, or a "0.05 probability of disease" is shown to a doctor as an actual risk estimate, or a "0.8 confidence" score is used to decide how much budget to allocate in an ad auction — the *raw number* matters, not just whether higher-scored examples rank above lower-scored ones. A model can have great AUC-ROC (perfect ranking) and still be terribly calibrated (the actual numbers are nonsense) — these are orthogonal properties, and this is the single most important fact in this entire topic.

---

## 2. Why Models Become Miscalibrated in the First Place

**Not all models start out miscalibrated — some model families are naturally well-calibrated, others systematically distort probabilities. Know which is which:**

| Model | Typically... | Why |
|---|---|---|
| Logistic Regression | Well-calibrated out of the box | Directly optimizes log-loss, which is a *proper scoring rule* that rewards calibrated probabilities |
| Naive Bayes | Often **overconfident** (pushes probabilities toward 0 or 1) | The (usually false) conditional-independence assumption compounds — multiplying many "independent" likelihoods makes the model more extreme than it should be |
| Decision Trees | Poorly calibrated, especially deep ones | Leaf-node probabilities are just raw class frequencies in that leaf — small leaves (from deep trees) give extreme, noisy 0%/100%-ish estimates |
| Random Forest | Better than a single tree, but still often **underconfident** in the extremes (compressed toward 0.5) | Averaging many trees' outputs smooths extreme predictions toward the middle, even when the true probability really is near 0 or 1 |
| Gradient Boosting (XGBoost/LightGBM) | Reasonably good but can still be off, especially with heavy regularization or class-imbalance handling | Techniques like class weighting or aggressive regularization shift the score distribution away from true probabilities |
| SVMs | Poorly calibrated by default (raw output isn't even a probability — it's a distance from the margin) | SVMs don't optimize a probabilistic objective at all; the "probability" is a post-hoc transformation of a geometric margin distance |
| Neural Networks (deep, modern) | Often **overconfident**, especially larger/deeper models | Well-documented empirical phenomenon (Guo et al. 2017) — increasing depth/capacity tends to increase confidence faster than accuracy, even with techniques like batch norm |

**Other causes of miscalibration, regardless of model family:**
- **Class rebalancing techniques** (oversampling minority class, class weighting, SMOTE) — these deliberately shift the training distribution away from the real-world class balance, which distorts the model's output probabilities away from true posterior probabilities unless corrected afterward (see §6).
- **Threshold-agnostic loss functions or heavy regularization** that don't specifically reward well-calibrated outputs even while optimizing overall discrimination well.
- **Train/serving distribution shift** — a model calibrated at training time can silently drift out of calibration in production as the real-world class balance or feature distribution changes (this connects directly to the skew/drift material in Cross-Validation and Imbalanced Data notes).

---

## 3. How to Diagnose Miscalibration — the Reliability Diagram

**This is the primary tool, and the one interviewers expect you to be able to describe from memory.**

**Construction steps:**
1. Take your model's predicted probabilities on a held-out validation set.
2. Bin the predictions into buckets (commonly 10 bins: [0-0.1), [0.1-0.2), ..., [0.9-1.0]).
3. For each bin, compute (a) the **average predicted probability** of examples in that bin, and (b) the **actual observed fraction of positives** in that bin.
4. Plot observed fraction (y-axis) against average predicted probability (x-axis) for each bin.
5. A perfectly calibrated model produces points exactly on the diagonal $y=x$.

```
Observed
fraction
positive
  1.0│                                    ╱
     │                                  ╱  ← perfect calibration (diagonal)
     │                                ╱
     │                    ●        ╱
     │                          ╱
     │              ●        ╱
     │           ╱ ●
     │        ╱●
     │  ● ╱  ← points BELOW diagonal here = OVERCONFIDENT
     │╱      (model says e.g. 0.3, but real rate is lower, ~0.15)
  0.0└──────────────────────────────────────► Average Predicted Probability
    0.0                                    1.0

     Points ABOVE the diagonal = UNDERCONFIDENT
     (model says e.g. 0.6, but real rate is actually higher, ~0.8)

     Points BELOW the diagonal = OVERCONFIDENT
     (model says e.g. 0.8, but real rate is actually lower, ~0.6)
```

**Worked numerical example (10-bin reliability check on a fraud model, validation set):**

| Predicted prob. bin | # examples in bin | Mean predicted prob | # actually fraud | Observed fraction | Diagnosis |
|---|---|---|---|---|---|
| 0.0–0.1 | 500 | 0.04 | 15 | 0.030 | Slightly overconfident (said 4%, real 3%) |
| 0.4–0.5 | 80 | 0.45 | 28 | 0.350 | **Overconfident** (said 45%, real 35%) |
| 0.7–0.8 | 40 | 0.75 | 22 | 0.550 | **Badly overconfident** (said 75%, real 55%) |
| 0.9–1.0 | 20 | 0.95 | 12 | 0.600 | **Severely overconfident** (said 95%, real 60%!) |

This pattern — increasing overconfidence at higher predicted probabilities — is extremely common in tree ensembles and deep neural nets, and is exactly what a reliability diagram is designed to surface at a glance.

---

## 4. Numerical Summary Metrics for Calibration

**Expected Calibration Error (ECE)** — the standard scalar summary of a reliability diagram:

$$ECE = \sum_{m=1}^{M} \frac{n_m}{N} \left| \text{acc}(m) - \text{conf}(m) \right|$$

where $M$ = number of bins, $n_m$ = examples in bin $m$, $N$ = total examples, $\text{acc}(m)$ = observed fraction of positives in bin $m$, $\text{conf}(m)$ = mean predicted probability in bin $m$.

**Worked example using the table from §3** (assume these 4 bins account for all 1000 examples for simplicity — real ECE would use all 10 bins):

$$ECE = \frac{500}{1000}|0.030-0.04| + \frac{80}{1000}|0.350-0.45| + \frac{40}{1000}|0.550-0.75| + \frac{20}{1000}|0.600-0.95|$$
$$= 0.5(0.010) + 0.08(0.100) + 0.04(0.200) + 0.02(0.350)$$
$$= 0.0050 + 0.0080 + 0.0080 + 0.0070 = 0.0280$$

**ECE ≈ 0.028** — on average, predicted probabilities are off by about 2.8 percentage points, weighted by how many examples fall in each bin. Lower is better; 0 is perfect calibration.

**Brier Score** — mean squared error between predicted probability and the actual binary outcome (0 or 1):

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2$$

- Ranges 0 (perfect) to 1 (worst possible). Unlike ECE, Brier score conflates calibration **and** discrimination/refinement into one number — it can be decomposed into a calibration term plus a "refinement" (sharpness/resolution) term. Two models can have the same Brier score for different reasons (one poorly calibrated but sharp, another well-calibrated but not very discriminative) — this is why ECE and reliability diagrams remain the more direct calibration-specific diagnostic.

---

## 5. Fixing Miscalibration — the Methods, End to End

**A) Platt Scaling** — fit a simple logistic regression on top of the model's raw outputs:

$$P(y=1 \mid s) = \frac{1}{1 + e^{-(As + B)}}$$

where $s$ is the model's raw score/logit and $A, B$ are two scalar parameters learned on a held-out calibration set (never the original training set — that would just re-fit to already-seen noise).

```
Raw score s ──► [ learn A, B via logistic regression on held-out cal set ] ──► calibrated probability
```

- Works well when the miscalibration follows a roughly sigmoid-shaped distortion (common for SVMs and boosted trees).
- Only 2 parameters — very data-efficient, works fine even with a small calibration set (few hundred to a few thousand examples).

**B) Isotonic Regression** — fit a non-parametric, monotonically increasing step function mapping raw scores to calibrated probabilities.

```
Raw score:        0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9
Isotonic output:  0.05  0.05  0.15  0.15  0.15  0.40  0.40  0.75  0.90
                   └────┘            └──────────┘      
                  flat "step" where the raw scores didn't reflect
                  real differences in outcome — isotonic regression
                  can flatten or bend arbitrarily, unlike Platt's
                  fixed sigmoid shape
```

- More flexible than Platt scaling (can fit any monotonic shape, not just a sigmoid) — but needs **more data** to avoid overfitting the calibration mapping itself (a common rule of thumb: isotonic wants at least ~1000+ calibration examples; with less data, Platt scaling tends to generalize better).
- **Monotonicity constraint is the key structural assumption**: it will never say a higher raw score should map to a lower calibrated probability — this preserves the model's ranking/AUC exactly while fixing the actual probability values, which is a nice guarantee to state in an interview.

**C) Temperature Scaling** (the modern deep-learning-era standard, Guo et al. 2017) — a simplified, single-parameter version of Platt scaling applied to the pre-softmax logits:

$$P(y=k \mid \mathbf{z}) = \text{softmax}(\mathbf{z}/T)$$

- $T$ (temperature) is a single scalar learned on a held-out set, minimizing negative log-likelihood.
- $T > 1$ "softens" overconfident predictions (spreads the probability mass out, pulling extreme values back toward the middle) — this is the most common correction needed for modern deep nets, which are systematically overconfident (see §2 table).
- $T = 1$ is the identity (no change); $T < 1$ sharpens/increases confidence (rarely needed in practice).
- Extremely cheap: one parameter, doesn't change the model's ranking/argmax predictions at all (softmax with any $T>0$ preserves the ordering) — purely fixes the confidence values.

**D) Practical workflow, step by step:**
1. Train your model as normal on the training set.
2. Hold out a **separate calibration set** (distinct from both training and final test data) — reusing training data for calibration re-fits to already-memorized noise; reusing your test set for calibration then evaluating on it leaks information and inflates your reported calibration quality.
3. Generate raw predicted probabilities/scores on the calibration set.
4. Fit Platt scaling, isotonic regression, or temperature scaling (choose based on model type and calibration-set size — see comparison below) using the calibration set's raw scores and true labels.
5. Apply the fitted calibration mapping to new predictions at inference/serving time.
6. Re-evaluate the reliability diagram and ECE on a held-out test set (not the calibration set) to confirm the fix actually worked, and periodically re-check in production since calibration can drift (see §2's train/serving shift point).

**Which method to pick — quick decision guide:**

| Situation | Recommended method |
|---|---|
| Small calibration set (hundreds of examples) | Platt scaling (fewer parameters, less overfitting risk) |
| Large calibration set (thousands+), non-sigmoid miscalibration shape | Isotonic regression |
| Deep neural network, multi-class softmax output | Temperature scaling |
| SVM raw margin outputs | Platt scaling (this is literally what it was originally designed for) |
| Need to preserve exact ranking/AUC while only fixing probability values | Any of the three — all are monotonic transformations, so none change ranking/AUC |

---

## 6. Special Case: Calibration After Class Rebalancing

If you trained on an artificially rebalanced dataset (oversampled minority class, SMOTE, or class-weighted loss) to handle imbalance (see Imbalanced Data notes), your model's raw output probabilities are now calibrated to the **rebalanced** training distribution, not the real-world one — a critical, very commonly-missed step.

```
True world:           2% fraud, 98% legit
Rebalanced training:  50% fraud, 50% legit  (via oversampling)

Model trained on rebalanced data says "60% chance of fraud" for some transaction.
This 60% reflects the REBALANCED world's base rate, not the real 2% world.

Correction needed (via Bayes' rule, if you know the rebalancing ratio):

  p_corrected = (p_model × true_prior / rebalanced_prior) 
                ────────────────────────────────────────────────────────
                (p_model × true_prior/rebalanced_prior) + (1-p_model) × (true_prior_neg/rebalanced_prior_neg)
```

- In practice, this is most commonly handled by simply **re-calibrating** (Platt/isotonic) on a held-out set drawn from the *true* (unrebalanced) distribution, rather than manually deriving the Bayes correction factor — the recalibration step naturally absorbs the prior-shift distortion.
- This is a favorite FAANG follow-up question precisely because it's an easy-to-miss real-world bug: teams handle imbalance correctly for training, then forget the resulting probabilities are no longer meaningful as real-world probabilities without a recalibration pass.

---

## 7. Common Pitfalls (interviewers love probing these)

1. **Believing high AUC-ROC implies good calibration.** These are orthogonal — a model can perfectly rank examples (AUC=1.0) while being wildly miscalibrated in its actual probability values (e.g., always outputting either 0.51 or 0.99 regardless of true risk, as long as the ordering is right).
2. **Calibrating on the training set.** This just re-fits the calibration mapping to noise the model has already memorized — always use a separate held-out calibration set.
3. **Evaluating calibration quality on the same set used to fit the calibration mapping.** Leaks information, inflates the apparent fix — use a third, untouched set for the final reliability diagram/ECE check.
4. **Using isotonic regression with too little calibration data.** Its flexibility (no fixed shape assumption) is a double-edged sword — with only a few hundred examples it can overfit noisy bins into a jagged, unreliable step function; Platt scaling's 2-parameter simplicity generalizes better in that regime.
5. **Forgetting that class rebalancing during training breaks calibration** (see §6) — a very commonly missed real-world step.
6. **Assuming calibration is a "set it and forget it" fix.** Just like model performance, calibration can drift in production as the real-world class balance or feature distribution shifts over time — needs periodic re-checking, not a one-time fix.
7. **Conflating Brier score with pure calibration quality.** Brier score bundles calibration and discrimination together; two models can have identical Brier scores for very different reasons (see §4) — use ECE/reliability diagrams when you specifically want to isolate calibration.
8. **Choosing bin count for reliability diagrams carelessly.** Too few bins (e.g., 2-3) hides fine-grained miscalibration patterns; too many bins (e.g., 50+) makes each bin too sparse to give a statistically meaningful observed fraction — 10 bins is a common, reasonable default, but always check bin sample sizes before trusting the diagram.

---

## 8. FAANG-Level Interview Q&A

**Q1: A model has AUC-ROC of 0.95 — does that mean its predicted probabilities can be trusted as real probabilities (e.g., for a downstream pricing decision)?**
No — AUC-ROC only measures ranking quality (whether positives tend to score above negatives), which is completely orthogonal to calibration (whether "0.7" really means a 70% chance). A model can have near-perfect AUC while its raw scores are systematically overconfident or underconfident — before using the raw probability values for any decision where the *number itself* matters (pricing, risk budgets, thresholding based on expected cost), you'd need to check a reliability diagram / ECE and calibrate if necessary.

**Q2: You trained a model using SMOTE to handle 2% positive-class imbalance. It's now well-calibrated on your (rebalanced) validation set. Is it calibrated in production?**
No, and this is a very common real-world bug — the model's output probabilities reflect the artificially rebalanced 50/50 training distribution, not the true 2% production base rate. A "60% probability of fraud" from this model doesn't mean a 60% real-world chance; it needs to be recalibrated against a held-out set drawn from the true (unrebalanced) production distribution, either via an explicit Bayes prior-correction or (more commonly) by fitting Platt scaling/isotonic regression on properly-representative data.

**Q3: Why do modern deep neural networks tend to be overconfident, and how would you fix it cheaply without touching the underlying model?**
This is a well-documented empirical phenomenon (Guo et al. 2017) — increasing model capacity/depth tends to increase confidence in predictions faster than it increases actual accuracy, even with regularizers like batch normalization. The cheapest fix is **temperature scaling**: learn a single scalar $T$ on a held-out set that divides the pre-softmax logits before the softmax, softening (T>1) the overconfident distribution without changing the model's ranking or top-1 predictions at all (since softmax with any positive T preserves ordering) — it's a one-parameter, post-hoc fix that requires no retraining.

**Q4: What's the difference between Platt scaling and isotonic regression, and how would you decide which to use given a specific calibration dataset size?**
Platt scaling fits a 2-parameter logistic sigmoid mapping raw scores to probabilities — data-efficient, works well when the miscalibration distortion is roughly sigmoid-shaped, but can't correct non-sigmoid distortions. Isotonic regression fits a non-parametric monotonically-increasing step function — much more flexible (can correct any monotonic miscalibration shape) but needs substantially more calibration data (roughly 1000+ examples as a rule of thumb) to avoid overfitting the mapping itself to noisy bins. With only a few hundred calibration examples, I'd default to Platt scaling; with a large held-out calibration set and evidence of non-sigmoid miscalibration on the reliability diagram, isotonic regression.

**Q5: Two models have identical Brier scores. Does that mean they have identical calibration quality?**
Not necessarily — Brier score is a combination of calibration error and "refinement"/sharpness (discrimination ability), and it can be formally decomposed into those two components. Two models can land on the same overall Brier score for very different reasons: one might be well-calibrated but not very discriminative (predictions cluster near the base rate), while another might be poorly calibrated but sharper/more discriminative (confident, spread-out predictions that are systematically biased). If you specifically want to isolate calibration quality, use ECE and a reliability diagram rather than relying on Brier score alone.

**Q6: Does calibrating a model (via Platt scaling, isotonic regression, or temperature scaling) change its AUC-ROC?**
No — all three are **monotonic** transformations of the raw score (Platt and temperature scaling are strictly monotonic sigmoids; isotonic regression is monotonic by construction/constraint). Since AUC-ROC depends only on the relative *ranking* of scores, not their absolute values, any monotonic recalibration leaves AUC-ROC (and AUC-PR, and the model's argmax/threshold-0.5 classification decisions under the corresponding recalibrated threshold) mathematically unchanged. This is a good guarantee to state explicitly — calibration is a "free" fix for probability values that doesn't cost you anything on the ranking-based metrics.

**Q7: Your production model was well-calibrated at launch six months ago. A dashboard now shows increasing complaints about risk scores "feeling off." How would you investigate?**
First pull recent production data and rebuild a reliability diagram / compute ECE on a fresh sample — compare against the original launch-time calibration curve to see if and how the distortion pattern has shifted (e.g., is it now systematically overconfident where it wasn't before?). This is likely a symptom of population/concept drift — the real-world class balance or feature distribution has shifted since launch (directly analogous to the training-serving skew material in the Cross-Validation notes) — and the fix is typically to refit the calibration mapping (or the underlying model) on more recent data, plus set up ongoing calibration monitoring rather than treating the original fix as permanent.

**Q8 (clever): Can a model be perfectly calibrated overall (ECE = 0 across the whole population) while still being badly miscalibrated for a specific subgroup?**
Yes — this is a well-known and important failure mode. Overall ECE aggregates across the entire population, so a model could be simultaneously overconfident for one subgroup and underconfident for another in a way that exactly cancels out in the aggregate reliability diagram, while both subgroups individually experience real calibration problems. The fix is the same principle as the fairness point in the Confusion Matrix notes — compute separate reliability diagrams/ECE per relevant subgroup, not just one pooled diagram, whenever subgroup-level trust in the probability values matters (e.g., a risk score used differently across demographic or geographic segments).

---

## 9. One-Line Interview Closers

- *"Calibration and discrimination are orthogonal — a model can rank perfectly and still be numerically dishonest about its confidence, and AUC-ROC will never tell you which one you have."*
- *"Any of the standard fixes — Platt, isotonic, temperature scaling — are monotonic, so they fix the numbers without touching the ranking; calibration is essentially a free lunch on top of a model you've already built."*
- *"If you rebalanced your training data to handle class imbalance, your model's probabilities are calibrated to a world that doesn't exist in production — recalibrating on the true distribution isn't optional, it's the step everyone forgets."*
