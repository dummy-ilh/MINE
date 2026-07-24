# Precision, Recall, F1 — Interview Notes

## 1. Core Definitions (from the Confusion Matrix)

```
                              PREDICTED
                        Positive        Negative
                    ┌───────────────┬───────────────┐
           Positive │      TP        │      FN        │
  ACTUAL             │                │                │
           Negative │      FP        │      TN        │
                    └───────────────┴───────────────┘
```

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{— "of everything I flagged as positive, how much was actually positive?"}$$

$$\text{Recall (Sensitivity, TPR)} = \frac{TP}{TP + FN} \qquad \text{— "of everything that was actually positive, how much did I catch?"}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \qquad \text{— harmonic mean of the two}$$

- Precision is about the **quality** of positive predictions (how much you can trust a "yes").
- Recall is about the **coverage** of actual positives (how much of the truth you found).
- They trade off against each other as you move the classification threshold (see §4) — you cannot generally maximize both simultaneously without more/better information (better features, more data, a better model).

---

## 2. Why F1 Uses the Harmonic Mean, Not the Arithmetic Mean

```
Arithmetic mean rewards extremes too generously:
  Precision = 1.0, Recall = 0.01  →  Arithmetic mean = 0.505   ← looks decent, but recall is basically 0!

Harmonic mean punishes imbalance between the two:
  Precision = 1.0, Recall = 0.01  →  F1 = 2·(1.0×0.01)/(1.0+0.01) = 0.0198   ← correctly reflects the model is nearly useless
```

- The harmonic mean is always **≤** the arithmetic mean of the same two numbers, and it's pulled sharply toward whichever of the two values is **smaller**. This is exactly why F1 is preferred as a single blended score — it penalizes a model that's great on one axis and terrible on the other, rather than letting a high value on one side hide a collapse on the other.
- F1 only equals precision (=recall) when precision = recall exactly.

---

## 3. Worked Numerical Sensitivity Analysis — "What Happens When TP Is Added?"

Start with a baseline confusion matrix (binary fraud detection, out of 1000 transactions, 50 actual fraud):

```
Baseline:            TP=40   FN=10   FP=30   TN=920
Precision = 40/70  = 0.571
Recall    = 40/50  = 0.800
F1        = 2(0.571×0.800)/(0.571+0.800) = 0.914/1.371 = 0.667
```

### Case A — Add 1 TP (model correctly catches one more fraud it previously missed; FN decreases by 1 since that example moves from FN → TP)

```
New:                  TP=41   FN=9    FP=30   TN=920
Precision = 41/71  = 0.577   (↑ from 0.571 — up slightly, since TP grew and FP stayed same)
Recall    = 41/50  = 0.820   (↑ from 0.800 — meaningful jump, since FN shrank)
F1        = 2(0.577×0.820)/(0.577+0.820) = 0.946/1.397 = 0.677  (↑ from 0.667)
```
**Takeaway:** adding a TP (by correctly converting a previous FN) improves **both** precision and recall — always a strict win. This is the only lever that helps both simultaneously without cost.

### Case B — Add 1 FP (model now also flags one more legit transaction as fraud, everything else unchanged)

```
New:                  TP=40   FN=10   FP=31   TN=919
Precision = 40/71  = 0.563   (↓ from 0.571 — precision drops, more false alarms diluting it)
Recall    = 40/50  = 0.800   (unchanged — recall only depends on TP and FN)
F1        = 2(0.563×0.800)/(0.563+0.800) = 0.900/1.363 = 0.661  (↓ from 0.667)
```
**Takeaway:** an added FP hurts precision and F1, but recall is completely untouched — recall's denominator ($TP+FN$) never includes FP at all. This is a common "gotcha" fact: recall is structurally blind to false positives.

### Case C — Add 1 FN (model now misses one it previously caught; that example moves from TP → FN)

```
New:                  TP=39   FN=11   FP=30   TN=920
Precision = 39/69  = 0.565   (↓ slightly from 0.571)
Recall    = 39/50  = 0.780   (↓ meaningfully from 0.800)
F1        = 2(0.565×0.780)/(0.565+0.780) = 0.441/1.345 = 0.655  (↓ from 0.667)
```
**Takeaway:** losing a TP to an FN hurts both metrics — the mirror image of Case A.

### Case D — Threshold moved to be more "aggressive" (lower threshold, flags more as positive): TP +5, FP +25, FN −5, TN −25

```
New:                  TP=45   FN=5    FP=55   TN=895
Precision = 45/100 = 0.450   (↓ sharply — precision collapses as many more FPs pour in)
Recall    = 45/50  = 0.900   (↑ sharply — almost everything actual is now caught)
F1        = 2(0.450×0.900)/(0.450+0.900) = 0.810/1.350 = 0.600  (↓ from 0.667 despite recall improving!)
```
**Takeaway — the classic interview surprise:** recall went up a lot, but F1 went *down*, because precision collapsed even faster. This demonstrates F1 is not simply "higher recall = higher F1" — it's a balance, and pushing too hard on one side without regard for the other can net a *worse* blended score.

### Summary Table of All Four Cases

| Change | Precision | Recall | F1 | Net effect |
|---|---|---|---|---|
| Baseline | 0.571 | 0.800 | 0.667 | — |
| +1 TP (FN→TP) | 0.577 ↑ | 0.820 ↑ | 0.677 ↑ | Strict improvement |
| +1 FP | 0.563 ↓ | 0.800 (unchanged) | 0.661 ↓ | Precision-only damage |
| +1 FN (TP→FN) | 0.565 ↓ | 0.780 ↓ | 0.655 ↓ | Both hurt |
| Lower threshold (+5TP, +25FP, −5FN, −25TN) | 0.450 ↓↓ | 0.900 ↑↑ | 0.600 ↓ | Recall up, F1 still down |

---

## 4. Precision-Recall Tradeoff via Threshold

```
Score/probability axis:  0.0 ────────────────────────────── 1.0
                                    ↑              ↑
                          low threshold       high threshold
                          (predict positive    (predict positive
                          more often)           more rarely)

Low threshold  →  more predicted positives  →  TP↑ and FP↑  →  Recall↑, Precision↓ (usually)
High threshold →  fewer predicted positives →  TP↓ and FP↓  →  Precision↑ (usually), Recall↓
```

```
Precision
   │╲
   │ ╲___
   │     ╲____
   │          ╲_____
   │                ╲________
   │                         ╲___________
   └──────────────────────────────────────► Recall
   0                                        1

Typical shape: precision stays high while recall is low (only very confident
predictions made), then drops off as you push recall higher by lowering
the threshold and sweeping in noisier, less-confident positive calls.
```

- This curve — precision vs. recall swept across every threshold — is the **PR curve**, and the area under it (**PR-AUC / Average Precision**) is the threshold-independent summary metric, directly relevant when discussing imbalanced classification (see Imbalanced Data notes).
- **F1 at a single threshold is a snapshot**, exactly like a single confusion matrix (see Confusion Matrix notes §7 pitfall 4) — always ask "F1 at what threshold?" when someone quotes a single F1 number without specifying.

---

## 5. How to "Boost F1" — the Actual Levers

**Because F1 is a snapshot at one threshold, there are two separate categories of levers:**

**A) Threshold-only levers (no retraining, same model, just moving the cutoff):**
- Sweep the threshold and pick the value that **maximizes F1 directly** on a held-out validation set — this is the simplest, most common "boost F1" answer, and it costs nothing beyond re-scoring.
- Because F1 weights precision and recall equally, the F1-optimal threshold is typically somewhere in the middle of the precision-recall tradeoff curve — not at the default 0.5, and not at either extreme.
- Use **F-beta** ($F_\beta = (1+\beta^2)\cdot\frac{P \cdot R}{\beta^2 P + R}$) instead of plain F1 if precision and recall genuinely matter unequally to the business — $\beta>1$ weights recall more (e.g., $F_2$ for medical screening), $\beta<1$ weights precision more (e.g., $F_{0.5}$ for spam filtering) — this isn't "cheating," it's making the objective match the actual cost asymmetry.

**B) Model-improvement levers (actually change the underlying TP/FP/FN counts, not just the threshold):**
- **More/better features** — can genuinely push both precision and recall up simultaneously (the "strict win" from Case A), unlike threshold-moving which trades one for the other.
- **More training data**, especially more positive-class examples if the class is rare — reduces variance, helps the model separate classes more cleanly at every threshold.
- **Better handling of class imbalance** (resampling, class weighting, focal loss) — shifts the underlying score distribution so precision and recall can both improve at a shared threshold, rather than just relocating where you cut.
- **Ensembling / better model architecture** — a stronger base classifier moves the entire PR curve up and to the right (higher precision at every recall level), which is a real improvement, distinct from sliding along a fixed curve.
- **Fixing label noise or data leakage** — mislabeled training examples cap how good precision/recall can ever get regardless of threshold; cleaning labels can unlock gains no threshold-tuning could reach.

**Common mistake to flag in an interview:** "boosting F1" by just moving the threshold to the model's own F1-optimal point is a legitimate and cheap first step, but if asked "how would you actually improve the model," threshold-tuning is NOT the answer — that's optimizing measurement, not the underlying model. The real answer is category B.

---

## 6. Common Pitfalls (interviewers love probing these)

1. **Reporting F1 without specifying the threshold it was computed at.** F1 (like the confusion matrix it's derived from) is threshold-specific — always state or ask for the threshold, or better, report PR-AUC for a threshold-independent comparison.
2. **Believing recall can be improved by fixing false positives.** Recall's formula ($TP/(TP+FN)$) never contains FP — reducing FPs affects precision only. This is a very common conceptual slip under time pressure (see Case B above).
3. **Assuming raising recall always helps F1.** As Case D shows, aggressively lowering the threshold can raise recall while F1 falls, because precision can collapse faster than recall climbs.
4. **Using F1 when the actual costs of FP vs FN are very unequal.** Plain F1 implicitly assumes precision and recall matter equally — if they don't (e.g., cancer screening), use $F_\beta$ with an appropriate $\beta$, or optimize directly for a cost-weighted objective, not vanilla F1.
5. **Averaging F1 across classes the wrong way in multi-class settings.** See the Confusion Matrix notes §6 numerical example — macro-F1 and micro-F1 (which equals accuracy in single-label settings) can diverge sharply under imbalance; conflating them misrepresents which behavior you're actually measuring.
6. **Forgetting F1 is undefined (0/0) when both TP and (FP+FN) or the relevant sums are zero.** E.g., if a class never appears and the model never predicts it, precision and recall are both undefined by the raw formula — libraries typically return 0 or raise a warning; know your library's convention so you don't silently misinterpret a "0" as "the model performed terribly" when it's actually "there was nothing to evaluate."
7. **Treating F1 as inherently superior to accuracy in all cases.** F1 is the right choice under class imbalance or asymmetric costs — for balanced classes with symmetric costs, plain accuracy is simpler and just as valid; reaching for F1 reflexively without checking if imbalance/asymmetry actually exists is cargo-culting.

---

## 7. FAANG-Level Interview Q&A

**Q1: You add a false positive to your confusion matrix. What happens to precision, recall, and F1, and why does one of them not move at all?**
Precision drops (FP is in its denominator, so the same TP is now divided by a larger number). Recall is completely unchanged — its formula is $TP/(TP+FN)$, which structurally never involves FP at all. F1 drops because it's a function of precision (which dropped) and recall (unchanged) — the harmonic mean moves down whenever either input moves down. This tests whether the candidate actually understands the formulas rather than pattern-matching "more errors = everything gets worse."

**Q2: A model's recall goes from 0.80 to 0.90 after lowering the classification threshold, but F1 drops from 0.667 to 0.600. Is this model strictly worse now?**
Not necessarily "worse" in an absolute sense — it depends on the actual cost of false positives vs. false negatives in the business context. F1 assumes precision and recall matter equally, but if this is (say) a cancer-screening use case where missing a positive (FN) is far more costly than a false alarm (FP), the higher-recall, lower-F1 model might genuinely be the better operating point for that domain — F1 dropping only means the *equally-weighted* blend got worse, not that every reasonable weighting agrees. This is the exact scenario where $F_\beta$ (beta>1) or a direct cost-weighted metric would tell a very different story than vanilla F1.

**Q3: How would you find the threshold that maximizes F1 for a given model, and what are the pitfalls of doing so?**
Sweep the threshold across the model's score range on a held-out validation set, compute precision/recall/F1 at each point, and pick the threshold with the highest F1 (or plot the full PR curve and read off the F1-maximizing point directly, since F1 is derivable pointwise along the curve). Pitfall: this optimizes a single equally-weighted blend, which may not reflect the true business cost asymmetry — and doing this tuning on the same data used for final reporting risks the same kind of overfitting-to-the-validation-set leakage discussed in the Cross-Validation notes (use a separate held-out set, or nested CV, for an honest final number).

**Q4: True or false — a model with higher precision AND higher recall than another model will always have a higher F1.**
True — F1 is a monotonically increasing function of both precision and recall individually (holding the other fixed), so if Model A dominates Model B on both axes simultaneously, A's F1 is guaranteed to be higher. The tricky/interesting case (and where most interview questions live) is when one model has higher precision but lower recall than another — there's no such guarantee, and you must actually compute F1 (or compare the full PR curves) rather than eyeballing which "seems better."

**Q5: Why is the harmonic mean used for F1 instead of the arithmetic mean of precision and recall?**
The harmonic mean is much more sensitive to the smaller of the two input values — it heavily penalizes a large imbalance between precision and recall (e.g., precision=1.0, recall=0.01 → arithmetic mean ≈ 0.5 which looks respectable, but harmonic mean ≈ 0.02 which correctly signals a nearly useless model since it's essentially catching nothing). The arithmetic mean would let a model that's extreme on one axis and terrible on the other look artificially decent; the harmonic mean won't let it hide.

**Q6: Your team reports "F1 improved from 0.70 to 0.72" after a model change. What follow-up questions would you ask before trusting this as a real improvement?**
(1) Was this measured at the same threshold, or was the threshold re-tuned for the new model — is this a genuine model improvement or a threshold-optimization artifact? (2) Was this evaluated via proper cross-validation with reported variance across folds, or a single train/val split — is a 0.02 F1 gap within CV noise? (3) Is this macro, micro, or weighted F1 if multi-class — the three can disagree, especially under imbalance, so "F1 improved" is ambiguous without specifying which. (4) Did anything about the data distribution (train/val split, class balance) change between the two evaluations in a way that isn't a true apples-to-apples comparison?

**Q7: In a multi-class setting, you're told to "optimize for F1." What clarifying question is essential before you start, and why?**
"Macro, micro, or weighted F1?" — because they can diverge sharply, especially under class imbalance (see Confusion Matrix notes §6 worked example). Optimizing for macro-F1 pushes you to care disproportionately about rare classes (a mistake catastrophically hurting a tiny class tanks macro-F1 even if it barely affects overall performance), while micro-F1 (equivalent to accuracy in single-label multi-class) lets frequent classes dominate and can mask poor performance on rare-but-important classes entirely. Building the wrong objective silently optimizes for the wrong business outcome.

**Q8 (clever): Can F1 be higher than both precision and recall?**
No — the harmonic mean of two positive numbers always lies between the smaller and larger of the two values (specifically, closer to the smaller one), so F1 is always bounded between min(precision, recall) and max(precision, recall). If someone reports an F1 that falls outside that range, it's a calculation bug, not a legitimate result — a good sanity-check fact to keep in your back pocket when reviewing someone else's metrics dashboard.

---

## 8. One-Line Interview Closers

- *"Recall is structurally blind to false positives — its formula simply doesn't contain FP, which is why 'just reduce false positives' never moves recall."*
- *"F1 is a snapshot at one threshold, exactly like a confusion matrix — 'boost F1' by threshold-tuning is optimizing measurement, not the model; real improvement moves the whole PR curve, not just where you cut it."*
- *"The harmonic mean exists specifically so a model can't hide a collapse on one axis behind strength on the other — that's the whole reason F1 isn't just the average of precision and recall."*
