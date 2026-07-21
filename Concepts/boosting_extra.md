## DART, Monotonic Constraints, Class Imbalance

---

# PART 1: DART

---

## The Problem DART Solves

In standard GBM, early trees dominate. Here's why.

Tree 1 fits the biggest, most obvious patterns in the data. It captures the bulk of the signal. Its contribution to the final prediction is large.

Tree 2 fits residuals of tree 1 — smaller signal. Smaller contribution.

Tree 50 fits tiny residuals — almost noise. Tiny contribution.

```
Final prediction:
F(x) = 0.8×Tree1 + 0.3×Tree2 + 0.1×Tree3 + ... + 0.001×Tree50
```

Early trees have **outsized influence.** Late trees barely matter. This is called **over-specialization** — the ensemble is essentially a few early trees with decoration.

---

## What DART Does

DART = **Dropouts meet Multiple Additive Regression Trees.**

Borrowed directly from dropout in neural networks.

At each boosting round, instead of fitting residuals of ALL previous trees, DART **randomly drops a subset of previous trees** and fits residuals of the remaining ones.

```
Standard GBM round 5:
  residual = y - (Tree1 + Tree2 + Tree3 + Tree4)
  fit Tree5 to this residual

DART round 5 (drop Tree1, Tree3):
  residual = y - (Tree2 + Tree4)
  fit Tree5 to THIS residual instead
```

Tree5 now has to compensate for the missing trees — it learns to be more **self-sufficient**, not just a tiny correction on top of everything else.

---

## Why This Helps

By randomly dropping early trees during training, later trees are forced to learn patterns **as if the early trees weren't there.** They become more robust, more independent, more capable on their own.

The result: all trees contribute more equally to the final prediction. No single early tree dominates.

```
Standard GBM: Tree1=80%, Tree2=12%, Tree3=5%, ...TreeN=0.001%
DART:         Tree1=25%, Tree2=22%, Tree3=20%, ...TreeN=15%
```

More democratic ensemble. Better generalization.

---

## The Normalization Trick

When you drop trees during training but use all trees at inference, there's a scale mismatch. DART fixes this by **rescaling** the dropped trees' contributions at prediction time.

```
If you dropped 2 out of 10 trees during training:
Scale factor = (dropped trees) / (total trees) = 2/10 = 0.2

Rescale surviving trees by (1 - 0.2) = 0.8
```

This keeps predictions on the same scale whether trees were dropped or not.

---

## Key Hyperparameters

**drop_rate** — fraction of trees dropped each round
- Too low (0.0): degenerates to standard GBM
- Too high (0.9): almost no context, trees learn in isolation, underfits
- Sweet spot: 0.1–0.5

**skip_drop** — probability of skipping dropout entirely for a round
- Gives some rounds standard GBM behavior
- Adds stability

**normalize_type** — how to rescale after dropping
- `tree`: normalize by number of trees dropped
- `forest`: normalize by contribution of dropped trees

---

## DART vs Standard GBM

| | Standard GBM | DART |
|---|---|---|
| Early tree dominance | Yes — big problem | Solved |
| Training speed | Fast | Slower (rescaling overhead) |
| Early stopping | Works cleanly | Broken — see below |
| Overfitting control | η + depth + subsampling | Dropout adds another layer |
| When to use | Default choice | When GBM overfits despite tuning |

---

## The Early Stopping Problem in DART

This is critical and almost never mentioned.

In standard GBM, adding tree N always reduces training loss. You can track val loss and stop when it plateaus.

In DART, adding tree N **might increase training loss** temporarily — because you're dropping trees that were helping. Val loss becomes noisy and non-monotonic.

```
Standard GBM val loss: 0.45 → 0.38 → 0.31 → 0.28 → 0.27 → 0.27 (stop)
DART val loss:         0.45 → 0.40 → 0.35 → 0.38 → 0.33 → 0.36 → 0.31
                                              ↑ went up  ↑ went up again
```

Early stopping triggers falsely. You stop too early.

> **Fix:** Don't use early stopping with DART. Set n_estimators manually based on cross-validation. Or use a very large patience window.

---

## Interview Answer

> "DART addresses over-specialization in GBM where early trees dominate the ensemble. By randomly dropping previous trees during each boosting round, later trees learn more robust representations instead of tiny residual corrections. The tradeoff is that early stopping breaks because val loss becomes non-monotonic — so I'd set n_estimators via cross-validation instead."

---

---

# PART 2: MONOTONIC CONSTRAINTS

---

## The Problem

GBM is a black box. It'll find any pattern in the data — including ones that are statistically real but **business-nonsensical.**

```
Feature: credit_score
Pattern GBM might learn:
  credit_score 300-500  → default probability 0.8
  credit_score 500-650  → default probability 0.6
  credit_score 650-700  → default probability 0.7  ← goes UP? nonsensical
  credit_score 700-850  → default probability 0.3
```

That spike at 650-700 might be a real artifact of training data (maybe high scorers in that range took bigger loans). But it violates business logic — higher credit score should always mean lower default risk.

Without constraints, GBM will learn whatever pattern minimizes loss. With **monotonic constraints**, you force the relationship to always go in one direction.

---

## What Monotonic Constraints Do

You tell GBM: *"feature X must have a monotonically increasing (or decreasing) relationship with the target — no exceptions."*

GBM then only considers splits that **maintain** that relationship. Any split that would create a non-monotonic pattern is rejected.

```python
# In LightGBM
model = lgb.LGBMClassifier(
    monotone_constraints=[1, -1, 0, 0]
    # +1 = must increase with target
    # -1 = must decrease with target
    #  0 = no constraint
)
# Feature order: [credit_score, debt_ratio, age, city]
# credit_score ↑ → default prob ↓ → constraint = -1
# debt_ratio ↑ → default prob ↑ → constraint = +1
```

---

## How It Works Internally

At each split, GBM evaluates candidate thresholds. For a constrained feature, after the split is made, it checks:

*"Does the left leaf predict lower/higher than the right leaf in the direction I specified?"*

If not — **split rejected.** Try next threshold.

This happens recursively at every node. The entire tree path must respect the monotonic direction.

```
Feature: credit_score (constraint = -1, must decrease default prob)

Split: credit_score > 650
  Left (≤650):  predicted default = 0.75  ✅ higher than right
  Right (>650): predicted default = 0.40

Split: credit_score > 700 (within right node)
  Left (≤700):  predicted default = 0.45  ❌ HIGHER than parent right node
                                              violates monotonicity → REJECTED
  Left (≤700):  predicted default = 0.38  ✅ lower → accepted
```

---

## When To Use Monotonic Constraints

**Use when:**
- Business logic dictates a clear direction (credit score↑ → risk↓, price↑ → demand↓)
- Regulatory compliance requires explainable, logical predictions
- Training data has noise that creates spurious non-monotonic patterns
- You need to defend model decisions to stakeholders

**Don't use when:**
- Relationship is genuinely non-linear and non-monotonic (age vs income — peaks mid-career)
- You're not sure of the direction — a wrong constraint hurts more than no constraint
- Exploratory modeling — constraints hide patterns worth knowing about

---

## The Accuracy Tradeoff

Constraints reduce the model's flexibility. You're removing valid splits from consideration.

```
Without constraints: val AUC = 0.847
With constraints:    val AUC = 0.831  ← small accuracy cost
```

This is the price of interpretability and business logic. Usually worth it in regulated industries (finance, healthcare, insurance). Less worth it in pure accuracy competitions.

---

## Interview Answer

> "Monotonic constraints force GBM to respect known business logic in its predictions — for example, that higher credit scores must always produce lower default probabilities. Internally, GBM rejects any split that would violate the specified direction. The cost is a small accuracy reduction because you're limiting the hypothesis space. I'd use them in regulated industries or when model decisions need to be defended to non-technical stakeholders."

---

---

# PART 3: HANDLING CLASS IMBALANCE IN GBM

---

## Why Imbalance Breaks GBM

GBM minimizes a loss function summed over all samples. When 95% of samples are class 0, the model can get very low loss by **always predicting class 0.**

```
Dataset: 950 non-churn (0), 50 churn (1)

Predict all 0:
  Loss on 950 non-churn: 0 (all correct)
  Loss on 50 churn: high (all wrong)
  Average loss: still very low ← GBM is happy

But you've built a useless model
```

GBM's residuals will be dominated by the majority class. Minority class residuals are tiny in aggregate — trees barely try to fix them.

---

## Five Fixes — In Order of What To Try First

---

### Fix 1 — Change Your Metric (Always Do This)

Accuracy is meaningless with imbalance. Switch to:

**AUC-ROC** — ranks positive samples above negative. Imbalance doesn't affect it directly.

**Precision-Recall AUC** — better than ROC when positives are very rare. Focuses entirely on how well you find the minority class.

**F1 score** — harmonic mean of precision and recall. Forces you to care about both.

```
Always predicting 0:
  Accuracy = 95%  ← looks great, is garbage
  AUC-ROC  = 0.50 ← exposed as random
  F1       = 0.0  ← exposed as useless
```

---

### Fix 2 — scale_pos_weight / class_weight

Tell GBM to penalize mistakes on the minority class more heavily.

```python
# ratio of negative to positive samples
scale_pos_weight = 950 / 50 = 19

model = lgb.LGBMClassifier(scale_pos_weight=19)
```

Internally this multiplies the loss contribution of positive samples by 19. A missed churn now costs 19x a missed non-churn. GBM can no longer ignore the minority class.

**Why it works:** Pseudo-residuals for minority class samples are now 19x larger. Trees are forced to focus on them.

**When it's enough:** Most of the time. Try this first before anything else.

---

### Fix 3 — Oversampling (SMOTE)

Instead of reweighting, create **synthetic minority samples.**

SMOTE (Synthetic Minority Oversampling Technique):
1. Take a minority sample
2. Find its K nearest neighbors (also minority)
3. Create synthetic sample along the line between them

```
Real churn user A:    sessions=3, age=28
Real churn user B:    sessions=5, age=32
Synthetic user:       sessions=4, age=30  ← interpolated
```

Now your dataset has more minority samples → GBM sees them more often → fits them better.

**Critical rule:** Only oversample the **training set.** Never touch validation or test. Otherwise you're validating on synthetic data and your metrics are fake.

```python
from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)
# Then train GBM on X_train_res, y_train_res
# Validate on original X_val, y_val
```

---

### Fix 4 — Undersampling Majority Class

Instead of adding minority samples, remove majority samples.

```
Original: 950 non-churn, 50 churn
After undersampling: 100 non-churn, 50 churn  (2:1 ratio)
```

**Pros:** Faster training, simpler than SMOTE.
**Cons:** You're throwing away real data. Information loss. Only use when dataset is large enough that losing majority samples doesn't hurt.

---

### Fix 5 — Change the Decision Threshold

GBM outputs a probability. Default threshold = 0.5 (predict churn if p > 0.5).

With imbalance, even a good model might output p=0.3 for true churners — because it's calibrated to the 5% base rate.

Lower the threshold:

```
threshold = 0.5: precision=0.8, recall=0.3  ← missing most churners
threshold = 0.2: precision=0.5, recall=0.7  ← catching more churners
threshold = 0.1: precision=0.3, recall=0.9  ← catching almost all
```

Choose threshold based on your business cost:
- Missing a churner costs $100 (retention campaign lost)
- False alarm costs $5 (unnecessary email)
- → Lower threshold, accept more false alarms to catch more churners

---

## Which Fix When

```
Always:        Change metric to AUC/F1
First try:     scale_pos_weight — simple, effective, no data changes
Mild imbalance (80/20):  scale_pos_weight is enough
Severe imbalance (99/1): SMOTE + scale_pos_weight together
Large dataset: Undersampling majority (you can afford to lose data)
Post-modeling: Adjust decision threshold to match business costs
```

---

## The Interview Trap

Interviewer asks: *"You have 1% positive class. How do you handle it?"*

**Wrong answer:** "I'll use SMOTE."

**Right answer:**
> "First I'd make sure I'm using the right metric — AUC-PR or F1, not accuracy. Then I'd try scale_pos_weight first since it's the simplest fix with no data modification. If that's not enough I'd combine SMOTE on the training set only with scale_pos_weight. Finally I'd tune the decision threshold based on the business cost of false positives vs false negatives — in a churn model, missing a churner is usually far more expensive than a false alarm."

---

## The Mental Models

> **DART:** Standard GBM is a band where the lead guitarist (Tree 1) drowns everyone out. DART randomly mutes the lead guitarist during rehearsal so other musicians develop their own voice. The full band plays together at the concert.

> **Monotonic Constraints:** GBM is a student who'll learn any pattern to pass the exam, even wrong ones. Monotonic constraints are the teacher saying "I don't care how you get there, but higher credit score must always mean lower risk — no exceptions."

> **Class Imbalance:** GBM is optimizing for the average student in a class of 95 A-students and 5 F-students. Without intervention it ignores the F-students. scale_pos_weight tells it each F-student counts 19x. SMOTE adds more F-students to the class. Threshold tuning says "I'd rather flag a B-student as struggling than miss an F-student."
