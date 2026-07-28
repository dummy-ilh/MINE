# Chapter 5 — Mitigation: Pre-processing

Chapters 3–4 gave you the tools to *measure* unfairness. Starting here, Chapters 5–7 cover how to actually *fix* it — organized by where in the pipeline (from Chapter 1's diagram) the fix is applied. Pre-processing means: change the data before training even begins, leaving the model and training procedure untouched.

**Why start here:** pre-processing techniques are the cheapest to apply (no change to model architecture, no change to the loss function, no special inference-time logic) and the easiest to explain to a non-technical stakeholder ("we adjusted how much weight different examples get"). That simplicity is also their main limitation, covered in 5.5.

## 5.1 Re-weighting

**The idea:** instead of changing which examples are in the training set, change *how much each example counts* toward the loss function. Give more weight to combinations of (group, label) that are underrepresented relative to what a "fair" world would look like, so the model can't just ignore them to optimize average loss.

**How the weights are actually computed.** The standard approach (Kamiran & Calders, 2012) computes a weight for each (group, label) combination as:

**w(a, y) = [P(A=a) × P(Y=y)] / P(A=a, Y=y)**

In words: compare the probability you'd expect for this (group, label) combination *if group and label were independent* (the numerator) to the probability you *actually* observe in the data (the denominator). If a combination is rarer than it "should" be under independence, its weight is > 1 (up-weighted); if it's more common than expected, its weight is < 1 (down-weighted).

**Worked example.** Suppose a training set of 1,000 people:
- 600 are Group A, 400 are Group B (so P(A=A)=0.6, P(A=B)=0.4)
- Overall, 300 people are labeled positive, 700 negative (so P(Y=1)=0.3, P(Y=0)=0.7)
- But within Group A: 240 of 600 are positive (40% positive rate)
- Within Group B: only 60 of 400 are positive (15% positive rate)

Let's compute the weight for the (Group B, Y=1) combination — the most underrepresented cell:
- P(A=B) × P(Y=1) = 0.4 × 0.3 = 0.12 — "expected" joint probability if group and label were independent
- P(A=B, Y=1) = 60/1000 = 0.06 — actual observed joint probability

w(B, 1) = 0.12 / 0.06 = **2.0**

So each positive-labeled Group B example gets counted **twice as much** in the loss function as it currently is — directly counteracting the fact that Group B positives are half as common (relative to independence) as they "should" be.

Compare to (Group A, Y=1): P(A=A)×P(Y=1) = 0.6×0.3 = 0.18, actual P(A=A,Y=1) = 240/1000 = 0.24, so w(A,1) = 0.18/0.24 = 0.75 — Group A positives get slightly *down*-weighted, since they're already somewhat overrepresented relative to independence.

**Effect:** the model can no longer get away with under-serving Group B's positive cases just because they're rare in raw counts — each one now "counts" more during gradient updates.

## 5.2 Re-sampling

**The idea:** instead of changing example *weights* (which requires a training loop that supports weighted loss), physically change which examples appear in the training set — oversample underrepresented (group, label) combinations, or undersample overrepresented ones, until the resulting dataset is closer to the target distribution.

- **Oversampling:** duplicate (or synthetically generate, e.g., via SMOTE-style interpolation) examples from underrepresented combinations.
- **Undersampling:** drop examples from overrepresented combinations.

**Tradeoff between the two:** oversampling risks overfitting to the (duplicated) minority examples, since the model may see the exact same points many times. Undersampling risks throwing away useful information from the majority combinations, which can hurt overall accuracy more than necessary. In practice, a moderate combination of both is common, and this is mathematically closely related to re-weighting — uniform oversampling by a factor of w is roughly equivalent to re-weighting by w, just implemented by duplicating rows instead of adjusting the loss function directly.

## 5.3 Disparate impact removal / feature transformation

**The idea:** even after dropping the protected attribute itself as a feature (Chapter 1, §1.2 — the proxy problem), other features can still be highly correlated with it. Disparate impact removal transforms the *remaining* features to reduce their correlation with the protected attribute, while trying to preserve as much of their usefulness for the actual prediction task as possible.

**One common approach (Feldman et al., 2015):** for each feature, instead of using each group's raw values, map every individual's value to their **percentile rank within their own group**, then convert that percentile back to a value using the *overall* (combined-group) distribution. Intuitively: "this person scored at the 80th percentile within their own group" gets converted to "here's what an 80th-percentile score looks like across everyone" — which removes the systematic group-level shift in the feature while preserving each person's *relative* standing within their group.

**Why this matters even after "dropping" the sensitive feature:** this technique is the formal fix for the "zip code as a race proxy" problem from Chapter 1 — you're not just deleting a column, you're actively reducing how much *any* remaining column can be used to reconstruct the group information.

## 5.4 Data augmentation for underrepresented groups

**The idea:** rather than reweighting/resampling existing examples, actively collect or generate *new* examples for underrepresented groups — e.g., targeted data collection efforts, synthetic data generation, or partnerships to source more diverse examples (this is the intervention that addresses the Gender Shades problem from Chapter 1 at its root: broaden the benchmark/training data itself, rather than just reweighting the skewed data you already have).

## 5.5 Tradeoffs: why pre-processing alone often isn't enough

Pre-processing is cheap and simple, but it has a structural limitation: **you're only controlling the input distribution, not what the model does with it.** A sufficiently flexible model can still learn a biased decision boundary from re-weighted data if there are other correlated signals it can latch onto, or if the re-weighting doesn't perfectly capture the real-world relationship you're trying to correct. Pre-processing shifts the odds in your favor, but doesn't *guarantee* any particular fairness metric on the trained model's actual predictions — you still need to measure the resulting model (Chapter 4) to confirm it worked, and if it didn't fully work, that's when in-processing (Chapter 6) or post-processing (Chapter 7) techniques come in, which act directly on the model's training objective or its final decisions rather than only on the data it started from.

---

**Next: Chapter 6 — Mitigation: In-processing**, where fairness constraints are built directly into the training objective — including adversarial debiasing, which ties back to the constrained-optimization ideas from your Lagrange multipliers chapter.
