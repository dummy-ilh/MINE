
# Module 8 — Evaluation Metrics Deep Dive

## 1. WHY

Once your model is trained, you need a way to answer: **"is this model actually good?"** The obvious first instinct is to check **accuracy** — the percentage of predictions the model got right. Let's see why this instinct fails you far more often than people expect.

**What breaks with accuracy alone:** Imagine a fraud detection model where only 1 in 1,000 transactions is actually fraudulent. A lazy model that predicts **"not fraud" for every single transaction** would be **99.9% accurate** — and completely useless, since it catches zero actual fraud. This is the classic **imbalanced data trap**, and it's one of the most common reasons a seemingly "great" model (by accuracy) turns out to be worthless in production. We need metrics that account for this.

## 2. INTUITION — The Confusion Matrix

Before any metric, we need a way to categorize every prediction into one of four buckets. Think of a security guard checking IDs at a bar:

- **True Positive (TP):** guard correctly stops an underage person — predicted "fraud/positive," and it WAS actually fraud/positive.
- **True Negative (TN):** guard correctly lets in an adult — predicted "not fraud/negative," and it WAS actually not fraud/negative.
- **False Positive (FP):** guard wrongly stops an adult (asks for ID, they're annoyed) — predicted "positive," but it was ACTUALLY negative. (Also called a "Type I error.")
- **False Negative (FN):** guard wrongly lets in an underage person — predicted "negative," but it was ACTUALLY positive. (Also called a "Type II error.")

**In a simple grid (the "confusion matrix"):**

|  | Actually Positive | Actually Negative |
|---|---|---|
| **Predicted Positive** | True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN) |

Every single prediction your model makes lands in exactly one of these 4 boxes. Every metric in this module is just a different way of combining these 4 numbers.

## 3. SIMPLE FORMULA — Precision, Recall, F1

### Accuracy (for context, and to show its flaw)

**In words:**
> Out of all predictions made, what fraction were correct (either correctly flagged positive, or correctly flagged negative)?

**In notation:**
```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

**WHY this metric exists:** Precision answers: **"Of everything I FLAGGED as positive, how much did I actually get right?"** This matters when **false positives are costly** — e.g., flagging legitimate transactions as fraud annoys customers, so you want to know how "trustworthy" your positive flags are.

**In words:**
> Out of all the cases I predicted as positive, what fraction were actually positive?

**In notation:**
```
precision = TP / (TP + FP)
```

- `TP` = true positives (correctly flagged)
- `FP` = false positives (wrongly flagged — the "false alarms")

### Recall (a.k.a. Sensitivity, True Positive Rate)

**WHY this metric exists:** Recall answers: **"Of everything that was ACTUALLY positive, how much did I successfully catch?"** This matters when **false negatives are costly** — e.g., missing actual fraud costs real money, so you want to know how much real fraud slips through undetected.

**In words:**
> Out of all the cases that were actually positive, what fraction did I correctly catch?

**In notation:**
```
recall = TP / (TP + FN)
```

- `TP` = true positives (correctly caught)
- `FN` = false negatives (missed — the ones that got away)

**The fundamental tradeoff (connects directly to Module 6):** Lowering your threshold catches more true positives (higher recall) but also flags more false positives (lower precision). Raising your threshold does the opposite. **You cannot maximize both simultaneously** — improving one typically costs you some of the other. This is why threshold selection (Module 6) and metric choice are deeply linked.

### F1 Score

**WHY this metric exists:** Precision and recall can pull in opposite directions, and reporting two numbers is often less convenient than reporting one. F1 combines them into a single number — but not with a simple average, for an important reason we'll see below.

**In words:**
> Take the "harmonic mean" of precision and recall — a special kind of average that punishes having one very low value, even if the other is high.

**In notation:**
```
F1 = 2 × (precision × recall) / (precision + recall)
```

**Why harmonic mean instead of a regular average?** A regular average would let one very high number "hide" one very low number. Example: precision = 1.0 (perfect), recall = 0.01 (terrible). A regular average gives `(1.0 + 0.01)/2 = 0.505` — looks mediocre-but-okay. But this model is nearly useless (it barely catches anything)! **Harmonic mean is designed to be pulled DOWN heavily by the smaller number** — it refuses to let a great precision mask a terrible recall (or vice versa), giving you a much more honest single-number summary.

## 4. WORKED NUMERIC EXAMPLE

Let's build a confusion matrix for a fraud model tested on 1,000 transactions, where 10 are actually fraud (a realistic imbalanced scenario).

Suppose the model's results are:
```
TP = 7   (caught 7 of the 10 real fraud cases)
FN = 3   (missed 3 real fraud cases)
FP = 20  (wrongly flagged 20 legitimate transactions as fraud)
TN = 970 (correctly identified 970 legitimate transactions)
```

**Check: does this add up to 1,000 total?** `7 + 3 + 20 + 970 = 1000`. ✓

**Compute accuracy:**
```
accuracy = (TP + TN) / (TP+TN+FP+FN)
accuracy = (7 + 970) / 1000
accuracy = 977 / 1000
accuracy = 0.977 → 97.7%
```

Looks great at first glance! But watch what happens with precision and recall:

**Compute precision:**
```
precision = TP / (TP + FP)
precision = 7 / (7 + 20)
precision = 7 / 27
precision = 0.259 → 25.9%
```

**In plain English: only about 1 in 4 of the transactions we flagged as fraud were ACTUALLY fraud.** That's a lot of false alarms hiding behind a 97.7% accuracy headline number.

**Compute recall:**
```
recall = TP / (TP + FN)
recall = 7 / (7 + 3)
recall = 7 / 10
recall = 0.70 → 70%
```

**In plain English: we caught 70% of actual fraud cases, missing 30%.**

**Compute F1:**
```
F1 = 2 × (0.259 × 0.70) / (0.259 + 0.70)
F1 = 2 × 0.1813 / 0.959
F1 = 0.3626 / 0.959
F1 = 0.378 → 37.8%
```

**The huge gap between accuracy (97.7%) and F1 (37.8%) is the entire lesson of this module** — accuracy was hiding a mediocre model behind class imbalance, while precision/recall/F1 reveal the real, much less flattering picture.

## 5. ROC-AUC vs PR-AUC

Both of these are **threshold-independent** metrics — instead of picking one threshold and computing one confusion matrix, they evaluate the model's ranking ability **across ALL possible thresholds at once**, then summarize that into a single score.

### ROC-AUC

**WHY:** ROC (Receiver Operating Characteristic) curve plots **True Positive Rate (recall)** against **False Positive Rate** at every possible threshold, then AUC ("Area Under the Curve") condenses this into one number between 0 and 1.

**Plain-English meaning of the AUC number:** *"If I randomly pick one actual positive case and one actual negative case, what's the probability my model correctly ranks the positive one higher (higher predicted probability) than the negative one?"*
- AUC = 1.0 → perfect ranking, always
- AUC = 0.5 → no better than random guessing (a coin flip)
- AUC = 0.0 → perfectly backwards (worse than useless — flip your predictions and you'd be perfect!)

**False Positive Rate (a new term needed for ROC):**
```
FPR = FP / (FP + TN)
```
In words: "Out of all the actually-negative cases, what fraction did we wrongly flag as positive?"

### PR-AUC (Precision-Recall AUC)

**WHY it exists as an alternative:** ROC-AUC has a known weakness — with **heavily imbalanced data** (like our fraud example, where negatives vastly outnumber positives), the False Positive Rate denominator `(FP + TN)` is dominated by the huge TN count, making FPR look artificially tiny and ROC-AUC look artificially great, even when precision is actually terrible (like our 25.9% precision above). **PR-AUC doesn't use TN at all** — it plots precision against recall across thresholds, which stays sensitive to how the model handles the rare positive class specifically.

**The practical rule of thumb:**
| Situation | Preferred metric |
|---|---|
| Roughly balanced classes | ROC-AUC is fine |
| Heavily imbalanced classes (rare positive class, like fraud/disease) | PR-AUC (more honest signal) |

## 6. CALIBRATION

**WHY this matters separately from accuracy/precision/recall:** A model can correctly RANK customers (high-risk customers get higher scores than low-risk ones) while still being **badly calibrated** — meaning its actual probability NUMBERS don't mean what they claim.

**Definition in plain English:** A model is "well calibrated" if, **among all the times it predicts 70% probability of an event, the event actually happens about 70% of the time** — not 40%, not 95%, but genuinely close to 70%.

**Why this matters in production:** If your churn model says "this customer has an 80% chance of churning" and your business logic spends real marketing dollars proportional to that percentage (e.g., "offer bigger discounts to higher-probability churners"), a miscalibrated model will misallocate that budget — even if it correctly RANKS customers from most-to-least likely to churn. Ranking quality (what AUC measures) and calibration quality are **two different properties**, and a model can be good at one while being poor at the other.

**How to check calibration in practice (conceptually):** group predictions into buckets (e.g., all predictions between 70-80%), then check what fraction of THOSE customers actually experienced the event. A "calibration curve" (or "reliability diagram") plots predicted probability (x-axis) against actual observed frequency (y-axis) — a perfectly calibrated model produces a straight 45-degree line.

## 7. INTERPRETATION

In real terms: if you only report accuracy to a stakeholder for an imbalanced problem, you risk **massively overselling** a mediocre model (our 97.7%-accurate-but-25.9%-precision fraud example). The right conversation is: "here's our precision/recall tradeoff at different thresholds, here's the PR-AUC since fraud is rare, and here's our calibration check since the fraud team is using our raw probability to prioritize investigation effort." This is a genuinely more sophisticated, trustworthy way to communicate model quality than a single accuracy number.

## 8. FAANG L5 ANGLE

**Common interview question:** *"Why is accuracy a bad metric for imbalanced classification problems?"*
Strong answer: state the "predict majority class always" trap directly, ideally with a quick example like ours (99%+ accuracy while catching 0% of the rare class), then pivot to precision/recall/F1 or PR-AUC as better alternatives.

**Common follow-up:** *"When would you use ROC-AUC vs PR-AUC?"*
Good answer: ROC-AUC for roughly balanced classes; PR-AUC for heavily imbalanced classes, because ROC-AUC's False Positive Rate can look deceptively good when true negatives vastly outnumber everything else.

**Common follow-up:** *"What does an AUC of 0.5 mean, and what would an AUC of 0.3 mean?"*
Sharp answer: 0.5 = random guessing, no ranking skill at all. 0.3 is actually informative, just backwards — the model's ranking is worse than random in a *systematic* way, meaning if you simply flipped its predictions (predict low where it says high), you'd get an AUC of 0.7. An AUC of exactly 0.5 is the only "truly useless" value; anything consistently far from 0.5 (even below it) contains exploitable signal.

**Common follow-up:** *"Can a model have high AUC but be poorly calibrated? Give an example."*
Good answer: yes — AUC only measures ranking (relative ordering), not the actual probability values. A model could correctly rank customer A above customer B in risk, while both its outputted probabilities are systematically too high or too low (e.g., always outputting 90%+ for anyone even slightly risky) — great ranking, terrible calibration.

**Common trap:** Candidates say "higher accuracy is always better" without qualifying it based on class balance — a red flag question at L5, since it's foundational to knowing when a chosen metric is actually appropriate for the problem.

**Another trap:** Confusing precision and recall directionally (mixing up which one has FP vs FN in the denominator) under interview pressure — a good trick to remember it: **"Precision cares about your Positive predictions being right" (denominator = all predicted positives = TP+FP). "Recall cares about Recalling/catching everything that's actually positive" (denominator = all actual positives = TP+FN).**

## 9. QUICK PYTHON CHECK

```python
import numpy as np

TP, FN, FP, TN = 7, 3, 20, 970

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")
```

## 10. CHECK — before Module 9

1. A hospital builds a model to flag patients for a follow-up cancer screening. Would they likely prioritize precision or recall, and why? What's the real-world cost of getting this priority backwards?
2. A colleague says "our fraud model has 99.5% accuracy, so it must be great." Using what you learned this module, what's the FIRST question you'd ask them before believing that claim?

Check 1 — confirmed
Correct: the hospital prioritizes recall. A false negative here means an actual cancer patient gets missed and doesn't receive follow-up screening — potentially fatal, since early detection is often the difference between treatable and terminal. A false positive just means a healthy patient gets an extra (uncomfortable but low-risk) screening — an inconvenience, not a tragedy. Getting this backwards (optimizing for precision instead) would mean the model is "confident" in its positive flags but silently lets real cancer cases slip through undetected — the worst possible failure mode for this use case.
Check 2 — confirmed, sharpened
Right instinct — the first question should be about class balance: "What fraction of transactions in your test set are actually fraud?" If fraud is rare (say, 1%), a 99.5% accuracy number is nearly meaningless on its own — a model that predicts "not fraud" for everything would already score ~99% accuracy. The natural follow-up questions after that: "What's the precision and recall? What's the PR-AUC?" — that's where the real picture lives.
