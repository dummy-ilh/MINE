

# Module 3 — The Model Itself

## 1. WHY

We now have all the pieces: a linear combination gives us log-odds, and sigmoid converts log-odds back into probability. But we haven't actually put a **real dataset with real features** through this machine yet. This module answers: *"What is logistic regression actually computing, feature by feature, and what does a coefficient MEAN?"*

This is the module that turns "I understand the pieces" into "I can explain a model to a stakeholder" — and coefficient interpretation is the single most-tested conceptual checkpoint at L5.

## 2. INTUITION

Think of it as a **two-stage assembly line**:

**Stage 1 (the "vote counting" stage):** Every feature casts a weighted vote — some features push toward "yes" (churn), some push toward "no" (stay). We add up all these votes into one combined score. This combined score is unbounded — it can be any number, positive or negative, large or small. This is our `z`, the log-odds.

**Stage 2 (the "translate to probability" stage):** Take that combined vote score and feed it through sigmoid, which squashes it into a clean 0–1 probability.

That's it. That's the entire model. Everything else (regularization, evaluation metrics, etc.) is refinement around this two-stage core.

## 3. SIMPLE FORMULA

**Stage 1 — the linear part (in words):**
> Start with a baseline value. Then, for each feature, multiply that feature's value by its own importance weight. Add all of these up, plus the baseline.

**In simple notation, with 2 features:**

```
z = b + (w1 × x1) + (w2 × x2)
```

- `z` = the combined score (log-odds) — this is the SAME `z` from Module 2
- `b` = baseline/intercept — the log-odds when all features are 0
- `x1`, `x2` = the actual feature values for one customer (e.g., number of complaints, months as customer)
- `w1`, `w2` = the weights (also called coefficients) — how much each feature matters, and in which direction

**Stage 2 — squash it into probability (already known from Module 2):**

```
p = sigmoid(z) = 1 / (1 + e^(-z))
```

## 4. WORKED NUMERIC EXAMPLE — Full Pipeline, 2 Features

Let's build a tiny churn model with two features:
- `x1` = number of complaints filed
- `x2` = months as a customer (tenure)

Suppose after training, the model learned these values:

```
b  = -1.0   (baseline)
w1 = +0.8   (complaints push toward churn)
w2 = -0.05  (tenure pushes AWAY from churn — loyal customers churn less)
```

**Customer A:** 3 complaints, 6 months tenure.

**Step 1 — compute z:**
```
z = -1.0 + (0.8 × 3) + (-0.05 × 6)
z = -1.0 + 2.4 + (-0.3)
z = 1.1
```

**Step 2 — squash through sigmoid:**
```
p = 1 / (1 + e^(-1.1))
p = 1 / (1 + 0.3329)
p = 1 / 1.3329
p = 0.7502
```

**Customer A has a ~75% predicted probability of churning.**

Let's also check **Customer B:** 0 complaints, 24 months tenure.

```
z = -1.0 + (0.8 × 0) + (-0.05 × 24)
z = -1.0 + 0 - 1.2
z = -2.2

p = 1 / (1 + e^(2.2))
p = 1 / (1 + 9.025)
p = 1 / 10.025
p = 0.0998
```

**Customer B has a ~10% predicted probability of churning.** Makes total sense — no complaints, long tenure, low churn risk.

## 5. COEFFICIENT INTERPRETATION — the part that "trips up most candidates"

This is the crux of Module 3. Let's slow way down here.

**The wrong instinct (what trips people up):** People want to say "w1 = 0.8 means a 1-unit increase in complaints increases the probability of churn by 0.8" — **this is WRONG.** Coefficients do NOT operate on probability directly. They operate on **log-odds** — remember, that's literally what `z` is.

**The correct interpretation, step by step:**

`w1 = 0.8` means: **"A 1-unit increase in complaints increases the LOG-ODDS of churning by 0.8, holding all other features fixed."**

That's still a bit abstract, so let's convert it into something more intuitive — **odds** (not log-odds, not probability):

**WHY convert to odds:** Log-odds units aren't intuitive to a human ("increases log-odds by 0.8" — okay, so what?). But if we undo the "log" part (exponentiate), we get a number expressed as a **multiplier on the odds**, which IS intuitive ("doubles the odds," "triples the odds").

**Formula (in words):**
> Take e (Euler's number) and raise it to the power of the coefficient. That gives you the "odds ratio" — how much the odds get multiplied by, for each 1-unit increase in that feature.

**In notation:**
```
odds ratio = e^w
```

**Worked example:**
```
odds ratio for complaints = e^0.8 = 2.2255
```

**Plain English: "Each additional complaint MULTIPLIES a customer's odds of churning by about 2.23x, holding tenure constant."** That's a sentence you could say out loud to a business stakeholder and they'd understand it immediately — this is why odds ratios are the standard way logistic regression coefficients get communicated in practice.

Let's also do tenure's coefficient:
```
odds ratio for tenure = e^(-0.05) = 0.9512
```

**Plain English: "Each additional month of tenure MULTIPLIES the odds of churning by about 0.95x"** — i.e., **reduces** the odds by about 4.9% per month, since 0.9512 is less than 1. A coefficient below 1 after exponentiating always means "this odds ratio makes the event LESS likely."

**Quick reference rule:**
| Odds ratio value | Meaning |
|---|---|
| > 1 | Feature increases the odds of the event |
| = 1 | Feature has no effect on the odds |
| < 1 | Feature decreases the odds of the event |

## 6. INTERPRETATION (business framing)

Putting it together for a real conversation with a product manager: *"For every extra complaint a customer files, they become about 2.2 times more likely to churn than to stay, all else being equal. Meanwhile, every extra month of tenure reduces their churn odds by about 5%. This tells the retention team that complaint volume is a much stronger churn signal than raw tenure, and interventions should prioritize customers with recent complaint spikes."* That's a coefficient interpretation stated the way an L5 candidate should be able to say it, live, in an interview.

## 7. FAANG L5 ANGLE

**Common interview question:** *"You have a logistic regression coefficient of 0.4 for a feature. What does that mean?"*
Strong answer: two-part. (1) "It means a 1-unit increase in that feature increases the log-odds of the outcome by 0.4, holding other features constant." (2) "More intuitively, exponentiating gives e^0.4 ≈ 1.49, meaning the odds of the outcome increase by about 49% per unit increase in that feature."

**Common trap #1:** Saying the coefficient directly represents a change in probability. This is **always wrong** — probability effects are non-linear (they depend on where you are on the S-curve), while log-odds effects are constant everywhere. If asked "how much does probability change," the honest answer is: "it depends on the current baseline probability — the effect on probability isn't constant, only the effect on log-odds is." Candidates who don't realize the effect is non-linear on the probability scale get flagged immediately.

**Common trap #2:** Forgetting "holding other features constant" — coefficients in a multi-feature model are conditional on other features being fixed; interviewers will ask what happens if you drop this caveat (answer: without it, the interpretation is incomplete/misleading, especially if features are correlated — foreshadowing Module 10's multicollinearity discussion).

**Common follow-up:** *"Why do we exponentiate the coefficient instead of interpreting it raw?"* Because raw coefficients are in log-odds units (unintuitive), and exponentiating converts them into a multiplicative effect on odds — a scale humans can reason about ("doubles," "cuts in half," etc.).

## 8. QUICK PYTHON CHECK

```python
import numpy as np

def predict_proba(x1, x2, b, w1, w2):
    z = b + w1 * x1 + w2 * x2
    p = 1 / (1 + np.exp(-z))
    return z, p

b, w1, w2 = -1.0, 0.8, -0.05

for name, x1, x2 in [("Customer A", 3, 6), ("Customer B", 0, 24)]:
    z, p = predict_proba(x1, x2, b, w1, w2)
    print(f"{name}: z={z:.3f}, p={p:.4f}")

print("Odds ratio (complaints):", np.exp(w1))
print("Odds ratio (tenure):", np.exp(w2))
```

## 9. CHECK — before Module 4

1. A model has a coefficient of `w = 1.2` for "number of prior purchases." Compute the odds ratio, and state in one plain-English sentence what it means for a customer's odds of the target outcome.
2. True or false, and explain why: "If w = 0.8, then a 1-unit increase in that feature increases the probability of the outcome by 0.8 (i.e., 80 percentage points)."

Close, but there's a mix-up in the wording. Let's be precise:
w = 1.2 is measured in log-odds units, and it's additive, not multiplicative. So the correct statement is:

"A 1-unit increase in this feature increases the log-odds by 1.2 (added on), holding other features constant."

The "1.2x" multiplicative language only applies after you exponentiate — that's the odds ratio, and it lives on a different scale (odds, not log-odds):
odds ratio = e^1.2 = 3.32
Correct plain-English sentence: "Each 1-unit increase in this feature multiplies the customer's ODDS of the outcome by about 3.32x, holding other features constant."
The mix-up to watch for (this is exactly the trap interviewers probe): "1.2" by itself is a log-odds shift (additive). "3.32x" is an odds multiplier (multiplicative). Never say "increases log-odds by 1.2x" — log-odds moves by addition, odds moves by multiplication. Keep those two verbs (add vs multiply) tied to their correct scale (log-odds vs odds) and you'll never mix this up again.

Confirming Check Question 2 — and showing you the proof
You said "No, it's log-odds" — correct. The statement was false because coefficients act on log-odds (constant effect), not probability (non-constant effect).
Here's the "still need to see" proof — same coefficient, two different starting points, watch the probability move by a different amount each time:
pythonimport numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = 0.8  # same coefficient both times
---

This is one of the most important concepts in logistic regression.

The connection isn't something we're *assuming*—it's how **logistic regression is defined**.

### Step 1: Start with the Odds

If the probability of the positive class is `p`, then the odds are

```text
Odds = p / (1 - p)
```

Odds answer:

> "How many times more likely is success than failure?"

For example,

```text
p = 0.8

Odds = 0.8 / 0.2 = 4
```

Meaning:

> Success is **4 times as likely** as failure.

---

### Step 2: Take the Log

Now take the natural logarithm of the odds:

```text
z = log(Odds)
```

or

```text
z = ln(p / (1 - p))
```

This quantity is called the **log-odds** or **logit**.

**This is where `z` comes from.**

It is **defined** to be the log-odds.

---

### Step 3: Why Introduce `z`?

The problem is that probabilities are constrained:

```text
0 ≤ p ≤ 1
```

A linear model, however, can output any real number:

```text
-∞ to +∞
```

We need a bridge between these two worlds.

```text
Linear Model
    │
    ▼
z = w₁x₁ + w₂x₂ + ... + b
    │
    │  (Interpret z as log-odds)
    ▼
log(p / (1-p)) = z
    │
    ▼
Solve for p
    │
    ▼
p = 1 / (1 + e^-z)
```

This is the entire idea behind logistic regression.

---

# The Connection Formula

The relationship is

```text
z = ln(p / (1 - p))
```

This is called the **logit transformation**.

The inverse is

```text
p = 1 / (1 + e^-z)
```

This is the **sigmoid function**.

So they are inverse functions of each other.

```text
Probability (p)
       │
       ▼
Logit

z = ln(p / (1-p))

       ▲
       │
Inverse (Sigmoid)

p = 1 / (1 + e^-z)
```

---

# Why Did We Choose Log-Odds?

Suppose we want a linear model:

```text
z = wx + b
```

Since `z` can be any real number (`-∞` to `+∞`), we need a transformation of probability that also spans all real numbers.

Let's see:

| Probability `p` | Log-Odds `z` |
| --------------: | -----------: |
|               0 |           -∞ |
|            0.01 |        -4.60 |
|            0.10 |        -2.20 |
|            0.50 |            0 |
|            0.90 |         2.20 |
|            0.99 |         4.60 |
|               1 |           +∞ |

Notice what happened:

```text
Probability

0 ---------------------- 1
```

became

```text
Log-Odds

-∞ ---------------------- +∞
```

The logit transformation converts a bounded probability into an unbounded value that a linear model can predict naturally.

---

# The Complete Pipeline

```text
Input Features (x)
        │
        ▼
Linear Model

z = wx + b

        │
        ▼
Interpret z as Log-Odds

z = ln(p / (1-p))

        │
        ▼
Invert the Log-Odds

p = 1 / (1 + e^-z)

        │
        ▼
Predicted Probability
```

## The key insight

We're **not discovering** that `z` equals the log-odds. We **define** it that way because it gives us exactly the transformation we need:

* A linear model predicts any real number (`z ∈ (-∞, +∞)`).
* Probabilities must lie in `[0, 1]`.
* The **logit** maps probabilities to all real numbers:

  ```text
  z = ln(p / (1 - p))
  ```
* The **sigmoid** maps those real numbers back to probabilities:

  ```text
  p = 1 / (1 + e^-z)
  ```

This pair of inverse functions is what makes logistic regression possible. Once you define the linear model's output `z` as the **log-odds**, the sigmoid function follows naturally by solving the equation for `p`.
