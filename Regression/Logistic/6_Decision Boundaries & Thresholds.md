

# Module 6 — Decision Boundaries & Thresholds

## 1. WHY

Everything so far gives us a **probability** — "this customer has a 73% chance of churning." But at some point, a business system usually needs a **hard decision**: do we send this customer a retention offer, or not? Do we flag this transaction as fraud, or not? Probability alone doesn't make decisions — **something has to convert probability into action.**

**What breaks if we don't think carefully about this:** Most people's gut instinct is "if probability > 50%, say yes; otherwise, say no." This default threshold of 0.5 feels obvious, but it's actually an **arbitrary choice** that quietly assumes something very specific: that a false positive (wrongly saying "yes") and a false negative (wrongly saying "no") cost the business **exactly the same amount.** That's almost never true in the real world, and blindly using 0.5 without checking this assumption is one of the most common silent mistakes in applied ML.

## 2. INTUITION

Think of the model's probability output as a **dial that goes from 0% to 100%.** The **threshold** is a line you draw on that dial — everything above the line gets labeled "yes," everything below gets labeled "no." 0.5 is just the halfway mark on the dial — there's nothing mathematically special about it beyond being the midpoint.

**Analogy:** imagine a airport security screening. If you set the "flag as suspicious" threshold very low (catch anyone with even a small chance of being a risk), you'll catch almost every real threat — but you'll also flag tons of innocent travelers (many false positives), causing massive delays. If you set the threshold very high (only flag near-certain threats), you'll have very few innocent people flagged — but you risk letting real threats slip through (false negatives). **There's no single "correct" threshold in a vacuum — the right choice depends entirely on which type of mistake is more costly to you.**

## 3. SIMPLE FORMULA

**In words — the basic decision rule:**
> If the model's predicted probability is greater than or equal to some chosen cutoff value, classify it as "yes" (positive class). Otherwise, classify it as "no" (negative class).

**In simple notation:**

```
predicted_class = 1  if  p >= threshold
predicted_class = 0  if  p <  threshold
```

- `p` = the model's predicted probability (output of sigmoid)
- `threshold` = the cutoff value we choose (default assumption is often 0.5, but this is a choice, not a law of nature)
- `predicted_class` = the final yes/no decision the business actually acts on

## 4. WORKED NUMERIC EXAMPLE — Same Model, Different Thresholds, Different Decisions

Let's take 5 customers with predicted churn probabilities from our model, and see how the decision changes depending on where we set the threshold.

| Customer | Predicted p | Decision @ threshold=0.5 | Decision @ threshold=0.3 | Decision @ threshold=0.7 |
|---|---|---|---|---|
| A | 0.20 | No | No | No |
| B | 0.40 | No | **Yes** | No |
| C | 0.55 | **Yes** | Yes | No |
| D | 0.65 | **Yes** | Yes | No |
| E | 0.85 | **Yes** | Yes | **Yes** |

**Look at Customer B (p=0.40):** at threshold=0.5, they're labeled "will NOT churn" — no retention offer sent. But at threshold=0.3, they flip to "WILL churn" — now they get an offer. **Same model, same probability, completely different business action** — purely because of where we drew the line. This is the entire point: the model itself doesn't change; only our decision rule does.

**Look at Customer C (p=0.55):** at threshold=0.5 they're flagged as churn risk, but at threshold=0.7 they're not. Notice how much the final decision set changes just by moving the threshold from 0.5 to 0.7 — we go from 3 "yes" decisions down to just 1.

## 5. LINEAR DECISION BOUNDARY — What It Looks Like Geometrically

Here's a deeper geometric fact worth knowing: **the boundary where `p = threshold` corresponds to a specific, fixed value of z (log-odds)** — and since `z = b + w1*x1 + w2*x2` is a straight-line equation, **the decision boundary itself is literally a straight line** (or, with more features, a flat plane/hyperplane) cutting through your feature space.

**Why does this matter?** Because it tells you something important about logistic regression's limitations: **it can only separate classes cleanly if a straight line (or flat plane) can do the job.** If your two classes are arranged in a pattern that needs a curved or wiggly boundary to separate well (imagine one class forming a ring around the other), plain logistic regression will struggle — you'd need to manually engineer curved features (like x², interaction terms) or switch to a model that naturally learns curves (like gradient boosting or neural networks).

**Quick derivation of why the boundary is where z=0 (at the default threshold=0.5):** Recall from Module 2, `sigmoid(0) = 0.5`. So if your threshold is 0.5, the decision boundary is exactly where `z = 0` — i.e., where `b + w1*x1 + w2*x2 = 0`. That equation, plotted on a 2D graph of x1 vs x2, is literally the equation of a straight line.

## 6. BUSINESS-DRIVEN THRESHOLD SELECTION

This is the practically important part for L5 interviews: **choosing the right threshold isn't a math problem, it's a business-cost problem.**

**The core question to always ask:** *"What does it cost us if we get this wrong in each direction?"*

**Example 1 — Fraud detection:**
- **False positive** (flagging a legitimate transaction as fraud): annoys a customer, maybe costs a small amount in customer service friction.
- **False negative** (missing actual fraud): can cost thousands of dollars directly, plus reputational damage.
- Since missing fraud is far more costly than annoying a customer, you'd set a **LOWER threshold** — catch more potential fraud, even at the cost of more false alarms.

**Example 2 — Spam email filtering:**
- **False positive** (flagging a real, important email as spam — e.g., a job offer buried in the spam folder): can be very costly to the user (missed opportunity).
- **False negative** (letting one spam email through to the inbox): mildly annoying, but low cost.
- Since a false positive here is worse than a false negative, you'd set a **HIGHER threshold** — only flag emails as spam when very confident, tolerating a few spam emails slipping through.

**The general rule:**
| If... | Then set the threshold... |
|---|---|
| False negatives are more costly | LOWER the threshold (catch more positives, accept more false alarms) |
| False positives are more costly | RAISE the threshold (be more conservative before saying "yes") |

## 7. INTERPRETATION

In real terms: two teams could take the exact same trained model, with the exact same predicted probabilities, and make completely different production decisions just by choosing different thresholds — one team optimizing to catch every possible fraud case (accepting more customer friction), another optimizing for a smooth customer experience (accepting a bit more fraud risk). **The model's job is to produce a well-calibrated probability. Choosing the threshold is a separate, business-driven decision that should involve stakeholders who understand the real-world cost of each type of mistake** — not something a data scientist should quietly hardcode to 0.5 without discussion.

## 8. FAANG L5 ANGLE

**Common interview question:** *"Why isn't 0.5 always the right threshold for a binary classifier?"*
Strong answer: 0.5 implicitly assumes false positives and false negatives have equal cost, which is rarely true. The right threshold should be chosen based on the actual business cost of each error type, often by analyzing the precision-recall tradeoff at different thresholds (bridges directly into Module 8).

**Common follow-up:** *"How would you actually go about choosing the threshold in practice?"*
Good answer: plot metrics (precision, recall, F1, or a cost curve incorporating actual dollar costs of FP vs FN) across a range of thresholds, then pick the threshold that optimizes for the business objective — not necessarily the one that maximizes accuracy.

**Common follow-up:** *"What does it mean that logistic regression has a 'linear decision boundary,' and when is that a limitation?"*
Good answer: the boundary between predicted classes is a straight line/hyperplane in feature space, because it's determined by `z = 0`, and z is a linear combination of features. This becomes a limitation when the true relationship between classes needs a curved boundary — in those cases, you'd add engineered nonlinear features (polynomial/interaction terms) or use a model that learns nonlinearity natively.

**Common trap:** Candidates say "we should just use whatever threshold gives the highest accuracy." This ignores class imbalance and unequal error costs — a classic case (foreshadowing Module 8) is a rare-disease detector where 99% accuracy is trivially achieved by predicting "no disease" for everyone, which is a useless model despite the high accuracy number.

**Another trap:** Assuming the decision boundary concept doesn't matter if you're "just predicting probabilities" — even if the immediate output is a probability, nearly every production system eventually needs SOME threshold to trigger an action (send email, block transaction, etc.), so this decision is rarely avoidable.

## 9. QUICK PYTHON CHECK

```python
import numpy as np

probabilities = np.array([0.20, 0.40, 0.55, 0.65, 0.85])

for threshold in [0.3, 0.5, 0.7]:
    decisions = (probabilities >= threshold).astype(int)
    print(f"Threshold={threshold}: decisions={decisions}")
```

## 10. CHECK — before Module 7

1. You're building a model to screen resumes for interview callbacks, where a "false negative" means a great candidate gets rejected, and a "false positive" means a mediocre candidate gets an interview slot (a relatively low-cost mistake). Would you raise or lower the classification threshold from 0.5, and why?
2. Why is it geometrically true that logistic regression's decision boundary is always a straight line (or flat plane), rather than a curve — connect this back to the formula for z?
---
Good work on Check 1 — that's exactly right, and let's lock in the reasoning: since a false negative (rejecting a great candidate) is costly and a false positive (mediocre candidate gets an interview) is cheap, you'd lower the threshold. This catches more candidates as "call back" (fewer real false negatives), accepting the cost of interviewing a few more mediocre candidates. Correct.

---
Check 2 needs a bit more — let's nail it down precisely, since this is a common trap question.
Completing Check 2 — Why the boundary is a straight line
The key connection is this: z is always a linear combination of your features — no matter what the actual weights turn out to be, the formula z = b + w1*x1 + w2*x2 is structurally just "add up features multiplied by constants." There are no squared terms, no products of features, nothing curved baked into it — it's degree-1 math (like y = mx + c, just with more variables).
The decision boundary is defined by the equation z = 0 (at the default threshold). Since z itself is a straight-line equation, setting it equal to a constant (0) gives you... another straight-line equation. Plotted on a graph of x1 vs x2, b + w1*x1 + w2*x2 = 0 traces out a straight line — the same way 2x + 3y = 6 is a straight line on a basic algebra graph, not a curve.
The only way to get a curved boundary is to feed logistic regression features that are already curved — like manually adding x1² or x1×x2 as new input features. Then z becomes a linear combination of THOSE terms, and while z=0 is still "linear" in the new features, it traces a curve back in the original x1, x2 space. This is a real technique (polynomial feature engineering) and a common interview follow-up.



# Decision Boundary in Logistic Regression — Made Simple

This is a **critical concept** that trips up many people. Let's break it down visually and intuitively.

---

## 🎯 **The Core Idea — In One Sentence**

> The **decision boundary** is the line (or curve) that separates **"predict class 0"** from **"predict class 1"** in the feature space.

---

## 🔑 **The Threshold Connection — Your Starting Point**

You said you understand **threshold** — that's perfect because the decision boundary and threshold are **two sides of the same coin**:

| Concept | What it is | Example |
|---------|-----------|---------|
| **Threshold** | A cutoff on the **probability** output | "Predict churn if p ≥ 0.5" |
| **Decision Boundary** | The set of **feature values** where p = threshold | "The line where predicted probability = 0.5" |

**Think of it this way:**
- **Threshold** = "How sure do I need to be?" (a number between 0 and 1)
- **Decision boundary** = "Where in feature-space does my prediction flip from 0 to 1?"

---

## 📐 **Step-by-Step Derivation (The Math)**

Let's find exactly where the decision boundary lives.

### Step 1: The logistic regression equation

```
z = b + w₁x₁ + w₂x₂ + ... + wₙxₙ
p = sigmoid(z) = 1 / (1 + e^(-z))
```

### Step 2: The decision rule with threshold = 0.5

We predict class 1 when:
```
p ≥ 0.5
```

### Step 3: What input z gives p = 0.5?

Let's solve:
```
sigmoid(z) = 0.5
1 / (1 + e^(-z)) = 0.5
1 + e^(-z) = 2
e^(-z) = 1
-z = 0
z = 0
```

**Key insight:** `p = 0.5` happens **exactly when z = 0**.

### Step 4: So the decision boundary is where z = 0

```
b + w₁x₁ + w₂x₂ + ... + wₙxₙ = 0
```

This is the **equation of a hyperplane** (a line in 2D, a plane in 3D, etc.)

---

## 👁️ **Visual Examples — Build Intuition**

### Example 1: 1 Feature (1D)

**Data:** Churn based on complaints (x)

```
z = b + w·x
Decision boundary: b + w·x = 0 → x = -b/w
```

**Visual:**
```
p(x) →
1.0 |                    ████████████████
    |                 ████
0.5 |──────────────●───────────────
    |           ████
0.0 |████████████
    └────────────────────────────→ x
              x* = -b/w
              (decision boundary)
```

- **Left of x*:** p < 0.5 → predict class 0
- **Right of x*:** p ≥ 0.5 → predict class 1
- **At x*:** p = 0.5 → tie (often predict class 1 by convention)

**Example with numbers:**
```
b = -1, w = 0.5
Decision boundary: -1 + 0.5·x = 0 → x = 2
```
- Customer with 1 complaint: z = -1 + 0.5(1) = -0.5 → p = 0.38 → predict "no churn"
- Customer with 3 complaints: z = -1 + 0.5(3) = 0.5 → p = 0.62 → predict "churn"

---

### Example 2: 2 Features (2D) — Most Common for Visualization

**Data:** Churn based on complaints (x₁) and calls to support (x₂)

```
z = b + w₁x₁ + w₂x₂
Decision boundary: b + w₁x₁ + w₂x₂ = 0
```

**Rearrange to get slope-intercept form (if w₂ ≠ 0):**
```
w₂x₂ = -b - w₁x₁
x₂ = (-b/w₂) + (-w₁/w₂)·x₁
```

This is a **straight line**!

**Visual:**
```
x₂ (calls)
    ↑
    |    ● class 1 (churn)
    |  ●
    |    ●
    |  ╲    ●
    |   ╲ ●
    |    ╲      ●
    |     ╲  ●
    |      ╲
    |   ●   ╲      ●
    | ●     ╲
    |●      ╲    ●
    |   ●    ╲
    |        ╲ ●
    |         ╲
    |  ●       ╲
    |           ╲
    |●           ╲
    └─────────────────────────→ x₁ (complaints)
               Decision boundary: b + w₁x₁ + w₂x₂ = 0
               (line separating churners from non-churners)
```

**Interpreting the line:**
- Points **above** the line → z > 0 → p > 0.5 → predict class 1
- Points **below** the line → z < 0 → p < 0.5 → predict class 0
- Points **on** the line → z = 0 → p = 0.5 → tie

---

## 🧠 **The Most Important Insight**

> **The decision boundary is ALWAYS linear for standard logistic regression.**

Why? Because:
- The boundary is defined by `z = 0`
- `z` is a **linear combination** of the features
- Setting a linear expression to 0 gives a line (or plane, or hyperplane)

**This is why logistic regression is called a "linear" classifier** — even though it outputs probabilities, its decision boundary is straight!

---

## 🎨 **Changing the Threshold → Moving the Boundary**

Here's where your understanding of thresholds really pays off!

**Default threshold = 0.5:** Boundary where z = 0

**If threshold = 0.7** (more conservative — only predict churn if very sure):

```
Decision rule: predict 1 if p ≥ 0.7
sigmoid(z) = 0.7
z = log(0.7/0.3) ≈ 0.847
Boundary: b + w₁x₁ + w₂x₂ = 0.847
```

**Effect:** The boundary **shifts** to make it harder to predict class 1.

**Visual:**
```
        threshold = 0.5        threshold = 0.7
        (default)              (conservative)
        
        ╲                      ╲
         ╲                      ╲
          ╲         ●            ╲    ●
           ╲    ●                ╲ ●
            ╲                    ╲
         ●   ╲                  ╲
              ╲         ●        ╲
           ●   ╲                 ╲
                ╲ ●            ●  ╲
              ●  ╲                ╲
    ●           ╲                 ╲
                ╲ ●               ╲
                 ╲                ╲
                 ← boundary        ← boundary shifted
                   (p=0.5)           (p=0.7)
```

Notice: The boundary moves **away** from the class 0 region, making it harder to classify points as class 1.

---

## 🔥 **Real Business Example**

**Problem:** Predict which customers will cancel their subscription (churn).

**Features:**
- `x₁` = months since last purchase
- `x₂` = number of support tickets opened

**Trained model:**
```
b = -2.5
w₁ = 0.3
w₂ = 0.8
```

**Decision boundary (threshold = 0.5):**
```
-2.5 + 0.3x₁ + 0.8x₂ = 0
x₂ = (2.5 - 0.3x₁) / 0.8
```

**Interpretation:**
- Customer A: last purchase 2 months ago, 1 support ticket
  ```
  z = -2.5 + 0.3(2) + 0.8(1) = -1.1 → p = 0.25 → predict "no churn"
  ```
- Customer B: last purchase 8 months ago, 4 support tickets
  ```
  z = -2.5 + 0.3(8) + 0.8(4) = 0.7 → p = 0.67 → predict "churn"
  ```

**Business meaning:** The decision boundary tells you the **combinations** of "time since purchase" and "support tickets" that predict churn. You could use this to identify at-risk customers (even those with high p but just below 0.5) for targeted retention campaigns.

---

## 🧪 **Interactive Intuition Builder**

Try these thought experiments:

1. **If `w₁` is very large** (complaints are very predictive):
   - Decision boundary becomes almost vertical
   - Even a few complaints push you across the boundary
   - The feature is "dominant"

2. **If `w₁` is very small** (complaints don't matter much):
   - Decision boundary becomes almost horizontal
   - Complaints have little effect on the prediction
   - Other features dominate

3. **If all weights are 0**:
   - `z = b`
   - Boundary: b = 0 → a flat line parallel to all axes
   - All predictions are constant (if b=0, everyone gets p=0.5)

---

## ⚠️ **Common Misconceptions to Avoid**

| Misconception | Reality |
|---------------|---------|
| "The decision boundary is curved" | **No** — standard logistic regression has a **linear** boundary |
| "The boundary is the sigmoid curve" | **No** — the sigmoid is the **probability curve**; the boundary is where p=0.5 (one point on that curve) |
| "Changing threshold changes the model" | **No** — it changes the **decision rule**, not the underlying weights |
| "Decision boundary only exists in 2D" | **No** — it exists in any dimension; we just can't visualize it beyond 3D (call it a hyperplane) |

---

## 🔑 **Key Takeaways**

1. **Decision boundary = where z = 0** (which is where p = 0.5)
2. **Equation:** `b + w₁x₁ + w₂x₂ + ... = 0`
3. **It's always linear** (straight line in 2D, plane in 3D, hyperplane in higher dimensions)
4. **Threshold changes the boundary** — higher threshold shifts boundary toward class 1 region (harder to predict 1)
5. **The weights control orientation** — large weights make that feature more influential on the boundary's position

---

## 🎯 **Check Your Understanding**

1. **If you have 3 features, what shape is the decision boundary?**
   *Answer: A plane in 3D space (hyperplane).*

2. **If you increase the threshold from 0.5 to 0.8, what happens to the decision boundary?**
   *Answer: It shifts so that a larger z is needed to predict class 1 — effectively making it harder to classify points as class 1.*

3. **What happens to the decision boundary if you double all weights?**
   *Answer: Nothing! Because b + w₁x₁ + ... = 0 is unchanged if you multiply everything by 2 — the probability changes, but the p=0.5 boundary stays the same.*

---

## 📝 **Final Visual Summary**

```
                    The Decision Boundary Story
                    ===========================
                    
     Feature Space                Probability Space
     (x₁, x₂, ...)                (z → p)
     
         ● class 1                   p
         ● ●                         1.0 ──────────
        ● ● ●                           ●  ●
      ● ○ ○ ○   ○                     ● ●
     ● ○ ○   ○ ○                    ●
    ○ ○ ○   ○ ○                   0.5 ──●──────────
   ○ ○ ○ ○ ○ ○                        ●
    ○ ○ ○ ○ ○                        ●
         ○ class 0                   ●
                                    0.0 ──────────→ z
                                    
    Decision boundary           Threshold = 0.5
    (line separating           at p=0.5 → z=0
    ● from ○)
```

**The decision boundary is just a shadow** — it's the projection of the p=0.5 threshold back into feature-space. Once you see this, it all clicks! 🎯
