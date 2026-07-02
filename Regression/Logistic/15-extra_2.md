Got it — this is the exact format we've established since Module 0, restated in full. I'll work through all 10 items in that same structure. Given several of these build directly on things already covered (Modules 0–13, plus our recent topics), I'll keep each tight and cross-reference rather than re-deriving from scratch, but nothing will be skipped.

---

# 1) Plain-Language Rewrite of the Dense Text

## 1. WHY

The dense version you pasted packs 4 ideas into heavy notation. Let's unpack each one, in plain words first, connecting directly back to what you already know.

## 2. INTUITION + PLAIN REWRITE

**Idea 1 — "Linear in log-odds, not probability."**
This is Module 1 + Module 3, restated: we don't ask the straight-line part of the model to predict probability directly (since probability is trapped between 0 and 1, and a straight line can't respect that boundary — Module 0). Instead, we ask the straight line to predict **log-odds** — a number that can be anything from very negative to very positive — and THEN translate that back into probability using sigmoid. The dense notation `log(p/(1-p)) = β0 + β1X1 + ...` is just: **"take the log-odds; set it equal to baseline plus weighted features."** Same thing you built in Module 3.

**Idea 2 — "Coefficients are interpretable via odds ratios."**
This is Module 3's coefficient interpretation section: a coefficient `w` (they call it β) tells you how much the LOG-ODDS moves (additively) per 1-unit feature increase. Exponentiating it (`e^w`) converts that into how much the ODDS get MULTIPLIED by. `β1 = 0.7` → `e^0.7 ≈ 2.01` → "each 1-unit increase roughly doubles the odds." You already know this precisely.

**Idea 3 — "Natural output between 0 and 1."**
This is Module 2 (sigmoid) — restating that squashing z through sigmoid guarantees a valid probability, no matter what the linear part computed.

**Idea 4 — "Well-behaved optimization / convex loss."**
This is Module 4 — log-loss is convex when paired with logistic regression's structure, guaranteeing gradient descent finds ONE true minimum, unlike squared error, which is non-convex here.

**Bottom line: this whole dense paragraph is Modules 0-4, compressed into textbook language.** Nothing new — just recognize the translation.

## 3. CHECK

If someone showed you `log(p/(1-p)) = β0 + β1X1`, could you say out loud, in plain words, what each piece means without looking at any notation? Try it before moving on.

---

# 2) Hand Calculation — Forward Pass → Loss → One Gradient Descent Step

## 1. WHY

This stitches together Modules 3, 4, and 5 into ONE continuous worked example — exactly what an interviewer might ask you to do live on a whiteboard.

## 2. INTUITION

Forward pass = "make a prediction." Loss = "how wrong was it." Gradient descent step = "nudge the weights to be less wrong." One smooth pipeline.

## 3. WORKED NUMERIC EXAMPLE

**Setup:** 1 feature (complaints), 2 customers, starting weights `b=0, w=0`, learning rate `0.1`.

| Customer | x (complaints) | y (actual churn) |
|---|---|---|
| 1 | 2 | 1 |
| 2 | 1 | 0 |

**STEP 1 — FORWARD PASS (compute z, then p, for each customer):**

```
Customer 1: z = 0 + (0 × 2) = 0        → p = sigmoid(0) = 0.5
Customer 2: z = 0 + (0 × 1) = 0        → p = sigmoid(0) = 0.5
```

**STEP 2 — LOSS (log-loss per point, then average):**

```
Customer 1: y=1, p=0.5 → loss = -log(0.5) = 0.693
Customer 2: y=0, p=0.5 → loss = -log(1-0.5) = -log(0.5) = 0.693

average loss = (0.693 + 0.693) / 2 = 0.693
```

**STEP 3 — GRADIENT (average of (p-y)×x, and (p-y) for bias):**

```
Customer 1: (p-y) = 0.5 - 1 = -0.5
Customer 2: (p-y) = 0.5 - 0 = 0.5

gradient_w = [(-0.5 × 2) + (0.5 × 1)] / 2 = [-1.0 + 0.5] / 2 = -0.25
gradient_b = [-0.5 + 0.5] / 2 = 0
```

**STEP 4 — UPDATE WEIGHTS:**

```
new_w = 0 - (0.1 × -0.25) = 0 + 0.025 = 0.025
new_b = 0 - (0.1 × 0) = 0
```

**After ONE full pass: w = 0.025, b = 0.** Notice `w` moved positive — correctly, since Customer 1 (more complaints) churned, and Customer 2 (fewer complaints) didn't. The model is starting to learn the right direction after just one step.

## 4. INTERPRETATION

This exact 4-step sequence — forward pass, loss, gradient, update — is what happens EVERY iteration during `.fit()`. Being able to walk through it by hand, with real numbers, is one of the highest-value things you can demonstrate live in an L5 interview.

## 5. FAANG L5 ANGLE

**Common ask:** "Walk me through one iteration of training logistic regression, with numbers." This IS that answer — practice reproducing it fluently without notes.

## 6. CHECK

If Customer 1 had 5 complaints instead of 2, would you expect `gradient_w` to become MORE negative or MORE positive (before the update)? Reason through it without recomputing everything.

---

# 3) Calibration — Diagnosis and Fix

## 1. WHY

Module 8 introduced calibration as a concept. Now: **how do you actually DETECT a calibration problem, and what do you DO about it?**

## 2. INTUITION

You're checking: "when the model says 70%, does the event really happen ~70% of the time?" Diagnosis = measuring the gap. Fix = a second, small correction step layered on top of your (already-trained) model's raw output.

## 3. DIAGNOSIS — The Calibration Curve (Reliability Diagram)

**In words:**
> Take your validation predictions. Group them into buckets by predicted probability (e.g., 0-10%, 10-20%, ..., 90-100%). Within each bucket, compute the ACTUAL fraction of positives. Plot predicted probability (x-axis) against actual fraction (y-axis). A perfectly calibrated model traces a 45-degree diagonal line.

**Worked numeric example:**

| Predicted bucket | Customers in bucket | Actual churners | Actual rate |
|---|---|---|---|
| 60-70% | 100 | 45 | 45% |
| 70-80% | 100 | 55 | 55% |
| 80-90% | 100 | 60 | 60% |

**Reading this:** the model predicts 60-70% but reality is only 45% — the model is **overconfident** in this range (systematically predicting too high). This pattern — predicted probability consistently exceeding actual rate — is the calibration problem made visible.

## 4. FIX — Two Standard Techniques

**Platt Scaling:** fit a SECOND, small logistic regression, using your ORIGINAL model's output probability as the only input feature, predicting the true label. This learns a simple correction curve (essentially "squash/stretch" the original probabilities) to realign them with reality. Works well when the miscalibration follows a roughly sigmoid-shaped distortion.

**Isotonic Regression:** fits a more flexible, non-parametric (step-function-like) correction — doesn't assume any specific shape, just enforces that the correction is monotonically increasing (higher raw score always maps to higher-or-equal corrected score). More flexible than Platt scaling, but needs more data to fit reliably without overfitting the correction itself.

## 5. INTERPRETATION

In real terms: you keep your original model's RANKING ability (AUC) untouched, and layer a small correction on top just to fix the NUMBERS. This is common in production — train once for ranking quality, calibrate separately for probability accuracy, since they're different properties (Module 8).

## 6. FAANG L5 ANGLE

**Common question:** *"Your model ranks well (AUC=0.85) but the raw probabilities are unreliable. Fix it without retraining from scratch."* → Platt scaling or isotonic regression, fit on a held-out calibration set (not the same data used for the original training, to avoid overfitting the correction).

**Common trap:** recalibrating using the SAME data the original model trained on — this doesn't reveal genuine miscalibration, since the model has already "seen" and adapted to that exact data.

## 7. CHECK

If your calibration curve showed predicted 20% but actual rate was 35% (model UNDER-confident, not over), would Platt scaling still be applicable? What would the correction curve look like directionally?

---

# 4) Discriminative vs. Generative Models

## 1. WHY

This is a favorite "does the candidate understand the landscape" question — logistic regression belongs to a family called **discriminative** models, and interviewers often ask you to contrast it with **generative** alternatives (like Naive Bayes or LDA) to test whether you understand WHY you'd pick one over the other.

## 2. INTUITION

**Discriminative models** learn to draw the BOUNDARY between classes directly — "given these features, which side of the line are you on?" They don't bother modeling how the data itself was generated, just how to separate it.

**Generative models** learn how EACH CLASS actually generates its data — "if this were a churner, what would their features typically look like? If this were a non-churner, what would THEIR features typically look like?" — then use Bayes' rule to flip that around into a classification decision.

**Analogy:** Discriminative = a bouncer who's learned "people wearing X get turned away, people wearing Y get in" without knowing anything about WHY. Generative = a bouncer who deeply understands "regular customers tend to dress like THIS, troublemakers tend to dress like THAT" — and uses that fuller understanding to decide.

## 3. SIMPLE FORMULA — What Each Actually Models

**Discriminative (logistic regression):** directly models `P(y | x)` — "given the features, what's the probability of the outcome?" That's it. That's ALL logistic regression ever tries to learn.

**Generative (e.g., Naive Bayes):** models `P(x | y)` for EACH class (what do features look like, given the class?) AND `P(y)` (how common is each class overall?), then combines them via Bayes' rule:
```
P(y | x) = [ P(x | y) × P(y) ] / P(x)
```
- `P(x | y)` = how likely these feature values are, if this row truly belongs to class y
- `P(y)` = the overall base rate of class y (before seeing any features)
- `P(x)` = a normalizing constant (doesn't depend on y)

## 4. WORKED NUMERIC EXAMPLE (conceptual comparison)

Suppose you're classifying emails as spam/not-spam.

**Discriminative (logistic regression) approach:** directly learns "if this email has word 'free' and 3 exclamation marks, what's P(spam)?" — a direct mapping from features to outcome probability, no modeling of what "typical spam" looks like as a whole.

**Generative (Naive Bayes) approach:** separately learns "what's the typical word distribution in spam emails?" and "what's the typical word distribution in real emails?" and "what fraction of all emails are spam overall?" — then, for a NEW email, asks "which of these two generating processes more plausibly produced this exact email?"

## 5. INTERPRETATION

In real terms: generative models can do things discriminative models can't — like GENERATE new synthetic examples of a class (since they model the full data distribution), and they often need LESS data to train well when their generative assumptions are roughly correct. Discriminative models, like logistic regression, typically achieve BETTER classification accuracy asymptotically (with enough data), since they focus their entire modeling effort directly on the boundary, not on the (harder, more assumption-laden) task of modeling the full data distribution for each class.

## 6. FAANG L5 ANGLE

**Common question:** *"When would you prefer a generative model like Naive Bayes over logistic regression?"*
Strong answer: smaller datasets (generative models often converge faster with less data, given their assumptions hold), when you need to handle missing features gracefully (generative models can more naturally marginalize over missing data), or when you actually WANT to generate synthetic samples. Logistic regression is usually preferred with more data available and when raw classification accuracy is the main goal, since it makes no assumptions about how features are distributed within each class.

**Common trap:** confusing "generative" with "generative AI" (like GPT) — related conceptually (both model a data-generating process) but this is classical ML terminology predating and distinct from modern generative AI usage; worth clarifying if the term comes up ambiguously.

## 7. CHECK

Naive Bayes assumes features are conditionally independent given the class (a strong, often unrealistic assumption). Why might logistic regression, which makes NO such independence assumption, often outperform Naive Bayes when you have enough data?

---

# 5) MLE for Logistic Regression — Explained Simply

## 1. WHY

Module 4 gave you log-loss and mentioned it comes from MLE. Let's build the ACTUAL MLE derivation, in plain steps, so "log-loss = negative log-likelihood" stops being a fact you're told and becomes something you can derive yourself.

## 2. INTUITION

Recall the detective analogy from Module 4: find the weights that make the DATA YOU ACTUALLY OBSERVED as probable as possible. Let's make this concrete, step by step, for a tiny dataset.

## 3. SIMPLE FORMULA — Building the Likelihood From Scratch

**Step 1 — the probability of ONE observation, given the model.**

For a single customer, the model predicts `p` = probability they churn. If they ACTUALLY churned (`y=1`), the model "explains" that outcome with probability `p`. If they did NOT churn (`y=0`), the model explains it with probability `(1-p)`.

**In words, compactly:** "the probability the model assigns to what ACTUALLY happened for this one row" = `p` if y=1, or `(1-p)` if y=0. This can be written as ONE unified expression:
```
P(this row's outcome | model) = p^y × (1-p)^(1-y)
```
- when y=1: this becomes `p^1 × (1-p)^0 = p` ✓ (matches "if y=1, use p")
- when y=0: this becomes `p^0 × (1-p)^1 = (1-p)` ✓ (matches "if y=0, use 1-p")

**Step 2 — the probability of the WHOLE dataset, given the model.**

**In words:** assuming each row is independent (Module 10's independence assumption!), the probability of seeing your ENTIRE dataset is the PRODUCT of each individual row's probability (this is just "AND" logic — probability of independent events all happening together = multiply their individual probabilities).

```
Likelihood = (row 1's probability) × (row 2's probability) × ... × (row N's probability)
```

**Step 3 — why we take the LOG of this (this is the key trick).**

Multiplying hundreds or thousands of small probabilities together produces a TINY, computationally awkward number (risk of numerical underflow). Taking the LOG of a product turns it into a SUM (a basic log rule: `log(a×b) = log(a) + log(b)`) — much easier to compute and optimize.

```
Log-Likelihood = sum of [ y×log(p) + (1-y)×log(1-p) ]  across all rows
```

**Step 4 — MLE says: find the weights that MAXIMIZE this.** Equivalently (multiplying by -1 and averaging instead of summing), we can instead MINIMIZE:

```
-Log-Likelihood (averaged) = -average of [ y×log(p) + (1-y)×log(1-p) ]
```

**This is EXACTLY Module 4's log-loss formula.** You've now derived it from first principles, rather than just being handed it.

## 4. WORKED NUMERIC EXAMPLE

Reuse Module 4's 4-customer table:

| Customer | y | p |
|---|---|---|
| 1 | 1 | 0.9 |
| 2 | 1 | 0.1 |
| 3 | 0 | 0.2 |
| 4 | 0 | 0.8 |

**Likelihood of the whole dataset (product of individual row probabilities):**
```
L = 0.9 × 0.1 × (1-0.2) × (1-0.8)
L = 0.9 × 0.1 × 0.8 × 0.2
L = 0.0144
```

**Log-Likelihood (sum of logs):**
```
log(L) = log(0.9) + log(0.1) + log(0.8) + log(0.2)
log(L) = -0.105 + (-2.303) + (-0.223) + (-1.609)
log(L) = -4.240
```

**Negative log-likelihood, averaged over 4 rows:**
```
-log(L) / 4 = 4.240 / 4 = 1.06
```

**This 1.06 matches Module 4's "average loss" computation exactly** — same numbers, now understood as coming directly from the MLE derivation, not handed down as a formula.

## 5. INTERPRETATION

In real terms: "minimize log-loss" and "find the maximum likelihood weights" are the SAME optimization problem, just phrased with opposite signs. Every time gradient descent (Module 5) reduces log-loss, it's simultaneously increasing the likelihood of the observed data under the model — literally making your training data look more and more "expected" given the fitted weights.

## 6. FAANG L5 ANGLE

**Common question:** *"Derive log-loss from first principles."* — walk through exactly the 4 steps above: per-row probability → whole-dataset likelihood via independence → log to convert product to sum → negate and average to get something to MINIMIZE.

**Common trap:** stating "log-loss comes from MLE" without being able to show the actual product→log→sum chain — a strong L5 candidate can reproduce this derivation live, not just cite it.

## 7. CHECK

Why do we take the PRODUCT of individual row probabilities (Step 2) rather than the SUM, to get the whole dataset's likelihood?

---

# 6) Why Not MSE — Quick Recap

Already covered in depth in Module 4 — the short version, tightened:

MSE + sigmoid produces a **non-convex** cost surface (multiple local minima, unreliable gradient descent). Log-loss, derived from MLE (see #5 above), is **convex** for logistic regression, guaranteeing gradient descent reliably finds the single global minimum. Additionally, log-loss's `-log(p)` structure specifically punishes confident-wrong predictions far more severely than MSE would (Module 4, Section 5) — a property well-suited to probability estimation, not just an optimization convenience.

**FAANG framing:** always mention BOTH reasons — convexity (optimization) AND the confident-wrong penalty shape (statistical behavior) — citing only one is a partial answer.

---

# 7) "Decision Boundary Is the Line of Threshold??" — Clarifying

## Direct answer: Yes, essentially — let's make the wording precise.

The decision boundary is the set of points in feature space where the model is **exactly on the fence** at your chosen threshold — where predicted probability EQUALS the threshold value exactly. On one side of this boundary, the model predicts "yes"; on the other side, "no."

**Precise statement:** at threshold=0.5 (the common default), the boundary is where `z = 0` (since sigmoid(0)=0.5, Module 2). If you pick a DIFFERENT threshold (say 0.7, Module 6), the boundary shifts to wherever `z` equals whatever log-odds value corresponds to `sigmoid(z) = 0.7` (which you can solve for: `z = logit(0.7) ≈ 0.847`). **So yes — the decision boundary IS literally the geometric line traced out by your threshold condition, plotted in feature space.** Different thresholds → different (but still straight, parallel) boundary lines, since they're all still just "z = some constant" equations.

**CHECK:** if you moved your threshold from 0.5 to 0.9, would the decision boundary line move CLOSER to the "positive" cluster of points or FARTHER from it, geometrically? (Hint: think about which customers now need to be MORE convincingly at-risk to get flagged.)

---

# 8) Likelihood Ratio Test as a Goodness-of-Fit Test

## 1. WHY

You've already seen the LR test used to compare TWO nested models (does adding a feature help?). There's a related but distinct use: **testing whether your fitted model fits the data well AT ALL**, by comparing it against a hypothetical "perfect" model.

## 2. INTUITION

Imagine a **"saturated model"** — a hypothetical model so flexible it perfectly predicts every single training row (essentially, one parameter per row, memorizing everything). This isn't a MODEL you'd ever deploy (pure overfitting) — it's a theoretical BEST-CASE reference point representing "the absolute best any model could conceivably do on this exact data." Goodness-of-fit asks: **"how far is MY model's likelihood from that theoretical best-case?"**

## 3. SIMPLE FORMULA — Deviance

**In words:**
> Take -2 times your model's log-likelihood, minus -2 times the saturated model's log-likelihood. This difference is called "deviance" — smaller deviance means your model is closer to the theoretical best fit.

**In notation:**
```
Deviance = -2 × (log_likelihood_your_model - log_likelihood_saturated_model)
```

This deviance, like the LR statistic from before, follows a chi-squared distribution, letting you test: "is my model's fit significantly WORSE than the best possible fit, more than you'd expect from noise alone?"

## 4. WORKED EXAMPLE (conceptual)

If your model's deviance comes out LOW relative to the chi-squared threshold for your degrees of freedom (roughly: number of data points minus number of parameters), you fail to reject "my model fits reasonably well." A HIGH deviance suggests systematic lack of fit — maybe missing an important non-linear term (back to the Non-Linearity topic) or an important interaction.

## 5. FAANG L5 ANGLE

**Common question:** *"How would you assess whether your logistic regression model fits the data well overall, not just whether one feature helps?"* — mention deviance/goodness-of-fit testing against the saturated model, alongside more practical checks (calibration curves, residual analysis) — in practice, most working data scientists lean more on calibration curves and validation metrics day-to-day, but knowing the formal deviance-based test signals depth.

**Common trap:** confusing THIS use of the LR test (goodness-of-fit vs. a hypothetical saturated model) with the EARLIER use (comparing two of YOUR OWN nested models to test one feature) — same math machinery, genuinely different purpose and interpretation.

---

# 9) Proof: Decision Boundary Is Linear

Already built the intuition in Module 6, Section 5 and in item #7 above — here's the clean, step-by-step proof form:

**Given:** `z = b + w1*x1 + w2*x2` (linear combination — this is definitional, always true for logistic regression).

**Decision rule:** classify as positive when `p ≥ threshold`.

**Step 1:** Since sigmoid is a strictly increasing function (Module 2 — bigger z always gives bigger p, never reverses), the condition `p ≥ threshold` is EQUIVALENT to `z ≥ some_constant` (specifically, `some_constant = logit(threshold)`).

**Step 2:** Substitute the actual formula for z:
```
b + w1*x1 + w2*x2 ≥ some_constant
```

**Step 3:** This is a linear inequality in x1 and x2 — the BOUNDARY (where it's exactly equal, not just ≥) is:
```
b + w1*x1 + w2*x2 = some_constant
```

**Step 4:** Rearranging, this is the equation of a straight line (in 2D) or a flat hyperplane (in higher dimensions) — the same form as `Ax + By = C`, textbook linear algebra. QED.

**CHECK:** if you added an `x1²` term to `z` (non-linear feature engineering, from the earlier Non-Linearity topic), would this proof still hold with x1² substituted in as if it were just another linear input? What would the boundary look like when translated back into terms of the ORIGINAL x1 (not x1²)?

---

# 10) Confidence Intervals (on Coefficients / Odds Ratios)

## 1. WHY

The Wald test told you WHETHER a coefficient is significantly different from zero. A confidence interval tells you a full RANGE of plausible values for the true coefficient — richer information than a single significant/not-significant verdict.

## 2. INTUITION

Instead of "is this coefficient probably not zero" (yes/no), a CI answers: "if I re-ran this experiment/data-collection many times, what RANGE of values would the estimated coefficient typically fall into?" A narrow CI = precise estimate. A wide CI = a lot of remaining uncertainty, even if the point estimate looks confident.

## 3. SIMPLE FORMULA

**In words:**
> Take the coefficient's estimated value. Go out a certain number of standard errors (about 1.96, for a common 95% confidence level) in BOTH directions. That range is your confidence interval.

**In notation:**
```
CI = coefficient ± (1.96 × standard_error)
```

## 4. WORKED NUMERIC EXAMPLE

Recall from the Wald Test topic: `coefficient = 0.50`, `standard_error = 0.08`.

```
lower_bound = 0.50 - (1.96 × 0.08) = 0.50 - 0.157 = 0.343
upper_bound = 0.50 + (1.96 × 0.08) = 0.50 + 0.157 = 0.657

95% CI for the coefficient: [0.343, 0.657]
```

**Convert to odds ratio CI (exponentiate each endpoint):**
```
lower odds ratio = e^0.343 = 1.41
upper odds ratio = e^0.657 = 1.93
```

**Plain English:** "We're 95% confident the true odds ratio for this feature lies between about 1.41x and 1.93x" — a genuine range, more informative than just "the odds ratio is 1.65 and it's significant."

## 5. INTERPRETATION

In real terms: if a stakeholder asks "how confident are we in this 2x effect," a CI gives you a defensible, precise answer ("somewhere between 1.4x and 1.9x, with 95% confidence") rather than a single potentially-overconfident point number.

## 6. FAANG L5 ANGLE

**Common trap:** the classic frequentist misinterpretation — "there's a 95% probability the true value is in this interval" is technically WRONG phrasing; correct: "if we repeated this sampling process many times, 95% of such intervals would contain the true value." A subtle distinction some interviewers specifically probe.

---

# 11) "Levels of Linearity"

## 1. WHY

This phrase is worth unpacking because "linear" gets used in at least THREE different senses in this curriculum, and mixing them up is a common, sneaky source of confusion.

## 2. THE THREE SENSES

**Sense 1 — Linear in the PARAMETERS (always true, by definition):** `z = b + w1*x1 + w2*x2` is linear in `b, w1, w2` — this is what makes logistic regression a "Generalized LINEAR Model" (GLM). This is ALWAYS true, no matter what features you feed in (even x1² or interaction terms) — the weights themselves always combine additively.

**Sense 2 — Linear in the ORIGINAL features, with respect to log-odds (Module 10's assumption, and the one that can be VIOLATED):** whether log-odds moves in a straight line as your RAW, un-engineered feature (e.g., raw tenure, not tenure²) changes. This is the assumption we CHECK (binned log-odds plots) and can FIX (polynomial features, binning — the Non-Linearity topic).

**Sense 3 — NOT linear in probability (this is NEVER true, and that's intentional):** the relationship between z and PROBABILITY is always the S-curve (Module 2) — this is never linear, by design, and isn't something you'd want to "fix."

## 3. WHY THIS MATTERS

When someone says "is logistic regression linear?" — the honest answer is "it depends which of these three senses you mean," and giving a one-word answer without this distinction is exactly the kind of imprecision that gets flagged in Module 10's traps.

## 4. FAANG L5 ANGLE

**Common trap question:** *"Is logistic regression a linear model?"* — Strong answer: "Yes, in the sense that it's linear in its parameters (that's the 'linear' in Generalized Linear Model) and assumes linearity between features and log-odds — but the relationship with raw probability is explicitly non-linear (the sigmoid S-curve), and if the log-odds relationship with a raw feature is genuinely non-linear, we fix that through feature engineering, not by changing the model's fundamental linear-in-parameters structure."
