# Module 0 — Why Logistic Regression Exists At All

## 1. WHY (the problem)

Imagine you're building a model to predict: **"Will this customer churn?" (Yes/No)**

Your first instinct: "I already know linear regression, let me just use that."

You feed in features (age, usage, complaints) and try to predict `y = 1` (churn) or `y = 0` (stay). Let's see what happens.

**If we DON'T have logistic regression and just use linear regression for this:**
- Linear regression can output **any number**: -47, 0.3, 1.8, 500. But your actual answer is only ever 0 or 1.
- A prediction of "1.8" isn't a valid probability. A prediction of "-0.3" is nonsense — you can't have negative probability of churning.
- The line that best fits your 0/1 dots also gets **dragged around by outliers**, giving you a garbage decision boundary.

This isn't a small cosmetic issue — it's a structural mismatch: you're using a tool built for **unbounded continuous output** on a problem where the output is **bounded and categorical**.

## 2. INTUITION

Think of linear regression as a **ruler** — it measures things on an infinite scale, stretching in both directions forever.

Classification problems don't need a ruler. They need something more like a **dimmer switch that's capped between fully off (0) and fully on (1)**. You want a tool whose output can *only* live between 0 and 1, no matter what garbage input values you throw at it.

Logistic regression exists because we needed a way to say: *"Take the same kind of linear combination of inputs you'd use in linear regression, but force the output to always behave like a probability."*

## 3. SIMPLE FORMULA (no notation yet — just the numeric problem)

Let's just show what goes wrong with actual numbers using linear regression on a classification problem.

Say we're predicting whether someone churns (1) or stays (0), using **"number of complaints filed"** as the only feature.

Suppose linear regression fits this line:

**predicted value = 0.1 + 0.3 × (number of complaints)**

In words: "Start at a baseline of 0.1, and for every complaint, add 0.3."

## 4. WORKED NUMERIC EXAMPLE

Let's plug in different numbers of complaints and see what linear regression spits out:

| Complaints | Calculation | Predicted "probability" |
|---|---|---|
| 0 | 0.1 + 0.3×0 | 0.10 |
| 1 | 0.1 + 0.3×1 | 0.40 |
| 3 | 0.1 + 0.3×3 | 1.00 |
| 5 | 0.1 + 0.3×5 | **1.60** ⚠️ |
| -1 (imagine a scaled feature that can go negative) | 0.1 + 0.3×(-1) | **-0.20** ⚠️ |

Look at rows 4 and 5. A "probability" of **1.60** means 160% chance of churning. A "probability" of **-0.20** means negative-20% chance. Both are mathematically meaningless. Probabilities **must** live strictly between 0 and 1.

This isn't a rare edge case — with real data (especially with multiple features, or values outside your training range), linear regression **will** eventually produce impossible outputs. It has no built-in mechanism to stop itself at 0 or 1 — it just keeps going in a straight line forever.

## 5. INTERPRETATION

In real terms: if you shipped this model to a business dashboard and told a manager *"Customer X has a 160% chance of churning,"* they'd rightly lose trust in the model. Worse — if you used this to rank customers by churn risk for an intervention campaign, the ranking near the extremes gets distorted because the straight line doesn't curve/flatten near 0 and 1 the way real probabilities should.

There's also a second, quieter problem: the **decision boundary**. With linear regression forced into classification (sometimes called "linear probability model"), a few extreme outlier points can tilt the whole line, shifting your 0/1 cutoff in a way that misclassifies points that were previously fine. Logistic regression's curved shape is much more robust to this.

## 6. FAANG L5 ANGLE

**Common interview question:** *"Why not just use linear regression for a binary classification problem?"*

Strong answer hits three points:
1. Outputs aren't bounded to [0,1] → not valid probabilities.
2. Errors (residuals) aren't normally distributed / homoscedastic for a 0/1 target → violates linear regression assumptions, so your statistical inference (p-values, confidence intervals) is invalid.
3. The relationship between features and probability of an event is typically **not linear** near the extremes — real probabilities flatten out near 0 and 1 (diminishing returns), which a straight line can't capture but an S-shaped curve can.

**Common trap:** Candidates say "linear regression doesn't work for classification" without explaining *why* — interviewers will push with "what specifically breaks?" and expect the bounded-output argument plus the S-curve argument, not just a vague gut feeling.

**Follow-up they might ask:** *"Could you just clip the linear regression output to [0,1]?"* — Good answer: clipping is a band-aid; it doesn't fix the shape of the relationship, it just truncates it. You'd get a huge pileup of predictions exactly at 0 or 1, and the model still isn't optimized to produce well-calibrated probabilities in between. It's a hack, not a solution.

## 7. CHECK — before we move to Module 1

1. In your own words: why is it specifically a problem that linear regression's output is unbounded, when we're trying to predict something like "probability of churn"?
2. If I showed you a linear regression model predicting "probability of default" and it output **-0.05** for a very low-risk customer, what would you say is fundamentally going wrong under the hood (not just "it's negative")?

it's not just "makes no sense," it's that linear regression has no mechanism to stop at 0 or 1. It just keeps extrapolating in a straight line forever, so any sufficiently large or small input will eventually break the [0,1] bound. That's the structural flaw we need to fix.
