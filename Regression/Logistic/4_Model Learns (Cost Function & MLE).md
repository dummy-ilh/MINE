

# Module 4 — How the Model Learns (Cost Function & MLE)

## 1. WHY

So far, we've been **assuming** the model already knows its weights (`b`, `w1`, `w2`). But in real life, nobody hands you those numbers — the model has to **learn** them from data. This module answers: *how does the model figure out which weights are "good"?*

**Why can't we just reuse linear regression's approach (squared error)?** Linear regression finds its weights by minimizing squared error — the classic "sum of (actual − predicted)²." It seems natural to try the same trick here: compare actual label (0 or 1) to predicted probability, square the difference, minimize it. Let's see why this breaks.

**What breaks:** If you use squared error with a sigmoid-based model, the resulting cost function is **non-convex** — meaning, if you plotted it, it would look like a bumpy landscape with multiple valleys (local minima), not one smooth bowl. Gradient descent (the "roll downhill" optimization method) can get stuck in a shallow local valley instead of finding the true best answer at the bottom of the deepest valley. This makes training unreliable — you might get a mediocre model depending on where you randomly started. We need a cost function that's guaranteed to have exactly **one** valley — a smooth bowl shape — so gradient descent always finds the true best answer.

## 2. INTUITION

Here's the plain-English idea behind how we're going to pick "good" weights: **choose the weights that make the data you actually observed look as likely as possible, according to the model.**

Think of it like being a detective: you have evidence (your actual dataset — who churned, who didn't). You're trying to find the "suspect" (the set of weights) whose story best explains that evidence. Weights that would have predicted "no one churns" when half your customers churned are bad suspects — they make the observed data look almost impossible. Weights that correctly predict high churn probability for people who actually churned, and low churn probability for people who didn't, are good suspects — they make the observed data look very plausible.

This detective-style approach has an official name: **Maximum Likelihood Estimation (MLE)** — "find the weights that MAXIMIZE the LIKELIHOOD of the data we actually saw."

## 3. SIMPLE FORMULA — Building Log-Loss from Scratch

Let's build this up piece by piece, starting from a **single data point**, in plain words first.

**Step 1 — "How good was this one prediction?"**

For a single customer, the model predicted some probability `p` that they'd churn. We also know the truth: did they actually churn (`y=1`) or not (`y=0`)?

**In words:**
> If the customer actually churned (y=1), we want to reward the model for predicting a HIGH probability of churning, and punish it for predicting a LOW probability.
> If the customer did NOT churn (y=0), we want to reward the model for predicting a LOW probability of churning, and punish it for predicting a HIGH probability.

**In simple notation, using natural log to turn this into a smooth, well-behaved penalty:**

```
loss for one point = 
    -log(p)       if the actual label y = 1
    -log(1 - p)   if the actual label y = 0
```

- `p` = the model's predicted probability that y=1 (churn)
- `y` = the true label (1 = churned, 0 = did not churn)
- `-log(...)` = negative natural log — this is our "penalty" function; we'll see exactly why it behaves the way we want in the worked example

**Step 2 — combine both cases into ONE formula (this is just a trick to avoid writing "if/else"):**

```
loss for one point = -[ y × log(p) + (1-y) × log(1-p) ]
```

- When `y=1`: the second term `(1-y) × log(1-p)` becomes `0 × log(1-p) = 0`, so we're left with just `-log(p)` — matches Step 1.
- When `y=0`: the first term `y × log(p)` becomes `0 × log(p) = 0`, so we're left with just `-log(1-p)` — also matches Step 1.

This combined formula is called **log-loss**, also known as **binary cross-entropy**. It's not a new idea — it's just a compact way of writing the two-case rule above.

**Step 3 — average this over your whole dataset:**

> Total cost = the average of this per-point loss, computed across every data point in your dataset.

That total average is what the model tries to **minimize** during training (equivalent to *maximizing* the likelihood from Module 4's intuition — minimizing loss and maximizing likelihood are two sides of the same coin, since loss is just negative log-likelihood).

## 4. WORKED NUMERIC EXAMPLE

Let's compute log-loss by hand for a tiny set of 4 customers, using their actual outcome and the model's predicted probability.

| Customer | Actual y | Predicted p | Loss formula | Loss value |
|---|---|---|---|---|
| 1 | 1 (churned) | 0.9 (confident, correct) | -log(0.9) | 0.105 |
| 2 | 1 (churned) | 0.1 (confident, WRONG) | -log(0.1) | 2.303 |
| 3 | 0 (stayed) | 0.2 (confident, correct) | -log(1-0.2) = -log(0.8) | 0.223 |
| 4 | 0 (stayed) | 0.8 (confident, WRONG) | -log(1-0.8) = -log(0.2) | 1.609 |

**Let's hand-verify Customer 2, since it's the most instructive:**
```
y = 1, p = 0.1
loss = -[ y × log(p) + (1-y) × log(1-p) ]
loss = -[ 1 × log(0.1) + 0 × log(0.9) ]
loss = -[ log(0.1) + 0 ]
loss = -(-2.303)
loss = 2.303
```

**Total cost (average over all 4 customers):**
```
average loss = (0.105 + 2.303 + 0.223 + 1.609) / 4
average loss = 4.240 / 4
average loss = 1.06
```

## 5. WHY LOG-LOSS PUNISHES CONFIDENT WRONG ANSWERS HARSHLY

Look closely at Customers 1 and 2 above. Both were "confident" predictions (0.9 and 0.1 are both far from the undecided 0.5), but Customer 1 was confidently RIGHT (loss = 0.105 — tiny) while Customer 2 was confidently WRONG (loss = 2.303 — over 20x bigger!).

**Why does the penalty explode like that?** Because of how `-log(p)` behaves as `p` approaches 0. Let's watch it happen:

| p (predicted probability of the TRUE outcome) | -log(p) |
|---|---|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |
| 0.001 | 6.908 |

**The pattern:** as your predicted probability for the correct answer gets closer and closer to 0 (meaning you were VERY confidently wrong), the penalty doesn't just increase gradually — it **shoots up toward infinity**. This is intentional design, not an accident: **log-loss is built to punish confident-and-wrong far more severely than uncertain-and-wrong.** A model that says "I'm 51% sure" and is wrong gets a small slap on the wrist. A model that says "I'm 99.9% sure" and is wrong gets hammered. This pushes the model, during training, to only be confident when it has strong evidence — exactly the behavior you want in a production system (nobody wants a model that's wildly overconfident and often wrong).

## 6. INTERPRETATION

In real terms: log-loss is the metric that actually drives training — it's what the optimizer is minimizing behind the scenes. Lower log-loss on a validation set means the model's predicted probabilities are both **accurate and well-calibrated**, not just "getting the right side of 0.5." Two models could have identical accuracy (both get 90% of customers correctly classified as churn/no-churn) but very different log-loss if one is confidently right/wrong more often — log-loss is a stricter, more information-rich metric than plain accuracy, which is why it's the standard training objective (we'll dig deeper into metric choice in Module 8).

## 7. FAANG L5 ANGLE

**Common interview question:** *"Why doesn't logistic regression use mean squared error like linear regression?"*
Strong answer: MSE with a sigmoid produces a **non-convex** cost surface with multiple local minima, making gradient descent unreliable. Log-loss (derived from Maximum Likelihood Estimation under a Bernoulli/binomial assumption) is **convex** for logistic regression, guaranteeing gradient descent converges to the global minimum.

**Common follow-up:** *"What is Maximum Likelihood Estimation, in plain terms?"*
Good answer: "MLE picks the model parameters that make the observed data most probable, according to the model." Then connect it: "Minimizing log-loss is mathematically the same thing as maximizing likelihood — log-loss is just negative log-likelihood, averaged over the dataset."

**Common follow-up:** *"What happens if p=0 or p=1 exactly, and the model is wrong?"*
Sharp answer: `-log(0)` is mathematically undefined (approaches infinity) — this is why in practice, libraries clip predicted probabilities to something like [1e-15, 1-1e-15] to avoid numerical errors (`log(0)` crashing the computation).

**Common trap:** Candidates say "log-loss and cross-entropy are different things" — they're not, for binary classification these terms are used interchangeably. Worth clarifying if it comes up, so you don't look confused by either term.

**Another trap:** Confusing "convex" with "always finds a global optimum regardless of learning rate/starting point" — convexity guarantees there's only ONE minimum to find, but poor learning rate choices can still cause slow convergence or overshooting (we'll cover this exact nuance in Module 5).

## 8. QUICK PYTHON CHECK

```python
import numpy as np

def log_loss_single(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

data = [(1, 0.9), (1, 0.1), (0, 0.2), (0, 0.8)]

losses = [log_loss_single(y, p) for y, p in data]
for (y, p), loss in zip(data, losses):
    print(f"y={y}, p={p}, loss={loss:.3f}")

print("Average loss:", np.mean(losses))
```

## 9. CHECK — before Module 5

1. Two models both predict for a customer who actually churned (y=1). Model A predicts p=0.6, Model B predicts p=0.99. Both are "correct" (p > 0.5). Which model gets a lower loss for this point, and why does that make sense given what we just learned?
2. In your own words, why is it a problem for gradient descent if the cost function is non-convex (has multiple valleys), rather than one smooth bowl shape?

Check 1 — confirmed, with the "why"
Yes, Model B (p=0.99) gets the lower loss. Let's verify with the actual numbers:
Model A: loss = -log(0.6) = 0.511
Model B: loss = -log(0.99) = 0.010
Model B's loss is 50x smaller than Model A's — even though both technically "got it right" (p > 0.5). This confirms the core lesson from Module 4: log-loss doesn't just care whether you're on the correct side of 0.5 — it cares how confidently correct you are. A model that's barely right gets a real (if modest) penalty; the loss function is nudging it during training to push its confidence higher when the evidence supports it.
Check 2 — confirmed, let's add the "why it's bad"
Yes — a non-convex cost function can have local minima (small dips that look like the bottom but aren't). Here's why that's a real practical problem, not just a theoretical curiosity:
Gradient descent works by "rolling downhill" from wherever it starts. If the landscape has multiple valleys, the algorithm has no way of knowing whether it's landed in the true best valley (global minimum) or just a decent-but-not-best valley (local minimum) nearby. It just stops when the ground flattens out under its feet — it can't see the rest of the landscape.
Practical consequence: train the same model twice with two different random starting weights, and with a non-convex cost function you could get two different final models with different performance — bad for reproducibility and reliability in production. This is exactly why log-loss's convexity (guaranteed single bowl-shaped valley for logistic regression) is such a big deal — no matter where you start, gradient descent is guaranteed to find the one true best answer.
