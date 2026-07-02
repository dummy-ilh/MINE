# Module 7 — Regularization (L1/L2)

## 1. WHY

Imagine you train your churn model on 500 customers, using 50 features (some genuinely predictive, others just noisy nonsense — like "customer's favorite color," which has no real relationship to churn). Without any safeguard, gradient descent will happily assign a large weight to ANY feature that seems to reduce training loss — **even if that "pattern" is pure coincidence in your specific training data.**

**What breaks without regularization:** The model ends up **overfitting** — it memorizes quirks and noise specific to your training set (like "every customer named Dave happened to churn in our sample" — an absurd coincidence, but the model doesn't know that). It then performs great on training data but **falls apart on new, unseen data**, because those coincidental patterns don't hold up in the real world. Regularization exists to actively discourage the model from leaning too heavily on any single feature, keeping it more honest and generalizable.

## 2. INTUITION

Think of training a model like grading an exam where **the only rule is "get the highest score possible."** Without any other constraint, a student might resort to memorizing the exact answer key from a leaked copy of THIS SPECIFIC exam — total mastery of this test, zero real understanding, complete failure on any new exam.

**Regularization is like adding a second rule: "get a high score, AND keep your effort/complexity reasonable."** It's a penalty added directly into the training process that says: *"Every time you consider making a weight really large, remember that large weights come with a cost — only do it if the improvement to your predictions is worth that cost."* This keeps the model from swinging wildly in response to any single feature, forcing it to rely on patterns that are broadly, robustly true — not just convenient coincidences in this one dataset.

## 3. SIMPLE FORMULA

**Recall from Module 4:** normally, training minimizes just the log-loss (cost) function. Regularization **adds an extra penalty term** on top of that, based on the size of the weights.

**In words — the general idea:**
> Take the normal cost (log-loss, from Module 4). Add an extra penalty that grows larger the bigger your weights get. The optimizer must now balance "predict well" against "keep weights small."

**In simple notation:**

```
new_cost = original_log_loss + (penalty_strength × penalty_on_weights)
```

- `original_log_loss` = the same cost function from Module 4 (how wrong the predictions are)
- `penalty_strength` = a number you choose (often called lambda, λ) — controls HOW MUCH we care about keeping weights small. Bigger number = stronger penalty = simpler model.
- `penalty_on_weights` = a formula measuring "how large are the weights overall" — this is where L1 and L2 differ (next section)

---

### L2 Regularization (a.k.a. "Ridge")

**In words:**
> Take every weight, square it, and add up all these squared values.

**In notation:**
```
penalty_on_weights (L2) = sum of (each weight squared)
```

**Plain-English effect:** squaring means big weights get punished MUCH more harshly than small weights (since squaring a big number makes it much bigger, but squaring a small number keeps it small). This pushes ALL weights to become smaller and more evenly spread out — but rarely forces any weight all the way down to exactly zero. Think of it as **"shrinkage"** — everything gets gently squeezed toward zero, but nothing gets eliminated entirely.

---

### L1 Regularization (a.k.a. "Lasso")

**In words:**
> Take every weight, use its absolute value (ignore the sign, just the size), and add up all these values.

**In notation:**
```
penalty_on_weights (L1) = sum of (absolute value of each weight)
```

**Plain-English effect:** unlike squaring, using absolute value treats all weight sizes more "evenly" in a specific mathematical sense that causes something surprising: L1 tends to push **some weights all the way to EXACTLY zero**, effectively **deleting those features from the model entirely.** Think of it as **"selection"** — L1 acts like an automatic feature selector, keeping only the features it deems truly useful and zeroing out the rest.

## 4. WORKED NUMERIC EXAMPLE

Let's use a tiny model with 3 weights: `w1 = 2.0`, `w2 = 0.1`, `w3 = -3.0` (ignore the intercept for simplicity).

**Suppose original log-loss = 0.40** (from Module 4's calculations, just a placeholder number).

**Compute the L2 penalty:**
```
L2 penalty = w1² + w2² + w3²
L2 penalty = (2.0)² + (0.1)² + (-3.0)²
L2 penalty = 4.0 + 0.01 + 9.0
L2 penalty = 13.01
```

**Compute the L1 penalty:**
```
L1 penalty = |w1| + |w2| + |w3|
L1 penalty = |2.0| + |0.1| + |-3.0|
L1 penalty = 2.0 + 0.1 + 3.0
L1 penalty = 5.1
```

**Now apply a penalty_strength (λ) of 0.1, and compute the new total cost for each:**

```
new_cost (L2) = 0.40 + (0.1 × 13.01) = 0.40 + 1.301 = 1.701
new_cost (L1) = 0.40 + (0.1 × 5.1)  = 0.40 + 0.51  = 0.91
```

**What this means in practice:** during training, the optimizer is no longer just trying to minimize 0.40 — it's trying to minimize 1.701 (for L2) or 0.91 (for L1). To reduce THIS new number, the optimizer has two levers: improve predictions (lower the log-loss part) OR shrink the weights (lower the penalty part). **This is the core mechanism — the optimizer will now voluntarily sacrifice a little bit of raw training accuracy if it means keeping weights smaller,** because the total cost formula rewards that tradeoff.

**Now let's see WHY L1 zeroes out weights but L2 doesn't**, using `w2 = 0.1` (a small, weak feature) as the example:

- Under L2, `w2`'s contribution to the penalty is `0.1² = 0.01` — tiny. There's very little pressure to shrink this weight further; the penalty barely notices it.
- Under L1, `w2`'s contribution to the penalty is `|0.1| = 0.1` — proportionally much larger relative to its size than the L2 case. Even a small weight still "costs" something meaningful under L1, so if this feature isn't pulling its weight in reducing log-loss, the optimizer will keep shrinking it — often all the way to exactly 0 — because there's no diminishing benefit to shrinking small weights the way there is with squaring.

This asymmetry (squaring shrinks proportionally less for small numbers; absolute value doesn't) is the real mathematical reason L1 produces sparse (many-zero) models while L2 produces small-but-nonzero models across the board.

## 5. WHEN TO USE WHICH

| Situation | Preferred | Why |
|---|---|---|
| You suspect many features are irrelevant/noisy | **L1** | Automatically zeroes out useless features — built-in feature selection |
| You want to keep all features but prevent any one from dominating | **L2** | Shrinks everything smoothly, keeps all features "in play" at smaller scale |
| You have highly correlated features | **L2** (or Elastic Net) | L1 tends to arbitrarily pick one of several correlated features and zero out the others, which can be unstable; L2 spreads weight across correlated features more evenly |
| You want interpretability with a small final feature list | **L1** | Sparse output is easier to explain — "these 8 features matter, the other 40 don't" |
| You want both benefits | **Elastic Net** | Combines L1 + L2 penalties together — sparsity AND smooth shrinkage |

## 6. INTERPRETATION

In real terms: regularization is the knob you turn to trade off between "fit the training data as closely as possible" and "generalize well to new data." If a model performs great on training data but poorly on a held-out validation set (a classic overfitting symptom), increasing the regularization strength (λ) is one of the first things to try — it forces the model to rely on more robust, broadly-true patterns rather than memorized noise. If you're doing L1 and see a feature's weight go to exactly zero, that's the model telling you **"this feature didn't earn its keep — removing it barely hurt predictions."**

## 7. FAANG L5 ANGLE

**Common interview question:** *"What's the difference between L1 and L2 regularization?"*
Strong answer: both add a penalty on weight size to the cost function to fight overfitting. L2 (Ridge) squares the weights, which shrinks all weights smoothly toward zero but rarely to exactly zero. L1 (Lasso) uses absolute value, which tends to push some weights to exactly zero, effectively performing automatic feature selection.

**Common follow-up:** *"Why does L1 produce sparsity but L2 doesn't? Explain geometrically or intuitively, not just formula-wise."*
Good answer (the intuitive version, since you can also cite the geometric "diamond vs. circle constraint region" explanation as a bonus if asked): small weights get squared into even tinier penalties under L2 (diminishing pressure to shrink further), while under L1 the penalty per unit of weight stays constant regardless of size, so there's just as much pressure to zero out a small weight as a large one — pushing weak features all the way to zero.

**Common follow-up:** *"How do you choose the regularization strength (λ)?"*
Good answer: typically via cross-validation — try several λ values, measure validation performance (e.g., log-loss or AUC) for each, pick the one that generalizes best. Too small a λ risks overfitting (barely any regularization effect); too large a λ risks underfitting (model becomes too simple, ignoring even genuinely useful features).

**Common trap:** Saying "regularization always improves the model." This is wrong — regularization trades a bit of training-set fit for better generalization; if λ is set too high, you can actively hurt performance by over-simplifying a model that didn't need it (underfitting). It's a tradeoff, not a free win.

**Another trap:** Confusing L1/L2 regularization with L1/L2 *distance* or *norm* used elsewhere in ML (e.g., in KNN or vector similarity) — same underlying math concept (sum of absolute values vs. sum of squares), but applied in a totally different context. Worth clarifying if the interviewer's question is ambiguous.

**Bridge to Module 12 (deep learning):** the exact same L1/L2 penalty concept applies directly to neural network weights — it's called "weight decay" in that context (L2 specifically) and is a standard tool to fight overfitting in deep learning, so this isn't a concept you'll retire after logistic regression.

## 8. QUICK PYTHON CHECK

```python
import numpy as np

w = np.array([2.0, 0.1, -3.0])
original_log_loss = 0.40
lam = 0.1

l2_penalty = np.sum(w ** 2)
l1_penalty = np.sum(np.abs(w))

new_cost_l2 = original_log_loss + lam * l2_penalty
new_cost_l1 = original_log_loss + lam * l1_penalty

print(f"L2 penalty: {l2_penalty:.3f}, new cost: {new_cost_l2:.3f}")
print(f"L1 penalty: {l1_penalty:.3f}, new cost: {new_cost_l1:.3f}")
```

## 9. CHECK — before Module 8

1. You train a logistic regression model with L1 regularization and notice that 30 out of 50 features end up with a weight of exactly 0. What does this tell you, in plain English, about those 30 features?
2. If you set λ (penalty_strength) extremely high — like 1000 — what do you predict would happen to the model's weights, and what problem would this likely cause?

Once you've got these, I'll append Module 7 to your cheat sheet and we'll move to **Module 8 — Evaluation Metrics Deep Dive** — precision, recall, F1, ROC-AUC vs PR-AUC, and calibration.


Check 1 — confirmed
Correct. Those 30 zeroed-out weights mean L1 has effectively performed feature selection for you — it decided those features contributed so little to reducing log-loss that they weren't worth the "cost" of a nonzero weight. In practice, you'd walk away from this model with a much shorter, more interpretable feature list (20 features instead of 50), and you could confidently drop the other 30 from your pipeline entirely — simpler data collection, faster inference, easier to explain to stakeholders.


Check 2 — confirmed, one addition
Correct — extremely high λ crushes all weights toward zero, and in the limit, if every weight is essentially 0, your model's prediction collapses to just the intercept b — meaning it predicts the same probability for every single customer, regardless of their features. That's the textbook definition of underfitting: the model has become so simple it ignores real, useful signal in the data, trading away genuine predictive power purely to satisfy the penalty term. This is the flip side of the overfitting problem regularization was meant to solve — too little regularization overfits, too much underfits, and the right λ sits somewhere in between (found via cross-validation, as mentioned in the FAANG angle above).
