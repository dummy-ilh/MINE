-

# Module 5 — Gradient Descent for Logistic Regression

## 1. WHY

Module 4 told us **what** makes a set of weights "good" (low log-loss). But it didn't tell us **how to find** those good weights. We can't just guess-and-check every possible combination of weights forever — with even a handful of features, the number of possible weight combinations is infinite. We need a systematic, repeatable process that **starts somewhere and improves step by step** until it lands on (or very near) the best weights. That process is **gradient descent**.

**What breaks without it:** Without a method to *search* for good weights, we'd be stuck with randomly guessed weights and no way to systematically improve them. Gradient descent gives us a reliable, repeatable recipe: look at how wrong we currently are, figure out which direction reduces that wrongness, and take a small step in that direction. Repeat.

## 2. INTUITION

Picture yourself standing somewhere on a **foggy hillside** at night, and your goal is to reach the very bottom of the valley (where the log-loss cost is lowest). You can't see the whole landscape because of the fog — you can only feel the slope of the ground right under your feet.

**Your strategy:** feel which direction is steepest downhill from where you're standing right now, take one small step in that direction, then repeat — feel the new slope, step again, feel again, step again — until the ground feels flat (you've reached the bottom).

That's gradient descent, described without any math: **repeatedly move in the direction that decreases the cost, by a small step, until you stop improving.**

- "Feeling the slope" = computing the **gradient** (a fancy word for "the direction and steepness of the cost function at your current weights").
- "Taking a small step" = updating the weights slightly, guided by that slope.
- "How big a step" = the **learning rate** — a setting you choose. Too big a step and you might overshoot the valley floor and bounce around; too small a step and it takes forever to get there.

## 3. SIMPLE FORMULA — The Update Rule

**In words, for ONE weight:**
> Take the current value of the weight. Subtract a small fraction (the learning rate) of "how much the cost would increase if we increased this weight" (the slope/gradient). The result is the new, updated value of the weight.

**In simple notation:**

```
new_weight = old_weight - (learning_rate × gradient)
```

- `old_weight` = the weight's current value, before this update
- `learning_rate` = a small number you choose (e.g., 0.1) — controls step size
- `gradient` = how much the cost function would change if we nudged this weight upward (tells us the "slope" at our current position)
- `new_weight` = the weight's value after taking one step

**Why subtract, and not add?** Because the gradient points in the direction of **increasing** cost. Since we want to **decrease** cost, we move in the *opposite* direction — hence the minus sign.

**The specific gradient formula for logistic regression (built from what we already know):**

It turns out that thanks to the clean math of sigmoid + log-loss together (their derivatives cancel nicely — this combination isn't an accident, it's *why* log-loss was chosen), the gradient for one feature's weight, averaged over your dataset, simplifies to something surprisingly clean:

**In words:**
> For each data point: take the model's predicted probability, subtract the actual label (0 or 1), then multiply that difference by that data point's feature value. Average this across all data points.

**In notation, for one feature's weight, averaged over N data points:**

```
gradient = average of [ (p - y) × x ]
```

- `p` = the model's predicted probability for that data point
- `y` = the actual label (0 or 1) for that data point
- `x` = the value of this particular feature for that data point
- The average is taken across all data points in your training set (or a batch of them)

**Plain-English meaning:** `(p - y)` is simply **"how wrong was this prediction, and in which direction?"** If p is too high compared to y, this is positive (we over-predicted — nudge the weight down). If p is too low compared to y, this is negative (we under-predicted — nudge the weight up). Multiplying by `x` scales this correction by how much that feature was even present for this data point (a feature that was 0 for this customer shouldn't get "blamed" for the error).

## 4. WORKED NUMERIC EXAMPLE — 2 Iterations by Hand

Let's use a **tiny** dataset: 2 customers, 1 feature (number of complaints), and walk through 2 full iterations of gradient descent by hand.

**Data:**
| Customer | x (complaints) | y (actual churn) |
|---|---|---|
| 1 | 1 | 0 |
| 2 | 3 | 1 |

**Starting point:** let's initialize both weights to 0 (a common simple starting choice).
```
b = 0
w = 0
learning_rate = 0.1
```

---

### ITERATION 1

**Step 1 — forward pass: compute z and p for each customer, using current weights (b=0, w=0):**

Customer 1: `z = 0 + (0 × 1) = 0` → `p = sigmoid(0) = 0.5`
Customer 2: `z = 0 + (0 × 3) = 0` → `p = sigmoid(0) = 0.5`

(Makes sense — with all weights at 0, the model has no information yet, so it predicts 50/50 for everyone.)

**Step 2 — compute the error term (p - y) for each customer:**

Customer 1: `p - y = 0.5 - 0 = 0.5` (over-predicted — actual was "no churn," model said 50%)
Customer 2: `p - y = 0.5 - 1 = -0.5` (under-predicted — actual was "churn," model said only 50%)

**Step 3 — compute the gradient for w (multiply error by x, then average):**

```
gradient_w = average of [ (p-y) × x ]
gradient_w = [ (0.5 × 1) + (-0.5 × 3) ] / 2
gradient_w = [ 0.5 + (-1.5) ] / 2
gradient_w = -1.0 / 2
gradient_w = -0.5
```

**Step 4 — compute the gradient for b (same idea, but x is effectively always 1 for the intercept):**

```
gradient_b = average of [ (p-y) × 1 ]
gradient_b = [ 0.5 + (-0.5) ] / 2
gradient_b = 0.0 / 2
gradient_b = 0.0
```

**Step 5 — update both weights using the update rule:**

```
new_w = old_w - (learning_rate × gradient_w)
new_w = 0 - (0.1 × -0.5)
new_w = 0 + 0.05
new_w = 0.05

new_b = old_b - (learning_rate × gradient_b)
new_b = 0 - (0.1 × 0.0)
new_b = 0
```

**After Iteration 1: w = 0.05, b = 0.** Notice `w` moved slightly positive — which makes sense! Customer 2 (who churned) had MORE complaints than Customer 1 (who didn't churn), so the model is correctly starting to learn "more complaints → more churn," one small nudge at a time.

---

### ITERATION 2

**Step 1 — forward pass with updated weights (b=0, w=0.05):**

Customer 1: `z = 0 + (0.05 × 1) = 0.05` → `p = sigmoid(0.05) ≈ 0.5125`
Customer 2: `z = 0 + (0.05 × 3) = 0.15` → `p = sigmoid(0.15) ≈ 0.5374`

**Step 2 — error terms:**

Customer 1: `p - y = 0.5125 - 0 = 0.5125`
Customer 2: `p - y = 0.5374 - 1 = -0.4626`

**Step 3 — gradient for w:**

```
gradient_w = [ (0.5125 × 1) + (-0.4626 × 3) ] / 2
gradient_w = [ 0.5125 + (-1.3878) ] / 2
gradient_w = -0.8753 / 2
gradient_w = -0.4377
```

**Step 4 — gradient for b:**

```
gradient_b = [ 0.5125 + (-0.4626) ] / 2
gradient_b = 0.0499 / 2
gradient_b = 0.0250
```

**Step 5 — update weights:**

```
new_w = 0.05 - (0.1 × -0.4377) = 0.05 + 0.0438 = 0.0938
new_b = 0 - (0.1 × 0.0250) = -0.0025
```

**After Iteration 2: w ≈ 0.0938, b ≈ -0.0025.**

**The trend so far:** `w` is steadily climbing (0 → 0.05 → 0.094), confirming the model is progressively learning that complaints predict churn. If we kept running this for many more iterations, `w` would keep climbing (and `b` adjusting) until the gradients shrink close to zero — meaning we've reached the bottom of the valley, where further steps stop meaningfully improving the cost.

## 5. LEARNING RATE INTUITION

Picture three versions of the same foggy-hillside walk:

- **Learning rate too small** (e.g., 0.0001): you take tiny, cautious steps. You'll eventually reach the bottom, but it might take thousands of iterations — painfully slow, wasting compute and time.
- **Learning rate too large** (e.g., 10): you take huge leaps. You might overshoot the valley entirely, land on the opposite hillside, then overshoot again on the way back — bouncing back and forth without ever settling down (or even diverging further away each time).
- **Learning rate just right:** steady, confident progress toward the bottom in a reasonable number of steps.

**Convergence issues to know for interviews:**
- If your cost is oscillating wildly or increasing over iterations, your learning rate is too high — reduce it.
- If your cost is decreasing but painfully slowly, your learning rate may be too low — increase it, or consider fancier optimizers (Adam, RMSProp, etc. — the deep learning versions of "smarter step sizing," which you'll meet in your MLP curriculum).
- A common practical trick: start with a slightly larger learning rate and gradually shrink it over training ("learning rate decay/scheduling") — big steps early when far from the answer, tiny careful steps late when close.

## 6. INTERPRETATION

In real business terms: gradient descent is the "training" your ML pipeline runs when you call `.fit()` on a logistic regression model. Every iteration, it's asking "given my current guess at the weights, how wrong am I on the training data, and which direction should I nudge each weight to be less wrong?" After enough iterations, the weights settle into values that make the model's predictions match reality as closely as log-loss allows. Understanding this process matters in production because if a model **fails to converge** (cost isn't decreasing) or **converges very slowly**, gradient descent + learning rate diagnostics are the first place you look — long before assuming something is wrong with your data or features.

## 7. FAANG L5 ANGLE

**Common interview question:** *"Walk me through what happens in one step of gradient descent for logistic regression."*
Strong answer: (1) forward pass — compute z, then p via sigmoid, for each data point; (2) compute error (p - y) for each point; (3) compute the gradient — average of (p-y)×x per weight; (4) update each weight by subtracting learning_rate × gradient.

**Common follow-up:** *"What's the difference between batch, stochastic, and mini-batch gradient descent?"*
Good answer: "Batch" uses the ENTIRE dataset to compute the gradient before each update (accurate but slow per step, especially at scale). "Stochastic" (SGD) uses just ONE data point per update (fast per step, but noisy/jumpy path to the minimum). "Mini-batch" is the practical middle ground used almost everywhere in industry — a small random subset (e.g., 32 or 256 points) per update, balancing speed and stability.

**Common follow-up:** *"How do you know when to stop training (convergence)?"*
Good answer: monitor the cost (log-loss) on a validation set each iteration; stop when it stops meaningfully decreasing (or starts increasing — a sign of overfitting, foreshadowing Module 7 regularization), or after a fixed max number of iterations as a safety net.

**Common trap:** Candidates describe gradient descent purely as "an optimization algorithm" without being able to state the actual update rule or gradient formula for logistic regression specifically — L5 interviewers often want you to derive or at least state the `(p-y)×x` gradient by hand, since it demonstrates you understand *why* sigmoid + log-loss were paired together (their derivatives combine cleanly into this simple form).

**Another trap:** Forgetting that a poorly chosen learning rate can make even a perfectly convex cost function (Module 4) fail to converge in practice — convexity guarantees a single minimum EXISTS, not that any learning rate will find it efficiently.

## 8. QUICK PYTHON CHECK

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Data
X = np.array([1, 3])   # complaints
y = np.array([0, 1])   # actual churn

b, w = 0.0, 0.0
lr = 0.1

for iteration in range(2):
    z = b + w * X
    p = sigmoid(z)
    error = p - y
    grad_w = np.mean(error * X)
    grad_b = np.mean(error)
    w = w - lr * grad_w
    b = b - lr * grad_b
    print(f"Iteration {iteration+1}: w={w:.4f}, b={b:.4f}")
```

## 9. CHECK — before Module 6

1. In your own words: what does the term `(p - y)` represent in the gradient formula, and why does multiplying it by `x` make intuitive sense (rather than just using `(p-y)` alone)?
2. You train a model and notice the cost function value is bouncing up and down wildly between iterations instead of steadily decreasing. What's the most likely cause, and what would you try first to fix it?

Got it — here are the model answers to the Module 5 checks, then straight into Module 6.
Model Answers — Module 5 Checks
1. What does (p - y) represent, and why multiply by x?
(p - y) is "how wrong was this prediction, and in which direction." If p is bigger than y, we over-predicted (nudge the weight down). If p is smaller than y, we under-predicted (nudge the weight up).
Multiplying by x matters because it scales the blame by how much that feature was even involved. If a feature was 0 for a given customer (e.g., they had 0 complaints), that feature didn't contribute to the wrong prediction for that customer — so it shouldn't get "blamed" or adjusted based on that data point. A feature with a large value that was present when the model was wrong deserves a bigger correction than a feature that barely showed up.
2. Cost is bouncing wildly instead of steadily decreasing — cause and fix?
Most likely cause: learning rate is too high — you're taking steps so large that you keep overshooting the valley floor and landing on the opposite hillside, bouncing back and forth (or even diverging further away each time). First fix to try: reduce the learning rate (e.g., cut it by 10x) and re-run; if the bouncing stops and cost decreases steadily, that confirms the diagnosis.
