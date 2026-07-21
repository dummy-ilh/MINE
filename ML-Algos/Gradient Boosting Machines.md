# Gradient Boosting Machines (GBM) — Reorganized Reference

---

## 1. The Core Idea (ELI20)

Imagine you're trying to guess someone's age from a photo. You make a guess. You're off by 5 years. So your friend looks at *just your mistake* and tries to correct it. Then another friend corrects *their* residual. Each person only focuses on fixing what the last person got wrong.

That's GBM. A sequence of weak learners (shallow trees), each one trained on the *errors* of the previous ensemble.

> **One sentence:** GBM builds trees one at a time, where each new tree fits the residual errors of all previous trees, and predictions are summed together.

---

## 2. Why Not Just One Big Tree?

A deep tree memorizes the data (overfits). A shallow tree is weak but generalizes. GBM asks: *what if we add many weak learners slowly, each correcting the last?* You get the power of a complex model with better generalization. This is the **bias-variance tradeoff in action** — each tree has high bias but low variance, and together they reduce bias without exploding variance.

---

## 3. Pseudocode

```
1. Initialize prediction: F₀(x) = mean(y)

2. For m = 1 to M (number of trees):
   a. Compute residuals: rᵢ = yᵢ - Fₘ₋₁(xᵢ)
   b. Fit a shallow tree hₘ(x) to the residuals {xᵢ, rᵢ}
   c. Update: Fₘ(x) = Fₘ₋₁(x) + η * hₘ(x)

3. Final prediction: F_M(x)
```

`η` = learning rate (how much we trust each tree). Lower = more conservative = better generalization, but needs more trees.

---

## 4. Formal Definition

Let the loss function be $L(y, F(x))$. GBM minimizes this loss in **function space** using gradient descent.

**Residuals are the negative gradient of the loss:**

$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

For **MSE loss** $L = \frac{1}{2}(y - F)^2$, the gradient is:

$$-\frac{\partial L}{\partial F} = y - F$$

Which is just the plain residual. That's why MSE-GBM is the easiest to understand — residuals *are* the gradient.

For **log-loss** (classification), it gets more complex — the gradient becomes the difference between true labels and predicted probabilities.

**Final model:**

$$F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$$

---

## 5. Why Each Step — The Intuition

| Step | What's happening | Why we do it |
|---|---|---|
| Initialize with mean | Best single-number guess | Minimizes MSE before any tree |
| Compute residuals | What the model currently gets wrong | Trees learn to fix mistakes, not re-learn what's already captured |
| Fit tree to residuals | A weak learner targets remaining signal | Simple trees avoid overfitting |
| Multiply by η | Shrink the correction | Prevents overshooting; forces more trees = more regularization |
| Sum everything | Accumulate all corrections | Each tree adds a small piece of the truth |

---

## 6. Full Worked Example (By Hand)

### Setup

We want to predict **house price** from **size (sqft)**.

| House | Size (sqft) | Price ($000s) |
|---|---|---|
| A | 1000 | 200 |
| B | 1500 | 300 |
| C | 2000 | 250 |
| D | 2500 | 400 |

Learning rate η = 0.5, using stumps (depth-1 trees).

---

### Step 0 — Initialize

$$F_0(x) = \text{mean}(y) = \frac{200 + 300 + 250 + 400}{4} = 287.5$$

**Why mean?** It minimizes squared error before we've seen any features.

| House | y | F₀ |
|---|---|---|
| A | 200 | 287.5 |
| B | 300 | 287.5 |
| C | 250 | 287.5 |
| D | 400 | 287.5 |

---

### Step 1 — First Tree

**Compute residuals** $r = y - F_0$:

| House | y | F₀ | r₁ |
|---|---|---|---|
| A | 200 | 287.5 | **-87.5** |
| B | 300 | 287.5 | **+12.5** |
| C | 250 | 287.5 | **-37.5** |
| D | 400 | 287.5 | **+112.5** |

**Fit a stump to residuals.** Try split at size = 1750:
- Left (A, B, C — sizes ≤ 1750): mean residual = (-87.5 + 12.5 - 37.5) / 3 = **-37.5**
- Right (D — size > 1750): mean residual = **+112.5**

**Tree 1 predicts:**

| House | h₁(x) |
|---|---|
| A | -37.5 |
| B | -37.5 |
| C | -37.5 |
| D | +112.5 |

**Update predictions** with η = 0.5:

$$F_1(x) = F_0(x) + 0.5 \times h_1(x)$$

| House | F₀ | 0.5 × h₁ | F₁ |
|---|---|---|---|
| A | 287.5 | -18.75 | **268.75** |
| B | 287.5 | -18.75 | **268.75** |
| C | 287.5 | -18.75 | **268.75** |
| D | 287.5 | +56.25 | **343.75** |

**Why shrink by 0.5?** If we fully applied -37.5, we might overcorrect. η=0.5 lets future trees also contribute. It's cautious learning.

---

### Step 2 — Second Tree

**New residuals** $r = y - F_1$:

| House | y | F₁ | r₂ |
|---|---|---|---|
| A | 200 | 268.75 | **-68.75** |
| B | 300 | 268.75 | **+31.25** |
| C | 250 | 268.75 | **-18.75** |
| D | 400 | 343.75 | **+56.25** |

**Fit stump.** Try split at size = 1250:
- Left (A — size ≤ 1250): mean = **-68.75**
- Right (B, C, D — size > 1250): mean = (31.25 - 18.75 + 56.25) / 3 = **+22.92**

**Update:**

$$F_2(x) = F_1(x) + 0.5 \times h_2(x)$$

| House | F₁ | 0.5 × h₂ | F₂ | True y | Error |
|---|---|---|---|---|---|
| A | 268.75 | -34.38 | **234.38** | 200 | -34.38 |
| B | 268.75 | +11.46 | **280.21** | 300 | -19.79 |
| C | 268.75 | +11.46 | **280.21** | 250 | +30.21 |
| D | 343.75 | +11.46 | **355.21** | 400 | -44.79 |

Notice errors are **shrinking** each round. After many trees, we converge toward the true values.

---

## 7. Why Fit Residuals Instead of Predicting y Directly? (The "Aha" Moment)

### Why Not Just Predict y Directly Each Time?

Say y = 300 and your current prediction is 250.

If you train a new tree on **y (300)**, that tree just learns the same thing the first tree already learned. You're re-teaching what's already captured. Wasted effort.

If you train on **the residual (50)**, the tree only learns **what's missing**. The gap. The unexplained part.

### Analogy

You're building a tower to reach 300cm.

- First block gets you to 250cm
- You measure the gap: **50cm remaining**
- You build the next block to be **exactly 50cm** — not 300cm

You're not rebuilding the whole tower each time. You're only building **what's missing**.

If you aimed for 300cm every time, each block would try to be a full tower on its own — and they'd stack into something way over 300.

### What Happens If You Train on y Each Time

```
Tree 1 predicts: 250  (trained on y=300, underfits slightly)
Tree 2 predicts: 260  (trained on y=300 again, learns same thing)
Tree 3 predicts: 255  (same signal, same result)

Sum: 250 + 260 + 255 = 765  ← way off
```

The trees overlap, double-count, and explode.

### What Happens With Residuals

```
Tree 1 predicts: 250  (trained on y=300)
Residual: 50

Tree 2 predicts: 25   (trained on residual=50, with η=0.5)
Residual: 25

Tree 3 predicts: 12.5 (trained on residual=25, with η=0.5)

Sum: 250 + 25 + 12.5 = 287.5  ← converging to 300
```

Each tree adds a smaller and smaller correction. You're **homing in** on the truth.

### The Core Principle

> Each tree should only learn what all previous trees **failed to explain**. Residuals are exactly that — the unexplained signal. Predicting y would re-explain what's already known.

Residuals keep the trees honest. Each one has a specific, non-overlapping job.

---

## 8. Pseudo-Residuals — The Generalization of "What's Left to Fix"

### The Problem With Plain Residuals

In the worked example, residuals were simply $y - F(x)$. That works perfectly for MSE loss. But what if you're predicting **churn** (classification)? Or **revenue with outliers** (where you want MAE not MSE)? You can't just subtract — the "error" concept changes depending on your loss function.

**Pseudo-residuals solve this.** They generalize the idea of "what direction should the next tree push?" to *any* loss function.

### Formal Definition

$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

In plain English: **the negative gradient of the loss with respect to the current prediction.**

It answers: *if I could nudge my prediction $F(x_i)$ by a tiny amount, which direction reduces loss the most?* That direction is what the next tree learns to predict.

### Why Negative?

Gradient descent moves **against** the gradient to minimize loss. The gradient tells you the direction of steepest *increase*. Flip it → steepest *decrease*. The tree learns to push predictions in the loss-reducing direction.

### Three Loss Functions, Three Pseudo-Residuals

**1. Regression — MSE Loss**

$$L = \frac{1}{2}(y - F)^2$$

$$\frac{\partial L}{\partial F} = -(y - F) = F - y$$

$$r_i = -[F - y] = y - F$$

Pseudo-residual = plain residual. They're identical. This is why MSE-GBM is the intuitive entry point.

---

**2. Classification — Log Loss (Binary Cross-Entropy)**

$$L = -[y \log p + (1-y) \log(1-p)]$$

where $p = \sigma(F) = \frac{1}{1 + e^{-F}}$ and $F$ is the raw log-odds score.

$$\frac{\partial L}{\partial F} = p - y$$

$$r_i = y_i - p_i$$

Pseudo-residual = **true label minus predicted probability.** This is elegant — if you predict 0.9 churn and user churns (y=1), residual = 0.1 (small correction needed). If user doesn't churn (y=0), residual = -0.9 (big correction — you were very wrong).

---

**3. Robust Regression — MAE Loss**

$$L = |y - F|$$

$$\frac{\partial L}{\partial F} = -\text{sign}(y - F)$$

$$r_i = \text{sign}(y_i - F_i) = \begin{cases} +1 & \text{if } y > F \\ -1 & \text{if } y < F \end{cases}$$

Pseudo-residual = just **+1 or -1** — only the direction matters, not the magnitude. This makes it robust to outliers. A prediction that's off by 1000 gets the same pseudo-residual as one off by 1. The size of the mistake doesn't dictate how hard you correct.

### Side-by-Side Intuition

| Loss | Pseudo-residual | Sensitive to outliers? | Use when |
|---|---|---|---|
| MSE | $y - F$ | Yes — big errors dominate | Clean targets, symmetric errors |
| Log-loss | $y - p$ | Moderate | Classification (churn, click) |
| MAE | $\text{sign}(y - F)$ | No | Noisy targets, revenue prediction |
| Huber | Blend of MSE + MAE | Partially | Best of both worlds |

### Worked Example — Classification Pseudo-Residuals

Predict churn (1 = churned, 0 = stayed).

| User | y | F₀ (log-odds) | p₀ = σ(F₀) | r = y - p |
|---|---|---|---|---|
| A | 1 | 0.0 | 0.50 | **+0.50** |
| B | 0 | 0.0 | 0.50 | **-0.50** |
| C | 1 | 0.0 | 0.50 | **+0.50** |
| D | 0 | 0.0 | 0.50 | **-0.50** |

Initialize $F_0 = 0$ (log-odds of 50% churn rate). All pseudo-residuals are ±0.5 — makes sense, we have zero information yet, so we're equally wrong about everyone.

After tree 1 fits these pseudo-residuals and we update $F_1$, the probabilities shift. Users predicted as high-churn get pseudo-residuals closer to 0 (we're getting them right). Users we're still wrong about keep large pseudo-residuals — **the next tree focuses its attention there.**

### The Key Insight

> Pseudo-residuals are the model's **confession of ignorance** — expressed in the language of whatever loss function you care about. Each tree reads that confession and tries to fix it.

Plain residuals only work for MSE. Pseudo-residuals work for *any differentiable loss* — that's the entire reason GBM is so general-purpose. Swap the loss function, recompute the gradient, and the rest of the algorithm is identical.

---

## 9. The Gradient Descent Connection

Here's the deep insight most people miss.

Regular gradient descent moves **parameters** in the direction of steepest loss descent:

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

GBM does gradient descent in **function space**. Instead of updating parameters, we update the *prediction function* itself. The "gradient" is the residual vector, and each tree approximates that gradient direction.

This is why GBM works for **any differentiable loss function** — just swap the gradient formula. MSE for regression, log-loss for classification, quantile loss for prediction intervals.

---

## 10. Key Hyperparameters — What They Do & How to Tune

**n_estimators (number of trees)**
- More trees = lower bias, but more compute and potential overfit
- Use early stopping: monitor validation loss, stop when it plateaus
- Typical range: 100–3000

**learning_rate (η)**
- Lower η → needs more trees, but generalizes better
- Rule of thumb: η × n_estimators ≈ constant. Half η → double trees
- Typical: 0.01–0.1

**max_depth**
- Controls tree complexity. Depth 3–5 is almost always right for GBM
- GBM works best with *weak* learners. Don't go deep
- Unlike Random Forest, deeper ≠ better here

**subsample**
- Fraction of rows sampled per tree (stochastic GBM)
- Adds randomness → reduces variance, speeds training
- Typical: 0.6–0.9

**min_samples_leaf / min_child_weight**
- Minimum samples in a leaf node
- Prevents trees from splitting on tiny groups → regularization

**Tuning order:** n_estimators + learning_rate first (they interact), then max_depth, then subsample.

---

## 11. η and n_estimators — They're Coupled

### The Intuition

η controls how big each step is. n_estimators controls how many steps you take.

If you take **small steps** (low η), you need **more steps** to reach the same place. If you take **big steps** (high η), you get there faster but might overstep.

They're not independent dials. They multiply together.

### The Math

Total "movement" of your model ≈ η × n_estimators

```
η=0.1,  n=100  →  total movement ≈ 10
η=0.05, n=200  →  total movement ≈ 10  (same)
η=0.01, n=100  →  total movement ≈ 1   (underfits)
η=0.1,  n=1000 →  total movement ≈ 100 (overfits)
```

Same η × n product → roughly same model complexity.

### Why Lower η Is Better (With More Trees)

Lower η forces GBM to take more, smaller corrections. Each tree specializes less aggressively. The ensemble explores more of the error landscape before committing.

It's like the difference between one surgeon making one big cut vs ten surgeons making ten small precise cuts. More cuts, more control.

> **Rule of thumb:** Start with η=0.1, n=100 as baseline. Then halve η and double n — if val loss improves, keep going smaller. Use early stopping to find the right n automatically.

### How To Tune Them Together

```
Step 1: Set η=0.1, n_estimators=1000, use early stopping
        → early stopping finds optimal n (say 200)

Step 2: Set η=0.05, n_estimators=2000, use early stopping
        → optimal n now ~400, check if val loss improved

Step 3: Set η=0.01, n_estimators=5000
        → slower but often best val loss
```

Never tune η and n separately. Always use early stopping so n finds itself.

---

## 12. Tree Depth — Is a Stump the Only Option?

No — but stumps are the **extreme end** of a spectrum.

### The Spectrum

```
depth=1 (stump)    → one split, two leaves
depth=3            → up to 8 leaves  ✅ GBM sweet spot
depth=5            → up to 32 leaves
depth=10+          → nearly a full tree ❌ GBM breaks down
```

### Why Stumps Work in AdaBoost But Not Always GBM

AdaBoost was designed around stumps — each stump captures one feature at a time, and you need hundreds to build up complexity.

GBM is more powerful per tree, so it can afford **slightly deeper trees** and needs fewer of them. Depth 3-5 is the real-world default.

### What Depth Controls

A depth-1 stump can only learn:
> "if feature A > threshold, go right"

A depth-3 tree can learn:
> "if feature A > x **AND** feature B > y **AND** feature C > z"

That's a **3-way interaction**. Real product data has interactions everywhere — time of day AND device type AND user tenure all jointly affecting churn. A stump can't capture that in one tree.

### The Trade-off

| Depth | Captures | Risk |
|---|---|---|
| 1 (stump) | Single features only | Needs many more trees |
| 3 | 2-3 way interactions | Sweet spot |
| 5 | Complex interactions | Watch for overfit |
| 8+ | Near-memorization per tree | GBM philosophy breaks |

### Why GBM Philosophy Breaks With Deep Trees

GBM works because each tree is **weak** — it only explains a small piece of the residual. Future trees have meaningful residuals left to learn from.

If tree 1 is deep enough to fit most of the residual by itself, trees 2-N have almost nothing left. You've essentially built one strong tree disguised as boosting.

> **Rule:** Keep trees weak enough that residuals stay meaningful across many rounds. Depth 3-5 almost always right. Stumps only if you have very simple data or need extreme interpretability.

---

## 13. Regularization in GBM — The Four Levers

### max_depth — Controls Tree Complexity

Shallow trees = weak learners = what GBM wants.

```
depth=1  (stump):  can only capture one feature at a time
depth=3:           can capture 2-way interactions  ✅ sweet spot
depth=5:           captures complex interactions, risks overfit
depth=8+:          each tree becomes too powerful, GBM breaks down
```

**Why GBM hates deep trees:** A deep tree already fits the residuals well on its own. Future trees have nothing meaningful left to learn. You get a powerful tree 1 and useless trees 2-N.

### min_samples_leaf — Controls Leaf Purity

Minimum number of training samples required in a leaf node.

Low value → tree can split on very small groups → fits noise.

```
min_samples_leaf=1:   fits every outlier
min_samples_leaf=20:  needs 20 samples to make a split → more robust
```

Think of it as: "I won't make a rule based on fewer than 20 examples." Prevents the model from memorizing edge cases.

### Subsampling (subsample) — Stochastic GBM

Each tree is trained on a **random fraction** of the training data.

```
subsample=1.0:  use all data (standard GBM)
subsample=0.8:  each tree sees 80% of rows, randomly sampled
```

Two benefits:
- Adds randomness → trees are less correlated → better ensemble
- Faster training

It's similar to the randomness in Random Forest, but applied to boosting. Called **Stochastic GBM**.

> Typical value: 0.6–0.8. Below 0.5 and you're throwing away too much data per tree.

### The Four Levers Together

| Lever | What it controls | Overfit signal to fix it |
|---|---|---|
| max_depth | Tree complexity | Reduce from 5→3 |
| min_samples_leaf | Leaf node purity | Increase from 1→20 |
| subsample | Row sampling per tree | Reduce from 1.0→0.8 |
| η + n_estimators | Step size + steps | Lower η, use early stopping |

> **Interview answer:** "I treat regularization in GBM as four dials. I fix max_depth at 3-5 first, set subsample to 0.8, then tune η and n_estimators together with early stopping. min_samples_leaf I tune last if I'm still seeing overfit."

---

## 14. Early Stopping

### The Problem Without It

Every tree you add reduces **training error**. Always. A new tree can always find some residual to fit. So training loss never tells you to stop — it just keeps dropping forever.

But validation loss tells the truth.

### What's Happening Tree by Tree

```
Tree 1:   train loss 0.45   val loss 0.44  ✅ still learning
Tree 10:  train loss 0.30   val loss 0.29  ✅ still learning  
Tree 50:  train loss 0.18   val loss 0.20  ✅ ok
Tree 100: train loss 0.10   val loss 0.23  ⚠️ gap opening
Tree 150: train loss 0.05   val loss 0.28  ❌ overfitting
Tree 200: train loss 0.02   val loss 0.31  ❌ getting worse
```

Training loss always goes down. Validation loss bottoms out then rises. The gap is the model memorizing residuals that are just **noise**, not signal.

Early stopping says: monitor val loss, stop when it hasn't improved in N rounds.

### Why It's Subtle in GBM Specifically

In GBM, later trees fit **smaller and smaller residuals**. Eventually those residuals are pure noise — random fluctuations in the training data. The model starts fitting noise patterns that don't generalize.

The learning rate makes this worse — low η means you need many trees to overfit, so you don't notice it happening until late.

> **Interview answer:** "I always use early stopping with a held-out validation set. I set n_estimators high (say 2000) and let early stopping find the right number. Without it you're guessing."

---

## 15. Feature Importance — And Why It Lies

### How GBM Calculates It

For each feature, sum up the **impurity reduction** (improvement in loss) across every split that used that feature, across all trees.

Feature with highest total impurity reduction = most important.

Simple. But wrong in one key situation.

### Why It Lies With Correlated Features

Say you have two features: `time_on_app` and `sessions_per_week`. They're highly correlated — basically measuring the same thing.

GBM splits on one of them first (say `time_on_app`). Now `sessions_per_week` explains very little additional variance — `time_on_app` already captured it. So it gets low importance.

But if you removed `time_on_app`, `sessions_per_week` would suddenly become very important.

**Neither importance score is trustworthy.** The credit is being arbitrarily assigned based on which feature the tree happened to split on first.

### The Fix — SHAP

SHAP assigns credit by asking: *"for this specific prediction, how much did each feature contribute?"* It accounts for feature interactions and correlations properly.

```
GBM importance:   time_on_app=0.45, sessions=0.03  ← misleading
SHAP:             time_on_app=0.24, sessions=0.22  ← true picture
```

> **Interview answer:** "GBM feature importance is a fast first signal but I don't trust it for correlated features. I use SHAP when I need to explain the model to stakeholders or make feature selection decisions."

---

## 16. Missing Values

### A) How GBM Handles Them Internally

When GBM hits a missing value during a split, it tries sending that sample **both left and right**, sees which direction reduces loss more, and permanently assigns that direction as the default for missing values of that feature.

```
Split: time_on_app > 10 mins?

User A: time_on_app = missing
→ try sending left: loss = 0.42
→ try sending right: loss = 0.38
→ assign right as default for missing ✅
```

It's learning *where missing values tend to belong* from the data itself. Not guessing. Not breaking.

This is why **GBM is one of the few algorithms that handles missings natively** — you don't always need to impute.

### But Native Handling Isn't Always Enough

GBM's native handling assumes **missing at random** — the missingness itself carries no signal.

But sometimes **missing IS the signal.**

```
purchase_amount = null  →  user never purchased  →  strong churn signal
```

If you just let GBM handle it natively, it learns *what to predict given the value is missing* — but it doesn't explicitly learn that *the act of being missing* matters.

**Fix:** Add a binary indicator feature.

```python
df['purchase_amount_missing'] = df['purchase_amount'].isnull().astype(int)
```

Now GBM can use both the value AND the missingness pattern as separate signals.

### B) Preprocessing Missings Before GBM

When should you bother imputing if GBM handles it natively?

| Situation | What to do |
|---|---|
| Missing at random, <20% missing | Let GBM handle it natively |
| Missing IS the signal | Add indicator column + let GBM handle |
| >50% missing | Consider dropping the feature or flagging it |
| Missing due to pipeline bug | Fix the pipeline — don't impute garbage |

> **Never** do mean/median imputation blindly before GBM. You're destroying the signal that missingness carries, and GBM didn't need the imputation anyway.

---

## 17. Outliers

### A) How GBM Handles Them Internally

This depends entirely on your **loss function** — and this is what interviewers want to hear.

**MSE Loss — Outliers Dominate**

Residual for an outlier is huge. Pseudo-residual = $y - F$ = massive number.

GBM sees a giant residual and dedicates significant tree capacity to fitting it. Multiple trees end up specializing on that one outlier. The rest of your predictions suffer.

```
Normal user LTV: 50-200
Whale user LTV:  50,000

Residual for whale after tree 1: ~49,800
Next 5 trees: almost entirely focused on that one whale
```

**MAE Loss — Outliers Ignored**

Pseudo-residual = sign(y - F) = just +1 or -1.

The whale gets the same gradient signal as everyone else. GBM doesn't care how wrong it is — only which direction to correct.

**Outliers have zero extra influence.**

**Huber Loss — Best of Both**

Behaves like MSE for small errors (precise corrections) and like MAE for large errors (ignores outliers).

```
If |residual| < δ:  use MSE gradient  (normal samples)
If |residual| > δ:  use MAE gradient  (outliers)
```

δ is a threshold you set. This is usually the right answer in production.

### B) Preprocessing Outliers Before GBM

Even though you can handle outliers via loss function, sometimes you want to handle them in the data itself.

**When to clip**

If an outlier is a **data error** (someone entered age=999), clip it before it corrupts your trees.

```python
df['age'] = df['age'].clip(upper=100)
```

If it's **real signal** (a genuine whale user), don't clip — let Huber loss handle it.

**When to log-transform**

Revenue, LTV, counts — naturally right-skewed with extreme values. Log transform compresses the scale so residuals are more balanced.

```python
df['revenue'] = np.log1p(df['revenue'])
```

GBM then splits on log-revenue instead of raw revenue. Splits become more meaningful across the distribution instead of obsessing over the top 0.1%.

**When to do nothing**

If you're using MAE or Huber loss, and outliers are real signal you don't want to ignore — just leave them. The loss function is your protection.

### The Decision Framework

```
Is the outlier a data error?
  → Yes: clip or remove it

Is the outlier real signal?
  → Do you want to predict it accurately?
      → Yes: keep it, use Huber loss
      → No (it'll distort everything else): log-transform or clip

Are you using MSE?
  → Outliers will hurt you — switch to Huber or log-transform target
```

### The One-Line Versions For Interviews

> **Missing values:** GBM handles them natively by learning which branch missing values belong to — but add an indicator column if missingness itself is a signal.

> **Outliers:** MSE loss amplifies them, MAE ignores them, Huber splits the difference. Choose your loss function before reaching for preprocessing.

---

## 18. Leakage in GBM on Time Series

### Why Standard Train/Test Split Breaks

Say you have daily user activity data from Jan–Dec. You randomly split 80/20.

Your training set contains **December data**. Your test set contains **March data**.

The model trained on December learns patterns from the future — seasonality, trends, events that haven't happened yet in March. It looks great on val. It fails in production.

**GBM is especially dangerous here** because it fits residuals so aggressively — it'll happily learn future patterns if you let it.

---

## 19. GBM vs Random Forest — When to Use Which

| | GBM | Random Forest |
|---|---|---|
| Accuracy on tabular data | Usually higher | Slightly lower |
| Training speed | Slower (sequential) | Faster (parallel) |
| Tuning required | More sensitive | More forgiving |
| Overfitting risk | Higher without tuning | Lower naturally |
| Interpretability | Similar (both tree-based) | Similar |
| Noisy data | More sensitive | More robust |

**Use GBM when:** you're optimizing for accuracy on a Kaggle-style problem or structured product data (clicks, conversions, churn).

**Use Random Forest when:** you need a quick reliable baseline, data is noisy, or you have limited time to tune.

---

## 20. GBM Variants You Must Know

**XGBoost** — adds L1/L2 regularization on tree weights, second-order gradients (Newton's method), and is highly optimized. Industry standard.

**LightGBM** — grows trees *leaf-wise* instead of level-wise, much faster on large datasets, handles high-cardinality categoricals natively.

**CatBoost** — handles categorical features automatically, uses ordered boosting to prevent target leakage during training.

For interviews: "I'd default to LightGBM for speed on large data, XGBoost when I need fine control over regularization, CatBoost when the data has many categoricals."

---

## 21. Interview Questions — With Model Answers

**Q1: What is the difference between bagging and boosting?**

Bagging (Random Forest) trains trees *in parallel* on random subsets of data and averages predictions — reduces variance. Boosting trains trees *sequentially*, each correcting the last — reduces bias. GBM is boosting.

---

**Q2: Why do we fit trees to residuals and not the original target?**

Because the current model already explains part of the signal. Residuals are what's *unexplained*. Fitting to residuals means each new tree only needs to learn what we don't yet know. Fitting to y would re-learn the same patterns already captured.

---

**Q3: What happens if you set learning rate = 1?**

Each tree's output is applied in full. The model may overshoot corrections and oscillate or overfit. A learning rate of 1 with enough trees essentially memorizes the training data. You lose the regularization benefit of slow learning.

---

**Q4: How does GBM handle overfitting?**

Three main levers: lower learning rate (more conservative updates), reduce n_estimators or use early stopping, and add subsampling (stochastic GBM). max_depth and min_child_weight also constrain tree complexity directly.

---

**Q5: A product feature predicting user churn is underperforming. You're using GBM. How do you debug it?**

First check data quality — class imbalance, leakage, missing values. Then check feature importance — are the expected signals actually being used? Look at learning curves (train vs val loss) — if val loss plateaus early, you're overfitting; if both are high, underfitting. Try subsampling if noisy. Finally, use SHAP to understand *how* the model uses features and catch counterintuitive patterns.

---

**Q6: Why are residuals the negative gradient of MSE loss?**

MSE loss: $L = \frac{1}{2}(y - F)^2$. Gradient w.r.t. F: $\frac{\partial L}{\partial F} = -(y - F)$. Negative gradient = $y - F$ = residual. So for MSE, fitting to residuals *is* gradient descent. For other losses, the "pseudo-residuals" are just the corresponding negative gradients.

---

**Q7: When would you NOT use GBM?**

When you need fast iteration and a quick baseline (use linear model or RF). When data is very small (<500 rows) — GBM may overfit. When you need strict interpretability for regulators (logistic regression is easier to explain to non-technical stakeholders). When latency at inference is critical and you can't afford to run 500 trees.

---

**Q8: How do you handle class imbalance in GBM?**

XGBoost has `scale_pos_weight` (ratio of negatives to positives). You can also oversample the minority class (SMOTE), undersample majority, or change the evaluation metric from accuracy to AUC or F1. In a product context, also ask whether imbalance is real or a data pipeline bug.

---

## 22. Company-Specific Interview Questions (Reported)

Here are the actual reported questions, with exactly what Google and Apple are testing beneath each one.

### Google GBM Questions

**Q: "What is the difference between bagging and boosting?"**

Google explicitly asks candidates to discuss ensemble methods and when they'd use Random Forests (bagging) vs. GBM (boosting), including strengths and weaknesses in various situations.

**What they want to hear:**
- Bagging = parallel, independent trees, reduces **variance**
- Boosting = sequential, each tree fixes the last, reduces **bias**
- GBM can overfit; RF is more forgiving
- Tie it to a product scenario: "For a noisy engagement signal, I'd start with RF. For a clean churn label where I need max accuracy, GBM."

---

**Q: "Explain gradient descent and how it connects to GBM."**

Google asks candidates to explain gradient descent as an optimization technique that minimizes a loss function by iteratively adjusting parameters in the direction of steepest decrease — and separately how GBM builds trees sequentially with each new tree focusing on correcting errors of the previous one.

**The trap:** Most people explain them separately. Google wants you to connect them — GBM *is* gradient descent but in function space. The trees approximate the gradient, not parameter updates.

---

**Q: "You need to forecast per-region request volume for Google Search to set load-shedding thresholds. How would you approach it?"**

A real Google interview problem involves forecasting Search traffic which has strong daily seasonality plus sudden step-changes from launches — the suggested approach uses gradient-boosted trees with lag features, seasonality and calendar features, plus event flags for promotions, evaluated with rolling-origin backtests.

**What they're testing:** Can you apply GBM to time series? Key points: lag features as inputs, don't use standard train/test split (use rolling origin), mention quantile loss if you want to impress.

---

### Apple GBM Questions

**Q: "Explain XGBoost. How does it handle bias and variance?"**

Apple directly asks about XGBoost as an ensemble algorithm that leverages gradient boosting — they want candidates to explain how boosting reduces bias by focusing on weak predictions iteratively, while taking a weighted average of many weak models keeps variance low.

**What they want to hear:** Don't just say "it's fast." Say: each tree targets residual error (bias reduction), but because each tree is shallow and we shrink with η, no single tree dominates (variance control). L1/L2 regularization in XGBoost adds another variance-control layer vanilla GBM lacks.

---

**Q (Apple Intern, reported): "You're in a hackathon — GBM vs neural net, limited data, 6 hours. What do you pick and why?"**

A reported Apple interview scenario involves choosing between gradient boosting and a neural network under time and data constraints — GBM won with RMSE 15% lower, and the interviewer valued the candidate letting data drive the decision rather than defaulting to the more complex model.

**The answer they love:** GBM. Small data + limited time = GBM wins almost always. Neural nets need large data and lots of tuning. Say: "I'd run a quick 2-hour proof of concept with both using default params and let validation performance decide."

---

### The Pattern Across Both Companies

| Company | Focus | What they're really testing |
|---|---|---|
| Google | Scale + systems thinking | Can you apply GBM to real infra problems? Do you know the math cold? |
| Apple | Practical trade-offs | Would you reach for GBM at the right moment, not just because it's powerful? |

Both companies push you to **justify why GBM over alternatives** — not just recite how it works. Always anchor your answer to data size, noise level, interpretability needs, and time constraints.

---

## 23. The Mental Model to Carry Forever

> GBM is a **committee of specialists**. The first tree is a generalist who takes a rough stab. Each subsequent tree is a specialist who only looks at what the generalist got wrong, and tries to fix that. The learning rate is how much you trust each specialist. After enough specialists, you have a very accurate ensemble.
