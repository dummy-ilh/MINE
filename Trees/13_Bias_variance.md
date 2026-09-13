Let's build this up piece by piece, starting from a single model and then adding more.

## Step 1: One model, one number to explain (Bias² + Variance + Noise)

Any model's error at a point splits into three additive pieces:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

- **Bias** = if you retrained this model on every possible dataset and averaged all those predictions, how far off is that average from the truth? This is a property of the *model type* (e.g. "shallow trees underfit curvy data") — it doesn't depend on which one training set you happened to get.
- **Variance** = how much does any *one* trained model wobble around that average, just because of which specific training set it saw?
- **Noise** = randomness in the world nobody can predict. Ignore it from here on — no model touches it.

## Step 2: Now average $M$ models together

This is what bagging/Random Forest do — train $M$ versions of the model (on different bootstrap samples), average their predictions.

### Why bias doesn't change

$$\mathbb{E}[\text{average of } M \text{ models}] = \text{average of } \mathbb{E}[\text{each model}] = \mathbb{E}[\text{one model}]$$

This is just arithmetic: the average of five numbers that are each "wrong by 3 on average" is *also* wrong by 3 on average. Averaging can't un-bias something — if every single model has the same systematic lean, their average has that exact same lean. **Bias is a property of *how* the model was trained, not of which random draw it got — averaging doesn't touch that.**

### Why variance *does* change — this is where correlation comes in

If the $M$ models were **totally independent** (never true in practice, but imagine it), their random wobbles would partially cancel out when averaged, the same way flipping 100 coins averages closer to 50% than flipping 1 coin does. More models → more cancellation → variance shrinks toward zero.

But real bootstrap-trained trees are **not independent** — they're all trained on overlapping data from the same original dataset, so their wobbles move *together* to some degree. That "move together" amount is the correlation $\rho$ (0 = fully independent, 1 = identical twins).

## Step 3: The formula, explained term by term

$$\text{Var(average)} = \underbrace{\rho\sigma^2}_{\text{floor}} + \underbrace{\frac{(1-\rho)\sigma^2}{M}}_{\text{shrinks with more models}}$$

Think of it as **two separate pools of wobble**:

| Piece | What it represents | Can $M$ (more trees) fix it? |
|---|---|---|
| $\rho\sigma^2$ | The part of each tree's wobble that's *shared* — every tree wobbles the same way because they're all built from the same base dataset | **No.** This is a hard floor. |
| $\frac{(1-\rho)\sigma^2}{M}$ | The part of each tree's wobble that's *unique* to that tree | **Yes.** Averaging more trees cancels more of this out, shrinking it toward 0. |

**Intuition for why $\rho$ can't be beaten by $M$:** imagine 1000 people each estimating a jar of jellybeans, but they all glanced at the jar from the exact same angle in the exact same lighting. Averaging their guesses helps a little (kills the part where each person second-guessed differently), but if they're all systematically fooled the same way by that one angle, no amount of averaging removes *that* shared error. Only getting people to look from genuinely different angles (lowering $\rho$) fixes it.

## Step 4: Sanity-check the extremes

- **$\rho = 1$** (all trees identical): formula becomes $1\cdot\sigma^2 + 0 = \sigma^2$. Averaging clones does literally nothing — you're just computing the same number $M$ times.
- **$\rho = 0$** (all trees fully independent): formula becomes $0 + \frac{\sigma^2}{M} \to 0$ as $M$ grows. Variance vanishes completely — the dream scenario, never fully achievable in practice.
- **Real life, $0 < \rho < 1$**: some benefit from $M$, but it caps out at $\rho\sigma^2$ no matter how large $M$ gets.

## Step 5: The two knobs, and how to actually "move" each one

| Knob | What it does | How you turn it, in practice |
|---|---|---|
| **$M$ (number of trees)** | Shrinks only the second term, `(1-ρ)σ²/M` | `n_estimators` in sklearn. Free to increase (never overfits), but returns diminish fast — most of the benefit happens in the first ~10-50 trees, then it flattens near the floor. |
| **$\rho$ (correlation between trees)** | Lowers the floor itself — the *only* knob that can push you below the $M\to\infty$ limit | This is the entire reason bagging uses bootstrap sampling (different data → somewhat different trees) and why Random Forest goes further, randomly hiding most features at every single split (`max_features`), forcing trees to disagree more structurally. |
| **$\sigma^2$ (how wobbly a single tree is)** | Scales both terms — smaller $\sigma^2$ shrinks everything | You'd do this by using a less unstable base learner (e.g. shallower tree) — but **careful**: a shallower tree usually has *higher bias*, so you're trading one error source for another, not getting a free lunch. |

## Worked numbers, so it's concrete

$\sigma^2 = 4.0$ (one tree's own variance), $M=200$ trees.

**Case A — high correlation, $\rho=0.5$:**
$$0.5 \times 4.0 + \frac{0.5\times4.0}{200} = 2.0 + 0.01 = 2.01$$

**Case B — lower correlation, $\rho=0.25$ (this is what Random Forest buys you over plain bagging):**
$$0.25\times4.0 + \frac{0.75\times4.0}{200} = 1.0 + 0.015 = 1.015$$

Same $M$, same $\sigma^2$ — just cutting $\rho$ in half roughly **halves the total variance**. Compare that to what happens if you leave $\rho=0.5$ fixed and just crank $M$ from 200 to 2000:
$$0.5\times4.0 + \frac{0.5\times4.0}{2000} = 2.0 + 0.001 = 2.001$$

Barely moved. **That's the core lesson of the formula: once $M$ is moderately large, adding more trees is nearly worthless — the only lever left that still does real work is lowering $\rho$.**

## One-line summary to keep

> Bias is stuck no matter how many models you average — it's baked into the model type, not the training draw. Variance splits into a floor set by correlation (which only decorrelating the models can lower) and a shrinking piece that more models can cancel out — but that shrinking piece runs out of gas fast, so past a certain point, the only way to keep improving is to make your models disagree with each other *more*, not to make more of them.
