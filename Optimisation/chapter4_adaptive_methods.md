# Chapter 4: Adaptive Methods — AdaGrad, RMSProp, Adam — Interview Notes (Beginner-Friendly)

This chapter builds on Chapters 2 (gradient descent) and 3 (momentum). Same style: plain English first, formulas second, every formula translated back into words.

---

## 1. The Problem Adaptive Methods Are Trying to Solve

So far, every method we've covered (plain GD, momentum, Nesterov) uses **the exact same learning rate $\eta$ for every single weight in the model.** That sounds reasonable, but real loss landscapes rarely treat all directions equally.

Imagine a loss landscape shaped like a long, stretched-out valley — very steep in one direction, but very gently sloped in another (picture an elongated canyon rather than a round bowl):

![elongated valley with steep and shallow directions](contour plot narrow steep valley loss surface)

If you use one learning rate for both directions:
- Make it small enough to be safe on the **steep** direction (avoid overshooting/bouncing, as in Chapter 2) → progress on the **shallow** direction becomes painfully slow, since the same small step barely moves you there.
- Make it big enough to make good progress on the **shallow** direction → you'll overshoot and bounce around on the **steep** direction.

**The core idea of adaptive methods, in one sentence:** *give every individual weight its own personal, automatically-adjusted learning rate, based on that weight's own history of gradients — instead of forcing one global learning rate onto every direction.*

This matters enormously in real models, where different weights genuinely behave very differently — e.g., a weight connected to a rare word in a vocabulary gets gradient updates rarely and should take bigger steps when it does; a weight connected to a common feature gets updated constantly and should be more cautious.

---

## 2. AdaGrad — "Slow Down Weights That Have Already Moved a Lot"

**Plain-language idea:** for each individual weight, keep a running total of how big its gradients have been *so far* (squared, so both positive and negative gradients count as "movement," not canceling out). Weights that have already seen a lot of movement get their effective learning rate **shrunk** — the idea being "you've already made a lot of progress/noise in this direction, so be more careful/precise now." Weights that have barely moved keep something close to the full learning rate.

In symbols, per weight $w_i$:
$$G_i \mathrel{+}= (\nabla f(w_i))^2$$
$$w_i \leftarrow w_i - \frac{\eta}{\sqrt{G_i} + \epsilon} \cdot \nabla f(w_i)$$

Translating every symbol:
- $G_i$ = a running sum, *just for this one weight*, of every squared gradient it's ever received. It only ever grows (never shrinks), since you're always adding a squared — therefore non-negative — number.
- $\sqrt{G_i}$ = the square root of that running total, used to shrink the effective step size — the bigger the accumulated history, the more this divides down the step.
- $\epsilon$ (epsilon) = a tiny constant (like $10^{-8}$) purely to prevent dividing by zero on the very first step, when $G_i$ is still $0$. It has no other conceptual role — you can basically ignore it except to note it exists.
- Everything else ($\eta$, $\nabla f$) means exactly what it meant in Chapter 2.

### 2.1 A simple numeric example

Say weight A gets gradients $3, 3, 3$ over three steps (a consistently active weight), and weight B gets gradients $0.1, 0.1, 0.1$ (a rarely-active weight), with base learning rate $\eta = 1$ and ignoring $\epsilon$ for simplicity.

**Weight A:**
- Step 1: $G_A = 9$. Effective step size $= \frac{1}{\sqrt 9} = 0.333$.
- Step 2: $G_A = 9+9=18$. Effective step size $=\frac{1}{\sqrt{18}} = 0.236$.
- Step 3: $G_A = 27$. Effective step size $= \frac{1}{\sqrt{27}} = 0.192$.

**Weight B:**
- Step 1: $G_B = 0.01$. Effective step size $= \frac{1}{\sqrt{0.01}} = 10$.
- Step 2: $G_B = 0.02$. Effective step size $=\frac{1}{\sqrt{0.02}} = 7.07$.
- Step 3: $G_B = 0.03$. Effective step size $= \frac{1}{\sqrt{0.03}} = 5.77$.

Notice: weight B (which barely moves each step in raw terms) ends up taking a **much bigger relative step** than weight A. That's AdaGrad's whole point — automatically compensating for weights that otherwise wouldn't get much attention.

### 2.2 AdaGrad's Big Flaw

Because $G_i$ only ever accumulates (never shrinks — it's a running sum of squares, always growing), the effective learning rate for every weight **monotonically shrinks forever** as training continues. Eventually, $G_i$ gets so large that the effective step size becomes vanishingly small, and the model **effectively stops learning**, even if it hasn't actually reached a good solution yet. This is the single most important thing to know about AdaGrad for an interview: it works well early on, but decays too aggressively over long training runs.

---

## 3. RMSProp — Fixing AdaGrad's "Decays to Zero" Problem

**Plain-language fix:** instead of keeping a running sum of *every* squared gradient ever seen (which only grows), keep a running **average** that gives more weight to *recent* gradients and gradually "forgets" old ones. This way the accumulated value doesn't just grow forever — it can go up or down depending on recent behavior.

This is done using an **exponential moving average** — a weighted average where recent values count more and older values fade out gradually (think of it like a "leaky memory" — new information mostly overwrites old information, rather than just piling on top of it forever).

In symbols, per weight:
$$G_i \leftarrow \gamma \cdot G_i + (1-\gamma)(\nabla f(w_i))^2$$
$$w_i \leftarrow w_i - \frac{\eta}{\sqrt{G_i}+\epsilon}\cdot \nabla f(w_i)$$

Translating: $\gamma$ (gamma, typically $0.9$ or $0.99$) controls how much of the *old* running average you keep vs. how much weight the *newest* squared gradient gets. This is structurally identical to AdaGrad's update rule — the only change is swapping "keep adding forever" for "blend old and new, with old fading out over time." That one change is enough to fix the "learning grinds to a halt" problem, because $G_i$ now settles into a stable range reflecting *recent* gradient sizes, rather than growing without bound.

---

## 4. Adam — Combining Momentum (Chapter 3) With RMSProp's Per-Weight Scaling

**Plain-language idea:** Adam ("Adaptive Moment Estimation") is just "take the momentum idea from Chapter 3 (smooth out the direction using a running average of gradients) **and** the RMSProp idea from Section 3 (scale each weight's step by a running average of its squared gradients) — and use both at the same time."

Adam tracks **two** running averages per weight:
1. A running average of the (plain, not squared) gradients — this is exactly the momentum "velocity" $v$ from Chapter 3, telling you the *smoothed direction*.
2. A running average of the *squared* gradients — this is exactly RMSProp's $G$ from Section 3, telling you the *typical recent magnitude*, used to scale the step size per-weight.

In symbols (using Adam's standard names $m$ for the first, $v$ for the second — note this "$v$" plays the same role as momentum's velocity, just renamed in the Adam paper's notation):

$$m \leftarrow \beta_1 m + (1-\beta_1)\nabla f(w)$$
$$v \leftarrow \beta_2 v + (1-\beta_2)(\nabla f(w))^2$$

These two lines are literally just "momentum's smoothing" and "RMSProp's smoothing," done side by side, with two separate decay rates $\beta_1$ (commonly $0.9$) and $\beta_2$ (commonly $0.999$ — note this is deliberately much closer to 1, meaning the squared-gradient average changes more slowly/cautiously than the direction average).

### 4.1 The Bias-Correction Step (the one genuinely new piece)

There's one wrinkle Adam adds that momentum/RMSProp alone don't need: **both $m$ and $v$ start at zero**, and because the update formulas above are weighted averages that lean partly on that zero starting value, the very first several steps are **artificially biased toward zero** — they haven't had enough steps yet to "fill up" with real gradient information.

**Plain-language fix:** divide by a correction factor that's large at the very start (to compensate for the artificial bias toward zero) and fades to $1$ (no correction needed) after enough steps have passed:

$$\hat m = \frac{m}{1-\beta_1^t}, \qquad \hat v = \frac{v}{1-\beta_2^t}$$

Here $t$ is which step number you're on. At $t=1$, $1-\beta_1^t$ is small, so you divide by a small number — a big correction. As $t$ grows, $\beta_1^t \to 0$, so $1-\beta_1^t \to 1$, and the correction fades away to "no adjustment needed."

### 4.2 Final Adam Update

$$w \leftarrow w - \frac{\eta}{\sqrt{\hat v}+\epsilon}\cdot \hat m$$

In words: *take the bias-corrected smoothed direction ($\hat m$, momentum's contribution), and scale its step size per-weight using the bias-corrected smoothed squared-gradient history ($\hat v$, RMSProp's contribution).* Every single piece of this formula is something you've already learned in Chapter 3 or earlier in this chapter — Adam's real contribution is combining them cleanly, plus the bias-correction fix.

### 4.3 A simple numeric example (one weight, first 2 steps)

Say $\beta_1=0.9$, $\beta_2=0.999$, $\eta=0.1$, $\epsilon\approx 0$, starting $m_0=0, v_0=0$, and the gradient at step 1 is $g_1 = 4$.

**Step 1:**
- $m_1 = 0.9(0) + 0.1(4) = 0.4$
- $v_1 = 0.999(0)+0.001(16) = 0.016$
- Bias-corrected: $\hat m_1 = \frac{0.4}{1-0.9^1} = \frac{0.4}{0.1} = 4$ ; $\hat v_1 = \frac{0.016}{1-0.999^1} = \frac{0.016}{0.001} = 16$
- Update: $w \leftarrow w - \frac{0.1}{\sqrt{16}}\times 4 = w - \frac{0.1}{4}\times 4 = w - 0.1$

Notice something neat: at step 1 with a single gradient, the bias correction perfectly recovers $\hat m_1 = g_1 = 4$ and $\hat v_1 = g_1^2 = 16$ exactly — the correction is doing exactly its job of "undoing" the zero-initialization bias so the very first step behaves sensibly rather than being artificially shrunk toward zero.

---

## 5. AdamW — One Important Practical Correction

**The problem:** a common technique called "weight decay" (a form of regularization — gently shrinking weights toward zero each step to reduce overfitting, similar in spirit to the L2 regularization idea) was traditionally implemented by just adding a penalty term into the loss function before computing gradients. But because Adam *rescales* every gradient per-weight (via $\hat v$), that rescaling **also distorts the weight-decay penalty** in an unintended way — the decay ends up interacting with Adam's adaptive scaling rather than behaving as a clean, separate "shrink everything a little" force.

**AdamW's fix, in plain English:** apply the weight-decay shrinkage **directly to the weights**, as a separate step, completely outside of and after Adam's gradient-based update — rather than folding it into the loss/gradient where Adam's per-weight rescaling would distort it. "Decoupled" weight decay just means: keep the two forces (Adam's adaptive gradient step, and weight decay's gentle shrinkage) mathematically separate instead of letting them tangle together.

This is a genuinely popular interview trick question: **"What's the difference between Adam with L2 regularization and AdamW?"** — the answer is exactly this decoupling, and AdamW is now the standard default in most modern deep learning training (especially for transformers).

---

## 6. Quick Comparison Table

| Method | Per-weight learning rate? | Uses momentum (direction smoothing)? | Main weakness |
|---|---|---|---|
| Plain GD (Ch. 2) | No — one rate for everyone | No | Struggles with ravines, uniform treatment of all weights |
| Momentum / NAG (Ch. 3) | No | Yes | Still one global learning rate for all weights |
| AdaGrad | Yes | No | Learning rate shrinks to ~zero over long training |
| RMSProp | Yes | No | Fixes AdaGrad's decay problem, but still no direction-smoothing |
| Adam | Yes | Yes | Slightly more hyperparameters to tune; sometimes generalizes worse than well-tuned SGD+momentum |
| AdamW | Yes | Yes | Same as Adam, but with correctly decoupled weight decay — today's default choice |

---

## 7. Common Interview Follow-Ups

**"Why does $\beta_2$ default to 0.999 while $\beta_1$ defaults to 0.9?"** The squared-gradient average ($v$) is meant to capture a *stable estimate of typical gradient magnitude* — you want that to change slowly and cautiously, since it directly controls the step size (an unstable step-size estimate is dangerous). The direction average ($m$) can afford to adapt a bit faster, since responding to genuine direction changes sooner is usually beneficial.

**"When would you pick plain SGD+momentum over Adam?"** Adam's aggressive per-weight adaptivity can sometimes converge to solutions that generalize slightly worse than a well-tuned SGD+momentum run, especially in some computer vision settings — this is an empirically observed pattern in parts of the literature, not a universal law, so state it with that hedge. In practice: Adam/AdamW for transformers and most modern large-scale training (fast, robust to hyperparameter choice); SGD+momentum sometimes preferred when squeezing out the best possible generalization on well-studied architectures where there's time to carefully tune the learning rate schedule.

**"Does Adam still need a learning rate schedule if it's already adaptive?"** Yes — the per-weight adaptivity handles differences *between* weights, but the *overall* scale $\eta$ still benefits from schedules (like warmup, discussed in Chapter 2) for a different reason: at the very start of training, gradients can be unreliable/large, and a warmup period avoids letting Adam's adaptivity lock in a bad early scaling estimate.

---

## 8. Quick Summary Table

| Concept | Plain-language meaning |
|---|---|
| The problem | One global learning rate treats every weight/direction the same, even though real landscapes are steep in some directions and shallow in others |
| AdaGrad | Give each weight a shrinking learning rate based on its *total* accumulated squared-gradient history — decays to zero over time |
| RMSProp | Same idea as AdaGrad, but use a fading (exponential) average instead of an ever-growing sum — fixes the decay-to-zero problem |
| Adam | RMSProp's per-weight scaling + momentum's direction smoothing, combined, plus a bias-correction fix for the first few steps |
| Bias correction | Compensates for $m$ and $v$ starting at zero, which would otherwise artificially shrink the first several updates |
| AdamW | Adam, but weight decay is applied directly to the weights, separately from the adaptive gradient step, instead of tangled into it |
