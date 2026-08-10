# Chapter 4: Vanishing & Exploding Gradients

## Picking up where Chapter 3 left off

In Chapter 3 we computed $D_t = \frac{dh_t}{dW_{hh}}$ and noticed the factors $(1-h_t^2)$ — 0.712, 0.172, 0.038 — shrinking fast across just 3 steps. This chapter asks: what happens to gradients over *longer* sequences? The answer explains one of the biggest limitations of vanilla RNNs.

---

## Part A: Building the idea gently, before any formula

### Strip the problem down to its simplest version

Chapter 3 tracked how $W_{hh}$ affects $h_t$. Let's simplify even further and ask a cleaner question first: forget $W_{hh}$ entirely for a second — how much does $h_t$ (a *late* hidden state) respond to $h_k$ (an *early* hidden state), sitting $n$ steps earlier in the same sequence?

Think about just **one single step** first: how does $h_{s-1}$ affect $h_s$? Look at the update formula from Chapter 2:

$$h_s = \tanh(W_{xh}x_s + W_{hh}h_{s-1}+b_h)$$

$h_{s-1}$ enters this formula in exactly one place: multiplied by $W_{hh}$, then passed through $\tanh$. So by the chain rule (slope of $\tanh$ times the coefficient in front of $h_{s-1}$):

$$\frac{\partial h_s}{\partial h_{s-1}} = \underbrace{(1-h_s^2)}_{\text{tanh's slope here}} \cdot \underbrace{W_{hh}}_{\text{coefficient in front of }h_{s-1}}$$

This is just **one link in the chain** — the effect of one timestep on the very next one. Call this single-step number the "per-step factor."

### Now stretch that one link into a whole chain

To find out how $h_k$ (early) affects $h_t$ (late), you don't just look at one link — you have to travel through *every* link in between: $h_k \to h_{k+1} \to h_{k+2} \to \cdots \to h_t$. Each arrow is one instance of the single-step formula above. And when you want the *combined* effect of a chain of dependencies — this one causes that one, which causes the next one — the chain rule says: **multiply the per-link effects together.**

$$\frac{\partial h_t}{\partial h_k} = \frac{\partial h_t}{\partial h_{t-1}}\cdot\frac{\partial h_{t-1}}{\partial h_{t-2}}\cdots\frac{\partial h_{k+1}}{\partial h_k} = \prod_{s=k+1}^{t}(1-h_s^2)\cdot W_{hh}$$

That $\prod$ symbol just means "multiply all of these together, one per step from $k+1$ up to $t$." Nothing new conceptually — it's the one-link formula, reused once per step, all multiplied.

### Why a product of many numbers is dangerous

Here's the intuitive crux of the whole chapter, before any numbers: **if you multiply together a long list of numbers that are each a bit less than 1, the result shrinks toward zero — fast.** ($0.5 \times 0.5 \times 0.5 \times 0.5 = 0.0625$ — four steps, and you're already below one-sixteenth.) Conversely, **if each number is a bit more than 1, the product explodes** — grows without bound, also fast, in the other direction. A gradient traveling backward $n$ steps through an RNN is exactly this kind of product: $n$ numbers multiplied together, one per step crossed. Whether it vanishes or explodes depends entirely on whether those per-step factors tend to be below or above 1.

---

## Part B: Using our actual numbers

From Chapter 2/3: $h_1=0.537,\ h_2=0.910,\ h_3=0.981$, and $W_{hh}=0.8$.

**Compute each per-step factor, showing both pieces:**

**Step $s=1$:**
- $\tanh'$ piece: $1-h_1^2 = 1-0.288 = 0.712$
- Multiply by $W_{hh}$: $0.712 \times 0.8 = 0.570$

**Step $s=2$:**
- $\tanh'$ piece: $1-h_2^2 = 1-0.828 = 0.172$
- Multiply by $W_{hh}$: $0.172 \times 0.8 = 0.138$

**Step $s=3$:**
- $\tanh'$ piece: $1-h_3^2 = 1-0.962 = 0.038$
- Multiply by $W_{hh}$: $0.038 \times 0.8 = 0.030$

| step | $(1-h_s^2)$ | $\times\ W_{hh}(=0.8)$ | per-step factor |
|---|---|---|---|
| $s=1$ | 0.712 | $\times 0.8$ | 0.570 |
| $s=2$ | 0.172 | $\times 0.8$ | 0.138 |
| $s=3$ | 0.038 | $\times 0.8$ | 0.030 |

**Now chain all three together** to get the gradient traveling from $h_3$ all the way back to $h_0$ — multiply step by step so you can see it shrink at each stage:

$$0.570 \times 0.138 = 0.0787 \quad\text{(after 2 links)}$$
$$0.0787 \times 0.030 = 0.00236 \quad\text{(after all 3 links)}$$

$$\frac{\partial h_3}{\partial h_0} \approx 0.00236$$

In just **3 steps**, the gradient has already shrunk to about a quarter of one percent of its starting size (since it started effectively at 1 — a change to $h_0$ would, before any shrinkage, move $h_0$ itself by exactly that much). Whatever learning signal was supposed to reach $h_0$ (and the weights that produced it) has nearly disappeared.

---

## Part C: Zooming out — what happens over 10, 20, 50 steps?

Each additional timestep multiplies in one more factor — the product just keeps getting longer. If the typical factor stabilizes around 0.15 (roughly what we're seeing above, once the hidden state saturates near $\pm1$), we can predict the pattern without hand-computing every step, just by raising 0.15 to a power:

**3 steps:** $0.15 \times 0.15 \times 0.15 = 0.15^3$. Compute: $0.15^2 = 0.0225$, then $0.0225\times0.15 = 0.003375 \approx 0.0034$.

**10 steps:** $0.15^{10}$. Doubling the exponent from 3 isn't quite enough to reach 10, but the pattern is: each extra factor of 0.15 shrinks the running product by another ~85%. By 10 steps: $0.15^{10}\approx 5.8\times10^{-9}$ — under one-billionth.

**20 steps:** $0.15^{20} = (0.15^{10})^2 \approx (5.8\times10^{-9})^2 \approx 3.3\times10^{-17}$ — squaring the 10-step result, since 20 steps is just two 10-step chains multiplied together.

| sequence length | approx. gradient reaching the start |
|---|---|
| 3 steps | $0.15^3 \approx 0.0034$ |
| 10 steps | $0.15^{10} \approx 5.8\times10^{-9}$ |
| 20 steps | $0.15^{20} \approx 3.3\times10^{-17}$ |

For context on how small $3.3\times10^{-17}$ really is: a standard 32-bit float can only reliably distinguish numbers down to around $10^{-7}$ relative precision — so by 20 steps, this gradient isn't just "small," it's smaller than the numerical noise floor of the computation itself. It rounds to effectively zero.

This is the **vanishing gradient problem**: for long sequences, the earliest timesteps receive essentially zero learning signal. The network becomes effectively unable to learn long-range dependencies — it can only "see" a handful of recent steps.

---

## Part D: Why does this happen? Two multiplied culprits

1. **$\tanh'$ is always $\le 1$**, and gets close to 0 whenever the hidden state saturates (pushed near $+1$ or $-1$) — which happens often, especially as inputs accumulate. Look back at our table: 0.712 → 0.172 → 0.038. The hidden state moved toward saturation fast.
2. **$W_{hh}$ is a fixed number multiplied in at every step**, regardless of whether that's helpful.

Multiply a bunch of numbers under 1 together, many times, and you get something very close to 0. This isn't a bug specific to our toy example — it's structural. Any vanilla RNN, trained on any long-ish sequence, faces this.

---

## Part E: The flip side — exploding gradients

Same mechanism, opposite direction. What if $W_{hh}$ is large, and the hidden state *isn't* saturated (so $\tanh' \approx 1$, which is true when $h_s$ is near 0 — e.g. early in training, or with small inputs)?

Say $W_{hh} = 3.0$ and $(1-h_s^2)\approx 1$:

$$\text{per-step factor} \approx 1 \times 3.0 = 3.0$$

Chain this over 5 steps, one multiplication at a time:
$$3.0\to 3.0\times3.0=9.0\to 9.0\times3.0=27\to27\times3.0=81\to81\times3.0=243$$

$$3.0^5 = 243$$

Over 10 steps, same doubling trick as before: $3.0^{10} = (3.0^5)^2 = 243^2 \approx 59{,}000$.

The gradient doesn't vanish — it **explodes**. Weight updates become enormous, the loss spikes or turns to `NaN`, and training collapses. This is the **exploding gradient problem**: the mirror image of vanishing, caused by the same product-of-many-factors mechanism, just with factors bigger than 1 instead of smaller.

---

## Vanishing vs. exploding, side by side

| | Vanishing | Exploding |
|---|---|---|
| Cause | per-step factor $< 1$, repeated many times | per-step factor $> 1$, repeated many times |
| Typical trigger | saturated $\tanh$ (small $\tanh'$), or small $W_{hh}$ | large $W_{hh}$, unsaturated $\tanh$ |
| Symptom | early timesteps stop learning; network "forgets" long-range context | loss spikes, weights blow up, training destabilizes |
| More common in practice? | Yes — $\tanh' \le 1$ always caps growth, so shrinkage is the default failure mode | Less common, but sharper and more sudden when it happens |

## Why this matters for the toy example

Our 3-day rain example only had 3 steps, so the effect was mild (gradient shrank to ~0.2%, not literally zero). But scale this up to something realistic — a 100-word sentence, a year of daily stock prices — and the vanilla RNN's memory effectively runs out after a handful of steps, no matter how the weights are trained.

---

## Part F: Interview questions (Google / Apple style)

**Q1. In plain words, why is the vanishing gradient problem a product of many numbers, rather than a sum?**
Because the dependency between $h_k$ and $h_t$ runs through a *chain* — $h_k$ affects $h_{k+1}$, which affects $h_{k+2}$, and so on, up to $h_t$. The chain rule for a chain of dependencies (as opposed to multiple parallel paths, which get summed, as in Chapter 3) is to multiply the per-link derivatives together. Each link contributes a factor typically less than 1 (from $\tanh'\le1$ combined with $W_{hh}$), and multiplying many sub-1 numbers together shrinks the product exponentially with the number of links.

**Q2. Why does $\tanh'(z_s) = 1-h_s^2$ specifically encourage vanishing, rather than exploding, as the default behavior?**
Because $\tanh$'s output is bounded in $(-1,1)$, its derivative $1-h_s^2$ is always $\le 1$, hitting exactly 1 only when $h_s=0$ and approaching 0 as $h_s\to\pm1$. This means the $\tanh'$ factor alone can never push the per-step product above 1 — it can only leave it unchanged (at best) or shrink it. Whether the *overall* per-step factor (which also includes $W_{hh}$) exceeds 1 depends on $W_{hh}$, but the $\tanh'$ term structurally biases things toward shrinkage, which is why vanishing is the more common failure mode in practice.

**Q3. If $W_{hh}=1$ exactly, does that guarantee gradients neither vanish nor explode?**
No — even with $W_{hh}=1$, the per-step factor is $(1-h_s^2)\times 1 = 1-h_s^2$, which is still $<1$ whenever $h_s\neq0$ (and $h_s$ is essentially always nonzero once inputs start flowing in). So gradients would still vanish, just somewhat more slowly than with $W_{hh}<1$. $W_{hh}=1$ alone doesn't neutralize the $\tanh'$ contribution — you'd need the *combined* per-step factor to average to exactly 1 across the sequence, which is a much harder condition to engineer or guarantee.

**Q4. Chapter 3 computed $D_t = dh_t/dW_{hh}$, a single weight's total gradient. How does that relate to the $\partial h_t/\partial h_k$ quantity in this chapter?**
$\partial h_t/\partial h_k$ measures pure state-to-state sensitivity — ignoring which specific weight caused a nudge to $h_k$, just "if $h_k$ changed, how much would $h_t$ change." $D_t$ from Chapter 3 folds a specific weight's *direct* contribution at each step on top of this same backward-flowing product — in fact, the recursive $D_t$ formula from Chapter 3 is built from exactly the same $(1-h_t^2)$ and $W_{hh}$ factors used here. This chapter's $\partial h_t/\partial h_k$ is really the "vanishing/exploding engine" sitting underneath every weight's gradient computation, isolated and studied on its own.

**Q5. Why does exploding gradient training typically produce loss spikes or `NaN`s, while vanishing gradient training produces no dramatic symptom at all?**
An exploding gradient directly and suddenly inflates the weight update ($\Delta W \propto \text{gradient}$), so a single bad batch can push weights to extreme values in one step — producing a visible loss spike, or an overflow to `NaN`/`inf` in floating-point arithmetic. A vanishing gradient does the opposite: the update becomes *too small to matter*, so training simply proceeds as if those weights (or those early timesteps' influence) had stopped updating at all — no crash, no spike, just silent stagnation, which is part of why vanishing gradients are historically harder to detect and debug than exploding ones.

**Q6. Name two mitigations — one for each problem — and briefly say why each works, without going into full detail (Chapter 5 territory).**
For exploding gradients: **gradient clipping** — rescale the gradient vector if its norm exceeds a threshold, so a single extreme update can't blow up the weights, without changing the update's *direction*. For vanishing gradients: architectures like **LSTM/GRU** replace the plain multiply-by-$W_{hh}$-every-step recurrence with a gated additive pathway (a "cell state" that information can flow through with fewer forced multiplications by sub-1 factors), which is far less prone to shrinking to zero over long sequences.

**Q7. Given the formula $\frac{\partial h_t}{\partial h_k} = \prod_{s=k+1}^t (1-h_s^2)W_{hh}$, what's the maximum possible value this product could take if $W_{hh}=2$ and every $h_s=0$ across 5 steps?**
If every $h_s=0$, then every $(1-h_s^2)=1$, so each per-step factor is exactly $1\times2=2$. Over 5 steps: $2^5=32$. This is the theoretical ceiling under these conditions — in practice $h_s=0$ at every step is unrealistic once real inputs are flowing through the network, but it illustrates that the $\tanh'$ term can only ever pull the factor *down* from $W_{hh}$, never push it up.

**Q8. Is truncated BPTT (only backpropagating through the last $k$ timesteps instead of the full sequence) a fix for vanishing gradients, or a workaround?**
It's a workaround, not a fix. Truncated BPTT doesn't change the underlying $(1-h_s^2)W_{hh}$ per-step factor or make it closer to 1 — it just avoids computing (and paying the compute cost for) the extremely long products that would vanish anyway, and accepts that the model can't learn dependencies longer than the truncation window. The actual vanishing mechanism is untouched; you're choosing not to look past the point where the gradient would've been useless regardless.

## What's ahead

Chapter 5 introduces the fixes: gradient clipping (a band-aid for exploding gradients), truncated BPTT (a practical compromise), and previews *why* LSTM and GRU exist — they replace the multiply-by-$W_{hh}$-every-step mechanism with something that doesn't force this shrink/explode tradeoff.

---

**One-line summary:** a gradient traveling back $n$ timesteps in an RNN is a product of $n$ factors — each one $(1-h_s^2)\cdot W_{hh}$ — so if those factors are consistently below 1 the gradient vanishes exponentially fast, and if consistently above 1 it explodes exponentially fast; vanilla RNNs have no built-in mechanism to prevent either.
