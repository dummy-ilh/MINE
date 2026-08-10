# Chapter 4: Vanishing & Exploding Gradients

## Picking up where Chapter 3 left off

In Chapter 3 we computed $D_t = \frac{dh_t}{dW_{hh}}$ and noticed the factors $(1-h_t^2)$ — 0.712, 0.172, 0.038 — shrinking fast across just 3 steps. This chapter asks: what happens to gradients over *longer* sequences? The answer explains one of the biggest limitations of vanilla RNNs.

## The general rule: gradient flowing backward through time

Forget $W_{hh}$ for a moment and ask a simpler question: how much does $h_t$ (a late hidden state) respond to $h_k$ (an early hidden state), $k$ steps earlier?

Each step from $h_{s-1}$ to $h_s$ passes through one $\tanh$ and one multiplication by $W_{hh}$:

$$\frac{\partial h_s}{\partial h_{s-1}} = (1-h_s^2)\cdot W_{hh}$$

To go from $h_k$ all the way to $h_t$, you chain these together — one factor per step in between:

$$\frac{\partial h_t}{\partial h_k} = \prod_{s=k+1}^{t} (1-h_s^2)\cdot W_{hh}$$

This product is the heart of the problem. **A gradient traveling back $n$ steps is a product of $n$ numbers.** If those numbers are consistently less than 1, the product shrinks toward zero. If they're consistently greater than 1, it grows without bound.

## Using our actual numbers

From Chapter 2/3: $h_1=0.537,\ h_2=0.910,\ h_3=0.981$, and $W_{hh}=0.8$.

| step | $(1-h_s^2)$ | $\times\ W_{hh}$ | per-step factor |
|---|---|---|---|
| $s=1$ | 0.712 | $\times 0.8$ | 0.570 |
| $s=2$ | 0.172 | $\times 0.8$ | 0.138 |
| $s=3$ | 0.038 | $\times 0.8$ | 0.030 |

The gradient from $h_3$ back to $h_0$:

$$\frac{\partial h_3}{\partial h_0} = 0.570 \times 0.138 \times 0.030 \approx 0.00236$$

In just **3 steps**, the gradient has already shrunk to about a quarter of one percent of its starting size. Whatever learning signal was supposed to reach $h_0$ (and the weights that produced it) has nearly disappeared.

## Zooming out: what happens over 10, 20, 50 steps?

Each additional timestep multiplies in one more factor. If the typical factor is around 0.15 (roughly what we're seeing above, once the hidden state saturates), here's what the product looks like:

| sequence length | approx. gradient reaching the start |
|---|---|
| 3 steps | $0.15^3 \approx 0.0034$ |
| 10 steps | $0.15^{10} \approx 5.8\times10^{-9}$ |
| 20 steps | $0.15^{20} \approx 3.3\times10^{-17}$ |

By 20 steps, the gradient is smaller than floating-point precision can meaningfully represent. This is the **vanishing gradient problem**: for long sequences, the earliest timesteps receive essentially zero learning signal. The network becomes effectively unable to learn long-range dependencies — it can only "see" a handful of recent steps.

## Why does this happen? Two multiplied culprits

1. **$\tanh'$ is always $\le 1$**, and gets close to 0 whenever the hidden state saturates (pushed near $+1$ or $-1$) — which happens often, especially as inputs accumulate. Look back at our table: 0.712 → 0.172 → 0.038. The hidden state moved toward saturation fast.
2. **$W_{hh}$ is a fixed number multiplied in at every step**, regardless of whether that's helpful.

Multiply a bunch of numbers under 1 together, many times, and you get something very close to 0. This isn't a bug specific to our toy example — it's structural. Any vanilla RNN, trained on any long-ish sequence, faces this.

## The flip side: exploding gradients

What if $W_{hh}$ is large, and the hidden state *isn't* saturated (so $\tanh' \approx 1$)? Say $W_{hh} = 3.0$ and $(1-h_s^2)\approx 1$ (true when $h_s$ is near 0, i.e., early in training or with small inputs):

$$\text{per-step factor} \approx 1 \times 3.0 = 3.0$$

Over 5 steps:

$$3.0^5 = 243$$

Over 10 steps:

$$3.0^{10} \approx 59{,}000$$

The gradient doesn't vanish — it **explodes**. Weight updates become enormous, the loss spikes or turns to `NaN`, and training collapses. This is the **exploding gradient problem**: the mirror image of vanishing, caused by the same product-of-many-factors mechanism, just with factors bigger than 1 instead of smaller.

## Vanishing vs. exploding, side by side

| | Vanishing | Exploding |
|---|---|---|
| Cause | per-step factor $< 1$, repeated many times | per-step factor $> 1$, repeated many times |
| Typical trigger | saturated $\tanh$ (small $\tanh'$), or small $W_{hh}$ | large $W_{hh}$, unsaturated $\tanh$ |
| Symptom | early timesteps stop learning; network "forgets" long-range context | loss spikes, weights blow up, training destabilizes |
| More common in practice? | Yes — $\tanh' \le 1$ always caps growth, so shrinkage is the default failure mode | Less common, but sharper and more sudden when it happens |

## Why this matters for the toy example

Our 3-day rain example only had 3 steps, so the effect was mild (gradient shrank to ~0.2%, not literally zero). But scale this up to something realistic — a 100-word sentence, a year of daily stock prices — and the vanilla RNN's memory effectively runs out after a handful of steps, no matter how the weights are trained.

## What's ahead

Chapter 5 introduces the fixes: gradient clipping (a band-aid for exploding gradients), truncated BPTT (a practical compromise), and previews *why* LSTM and GRU exist — they replace the multiply-by-$W_{hh}$-every-step mechanism with something that doesn't force this shrink/explode tradeoff.

---

**One-line summary:** a gradient traveling back $n$ timesteps in an RNN is a product of $n$ factors; if those factors are consistently below 1 the gradient vanishes, if consistently above 1 it explodes — and vanilla RNNs have no built-in way to prevent either.
