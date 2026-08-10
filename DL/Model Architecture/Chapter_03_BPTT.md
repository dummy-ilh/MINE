# Chapter 3: Backpropagation Through Time (BPTT)

## Why this needs special treatment

In a normal feedforward net, each weight affects the loss through exactly one path. You take the derivative, done.

In an RNN, one weight — say $W_{hh}$ — is used at *every* timestep. So $W_{hh}$ affects $h_1$, which affects $h_2$ (because $h_2$ is built from $h_1$), which affects $h_3$, which affects the loss. Changing $W_{hh}$ ripples forward through the whole chain. To get the true gradient, you have to add up its effect through **every path it touches**, not just the most direct one.

That's what BPTT is: ordinary backprop, applied to the unrolled chain, carefully accounting for the fact that the same weight appears at every link.

## Setup: give it something to learn from

We need a target to compare against. Say the true rain probabilities were:

$$y_1^* = 0.6, \quad y_2^* = 0.8, \quad y_3^* = 1.0$$

Using the predictions from Chapter 2 ($\hat{y}_1=0.537$, $\hat{y}_2=0.910$, $\hat{y}_3=0.981$), and squared error loss $L_t = \tfrac{1}{2}(\hat{y}_t - y_t^*)^2$ at each step:

| $t$ | $\hat{y}_t$ | $y_t^*$ | error $e_t = \hat{y}_t - y_t^*$ |
|---|---|---|---|
| 1 | 0.537 | 0.6 | -0.063 |
| 2 | 0.910 | 0.8 | +0.110 |
| 3 | 0.981 | 1.0 | -0.019 |

Total loss: $L = L_1 + L_2 + L_3$. We want $\frac{\partial L}{\partial W_{hh}}$ — how to nudge $W_{hh}$ to reduce the loss.

## The key idea: total derivative, not partial derivative

$h_t$ depends on $W_{hh}$ in two ways:

1. **Directly** — $W_{hh}$ appears explicitly in the formula for $z_t$.
2. **Indirectly** — through $h_{t-1}$, which itself depended on $W_{hh}$ one step earlier.

So the *total* effect of $W_{hh}$ on $h_t$ is:

$$\frac{dh_t}{dW_{hh}} = \underbrace{(1 - h_t^2)}_{\tanh'(z_t)} \cdot \left[\, \underbrace{h_{t-1}}_{\text{direct path}} + \underbrace{W_{hh} \cdot \frac{dh_{t-1}}{dW_{hh}}}_{\text{indirect path, via earlier steps}} \,\right]$$

This is a **recursive formula** — to get the gradient at step $t$, you need the gradient at step $t-1$. Call this total derivative $D_t \equiv \frac{dh_t}{dW_{hh}}$. Then:

$$D_t = (1-h_t^2)\big(h_{t-1} + W_{hh} \cdot D_{t-1}\big), \qquad D_0 = 0$$

($D_0 = 0$ because $h_0$ is a fixed starting point — it doesn't depend on $W_{hh}$ at all.)

## Computing $D_t$ by hand

Recall from Chapter 2: $h_0=0,\ h_1=0.537,\ h_2=0.910,\ h_3=0.981$, and $W_{hh}=0.8$.

**$t=1$:**
$$D_1 = (1-0.537^2)(h_0 + 0.8 \cdot D_0) = (0.712)(0 + 0) = 0$$

Makes sense: at the very first step, there's no previous hidden state for $W_{hh}$ to act on ($h_0=0$), so it contributes nothing yet.

**$t=2$:**
$$D_2 = (1-0.910^2)(h_1 + 0.8 \cdot D_1) = (0.172)(0.537 + 0) = 0.0924$$

**$t=3$:**
$$D_3 = (1-0.981^2)(h_2 + 0.8 \cdot D_2) = (0.038)(0.910 + 0.8 \times 0.0924) = (0.038)(0.984) = 0.0374$$

| $t$ | $1-h_t^2$ | $h_{t-1} + W_{hh}D_{t-1}$ | $D_t$ |
|---|---|---|---|
| 1 | 0.712 | 0.000 | 0.000 |
| 2 | 0.172 | 0.537 | 0.0924 |
| 3 | 0.038 | 0.984 | 0.0374 |

## Turning $D_t$ into the actual gradient

Each timestep's loss $L_t$ depends on $h_t$ through $\hat{y}_t = W_{hy}h_t + b_y$ (with $W_{hy}=1$ from Chapter 2), so $\frac{\partial L_t}{\partial h_t} = e_t \cdot W_{hy} = e_t$.

The full gradient sums each step's contribution:

$$\frac{\partial L}{\partial W_{hh}} = \sum_t e_t \cdot D_t = (-0.063)(0) + (0.110)(0.0924) + (-0.019)(0.0374)$$

$$= 0 + 0.0102 - 0.0007 = 0.0095$$

So nudging $W_{hh}$ down slightly would reduce the loss — a tiny, gentle signal.

## The picture: multiple backward paths merging

```
       dL/dW_hh
           ^
           |
   +-------+-------+-------+
   |               |       |
 (t=1 path)   (t=2 path)  (t=3 path)
   |               |       |
  h1 <---W_hh--- h2 <---W_hh--- h3
```

$W_{hh}$ gets *one* gradient update, but that update is the sum of its influence at every timestep it was used. This is the "through time" in backpropagation through time — you backprop the loss from the end of the sequence all the way to the start, accumulating $W_{hh}$'s contribution at each stop along the way.

## Something to notice (preview of Chapter 4)

Look at the $(1-h_t^2)$ column: **0.712 → 0.172 → 0.038**. It shrinks fast. This is $\tanh'$, and it shrinks toward zero whenever $h_t$ gets close to $\pm1$ (which happens as the hidden state saturates). Every extra timestep multiplies in another one of these shrinking factors. Chapter 4 shows what happens to this product over a *long* sequence — and why it's a real problem.

## $W_{xh}$ and $b_h$ follow the same pattern

The exact same recursive logic applies to the other two learnable parameters — just swap what's added at the "direct path" step:

$$\frac{dh_t}{dW_{xh}} = (1-h_t^2)\Big(x_t + W_{hh}\cdot\frac{dh_{t-1}}{dW_{xh}}\Big), \qquad \frac{dh_t}{db_h} = (1-h_t^2)\Big(1 + W_{hh}\cdot\frac{dh_{t-1}}{db_h}\Big)$$

Same idea: direct contribution at this step, plus everything inherited from before, scaled by $\tanh'$.

## What's ahead

Chapter 4 stretches this same mechanism over a longer sequence and shows, numerically, why the gradient can shrink toward zero (vanish) — meaning early timesteps stop getting any learning signal at all.

---

**One-line summary:** BPTT is regular backprop applied to the unrolled RNN, where each weight's total gradient is the *sum of its effect at every timestep*, computed recursively backward from the end of the sequence.
