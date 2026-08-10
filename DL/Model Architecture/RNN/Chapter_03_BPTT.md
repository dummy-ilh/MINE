# Chapter 3: Backpropagation Through Time (BPTT)

## Why this needs special treatment

In a normal feedforward net, each weight affects the loss through exactly one path. You take the derivative, done.

In an RNN, one weight — say $W_{hh}$ — is used at *every* timestep. So $W_{hh}$ affects $h_1$, which affects $h_2$ (because $h_2$ is built from $h_1$), which affects $h_3$, which affects the loss. Changing $W_{hh}$ ripples forward through the whole chain. To get the true gradient, you have to add up its effect through **every path it touches**, not just the most direct one.

That's what BPTT is: ordinary backprop, applied to the unrolled chain, carefully accounting for the fact that the same weight appears at every link.

---

## Part A: Building the idea gently, before any formula

### Start with something you already believe

If you nudge $W_{hh}$ up a tiny bit, does $h_1$ change? Look back at Chapter 2:

$$h_1 = \tanh(W_{xh}x_1 + W_{hh}\cdot h_0 + b_h)$$

With $h_0 = 0$, the term $W_{hh}\cdot h_0$ is $W_{hh}\cdot 0 = 0$, no matter what $W_{hh}$ is. So nudging $W_{hh}$ does **nothing** to $h_1$ — there's nothing for it to multiply against yet. Good, that's a fact we can check by eye, and it'll show up again below as a sanity check.

Now ask the same question about $h_2$:

$$h_2 = \tanh(W_{xh}x_2 + W_{hh}\cdot h_1 + b_h)$$

This time $W_{hh}$ multiplies $h_1$, which is *not* zero. So nudging $W_{hh}$ directly changes $h_2$ — that's one path of influence. But there's a second, sneakier path: $h_1$ itself was computed using $W_{hh}$ too (see the equation above for $h_1$). So if we'd nudged $W_{hh}$ *before* computing $h_1$, then $h_1$ would have come out slightly different, which would *also* change $h_2$ — a second, indirect path, riding through $h_1$.

This is the whole idea of the chapter: **as you move later into the sequence, $W_{hh}$ has more and more paths by which it can influence the current hidden state** — one direct path (it's used right here, right now) and one indirect path per earlier timestep (it was also used back then, and that echoes forward).

### Why we can't just use the ordinary chain rule once

In a normal feedforward layer, a weight touches the output through one clean path, so $\frac{\partial \text{output}}{\partial \text{weight}}$ is a single chain-rule product. Here, $W_{hh}$ touches $h_t$ through *multiple* paths simultaneously (direct + every earlier indirect echo). When something affects an output through multiple paths, calculus says: **compute the effect of each path separately, then add them up.** That's the one new idea this chapter introduces — everything else is chain rule you already know.

### Turning that into a recursive formula

Rather than tracing every path separately by hand each time (which gets combinatorially painful as $t$ grows), we build a running total. Define:

$$D_t \equiv \frac{dh_t}{dW_{hh}} = \text{"the total effect of a nudge to } W_{hh} \text{ on } h_t\text{, through every path combined"}$$

We want a formula for $D_t$ in terms of $D_{t-1}$ — so each step only has to account for what's *new* at that step, and trust that $D_{t-1}$ already correctly totaled up everything before it.

Recall $h_t = \tanh(z_t)$ where $z_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h$. Two things affect $D_t$:

1. **How much a change in $z_t$ moves $h_t$.** That's the slope of $\tanh$ at this point, $\tanh'(z_t) = 1-h_t^2$ — a number that's large when $h_t$ is near 0, and small when $h_t$ is saturated near $\pm1$ (this fact becomes very important in Chapter 4).
2. **How much a change in $W_{hh}$ moves $z_t$.** Looking at $z_t = W_{xh}x_t + W_{hh}h_{t-1}+b_h$, and remembering $W_{hh}$ appears in **two places** — explicitly multiplying $h_{t-1}$, *and* implicitly inside $h_{t-1}$ itself (since $h_{t-1}$ was itself built using $W_{hh}$) — product-rule-style, that total effect on $z_t$ is:
$$\underbrace{h_{t-1}}_{\text{direct: } W_{hh}\text{ multiplies this value right now}} + \underbrace{W_{hh}\cdot D_{t-1}}_{\text{indirect: this value itself shifts by } D_{t-1}\text{ per unit of }W_{hh}}$$

Multiply piece 1 and piece 2 together (chain rule: slope of the outer function times the derivative of what's inside it), and you get the full recursive formula:

$$D_t = (1-h_t^2)\big(h_{t-1} + W_{hh}\cdot D_{t-1}\big), \qquad D_0 = 0$$

($D_0=0$ because $h_0$ is a fixed starting point we chose ourselves — it doesn't depend on $W_{hh}$ at all, so a nudge to $W_{hh}$ can't move it.)

Read the formula in words: *"$W_{hh}$'s total effect on $h_t$ equals (how sharply $\tanh$ responds right now) times (its brand-new direct contribution this step, plus everything it already contributed before, carried forward and re-scaled by one more factor of $W_{hh}$)."*

---

## Part B: Setup — give it something to learn from

We need a target to compare against. Say the true rain probabilities were:

$$y_1^* = 0.6, \quad y_2^* = 0.8, \quad y_3^* = 1.0$$

Using the predictions from Chapter 2 ($\hat{y}_1=0.537$, $\hat{y}_2=0.910$, $\hat{y}_3=0.981$), and squared error loss $L_t = \tfrac{1}{2}(\hat{y}_t - y_t^*)^2$ at each step:

| $t$ | $\hat{y}_t$ | $y_t^*$ | error $e_t = \hat{y}_t - y_t^*$ |
|---|---|---|---|
| 1 | 0.537 | 0.6 | -0.063 |
| 2 | 0.910 | 0.8 | +0.110 |
| 3 | 0.981 | 1.0 | -0.019 |

*(Why $e_t = \hat y_t - y_t^*$ and not the other way round: with $L_t = \tfrac12(\hat y_t - y_t^*)^2$, calculus gives $\frac{\partial L_t}{\partial \hat y_t} = \hat y_t - y_t^*$ directly — that derivative is exactly what we're calling $e_t$, so nothing extra needs computing there.)*

Total loss: $L = L_1 + L_2 + L_3$. We want $\frac{\partial L}{\partial W_{hh}}$ — how to nudge $W_{hh}$ to reduce the loss.

---

## Part C: Computing $D_t$ by hand, in full detail

Recall from Chapter 2: $h_0=0,\ h_1=0.537,\ h_2=0.910,\ h_3=0.981$, and $W_{hh}=0.8$.

### $t=1$:

First, the $\tanh'$ factor:
$$1 - h_1^2 = 1 - (0.537)^2 = 1 - 0.288 = 0.712$$

Then the bracket — direct term plus indirect term:
$$h_0 + W_{hh}\cdot D_0 = 0 + (0.8)(0) = 0$$

Multiply:
$$D_1 = (0.712)(0) = 0$$

This matches the sanity check from Part A: at the very first step, there's no previous hidden state for $W_{hh}$ to act on ($h_0=0$), so it contributes nothing yet — confirmed by the formula, not just eyeballed.

### $t=2$:

$\tanh'$ factor:
$$1 - h_2^2 = 1 - (0.910)^2 = 1 - 0.828 = 0.172$$

Bracket:
- Direct term: $h_1 = 0.537$
- Indirect term: $W_{hh}\cdot D_1 = (0.8)(0) = 0$
- Sum: $0.537 + 0 = 0.537$

Multiply:
$$D_2 = (0.172)(0.537) = 0.0924$$

### $t=3$:

$\tanh'$ factor:
$$1 - h_3^2 = 1 - (0.981)^2 = 1 - 0.962 = 0.038$$

Bracket:
- Direct term: $h_2 = 0.910$
- Indirect term: $W_{hh}\cdot D_2 = (0.8)(0.0924) = 0.0739$
- Sum: $0.910 + 0.0739 = 0.984$

Multiply:
$$D_3 = (0.038)(0.984) = 0.0374$$

| $t$ | $1-h_t^2$ | direct: $h_{t-1}$ | indirect: $W_{hh}D_{t-1}$ | sum | $D_t$ |
|---|---|---|---|---|---|
| 1 | 0.712 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.172 | 0.537 | 0.000 | 0.537 | 0.0924 |
| 3 | 0.038 | 0.910 | 0.0739 | 0.984 | 0.0374 |

Notice: even though the "sum" column keeps growing (0 → 0.537 → 0.984, i.e. $W_{hh}$'s raw combined influence is accumulating more with each timestep, exactly as you'd expect since it's used more times), $D_t$ itself goes *down* from step 2 to step 3 (0.0924 → 0.0374). That's the $\tanh'$ factor overpowering the accumulation — it shrinks from 0.172 to 0.038, more than 4x, which more than cancels the growth in the sum. Keep this tug-of-war in mind; it's the entire subject of Chapter 4.

---

## Part D: Turning $D_t$ into the actual gradient

Each timestep's loss $L_t$ depends on $h_t$ through $\hat{y}_t = W_{hy}h_t + b_y$ (with $W_{hy}=1$ from Chapter 2), so $\frac{\partial L_t}{\partial h_t} = e_t \cdot W_{hy} = e_t \cdot 1 = e_t$ — the error at that step passes straight through unchanged, because $W_{hy}=1$ was chosen to keep this arithmetic simple.

The full gradient is the sum, over every timestep, of "how wrong were we there" times "how much did $W_{hh}$ influence the hidden state feeding that step's prediction":

$$\frac{\partial L}{\partial W_{hh}} = \sum_t e_t \cdot D_t$$

Compute each product separately:
- $t=1$: $(-0.063)(0) = 0$
- $t=2$: $(0.110)(0.0924) = 0.01016$
- $t=3$: $(-0.019)(0.0374) = -0.00071$

Sum them:
$$\frac{\partial L}{\partial W_{hh}} = 0 + 0.01016 - 0.00071 = 0.00945 \approx 0.0095$$

So nudging $W_{hh}$ down slightly would reduce the loss — a tiny, gentle signal. Notice the $t=2$ term dominates: that's the step with the largest error ($e_2=0.110$) landing on a $D_t$ that hadn't yet been shrunk as much by repeated $\tanh'$ multiplication.

---

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

The exact same recursive logic applies to the other two learnable parameters — just swap what's added at the "direct path" step. Compare directly against the $D_t$ formula above:

$$\frac{dh_t}{dW_{xh}} = (1-h_t^2)\Big(x_t + W_{hh}\cdot\frac{dh_{t-1}}{dW_{xh}}\Big), \qquad \frac{dh_t}{db_h} = (1-h_t^2)\Big(1 + W_{hh}\cdot\frac{dh_{t-1}}{db_h}\Big)$$

Why the swap makes sense: for $W_{hh}$, the "direct" term was $h_{t-1}$ because that's literally what $W_{hh}$ multiplies inside $z_t$. For $W_{xh}$, the thing it multiplies inside $z_t$ is $x_t$ — so $x_t$ takes over that slot. For $b_h$, it's added with an implicit coefficient of 1 — so $1$ takes that slot. The "indirect, inherited from before" term always keeps the same shape, $W_{hh}\cdot(\text{previous step's total derivative})$, because that inheritance always flows through the recurrence $h_{t-1}\to h_t$ regardless of which weight you're differentiating with respect to.

---

## Part E: Interview questions (Google / Apple style)

**Q1. Why can't you just apply the chain rule once to get $\partial L/\partial W_{hh}$, the way you would in a feedforward network?**
Because $W_{hh}$ isn't used once — it's reused at every timestep, so it influences the loss through multiple simultaneous paths (one direct use per timestep, plus every indirect echo carried forward from earlier timesteps). Calculus's rule for a quantity that affects an output through multiple paths is to differentiate along each path and sum the results — that's the multivariable/total-derivative chain rule, not the single-path chain rule from a feedforward layer.

**Q2. What does $D_0=0$ mean, intuitively, and why is it necessary as a base case?**
$D_0$ is $\frac{dh_0}{dW_{hh}}$ — how much $W_{hh}$ affects the *initial* hidden state. Since $h_0$ is a fixed constant we chose before any weight is applied (usually zero), no value of $W_{hh}$ can change it, so its derivative with respect to $W_{hh}$ is exactly 0. It's necessary as a base case because $D_t$ is defined recursively in terms of $D_{t-1}$ — without a starting value, the recursion has nothing to bottom out on.

**Q3. In the $D_t$ recursion, what do the "direct" and "indirect" terms represent, and why are they added rather than multiplied?**
The direct term ($h_{t-1}$) is $W_{hh}$'s brand-new contribution at this exact timestep — it appears explicitly in this step's formula for $z_t$. The indirect term ($W_{hh}\cdot D_{t-1}$) is the *inherited* effect: $W_{hh}$ already nudged $h_{t-1}$ at the previous step, and that nudge is still present, carried into $z_t$ (scaled once more by $W_{hh}$, since $h_{t-1}$ is itself multiplied by $W_{hh}$). They're added, not multiplied, because they're two independent contributions to the same sum ($z_t$) — this mirrors the sum rule for derivatives: the derivative of a sum is the sum of the derivatives.

**Q4. Why does the $t=2$ term dominate the final gradient sum in this example, rather than $t=3$, even though $t=3$ is later in the sequence?**
The final per-step contribution to the gradient is $e_t \cdot D_t$, a product of two competing factors. $D_t$ tends to shrink at later steps once $h_t$ saturates near $\pm1$ (because $\tanh'(z_t)=1-h_t^2$ gets small there) — in this example $D_t$ actually goes down from $t=2$ to $t=3$ (0.0924 → 0.0374) despite $t=3$ having more accumulated paths, because the shrinking $\tanh'$ factor (0.172→0.038) outweighs the growth. Combined with $t=2$ also having the largest raw error ($e_2=0.110$), its product dominates the sum.

**Q5. If the sequence were 50 steps long instead of 3, what would you expect to happen to $D_{50}$ relative to $D_2$, and why?**
$D_t$ is built by repeatedly multiplying by $\tanh'(z_t)=1-h_t^2$, a factor that's strictly less than 1 whenever $h_t\neq0$, and much less than 1 once $h_t$ saturates near $\pm1$. Multiplying many such sub-1 factors together compounds — over 50 steps, $D_{50}$ would typically be many orders of magnitude smaller than $D_2$, unless something (careful initialization, gating, gradient clipping, etc.) intervenes. This is precisely the vanishing gradient problem, covered numerically in Chapter 4.

**Q6. How would the recursion for $D_t$ change if the activation function were ReLU instead of $\tanh$?**
The $(1-h_t^2)$ factor is specifically $\tanh'(z_t)$; for ReLU, the derivative is a step function — 1 wherever $z_t>0$, and 0 wherever $z_t\le0$ (with a defined convention at exactly 0). The recursion structure stays the same ($D_t = f'(z_t)\big(h_{t-1}+W_{hh}D_{t-1}\big)$), but ReLU's derivative never lies strictly *between* 0 and 1 the way $\tanh'$ does — so instead of a gradual shrink, ReLU-based recurrent gradients either pass through completely unchanged (times 1) or die completely (times 0) at each step, which trades vanishing gradients for a different failure mode (exploding gradients from repeated ×1 multiplication by $W_{hh}$, if $|W_{hh}|>1$).

**Q7. Why is $\partial L_t/\partial h_t$ equal to $e_t$ in this example, and would that still hold with a different readout formula?**
It's $e_t$ here specifically because $\hat y_t = W_{hy}h_t+b_y$ with $W_{hy}=1$: by the chain rule, $\frac{\partial L_t}{\partial h_t} = \frac{\partial L_t}{\partial \hat y_t}\cdot\frac{\partial \hat y_t}{\partial h_t} = e_t \cdot W_{hy} = e_t\cdot 1$. If $W_{hy}$ were, say, 2, this would become $2e_t$ instead. With a different readout entirely (e.g. softmax + cross-entropy for classification), the exact form of $\partial L_t/\partial h_t$ changes, but the *role* it plays — "how much does this step's local loss push back on the hidden state" — stays the same, and everything downstream (the $D_t$ machinery) is unaffected.

**Q8. This chapter computed the gradient for $W_{hh}$ using an explicit recursive formula for $D_t$. What does an autodiff framework like PyTorch actually do differently, if anything?**
Conceptually, nothing — autodiff builds the same computation graph (every use of $W_{hh}$ at every timestep is a separate node sharing the same underlying parameter) and accumulates gradients into that one shared parameter exactly the way the sum $\sum_t e_t D_t$ does here. The difference is mechanical: PyTorch doesn't require you to derive a closed-form recursion like $D_t$ by hand — it walks the actual unrolled graph backward, applying the chain rule at each node automatically and summing contributions into `W_hh.grad`, which is exactly what Chapter 9's from-scratch/PyTorch comparison verifies numerically.

## What's ahead

Chapter 4 stretches this same mechanism over a longer sequence and shows, numerically, why the gradient can shrink toward zero (vanish) — meaning early timesteps stop getting any learning signal at all.

---

**One-line summary:** BPTT is regular backprop applied to the unrolled RNN, where each weight's total gradient is the *sum of its effect at every timestep*, computed recursively backward from the end of the sequence — one direct contribution per step, plus everything inherited from before.
