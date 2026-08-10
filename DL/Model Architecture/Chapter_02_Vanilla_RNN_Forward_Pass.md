# Chapter 2: The Vanilla RNN Forward Pass

## Recall the loop

From Chapter 1:

```
hidden_state = 0
for each input x_t:
    hidden_state = update(hidden_state, x_t)
    output = readout(hidden_state)
```

This chapter turns `update` and `readout` into real numbers. We'll hand-compute every single value, and — before touching any formula — build up *why* it looks the way it does.

---

## Part A: Building the formula gently, piece by piece

Forget the equation for a second. Think about what "update the summary" actually needs to do.

You're standing at day $t$. You have two things in your hands:
1. **Today's input** $x_t$ (today's wetness score).
2. **Yesterday's summary** $h_{t-1}$ (everything you'd bothered to remember up through yesterday).

Your job: blend them into a new summary $h_t$.

**Piece 1 — let today's input matter.**
The simplest way to let $x_t$ influence anything is to scale it by some number:
$$W_{xh} \, x_t$$
$W_{xh}$ is just a dial. Turn it up, and today's input matters more. Turn it down (toward 0), and today barely registers.

**Piece 2 — let yesterday's summary matter.**
Same idea, a second dial, applied to yesterday's summary instead of today's input:
$$W_{hh} \, h_{t-1}$$
Turn $W_{hh}$ up, and the network leans heavily on its memory. Turn it down, and it forgets yesterday almost entirely.

**Piece 3 — add them together.**
If both pieces matter, the natural thing to do is add them — today's contribution plus yesterday's contribution:
$$W_{xh}\,x_t + W_{hh}\,h_{t-1}$$
This sum is a blend: part "what just happened," part "what I already knew."

**Piece 4 — add a baseline.**
Sometimes you want the summary to lean a certain direction by default, even before looking at $x_t$ or $h_{t-1}$ at all. A constant $b_h$ (the "bias") does that:
$$z_t = W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h$$
Call this raw blended number $z_t$. It's the pre-squashing version of the new summary. Notice: nothing so far stops $z_t$ from being enormous. If wetness scores are large and weights aren't tiny, $z_t$ could be 50, or -300. An unbounded "summary" is awkward to keep reusing over and over across a long sequence — it could blow up.

**Piece 5 — squash it into a stable range.**
This is where $\tanh$ comes in. $\tanh$ takes *any* real number and squeezes it into the range $(-1, 1)$:

- Very negative input → output near $-1$
- Zero input → output exactly $0$
- Very positive input → output near $+1$
- Middling input → somewhere smoothly in between

$$h_t = \tanh(z_t)$$

Think of $\tanh$ as a volume knob that can never go past max or min, no matter how hard you push it. That's exactly what we want for a summary that gets reused as input to itself, timestep after timestep — it keeps $h_t$ from spiraling out of control.

**Putting Pieces 1–5 together, the full update step is:**
$$h_t = \tanh(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h)$$

**Piece 6 — turn the summary into an actual prediction.**
$h_t$ is a private internal number, capped between $-1$ and $1$. But often you want an actual prediction on a different scale (a probability, a price, whatever). So you pass $h_t$ through one more dial-and-baseline pair:
$$\hat{y}_t = W_{hy}\,h_t + b_y$$
This is called the **readout step** — it doesn't feed back into the recurrence, it just reads the current summary out into an answer.

That's the whole model. Two steps, six numbers to tune ($W_{xh}, W_{hh}, b_h, W_{hy}, b_y$ — five weights, plus the reused $h_{t-1}$), applied again and again as $t$ increases.

In the general case, $x_t$ is a vector, $h_t$ is a vector, and $W_{xh}$, $W_{hh}$, $W_{hy}$ are matrices instead of single numbers. Nothing about the logic above changes — "scale and add" becomes "matrix-multiply and add," and $\tanh$ is just applied to every entry of the vector separately. To keep the arithmetic fully visible by hand, this chapter uses the smallest possible version: everything below is a scalar (a single number).

---

## Part B: Setting up numbers

Our sequence, from Chapter 1 — three days of wetness scores:

$$x_1 = 1.0, \quad x_2 = 2.0, \quad x_3 = 3.0$$

Pick some (untrained, arbitrary — just for illustration) weights:

| Weight | Value | Meaning |
|---|---|---|
| $W_{xh}$ | 0.5 | how much the current input shifts the summary |
| $W_{hh}$ | 0.8 | how much the previous summary carries forward |
| $b_h$ | 0.1 | a constant offset |
| $W_{hy}$ | 1.0 | how the summary maps to the output |
| $b_y$ | 0.0 | output offset |

Starting hidden state: $h_0 = 0$ (nothing has been read yet — day 0 has no history behind it).

---

## Part C: Hand-computing every step, in full detail

### Step 1: $t = 1$

**Update step.** Plug $x_1=1.0$ and $h_0=0$ into $z_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h$:

$$z_1 = (0.5)(1.0) + (0.8)(0) + 0.1$$

Compute each term separately:
- Input term: $(0.5)(1.0) = 0.5$
- Memory term: $(0.8)(0) = 0$ — there's no memory yet, so this contributes nothing
- Bias: $0.1$

$$z_1 = 0.5 + 0 + 0.1 = 0.6$$

**Squash it:**
$$h_1 = \tanh(0.6)$$

To see roughly where this lands without a calculator: $\tanh(0) = 0$ and $\tanh(1) \approx 0.762$, and $\tanh$ is roughly linear (slope $\approx 1$) near 0, so $\tanh(0.6)$ should land a bit under $0.6$ itself — squashing has started nudging it down, but only slightly since $0.6$ isn't very large yet. The precise value:

$$h_1 = \tanh(0.6) \approx 0.537$$

**Readout step.** Plug $h_1$ into $\hat y_t = W_{hy}h_t + b_y$:

$$\hat{y}_1 = (1.0)(0.537) + 0 = 0.537$$

*(Here $W_{hy}=1.0$ and $b_y=0$, so the readout is just a pass-through of $h_1$ — this was chosen deliberately, to keep this first pass simple. In general $\hat y_t \neq h_t$.)*

**Read this as:** the network saw wetness score 1.0, had no prior memory to draw on ($h_0=0$), and formed an initial summary of 0.537.

### Step 2: $t = 2$

**Update step.** Now $h_{t-1}$ is no longer zero — it's $h_1 = 0.537$, carrying forward everything step 1 produced.

$$z_2 = (0.5)(2.0) + (0.8)(0.537) + 0.1$$

Compute each term:
- Input term: $(0.5)(2.0) = 1.0$
- Memory term: $(0.8)(0.537) = 0.4296 \approx 0.430$
- Bias: $0.1$

$$z_2 = 1.0 + 0.430 + 0.1 = 1.530$$

**Squash it:**
$$h_2 = \tanh(1.530) \approx 0.910$$

Notice the squashing effect is stronger here: $z_2 = 1.530$ is a much bigger raw number than $h_2 = 0.910$, because $\tanh$ compresses harder the further you get from 0.

**Readout step:**
$$\hat{y}_2 = (1.0)(0.910) + 0 = 0.910$$

**The key thing to notice:** $h_2$ is *not* just a function of $x_2$. The term $(0.8)(0.537)$ pulls in everything the network produced at step 1. This is the "memory" — day 2's summary is built on top of day 1's summary, not from scratch. If you deleted that one term, $h_2$ would have no idea day 1 ever happened.

### Step 3: $t = 3$

**Update step.**

$$z_3 = (0.5)(3.0) + (0.8)(0.910) + 0.1$$

Compute each term:
- Input term: $(0.5)(3.0) = 1.5$
- Memory term: $(0.8)(0.910) = 0.728$
- Bias: $0.1$

$$z_3 = 1.5 + 0.728 + 0.1 = 2.328$$

**Squash it:**
$$h_3 = \tanh(2.328) \approx 0.981$$

Notice how close $h_3$ is getting to the ceiling of 1. $\tanh$'s range is $(-1,1)$, and as $z_t$ keeps growing, $h_t$ keeps approaching (but never reaching) 1 — the squashing gets more and more aggressive the further out you push it. This is worth remembering for Chapter 4, on vanishing gradients: once $h_t$ is this close to $\pm1$, $\tanh$'s slope there is nearly flat, and that flatness is exactly what causes gradients to shrink during backprop.

**Readout step:**
$$\hat{y}_3 = (1.0)(0.981) + 0 = 0.981$$

---

## Everything in one table

| $t$ | $x_t$ | Input term $W_{xh}x_t$ | Memory term $W_{hh}h_{t-1}$ | $z_t$ (sum + bias) | $h_t=\tanh(z_t)$ | $\hat{y}_t$ |
|---|---|---|---|---|---|---|
| 1 | 1.0 | 0.500 | 0.000 | 0.600 | 0.537 | 0.537 |
| 2 | 2.0 | 1.000 | 0.430 | 1.530 | 0.910 | 0.910 |
| 3 | 3.0 | 1.500 | 0.728 | 2.328 | 0.981 | 0.981 |

Rising wetness scores → rising hidden state → rising predicted rain probability. That's the network doing something sensible, even with arbitrary, untrained weights.

## The picture

```
h0=0 --[W_hh]--> h1=0.537 --[W_hh]--> h2=0.910 --[W_hh]--> h3=0.981
        ^                    ^                    ^
      [W_xh]               [W_xh]               [W_xh]
        |                    |                    |
      x1=1.0               x2=2.0               x3=3.0
```

Same $W_{xh}$, same $W_{hh}$, same $b_h$ used at every arrow. Nothing new is learned per timestep — one shared rule, applied three times.

## What to notice, and carry forward

1. **$h_{t-1}$ shows up inside every $z_t$.** This is the only thing that makes it "recurrent" — the previous output feeds back in as an input.
2. **Weights are shared across time.** $W_{xh}$ at $t=3$ is the exact same number as $W_{xh}$ at $t=1$. This is *why* the network can handle sequences of any length — you're not adding new parameters per timestep, just reapplying the same three numbers.
3. **$\tanh$ keeps $h_t$ bounded in $(-1,1)$ no matter how many timesteps pass** — but it also means $h_t$ saturates (flattens out near $\pm 1$) if $z_t$ gets large, and that flat region is where gradients start to vanish.
4. **The chain of $h_t$'s is what we'll differentiate through in Chapter 3.** Because $h_3$ depends on $h_2$, which depends on $h_1$, which depends on $h_0$ — a gradient flowing backward from the final output has to pass through every one of those links. That chain is exactly what makes training an RNN different from training an MLP.

---

## Part D: Interview questions (Google / Apple style)

**Q1. Why is there a nonlinearity ($\tanh$) inside the recurrence, rather than just summing $W_{xh}x_t + W_{hh}h_{t-1} + b_h$ directly as $h_t$?**
Two reasons. First, without a nonlinearity, stacking the recurrence over many timesteps collapses to a single linear map — a purely linear RNN could be rewritten as one big matrix multiply on the raw input sequence, so it couldn't represent nonlinear temporal patterns no matter how deep in time it goes. Second, unboundedly growing $z_t$ would make $h_t$ explode over long sequences; squashing keeps it numerically stable.

**Q2. Why $\tanh$ specifically, and not sigmoid or ReLU, for the vanilla RNN hidden state?**
$\tanh$ is zero-centered ($\tanh(0)=0$), which keeps the hidden state's average magnitude balanced around zero rather than always positive — this tends to make optimization better-behaved than sigmoid (which saturates between 0 and 1 and is not zero-centered). ReLU is avoided in the *vanilla* RNN because unbounded positive activations, reused every timestep via $W_{hh}$, can explode across a long sequence; $\tanh$'s bounded range is a built-in safety valve (though ReLU RNNs are used in some architectures with careful initialization).

**Q3. What's the difference between $h_t$ and $\hat y_t$, and why keep them separate?**
$h_t$ is the internal, recurrent state — it's what gets passed forward and is bounded to $(-1,1)$ by $\tanh$. $\hat y_t$ is a *task-specific readout* of that state, on whatever scale the task needs (unbounded for regression, or fed through softmax for classification). Keeping them separate means the same recurrent core can be reused with different readout heads for different tasks, and the internal state isn't artificially constrained to the output's scale.

**Q4. Why are the weights ($W_{xh}, W_{hh}, b_h$) shared across every timestep instead of learning separate weights per position?**
Two reasons interviewers look for: (1) it lets the model generalize to sequences of any length, since you're not adding parameters per timestep; (2) it enforces the inductive bias that "the rule for updating memory given new input" should be the same regardless of *when* in the sequence you are — e.g., the pattern for combining "today's word" with "the sentence so far" shouldn't depend on whether today is word 3 or word 30.

**Q5. If $h_0$ were initialized to a large random value instead of 0, how would that change $h_1$?**
$h_1 = \tanh(W_{xh}x_1 + W_{hh}h_0 + b_h)$ — a large $h_0$ would shift $z_1$ before $\tanh$, potentially pushing $h_1$ toward saturation ($\pm1$) regardless of $x_1$. Since $\tanh$'s gradient is near zero in the saturated region, this can hurt early training (vanishing gradients from step 1 onward), which is part of why $h_0=0$ is the standard default.

**Q6. Given this chapter's numbers, if $W_{hh}$ were 0 instead of 0.8, what would $h_2$ and $h_3$ become, and why does that matter conceptually?**
With $W_{hh}=0$, the memory term drops out of every $z_t$, so $z_t = W_{xh}x_t + b_h$ depends only on the current input — $h_2 = \tanh((0.5)(2.0)+0.1) = \tanh(1.1) \approx 0.800$, and $h_3 = \tanh((0.5)(3.0)+0.1) = \tanh(1.6) \approx 0.922$. Conceptually, $W_{hh}=0$ turns the "recurrent" network into a stateless per-timestep function — it's the cleanest way to see that $W_{hh}$ is *entirely* what gives the network memory.

**Q7. This example uses scalars for $x_t$ and $h_t$. What actually changes when you move to the real vector/matrix case?**
The scalar multiplications become matrix-vector products: $W_{xh}$ becomes a matrix of shape (hidden_size × input_size), $W_{hh}$ becomes (hidden_size × hidden_size), and $b_h$ becomes a vector of length hidden_size. $\tanh$ is applied elementwise to the resulting vector. The *logic* — blend today's input with yesterday's summary, add bias, squash — is identical; only the arithmetic scales up from numbers to matrices.

**Q8. What happens to $\hat y_t$'s expressiveness if $W_{hy}=1$ and $b_y=0$, as used in this toy example — is that a real limitation?**
With those specific values, $\hat y_t = h_t$ exactly, so the readout adds nothing beyond what $\tanh$ already bounds to $(-1,1)$. That's fine for this toy example (chosen purely so $\hat y_t$ is easy to read off), but in a real model $W_{hy}, b_y$ are learned separately from $W_{xh}, W_{hh}, b_h$ precisely so the output can be rescaled to whatever range the task needs, independent of the internal state's $(-1,1)$ bound.

## What's ahead

Chapter 3 takes this exact forward pass and runs backprop through it — computing how a change in $W_{hh}$ at $t=1$ affects the loss at $t=3$, step by step, using the same numbers above.

---

**One-line summary:** the forward pass is just the loop from Chapter 1 with real numbers plugged in — at each step, blend the new input with the previous summary (two dials, one bias), squash with $\tanh$ to keep it bounded, optionally read out a prediction on a different scale, and carry the new bounded summary forward.
