# Chapter 2: The Vanilla RNN Forward Pass

## Recall the loop

From Chapter 1:

```
hidden_state = 0
for each input x_t:
    hidden_state = update(hidden_state, x_t)
    output = readout(hidden_state)
```

This chapter turns `update` and `readout` into real numbers. We'll hand-compute every single value.

## The two equations

**Update step** (builds the new summary from the old summary + new input):

$$h_t = \tanh(W_{xh} \, x_t + W_{hh} \, h_{t-1} + b_h)$$

**Readout step** (turns the summary into a prediction):

$$\hat{y}_t = W_{hy} \, h_t + b_y$$

In the general case, $x_t$ is a vector, $h_t$ is a vector, and $W_{xh}$, $W_{hh}$, $W_{hy}$ are matrices. To keep the arithmetic fully visible, this chapter uses the smallest possible version: everything is a scalar (a single number). The mechanics are identical either way — matrices just mean "do this same kind of blending, but across more numbers at once."

## Setting up numbers

Our sequence, from Chapter 1:

$$x_1 = 1.0, \quad x_2 = 2.0, \quad x_3 = 3.0$$

Pick some (untrained, arbitrary — just for illustration) weights:

| Weight | Value | Meaning |
|---|---|---|
| $W_{xh}$ | 0.5 | how much the current input shifts the summary |
| $W_{hh}$ | 0.8 | how much the previous summary carries forward |
| $b_h$ | 0.1 | a constant offset |
| $W_{hy}$ | 1.0 | how the summary maps to the output |
| $b_y$ | 0.0 | output offset |

Starting hidden state: $h_0 = 0$ (nothing has been read yet).

## Step 1: $t = 1$

$$z_1 = W_{xh} \cdot x_1 + W_{hh} \cdot h_0 + b_h = (0.5)(1.0) + (0.8)(0) + 0.1 = 0.6$$

$$h_1 = \tanh(0.6) \approx 0.537$$

$$\hat{y}_1 = W_{hy} \cdot h_1 + b_y = (1.0)(0.537) + 0 = 0.537$$

Read this as: the network saw wetness score 1.0, had no prior memory to draw on ($h_0=0$), and formed an initial summary of 0.537.

## Step 2: $t = 2$

$$z_2 = W_{xh} \cdot x_2 + W_{hh} \cdot h_1 + b_h = (0.5)(2.0) + (0.8)(0.537) + 0.1 = 1.0 + 0.430 + 0.1 = 1.530$$

$$h_2 = \tanh(1.530) \approx 0.910$$

$$\hat{y}_2 = (1.0)(0.910) = 0.910$$

Notice what happened: $h_2$ isn't just a function of $x_2$. The term $(0.8)(0.537)$ pulls in everything the network learned from step 1. This is the "memory" — day 2's summary is built on top of day 1's summary, not from scratch.

## Step 3: $t = 3$

$$z_3 = W_{xh} \cdot x_3 + W_{hh} \cdot h_2 + b_h = (0.5)(3.0) + (0.8)(0.910) + 0.1 = 1.5 + 0.728 + 0.1 = 2.328$$

$$h_3 = \tanh(2.328) \approx 0.981$$

$$\hat{y}_3 = (1.0)(0.981) = 0.981$$

## Everything in one table

| $t$ | $x_t$ | $z_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h$ | $h_t = \tanh(z_t)$ | $\hat{y}_t$ |
|---|---|---|---|---|
| 1 | 1.0 | 0.600 | 0.537 | 0.537 |
| 2 | 2.0 | 1.530 | 0.910 | 0.910 |
| 3 | 3.0 | 2.328 | 0.981 | 0.981 |

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
3. **The chain of $h_t$'s is what we'll differentiate through in Chapter 3.** Because $h_3$ depends on $h_2$, which depends on $h_1$, which depends on $h_0$ — a gradient flowing backward from the final output has to pass through every one of those links. That chain is exactly what makes training an RNN different from training an MLP.

## What's ahead

Chapter 3 takes this exact forward pass and runs backprop through it — computing how a change in $W_{hh}$ at $t=1$ affects the loss at $t=3$, step by step, using the same numbers above.

---

**One-line summary:** the forward pass is just the loop from Chapter 1 with real numbers plugged in — at each step, blend the new input with the previous summary, squash with tanh, optionally read out a prediction, and carry the new summary forward.
