# Chapter 5: Fixes & Gated Variants (A Bridge to LSTM/GRU)

## Recap

Chapter 4 showed: gradients flowing backward through time are a product of many factors. Too many factors below 1 → vanishing. Too many above 1 → exploding. Two different failure modes, same root cause: repeated multiplication by $(1-h_s^2)\cdot W_{hh}$ at every step.

This chapter covers two practical patches, and then previews the real architectural fix.

## Fix 1: Gradient clipping (for exploding gradients)

The idea is blunt and effective: after computing the gradient, if its size (norm) is too large, shrink it back down before using it to update weights.

$$g \leftarrow g \cdot \frac{\text{threshold}}{\|g\|} \quad \text{if } \|g\| > \text{threshold}$$

**Worked example.** Recall from Chapter 4: with $W_{hh}=3.0$ and an unsaturated hidden state, the gradient over 10 steps grew to about $59{,}000$. Say we set a clipping threshold of $5.0$:

$$g_{\text{clipped}} = 59{,}000 \times \frac{5.0}{59{,}000} = 5.0$$

The *direction* of the gradient is preserved (it still points the same way in weight-space), but its *magnitude* is capped. This stops a single bad step from blowing up the weights, without touching the underlying cause.

**What clipping does and doesn't fix:** it caps the damage from exploding gradients. It does **nothing** for vanishing gradients — a gradient that's already near zero doesn't get any bigger by capping large ones elsewhere. It's a safety rail, not a cure.

## Fix 2: Truncated BPTT (a computational compromise)

Backpropagating through a 10,000-step sequence means storing 10,000 hidden states and running the chain rule back through all of them — expensive, and given Chapter 4, mostly pointless past the first dozen or so steps anyway (the gradient's already vanished).

**Truncated BPTT**: instead of backpropagating through the entire sequence, only backpropagate through the last $k$ steps (say $k=20$), even though the forward pass keeps running across the whole sequence.

```
Forward pass:  h0 -> h1 -> h2 -> ... -> h998 -> h999 -> h1000
Backward pass (truncated, k=20):                <------------->
                                          only these last 20 steps
                                          get gradient signal
```

This is a **speed and memory compromise**, not a fix for vanishing gradients — it just stops paying computational cost for gradient signal that would've vanished anyway. It doesn't help the network learn longer-range dependencies; it only makes training long sequences tractable.

## Why neither fix solves the real problem

Both patches operate *after* the fact — clipping reacts to an already-exploded gradient, truncation just stops computing a gradient that would've vanished. Neither changes the mechanism that causes the shrink/explode tradeoff in the first place: **the hidden state gets multiplied by $W_{hh}$ and squashed through $\tanh$ at every single step, with no alternative path for information to travel.**

To actually fix this, the architecture itself needs to change.

## The real fix: give information a path that doesn't shrink

Here's the core design change gated architectures (LSTM, GRU) make, in one sentence:

> Instead of forcing *all* information through a multiply-and-squash bottleneck at every step, add a second pathway that can carry information forward with **little or no shrinking**, controlled by learned gates that decide what to keep, what to throw away, and what to update.

Think of it like this: in the vanilla RNN, the hidden state is like passing a note through a chain of people who each rewrite it slightly (multiply and squash). Rewrite it enough times and the original message is unrecognizable — that's vanishing. LSTM instead adds a **conveyor belt** (called the "cell state") running alongside the note-passing — most people just let it pass through untouched, and only add or remove small, deliberate pieces of information via **gates**. Because the conveyor belt doesn't get squashed through $\tanh$ and multiplied by a weight at every single step, gradients can flow along it much further before vanishing.

That's the entire intuition. The gates are themselves small neural networks (sigmoid-based, outputting values between 0 and 1) that learn:

- **What to forget** from the cell state (forget gate)
- **What new information to add** (input gate)
- **What to output** as the next hidden state (output gate)

We won't derive the full LSTM equations here — that's outside the current curriculum's scope — but this is the idea to carry forward: **gating exists specifically to fix the vanishing gradient problem, by giving gradients a path through time that isn't forced through repeated shrinking multiplications.**

## Summary table: the three responses to Chapter 4's problem

| Fix | Solves exploding? | Solves vanishing? | What it actually does |
|---|---|---|---|
| Gradient clipping | Yes | No | Caps gradient magnitude after computing it |
| Truncated BPTT | Indirectly (bounds computation) | No | Limits how far back gradients are computed at all |
| Gated architectures (LSTM/GRU) | Yes | **Yes** | Changes the architecture so information has a low-decay path through time |

## What's ahead

Chapter 6 catalogs the 7 shapes RNN architectures come in — how many inputs, how many outputs, and how BPTT differs depending on that shape. Chapter 7 is your interview cheat sheet. Chapter 8 covers RNNs on tabular data. Chapter 9 builds everything from scratch in NumPy and PyTorch.

---

**One-line summary:** gradient clipping patches exploding gradients, truncated BPTT patches the cost of long sequences, but only an architectural change — giving information a low-decay path through time via gates — actually solves vanishing gradients. That's what LSTM and GRU are for.
