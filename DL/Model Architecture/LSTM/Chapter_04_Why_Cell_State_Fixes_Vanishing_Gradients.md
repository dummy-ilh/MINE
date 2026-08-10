# Chapter 4: Why the Cell State Fixes Vanishing Gradients

## Recall the vanilla RNN's problem

Every backward step in a vanilla RNN forces a multiply by $(1-h_t^2)\cdot W_{hh}$ — a fixed, uncontrollable shrink factor baked into the same computation that produces the forward output. Over 3 steps, that gradient shrank to about **0.00236** (0.24%) of its starting size. There was no way for the network to choose otherwise — $\tanh'$ is always $\le1$, and $W_{hh}$ is one fixed number doing double duty for both content and gradient flow.

## The cell state's recurrence

Recall from Chapter 2:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Look at how $c_{t-1}$ enters this equation: it's multiplied by $f_t$ and then **added**, not squashed through $\tanh$. The direct path from $c_{t-1}$ to $c_t$ is:

$$\frac{\partial c_t}{\partial c_{t-1}} \approx f_t$$

(This is the *dominant* term — technically $f_t$, $i_t$, and $\tilde{c}_t$ also depend on $h_{t-1}$, which depends on $c_{t-1}$, giving smaller secondary paths, same as the full BPTT treatment in the RNN curriculum. But $f_t$ is the term that matters for the headline result, so we isolate it here.)

Chain this across $n$ steps and the gradient from $c_t$ back to $c_{t-n}$ is approximately:

$$\frac{\partial c_t}{\partial c_{t-n}} \approx \prod_{s=t-n+1}^{t} f_s$$

**This is the same "product of many factors" structure as the vanilla RNN** — but with one crucial difference: $f_s$ is a **separately learned sigmoid gate**, not a fixed shrink factor forced by the same weights that compute content. The network can push $f_s$ close to 1 whenever "preserve this memory" is the right behavior — and when it does, the product barely shrinks at all.

## The numbers, side by side

Using the gate values computed in Chapter 3: $f_1=0.646,\ f_2=0.761,\ f_3=0.853$.

$$\prod_{s=1}^{3} f_s = 0.646 \times 0.761 \times 0.853 \approx 0.4195$$

| | Vanilla RNN (Ch. 4 of RNN curriculum) | LSTM (this chapter) |
|---|---|---|
| Per-step factors | 0.570, 0.138, 0.030 | 0.646, 0.761, 0.853 |
| Cumulative product over 3 steps | **0.00236** | **0.4195** |
| Gradient retained | ~0.2% | ~42% |

Same number of steps, **~178x more gradient survives** through the LSTM's cell-state path. And this is with untrained, arbitrary weights — a network actually trained to remember long-range dependencies would push its forget gates even closer to 1.

## Why this gap gets more dramatic over longer sequences

Project both patterns forward. Suppose the vanilla RNN's typical per-step factor stays around 0.15 (roughly what we saw), while the LSTM's forget gate — once trained for a task needing long memory — settles around 0.95:

| Sequence length | Vanilla RNN gradient remaining | LSTM gradient remaining (forget gate ≈ 0.95) |
|---|---|---|
| 3 steps | $0.15^3 \approx 0.0034$ | $0.95^3 \approx 0.857$ |
| 10 steps | $0.15^{10}\approx5.8\times10^{-9}$ | $0.95^{10}\approx0.599$ |
| 50 steps | effectively 0 | $0.95^{50}\approx0.077$ |

At 50 steps, the vanilla RNN's gradient is numerically indistinguishable from zero. The LSTM's is down to about 8% — much smaller, but very much alive, and if the forget gate learns to sit even closer to 1 (say 0.99), it stays around 60% even after 50 steps ($0.99^{50}\approx0.605$).

## Why the forget gate *can* learn to sit near 1 (and $W_{hh}$ couldn't)

This is the actual mechanism worth remembering: in a vanilla RNN, $W_{hh}$ has to simultaneously (a) transform the hidden state into something useful for the task, and (b) whatever value it ends up at also directly controls gradient decay through $(1-h_t^2)\cdot W_{hh}$. Those two jobs are coupled — you can't tune one without affecting the other.

In an LSTM, the forget gate is a **separate function** with its own weights ($W_{xf}, W_{hf}, b_f$), independent from the weights that compute the candidate content ($W_{xc}, W_{hc}, b_c$). The network is free to learn "keep almost everything" (forget gate near 1) for the memory pathway, while *independently* learning whatever content-update behavior the input gate and candidate need. Decoupling "how much to preserve" from "what to compute" is the actual architectural fix — not just "adding more parameters."

## An important caveat

LSTM doesn't make vanishing gradients *impossible* — if a task genuinely requires forgetting (forget gate learns values near 0), gradient flow through that path is deliberately cut, which is correct behavior, not a flaw. What LSTM provides is the **option** of near-lossless gradient flow when the task calls for it — an option vanilla RNNs structurally do not have.

## What's ahead

Chapter 5 introduces GRU — a simpler gated architecture (2 gates instead of 3, no separate cell state) that achieves a similar effect with fewer parameters.

---

**One-line summary:** the cell state's update is additive rather than squash-and-multiply, so the gradient path through it is a product of learned forget-gate values instead of a fixed shrink factor — and because the forget gate is decoupled from content computation, the network can learn to keep it near 1 whenever long-range memory matters, which a vanilla RNN structurally cannot do.
