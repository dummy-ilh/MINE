# Chapter 1: Why Vanilla RNNs Need Gates

## Where we left off

A vanilla RNN updates its hidden state like this:

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

Every timestep, the entire hidden state gets rebuilt from scratch: multiply by $W_{hh}$, add the new input, squash through $\tanh$. This works, but it has one structural flaw — that squash-and-multiply step happens whether it's useful or not, at *every* single timestep, to the *entire* hidden state.

## The problem, recapped numerically

Take the toy sequence $x=[1.0, 2.0, 3.0]$, with $W_{xh}=0.5$, $W_{hh}=0.8$, $b_h=0.1$. The forward pass gives:

$$h_1 \approx 0.537, \quad h_2 \approx 0.910, \quad h_3 \approx 0.981$$

Now trace a gradient backward from $h_3$ to $h_0$. Each step backward multiplies in a factor $(1-h_t^2)\cdot W_{hh}$:

| step | $(1-h_t^2)$ | $\times\ W_{hh}=0.8$ | factor |
|---|---|---|---|
| 1 | 0.712 | | 0.570 |
| 2 | 0.172 | | 0.138 |
| 3 | 0.038 | | 0.030 |

$$\frac{\partial h_3}{\partial h_0} = 0.570 \times 0.138 \times 0.030 \approx 0.0024$$

In just 3 steps, the gradient has shrunk to roughly a quarter of one percent of where it started. Stretch this to 20 or 50 steps (a real sentence, a real time series) and the gradient reaching the earliest timesteps is effectively zero. The network can't learn anything about long-range dependencies — not because the information isn't useful, but because the *architecture itself* destroys the gradient before it can get there.

## The core design flaw

Look again at the forward equation. There is exactly **one path** for information to travel from $h_{t-1}$ to $h_t$: multiply by $W_{hh}$, add stuff, squash through $\tanh$. Every single piece of information — whether it matters or not — is forced through that same bottleneck, every single step.

Compare this to how you actually read a long document. You don't re-derive your entire understanding from scratch at every sentence. Most of what you knew a paragraph ago, you just... keep. You only update your understanding when something actually changes it. Selective retention, not full rewrite every step.

Vanilla RNNs have no mechanism for "just keep this part unchanged." Everything gets multiplied and squashed, always.

## The fix, in one sentence

> Add a second pathway that information can travel along with little or no forced transformation, and use small learned gates to decide, at each step: what to forget, what to add, and what to output.

That's the entire idea behind **LSTM** (Long Short-Term Memory). The "second pathway" is called the **cell state** — think of it as a conveyor belt running alongside the hidden state, which most information rides on largely untouched, with small, deliberate edits applied by gates rather than a full rewrite at every step.

**GRU** (Gated Recurrent Unit), covered starting in Chapter 5, achieves something similar with a simpler design — no separate cell state, fewer gates, but the same underlying idea: *let the network learn what to keep and what to change, instead of forcing a full rewrite every step.*

## What a "gate" actually is

Nothing exotic — a gate is just a small neural network layer with a **sigmoid** output:

$$\text{gate} = \sigma(W \cdot [\text{input}] + b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

Sigmoid squashes any number into $(0, 1)$. A gate value near 0 means "block this — let almost nothing through." A gate value near 1 means "let this pass essentially unchanged." The network *learns* what these gate values should be, per input, per timestep — it's not a fixed rule, it's a small trainable layer just like anything else in the network.

## What's ahead

| Chapter | What you'll get |
|---|---|
| 2 | Full LSTM architecture: forget gate, input gate, output gate, cell state — equations, one at a time |
| 3 | LSTM forward pass — hand-computed on the same toy sequence |
| 4 | Why the cell state fixes vanishing gradients — shown numerically, contrasted with Chapter 1's 0.0024 |
| 5 | GRU architecture: reset gate, update gate — built as "LSTM's simpler cousin" |
| 6 | GRU forward pass — same toy sequence, compared side-by-side with LSTM |
| 7 | Bidirectional variants — BiLSTM and BiGRU |
| 8 | LSTM vs. GRU vs. Bi-variants — when to use which, plus common tweaks |
| 9 | Interview cheat sheet + Q&A |
| 10 | Built from scratch — NumPy, then PyTorch, verified numerically |

---

**One-line summary:** vanilla RNNs force every piece of information through the same multiply-and-squash bottleneck at every step, which is exactly what causes vanishing gradients; LSTM and GRU fix this by adding a low-decay pathway and learned gates that decide what to keep, forget, and update.
