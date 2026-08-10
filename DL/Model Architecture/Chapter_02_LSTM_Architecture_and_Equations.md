# Chapter 2: The LSTM Architecture & Equations

## The big idea: two states instead of one

A vanilla RNN carries one thing forward: the hidden state $h_t$. An LSTM carries **two**:

- $c_t$ — the **cell state**. This is the conveyor belt from Chapter 1: a running memory that information can travel along with minimal forced change.
- $h_t$ — the **hidden state**. This is what gets exposed to the outside world (used for predictions, passed to the next layer) — a filtered view of the cell state.

Everything in this chapter is about how three small gates control the flow of information between these two states.

## The three gates, in plain language first

| Gate | Question it answers | Sigmoid output near 0 means | Near 1 means |
|---|---|---|---|
| **Forget gate** $f_t$ | How much of the old cell state should I keep? | throw it away | keep it fully |
| **Input gate** $i_t$ | How much of the new candidate info should I add? | ignore the new info | fully add it |
| **Output gate** $o_t$ | How much of the cell state should I expose as hidden state? | hide it | expose it fully |

All three are small learned layers, sigmoid-activated (output between 0 and 1), computed the same way — just with their own weights.

## Building the equations, one gate at a time

**1. Forget gate.** Decides what fraction of the old cell state $c_{t-1}$ to keep:

$$f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$$

**2. Input gate.** Decides how much of the *new* information to let in:

$$i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$$

**3. Candidate cell state.** The actual *content* being proposed for addition — note this uses $\tanh$, not sigmoid, because it's a value (bounded between -1 and 1), not a gate (bounded between 0 and 1):

$$\tilde{c}_t = \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)$$

**4. Cell state update — the most important equation in this chapter:**

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

($\odot$ means elementwise multiply — in our scalar toy example this is just ordinary multiplication.)

Read this in plain English: *the new memory equals (how much of the old memory to keep) plus (how much of the new candidate to add)*. Notice what's **not** here: no $\tanh$ squashing the whole sum, no single weight matrix multiplying the entire previous state. The old cell state $c_{t-1}$ passes through mostly untouched if $f_t\approx1$ — just scaled, not transformed. This is precisely the "conveyor belt" behavior promised in Chapter 1, and it's the mechanical reason LSTM fixes vanishing gradients (full derivation in Chapter 4).

**5. Output gate.** Decides how much of the cell state to reveal as this step's hidden state:

$$o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$$

**6. Hidden state.** The cell state is squashed through $\tanh$ (to bound it between -1 and 1, same role $\tanh$ played in the vanilla RNN) and then filtered by the output gate:

$$h_t = o_t \odot \tanh(c_t)$$

## All six equations together

$$f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$$
$$i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$$
$$\tilde{c}_t = \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

Four sets of weights ($W_{xf},W_{hf}$; $W_{xi},W_{hi}$; $W_{xc},W_{hc}$; $W_{xo},W_{ho}$), four biases. Every one of them is learned during training — the network figures out on its own what "forget," "input," and "output" should mean for the task at hand. Nobody hand-designs what the gates do; they're just ordinary trainable parameters, structured so that *this particular shape of computation* tends to preserve long-range information well.

## The picture

```
                    c_{t-1} ---(x forget gate)---+---(+)---> c_t
                                                   |           |
                                    i_t x c~_t ----+           |
                                                                v
   x_t, h_{t-1} --> [forget, input, candidate, output gates]  tanh
                                                                |
                                                          (x output gate)
                                                                |
                                                                v
                                                               h_t
```

The top row (the $c_{t-1} \to c_t$ line) is the conveyor belt: multiply by the forget gate, add the gated candidate, done — no squashing of the whole running total. The bottom machinery (gates) reads $x_t$ and $h_{t-1}$ and decides how much to let through at each junction.

## Why four separate weight sets, not one?

Each gate needs to learn a *different* function of the same inputs $(x_t, h_{t-1})$ — "should I forget" is a different question from "should I output," even though both look at the same information. Giving each gate its own weights lets the network learn four independent answers instead of being forced to reuse one.

## Parameter count, just to be aware of it

A vanilla RNN has 3 weight matrices ($W_{xh}, W_{hh}, W_{hy}$ plus biases). An LSTM has 4 gates $\times$ 2 matrices each = 8 weight matrices for the recurrent part, plus whatever output layer you attach on top. **Roughly 4x the parameters of a vanilla RNN of the same hidden size** — this is the price paid for the gating mechanism, and it's a fact worth having ready in an interview.

## What's ahead

Chapter 3 hand-computes every one of these six equations on the same toy sequence ($x=[1,2,3]$) used throughout — you'll see actual numbers for $f_t, i_t, \tilde{c}_t, c_t, o_t, h_t$ at each timestep. Chapter 4 then traces a gradient through this structure and shows numerically why it doesn't vanish the way the vanilla RNN's did.

---

**One-line summary:** LSTM keeps two running states — a cell state that flows forward with minimal forced transformation (the conveyor belt) and a hidden state that's a gated, filtered view of it — controlled by three learned sigmoid gates (forget, input, output) plus one tanh candidate layer.
