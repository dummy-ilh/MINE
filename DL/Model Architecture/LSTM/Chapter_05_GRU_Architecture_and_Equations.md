# Chapter 5: The GRU Architecture & Equations

## The pitch: same idea, fewer parts

GRU (Gated Recurrent Unit) was designed to capture LSTM's core benefit — a gated path that lets information flow with minimal forced decay — using a **simpler** structure: no separate cell state, and 2 gates instead of 3.

| | LSTM | GRU |
|---|---|---|
| States carried forward | 2 ($c_t$, $h_t$) | 1 ($h_t$ only) |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Weight matrices (recurrent part) | 8 | 6 |

If LSTM's conveyor belt is a separate physical lane running alongside the hidden state, GRU merges the two lanes into one — but keeps the same core trick: gates that decide how much of the past to keep versus overwrite.

## The two gates, in plain language

| Gate | Question it answers |
|---|---|
| **Reset gate** $r_t$ | When computing a new candidate, how much of the *old* hidden state should influence it? |
| **Update gate** $z_t$ | How much of the *old* hidden state should I carry forward unchanged, versus replace with the new candidate? |

Notice the update gate in GRU does the job of **both** the forget gate and the input gate in LSTM, combined into one number — if the update gate says "keep 80% old," it's automatically also saying "let in 20% new." One gate, two decisions, coupled together (a real design tradeoff — more on this in Chapter 8).

## Building the equations

**1. Reset gate.** Decides how much past state to use when forming the new candidate:

$$r_t = \sigma(W_{xr}x_t + W_{hr}h_{t-1} + b_r)$$

**2. Update gate.** Decides the overall keep-old vs. take-new blend:

$$z_t = \sigma(W_{xz}x_t + W_{hz}h_{t-1} + b_z)$$

**3. Candidate hidden state.** Notice $r_t$ appears *inside* this equation, gating how much of $h_{t-1}$ contributes to the candidate:

$$\tilde{h}_t = \tanh\big(W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h\big)$$

**4. Final hidden state — the key equation.** A direct blend between old and candidate, controlled by $z_t$:

$$h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t$$

That's the whole architecture: 4 equations instead of LSTM's 6, and only one running state.

## Reading the final equation carefully

$$h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde{h}_t$$

This is a **linear interpolation** between old and new: if $z_t=0$, $h_t=h_{t-1}$ exactly (nothing changes — perfect memory). If $z_t=1$, $h_t=\tilde{h}_t$ exactly (fully overwritten — no memory carried). Anything in between is a blend. This is structurally the same trick as LSTM's forget-gate path: an additive (not squash-through-tanh) update that lets $h_{t-1}$ pass through with minimal forced transformation when $z_t$ is small.

## Why the reset gate is inside the candidate, not the final blend

The reset gate's job is different from the update gate's. It doesn't decide "how much old state to keep" — it decides "when computing a *proposal* for new content, how relevant is the old state to that proposal." If $r_t\approx0$, the candidate $\tilde{h}_t$ is computed almost as if starting fresh from $x_t$ alone, useful when the model wants to "forget everything and start a new topic." If $r_t\approx1$, the old state fully informs the candidate, useful for smooth, continuous updates.

## All four equations together

$$r_t = \sigma(W_{xr}x_t + W_{hr}h_{t-1} + b_r)$$
$$z_t = \sigma(W_{xz}x_t + W_{hz}h_{t-1} + b_z)$$
$$\tilde{h}_t = \tanh\big(W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h\big)$$
$$h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde{h}_t$$

## The picture

```
                 h_{t-1} ---(x (1-z_t))---+---(+)---> h_t
                    |                      |
              (x r_t)                z_t x h~_t
                    |                      |
                    v                      |
             [candidate h~_t] ------------+
                    ^
             x_t, h_{t-1}, r_t
```

Compare to LSTM's picture in Chapter 2: same top-level idea (an additive path that bypasses forced squashing), but here the "conveyor belt" *is* the hidden state itself, not a separate cell state.

## Parameter count

GRU: 2 gates + 1 candidate = 3 functions $\times$ 2 matrices each = 6 weight matrices, vs. LSTM's 8. **Roughly 25% fewer recurrent parameters than LSTM**, for a hidden state of the same size — a fact worth having ready in an interview, and the main practical reason GRU is sometimes preferred when data or compute is limited.

## What's ahead

Chapter 6 hand-computes this exact set of equations on the same toy sequence, and puts the resulting $h_t$ values side by side with both the vanilla RNN's and LSTM's from earlier chapters.

---

**One-line summary:** GRU keeps a single hidden state and uses two gates — reset (controls how much old state feeds into the new candidate) and update (blends old state and candidate into the final output) — achieving LSTM's core "additive, low-decay path" trick with 25% fewer recurrent parameters and no separate cell state.
