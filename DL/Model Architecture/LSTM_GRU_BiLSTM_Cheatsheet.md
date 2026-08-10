# LSTM / GRU / BiLSTM Interview Cheatsheet (One Page)

## 1. LSTM — memorize as "3 gates decide, 1 line updates memory"

$$f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f) \quad i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i) \quad o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)$$
$$\tilde{c}_t=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)$$
$$\boxed{c_t=f_t \cdot c_{t-1}+i_t \cdot \tilde{c}_t} \qquad h_t=o_t \cdot \tanh(c_t)$$

**Say it out loud:** "forget gate, input gate, output gate — all sigmoid, all same shape. Candidate is tanh, it's content not a gate. Cell state = keep×old + add×new. Hidden state = output gate times squashed cell state."

## 2. Replicate LSTM live (memorize these numbers)

$$x=[1,2,3] \quad W_{xf}{=}0.5,W_{hf}{=}0.3,b_f{=}0.1 \quad W_{xi}{=}0.4,W_{hi}{=}0.2,b_i{=}0 \quad W_{xc}{=}0.5,W_{hc}{=}0.8,b_c{=}0.1 \quad W_{xo}{=}0.6,W_{ho}{=}0.4,b_o{=}0$$

| $t$ | $f_t$ | $i_t$ | $\tilde{c}_t$ | $c_t$ | $o_t$ | $h_t$ |
|---|---|---|---|---|---|---|
| 1 | 0.646 | 0.599 | 0.537 | 0.322 | 0.646 | 0.201 |
| 2 | 0.761 | 0.699 | 0.851 | 0.839 | 0.783 | 0.536 |
| 3 | 0.853 | 0.787 | 0.966 | 1.477 | 0.882 | 0.795 |

**Trick to reproduce fast:** compute all 4 gate/candidate $z$'s the same way ($W_x x_t + W_h h_{t-1} + b$), squash 3 with sigmoid (f, i, o) and 1 with tanh ($\tilde c$), then $c_t = f_t c_{t-1}+i_t\tilde c_t$, then $h_t = o_t\tanh(c_t)$.

## 3. GRU — memorize as "2 gates, 1 blend"

$$r_t=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r) \qquad z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)$$
$$\tilde h_t=\tanh(W_{xh}x_t+W_{hh}(r_t\cdot h_{t-1})+b_h)$$
$$\boxed{h_t=(1-z_t)\cdot h_{t-1}+z_t\cdot \tilde h_t}$$

**Say it out loud:** "reset gate filters old state before it enters the candidate. Update gate linearly blends old and new — one number doing what LSTM splits into forget + input."

**Same toy numbers** ($W_{xr}{=}0.5,W_{hr}{=}0.3,b_r{=}0.1$; $W_{xz}{=}0.4,W_{hz}{=}0.2,b_z{=}0$; candidate same as LSTM's):

| $t$ | $r_t$ | $z_t$ | $\tilde h_t$ | $h_t$ |
|---|---|---|---|---|
| 1 | 0.646 | 0.599 | 0.537 | 0.322 |
| 2 | 0.768 | 0.704 | 0.861 | 0.701 |
| 3 | 0.859 | 0.793 | 0.969 | 0.914 |

## 4. Why gating fixes vanishing gradients — 1 sentence, 1 number

**Sentence:** "gradient path through $c_t$/$h_t$ is a product of *learned* gate values, not a fixed shrink factor — and since the gate is decoupled from content weights, the network can push it near 1 when long memory matters."

**Number:** vanilla RNN retains ~0.24% of gradient over 3 steps; LSTM retains ~42% (same toy example) — ~178x more, purely from decoupling "how much to keep" from "what to compute."

## 5. BiLSTM/BiGRU — memorize as "run twice, flip, concat"

$$h_t = [\overrightarrow{h_t} \, ; \, \overleftarrow{h_t}]$$

**Steps:** forward pass on $x$ in order → backward pass on $x$ reversed → reverse the backward output list to realign timesteps → concatenate at each $t$.

**One sentence:** "two independent networks, one each direction — needs the WHOLE sequence upfront, so never usable for real-time/streaming."

## 6. Parameter counts — memorize the ratio

| Cell | Recurrent matrices | vs. vanilla RNN |
|---|---|---|
| Vanilla RNN | 3 | 1x |
| GRU | 6 | 2x |
| LSTM | 8 | ~2.67x |
| Bi- (either) | 2× base cell | 2× base |

## 7. Decision tree — 2 questions

1. **Full sequence available upfront + future context helps?** → bidirectional.
2. **Limited data/compute or modest memory needs?** → GRU. Otherwise → LSTM.

## 8. Rapid-fire (under 15 sec each)

- **Why separate cell state?** Lets info travel additively, skipping the squash-every-step bottleneck that causes vanishing gradients.
- **Why tanh for candidate, sigmoid for gates?** Candidate is *content* (needs $\pm1$ range); gates answer "how much" (need $0$–$1$ range).
- **Why can $c_t$ exceed 1 but $h_t$ can't?** $c_t$ is a raw additive running sum, never squashed directly; $h_t=o_t\tanh(c_t)$ re-bounds it at the point of use.
- **What does GRU's update gate replace?** LSTM's forget + input gates, coupled into one: keep-fraction $=1{-}z_t$, add-fraction $=z_t$.
- **Does gating fully eliminate vanishing gradients?** No — a gate learning near 0 deliberately cuts that path; gating gives the *option* of near-lossless flow, not a guarantee.
- **Can BiLSTM run in real time?** No — backward pass needs the full sequence first.

## 9. Commonly reported Google/Apple-style questions

*(Aggregated patterns from public interview-prep sources, not verified company material — treat as representative.)*

**Q: Explain the mechanism, not just the buzzword, for why LSTM beats vanilla RNN on long sequences.**
A: Additive cell-state update ($c_t=f_tc_{t-1}+i_t\tilde c_t$) replaces the forced multiply-and-squash. The forget gate is a separately learned function, decoupled from content weights, so it can sit near 1 to preserve gradients — $W_{hh}$ in a vanilla RNN can't do that because it's coupled to both content and decay.

**Q: GRU vs LSTM — how do you choose in production?**
A: Comparable performance on most tasks. Default to GRU for efficiency (25% fewer params, faster train/inference), switch to LSTM for very long sequences with ample data where independently-controlled forget/input gates earn their extra cost.

**Q: What's the cost of going bidirectional?**
A: Roughly double the parameters and compute, and you lose real-time/streaming capability entirely — must wait for the full sequence before the backward pass can even begin.

**Q: Your LSTM is overfitting — what do you try first?**
A: Shrink hidden size, add dropout on input/output connections (not recurrent connections — that disrupts gradient flow through time), try GRU for fewer params, or L2-regularize the recurrent weights.

**Q: Can cell state magnitude grow unbounded — is that a problem?**
A: It can grow since it's an unbounded additive sum, but it's not a problem because it's never exposed directly — always passed through $\tanh(c_t)$ before contributing to $h_t$, which re-bounds it at the point of use.

---

**If you remember nothing else:** LSTM = `c_t = keep×old + add×new` (additive, gated). GRU = `h_t = (1-z)×old + z×new` (linear blend, one gate). Bidirectional = `[forward ; backward]`, no streaming. The whole reason any of this exists: decouple "how much to preserve" from "what to compute," which a vanilla RNN's single weight $W_{hh}$ can't do.
