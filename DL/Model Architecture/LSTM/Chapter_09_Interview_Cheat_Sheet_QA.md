# Chapter 9: LSTM/GRU/BiLSTM Interview Cheat Sheet + Q&A

## 1. LSTM equations — memorize as "3 gates, 1 candidate, 1 update"

$$f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f) \quad i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i) \quad o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)$$
$$\tilde{c}_t=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)$$
$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t \qquad h_t=o_t\odot\tanh(c_t)$$

**Say it out loud:** "forget old memory, input new candidate, blend additively into cell state, output a filtered view as hidden state."

**Mnemonic for the cell update — the one line that matters most:** `NEW MEMORY = KEEP × OLD + ADD × CANDIDATE` — no full squash of the running total, just a gated blend.

## 2. GRU equations — memorize as "2 gates, 1 candidate, 1 blend"

$$r_t=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r) \qquad z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)$$
$$\tilde{h}_t=\tanh(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h)$$
$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t$$

**Say it out loud:** "reset gate controls how much old state feeds the candidate; update gate linearly blends old state and candidate — one gate doing the job LSTM splits across two."

**Mnemonic:** `h_t = (1-z)·OLD + z·NEW` — pure linear interpolation, easiest line in the whole curriculum to remember.

## 3. Why gating fixes vanishing gradients — one sentence + one number

**Sentence:** "the gradient path through the cell/hidden state is a product of *learned* forget/update-gate values instead of a fixed shrink factor, and because the gate is decoupled from content computation, the network can push it near 1 whenever long memory matters."

**Number to have ready:** vanilla RNN retained ~0.24% of gradient over 3 steps; LSTM (same toy example) retained ~42% — about 178x more, from decoupling "how much to keep" from "what to compute."

## 4. BiLSTM/BiGRU — memorize as "two independent networks, concatenate"

$$h_t = [\overrightarrow{h_t} \, ; \, \overleftarrow{h_t}]$$

**One sentence:** "run the cell forward, run a second independent copy backward, reverse-align the backward outputs, concatenate at each timestep — needs the full sequence upfront, so no real-time/streaming use."

## 5. Parameter counts — memorize the ratio, not exact numbers

| Cell | Recurrent weight matrices | Relative to vanilla RNN |
|---|---|---|
| Vanilla RNN | 3 | 1x |
| GRU | 6 | 2x |
| LSTM | 8 | ~2.67x |
| BiLSTM/BiGRU | 2x base cell | 2x whichever base |

**One sentence:** "GRU is roughly 25% cheaper than LSTM at the same hidden size — fewer gates, no separate cell state."

## 6. Decision tree — memorize as 2 questions

1. **Full sequence available upfront, and future context helps?** → bidirectional variant.
2. **Limited data/compute, or modest memory needs?** → GRU. Otherwise → LSTM.

## 7. Rapid-fire answers (under 15 seconds each)

- **Why does LSTM have a separate cell state?** So information can travel with minimal forced transformation (additive update), instead of being squashed through tanh every step like the hidden state is.
- **What's the candidate cell state, and why tanh not sigmoid?** It's proposed new *content* (a value, not a gate), so it needs to be bounded $(-1,1)$ like any content representation — sigmoid is reserved for gates (0 to 1, "how much").
- **Why can $c_t$ exceed 1 in magnitude but $h_t$ can't?** $c_t$ is a running additive sum, never squashed directly. $h_t = o_t\odot\tanh(c_t)$ re-bounds it through tanh before exposing it.
- **What does GRU's update gate replace?** LSTM's forget gate AND input gate combined into one coupled decision: keep-fraction $=1-z_t$, add-fraction $=z_t$.
- **Can a bidirectional RNN be used for real-time prediction?** No — it needs the entire sequence before the backward pass can even start.
- **Does gating fully eliminate vanishing gradients?** No — if a task needs the gate near 0 (deliberate forgetting), that gradient path is legitimately cut. Gating provides the *option* of near-lossless flow, not a guarantee.

## 8. Commonly reported questions in Google/Apple-style ML interviews

*(Same caveat as the RNN cheatsheet: not official company material, aggregated patterns from public interview-prep sources. Treat as representative, not verbatim.)*

**Q: Walk me through why LSTM solves vanishing gradients better than a vanilla RNN — don't just say "gates," explain the mechanism.**
A: The cell state update is additive ($c_t=f_t c_{t-1}+i_t\tilde{c}_t$) rather than forced through a multiply-and-squash every step. The gradient path through it is a product of forget-gate values, and because the forget gate is a separately learned function (decoupled from the weights computing content), the network can learn to keep it near 1 when long-range memory is needed — something a vanilla RNN's $W_{hh}$ structurally cannot do, since it's coupled to both content and gradient decay simultaneously.

**Q: Compare GRU and LSTM — when would you pick one over the other in production?**
A: They perform comparably on most tasks. Pick GRU by default for efficiency (25% fewer recurrent parameters, faster training/inference) — especially with limited data or tight latency budgets. Pick LSTM when sequences are long, you have ample data/compute, and you need the extra expressiveness of independently-controlled forget/input gates rather than GRU's coupled update gate.

**Q: What's the tradeoff of using a bidirectional model in production?**
A: You gain access to future context at every position, which typically improves accuracy on tasks where the full input is available upfront (tagging, classification). The cost: you can't run it in streaming/real-time settings (must wait for the full sequence), and it roughly doubles parameter count and compute.

**Q: If your LSTM is overfitting, what would you try before adding more data?**
A: Reduce hidden size, add dropout (applied carefully — typically to the input/output connections, not the recurrent connections, to avoid disrupting gradient flow through time), consider switching to GRU for fewer parameters, or add weight regularization (L2) on the recurrent weight matrices.

**Q: Explain what happens to the cell state's magnitude over a very long sequence, and why that's not necessarily a problem.**
A: Since $c_t$ is a running additive sum, gated by $f_t$ each step, its magnitude can grow or shrink depending on the accumulated gate values — it's not bounded like $\tanh$-squashed states. This is fine because $c_t$ is never directly exposed; it's always passed through $\tanh(c_t)$ before contributing to $h_t$, which re-bounds it at the point of use.

**Q: How would you decide between a stacked BiLSTM and a single-layer BiLSTM with a larger hidden size?**
A: Stacking adds representational depth (compositional features across layers) at the cost of a harder optimization problem (gradients now flow through time, direction, and layers). A single wider layer is easier to train and often a reasonable first try; stacking tends to help more once you have enough data to support the extra depth, and is common in production tagging/NER systems where accuracy gains from depth are well-documented.

---

**If you remember nothing else:** LSTM's core line is `NEW MEMORY = KEEP×OLD + ADD×CANDIDATE` (additive, not squashed). GRU's core line is `h_t = (1-z)·OLD + z·NEW` (linear interpolation, one gate doing two jobs). Bidirectional is just "two independent networks, concatenate, no streaming." Everything else in this curriculum is detail on top of those three lines.
