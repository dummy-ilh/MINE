# RNN Interview Cheatsheet (One Page)

## 1. The forward pass — memorize this shape

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h) \qquad \hat{y}_t = W_{hy}h_t + b_y$$

**Say it out loud like this:** "new input times its weight, plus old memory times its weight, plus bias, squashed through tanh."

**Mnemonic:** `h_t = tanh(NEW + OLD + bias)` — new = $W_{xh}x_t$, old = $W_{hh}h_{t-1}$.

## 2. Replicate the toy example live (memorize these 5 numbers)

$$x=[1,2,3] \quad W_{xh}=0.5,\ W_{hh}=0.8,\ b_h=0.1,\ W_{hy}=1,\ b_y=0$$

| $t$ | $z_t$ | $h_t$ |
|---|---|---|
| 1 | $0.5(1)+0.8(0)+0.1=0.6$ | $\tanh(0.6)\approx0.54$ |
| 2 | $0.5(2)+0.8(0.54)+0.1=1.53$ | $\tanh(1.53)\approx0.91$ |
| 3 | $0.5(3)+0.8(0.91)+0.1=2.33$ | $\tanh(2.33)\approx0.98$ |

**Trick to reproduce fast:** you only need to remember $z_t = 0.5x_t + 0.8h_{t-1}+0.1$, then tanh it. Everything else follows.

## 3. BPTT — memorize this ONE recursive line

$$D_t = (1-h_t^2)\big(h_{t-1} + W_{hh}\cdot D_{t-1}\big), \quad D_0=0$$

**Say it out loud:** "local tanh slope, times (what's directly here, plus whatever carried over, scaled by $W_{hh}$)."

Then: $\dfrac{\partial L}{\partial W_{hh}} = \sum_t e_t \cdot D_t$, where $e_t = \hat{y}_t - y_t^*$.

**Why this is the whole trick:** every BPTT question boils down to "the same weight is used at every step, so sum its effect across all steps." If you remember only one sentence for BPTT, remember that one.

## 4. Vanishing/exploding — one formula, one sentence

$$\frac{\partial h_t}{\partial h_k} = \prod_{s=k+1}^{t}(1-h_s^2)\cdot W_{hh}$$

**One sentence:** "gradient across $n$ steps = product of $n$ numbers; mostly-under-1 → vanishes, mostly-over-1 → explodes." $\tanh'\le 1$ always, so vanishing is the default failure mode.

## 5. Fixes — 3 boxes, memorize which problem each solves

| Fix | Fixes exploding? | Fixes vanishing? |
|---|---|---|
| Gradient clipping | ✅ | ❌ |
| Truncated BPTT | partially (compute only) | ❌ |
| Gating (LSTM/GRU) | ✅ | ✅ |

**One sentence:** "clipping caps big gradients, truncation just stops paying for gradients that already vanished, gating is the only real architectural fix."

## 6. The 7 shapes — memorize as a spectrum, 1 input/output pattern each

| # | Shape | Pattern | Example |
|---|---|---|---|
| 1 | one-to-one | 1:1 | image classification |
| 2 | one-to-many | 1:N | image captioning |
| 3 | many-to-one | N:1 | sentiment classification |
| 4 | many-to-many aligned | N:N | POS tagging |
| 5 | many-to-many unaligned (seq2seq) | N:M | translation |
| 6 | bidirectional | N:N, 2 passes | masked word fill-in |
| 7 | deep/stacked | N:N, 2D backprop | complex sequence modeling |

**One sentence per hard one:** many-to-one is most vanishing-prone (only one loss, at the end); seq2seq is worst overall (two chained BPTT passes, encoder + decoder).

## 7. Rapid-fire answers (say these in under 15 seconds each)

- **Why not MLP for sequences?** Fixed input size + no order sensitivity. RNN fixes both via shared weights + hidden state.
- **What is BPTT, really?** Ordinary backprop on the unrolled graph; shared weights get gradients summed across timesteps.
- **Vanishing gradient in one line?** Product of many <1 factors → shrinks to ~0 → early steps stop learning.
- **Does clipping fix vanishing?** No — it only caps large gradients; a vanished gradient has nothing to cap.
- **How does LSTM fix it?** Adds a cell-state path that skips the multiply-and-squash bottleneck, gated by learned sigmoid gates.
- **Params grow with sequence length?** No — weight sharing means fixed param count regardless of length.

## 8. Commonly reported questions in Google/Apple-style ML interviews

*(Not official company material — these are patterns aggregated from public interview-prep sources like Glassdoor and ML interview compilations. Treat as representative, not verbatim.)*

**Q: What causes the vanishing gradient problem in RNNs, and how would you detect it during training?**
A: Repeated multiplication by $(1-h_t^2)\cdot W_{hh}$ across timesteps shrinks gradients toward zero. Detect it by watching gradient norms per layer/timestep during training — early-timestep gradients staying near-zero while later ones are healthy is the signature.

**Q: Walk me through how you'd decide between a vanilla RNN, LSTM, and GRU for a given task.**
A: Vanilla RNN only for very short sequences or as a baseline. LSTM when you need strong long-range memory and have enough data/compute for the extra gate parameters. GRU as a cheaper middle ground — fewer gates, often comparable performance, faster to train.

**Q: How would you handle a batch of sequences with different lengths in an RNN?**
A: Pad shorter sequences to the max length in the batch, then use masking (or `pack_padded_sequence` in PyTorch) so the loss and gradient updates ignore the padded positions.

**Q: Explain how gradient clipping works and why it's applied to the gradient norm rather than each parameter individually.**
A: Clipping rescales the entire gradient vector if its overall norm exceeds a threshold, preserving direction while capping magnitude. Clipping each parameter independently would distort the direction of the update, which can hurt convergence more than it helps.

**Q: Given a many-to-one RNN classifier that performs well on short sequences but poorly on long ones, what's your diagnosis and fix?**
A: Likely vanishing gradients — the single final loss can't push learning signal back far enough for long sequences. Fixes: switch to LSTM/GRU, add attention, or use bidirectional processing so relevant context isn't buried at the far end of a long chain.

**Q: What's the difference between a stacked RNN and a bidirectional RNN, and can you combine them?**
A: Stacked = depth through layers (each layer's output feeds the next layer, same time direction). Bidirectional = depth through direction (forward + backward passes over time, concatenated). Yes — you can stack multiple bidirectional layers; each layer processes both directions before passing its concatenated output up to the next layer.

---

**If you remember nothing else:** forward pass is "blend new input with old memory, squash, repeat." BPTT is "same weight, so sum its gradient across every step it touched." Vanishing/exploding is "that sum is a product of many factors — too small or too big, repeated, breaks things." Everything else is detail on top of those three sentences.
