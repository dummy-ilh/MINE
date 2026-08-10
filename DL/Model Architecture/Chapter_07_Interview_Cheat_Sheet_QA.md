# Chapter 7: Interview Cheat Sheet + Q&A

## Formula reference

**Forward pass (vanilla RNN):**
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$
$$\hat{y}_t = W_{hy}h_t + b_y$$

**Gradient through one step:**
$$\frac{\partial h_s}{\partial h_{s-1}} = (1-h_s^2)\cdot W_{hh}$$

**Gradient across $n$ steps (the core vanishing/exploding mechanism):**
$$\frac{\partial h_t}{\partial h_k} = \prod_{s=k+1}^{t}(1-h_s^2)\cdot W_{hh}$$

**Total derivative recursion (BPTT for a shared weight):**
$$D_t = \frac{dh_t}{dW_{hh}} = (1-h_t^2)\big(h_{t-1} + W_{hh}\cdot D_{t-1}\big), \quad D_0 = 0$$

**Full gradient for a shared weight (sum across timesteps):**
$$\frac{\partial L}{\partial W_{hh}} = \sum_t \frac{\partial L_t}{\partial h_t}\cdot D_t$$

**Gradient clipping:**
$$g \leftarrow g\cdot\frac{\text{threshold}}{\|g\|} \quad \text{if } \|g\|>\text{threshold}$$

## Key facts, fast

- An RNN is one small function, reused at every timestep — that reuse (weight sharing) is *why* it handles variable-length sequences.
- The hidden state $h_t$ is the network's compressed memory of everything seen so far.
- BPTT ≠ a different algorithm from backprop — it's ordinary backprop applied to the unrolled computation graph, with the twist that the *same* weight appears at every step, so its total gradient is a **sum across all the timesteps it touched**.
- Vanishing and exploding gradients both come from the same product: $\prod (1-h_s^2)\cdot W_{hh}$. Factors consistently $<1$ → vanish. Consistently $>1$ → explode.
- $\tanh' = 1-h^2$ is always $\le 1$, and shrinks toward 0 as $h_t$ saturates near $\pm1$ — this is why vanishing is the *default* failure mode, and exploding needs a large $W_{hh}$ to trigger.
- Gradient clipping fixes exploding gradients only. Truncated BPTT is a compute/memory compromise, not a vanishing-gradient fix. Only architectural changes (LSTM/GRU gating) fix vanishing gradients directly, by giving information a low-decay path through time.
- 7 architecture shapes exist because inputs/outputs/losses can attach at different points in the unrolled chain — this changes how many gradient "entry points" exist and how exposed each shape is to vanishing gradients (many-to-one is worst; many-to-many aligned is better; unaligned seq2seq is worst overall due to the two chained BPTT passes).

## Conceptual Q&A

**Q: Why can't you just use an MLP for sequence data?**
A: Two reasons: MLPs need a fixed input size, but sequences vary in length; and MLPs have no notion of order — shuffle the inputs and (without extra engineering) the output is unaffected. RNNs fix both by reading one element at a time and carrying a hidden state forward, using the same weights at every step.

**Q: What exactly is "recurrent" about a recurrent neural network?**
A: The same function is applied repeatedly, each time taking its own previous output ($h_{t-1}$) as part of its input. It's a loop, not a stack of distinct layers — even though unrolled, it visually resembles a deep network.

**Q: Explain vanishing gradients in one sentence.**
A: A gradient traveling backward through $n$ timesteps is a product of $n$ factors, each roughly $(1-h_s^2)\cdot W_{hh}$; since $\tanh'\le1$, these factors are usually below 1, so the product shrinks toward zero as $n$ grows, meaning early timesteps get essentially no learning signal.

**Q: Why doesn't gradient clipping fix vanishing gradients?**
A: Clipping only caps gradients that are *too large*. A vanished gradient is already near zero — there's nothing to cap, and clipping does nothing to make it bigger. It's a fix for the opposite failure mode.

**Q: What's the actual difference between truncated BPTT and full BPTT?**
A: Full BPTT backpropagates through every timestep of the sequence. Truncated BPTT only backpropagates through the last $k$ steps, even though the forward pass runs the full sequence — it trades some (usually already-vanished) gradient accuracy for tractable compute and memory.

**Q: How do LSTM/GRU solve vanishing gradients?**
A: They add a second pathway (the cell state, in LSTM) that information can travel along largely unchanged, controlled by learned gates, instead of forcing everything through a multiply-by-$W_{hh}$-and-squash-through-tanh bottleneck at every step. This gives gradients a route through time that doesn't shrink geometrically.

**Q: Give an example each of many-to-one, one-to-many, and many-to-many.**
A: Many-to-one: sentiment classification (sequence of words → one label). One-to-many: image captioning (one image → sequence of words). Many-to-many aligned: part-of-speech tagging (one tag per word). Many-to-many unaligned: machine translation (encoder-decoder, different lengths).

**Q: Why is many-to-one more vulnerable to vanishing gradients than many-to-many aligned?**
A: Many-to-one has a single loss at the final timestep — the only gradient signal for early timesteps has to survive the *entire* backward chain. Many-to-many aligned has a loss at every timestep, so early timesteps get their own direct gradient in addition to whatever survives from later losses.

**Q: What's the difference between depth "through time" and depth "through layers" in a stacked RNN?**
A: Through time: the same layer applied repeatedly across timesteps (horizontal chain, shared weights). Through layers: distinct RNN layers stacked vertically, each with its own weights, where layer $l$'s hidden-state sequence becomes layer $l{+}1$'s input sequence. Stacked RNNs backprop through both simultaneously.

**Q: Does an RNN's parameter count grow with sequence length?**
A: No — that's the entire point of weight sharing. $W_{xh}, W_{hh}, W_{hy}, b_h, b_y$ are fixed-size regardless of whether the sequence is 3 steps or 3,000.

## Common pitfalls (interviewer bait)

- Saying BPTT is a "different algorithm" from backprop — it isn't; it's the same chain rule, applied to an unrolled shared-weight graph.
- Confusing "vanishing gradient" with "small loss" — they're unrelated; you can have a large loss and a vanished gradient at the same time (the network just can't fix it via early-timestep weights).
- Thinking gradient clipping fixes vanishing gradients (it doesn't — see above).
- Forgetting that in many-to-many, the total gradient for a shared weight is a **sum** over timesteps, not a single term.
- Assuming a bidirectional RNN's forward and backward passes interact during BPTT — they're independent chains, only merged at the readout layer.

## What's ahead

Chapter 8 applies RNNs to tabular data — an unusual pairing, but worth knowing when and why. Chapter 9 builds a vanilla RNN from scratch in NumPy, then PyTorch.

---

**One-line summary:** if an interviewer asks about RNNs, you should be able to write the forward equation, derive why gradients vanish/explode as a product-of-factors argument, name which fixes solve which problem, and map any use case to one of the 7 architecture shapes.
