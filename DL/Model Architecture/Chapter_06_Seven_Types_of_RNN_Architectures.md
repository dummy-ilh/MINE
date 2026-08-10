# Chapter 6: The 7 Types of RNN Architectures

## Why architecture "shape" matters

Everything so far assumed a specific shape: 3 inputs in, 3 outputs out, one loss per timestep. That's just one option. The core RNN mechanism from Chapters 1–5 (shared weights, hidden state carried forward, BPTT) stays the same — what changes is **where inputs enter, where outputs leave, and where losses get computed.** That last part changes how gradients accumulate. This chapter walks through all 7 shapes and, for each, says exactly what's different about backprop.

## 1. One-to-one

```
x -> [NN] -> y
```

No sequence at all — a single input produces a single output. This is just a regular feedforward network. It's included here only as the baseline: zero recurrence, zero shared weights across time, ordinary backprop. Not really an RNN, but it anchors the spectrum.

**BPTT:** none needed — there's no "time" to backpropagate through.

## 2. One-to-many

```
x -----> h1 -> h2 -> h3
         |     |     |
         y1    y2    y3
```

A single input kicks off the sequence, and the network generates a sequence of outputs from it. Example: image captioning — one image in, a sequence of words out.

**BPTT:** losses exist at every output ($y_1, y_2, y_3$), but there's only **one** input, injected at $t=1$ (or fed into $h_0$). Gradients from all three losses flow backward through the chain toward that single input and — as always — accumulate into the shared weights $W_{hh}, W_{xh}$ at every step they were used. The later losses ($y_3$) have a longer path back to the input than the earlier ones ($y_1$), so they're more exposed to vanishing gradients.

## 3. Many-to-one

```
x1 -> h1 -> h2 -> h3
             (only)  |
                     y3
```

A full sequence goes in, but there's only **one** output, produced at (or after) the last timestep. Example: sentiment classification — read a whole review, output one label. This is the shape most textbook diagrams use for "sequence classification."

**BPTT:** exactly the setup we'd get if Chapters 2–3 had used a single loss at $t=3$ instead of three separate losses. There's just one term, $\frac{\partial L}{\partial h_3}$, and it has to travel all the way back through $h_2$ and $h_1$ to reach the earliest weights. This is the shape most exposed to vanishing gradients for early timesteps — there's no "shortcut" loss at $t=1$ giving the early steps their own direct gradient signal.

## 4. Many-to-many (aligned)

```
x1 -> h1 -> h2 -> h3
      |     |     |
      y1    y2    y3
```

Same number of inputs and outputs, and they're synced in time — output $y_t$ corresponds directly to input $x_t$. Example: part-of-speech tagging (tag every word), video frame classification (label every frame).

**This is actually the exact shape we used in Chapters 2 and 3** — we computed $\hat{y}_1, \hat{y}_2, \hat{y}_3$ from $x_1, x_2, x_3$ and summed three losses. Worth noting explicitly, since the task description ("predict rain probability from 3 days") sounded many-to-one but the worked numbers were many-to-many aligned.

**BPTT:** every timestep contributes its own gradient term, and — as derived in Chapter 3 — the total gradient for a shared weight is the *sum* over all timesteps' contributions: $\frac{\partial L}{\partial W_{hh}} = \sum_t e_t \cdot D_t$. Early timesteps get their own direct loss signal (unlike many-to-one), which partially offsets vanishing gradients — there's more than one "entry point" for gradient into the chain.

## 5. Many-to-many (unaligned / seq2seq)

```
Encoder:  x1 -> h1 -> h2 -> h3 -> [context vector c]
                                       |
Decoder:                    c ->  s1 -> s2 -> s3 -> s4
                                   |     |     |     |
                                   y1    y2    y3    y4
```

Input length and output length don't have to match, and there's no timestep-to-timestep alignment. Example: machine translation — a 3-word French sentence might become a 5-word English sentence. This uses **two** RNNs: an *encoder* that reads the whole input and compresses it into a summary (the final hidden state, often called the context vector $c$), and a *decoder* that generates the output sequence starting from that summary.

**BPTT:** gradients flow backward through the decoder's chain first (same mechanism as many-to-many aligned, but decoder-only), all the way back to the context vector $c$. Then — critically — the gradient **continues flowing backward through the encoder's chain too**, since $c$ was built from the encoder's $h_3$. So you get two separate BPTT passes chained together: decoder gradients accumulate all the way back through the encoder. This is a longer path than any single-RNN shape above, so unaligned seq2seq is the most vanishing-gradient-prone shape covered so far — one motivation for attention mechanisms (outside this curriculum's scope, covered elsewhere).

## 6. Bidirectional RNN

```
Forward:   h1_f -> h2_f -> h3_f
Backward:  h1_b <- h2_b <- h3_b
             |       |       |
           [h1_f;h1_b] [h2_f;h2_b] [h3_f;h3_b]
                 |          |          |
                y1         y2         y3
```

Two separate RNNs read the same sequence — one left-to-right, one right-to-left — and their hidden states are concatenated at each timestep before producing an output. Useful when the whole sequence is available upfront (not for real-time prediction) and future context matters. Example: filling in a masked word using both preceding *and* following words.

**BPTT:** the forward RNN and backward RNN are two **independent** chains — each gets its own BPTT pass, exactly like a many-to-many aligned RNN (Chapter 3's mechanism), run once forward-in-time and once backward-in-time. Their gradients don't interact except where they merge — at the shared output weights that consume the concatenated hidden state. So you effectively do Chapter 3's math twice, then combine gradients at the readout layer.

## 7. Deep (stacked) RNN

```
Layer 2:   h1^(2) -> h2^(2) -> h3^(2)
             ^          ^          ^
Layer 1:   h1^(1) -> h2^(1) -> h3^(1)
             ^          ^          ^
            x1          x2         x3
```

Multiple RNN layers stacked vertically — the hidden state sequence output by layer 1 becomes the *input* sequence to layer 2, and so on. This adds representational depth (like stacking layers in an MLP) on top of the recurrence-over-time depth.

**BPTT:** gradients now flow in **two directions**: backward through time (within each layer, exactly as in Chapter 3) *and* backward through layers (from layer 2 down to layer 1, like ordinary feedforward backprop between layers). A weight in layer 1 gets a gradient that's been passed back through every later timestep in layer 1 *and* down through every layer above it. This compounds the vanishing-gradient risk from Chapter 4 — more multiplied factors, in two dimensions instead of one.

## Summary table

| # | Shape | Input : Output | Example | Where losses live | BPTT characteristic |
|---|---|---|---|---|---|
| 1 | One-to-one | 1 : 1 | Image classification | Single loss | No recurrence |
| 2 | One-to-many | 1 : N | Image captioning | Loss at every output step | Single entry point for input; later losses have longer paths |
| 3 | Many-to-one | N : 1 | Sentiment classification | Single final loss | Most vanishing-prone; no early direct gradient signal |
| 4 | Many-to-many (aligned) | N : N | POS tagging | Loss at every step | Sum of per-timestep gradients (Chapter 3's exact mechanism) |
| 5 | Many-to-many (unaligned/seq2seq) | N : M | Translation | Loss at every decoder step | Two chained BPTT passes (decoder, then encoder); longest gradient path |
| 6 | Bidirectional | N : N | Masked word fill-in | Loss at every step | Two independent BPTT passes (forward + backward), merged at readout |
| 7 | Deep/stacked | N : N | Complex sequence modeling | Loss at every step (top layer) | BPTT through time *and* through layers — two-dimensional gradient flow |

## What's ahead

Chapter 7 turns everything so far into an interview-ready cheat sheet and Q&A. Chapter 8 covers RNNs on tabular data. Chapter 9 builds a vanilla RNN from scratch — NumPy first, then PyTorch.

---

**One-line summary:** the same shared-weight, hidden-state mechanism underlies all 7 shapes — what changes is where inputs enter and losses attach, which changes how many gradient paths exist and how exposed each shape is to vanishing gradients.
