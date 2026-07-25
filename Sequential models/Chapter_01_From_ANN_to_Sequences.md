# Chapter 1: From ANN to Sequences — Why We Need RNNs

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Target:** Apple/Google Senior ML Engineer (L5) interviews
**Prerequisite assumed:** ANN fundamentals, backpropagation

---

## 1.1 The Setup: What ANNs Assume

A standard feedforward ANN (or even a CNN) makes one core assumption:

> **Inputs are independent, fixed-size vectors.**

For `y = f(Wx + b)`, every input `x` is processed in isolation. There is no notion of "what came before." This is fine for:
- Tabular data (row-independent)
- Image classification (a cat photo doesn't depend on the previous photo)

It **breaks down** for data where **order and context matter**:
- Text: "The movie was not good" vs "The movie was good, not bad" — same words, different order, opposite/nuanced meaning
- Time series: stock price at t=10 depends heavily on t=9, t=8...
- Speech/audio: a phoneme's meaning depends on what was said a moment ago
- Search queries / session behavior (Apple Search & AI relevance): a query is often a *continuation* of previous queries in a session

## 1.2 Why Not Just Feed the Whole Sequence into an ANN?

Naive idea: concatenate a fixed window of tokens into one big vector, feed to an ANN.

**Problems (interview-important):**

1. **Fixed input size.** A sentence of 5 words and one of 50 words need different-sized inputs. ANNs need fixed `d`-dim vectors.
2. **No parameter sharing across positions.** If "Paris" appears at position 3 in one example and position 47 in another, an ANN with position-specific weights has to *relearn* what "Paris" means at every position. Massive sample inefficiency.
3. **No compositional memory.** There's no mechanism to accumulate information across arbitrary-length history — you'd need to hand-pick a window size, and long-range dependencies (e.g., subject-verb agreement across 20 words) are lost if they fall outside the window.

## 1.3 The Core Idea of RNNs

**Recurrent Neural Networks solve this with one trick: reuse the same weights at every timestep, and pass a "hidden state" forward as a summary of everything seen so far.**

Think of the hidden state `h_t` as a **running memory / notebook**:
- At each timestep, the RNN reads the current input `x_t` AND the notebook `h_{t-1}` from the previous step
- It updates the notebook: `h_t = f(x_t, h_{t-1})`
- The same update rule (same weights) is applied at every single timestep

This is the "unrolled" diagram you just saw: one cell, copied across time, sharing weights, with the hidden state as the thread connecting them.

**Analogy:** Reading a novel. You don't re-read the entire book from scratch at every new page — you carry forward a mental summary (who the characters are, what's happened) and update it with each new page. That mental summary *is* the hidden state.

## 1.4 Why This Fixes All Three Problems

| ANN Problem | RNN Fix |
|---|---|
| Fixed input size | Process one token at a time; sequence length is just "how many times you run the loop" |
| No parameter sharing across positions | Same `W` used at every timestep — "Paris" is processed identically no matter where it appears |
| No memory across time | Hidden state `h_t` explicitly carries a compressed summary of the past forward |

## 1.5 What an RNN Is NOT (Interview Trap #1)

- It is **not** a different neuron type — it's still just matrix multiply + nonlinearity, applied recurrently.
- It does **not** have unlimited memory — `h_t` is a *fixed-size vector* (e.g., 128-dim), so it must compress arbitrarily long history into a fixed budget. This compression is *lossy* — this is exactly the seed of why RNNs struggle with long-range dependencies, and exactly why LSTM/GRU (Ch. 5–9) and later attention (Ch. 13) exist.
- It does **not** process all timesteps in parallel during training in the naive formulation — computation at `t` depends on `t-1`, making RNNs inherently **sequential**, which is also why they're slow to train and why Transformers displaced them for large-scale NLP (but RNNs are still very relevant for streaming/online, low-latency, and time-series use cases — highly relevant to on-device Apple contexts).

## 1.6 Numerical Preview (fully worked in Chapter 2)

Just to plant the seed — here's the shape of what's coming. An RNN cell computes:

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
y_t = W_hy · h_t + b_y
```

- `W_xh`: maps input → hidden
- `W_hh`: maps previous hidden → new hidden (this is the "memory" weight)
- `W_hy`: maps hidden → output
- **Same three matrices are reused at every single timestep.**

In Chapter 2, we'll hand-compute this for a real toy sequence (3 timesteps, 2-dim hidden state) with actual numbers, so you can trace exactly how a number 3 steps back in the sequence influences the current output.

## 1.7 Interview Talking Points (L5 Signal)

A mid-level answer says: "RNNs process sequences and have a hidden state."

An **L5-differentiating** answer says:
- "RNNs trade the fixed-size-input constraint of ANNs for a fixed-size-*memory* constraint — you've moved the bottleneck, not removed it."
- "The parameter sharing across time is the same inductive bias as CNN weight-sharing across space — both exploit a symmetry in the problem (translation invariance in space vs. time) to reduce sample complexity."
- "The sequential dependency (`h_t` needs `h_{t-1}`) is why RNNs don't parallelize over the time dimension during training — this is a major reason the field moved toward attention/Transformers for training efficiency at scale, even though RNNs can be more efficient at *inference* for streaming, single-token-at-a-time generation."

## 1.8 Sample Interview Q&A

**Q: Why would you use an RNN instead of a feedforward network with a sliding window for time-series forecasting?**
A: A sliding window hard-codes a fixed lookback and treats each window position with separate implicit weight allocation via concatenation, so it can't generalize a pattern learned at lag-3 to lag-10. An RNN shares weights across all lags and, in principle, can capture dependencies beyond the window length, bounded only by the vanishing-gradient horizon (Ch. 4) rather than a hard-coded window.

**Q: What's the fundamental limitation of the hidden state?**
A: It's a fixed-dimensional vector, so it must lossy-compress unboundedly long history. This creates a capacity bottleneck — the same failure mode as an autoencoder's bottleneck layer — and motivates gating mechanisms (LSTM/GRU) and eventually attention, which lets the model access *all* past hidden states directly instead of relying on one compressed summary.

## 1.9 Comprehension Check (answer before moving to Ch. 2)

1. Why can't you just one-hot/concatenate a whole variable-length sentence and feed it to a plain ANN?
2. What are the two things an RNN cell takes as input at each timestep?
3. What weight matrix is responsible for "memory" (passing information from t-1 to t)?
4. True/False: RNNs have unlimited memory because the hidden state is updated every step. (Answer: False — explain why in your own words.)

---
**Next:** Chapter 2 — Vanilla RNN forward pass, fully hand-computed with real numbers.
