# Chapter 12: Sequence-to-Sequence (Encoder-Decoder)

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 2 (encoder reuses that forward pass), Chapter 10 (encoder can be bidirectional)

---

## 12.1 The Gap This Fills

Recall the three regimes from Chapter 2.7: many-to-one, many-to-many-aligned, many-to-many-**unaligned**. Machine translation is the classic unaligned case: "I am a student" (4 words) → "Je suis étudiant" (3 words) — **input and output lengths differ, and there's no natural 1-to-1 timestep alignment.** None of the architectures so far handle this directly — we need something new.

## 12.2 The Architecture: Two RNNs, Connected by a Context Vector

```
ENCODER: reads the entire input sequence, produces a single summary vector
  x_1, x_2, ..., x_T  →  [Encoder RNN/LSTM]  →  c  (context vector — typically the encoder's final hidden state)

DECODER: generates the output sequence one token at a time, conditioned on c
  c, <START>  →  [Decoder RNN/LSTM]  →  y_1
  c, y_1      →  [Decoder RNN/LSTM]  →  y_2
  c, y_2      →  [Decoder RNN/LSTM]  →  y_3
  ...continues until a <END> token is generated
```

**Key structural facts:**
- The encoder can be **bidirectional** (Ch. 10) — the whole input is available upfront, no streaming constraint.
- The decoder **must be unidirectional/autoregressive** — at generation time, future output tokens don't exist yet (you're producing them one at a time), so bidirectionality is structurally impossible here (this directly answers Ch.10's comprehension check #3).
- The decoder's hidden state is typically **initialized with the context vector `c`** (i.e., `h_0^dec = c`), and at each step also receives the *previous* output token (or its embedding) as input.

## 12.3 Numerical Walkthrough: One Decoding Step

**Context vector:** reuse Chapter 2's encoder final hidden state, `c = h_3 = [-0.3157, 0.3305]`.

**Setup:** decoder hidden dim = 2, vocabulary = {A, B, C} (3 tokens), `<START>` token embedding `y_0^emb = [1, 0]`.

```
Decoder weights:
  W_xh_dec = [[0.4, -0.2], [0.5, 0.3]]
  W_hh_dec = [[0.2, 0.1], [-0.3, 0.4]]
  b_h_dec  = [0, 0]

Output projection (hidden → vocab logits):
  W_hy_dec = [[0.5, -0.3], [0.2, 0.4], [-0.1, 0.6]]     (rows = tokens A, B, C)
  b_y_dec  = [0, 0, 0]

h_0^dec = c = [-0.3157, 0.3305]
```

**Step A: `W_xh_dec · y_0^emb`**
```
row1: 0.4(1) + (-0.2)(0) = 0.4
row2: 0.5(1) + 0.3(0) = 0.5
→ [0.4, 0.5]
```

**Step B: `W_hh_dec · h_0^dec`**
```
row1: 0.2(-0.3157) + 0.1(0.3305) = -0.0631 + 0.0331 = -0.0301
row2: -0.3(-0.3157) + 0.4(0.3305) = 0.0947 + 0.1322 = 0.2269
→ [-0.0301, 0.2269]
```

**Step C: sum + tanh**
```
a = [0.4-0.0301, 0.5+0.2269] = [0.3699, 0.7269]
h_1^dec = [tanh(0.3699), tanh(0.7269)] = [0.3538, 0.6215]
```

**Step D: project to vocabulary logits**
```
logit_A = 0.5(0.3538) + (-0.3)(0.6215) = 0.1769 - 0.1865 = -0.0096
logit_B = 0.2(0.3538) + 0.4(0.6215) = 0.0708 + 0.2486 = 0.3194
logit_C = -0.1(0.3538) + 0.6(0.6215) = -0.0354 + 0.3729 = 0.3375
```

**Step E: softmax**
```
e^-0.0096≈0.990, e^0.3194≈1.376, e^0.3375≈1.401
sum ≈ 3.767

P(A) = 0.990/3.767 = 0.263
P(B) = 1.376/3.767 = 0.365
P(C) = 1.401/3.767 = 0.372
```

**Decoded first token: C** (highest probability, 37.2% — though note B is nearly tied at 36.5%, illustrating the model's genuine uncertainty here). This token's embedding now becomes the input for the next decoding step (`y_1^emb → h_2^dec → ...`), and the process repeats until `<END>` is produced.

## 12.4 Teacher Forcing vs. Autoregressive Inference (Critical Distinction)

- **Training (teacher forcing):** at each decoder step, feed in the **ground-truth** previous token, not the model's own (possibly wrong) prediction. This stabilizes and speeds up training.
- **Inference:** there's no ground truth — you feed the model's **own previously generated token** back in. This is the actual autoregressive generation loop.

**This mismatch between training-time and inference-time input distributions is called *exposure bias*** — the model never learned to recover from its own mistakes during training (it always saw perfect ground-truth history), but at inference, errors can compound once a wrong token is fed back in. **Scheduled sampling** (gradually mixing in the model's own predictions during training, ramping up over training) is a common mitigation, though this is a real, only-partially-solved issue that resurfaces in most modern generation systems.

## 12.5 The Fundamental Bottleneck (Sets Up Chapter 13)

**The single context vector `c` is a fixed-size summary of the *entire* input sequence — regardless of whether the input is 5 tokens or 500.** This is the exact same fixed-capacity bottleneck problem from Chapter 1.5, now made concrete and severe: a 50-word source sentence must be compressed into the same-size vector as a 5-word one, and the decoder has **no way to look back at specific parts of the input** — it only ever sees `c`.

This bottleneck is precisely what **attention** (Chapter 13) solves: instead of forcing everything through one fixed-size vector, let the decoder access *all* encoder hidden states directly at every decoding step, and learn to weight them by relevance.

## 12.6 Decoding Strategies (Interview-Relevant, Briefly)

- **Greedy decoding:** always pick the highest-probability token at each step (what we did above). Fast, but can lock in early mistakes with no ability to reconsider.
- **Beam search:** keep the top-`k` partial sequences at each step instead of just 1, expanding each and pruning back to `k` — better global sequence quality at `k`× the compute cost. Standard in production translation/generation systems.
- **Sampling-based (temperature/top-k/nucleus):** sample rather than argmax, for more diverse output — common in open-ended generation, less so in translation where a single "correct" answer is expected.

## 12.7 Interview Talking Points (L5 Signal)

- "The encoder-decoder split cleanly separates *understanding* (encoding, can be bidirectional, no autoregressive constraint) from *generation* (decoding, must be autoregressive/unidirectional) — naming this asymmetry explicitly is a strong signal of structural understanding."
- "The fixed-size context vector is the *same* fixed-capacity bottleneck problem introduced all the way back in Chapter 1, now at the sequence-summary level rather than the single-hidden-state level — attention is the direct architectural answer, not an unrelated add-on."
- "Exposure bias is a genuine, still-relevant issue — worth distinguishing from garden-variety overfitting, since it's specifically about a *train/inference input distribution mismatch*, not a generalization gap in the usual sense."

## 12.8 Sample Interview Q&A

**Q: Why does the decoder have to be autoregressive, but the encoder doesn't?**
A: The encoder processes a fully-observed input sequence — nothing is missing, so bidirectionality (needing "future" input tokens) is available and often helpful. The decoder is *generating* the output sequence one token at a time — by definition, future output tokens don't exist yet during generation, so there's no future context to look at; each step can only condition on tokens already produced.

**Q: What's the practical failure mode caused by relying solely on a single context vector, especially for long source sequences?**
A: Translation/generation quality degrades noticeably as source sentence length grows, since the fixed-size context vector must compress increasingly more information into the same capacity — a concrete, measurable version of the bottleneck problem, historically one of the strongest empirical motivations for attention mechanisms.

**Q: How would you handle the mismatch between teacher-forced training and autoregressive inference in a production translation system?**
A: Options include scheduled sampling (gradually exposing the model to its own predictions during training), or evaluating/fine-tuning with beam search decoding to better match the actual inference-time decoding strategy, alongside standard practices like label smoothing to reduce overconfidence in single "correct" tokens. In practice, most modern systems mitigate rather than fully solve exposure bias.

## 12.9 Comprehension Check

1. Why must the decoder in a seq2seq model be strictly unidirectional, even if the encoder is bidirectional?
2. What's the difference between teacher forcing (training) and the actual generation loop (inference), and what problem does this mismatch cause?
3. What specific bottleneck does the encoder-decoder architecture still have, and what chapter's mechanism directly addresses it?
4. In one sentence, contrast greedy decoding and beam search.

---
**Next:** Chapter 13 — Attention as an RNN bridge: how attention removes the fixed-context-vector bottleneck by letting the decoder access all encoder states directly (brief chapter, since you've already covered Transformer attention in depth).
