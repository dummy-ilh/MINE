# Chapter 10: Bidirectional RNN/LSTM — Architecture & Hand Computation

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 2 (we directly reuse those forward-pass numbers)

---

## 10.1 The Motivating Gap

Everything so far (vanilla RNN, LSTM, GRU) processes a sequence **left to right only**. At position `t`, the hidden state `h_t` only knows about `x_1, ..., x_t` — it has **zero information about what comes after**.

For many tasks, that's a real handicap:
- **Named Entity Recognition:** in "Washington was elected," knowing the word *after* "Washington" ("was elected," suggesting a person, not the city) helps disambiguate — but a forward-only model at the word "Washington" hasn't seen that yet.
- **Part-of-speech tagging:** "I saw her **duck**" — is "duck" a verb (she ducked) or noun (her pet duck)? Often only resolved by what follows.
- **Search/query understanding (Apple-relevant):** at inference time, you typically have the **entire query string** available at once — there's no reason to throw away the back half of the query when interpreting the front half.

**Bidirectional models exist precisely to use future context when it's available at inference time.**

## 10.2 The Architecture: Two Independent Passes, Concatenated

A BiRNN (or BiLSTM/BiGRU — the "Bi" wrapper works identically over any cell type) runs **two completely separate recurrent networks over the same input sequence**:

- A **forward** RNN: processes `x_1 → x_2 → ... → x_T`, producing `h_1^fwd, h_2^fwd, ..., h_T^fwd` (each summarizing everything *up to and including* that position)
- A **backward** RNN: processes `x_T → x_{T-1} → ... → x_1` (reverse order), producing `h_T^bwd, ..., h_1^bwd` (each summarizing everything *from that position to the end*)

**These are two separate sets of weights** (`W_xh^fwd, W_hh^fwd` vs. `W_xh^bwd, W_hh^bwd`, entirely independently trained) — not the same weights run in two directions.

At each position `t`, the final representation is the **concatenation**:
```
h_t^bi = [ h_t^fwd ; h_t^bwd ]
```

So if each direction has hidden dim `d_h`, the bidirectional output at each position has dim `2·d_h`.

## 10.3 Numerical Walkthrough (Reusing Chapter 2's Exact Setup)

**Important caveat before we start:** for teaching efficiency, we'll reuse the *same* weight matrices (`W_xh, W_hh, b_h` from Chapter 2) for both directions below. In a real trained model, the backward direction has its own independently-learned weights — we're only sharing them here so you can hand-verify the arithmetic without introducing a second full weight set. The mechanics are identical either way.

**Forward pass (already computed in Chapter 2):**
```
h_1^fwd = [0.4219, 0.3799]
h_2^fwd = [0.0363, 0.5397]
h_3^fwd = [-0.3157, 0.3305]
```

**Backward pass:** process in order `x_3, x_2, x_1` (reverse), starting from `h_0^bwd = [0,0]`:

**Step 1 (consume `x_3` first):**
```
W_xh · x_3 = [-0.65, 0.30]     (computed in Ch. 2 §2.5)
+ W_hh · [0,0] = [0,0]
+ b_h = [0.1,-0.1]
a = [-0.55, 0.20]
h_3^bwd = [tanh(-0.55), tanh(0.20)] = [-0.5005, 0.1974]
```

**Step 2 (consume `x_2`, using `h_3^bwd` as "previous"):**
```
W_xh · x_2 = [-0.3, 0.8]     (Ch. 2 §2.4)
W_hh · h_3^bwd:
  row1: 0.2(-0.5005) + 0.4(0.1974) = -0.1001 + 0.0790 = -0.0211
  row2: -0.5(-0.5005) + 0.3(0.1974) = 0.2503 + 0.0592 = 0.3095
a = [-0.3-0.0211+0.1, 0.8+0.3095-0.1] = [-0.2211, 1.0095]
h_2^bwd = [tanh(-0.2211), tanh(1.0095)] = [-0.2175, 0.7658]
```

**Step 3 (consume `x_1`, using `h_2^bwd` as "previous"):**
```
W_xh · x_1 = [0.35, 0.50]     (Ch. 2 §2.3)
W_hh · h_2^bwd:
  row1: 0.2(-0.2175) + 0.4(0.7658) = -0.0435 + 0.3063 = 0.2628
  row2: -0.5(-0.2175) + 0.3(0.7658) = 0.1088 + 0.2297 = 0.3385
a = [0.35+0.2628+0.1, 0.50+0.3385-0.1] = [0.7128, 0.7385]
h_1^bwd = [tanh(0.7128), tanh(0.7385)] = [0.6127, 0.6274]
```

## 10.4 Final Bidirectional Representations (Concatenation)

| Position | Forward `h_t^fwd` | Backward `h_t^bwd` | Concatenated `h_t^bi` |
|---|---|---|---|
| 1 | [0.4219, 0.3799] | [0.6127, 0.6274] | [0.4219, 0.3799, 0.6127, 0.6274] |
| 2 | [0.0363, 0.5397] | [-0.2175, 0.7658] | [0.0363, 0.5397, -0.2175, 0.7658] |
| 3 | [-0.3157, 0.3305] | [-0.5005, 0.1974] | [-0.3157, 0.3305, -0.5005, 0.1974] |

**The critical thing to notice:** `h_1^bi` now carries information from `x_1` (via forward) **and** `x_2, x_3` (via backward `h_1^bwd`, which was built by processing `x_3` then `x_2` then `x_1`) — genuinely full-sequence context at *every* position, not just the ones near the end.

## 10.5 The Critical Interview Trap: You CANNOT Stream This

**Bidirectional models require the entire input sequence to be available before any output can be computed**, because the backward pass literally starts at the last token. This has an immediate, hard practical consequence:

- ❌ You **cannot** use a bidirectional model for autoregressive generation (predicting the next token — there is no "future" yet, by definition) or true real-time/streaming inference (e.g., live speech-to-text emitting words as the user speaks) without introducing buffering/latency.
- ✅ You **can** use bidirectional models whenever the **entire input is available upfront** at inference time — e.g., classifying a complete sentence's sentiment, tagging a complete sentence's parts of speech, encoding a complete search query, or as the **encoder** in an encoder-decoder architecture (Ch. 12) where the encoder sees the whole source sentence before decoding begins.

**This is a very common interview trap question** — being asked to design a live captioning or streaming voice-assistant system, and reflexively suggesting "let's use a BiLSTM" without noting that it fundamentally breaks the real-time constraint (you'd have to wait for the speaker to finish, or buffer with added latency to approximate a bounded look-ahead window).

## 10.6 Compute/Latency Cost

Bidirectional models **roughly double** parameter count and per-timestep compute (two independent recurrent networks instead of one) — on top of latency implications above, this is a genuine on-device/production cost worth naming explicitly in an Apple-context interview.

## 10.7 Interview Talking Points (L5 Signal)

- "A bidirectional model isn't a different cell type — it's an *architectural wrapper*: run the same cell type (vanilla/LSTM/GRU) twice, independently, in opposite directions, and concatenate. This modularity is worth stating explicitly since it shows you understand it's orthogonal to the RNN vs LSTM vs GRU choice."
- "The single biggest interview trap is forgetting that bidirectionality requires the *complete* sequence upfront — it's fundamentally incompatible with token-by-token generation or hard real-time streaming, and any system design answer proposing BiLSTM for a live/generative task needs to address that head-on (e.g., propose a fixed look-ahead window as a latency/accuracy trade-off instead of full bidirectionality)."
- "In encoder-decoder architectures, it's very common to make the **encoder** bidirectional (full input available) while keeping the **decoder** strictly unidirectional/autoregressive (each output token depends only on previously generated tokens) — this asymmetry is worth naming precisely in a seq2seq system design discussion (Chapter 12)."

## 10.8 Sample Interview Q&A

**Q: You're designing a live transcription feature for Siri. Would you use a bidirectional LSTM?**
A: Not directly for the streaming output — bidirectionality requires seeing future audio/tokens, which contradicts real-time emission. A common practical compromise is a **windowed** or **chunked** bidirectional model: buffer a small fixed window of audio (e.g., 300ms–1s) and run bidirectional processing within that window only, trading a small, bounded latency for the accuracy benefit of limited future context — rather than either a strictly causal unidirectional model or a fully-bidirectional model that would need to wait for the entire utterance.

**Q: Does making a model bidirectional fix the vanishing gradient problem?**
A: No — it's a completely orthogonal concern. Bidirectionality is about *what context is available* (past-only vs. past+future); vanishing gradients are about *how well gradient signal propagates across time* within a single directional pass. You can have a bidirectional vanilla RNN that still suffers badly from vanishing gradients (in both directions independently), or a unidirectional LSTM that handles long-range dependencies well without any bidirectionality at all. They're solving different problems and are freely combinable (e.g., BiLSTM combines both benefits).

**Q: In a BiLSTM, do the forward and backward LSTMs share weights?**
A: No — by convention (and in virtually all standard implementations) they have entirely independent, separately-learned weight matrices. Sharing them would be an unusual, non-standard design choice with no clear benefit, since the forward and backward directions are learning to solve genuinely different sub-problems (summarizing past vs. summarizing future).

## 10.9 Comprehension Check

1. Recompute `h_2^bwd` by hand — does it match `[-0.2175, 0.7658]`?
2. Why can't a bidirectional model be used for autoregressive text generation?
3. In an encoder-decoder architecture, which side is typically bidirectional and which side must remain unidirectional, and why?
4. Name a practical compromise for getting *some* future-context benefit in a genuinely real-time/streaming system, without full bidirectionality.

---
**Next:** Chapter 11 — Stacked/Deep RNNs: what happens when you stack multiple recurrent layers on top of each other, and how that interacts with everything covered so far.
