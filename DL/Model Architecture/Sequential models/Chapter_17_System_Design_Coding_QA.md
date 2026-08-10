# Chapter 17: Apple/Google System Design & Coding Q&A

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Format:** Full worked solutions to longer-form interview questions, spanning Chapters 1-16.

---

## 17.1 Coding Q1: Implement an LSTM Cell From Scratch

**Prompt:** "Implement a single LSTM cell's forward pass in NumPy, given `x_t`, `h_{t-1}`, `C_{t-1}` and the gate weight matrices."

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTMCell:
    def __init__(self, d_x, d_h, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.1  # illustrative init; real systems use Xavier/orthogonal init
        # One (W_x, W_h, b) triple per gate: forget, input, output, candidate
        self.W_xf, self.W_hf, self.b_f = rng.normal(scale=scale, size=(d_h, d_x)), rng.normal(scale=scale, size=(d_h, d_h)), np.zeros(d_h)
        self.W_xi, self.W_hi, self.b_i = rng.normal(scale=scale, size=(d_h, d_x)), rng.normal(scale=scale, size=(d_h, d_h)), np.zeros(d_h)
        self.W_xo, self.W_ho, self.b_o = rng.normal(scale=scale, size=(d_h, d_x)), rng.normal(scale=scale, size=(d_h, d_h)), np.zeros(d_h)
        self.W_xc, self.W_hc, self.b_c = rng.normal(scale=scale, size=(d_h, d_x)), rng.normal(scale=scale, size=(d_h, d_h)), np.zeros(d_h)

    def forward(self, x_t, h_prev, C_prev):
        f_t = sigmoid(self.W_xf @ x_t + self.W_hf @ h_prev + self.b_f)
        i_t = sigmoid(self.W_xi @ x_t + self.W_hi @ h_prev + self.b_i)
        o_t = sigmoid(self.W_xo @ x_t + self.W_ho @ h_prev + self.b_o)
        C_tilde = np.tanh(self.W_xc @ x_t + self.W_hc @ h_prev + self.b_c)
        C_t = f_t * C_prev + i_t * C_tilde     # elementwise — the additive gradient highway (Ch.5, 7)
        h_t = o_t * np.tanh(C_t)
        return h_t, C_t

    def forward_sequence(self, xs, h0=None, C0=None):
        d_h = self.b_f.shape[0]
        h = np.zeros(d_h) if h0 is None else h0
        C = np.zeros(d_h) if C0 is None else C0
        hs = []
        for x_t in xs:
            h, C = self.forward(x_t, h, C)
            hs.append(h)
        return np.array(hs)
```

**Validated against Chapter 6's hand-computed scalar example** (setting the weights explicitly to Ch. 6's values and running):
```
t=1: f=0.6900 i=0.6225 o=0.6457 c~=0.7616 C=0.4741 h=0.2850
t=2: f=0.6055 i=0.5762 o=0.5952 c~=0.4394 C=0.5402 h=0.2935
t=3: f=0.3163 i=0.3914 o=0.3747 c~=-0.7736 C=-0.1319 h=-0.0492
```
These match Chapter 6's by-hand values to within rounding error (e.g., `h_3 = -0.0492` here vs. `-0.0493` by hand) — a good sanity-check habit for any interview coding question: **always spot-check your code against a hand-computed example if one is available.**

**Interview follow-ups to expect:**
- "What's the time complexity per timestep?" → `O(d_h·d_x + d_h²)` per gate, ×4 gates.
- "How would you vectorize across a batch?" → add a batch dimension to `x_t` (shape `(B, d_x)`) and use `x_t @ W_xf.T` instead of `W_xf @ x_t`, batching all four gates' matmuls.
- "How would you make this more efficient?" → concatenate the four `W_x*` matrices into one `(4·d_h, d_x)` matrix and do a single matmul, then split the result into four gate pre-activations — standard practice in real implementations (this is what PyTorch's `nn.LSTM` does internally).

## 17.2 Coding Q2: Gradient Clipping by Global Norm

**Prompt:** "Implement gradient clipping by global norm, given a list of gradient arrays (one per parameter tensor)."

```python
import numpy as np

def clip_grad_norm(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))  # global norm across ALL params
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)  # epsilon avoids division-by-zero
        grads = [g * scale for g in grads]
    return grads, total_norm

# Sanity check against Ch. 14.3's hand example:
g_clipped, norm = clip_grad_norm([np.array([3.0, 4.0])], max_norm=2.0)
print(g_clipped, norm)   # expect ~[1.2, 1.6], norm=5.0
```
**Common interview trap here:** clipping each gradient tensor *independently* (per-parameter norm) instead of computing one **global** norm across all parameters together — this is a frequently-tested subtle bug, since independent per-tensor clipping changes the relative scaling between different parameters' updates in an unintended way.

## 17.3 System Design Q1: Design a Real-Time Speech Transcription System

**Prompt:** "Design a live, streaming speech-to-text system (Siri-adjacent). Walk through your architecture."

**Model answer structure:**

1. **Clarify the constraint that shapes everything:** this is a *streaming* task — output must be emitted incrementally as the user speaks, with a tight latency budget (e.g., sub-second). This immediately rules out full-sequence bidirectional processing (Ch. 10.5) and constrains decoding strategy.

2. **Architecture choice:** an LSTM/GRU-based **acoustic encoder** processing audio frames sequentially (unidirectional, or a small **chunked bidirectional** window — Ch. 10.8 — trading a bounded amount of latency, e.g. 200-500ms, for improved accuracy from limited look-ahead). Note explicitly: a full BiLSTM is disqualified by the streaming requirement, but a *windowed* bidirectional model is a legitimate middle ground worth naming.

3. **Decoding:** greedy or small-beam-width decoding (Ch. 12.6) per chunk to keep latency low — full beam search with a large beam is likely too slow for real-time; state the latency/accuracy trade-off explicitly rather than picking beam width arbitrarily.

4. **State management:** the model must carry hidden state across chunks within an utterance (Ch. 15.4) — design a per-utterance state cache, with clear expiry when the utterance ends (silence detection / explicit end-of-speech signal) to bound memory.

5. **On-device constraints (Apple-specific):** given privacy and latency requirements, likely want this fully on-device — invoke Ch. 15.5's toolkit: GRU over LSTM for the ~25% compute savings (validate this doesn't hurt WER — word error rate — meaningfully), quantization, and possibly distillation from a larger server-side model.

6. **Training-serving skew risks to flag explicitly (Ch. 15.6):** the chunking scheme used at serving time (fixed window size, overlap handling) must exactly match how training data was chunked, or the model will see a different effective "look-ahead" distribution than it was trained on — a concrete, model-specific skew risk worth naming unprompted.

7. **Metrics & monitoring:** word error rate (WER) is the core offline metric; in production, also track end-to-end latency percentiles (p50/p95/p99) and monitor for utterance-length distribution drift (Ch. 15.9 Q2's bucketing idea generalizes to monitoring, not just training-time efficiency).

**What separates an L5 answer here:** naming the streaming-vs-bidirectionality tension *unprompted*, proposing the concrete windowed-bidirectional compromise rather than defaulting to either extreme, and flagging the chunking-consistency training-serving skew risk without being asked.

## 17.4 System Design Q2: Design a Session-Aware Search Ranking Model

**Prompt:** "Design a system that re-ranks search results using a user's within-session query history (Apple Search & AI relevant)."

**Model answer structure:**

1. **Framing:** this is a sequence-conditioned ranking task — treat the session's query history as a sequence fed through an RNN/LSTM/GRU encoder, whose final (or attention-pooled, Ch. 13) hidden state becomes a "session intent" feature used alongside standard ranking features.

2. **Architecture choice:** likely GRU or a small LSTM (not necessarily bidirectional — at serving time, only *past* queries in the session exist, so bidirectionality wouldn't even be applicable here, mirroring the seq2seq decoder's constraint from Ch. 12.2) processing the session query sequence, producing a session-embedding used downstream by a ranking model.

3. **Handling variable, growing session lengths at serving:** must support **incremental** state updates — as each new query arrives, update the session's hidden state rather than reprocessing the full history from scratch (Ch. 15.4's stateful serving pattern) — critical for latency at query-time.

4. **Training-serving skew to flag (Ch. 15.6):** training must use **prefixes** of sessions (not just complete sessions) so the model has seen the sparse-early-session-history regime it will actually encounter for many real-time queries — directly the failure mode described in Ch. 15.8 Q2.

5. **Cold start:** first query in any session has no history — the model needs a well-defined behavior for empty/near-empty history (e.g., a learned "no history" initial state, rather than relying on ranking features alone in that case, or gracefully falling back to a session-agnostic ranker).

6. **Batching at training/eval time:** session lengths are highly skewed (mostly short, some very long) — bucketing (Ch. 14.5) is directly relevant to keep training efficient.

7. **Evaluation:** offline, evaluate specifically on short-session and long-session slices separately, not just aggregate metrics — aggregate metrics can mask exactly the early-session degradation failure mode from 17.4.4.

**What separates an L5 answer here:** connecting back explicitly to the *specific* training-serving skew failure mode from earlier chapters (not generic "watch for skew" hand-waving), and proactively addressing cold start and incremental-state-update requirements rather than just describing the model architecture in isolation.

## 17.5 Coding Q3: Implement Masked Attention

**Prompt:** "Implement additive (Bahdanau-style) attention with padding masking, in NumPy."

```python
import numpy as np

def masked_attention(s_prev, encoder_states, W_s, W_h, v, mask):
    """
    s_prev: (d_h,) decoder's previous hidden state
    encoder_states: (T, d_h) all encoder hidden states
    mask: (T,) boolean array, True = real token, False = padding
    """
    scores = np.array([
        v @ np.tanh(W_s @ s_prev + W_h @ h_i)
        for h_i in encoder_states
    ])                                    # (T,)

    scores = np.where(mask, scores, -1e9)  # Ch.14.4: mask BEFORE softmax, not after
    scores -= scores.max()                 # numerical stability (standard softmax trick)
    exp_scores = np.exp(scores)
    alpha = exp_scores / exp_scores.sum()  # (T,) attention weights, ~0 at padded positions

    context = alpha @ encoder_states       # (d_h,) weighted sum — Ch.13's c_t
    return context, alpha
```

**Interview follow-up:** "Why subtract `scores.max()` before exponentiating?" → Pure numerical stability — `exp()` of a large positive score can overflow to `inf`; subtracting the max shifts all scores to be `≤0` without changing the softmax output (since softmax is shift-invariant), avoiding overflow while `-1e9` (now very negative) still safely underflows to `exp()≈0`.

## 17.6 Practice Prompts (Do These Yourself Before Your Interview)

1. Extend the LSTM cell code (17.1) to a **GRU cell**, and validate it against Chapter 9's hand-computed scalar numbers.
2. Implement **truncated BPTT** conceptually in pseudocode: given a long sequence and chunk size `k`, show where you'd insert a "detach" operation.
3. Design question: "Your bidirectional NER model works great offline but a colleague wants to deploy it for real-time chat moderation, flagging toxic spans as users type." Identify the architectural conflict and propose two alternatives, with trade-offs.
4. Coding: implement bucketing — given a list of sequence lengths and a target batch size, write a function that groups sequences into batches to minimize total padding.

---
**Next:** Chapter 18 — Master Cheat Sheet: every equation, every key number, and the full comprehension-check answer key, in one condensed reference document.
