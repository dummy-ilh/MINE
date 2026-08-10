# Chapter 13: Attention — The Bridge Out of the RNN Bottleneck

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 12 (fixed context-vector bottleneck), your existing Transformer/attention knowledge

**Note:** since you've already worked through "Attention Is All You Need" in depth, this chapter is intentionally brief — it focuses specifically on how attention was *originally* introduced (Bahdanau et al., 2014) to patch a specific RNN problem, before it was later generalized into the self-attention that replaced recurrence entirely.

---

## 13.1 The Problem, Restated

Chapter 12 ended with the core bottleneck: the decoder only ever sees one fixed-size context vector `c`, no matter the source length. **Attention's fix: let the decoder look at *all* encoder hidden states directly, at every decoding step, weighted by relevance to what it's currently trying to generate.**

## 13.2 The Equations (Bahdanau-style additive attention)

At each decoder step `t`, given the decoder's previous hidden state `s_{t-1}` and **all** encoder hidden states `h_1, ..., h_T`:

```
score(s_{t-1}, h_i) = vᵀ · tanh(W_s·s_{t-1} + W_h·h_i)      — "how relevant is encoder position i right now?"
α_{t,i} = softmax_i( score(s_{t-1}, h_i) )                   — normalize scores into weights summing to 1
c_t = Σᵢ α_{t,i} · h_i                                        — weighted sum: THIS step's context vector
```

**Critical difference from Chapter 12:** `c_t` now has a **`t` subscript** — a fresh, differently-weighted combination of *all* encoder states is computed at every single decoding step, instead of one fixed `c` reused throughout decoding.

## 13.3 Numerical Walkthrough

**Encoder hidden states** (reusing Chapter 2's exact vanilla RNN outputs):
```
h_1 = [0.4219, 0.3799]
h_2 = [0.0363, 0.5397]
h_3 = [-0.3157, 0.3305]
```

**Decoder's previous hidden state** (reusing Chapter 12's `h_1^dec`, treated here as `s_1`):
```
s_1 = [0.3538, 0.6215]
```

**Attention weights:**
```
W_s = [[0.3, 0.2], [-0.1, 0.4]]
W_h = [[0.5, -0.2], [0.1, 0.3]]
v   = [0.6, -0.4]
```

**Step A: `W_s · s_1`**
```
row1: 0.3(0.3538) + 0.2(0.6215) = 0.1061 + 0.1243 = 0.2304
row2: -0.1(0.3538) + 0.4(0.6215) = -0.0354 + 0.2486 = 0.2132
→ [0.2304, 0.2132]
```

**Step B: compute score against `h_1`**
```
W_h · h_1: row1: 0.5(0.4219)+(-0.2)(0.3799)=0.2110-0.0760=0.1350
           row2: 0.1(0.4219)+0.3(0.3799)=0.0422+0.1140=0.1562
sum with W_s·s_1: [0.2304+0.1350, 0.2132+0.1562] = [0.3654, 0.3694]
tanh: [tanh(0.3654), tanh(0.3694)] = [0.3502, 0.3536]
score_1 = v·[0.3502, 0.3536] = 0.6(0.3502) + (-0.4)(0.3536) = 0.2101 - 0.1414 = 0.0687
```

**Step C: score against `h_2`**
```
W_h · h_2: row1: 0.5(0.0363)+(-0.2)(0.5397)=0.0182-0.1079=-0.0898
           row2: 0.1(0.0363)+0.3(0.5397)=0.0036+0.1619=0.1655
sum: [0.2304-0.0898, 0.2132+0.1655] = [0.1406, 0.3787]
tanh: [0.1398, 0.3612]
score_2 = 0.6(0.1398) + (-0.4)(0.3612) = 0.0839 - 0.1445 = -0.0606
```

**Step D: score against `h_3`**
```
W_h · h_3: row1: 0.5(-0.3157)+(-0.2)(0.3305)=-0.1579-0.0661=-0.2240
           row2: 0.1(-0.3157)+0.3(0.3305)=-0.0316+0.0992=0.0676
sum: [0.2304-0.2240, 0.2132+0.0676] = [0.0064, 0.2808]
tanh: [0.0064, 0.2735]
score_3 = 0.6(0.0064) + (-0.4)(0.2735) = 0.0038 - 0.1094 = -0.1056
```

**Step E: softmax over `[0.0687, -0.0606, -0.1056]`**
```
e^0.0687≈1.0711, e^-0.0606≈0.9412, e^-0.1056≈0.8998
sum ≈ 2.9121
α_1 = 1.0711/2.9121 = 0.3678
α_2 = 0.9412/2.9121 = 0.3232
α_3 = 0.8998/2.9121 = 0.3090
```

**Step F: weighted context vector**
```
c_t = 0.3678·[0.4219,0.3799] + 0.3232·[0.0363,0.5397] + 0.3090·[-0.3157,0.3305]
    = [0.1552, 0.1397] + [0.0117, 0.1744] + [-0.0976, 0.1021]
    = [0.0694, 0.4163]
```

**Interpretation:** the model gives slightly more weight (36.8%) to `h_1` than to `h_2` (32.3%) or `h_3` (30.9%) — a mild preference, not yet strongly peaked, which is realistic for untrained/illustrative weights. **In a well-trained model, these weights typically become much more sharply peaked** on the few positions genuinely relevant to generating the current output token (e.g., near-1.0 on the aligned source word, near-0 elsewhere) — that sharpness is exactly what makes attention weights visually interpretable in attention heatmaps.

## 13.4 This Is Also How the Bottleneck Actually Gets Fixed

`c_t` is now a **direct, differentiable function of every encoder hidden state** — no more forcing all information through one fixed-size vector regardless of source length. Longer sequences simply mean more terms in the weighted sum, not more compression pressure on a fixed-size bottleneck.

## 13.5 The Bridge to What You Already Know

This additive-attention mechanism, sitting *on top of* an RNN encoder-decoder, is the direct historical ancestor of Transformer self-attention. The generalization step (which you've already studied) was: **why require an underlying RNN at all?** If attention can already relate any two positions directly regardless of distance, the recurrent hidden-state pathway becomes largely redundant for capturing long-range dependency — leading directly to "Attention Is All You Need," which replaces the RNN encoder/decoder entirely with stacked self-attention, keeping only the alignment-score-then-weighted-sum idea you just hand-computed above (generalized to `Q, K, V` and scaled dot-product form).

## 13.6 Interview Talking Points (L5 Signal)

- "Attention wasn't originally proposed as a replacement for RNNs — it was a targeted fix for the fixed-context-vector bottleneck in RNN encoder-decoders. Understanding it in that original context makes the jump to 'why not remove the RNN entirely' (Transformers) a much more natural, motivated step rather than an arbitrary architecture choice."
- "The core computation — score, softmax-normalize, weighted sum — is identical in spirit to scaled dot-product attention; what changed for Transformers is (1) the scoring function (dot-product/scaled-dot-product vs. this additive form), and (2) removing the recurrent state entirely, letting attention be computed over *all* pairs of positions in parallel."
- "Bahdanau attention (additive, `tanh(W_s·s + W_h·h)`) vs. Luong attention (multiplicative, e.g. `sᵀ·W·h` or plain dot product) — the multiplicative form is what scaled up efficiently into Transformer's `QKᵀ` — worth naming both if asked to trace the lineage."

## 13.7 Sample Interview Q&A

**Q: Why did attention get proposed for RNN encoder-decoders specifically, rather than, say, for a single-RNN classification model?**
A: Because the bottleneck it fixes is specific to encoder-decoder architectures — the fixed-size context vector connecting two separate RNNs. A single-RNN classifier already has direct access to its own hidden states as it processes the sequence; there's no analogous "everything must funnel through one vector" chokepoint to fix.

**Q: In this example, why aren't the attention weights more sharply peaked?**
A: Because the weights `W_s, W_h, v` here are illustrative/untrained values, not the result of gradient descent optimizing for a real task. In a trained model, these weights are shaped specifically so that the score function produces large positive scores for genuinely relevant source positions and large negative scores elsewhere, which softmax then turns into a sharp, near-one-hot distribution.

**Q: What's the computational cost of attention relative to the fixed-context-vector approach, per decoding step?**
A: Attention requires computing a score against *every* encoder position (`O(T)` per decoding step, `O(T²)` total for a length-T sequence being decoded into a length-T output), versus `O(1)` for reusing a single fixed context vector. This quadratic cost is a genuine trade-off attention introduces — and is the same `O(T²)` cost that later became a well-known scaling bottleneck for Transformer self-attention on very long sequences.

## 13.8 Comprehension Check

1. Recompute `score_2` from scratch — did you get `-0.0606`?
2. Why does `c_t` (with attention) not suffer from the same fixed-capacity bottleneck as `c` in Chapter 12?
3. In one sentence, what did Transformers remove from this picture that Bahdanau attention still relied on?
4. Name the two attention variants (additive vs. multiplicative) and which one scales more directly into Transformer-style attention.

---
**Next:** Chapter 14 — Training mechanics: teacher forcing, truncated BPTT, gradient clipping, and padding/masking — the practical machinery every one of these architectures actually needs in production.
