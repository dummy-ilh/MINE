# Module 6 — Inference & Decoding (Master Notes, Maximum Depth)

## 0. The setup — what "inference" actually computes

At inference, the model has learned `P(x_t | x_1, ..., x_{t-1})` (Module 2's CLM formula) — a full probability distribution over the *entire vocabulary* for the next token, at every position. The model itself never "picks" a word — a separate **decoding algorithm** decides how to turn that probability distribution into an actual chosen token, and this choice happens at every single generation step. Different decoding strategies trade off **quality, diversity, and determinism** very differently, even using the exact same underlying trained model and the exact same probability distribution.

---

## 1. Greedy Decoding

### The rule
At every step, pick the single token with the highest probability:
```
x_t = argmax P(x_t | x_1, ..., x_{t-1})
```
### Why it fails — the concrete numerical problem
Greedy decoding is **locally optimal at each step but not globally optimal for the whole sequence** — picking the best next word right now can lock you into a worse overall sequence.

**Worked example**: suppose generating "The weather today is ___":
- Step 1 options: "sunny" (P=0.4), "cold" (P=0.35), "nice" (P=0.25) → greedy picks "sunny".
- Given "The weather today is sunny ___": "and" (P=0.9), "," (P=0.1) → greedy picks "and".
- Given "...sunny and ___": best continuation only reaches P=0.3 for "warm", because "sunny and warm" is a slightly awkward/redundant phrase, whereas if step 1 had picked "nice" instead:
- Given "...is nice ___": "and" (P=0.6), then "...nice and warm" (P=0.85) — a much more probable, fluent overall sequence.

Multiplying through: `P("sunny and warm") = 0.4 × 0.9 × 0.3 = 0.108`, while `P("nice and warm") = 0.25 × 0.6 × 0.85 = 0.1275` — the **globally better sequence was never found**, because greedy locked in "sunny" at step 1 purely because it had the highest single-step probability, without any lookahead. This exact failure mode — high locally-probable choices leading to a lower overall sequence probability — is the core argument for beam search.

**Additional practical failure**: greedy decoding is also notorious for producing **repetitive loops** ("I think that I think that I think that...") on open-ended generation, because once a repetitive pattern starts, continuing the repetition is often literally the highest-probability next token (the model has strong evidence for what comes next when it's already seen that exact pattern before in-context).

---

## 2. Beam Search

### The core idea
Instead of keeping only 1 candidate sequence (greedy) or trying literally all possible sequences (computationally impossible — grows exponentially with vocab_size^sequence_length), keep the **top-k partial sequences** (k = "beam width") at every step, expand each by all possible next tokens, then prune back down to the top-k highest-*cumulative*-probability sequences.

### Worked numerical example (beam width k=2)
Vocabulary simplified to {A, B, C} for illustration. Step 1 probabilities: P(A)=0.5, P(B)=0.3, P(C)=0.2 → keep top-2 beams: ["A"] (0.5) and ["B"] (0.3).

**Step 2** — expand each beam by all next-token options, compute cumulative probability (product):
- From "A": P(A→X)=0.6 → cum=0.30; P(A→Y)=0.4 → cum=0.20
- From "B": P(B→X)=0.9 → cum=0.27; P(B→Y)=0.1 → cum=0.03

All 4 candidates ranked by cumulative probability: "A→X" (0.30), "B→X" (0.27), "A→Y" (0.20), "B→Y" (0.03). Keep top-2: **["A","X"] (0.30) and ["B","X"] (0.27)**.

Notice: "B→X" (cumulative 0.27) survived into the beam even though "B" alone (0.3) was *less* probable at step 1 than "A" (0.5) — this is exactly the lookahead benefit beam search provides over greedy: it never permanently commits to only the single best step-1 choice, keeping a second candidate alive in case it leads to a better overall sequence, exactly like the "nice and warm" vs "sunny and warm" scenario in Section 1.

### The known downside — beam search is not free of problems either
Beam search tends to produce **generic, "safe," repetitive text** for open-ended generation (it systematically favors high-probability-but-bland sequences, since it's explicitly hunting for maximum cumulative probability) — this is why beam search is common for tasks with a fairly narrow "correct" answer space (machine translation, summarization) but is **rarely used for open-ended chat/creative generation**, where sampling-based methods (below) produce more natural, varied text.

### Cost note
Compute/memory cost scales with beam width k (you're tracking and expanding k sequences in parallel at every step instead of 1) — a direct, tunable quality-vs-cost tradeoff knob.

---

## 3. Sampling-based Decoding — Temperature, Top-k, Top-p

### Temperature — the formula and what it actually does
Before sampling, the model's raw output scores (**logits**, pre-softmax) are divided by a temperature `T` before applying softmax:
```
P(x_i) = exp(z_i / T) / Σ_j exp(z_j / T)
```
where `z_i` is the raw logit for token i.

**Numerical example**: say raw logits for 3 tokens are `z = [2.0, 1.0, 0.5]`.

**T=1.0 (no change, standard softmax)**:
```
exp(2.0)=7.389, exp(1.0)=2.718, exp(0.5)=1.649, sum=11.756
P = [0.629, 0.231, 0.140]
```

**T=0.5 (lower temperature — sharpens the distribution)**: divide logits by 0.5 first → `z/T = [4.0, 2.0, 1.0]`
```
exp(4.0)=54.60, exp(2.0)=7.389, exp(1.0)=2.718, sum=64.71
P = [0.844, 0.114, 0.042]
```
Notice the top token's probability jumped from 0.629 → 0.844 — **lower temperature makes the distribution more peaked/confident, pushing sampling behavior toward greedy** (as T→0, sampling becomes exactly equivalent to greedy argmax).

**T=2.0 (higher temperature — flattens the distribution)**: `z/T = [1.0, 0.5, 0.25]`
```
exp(1.0)=2.718, exp(0.5)=1.649, exp(0.25)=1.284, sum=5.651
P = [0.481, 0.292, 0.227]
```
The distribution flattened considerably (0.629→0.481 for the top token) — **higher temperature increases randomness/diversity, giving lower-probability tokens a meaningfully higher chance of being sampled**, at the cost of more risk of incoherent output.

### Top-k sampling
Restrict sampling to only the `k` highest-probability tokens (e.g., k=40), **renormalize their probabilities to sum to 1** among just that subset, then sample from that restricted, renormalized distribution. This prevents sampling from the "long tail" of very low-probability, likely-nonsensical tokens, while still allowing some randomness among the plausible top candidates.

**Weakness**: a *fixed* k doesn't adapt to context — sometimes the model is very confident and only 3 tokens are remotely reasonable (fixed k=40 would include 37 garbage options), and sometimes the model is quite uncertain and even the top 40 aren't enough to cover reasonable options. This context-insensitivity is exactly what top-p was designed to fix.

### Top-p (nucleus) sampling — the formula and worked example
Instead of a fixed *count* of tokens, choose the **smallest set of tokens whose cumulative probability exceeds threshold p** (e.g., p=0.9), then sample (renormalized) from just that dynamically-sized set.

**Worked numerical example**: suppose sorted probabilities are `[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]` (sums to 1.0). With p=0.9:
```
Cumulative: 0.5 → 0.7 → 0.85 → 0.93 (crosses 0.9 here) → stop
```
The nucleus includes the first 4 tokens (cumulative 0.93 ≥ 0.9 threshold), so we keep `[0.5, 0.2, 0.15, 0.08]`, renormalize by dividing each by their sum (0.93):
```
Renormalized: [0.538, 0.215, 0.161, 0.086]
```
Sample from just these 4 renormalized probabilities. **Why this adapts better than top-k**: if the distribution were instead very peaked, e.g. `[0.85, 0.08, 0.04, 0.03]`, the p=0.9 nucleus would only need the **first 2** tokens (cumulative 0.93) — the set size shrinks automatically when the model is confident, and grows automatically when the model is uncertain (a flatter distribution needs more tokens to reach the same cumulative threshold) — this dynamic sizing is the core practical advantage over fixed top-k.

### Practical combination
Production systems typically apply **temperature, then top-k or top-p (or both), then sample** — these are complementary, stackable knobs, not mutually exclusive alternatives; a common real-world default is something like temperature=0.7-1.0 combined with top-p=0.9.

---

## 4. KV Caching — the memory/compute mechanism that makes autoregressive generation fast

### The problem it solves
In self-attention, computing the output at position `t` requires the **Key (K) and Value (V) vectors of every previous position** (1 through t), attending over all of them. Naively, generating token t+1 would require **recomputing K and V for all positions 1 through t all over again** — even though positions 1 through t never change once they're generated. This is enormously wasteful: generating a sequence of length N naively would cost `O(N²)` redundant K/V computation (recomputing position 1's K/V N times total, position 2's K/V N-1 times, etc.).

### The fix
**Cache the K and V vectors for every position, the first time they're computed, and simply append the new token's K/V to the cache at each new step** — never recompute K/V for past positions. Generating the next token only requires computing Q (query), K, V for the *single new position*, then attending that one new query against the full cached K/V history.

### Numerical illustration of the savings
Say hidden dimension `d=4096`, and you're generating a sequence of length N=1000, with L=32 transformer layers.

**Without KV caching**: to generate token 1000, you'd recompute K and V for all 1000 previous positions, across all 32 layers, from scratch — and you'd repeat this entire recomputation at *every single one* of the 1000 generation steps. Total K/V computation work across the whole generation scales roughly as `O(N² × L × d)` — quadratic in sequence length purely from this redundant recomputation.

**With KV caching**: at each step, you compute K/V for exactly 1 new position (across all L layers), giving total K/V computation work scaling as `O(N × L × d)` — **linear** in sequence length. For N=1000, this is roughly a 1000x reduction in K/V-computation work compared to the naive approach (the quadratic term collapses to linear) — this is precisely why every production LLM serving system implements KV caching; without it, long-sequence generation would be computationally infeasible at production latency/cost targets.

### The memory cost tradeoff (the concrete downside to know)
KV cache memory grows **linearly with sequence length**, and must be stored per-layer, per-head:
```
KV cache size ≈ 2 (K and V) × L (layers) × N (sequence length) × d (hidden dim) × batch_size × bytes_per_value
```
**Numerical example**: for a 32-layer model, hidden dim 4096, sequence length 4096, batch size 1, using fp16 (2 bytes/value):
```
2 × 32 × 4096 × 4096 × 1 × 2 bytes = 2,147,483,648 bytes ≈ 2 GB
```
Just for the KV cache of a *single* sequence at this length — this is why long-context serving is memory-bound, not just compute-bound, and why techniques like **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** exist: they reduce the number of distinct K/V "heads" that need to be cached (sharing K/V projections across multiple query heads) specifically to shrink this cache memory footprint, at a small quality cost — worth naming these if asked "how do you reduce KV cache memory."

---

## 5. Speculative Decoding

### The core idea
Autoregressive generation is fundamentally **sequential and latency-bound** — you must generate token t before you can even start computing token t+1, because token t+1's computation depends on t as input. Speculative decoding breaks this bottleneck using a **small, fast "draft" model** to guess several tokens ahead cheaply, then verifies (or rejects) those guesses using the large "target" model in a single parallel pass.

### The mechanism, step by step
1. A small draft model (much cheaper/faster than the target model) autoregressively generates a short candidate sequence of, say, 4-5 tokens.
2. The large target model then processes **all of those candidate positions in a single forward pass** (not one at a time) — this is possible/cheap because verifying a *given* sequence of tokens in parallel is a much cheaper operation than *generating* that sequence token-by-token would have been, since parallel verification doesn't have the same sequential-dependency bottleneck.
3. Compare the target model's actual next-token distribution at each position against what the draft model guessed. **Accept** every draft token where the target model agrees was a good choice (using a rejection-sampling-style acceptance criterion so this remains mathematically equivalent to sampling directly from the target model — this is the key correctness property: speculative decoding provably produces exactly the same output distribution as the target model alone would have, just faster). At the first position where the target model rejects the draft's guess, **discard the rest of the draft sequence from that point on**, sample the correct token directly from the target model's own distribution at that position, and restart the drafting process from there.

### Numerical intuition for the speedup
If the draft model's guesses are accepted, on average, for 3 out of every 4 speculated tokens, you effectively get ~4 tokens' worth of output for roughly the cost of **1 target-model forward pass** (parallel verification) **plus a few cheap draft-model forward passes** — instead of needing 4 full, strictly-sequential target-model forward passes. Since the large target model's forward pass is by far the dominant cost, and parallel verification of several positions costs roughly the same as verifying just one position (both are dominated by fixed per-call overhead plus one comparable-sized matrix multiply, since modern GPUs are heavily underutilized during memory-bound single-token generation), the **wall-clock speedup can be substantial** (commonly cited real-world figures are in the 2-3x range) **without any change to output quality/distribution** — this last point (exactness, not an approximation) is the detail interviewers most want to hear stated explicitly.

---

## 6. Context Length Extension — RoPE and ALiBi (ties directly back to your Transformer architecture knowledge)

### The underlying problem
A Transformer trained on sequences up to length N (say, 4096) often performs poorly if you naively try to run it on longer sequences at inference (say, 16000) — positional encoding schemes need to **generalize/extrapolate** to positions never seen during training.

### RoPE (Rotary Position Embedding) — the mechanism
Instead of adding a separate positional embedding vector to the token embedding (as in the original Transformer's sinusoidal encoding), RoPE **rotates** the Query and Key vectors by an angle that depends on their absolute position, using 2D rotation matrices applied to pairs of dimensions within the vector. The key mathematical property: the dot product between a rotated Query at position `m` and a rotated Key at position `n` ends up depending **only on the relative distance `(m - n)`**, not on the absolute positions `m` and `n` individually — this relative-position property is exactly what helps generalization, since "attend to something 5 tokens back" is a pattern the model can learn once and reuse regardless of whether that pattern occurs at absolute position 10 or position 10,000.

**RoPE scaling for length extension**: since RoPE's rotation angle is a function of position, at inference time you can apply a **scaling factor** to effectively "compress" position indices (e.g., treat position 16000 as if it were position 4000 by dividing all position indices by 4) so the rotation angles stay within the range the model saw during training — this is the core trick behind techniques like "Position Interpolation" used to extend context windows of RoPE-based models (Llama and most modern LLMs use RoPE) without retraining from scratch.

### ALiBi (Attention with Linear Biases) — the mechanism
A different, arguably simpler approach: don't modify Query/Key vectors at all — instead, directly **subtract a penalty from the raw attention scores**, proportional to the distance between the query and key positions:
```
attention_score(i,j) = (Q_i · K_j) - m × |i - j|
```
where `m` is a fixed, head-specific slope (different attention heads get different, geometrically-spaced slope values `m`, so some heads focus more locally and others more globally). **In plain words**: tokens further away automatically get their raw attention score penalized more, with no learned positional embedding parameters at all — just a fixed, hard-coded linear penalty based on distance. ALiBi's authors demonstrated notably strong length-extrapolation performance (training on short sequences, evaluating well on much longer ones) specifically because this penalty structure naturally, smoothly discourages attending far away without ever needing position representations that go "out of distribution" the way learned/sinusoidal embeddings can at unseen lengths.

### One-line comparison to have ready
"RoPE modifies Q/K vectors via rotation to encode *relative* position implicitly in the dot product, and is the dominant choice in modern LLMs (Llama, GPT-NeoX-style models); ALiBi instead leaves Q/K untouched and directly biases the attention *scores* with a distance-proportional penalty, trading some peak in-distribution performance for often better raw length-extrapolation behavior in the original comparisons."

---

## 7. Side-by-side summary table (memorize this cold)

| | Greedy | Beam Search | Temperature/Top-k/Top-p Sampling |
|---|---|---|---|
| Determinism | Fully deterministic | Fully deterministic | Stochastic |
| Lookahead | None | k-way lookahead | None (per-step distribution shaping only) |
| Common failure mode | Repetition loops, myopic choices | Generic/bland "safe" text | Can be incoherent if temperature too high / p,k too loose |
| Typical use case | Rarely used alone in production | Translation, summarization (narrow answer space) | Open-ended chat/creative generation |

| | KV Caching | Speculative Decoding |
|---|---|---|
| What it optimizes | Avoids redundant K/V recomputation | Avoids strict sequential-only generation |
| Cost/tradeoff | Linear-growing memory footprint per sequence | Requires a second, smaller draft model |
| Changes output distribution? | No | No (provably exact, via rejection sampling) |

---

## 8. Quick-fire Q&A (self-test)

**Q: Why can greedy decoding produce a lower-probability overall sequence than an alternative path, even though it always picks the locally best token?**
A: Because it commits irreversibly to the single highest-probability token at each step with no lookahead — a slightly lower-probability first choice can lead to much higher-probability continuations later, and greedy can never discover or backtrack to that better overall path.

**Q: In beam search with width k=2, why might a beam that wasn't the single best option at step 1 still survive into the final beam set?**
A: Because beam search ranks by *cumulative* sequence probability across all surviving beams at each step, not by the step-1 probability alone — a lower step-1-probability beam can still be retained if its subsequent expansions give it a higher running cumulative probability than the top beam's worse expansions.

**Q: Write the temperature-scaled softmax formula and explain what T→0 and T→∞ each converge to.**
A: `P(x_i) = exp(z_i/T) / Σ_j exp(z_j/T)`. As T→0, the distribution becomes maximally peaked and sampling converges to greedy argmax; as T→∞, all logits are scaled toward 0 and the distribution converges to uniform random sampling over the vocabulary.

**Q: What's the core weakness of top-k sampling that top-p (nucleus) sampling fixes?**
A: Top-k uses a fixed token count regardless of context, which can be too permissive when the model is very confident (few tokens are reasonable) or too restrictive when the model is very uncertain (many tokens are reasonable). Top-p dynamically sizes the candidate set based on cumulative probability mass, adapting automatically to the model's confidence at each step.

**Q: Explain, with the complexity classes, why KV caching is necessary for practical autoregressive generation.**
A: Without caching, generating a sequence of length N requires recomputing K/V for all previous positions at every step, giving O(N²) redundant K/V computation work. Caching K/V once computed and only computing K/V for the new position at each step reduces this to O(N) — linear instead of quadratic in sequence length.

**Q: Why does the KV cache's memory cost become a serving bottleneck, and what's one architectural fix?**
A: KV cache memory grows linearly with sequence length, layers, hidden dimension, and batch size — for long contexts and large batches this can reach many GB per sequence, becoming memory-bound rather than compute-bound. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce this by sharing K/V projections across multiple query heads, shrinking the cache size at a small quality cost.

**Q: Why is speculative decoding described as "exact" rather than an approximation, despite using a smaller, less accurate draft model?**
A: It uses a rejection-sampling-style acceptance criterion when comparing draft-model guesses against the target model's actual distribution, which is mathematically constructed so the final accepted output distribution is provably identical to what the target model alone would have produced by itself — the draft model only affects speed, never correctness/quality.

**Q: What's the key mathematical property RoPE achieves that a naive absolute positional embedding does not?**
A: RoPE's rotation makes the Query-Key dot product depend only on the *relative* distance between two positions, not their absolute positions — this relative-position property helps the learned attention patterns generalize to sequence lengths not seen during training, unlike absolute positional schemes which can go fully out-of-distribution beyond the trained length.

**Q: How does ALiBi handle position information differently from RoPE?**
A: ALiBi leaves Query/Key vectors completely unmodified and instead directly subtracts a fixed, head-specific, distance-proportional penalty from the raw attention scores after the Q·K dot product — no learned or rotated positional representations are involved at all, just a hard-coded linear bias based on token distance.

---
*End of Module 6 (maximum depth). Next: Module 7 — Efficiency & Serving (quantization, mixed precision, MoE routing, distillation).*
