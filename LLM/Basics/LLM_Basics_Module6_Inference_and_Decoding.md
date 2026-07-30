# Module 6 — Inference & Decoding (Master Notes, Maximum Depth)

> **Note on this version:** This file preserves 100% of your original notes, in their original order and wording. Every addition is clearly tagged: `📌 Added Explanation`, `🧮 Numerical Example`, `❓ Interview Q&A`, or `🔎 Accuracy Flag`. Nothing original was deleted or shortened.

## 0. The setup — what "inference" actually computes

At inference, the model has learned `P(x_t | x_1, ..., x_{t-1})` (Module 2's CLM formula) — a full probability distribution over the *entire vocabulary* for the next token, at every position. The model itself never "picks" a word — a separate **decoding algorithm** decides how to turn that probability distribution into an actual chosen token, and this choice happens at every single generation step. Different decoding strategies trade off **quality, diversity, and determinism** very differently, even using the exact same underlying trained model and the exact same probability distribution.

### 📌 Added Explanation — "in simple terms" framing for Section 0

It's worth internalizing this separation clearly, because it's a common point of confusion: **the model and the decoding algorithm are two different things that get composed together.** The model's *only* job is to answer the question "given everything so far, how likely is each possible next word?" — it outputs one full probability distribution and its job is done. Everything downstream of that — "okay, given this distribution, which single word do we actually commit to?" — is a separate, swappable decision procedure. This is exactly why you can take one single trained model (with one fixed set of weights, producing one fixed distribution at each step) and get wildly different outputs — deterministic and terse, or varied and creative, or slow-but-optimal, or fast-but-greedy — purely by changing the decoding algorithm wrapped around it, with zero retraining. Every section below is really just "a different answer to the question: given a probability distribution, how do I pick a token?"

---

## 1. Greedy Decoding

### The rule
At every step, pick the single token with the highest probability:
```
x_t = argmax P(x_t | x_1, ..., x_{t-1})
```

### 📌 Added Explanation — unpacking the argmax notation itself
`argmax` means "the input that produces the maximum output," as opposed to `max` which would just give you the maximum *value* itself. So `x_t = argmax P(x_t | ...)` reads as: "look across every possible token in the vocabulary, compute how probable each one is given everything generated so far, and set `x_t` equal to *whichever specific token* achieved the highest probability" — not the probability value itself, but the token that earned it. This is a purely deterministic operation: given the exact same probability distribution, argmax always returns the exact same token, with no randomness involved — which is precisely why greedy decoding (unlike the sampling methods in Section 3) always produces the identical output every time you run it on the same input.

### Why it fails — the concrete numerical problem
Greedy decoding is **locally optimal at each step but not globally optimal for the whole sequence** — picking the best next word right now can lock you into a worse overall sequence.

**Worked example**: suppose generating "The weather today is ___":
- Step 1 options: "sunny" (P=0.4), "cold" (P=0.35), "nice" (P=0.25) → greedy picks "sunny".
- Given "The weather today is sunny ___": "and" (P=0.9), "," (P=0.1) → greedy picks "and".
- Given "...sunny and ___": best continuation only reaches P=0.3 for "warm", because "sunny and warm" is a slightly awkward/redundant phrase, whereas if step 1 had picked "nice" instead:
- Given "...is nice ___": "and" (P=0.6), then "...nice and warm" (P=0.85) — a much more probable, fluent overall sequence.

Multiplying through: `P("sunny and warm") = 0.4 × 0.9 × 0.3 = 0.108`, while `P("nice and warm") = 0.25 × 0.6 × 0.85 = 0.1275` — the **globally better sequence was never found**, because greedy locked in "sunny" at step 1 purely because it had the highest single-step probability, without any lookahead. This exact failure mode — high locally-probable choices leading to a lower overall sequence probability — is the core argument for beam search.

### 🧮 Numerical Example — showing the arithmetic behind the two products explicitly

To make sure the multiplication itself is transparent (it's easy to skim past): sequence probability under an autoregressive model is just the **product of each token's conditional probability given everything before it** (this is the chain rule of probability, the same one underlying the CLM objective from Module 2: `P(x_1,...,x_n) = Π_t P(x_t | x_1,...,x_{t-1})`). So:

```
P("sunny and warm") = P(sunny) × P(and | sunny) × P(warm | sunny, and)
                     = 0.4 × 0.9 × 0.3
                     = 0.108

P("nice and warm")  = P(nice) × P(and | nice) × P(warm | nice, and)
                     = 0.25 × 0.6 × 0.85
                     = 0.1275
```
`0.1275 > 0.108`, confirming "nice and warm" is the genuinely higher-probability full sequence (about 18% higher), even though "sunny" alone beat "nice" alone at step 1 (0.4 vs 0.25). **In simple terms**: greedy decoding is a strategy that only ever asks "what's best *right now*?" — it never asks "what's best *overall*?", and this example is a minimal, concrete demonstration that those two questions can have different answers.

**Additional practical failure**: greedy decoding is also notorious for producing **repetitive loops** ("I think that I think that I think that...") on open-ended generation, because once a repetitive pattern starts, continuing the repetition is often literally the highest-probability next token (the model has strong evidence for what comes next when it's already seen that exact pattern before in-context).

### 📌 Added Explanation — why repetition becomes *self-reinforcing*, mechanistically

This is worth being able to explain beyond just naming the phenomenon. Once a phrase like "I think that" has appeared once earlier in the generated text, the model's attention mechanism can directly attend back to that earlier occurrence (this is related to the induction-head mechanism from Module 4's ICL discussion) and treat it as strong evidence that the same continuation pattern should repeat — from the model's perspective, "I've already seen 'I think that' followed by 'I think that' once in this exact context" is genuine statistical evidence, since real text sometimes *does* legitimately repeat phrases for emphasis. Under greedy decoding specifically, there is no mechanism to ever break out of this loop once started — each repeated instance only reinforces the pattern further (each new instance becomes another piece of "evidence" supporting yet another repeat), and because greedy always takes the deterministic highest-probability token, there is zero chance of a random deviation ever escaping the loop. This is precisely why sampling-based methods (Section 3), which inject randomness, are much less prone to getting permanently stuck in this exact failure mode.

---

## 2. Beam Search

### The core idea
Instead of keeping only 1 candidate sequence (greedy) or trying literally all possible sequences (computationally impossible — grows exponentially with vocab_size^sequence_length), keep the **top-k partial sequences** (k = "beam width") at every step, expand each by all possible next tokens, then prune back down to the top-k highest-*cumulative*-probability sequences.

### 📌 Added Explanation — why "try all possible sequences" really is computationally impossible, with a number attached

The notes assert exponential blowup; here's the concrete scale of it. A realistic vocabulary size might be `V ≈ 50,000` tokens, and a modest generated sequence length of `N = 20` tokens. The total number of possible sequences is `V^N = 50,000^20` — a number with roughly 94 digits (`log10(50,000^20) = 20 × log10(50,000) ≈ 20 × 4.7 = 94`). For comparison, the number of atoms in the observable universe is estimated around `10^80` — so even this modest 20-token sequence has *more* possible completions than there are atoms in the universe, for just `V=50,000` and `N=20`. This is why beam search's approach — track only a small constant number `k` of partial sequences at every step, rather than the combinatorial explosion of all of them — isn't just an optimization, it's the only computationally conceivable approach.

### Worked numerical example (beam width k=2)
Vocabulary simplified to {A, B, C} for illustration. Step 1 probabilities: P(A)=0.5, P(B)=0.3, P(C)=0.2 → keep top-2 beams: ["A"] (0.5) and ["B"] (0.3).

**Step 2** — expand each beam by all next-token options, compute cumulative probability (product):
- From "A": P(A→X)=0.6 → cum=0.30; P(A→Y)=0.4 → cum=0.20
- From "B": P(B→X)=0.9 → cum=0.27; P(B→Y)=0.1 → cum=0.03

All 4 candidates ranked by cumulative probability: "A→X" (0.30), "B→X" (0.27), "A→Y" (0.20), "B→Y" (0.03). Keep top-2: **["A","X"] (0.30) and ["B","X"] (0.27)**.

Notice: "B→X" (cumulative 0.27) survived into the beam even though "B" alone (0.3) was *less* probable at step 1 than "A" (0.5) — this is exactly the lookahead benefit beam search provides over greedy: it never permanently commits to only the single best step-1 choice, keeping a second candidate alive in case it leads to a better overall sequence, exactly like the "nice and warm" vs "sunny and warm" scenario in Section 1.

### 🧮 Numerical Example — extending the worked example one more step, to see a beam actually get pruned away

Continuing the same toy example, suppose **Step 3** expansions of the two surviving beams give:
- From ["A","X"] (cum=0.30): P(X→P)=0.5 → new cum = 0.30×0.5 = 0.150; P(X→Q)=0.5 → new cum = 0.30×0.5 = 0.150
- From ["B","X"] (cum=0.27): P(X→P)=0.2 → new cum = 0.27×0.2 = 0.054; P(X→Q)=0.8 → new cum = 0.27×0.8 = 0.216

All 4 candidates now: ["A","X","P"] (0.150), ["A","X","Q"] (0.150), ["B","X","P"] (0.054), ["B","X","Q"] (0.216). Ranked: ["B","X","Q"] (0.216) > ["A","X","P"] (0.150) = ["A","X","Q"] (0.150) > ["B","X","P"] (0.054). Keeping the top-2 (beam width k=2): **["B","X","Q"] (0.216) and either "A","X","P" or "A","X","Q" (tied at 0.150)** — note that the branch starting with "A" now survives in the beam *only through one* of its two children, and the low-scoring "B","X","P" branch (0.054) is finally pruned away entirely, even though "B" was kept alive through steps 1 and 2. **In simple terms**: beam search keeps *hedging its bets* across multiple candidate continuations for as long as they remain competitive, and only discards a candidate once its cumulative probability has clearly fallen behind the pack — this is the entire mechanism that gives it more robustness than greedy's single, irreversible commitment at every step.

### The known downside — beam search is not free of problems either
Beam search tends to produce **generic, "safe," repetitive text** for open-ended generation (it systematically favors high-probability-but-bland sequences, since it's explicitly hunting for maximum cumulative probability) — this is why beam search is common for tasks with a fairly narrow "correct" answer space (machine translation, summarization) but is **rarely used for open-ended chat/creative generation**, where sampling-based methods (below) produce more natural, varied text.

### 📌 Added Explanation — why "maximum probability" and "most natural-sounding to a human" are not the same target

This is worth stating explicitly since it's a subtle but important point: beam search is very good at exactly the thing it was designed to do — find a high-cumulative-probability sequence under the model's own distribution. The issue is that **real human language is not itself maximally probable under the model's own distribution** — genuinely good writing includes surprising word choices, varied sentence structure, and legitimate departures from the single "safest" continuation at many points, precisely the kind of choices that lower a sequence's raw probability score even while making it read as more natural/interesting to a human. Because beam search explicitly optimizes for probability-maximization, it systematically gravitates toward exactly the bland, generic, "safest at every step" continuations that a genuinely well-written human sentence often deliberately avoids — which is why it does well on tasks where there really is one narrowly "correct" target sequence (like a specific accurate translation) but does poorly on open-ended creative or conversational generation, where "most probable" and "best" diverge.

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

### 📌 Added Explanation — deriving why dividing logits by T sharpens or flattens the distribution

It's worth being able to explain *why* this specific transformation (dividing by `T` before exponentiating) has the sharpening/flattening effect, rather than just citing the examples. The key insight is that exponentiation is extremely sensitive to the *differences* between logits, and dividing by `T` rescales those differences: if two logits differ by `Δz = z_i - z_j`, then after dividing by `T` their difference becomes `Δz/T`. When `T < 1`, dividing by a number less than 1 is equivalent to *multiplying*, which **amplifies** the gap between logits — so after exponentiating, the token that was already ahead pulls even further ahead in probability terms, sharpening the distribution. When `T > 1`, dividing shrinks the gap between logits, so after exponentiating, all tokens end up with more comparable probabilities, flattening the distribution. **In simple terms**: temperature doesn't change *which* token is most likely (the ranking of tokens by probability is preserved for any `T > 0`) — it only changes *how much more likely* the top token is relative to the rest, controlling the peakedness/flatness of the whole distribution.

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

### 🧮 Numerical Example — an extreme case, T→0 and T→∞, to confirm the limiting behavior stated in the Q&A section

**Very low temperature, T=0.01**: `z/T = [200, 100, 50]`. Here `exp(200)` so overwhelmingly dwarfs `exp(100)` and `exp(50)` (differing by 100+ in the exponent) that, to any reasonable numerical precision, `P ≈ [1.0, ~0, ~0]` — essentially indistinguishable from deterministically picking the top logit, i.e., greedy argmax. This confirms concretely (not just asserted) that T→0 converges to greedy decoding.

**Very high temperature, T=1000**: `z/T = [0.002, 0.001, 0.0005]` — these are all so close to 0 that `exp(x) ≈ 1 + x` for small x, giving all three exponentials very close to 1.0 and therefore all three probabilities very close to `1/3 ≈ 0.333` each — nearly uniform. This confirms concretely that T→∞ converges to uniform random sampling over the vocabulary, exactly as the Q&A answer at the end of the notes states.

### Top-k sampling
Restrict sampling to only the `k` highest-probability tokens (e.g., k=40), **renormalize their probabilities to sum to 1** among just that subset, then sample from that restricted, renormalized distribution. This prevents sampling from the "long tail" of very low-probability, likely-nonsensical tokens, while still allowing some randomness among the plausible top candidates.

### 📌 Added Explanation — why renormalization is a required step, not an optional nicety

If you simply *dropped* the tokens outside the top-k without renormalizing, the remaining probabilities would no longer sum to 1 (e.g., if the top-40 tokens' probabilities summed to only 0.85 before truncation, you'd be sampling from an invalid, sub-1 "distribution"). Renormalizing — dividing each remaining probability by the sum of just the kept probabilities — rescales them so they properly sum back to exactly 1, making them a valid probability distribution again over just the restricted candidate set. **In simple terms**: you're not just deleting bad options, you're redistributing the probability mass that *used to* belong to the deleted options proportionally among the surviving ones, so the relative odds among the top-k tokens stay the same as they were originally, just without the discarded tail diluting things.

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

### 🧮 Numerical Example — verifying the renormalized values, and checking they sum to 1

Double-checking the arithmetic given in the notes, since renormalization errors are an easy place to make mistakes:
```
0.5/0.93 = 0.5376... ≈ 0.538
0.2/0.93 = 0.2151... ≈ 0.215
0.15/0.93 = 0.1613... ≈ 0.161
0.08/0.93 = 0.0860... ≈ 0.086

Sum check: 0.538 + 0.215 + 0.161 + 0.086 = 1.000 ✓
```
Confirms the renormalized values form a valid probability distribution (sums to 1.0) restricted to just the 4-token nucleus, as claimed.

### Practical combination
Production systems typically apply **temperature, then top-k or top-p (or both), then sample** — these are complementary, stackable knobs, not mutually exclusive alternatives; a common real-world default is something like temperature=0.7-1.0 combined with top-p=0.9.

### 📌 Added Explanation — why the *order* of operations (temperature before truncation) matters

It's worth noting explicitly why the stated order — temperature first, then top-k/top-p truncation — is the standard pipeline rather than the reverse. Temperature reshapes the *whole* distribution's peakedness based on the raw logit gaps; applying it first ensures that the subsequent top-k/top-p truncation step operates on a distribution whose sharpness already reflects your desired randomness level, so the truncation threshold behaves consistently with your temperature setting. If you truncated first and applied temperature only to the surviving subset, the truncation decision (which tokens even get considered) would be made using the *un*-adjusted, original probabilities — potentially keeping or discarding a different set of candidate tokens than you'd get if temperature were allowed to reshape the field before deciding who makes the cut. In short: temperature decides "how peaked should my beliefs be," and truncation decides "which specific candidates do I even consider" — doing the peakedness adjustment first ensures the truncation decision reflects your actual intended confidence level.

---

## 4. KV Caching — the memory/compute mechanism that makes autoregressive generation fast

### The problem it solves
In self-attention, computing the output at position `t` requires the **Key (K) and Value (V) vectors of every previous position** (1 through t), attending over all of them. Naively, generating token t+1 would require **recomputing K and V for all positions 1 through t all over again** — even though positions 1 through t never change once they're generated. This is enormously wasteful: generating a sequence of length N naively would cost `O(N²)` redundant K/V computation (recomputing position 1's K/V N times total, position 2's K/V N-1 times, etc.).

### 📌 Added Explanation — why K and V for a given position never change, which is the entire reason caching is valid

This is worth justifying rather than just asserting, since it's the linchpin of the whole technique: the Key and Value vectors for position `i` are computed as `K_i = W_K · h_i` and `V_i = W_V · h_i`, where `h_i` is that position's hidden state and `W_K, W_V` are the (frozen, already-trained) projection matrices. Once token `i` has been generated and its hidden state `h_i` computed (at a given layer), nothing about `h_i` ever changes on subsequent generation steps — future tokens attend to it, but never modify it (self-attention only lets *later* positions attend to earlier ones in a causal/autoregressive model, never the reverse). Since `h_i` is fixed forever once computed, and `W_K, W_V` are also fixed (they're trained parameters, not changing at inference time), `K_i` and `V_i` are simply fixed numbers once computed — recomputing them again later would deterministically produce the exact identical values, making that recomputation pure wasted work. This fixed-once-computed property is precisely what licenses caching: you're not approximating anything or trading off any accuracy by caching — you're avoiding literally redundant, wasted arithmetic that would produce identical results anyway.

### The fix
**Cache the K and V vectors for every position, the first time they're computed, and simply append the new token's K/V to the cache at each new step** — never recompute K/V for past positions. Generating the next token only requires computing Q (query), K, V for the *single new position*, then attending that one new query against the full cached K/V history.

### Numerical illustration of the savings
Say hidden dimension `d=4096`, and you're generating a sequence of length N=1000, with L=32 transformer layers.

**Without KV caching**: to generate token 1000, you'd recompute K and V for all 1000 previous positions, across all 32 layers, from scratch — and you'd repeat this entire recomputation at *every single one* of the 1000 generation steps. Total K/V computation work across the whole generation scales roughly as `O(N² × L × d)` — quadratic in sequence length purely from this redundant recomputation.

**With KV caching**: at each step, you compute K/V for exactly 1 new position (across all L layers), giving total K/V computation work scaling as `O(N × L × d)` — **linear** in sequence length. For N=1000, this is roughly a 1000x reduction in K/V-computation work compared to the naive approach (the quadratic term collapses to linear) — this is precisely why every production LLM serving system implements KV caching; without it, long-sequence generation would be computationally infeasible at production latency/cost targets.

### 🧮 Numerical Example — computing the actual operation counts, not just the big-O order, to see where "1000x" comes from

To make the "roughly a 1000x reduction" claim concrete rather than just an order-of-growth statement, let's count total K/V-projection work-units (treating one position's K/V computation, across all layers, as "1 unit" for simplicity):

**Without caching**: at generation step `t` (generating token `t`), you recompute K/V for all `t` positions so far, i.e., `t` units of work. Summing over all steps from `t=1` to `t=N=1000`:
```
Total = 1 + 2 + 3 + ... + 1000 = N(N+1)/2 = 1000×1001/2 = 500,500 units
```

**With caching**: at each step `t`, you compute K/V for exactly 1 new position:
```
Total = 1 + 1 + 1 + ... + 1 (1000 times) = 1000 units
```

**Ratio**: `500,500 / 1,000 ≈ 500.5x` reduction for N=1000 with this precise unit-counting — the notes' "roughly 1000x" is the right *order of magnitude* (the exact ratio for the sum-of-integers formula is `(N+1)/2`, which for large N is indeed close to `N/2`, i.e., on the order of hundreds-to-thousands for N in the thousands range, growing larger for longer sequences) — worth knowing the exact `N(N+1)/2` derivation if an interviewer asks you to be precise rather than just quoting "roughly N times fewer."

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

### 🧮 Numerical Example — scaling this up to a realistic serving batch, and showing the MQA/GQA savings numerically

**Batch of 32 concurrent requests, same settings as above** (32 layers, d=4096, N=4096, fp16):
```
2 × 32 × 4096 × 4096 × 32 (batch) × 2 bytes = 68,719,476,736 bytes ≈ 68.7 GB
```
That's roughly 68.7 GB of *just KV cache*, for one batch of 32 sequences at 4096 tokens each — easily exceeding the memory of a single high-end GPU (e.g., an 80GB card would be almost entirely consumed by cache alone, leaving little room for the model weights themselves), which is exactly the "memory-bound" problem the notes flag.

**With Grouped-Query Attention, e.g. sharing K/V across a group of 8 query heads** (a common real-world GQA configuration — say the model has 32 query heads total, grouped into 4 groups of 8, so only 4 distinct K/V head-sets need to be cached instead of 32): if the hidden dimension `d=4096` were originally split across 32 query heads (128 dims/head) each with their own K/V, GQA with 4 KV-groups means the K/V cache only needs to store `4/32 = 1/8` as many distinct K/V vectors:
```
GQA cache ≈ 68.7 GB × (4/32) = 68.7 GB / 8 ≈ 8.6 GB
```
An **8x reduction** in this configuration (exact ratio depends on how many query heads share each KV group), turning an infeasible 68.7 GB requirement into a much more manageable ~8.6 GB — this is the concrete payoff behind "reduces cache memory at a small quality cost" that's worth having ready with real numbers in an interview.

---

## 5. Speculative Decoding

### The core idea
Autoregressive generation is fundamentally **sequential and latency-bound** — you must generate token t before you can even start computing token t+1, because token t+1's computation depends on t as input. Speculative decoding breaks this bottleneck using a **small, fast "draft" model** to guess several tokens ahead cheaply, then verifies (or rejects) those guesses using the large "target" model in a single parallel pass.

### 📌 Added Explanation — why "latency-bound" specifically, and why parallel verification sidesteps it

It's worth clarifying exactly *what kind* of bottleneck this is, since "sequential" alone doesn't fully explain why parallel verification helps. Modern GPUs are massively parallel — computing a matrix multiply for 5 positions at once is barely more expensive in wall-clock time than computing it for 1 position, because the GPU has far more parallel compute capacity than a single token's worth of matrix multiplication can saturate (single-token autoregressive generation is famously *memory-bandwidth*-bound, not compute-bound — the GPU spends most of its time waiting to load weights from memory rather than actually computing, so adding a few more positions to verify "in parallel" barely adds wall-clock cost). The bottleneck in ordinary autoregressive generation isn't that the *math* for computing multiple tokens is expensive — it's that you're *forced* to wait for token t's actual output before you even know what input to feed in for token t+1, so you can never batch multiple *sequential* generation steps together in the ordinary process. Speculative decoding's trick is to break that forced-waiting structure: the draft model guesses several tokens *without* waiting for target-model confirmation at each step, and then the target model checks all those guesses at once, in a single parallel pass — since parallel verification of several already-guessed positions doesn't have the same "must wait for the previous output" dependency that generation does.

### The mechanism, step by step
1. A small draft model (much cheaper/faster than the target model) autoregressively generates a short candidate sequence of, say, 4-5 tokens.
2. The large target model then processes **all of those candidate positions in a single forward pass** (not one at a time) — this is possible/cheap because verifying a *given* sequence of tokens in parallel is a much cheaper operation than *generating* that sequence token-by-token would have been, since parallel verification doesn't have the same sequential-dependency bottleneck.
3. Compare the target model's actual next-token distribution at each position against what the draft model guessed. **Accept** every draft token where the target model agrees was a good choice (using a rejection-sampling-style acceptance criterion so this remains mathematically equivalent to sampling directly from the target model — this is the key correctness property: speculative decoding provably produces exactly the same output distribution as the target model alone would have, just faster). At the first position where the target model rejects the draft's guess, **discard the rest of the draft sequence from that point on**, sample the correct token directly from the target model's own distribution at that position, and restart the drafting process from there.

### 📌 Added Explanation — a concrete sketch of the rejection-sampling acceptance rule itself

The notes describe the acceptance criterion at a high level ("rejection-sampling-style," "mathematically equivalent") — here's the actual mechanism, at the level of detail useful for an interview follow-up. Let `q(x)` be the draft model's probability for the guessed token `x` at some position, and `p(x)` be the target model's probability for that same token at that position:

- **Accept the draft token with probability `min(1, p(x)/q(x))`.** Intuitively: if the target model thinks this token is *at least as* likely as the draft model did (`p(x) ≥ q(x)`), accept it unconditionally (probability 1) — the draft's guess was, if anything, an *underestimate* of how good this token was, so there's no reason for the target model to object. If the target model thinks the token is *less* likely than the draft did (`p(x) < q(x)`), accept it only with probability `p(x)/q(x) < 1` — the draft may have overstated this token's quality, so it's only kept probabilistically, proportional to how much the target model's assessment agrees.
- **If rejected**, sample a replacement token not from `p(x)` directly, but from a specifically constructed "residual" distribution `max(0, p(x) - q(x))` (renormalized) — this residual distribution is exactly the leftover probability mass the target model assigns to outcomes the draft model *under*-weighted, ensuring that across the whole accept/reject procedure, the overall marginal probability of ending up with any given token exactly matches `p(x)`, the target model's own distribution, with no distortion introduced by having used the draft model at all.

This is exactly why the notes can say (accurately) that the output distribution is "provably identical" to sampling from the target model alone — the accept/reject/resample procedure is mathematically constructed so all the draft model's involvement washes out in expectation, leaving the target model's own distribution as the final result.

### Numerical intuition for the speedup
If the draft model's guesses are accepted, on average, for 3 out of every 4 speculated tokens, you effectively get ~4 tokens' worth of output for roughly the cost of **1 target-model forward pass** (parallel verification) **plus a few cheap draft-model forward passes** — instead of needing 4 full, strictly-sequential target-model forward passes. Since the large target model's forward pass is by far the dominant cost, and parallel verification of several positions costs roughly the same as verifying just one position (both are dominated by fixed per-call overhead plus one comparable-sized matrix multiply, since modern GPUs are heavily underutilized during memory-bound single-token generation), the **wall-clock speedup can be substantial** (commonly cited real-world figures are in the 2-3x range) **without any change to output quality/distribution** — this last point (exactness, not an approximation) is the detail interviewers most want to hear stated explicitly.

### 🧮 Numerical Example — a toy end-to-end throughput calculation

Suppose (illustrative numbers): one target-model forward pass costs `100 ms` regardless of whether it's verifying 1 or 5 positions at once (since it's memory-bandwidth-bound, as explained above), and one draft-model forward pass costs `10 ms`, with the draft proposing 4 tokens per round, of which on average 3 are accepted.

**Speculative decoding, per "round"**:
```
Draft model generates 4 tokens sequentially: 4 × 10 ms = 40 ms
Target model verifies all 4 in one parallel pass: 100 ms
Total time per round: 140 ms, yielding ~3 accepted tokens + 1 target-sampled replacement = 4 tokens
Effective time per token ≈ 140 ms / 4 = 35 ms/token
```

**Ordinary autoregressive generation with the target model alone**:
```
4 tokens, one full target forward pass each, strictly sequential: 4 × 100 ms = 400 ms
Effective time per token = 400 ms / 4 = 100 ms/token
```

**Speedup**: `100 ms / 35 ms ≈ 2.9x` — landing right in the "commonly cited 2-3x range" the notes mention, and showing concretely where that range comes from: it's driven by (a) how cheap the draft model is relative to the target model, and (b) how often the draft's guesses are actually accepted (a higher acceptance rate directly increases the number of "free" tokens obtained per expensive target-model pass).

---

## 6. Context Length Extension — RoPE and ALiBi (ties directly back to your Transformer architecture knowledge)

### The underlying problem
A Transformer trained on sequences up to length N (say, 4096) often performs poorly if you naively try to run it on longer sequences at inference (say, 16000) — positional encoding schemes need to **generalize/extrapolate** to positions never seen during training.

### 📌 Added Explanation — why "never seen during training" is specifically the problem, for learned/absolute schemes

It's worth spelling out exactly what goes wrong. If positional information is encoded as, say, a learned embedding vector indexed by absolute position (position 1 gets embedding vector #1, position 4097 would need embedding vector #4097), and the model was only ever trained with positions 1 through 4096, then embedding vector #4097 (and beyond) was **never updated by any gradient step during training** — it's either undefined, randomly initialized and untrained, or simply doesn't exist as an addressable index at all. Even schemes using fixed (non-learned) sinusoidal functions of absolute position can behave unpredictably past the trained range, because the *combinations* of positional and content information the attention mechanism learned to interpret were only ever exposed to position values within the training range — the network's weights (which were trained assuming positional signals stay within a certain numeric range/pattern) have no guarantee of behaving sensibly when fed inputs describing positions the training process never demonstrated. This is precisely the motivation for schemes like RoPE and ALiBi below, both of which are specifically designed around *relative*, not absolute, position information, since relative distances ("5 tokens back") are a pattern that recurs identically regardless of whether you're near the start or far into a long sequence — making them inherently more likely to generalize past the trained length.

### RoPE (Rotary Position Embedding) — the mechanism
Instead of adding a separate positional embedding vector to the token embedding (as in the original Transformer's sinusoidal encoding), RoPE **rotates** the Query and Key vectors by an angle that depends on their absolute position, using 2D rotation matrices applied to pairs of dimensions within the vector. The key mathematical property: the dot product between a rotated Query at position `m` and a rotated Key at position `n` ends up depending **only on the relative distance `(m - n)`**, not on the absolute positions `m` and `n` individually — this relative-position property is exactly what helps generalization, since "attend to something 5 tokens back" is a pattern the model can learn once and reuse regardless of whether that pattern occurs at absolute position 10 or position 10,000.

### 📌 Added Explanation — a minimal derivation sketch showing why the rotation makes the dot product relative-only

This is a genuinely elegant piece of math worth having a concrete feel for, even at a simplified (2D) level. Consider a single pair of dimensions, and represent a vector at that pair as a complex number (a standard trick, since 2D rotation is exactly what complex-number multiplication by `e^(iθ)` does). RoPE rotates the Query vector at position `m` by angle `mθ` and the Key vector at position `n` by angle `nθ` (`θ` is some fixed frequency parameter):
```
Rotated Query at m:  q'_m = q · e^(i·mθ)
Rotated Key at n:     k'_n = k · e^(i·nθ)
```
The relevant "dot product" in the complex-number analogy is `q'_m` multiplied by the *complex conjugate* of `k'_n` (this is how the real 2D dot product translates into complex-number form):
```
q'_m · conj(k'_n) = q · e^(i·mθ) · conj(k · e^(i·nθ))
                   = q · e^(i·mθ) · k · e^(-i·nθ)        [conjugate of e^(iφ) is e^(-iφ)]
                   = q·k · e^(i·(mθ - nθ))
                   = q·k · e^(i·(m-n)θ)
```
Notice: the absolute positions `m` and `n` no longer appear separately anywhere in the final expression — only their **difference** `(m-n)` does, multiplied by the fixed frequency `θ`. **In simple terms**: no matter what specific absolute positions `m` and `n` are, as long as the *distance between them* (`m-n`) stays the same, the resulting attention-relevant quantity is identical — which is exactly the claimed relative-position property, and this derivation shows concretely (not just asserted) where it comes from: it falls directly out of the algebra of how rotation angles combine when you take a rotated-query-vs-rotated-key dot product.

**RoPE scaling for length extension**: since RoPE's rotation angle is a function of position, at inference time you can apply a **scaling factor** to effectively "compress" position indices (e.g., treat position 16000 as if it were position 4000 by dividing all position indices by 4) so the rotation angles stay within the range the model saw during training — this is the core trick behind techniques like "Position Interpolation" used to extend context windows of RoPE-based models (Llama and most modern LLMs use RoPE) without retraining from scratch.

### 🧮 Numerical Example — walking through Position Interpolation's scaling concretely

Suppose a model was trained with a maximum context length of `N_train = 4096`, and you want to serve it at `N_target = 16384` (a 4x extension). Position Interpolation rescales every position index `p` (from 0 up to 15383) by the ratio `N_train / N_target = 4096/16384 = 0.25` before computing the RoPE rotation angle:
```
Effective position used in rotation = p × (N_train / N_target) = p × 0.25
```
So, for example, an actual token at absolute position `p = 16000` gets treated, for rotation-angle purposes, as if it were at position `16000 × 0.25 = 4000` — safely within the `[0, 4096)` range the model was originally trained on. **Why this works reasonably well (though not perfectly)**: because RoPE's relative-position property means what actually matters to attention is the *difference* between two scaled positions, and after uniform scaling, two positions that were originally, say, 8000 apart are now effectively only 2000 apart — the model still sees "in-range" relative distances, just at a different *density* (each unit of scaled-position now corresponds to 4 real tokens' worth of separation) — this compression is why Position Interpolation can extend context noticeably without retraining, though (worth flagging) it typically still benefits from at least a short amount of fine-tuning at the new target length to fully adapt to the altered relative-distance density, rather than being a perfectly free lunch with zero retraining at all.

### 🔎 Accuracy Flag
The original notes describe Position Interpolation as extending context "without retraining from scratch" — this is accurate (it avoids *full* pretraining-scale retraining), but published results on Position Interpolation generally report best results with some amount of brief additional fine-tuning at the new, longer context length, rather than being usable with literally zero additional training. Worth mentioning this nuance if asked for precise practical detail.

### ALiBi (Attention with Linear Biases) — the mechanism
A different, arguably simpler approach: don't modify Query/Key vectors at all — instead, directly **subtract a penalty from the raw attention scores**, proportional to the distance between the query and key positions:
```
attention_score(i,j) = (Q_i · K_j) - m × |i - j|
```
where `m` is a fixed, head-specific slope (different attention heads get different, geometrically-spaced slope values `m`, so some heads focus more locally and others more globally). **In plain words**: tokens further away automatically get their raw attention score penalized more, with no learned positional embedding parameters at all — just a fixed, hard-coded linear penalty based on distance. ALiBi's authors demonstrated notably strong length-extrapolation performance (training on short sequences, evaluating well on much longer ones) specifically because this penalty structure naturally, smoothly discourages attending far away without ever needing position representations that go "out of distribution" the way learned/sinusoidal embeddings can at unseen lengths.

### 📌 Added Explanation — explaining every symbol in the ALiBi formula, and why "geometrically-spaced slopes" specifically

- **`Q_i · K_j`**: the ordinary, unmodified content-based attention score between query position `i` and key position `j` — exactly the same dot product you'd compute in vanilla attention, with no positional modification to the vectors themselves at all.
- **`|i - j|`**: the absolute distance between the two positions — always non-negative, growing the further apart the two tokens are, regardless of which one comes first (though in a causal/autoregressive model, `j ≤ i` always, since you can't attend to future positions).
- **`m` (the slope)**: a fixed, non-learned constant that determines how harshly distance is penalized for a *given attention head*. A larger `m` means even modest distances rack up a large penalty (that head is forced to focus locally); a smaller `m` means the penalty grows very slowly with distance (that head can attend broadly across long ranges).
- **Why *different, geometrically-spaced* slopes across heads** (e.g., a common scheme uses slopes like `1/2, 1/4, 1/8, ..., 1/2^h` for `h` heads): this gives the model, essentially for free, a whole *spectrum* of attention behaviors in a single layer — some heads are hard-wired to be strongly local (large `m`, sharply penalizing anything beyond a few tokens), others are hard-wired to be much more global (tiny `m`, barely penalizing distant tokens at all) — without needing to *learn* this division of labor at all, since it's baked into the fixed slope values from the start. **In simple terms**: ALiBi doesn't ask the model to figure out "how far should I look" — it hard-codes a range of different look-distances across different heads directly into the architecture, and lets training decide how to make use of that fixed menu of options.

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

#### 📌 Added Explanation — fuller answer with reasoning
The underlying reason is that sequence probability is a **product** of conditional probabilities (chain rule), and a product's overall value depends jointly on *every* factor, not just the first one — a slightly smaller first factor can easily be more than compensated for by much larger later factors, as the "0.108 vs 0.1275" arithmetic in Section 1 demonstrates concretely. Greedy decoding, however, makes its choice at step 1 using *only* the step-1 probabilities, with zero visibility into what later factors might look like under each alternative — it has no mechanism to "look ahead" and check whether a currently-second-best option might unlock much better subsequent factors. Once it commits to the argmax choice at step 1, that choice becomes part of the fixed context for all subsequent steps, and there is no backtracking step in the algorithm at all — the decision, once made, can never be revisited even if a later step reveals it led somewhere worse than an alternative would have.

**Q: In beam search with width k=2, why might a beam that wasn't the single best option at step 1 still survive into the final beam set?**
A: Because beam search ranks by *cumulative* sequence probability across all surviving beams at each step, not by the step-1 probability alone — a lower step-1-probability beam can still be retained if its subsequent expansions give it a higher running cumulative probability than the top beam's worse expansions.

#### 📌 Added Explanation — fuller answer with reasoning
The mechanism that makes this possible is that beam search re-ranks *all* candidates (across *all* currently-tracked beams) freshly at every single step, based on their cumulative product-of-probabilities so far — it never locks in a "final ranking" of beams until generation is fully complete. In the worked example, "B" (0.3) was behind "A" (0.5) after step 1, but "B→X" (cumulative 0.27) outscored "A→Y" (cumulative 0.20) after step 2's expansion, because "B→X" happened to have a very high (0.9) step-2 conditional probability while "A→Y" only had 0.4. Since beam search's pruning decision at each step considers the *product* built up so far, not just the most recent step's probability or the very first step's probability, a beam that started slightly behind can and does overtake another beam that started ahead, exactly as demonstrated numerically — this cross-step re-ranking, repeated at every step, is the entire mechanism providing beam search's lookahead advantage over greedy's single irreversible step-1 decision.

**Q: Write the temperature-scaled softmax formula and explain what T→0 and T→∞ each converge to.**
A: `P(x_i) = exp(z_i/T) / Σ_j exp(z_j/T)`. As T→0, the distribution becomes maximally peaked and sampling converges to greedy argmax; as T→∞, all logits are scaled toward 0 and the distribution converges to uniform random sampling over the vocabulary.

#### 📌 Added Explanation — fuller answer with reasoning
As `T→0`, dividing any finite logit by an ever-smaller `T` sends the differences between logits `(z_i - z_j)/T` toward `±∞` — meaning whichever token had even a marginally higher raw logit gets its exponential term growing arbitrarily larger relative to every other token's exponential term, so its softmax probability approaches exactly 1 while every other token's approaches exactly 0. This is, by definition, indistinguishable from picking `argmax(z)` deterministically — hence T→0 recovers greedy decoding exactly, not just approximately. As `T→∞`, dividing every logit by an ever-larger `T` sends every scaled logit `z_i/T` toward 0 (regardless of the original gaps between them, since any finite difference divided by an infinitely large number vanishes) — and `exp(0) = 1` for every token equally, so the softmax reduces to `1/V` for every one of the `V` vocabulary tokens: a perfectly uniform distribution, matching the claim that extremely high temperature converges to uniform random sampling with no preference for any token based on the model's actual predictions at all.

**Q: What's the core weakness of top-k sampling that top-p (nucleus) sampling fixes?**
A: Top-k uses a fixed token count regardless of context, which can be too permissive when the model is very confident (few tokens are reasonable) or too restrictive when the model is very uncertain (many tokens are reasonable). Top-p dynamically sizes the candidate set based on cumulative probability mass, adapting automatically to the model's confidence at each step.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning centers on what a fixed *count* fails to capture versus what cumulative *probability mass* captures instead. A fixed `k` (say 40) treats "40 tokens" as equally meaningful regardless of how probability happens to be distributed among them at any given step — but the *actual* shape of the distribution varies enormously step to step: sometimes the top handful of tokens already account for 99% of the probability mass (a very confident prediction, where tokens ranked 4 through 40 are essentially noise/garbage that shouldn't be sampled at all), and sometimes probability is spread thinly across hundreds of plausible tokens (a genuinely uncertain prediction, where even the top 40 might only cover 60% of reasonable mass, cutting off many legitimately plausible options). Top-p directly targets the quantity that actually matters — "how much of the total probability mass have I covered" — rather than an arbitrary proxy ("how many tokens have I covered"), so it automatically includes exactly as many tokens as needed to hit the target coverage in any given context, shrinking to just 1-2 tokens when the model is very sure and expanding to dozens or more when the model is genuinely unsure, without needing any context-specific tuning of a count parameter.

**Q: Explain, with the complexity classes, why KV caching is necessary for practical autoregressive generation.**
A: Without caching, generating a sequence of length N requires recomputing K/V for all previous positions at every step, giving O(N²) redundant K/V computation work. Caching K/V once computed and only computing K/V for the new position at each step reduces this to O(N) — linear instead of quadratic in sequence length.

#### 📌 Added Explanation — fuller answer with reasoning
The `O(N²)` figure comes directly from summing the redundant work across all steps: at step `t`, you'd redundantly recompute K/V for all `t` previous positions, and summing `t` from 1 to `N` gives `Σt = N(N+1)/2`, which is `O(N²)` — quadratic growth, meaning that doubling the sequence length roughly quadruples the total redundant K/V computation work (not just doubles it). Caching eliminates essentially all of this redundancy by computing each position's K/V exactly once, ever, reducing the sum to simply `N` total units of K/V computation across the whole generation — `O(N)`, linear growth, where doubling sequence length only doubles the work, not quadruples it. This distinction — quadratic vs. linear scaling — becomes the dominant practical bottleneck specifically as sequence lengths grow large (a few hundred tokens might be tolerable either way, but for sequences in the thousands or tens-of-thousands of tokens that modern long-context models support, the quadratic-vs-linear gap becomes the difference between "feasible in production" and "computationally infeasible"), which is exactly why KV caching isn't an optional optimization but a load-bearing requirement for any practical autoregressive serving system.

**Q: Why does the KV cache's memory cost become a serving bottleneck, and what's one architectural fix?**
A: KV cache memory grows linearly with sequence length, layers, hidden dimension, and batch size — for long contexts and large batches this can reach many GB per sequence, becoming memory-bound rather than compute-bound. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce this by sharing K/V projections across multiple query heads, shrinking the cache size at a small quality cost.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning for why this specifically becomes a *memory*-bound (rather than compute-bound) bottleneck is that the KV cache must be held, in full, in GPU memory for the *entire duration* a sequence is being generated (every future generation step needs to attend back over the complete cached history) — and this memory requirement scales multiplicatively across four independent factors (layers × sequence length × hidden dimension × batch size), each of which is already large in modern production settings (dozens of layers, thousands of tokens of context, thousands of hidden dimensions, and potentially dozens-to-hundreds of concurrent requests batched together for throughput) — so even though computing K/V itself is comparatively cheap (a modest matrix multiply per new token, as established in the KV-caching discussion above), simply *storing* all of the previously-computed K/V vectors for every concurrent sequence can outstrip available GPU memory long before the GPU's raw compute capacity becomes the limiting factor. MQA/GQA address this at its root cause: since the memory cost scales with the *number of distinct K/V head-sets stored*, reducing the number of independent K/V projections (by having multiple query heads share the same underlying K/V projection, rather than each query head requiring its own separate K/V) directly and proportionally shrinks the memory footprint — at the cost of some representational flexibility (different query heads can no longer attend using entirely independent key/value spaces), which is the "small quality cost" the notes reference.

**Q: Why is speculative decoding described as "exact" rather than an approximation, despite using a smaller, less accurate draft model?**
A: It uses a rejection-sampling-style acceptance criterion when comparing draft-model guesses against the target model's actual distribution, which is mathematically constructed so the final accepted output distribution is provably identical to what the target model alone would have produced by itself — the draft model only affects speed, never correctness/quality.

#### 📌 Added Explanation — fuller answer with reasoning
The precise mechanism (spelled out fully in Section 5's added explanation above) is that a draft token is accepted with probability `min(1, p(x)/q(x))` (target probability over draft probability), and whenever rejected, the replacement token is drawn not from the target's raw distribution but from a specifically constructed residual distribution `max(0, p(x)-q(x))` (renormalized) representing exactly the probability mass the draft model under-weighted. This specific combination of accept-probability and residual-resampling-on-rejection is a classical rejection-sampling construction, and its defining mathematical property is that the *overall*, marginal probability of ending up with any given token `x` — averaging over both the "accepted from draft" and "rejected then resampled from residual" cases — works out to exactly `p(x)`, the target model's own probability, regardless of what the draft model's `q(x)` happened to be. This is why the draft model's quality only ever affects the *acceptance rate* (a worse draft model gets rejected more often, meaning less speedup, since more tokens have to be resampled directly from the target model) but never affects *which distribution* the final output is drawn from — the correctness guarantee holds even for an arbitrarily bad draft model, it would just provide little-to-no speedup in that case.

**Q: What's the key mathematical property RoPE achieves that a naive absolute positional embedding does not?**
A: RoPE's rotation makes the Query-Key dot product depend only on the *relative* distance between two positions, not their absolute positions — this relative-position property helps the learned attention patterns generalize to sequence lengths not seen during training, unlike absolute positional schemes which can go fully out-of-distribution beyond the trained length.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning connects directly to the derivation given in Section 6: because the rotated-Query/rotated-Key dot product algebraically simplifies to an expression involving only `(m-n)θ` (the position *difference* times a fixed frequency), two pairs of positions with the same relative distance — say, positions (10, 15) and positions (10,000, 10,005), both 5 apart — produce the *exact same* dot-product-modifying term, regardless of how far into the sequence they occur. This means any attention pattern the model learns during training about "how to weigh tokens 5 positions back" is expressed purely as a function of that distance-5 relationship, and is automatically valid at *any* absolute position, including positions never encountered during training — because the mathematical object the model actually learned to respond to (the relative-distance-dependent rotation term) simply doesn't have an absolute-position component that could go "out of range." Absolute schemes (a learned embedding indexed by position number, or even fixed sinusoidal functions evaluated directly at large position values) don't have this guarantee: whatever pattern the network learned to associate with, say, "position 4000's embedding vector" has no defined relationship to what "position 16,000's embedding vector" should look like, since the network never observed or learned anything about that specific input during training — there's no algebraic guarantee, only whatever accidental interpolation/extrapolation behavior the network's specific learned weights happen to exhibit.

**Q: How does ALiBi handle position information differently from RoPE?**
A: ALiBi leaves Query/Key vectors completely unmodified and instead directly subtracts a fixed, head-specific, distance-proportional penalty from the raw attention scores after the Q·K dot product — no learned or rotated positional representations are involved at all, just a hard-coded linear bias based on token distance.

#### 📌 Added Explanation — fuller answer with reasoning
The key structural difference, reasoned through: RoPE intervenes *before* the dot product is computed — it transforms the Q and K vectors themselves (via rotation) so that when the ordinary dot product is later taken, position information is already baked into the numbers being multiplied together. ALiBi instead leaves the dot product computation completely untouched (Q and K are exactly the same content-only vectors they'd be with zero positional information at all) and intervenes *after* the dot product, by simply subtracting a distance-dependent penalty term directly from the resulting score. This "after, not before" distinction has a practical consequence explaining ALiBi's strong extrapolation behavior: because the penalty `m×|i-j|` is a simple, fixed linear function that can be evaluated for *any* distance value at all (there's no learned parameter indexed by position that could be "out of range" — it's literally just an arithmetic multiplication that remains perfectly well-defined for a distance of 16,000 exactly as it is for a distance of 16), there is no possible sense in which ALiBi's positional mechanism could encounter an input it wasn't designed to handle — unlike even RoPE, whose rotation *angles* are still, in the unmodified/unscaled case, functions of specific position values that could in principle grow into numerically less-familiar ranges at extreme lengths (which is precisely why techniques like Position Interpolation/RoPE-scaling, discussed above, are needed for RoPE specifically, whereas ALiBi's linear penalty needs no such rescaling trick to remain well-defined at longer lengths).

---

## ❓ Interview Q&A — Apple / Google ML Engineer style questions

*(These go beyond the "quick-fire" self-test above — phrased the way an interviewer would actually ask them live, often layering a follow-up on top of a definition to test whether you understand the mechanism, not just the vocabulary.)*

**Q1. "You're serving a customer support chatbot. Would you use greedy decoding, beam search, or sampling? Walk me through your reasoning, not just your final pick."**
A: I'd lean toward sampling (temperature + top-p) over the other two, but the reasoning matters more than the pick. Greedy is cheap and deterministic, which is attractive for reproducibility, but its tendency toward repetition loops on longer, open-ended responses is a real risk in a conversational setting where responses aren't always short and templated. Beam search would actually be a poor fit here specifically because customer support chat is exactly the kind of open-ended generation task where beam search's bias toward generic, "safe" text becomes a liability rather than a virtue — it shines in narrow-answer-space tasks like translation, not open conversation. Sampling (with a moderate temperature like 0.7 and top-p around 0.9) gives natural-sounding, varied phrasing while still avoiding the genuinely nonsensical long tail, which is the right tradeoff for conversational quality — though I'd want low temperature or even greedy for any sub-task requiring strict determinism/reproducibility, like generating a structured field (e.g., a ticket category label) rather than free-form conversational text — so in practice I'd likely mix strategies by sub-task rather than pick one globally.

**Q2. "Derive why beam search's memory and compute cost scales linearly, not exponentially, with beam width k — some people assume it's exponential since it's exploring a tree of possibilities."**
A: The key is that beam search **prunes back down to exactly k beams at every single step**, regardless of how many candidates were generated during expansion — it never lets the tracked set grow beyond k. At each step, you expand each of the k current beams by all V vocabulary options, producing `k × V` candidates, then immediately sort and keep only the top k of those `k × V` candidates before moving to the next step. So the *tracked* state size is constant at k beams per step (linear in k, since work scales as roughly `k × V` per step, i.e., proportional to k), not exponential — the naive "explore everything" alternative would be exponential (`V^N` total sequences) precisely because it never prunes anything away; beam search's defining design choice is exactly this per-step pruning back to a fixed-size set, which is what converts what looks like tree exploration into a bounded, linear-in-k cost per step.

**Q3. "If speculative decoding is 'exact' (same output distribution), why doesn't everyone just always use temperature=high draft models to maximize guessing speed, since correctness is guaranteed regardless?"**
A: Because "exact in distribution" and "fast" are two separate properties, and the draft model's quality only affects the *speedup*, not correctness — but that speedup is exactly what you're trying to gain by using this technique in the first place. If the draft model's guesses are frequently wrong (low `q(x)` relative to the target's `p(x)` for the tokens it actually proposes), the acceptance probability `min(1, p(x)/q(x))` will be low on average, meaning most draft tokens get rejected and you fall back to sampling directly from the target model position-by-position anyway — at that point you've paid the extra cost of running the draft model repeatedly for close to zero speedup benefit, potentially making the overall pipeline *slower* than just running the target model alone with no draft model at all. So in practice, you want a draft model whose distribution is a genuinely good, cheap approximation of the target's distribution (high agreement, hence high acceptance rate) — not just "any fast model," since a fast-but-inaccurate draft model still guarantees exact correctness, but delivers little to none of the speed benefit the whole technique exists to provide.

**Q4. "A colleague suggests extending context length simply by fine-tuning at the new longer length directly, with ordinary (non-scaled) RoPE — no Position Interpolation. What tradeoff would you point out?"**
A: This is a real, valid alternative approach (sometimes called direct extrapolation or extended fine-tuning), but the tradeoff is compute cost and data requirements versus Position Interpolation's cheaper "compress and lightly fine-tune" approach. Extending purely via direct fine-tuning at the new length requires the model to actually learn to handle rotation angles it never saw during original pretraining, at whatever new position range you're training on — this generally demands a non-trivial amount of long-sequence training data and compute (attention computation itself scales roughly quadratically with sequence length within each training step, quite apart from the KV-cache discussion above, so training on much longer sequences is intrinsically more expensive per step). Position Interpolation instead rescales positions so the model only ever has to interpret rotation angles within the *original*, already-learned range, just at a different effective "density" — which is why it typically needs only a comparatively brief amount of additional fine-tuning (as flagged in the 🔎 Accuracy Flag above) rather than the more extensive training a from-scratch-style extension at the new length would likely require. The tradeoff to state clearly: Position Interpolation trades a small amount of peak fidelity (the compressed relative-distance density is an approximation, not identical to genuinely training at native resolution) for a substantially cheaper adaptation cost; direct extended fine-tuning can potentially reach higher fidelity at the new length but at meaningfully higher compute/data cost.

**Q5. "Explain what would happen, mechanistically, if you used KV caching but forgot to also cache/reuse the causal attention mask correctly across generation steps — is this purely a caching bug, or could it silently produce wrong (not just slow) output?"**
A: This would be a correctness bug, not just a performance one. KV caching's entire correctness guarantee rests on the fact that each cached K/V vector is exactly what would have been computed if you'd recomputed it from scratch (as established in the "why K/V never change" explanation above) — the technique is only valid because nothing else about the computation changes. The causal mask's job is to ensure that, at every step, the new query position only attends to the current and *past* cached positions and never to any (nonexistent, future) position — if the mask were mishandled (e.g., built assuming the wrong current sequence length, or omitted for the newly appended position), you could end up either attending to positions that shouldn't be visible yet, or failing to attend to positions that should be, producing an output that's simply computing a *different* function than the one the model was trained to compute — a genuine correctness error, not merely a slowdown, and one that could be especially insidious because the generation would still run without crashing, just quietly produce different (and likely lower-quality or inconsistent) results.

**Q6. "Compare ALiBi's fixed slopes to a hypothetical version where the per-head slope `m` is instead a learned parameter. What would you expect to gain or lose?"**
A: Making `m` learned would, in principle, let the model discover the *optimal* division of local-vs-global attention per head from data, rather than committing upfront to a fixed geometric schedule of slopes — potentially capturing a better-tuned allocation of "how far should this specific head look" for the specific task/data distribution at hand, which could plausibly improve peak in-distribution performance. The likely cost, reasoning from the rest of this module's themes, would be exactly the same kind of out-of-distribution fragility that motivated ALiBi's fixed-slope design in the first place: a *learned* slope value is only ever trained/validated using the distance ranges seen during training, so at inference time on much longer sequences than were seen in training, a learned `m` has no guarantee of continuing to behave sensibly at those larger, never-observed distance values — precisely the same generalization risk that fixed, non-learned schemes (Section 6's whole discussion) are specifically designed to avoid. So the tradeoff would likely mirror the classic "flexibility now vs. robustness later" pattern seen throughout this module (e.g., LoRA rank size, PPO's β, RoPE vs. ALiBi themselves): a learned slope might do somewhat better within the trained length range, but ALiBi's whole empirical claim to fame is specifically its length-*extrapolation* robustness, which a learned-and-therefore-potentially-out-of-distribution slope parameter would be reintroducing exactly the failure mode ALiBi's fixed-slope design was built to sidestep.

---
*End of Module 6 (maximum depth). Next: Module 7 — Efficiency & Serving (quantization, mixed precision, MoE routing, distillation).*
