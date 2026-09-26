Here are all the major parameters that shape LLM output at inference time, with the math behind each.

## 1. Temperature (T)

**What it does:** Rescales the logits before softmax, controlling randomness.

**Formula:**
$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $z_i$ are the raw logits.

**Why/how:**
- T = 1: standard softmax, no change
- T → 0: distribution sharpens toward argmax (nearly deterministic, greedy-like)
- T > 1: distribution flattens, more uniform, more randomness/creativity
- T → ∞: uniform random sampling over vocabulary

**Example:** Logits = [2.0, 1.0, 0.5] for tokens ["cat", "dog", "fish"]
- T=1: softmax → [0.51, 0.19, 0.11] (normalized-ish)
- T=0.5: divide logits by 0.5 first → [4.0, 2.0, 1.0] → softmax sharpens, "cat" dominates even more
- T=2: divide by 2 → [1.0, 0.5, 0.25] → softmax flattens, probabilities become closer

**Use case:** Low T (0.0–0.3) for factual/code tasks; higher T (0.7–1.0) for creative writing.

## 2. Top-k Sampling

**What it does:** Truncates the distribution to only the k highest-probability tokens, renormalizes, then samples.

**Formula:** Given sorted logits $z_{(1)} \geq z_{(2)} \geq ... $, keep only $\{z_{(1)}, ..., z_{(k)}\}$, set all others to $-\infty$, then apply softmax.

**Why:** Prevents sampling from the long, noisy tail of low-probability garbage tokens that pure temperature sampling might occasionally pick.

**Example:** k=3 with vocab probabilities [0.4, 0.3, 0.2, 0.05, 0.05] → keep only first three [0.4, 0.3, 0.2], renormalize to [0.444, 0.333, 0.222], sample from this.

**Downside:** Fixed k is inflexible — sometimes the model is very confident (should consider fewer tokens) and sometimes very uncertain (should consider more).

## 3. Top-p / Nucleus Sampling

**What it does:** Instead of a fixed count, keeps the smallest set of tokens whose cumulative probability ≥ p.

**Formula:** Find smallest set $V_p$ such that:
$$\sum_{x \in V_p} P(x) \geq p$$
Renormalize probabilities within $V_p$, sample from that.

**Why:** Adapts to the shape of the distribution — narrow set when the model is confident, wider set when it's uncertain. Generally preferred over top-k today.

**Example:** Probabilities sorted [0.5, 0.2, 0.15, 0.1, 0.05], p=0.7 → cumulative sum hits 0.7 after first two tokens (0.5+0.2) → nucleus = {token1, token2}, renormalize to [0.714, 0.286].

**Typical value:** p = 0.9 to 0.95 is common default.

## 4. Combining Temperature + Top-p/Top-k

Order of operations typically: apply temperature scaling to logits → apply top-k and/or top-p filtering → renormalize → sample. Many APIs (OpenAI, Anthropic) let you set both simultaneously; using both together, top-p/top-k acts as a safety net regardless of temperature.

## 5. Repetition Penalty

**What it does:** Discourages the model from repeating tokens it has already generated.

**Formula (CTRL-style, multiplicative):**
$$z_i' = \begin{cases} z_i / \theta & \text{if } z_i > 0 \\ z_i \times \theta & \text{if } z_i < 0 \end{cases}$$
for tokens $i$ already seen, where θ > 1 (typically 1.1–1.3).

**Why:** LLMs left to greedy/low-temperature decoding often fall into repetitive loops ("the the the..." or repeating whole sentences). This penalizes logits of already-generated tokens before softmax.

**Example:** If "the" was already generated and its logit is 3.0, with θ=1.2, new logit = 3.0/1.2 = 2.5 — less likely to be picked again.

## 6. Frequency Penalty & Presence Penalty (OpenAI-style, additive)

**Frequency penalty formula:**
$$z_i' = z_i - \alpha \cdot \text{count}(i)$$
where count(i) = number of times token i has appeared so far, α = frequency_penalty coefficient.

**Presence penalty formula:**
$$z_i' = z_i - \beta \cdot \mathbb{1}[\text{count}(i) > 0]$$
a flat penalty applied once if the token has appeared at all (regardless of how many times).

**Why the distinction matters:**
- Frequency penalty scales with how often a word is repeated — good for suppressing overused words.
- Presence penalty is binary — good for encouraging topic diversity/new vocabulary regardless of repeat count.

**Example:** Word "amazing" used 4 times, frequency_penalty=0.5 → penalty = 0.5×4 = 2.0 subtracted from its logit each subsequent time. Presence penalty=0.5 → flat -0.5 subtracted once it's appeared, no matter how many more times.

## 7. No-Repeat N-gram Size

**What it does:** Hard constraint — completely blocks any n-gram (sequence of n tokens) from being generated twice.

**How:** During generation, track all n-grams seen so far; if the next token would recreate a previously seen n-gram, set its probability to 0.

**Example:** no_repeat_ngram_size=3 with "New York City is great, New York City" → generating "is" again after the second "New York City" would recreate a 4-gram match on "New York City is" so it gets blocked.

**Why:** Common in beam search to avoid verbatim repeated phrases; too aggressive a setting can block legitimate repetition (like a phone number or a name that should repeat).

## 8. Max Tokens / Max Length

**What it does:** Hard cap on the number of tokens generated (or total sequence length).

**Why:** Controls cost, latency, and prevents runaway generation. Purely a stopping criterion, doesn't affect the probability distribution itself — generation just halts once the cap is hit (possibly mid-thought).

## 9. Min Length

**What it does:** Forces the end-of-sequence (EOS) token's probability to 0 until a minimum number of tokens have been generated.

**Why:** Prevents the model from stopping too early (e.g., generating a single short sentence when a longer answer is wanted).

## 10. Stop Sequences

**What it does:** Strings that, when generated, immediately terminate output (before max_tokens is reached).

**Why/how:** Useful for structured generation — e.g., stop at "\n\n" to end a single paragraph, or at "```" to end a code block. Checked as a post-hoc string match on generated text, not a distribution modification.

## 11. Beam Search & Num Beams

**What it does:** Instead of sampling, beam search keeps the top-B most probable partial sequences at each step (deterministic, not stochastic).

**Formula:** At each step, expand each of the B beams by all possible next tokens, score by cumulative log-probability:
$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i})$$
keep only the top-B sequences by this score, discard the rest.

**Why:** Greedy decoding (B=1) is myopic and can miss higher-quality overall sequences; beam search explores more of the search space. But large beams can produce bland, generic text (it optimizes for probability, not diversity/interestingness) — this is why beam search is common in machine translation/summarization but rare in open-ended chat.

## 12. Length Penalty (used with beam search)

**What it does:** Normalizes beam scores by sequence length, since raw log-probability sums naturally favor shorter sequences (each additional token multiplies in another probability ≤ 1, i.e., adds a negative log-prob).

**Formula:**
$$\text{score}(y) = \frac{1}{|y|^{\gamma}} \sum_{i=1}^{|y|} \log P(y_i \mid y_{<i})$$
γ > 1 favors longer sequences, γ < 1 favors shorter ones, γ=1 is neutral-ish.

## 13. Logit Bias

**What it does:** Directly adds/subtracts a constant to specific token logits before sampling.

**Formula:** $z_i' = z_i + b_i$ for a manually specified bias $b_i$ per token id.

**Why:** Fine-grained control — e.g., ban a specific word entirely by setting bias to -100 (effectively -∞ after softmax), or force a token to almost always appear with a large positive bias. Used for things like enforcing profanity filters or biasing toward a required format token.

## 14. Random Seed

**What it does:** Fixes the pseudo-random number generator state used in sampling, so that given identical logits and identical sampling parameters, the same "random" choice is made every time.

**Why:** Reproducibility for debugging/testing — without a fixed seed, the same prompt + same temperature can give different outputs on each API call because sampling is stochastic.

---

### Summary table of what to tune for what goal

| Goal | Parameters to adjust |
|---|---|
| Deterministic/factual output | Low temperature (0–0.3), or greedy decoding |
| Creative/diverse output | Higher temperature (0.7–1.2), top-p ~0.9 |
| Avoid repetition loops | Repetition penalty, no-repeat n-gram, frequency penalty |
| Encourage new topics/vocab | Presence penalty |
| Control output length | Max tokens, min length, stop sequences |
| Best "overall" sequence (not stochastic) | Beam search + length penalty |
| Ban/force specific tokens | Logit bias |
| Reproducible results | Fixed seed |

If it'd help, I can also walk through how these interact mathematically in a single decoding step (e.g., temperature → top-p → repetition penalty → sample), since order of application actually changes the resulting distribution.
