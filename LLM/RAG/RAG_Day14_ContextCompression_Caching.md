# RAG Interview Prep — Day 14
## Context Window Management & Compression

---

## 🚀 Quick Summary

Day 13 covered how to *arrange* what fits in a context window; today covers what to do when the genuinely relevant retrieved content **doesn't fit** in the first place, or when you want to shrink it anyway to save cost and latency. Context compression spans a spectrum from cheap and mechanical (deduplicating near-identical chunks) to sophisticated and LLM-powered (summarizing chunks down to their essential claims, or algorithmically dropping low-information tokens). Today also covers **caching** — a completely different lever for the same underlying goal (reduce cost/latency), by avoiding redundant computation entirely rather than shrinking what's computed.

**Think of it like packing for a trip with a strict luggage weight limit.** Deduplication is realizing you packed two identical toothbrushes by accident. Extractive compression is deciding you only need the relevant pages of a guidebook, not the whole book. Abstractive compression (summarization) is condensing that guidebook chapter into your own three-sentence notes. And caching is realizing you don't need to re-pack your toiletry bag from scratch every single trip if it's already packed the same way from last time — just reuse it.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Extractive compression** | Selecting the most relevant sentences/passages from retrieved content, discarding the rest verbatim |
| **Abstractive compression** | Using an LLM to summarize/rewrite retrieved content into a more compact form |
| **Contextual compression** | A retrieval-time filtering step that trims each retrieved document down to only the parts relevant to the specific query |
| **Prompt compression (token-level)** | Algorithmically removing low-information individual tokens from a prompt, using a smaller model to score token importance |
| **Chunk deduplication** | Removing redundant or near-duplicate chunks from a candidate set before they reach the generator |
| **Prefix / KV caching** | Reusing previously-computed attention key-value states for a repeated prompt prefix (e.g., a fixed system prompt), avoiding recomputation |
| **Semantic caching** | Caching and reusing a full response for queries that are *semantically* similar to a previously-answered query, not just exact string matches |

---

# PHASE 1 — Intuition & Visual Map

## The compression spectrum

```
   CHEAP, MECHANICAL                                    EXPENSIVE, SOPHISTICATED
   ────────────────────────────────────────────────────────────────────────▶

   Deduplication    Extractive          Abstractive         Token-level
   (remove exact/   compression         compression         algorithmic
   near-duplicate   (keep relevant      (LLM rewrites/      compression
   chunks)          sentences,          summarizes          (LLMLingua-style,
                     drop the rest       content down)       drops low-info
                     verbatim)                                tokens within
                                                                what's kept)

   Risk of losing information generally INCREASES as you move right — more
   aggressive compression saves more space/cost, but risks discarding
   nuance the generator actually needed, directly threatening faithfulness
   (Module 7) if done carelessly.
```

## Why compression matters beyond "it doesn't fit"

Even when content *does* technically fit in the context window, compression can still be worth doing for:
- **Cost** — fewer input tokens = lower API cost per query (a direct, linear relationship).
- **Latency** — processing fewer tokens is faster (relevant even within budget, not just at the limit).
- **Lost-in-the-middle mitigation (Day 13)** — a shorter, denser context has less "middle" for information to get lost in, in absolute terms.

---

# PHASE 2 — Deep Dive: Compression Techniques

## 1. Chunk Deduplication

**The problem it solves:** techniques from earlier weeks can independently introduce redundant chunks into a candidate set — sliding-window overlap (Day 3) means adjacent chunks share content by design, and multi-query retrieval (Day 11) can retrieve the same or highly similar chunk multiple times across different query variants before fusion.

**Mechanism:** compare candidate chunks pairwise (via embedding similarity, or exact/near-exact text matching) and drop chunks that are near-duplicates of a higher-ranked chunk already selected, before finalizing the context.

**Worked numerical example:**
```
Multi-query retrieval (Day 11) with 4 query variants, each returning
top-5 chunks → up to 20 raw candidates before deduplication.

After RRF fusion (Day 9) and deduplication (removing chunks with
>0.95 cosine similarity to an already-selected higher-ranked chunk):

Raw candidates: 20
Duplicates removed: 7 (many variants surfaced the same 2-3 core chunks)
Unique chunks remaining: 13

→ Without deduplication, up to 7 "slots" in your context budget
  (worth real tokens) would have been spent on redundant content
  instead of genuinely diverse, additional evidence.
```
**Why it matters in practice:** this is a nearly-free win — deduplication is cheap to compute (embedding similarity comparisons are fast) and directly frees up context budget for genuinely new information, rather than wasted redundancy — always worth doing before any more expensive compression technique.

---

## 2. Extractive Compression (Contextual Compression)

**Mechanism:** for each retrieved chunk, instead of passing the *entire* chunk to the generator, extract only the sentences/spans within that chunk that are actually relevant to the specific query — discarding the rest of the chunk's content, verbatim (no rewriting, just selection).

**Worked example:**
```
Retrieved chunk (200 tokens): "AirPods Pro feature Active Noise
Cancellation and Transparency mode. The battery provides up to 6
hours of listening time on a single charge, with the charging case
providing an additional 24 hours. AirPods Pro are sweat and water
resistant with an IPX4 rating. The touch-sensitive stem allows
control of playback and Siri."

Query: "What is the battery life of AirPods Pro?"

Extractive compression output (only the relevant sentence, ~25 tokens):
"The battery provides up to 6 hours of listening time on a single
charge, with the charging case providing an additional 24 hours."
```
**Compression ratio here:** roughly 200 → 25 tokens, an **8x reduction** for this one chunk, with (in this case) no loss of the actually-relevant information, since the extracted sentence directly answers the query.

**Trade-off:** extractive compression assumes relevance can be cleanly identified at the sentence level — but sometimes surrounding sentences provide necessary context for correctly interpreting the "relevant" sentence (e.g., a caveat or condition stated in an adjacent sentence), and naive extraction can strip that away, creating a faithfulness risk of a different kind (technically accurate extracted text, but missing a qualifying condition that changes its correct interpretation).

---

## 3. Abstractive Compression (LLM Summarization)

**Mechanism:** use an LLM to summarize/rewrite retrieved content into a more compact form, rather than just selecting a subset of the original text verbatim.

**When to prefer this over extractive:** when relevant information is spread thinly across a chunk (or across multiple chunks) in a way that no single clean extractable span captures — summarization can synthesize scattered relevant details into a compact, coherent statement that pure extraction couldn't produce.

**The real cost/risk trade-off (be explicit about this in an interview):**
- **Cost:** an extra LLM call per chunk (or per group of chunks) being summarized — directly adds latency and cost, the same fundamental trade-off pattern as Day 11's query transformation techniques.
- **Faithfulness risk:** summarization is itself a generation step, and generation steps can introduce the same hallucination risk that Module 7's faithfulness metric is designed to catch — a summary that subtly misrepresents or drops a nuance from the original chunk creates a compounding faithfulness risk (the final generator's answer is only as faithful as the summary it was given, which itself may have already drifted from the original source).

> **Why This Matters callout:** If asked "would you use abstractive compression in your RAG pipeline," a strong answer acknowledges this compounding risk explicitly — summarizing retrieved content before generation adds a second point where hallucination/information loss can be introduced, on top of the final generation step itself, so it should be reserved for cases where the token savings are genuinely necessary (context truly doesn't fit) rather than applied by default for minor cost savings.

---

## 4. Token-Level Prompt Compression (LLMLingua-style)

**Mechanism:** use a smaller, cheaper model to estimate the "information density" or importance of individual tokens within a prompt, and algorithmically remove the lowest-information tokens (often filler words, redundant phrasing) while preserving the tokens that carry the most meaning — operating at a much finer grain than extractive compression's sentence-level selection.

**Worked conceptual example:**
```
Original (18 tokens): "In addition to this, it should also be noted
that the battery typically lasts around 6 hours on a full charge."

Token-importance-scored compression might remove low-information
filler, producing something like (10 tokens): "battery typically
lasts around 6 hours full charge"

Compression ratio: 18 → 10 tokens ≈ 44% reduction, while preserving
the core factual content.
```
**Trade-off:** more aggressive than extractive compression's sentence-level granularity, and can produce grammatically broken (though still often LLM-parseable) text — the compression is optimized for information preservation as measured by a smaller scoring model, not for human readability, which is a reasonable trade-off if the compressed text is only ever consumed by the generator model, not shown to a human. Worth knowing this class of technique exists by name for breadth, even if it's less commonly hand-implemented than extractive/abstractive approaches.

---

## Compression Technique Comparison Table

| | Deduplication | Extractive | Abstractive | Token-level algorithmic |
|---|---|---|---|---|
| **Cost/latency added** | Very low | Low-moderate | Moderate-high (extra LLM call) | Low-moderate (smaller model) |
| **Faithfulness risk** | Minimal | Moderate (can strip needed context around the extracted span) | Higher (a second generation step can introduce drift/hallucination) | Moderate (can produce hard-to-interpret fragments) |
| **Compression ratio achievable** | Depends on redundancy present | Moderate-high (Day's example: ~8x) | High, flexible | High, flexible |
| **When to use** | Always, first, essentially free | When relevance is cleanly localized within chunks | When relevant info is scattered/needs synthesis | Very tight token budgets, willing to trade readability for density |

---

## Caching — A Different Lever Entirely

Compression shrinks *what* gets processed; caching avoids *reprocessing* things that haven't changed. Both reduce cost/latency, but through completely different mechanisms.

### Prefix / KV Caching

**Mechanism:** transformer inference computes "key" and "value" attention states for every token as part of generating a response. If a prompt's *prefix* is identical across multiple requests (e.g., a fixed system prompt, or the same retrieved context reused across a multi-turn conversation), the key-value states for that shared prefix can be **computed once and reused**, rather than recomputed from scratch on every single request.

**Worked numerical example:**
```
System prompt + instructions: 500 tokens (IDENTICAL across every request)
Retrieved context: 2,000 tokens (varies per query)
User query: 50 tokens (varies per query)

Without prefix caching: every request recomputes attention states
for all 2,550 tokens (500 + 2,000 + 50).

With prefix caching (assuming the 500-token system prompt is cached):
every request only needs fresh computation for the 2,050 remaining
tokens (2,000 + 50) — the system prompt's 500 tokens' KV states are
reused from cache.

Savings per request: 500/2550 ≈ 19.6% of the total input tokens'
computation avoided, purely from caching the fixed prefix.
```
**Why it matters in practice:** this is a "free" optimization in the sense that it requires no compression trade-off at all (no information loss risk) — it's purely an infrastructure/engineering win, and it compounds: any part of the prompt that's genuinely identical across requests (a fixed system prompt, a fixed set of instructions, even a commonly-reused document in a narrow-domain application) is a candidate for prefix caching savings. Many modern LLM APIs and serving frameworks support this natively (sometimes automatically, sometimes requiring the identical-prefix content to be structured consistently at the start of every request to enable cache hits).

### Semantic Caching

**Mechanism:** rather than caching based on exact query string matches (which would rarely hit in practice, since users rarely type the identical query twice), cache based on **semantic similarity** — if a new query's embedding is sufficiently close (above some similarity threshold) to a previously-answered query's embedding, serve the cached previous answer (or cached retrieval results) instead of recomputing from scratch.

**Worked example:**
```
Previously cached query: "What is the battery life of AirPods Pro?"
                          → cached answer + cached retrieved chunks

New incoming query: "How long does the AirPods Pro battery last?"
cosine_similarity(new query embedding, cached query embedding) = 0.94

If similarity threshold is set at 0.90:
  0.94 > 0.90 → CACHE HIT — serve the cached answer, skip retrieval
  and generation entirely for this request
```
**The threshold trade-off (a real design decision, not a fixed constant):**
- **Threshold too low** (e.g., 0.75): risks false-positive cache hits — semantically different queries get incorrectly served a cached answer meant for a different question, potentially returning an irrelevant or wrong answer.
- **Threshold too high** (e.g., 0.99): very few queries will actually hit the cache (since almost no two real user queries are near-identical in embedding space), minimizing the cost-saving benefit of caching at all.
- Setting this threshold correctly requires empirical validation (similar in spirit to chunk-size sweeps, Day 3) — there's no universally correct number, it depends on the query distribution and the cost of a false-positive cache hit for your specific application.

> **Gotcha:** Semantic caching introduces a genuine staleness/correctness risk beyond just threshold tuning — if the underlying knowledge base changes (a document gets updated), a cached answer from before that update could be served as if still current, unless cache invalidation is tied to the relevant document's update events. This is a real production concern worth raising proactively, not just the similarity-threshold tuning question alone.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What's the difference between extractive and abstractive compression, and which carries more faithfulness risk?

<details>
<summary>Show answer</summary>

Extractive compression selects relevant sentences/spans from the original text verbatim, discarding the rest without rewriting anything. Abstractive compression uses an LLM to summarize/rewrite the content into a more compact form. Abstractive compression carries more faithfulness risk, because it's itself a generation step that can introduce hallucination or subtle misrepresentation of the original content — the final answer's faithfulness now depends on both the summarization step and the final generation step, compounding the risk compared to extraction, which preserves the original wording exactly (though it can still risk stripping needed surrounding context).
</details>

---

**Q2 (Easy — calculation).** A chunk is 240 tokens; extractive compression keeps a single 40-token relevant sentence. Compute the compression ratio.

<details>
<summary>Show answer</summary>

```
240/40 = 6x compression ratio
```
</details>

---

**Q3 (Medium — conceptual).** Why is chunk deduplication considered a near-free optimization compared to other compression techniques?

<details>
<summary>Show answer</summary>

Deduplication only requires comparing candidate chunks for similarity (cheap embedding comparisons) and dropping near-duplicates — it involves no LLM calls, no risk of losing genuinely relevant information (since duplicates by definition carry no unique content), and directly frees up context budget for actually-distinct evidence. This makes it essentially risk-free and cheap relative to extractive/abstractive compression, which both involve real trade-offs between space savings and potential information loss.
</details>

---

**Q4 (Medium — conceptual).** Explain how prefix/KV caching reduces cost, and what has to be true about a prompt's structure for it to actually produce cache hits.

<details>
<summary>Show answer</summary>

Transformer inference computes key-value attention states for every input token; if a prompt's prefix (e.g., a fixed system prompt) is byte-identical across multiple requests, those KV states can be computed once and reused rather than recomputed on every request, avoiding redundant computation for that shared portion of the prompt. For this to actually produce cache hits, the identical content needs to be structured consistently at the *start* of every request (since it's a prefix cache — content after the first point of divergence between two requests can't benefit), and needs to be genuinely identical (not just similar) across requests, since KV caching operates on exact prefix matches, unlike semantic caching.
</details>

---

**Q5 (Medium — calculation + reasoning).** A semantic cache is set with similarity threshold 0.85. A new query has cosine similarity 0.88 to a cached query about a topic that was updated in the knowledge base yesterday. What happens, and why might this be a problem even though the similarity threshold logic worked "correctly"?

<details>
<summary>Show answer</summary>

Since 0.88 > 0.85, this is a cache hit by the similarity threshold's logic, and the system would serve the previously cached answer. This is a problem despite the threshold "working correctly" because the underlying knowledge base was updated after the cached answer was generated — the cache has no awareness that the source content changed, so it would serve a now-potentially-stale or incorrect answer. This illustrates that semantic caching needs cache invalidation tied to document update events, not just similarity-threshold tuning, to avoid serving stale answers after the knowledge base changes.
</details>

---

**Q6 (Hard — system design synthesis).** Design a context management strategy for a high-traffic RAG system where the same handful of common questions account for 40% of daily query volume, but the underlying knowledge base updates several times per day. Balance cost savings against staleness risk.

<details>
<summary>Show answer</summary>

Given that 40% of volume concentrates on a small set of common questions, semantic caching offers substantial cost/latency savings — I'd implement it with a conservatively-high similarity threshold (tuned empirically, likely in the 0.90+ range, to minimize false-positive cache hits on subtly different questions) specifically for this high-value common-query segment. Given frequent knowledge base updates, I'd tie cache invalidation directly to document update events — when a document changes, invalidate (or at minimum re-validate) any cached answers whose retrieved source chunks included that document, rather than relying on a fixed time-based cache expiry alone, which could either invalidate too eagerly (losing savings) or too rarely (serving stale answers) depending on how update frequency happens to align with a fixed TTL. I'd also apply prefix/KV caching for the fixed system prompt and instructions across all requests regardless of caching tier, since that's a risk-free optimization independent of the semantic caching strategy. For the long tail of less-common queries (the remaining 60% of volume, more diverse), I wouldn't rely heavily on semantic caching (low hit rate, not worth the false-positive risk) and would instead focus compression efforts (deduplication + extractive compression) on keeping their per-query cost reasonable.
</details>

---

**Q7 (Hard — conceptual, ties across days).** How does context compression interact with the lost-in-the-middle effect from Day 13 — does compression help or hurt that problem, and does it depend on which compression technique is used?

<details>
<summary>Show answer</summary>

Compression generally helps lost-in-the-middle in a mechanical sense — a shorter, denser context has proportionally less "middle" for information to get lost in, and removing redundant/irrelevant content (deduplication, extractive compression) means what remains is more likely to be genuinely relevant, reducing the amount of low-value content that could otherwise occupy prime start/end positions unnecessarily. However, this benefit depends on how compression interacts with chunk ordering: compression should generally be applied *before* final context ordering/sandwiching (Day 13), not as an afterthought, since compressing chunks changes their effective length and might shift what actually ends up positioned where. Abstractive compression specifically introduces an additional consideration — a summarized chunk might inadvertently lose the exact phrasing or specific detail that made it easy for the model to correctly attend to and use, even if it's now positioned favorably, meaning position and content quality/faithfulness need to be reasoned about together rather than treating compression and ordering as fully independent decisions.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Skipping deduplication before other compression techniques — it's nearly free and should almost always happen first.
- ❌ Treating abstractive compression (summarization) as risk-free — it's a generation step that can introduce its own hallucination/faithfulness issues, compounding with the final generation step.
- ❌ Confusing prefix/KV caching (exact-match, infrastructure-level, risk-free) with semantic caching (similarity-based, requires threshold tuning, has real staleness/correctness risk).
- ❌ Setting a semantic cache similarity threshold without empirical validation, or without considering the cost of a false-positive cache hit for your specific application.
- ❌ Implementing semantic caching without a cache invalidation strategy tied to underlying knowledge base updates — a classic staleness bug.
- ❌ Assuming prefix caching "just works" regardless of prompt structure — it requires the cacheable content to be a consistent, identical prefix at the start of every request.

---

# 📌 Cheat Sheet (Day 14)

**Compression spectrum:** deduplication (free, always do it) → extractive (moderate, risk of stripping needed surrounding context) → abstractive (highest faithfulness risk, adds a generation step) → token-level algorithmic (aggressive, sacrifices readability for density).

**Caching (a different lever — avoid recomputation, not shrink content):** prefix/KV caching = exact-match, risk-free, reuses attention states for identical prompt prefixes (e.g., fixed system prompts). Semantic caching = similarity-based, real cost savings on repeated query patterns, but requires threshold tuning AND cache invalidation tied to knowledge base updates to avoid serving stale answers.

**Golden interview line:** *"Compression and caching solve the same cost/latency problem through opposite mechanisms — compression reduces what you process, caching avoids reprocessing what hasn't changed — and the riskiest technique on either side, abstractive summarization and semantic caching respectively, both need explicit guardrails: faithfulness validation for the former, invalidation-on-update for the latter."*

---

*End of Day 14. Next up — Day 15: Citation & Faithfulness Enforcement at Runtime.*
