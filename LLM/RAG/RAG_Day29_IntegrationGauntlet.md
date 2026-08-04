# RAG Interview Prep — Day 29
## Buffer Day — Final Full-Curriculum Integration Gauntlet

---

## 🚀 How to use today

This is the first of your two open buffer days. There are two ways to spend it — pick based on Mock #2's actual result, honestly:

- **If Mock #2 scored 20+:** you're in good shape. Skip straight to the **Integration Gauntlet** below — a final set of hard, cross-curriculum synthesis questions you haven't seen before, designed to be harder than anything in Mocks #1 or #2. This is a ceiling-test, not a repair exercise.
- **If Mock #2 scored below 20, or scored lower than Mock #1:** don't do the gauntlet yet. Instead, repeat Day 26's repair process using Mock #2's specific misses, then save the gauntlet below for Day 30.

---

## Part A — The Integration Gauntlet (10 questions, no time limit — this is about depth, not speed today)

These questions are deliberately harder and more cross-cutting than anything in the daily reviews or mocks. Each one requires connecting three or more days. Answer fully before checking.

---

**G1.** You're told a RAG system uses semantic chunking (Day 3), ColBERT reranking (Day 10), and runs entirely on Apple's on-device tier (Day 23). Identify the single biggest tension between these three choices, and how you'd resolve it.

<details>
<summary>Show answer</summary>
The tension is storage/compute footprint: semantic chunking requires embedding calls at ingestion time (moderate cost, one-time), but ColBERT's late-interaction approach requires storing PER-TOKEN embeddings for every chunk rather than one collapsed vector — a substantially larger memory footprint than a standard bi-encoder index. On Apple's on-device tier, memory is an absolute, tight, shared ceiling (Day 23), not a scale-driven optimization question — ColBERT's storage overhead is likely incompatible with on-device constraints at almost any reasonable personal-corpus size, even before considering the generator model's own footprint. Resolution: on-device, a standard compact bi-encoder with PQ/binary-quantized compression (Day 4) is almost certainly the right choice over ColBERT, reserving late-interaction rerankers for the Private Cloud Compute tier or cloud deployments where memory is far less constrained.
</details>

---

**G2.** A system shows: high Recall@k, high nDCG, high context relevance, high faithfulness — and users are STILL unhappy. Using concepts beyond Module 7's triad alone, what else could be wrong?

<details>
<summary>Show answer</summary>
All four of these metrics can be excellent while the system still fails on dimensions they don't measure: (1) Answer relevance specifically wasn't mentioned — a faithful, well-grounded answer can still not address what was actually asked (Module 7's distinct third leg of the triad). (2) Refusal miscalibration (Day 17) — the system might be over-refusing on genuinely answerable questions, which none of these four metrics would flag, since they're typically measured on cases where an answer was actually given. (3) Latency — a system can be perfectly accurate but too slow to feel usable, which none of these correctness-focused metrics capture at all. (4) The eval set itself might not represent the actual production query distribution (Module 7 §7.6/7.7) — high scores on a golden set don't guarantee real user satisfaction if real queries differ meaningfully from what's tested. This question is really testing whether you over-rely on the RAG triad as a complete picture rather than one useful slice of a much larger quality surface.
</details>

---

**G3.** Explain how a single bad decision at ingestion time (Day 3's chunking) can simultaneously cause a Day 7 (sparse retrieval) problem, a Day 8 (dense retrieval) problem, AND a Day 17 (over-reliance on parametric knowledge) problem — using one concrete example.

<details>
<summary>Show answer</summary>
Example: chunking a document about a recently-updated product policy in a way that separates the policy's specific numeric detail from the sentence identifying WHICH product/date it applies to (e.g., one chunk ends with "...the return window is now" and the next begins with "30 days, effective March 2024"). Sparse retrieval (Day 7) suffers because BM25 matching on "return window" terms might not co-occur with "30 days" in the same chunk, hurting exact-term relevance scoring. Dense retrieval (Day 8) suffers because the embedding of either half-chunk is a blurred, incomplete representation of the actual policy, landing in a less accurate region of vector space than a complete chunk would. Over-reliance on parametric knowledge (Day 17) is then MORE likely to occur as a downstream consequence — if the generator receives an incomplete or ambiguous chunk (missing the specific "30 days, effective March 2024" detail clearly attached to its context), it has weaker grounding to override its own strong prior about typical/historical return windows, making it more likely to fall back on a well-known default. This demonstrates that a single upstream chunking mistake doesn't just cause one isolated symptom — it can degrade multiple downstream stages simultaneously, including making an otherwise-separate generation-stage failure mode MORE likely, not just retrieval quality.
</details>

---

**G4.** Design a single golden eval set slice that would simultaneously test: refusal calibration (Day 15/17), context relevance (Module 7), AND multi-hop error propagation (Day 16). Is this even possible with one query type, or does it require separate slices?

<details>
<summary>Show answer</summary>
This genuinely requires separate slices — these three failure modes test fundamentally different conditions that can't be cleanly satisfied by one query design. Refusal calibration needs queries with KNOWN sufficient vs. KNOWN insufficient context, deliberately split. Context relevance needs queries where retrieval is deliberately noisy (surrounded by plausible-but-irrelevant chunks) to see if the generator is misled by dilution. Multi-hop error propagation needs genuinely sequential multi-part queries with verifiable intermediate ground truth at each hop. Attempting to force one query to test all three simultaneously would conflate the signals — if such a combined query fails, you wouldn't know which of the three mechanisms caused the failure, defeating the entire purpose of having decomposed, interpretable eval metrics in the first place (this connects directly back to Module 7 §7.1's opening argument for why retrieval and generation need separate metrics — the same "don't conflate independently-failing things into one signal" principle applies to eval SET DESIGN, not just eval METRIC design).
</details>

---

**G5.** A cost-conscious team proposes removing the reranking stage entirely (Day 10) to cut costs, relying only on hybrid search with RRF (Day 9) for final ranking. Using Day 22's cost-modeling lens, when would this actually be a reasonable trade-off, and when would it be a mistake?

<details>
<summary>Show answer</summary>
Reasonable when: reranking cost is a genuinely large fraction of total cost (as in Day 22's worked example, where it was actually the dominant line item) AND the use case's latency/quality bar tolerates first-stage-only ranking — e.g., a lower-stakes internal tool where nDCG-level position precision matters less, or a use case with a short context window anyway (limiting how much reranking's position-optimization can even help, similar to Day 23's on-device reasoning where only 2-3 chunks fit regardless). A mistake when: the application is high-stakes or the context window is generous enough that WHICH chunks end up in the top 2-3 positions genuinely matters for lost-in-the-middle reasons (Day 13) — removing reranking there risks exactly the kind of position-quality degradation nDCG is designed to catch, potentially causing a larger downstream faithfulness/answer-quality cost than the reranking cost saved. The right call requires actually quantifying both sides (Day 22's cost model AND an eval-set-measured quality delta with/without reranking) rather than assuming cost-cutting is free of quality consequences.
</details>

---

**G6.** Connect Day 5 (multi-tenancy), Day 14 (semantic caching), and Day 23 (Apple's on-device tier): why would semantic caching be architecturally MUCH simpler to reason about correctly on Apple's on-device tier than in a multi-tenant cloud system?

<details>
<summary>Show answer</summary>
In a multi-tenant cloud system, semantic caching introduces a cross-tenant risk dimension on top of the usual staleness risk (Day 14): a cached answer could theoretically be served across tenant boundaries if the cache isn't correctly scoped per-tenant, compounding the caching problem with the Day 5 isolation problem. On Apple's on-device tier, there's inherently only ONE user's data on that device — there's no multi-tenancy dimension to get wrong in the first place, so semantic caching only needs to reason about the single staleness-vs-cache-invalidation problem from Day 14, not an additional cross-user isolation problem layered on top. This is a specific instance of Day 23's broader point that on-device architecture sidesteps certain classes of problems structurally rather than requiring them to be correctly mitigated.
</details>

---

**G7.** A hiring manager asks: "If you had to remove ONE technique from everything we've discussed today, and your system would degrade the least, which would it be, and why?" This is testing whether you understand relative importance, not just breadth. Answer for a **generic, mid-scale, moderate-stakes RAG system** (not a specific niche scenario).

<details>
<summary>Show answer</summary>
There's no single universally-correct answer here, and the strength of the response is in the reasoning, not the specific pick — but a defensible answer: query transformation techniques (Day 11 — multi-query, HyDE, decomposition, step-back) are the most removable for a generic moderate-stakes system, because (a) they're the most conditionally-applied techniques already (Day 11 explicitly argues against blanket application), meaning a well-tuned base pipeline (good chunking, good embeddings, hybrid retrieval, reranking) already handles the majority of queries without needing them, and (b) their absence degrades gracefully — queries that would have benefited from query transformation simply retrieve somewhat less optimally rather than failing outright, unlike removing something more structurally load-bearing like reranking (which affects EVERY query's final ranking quality) or faithfulness enforcement (Day 15, where removal risks the worst failure mode — confident wrong answers — rather than a graceful quality degradation). The key reasoning to demonstrate: distinguishing between techniques that provide broad, structural quality (harder to remove safely) vs. techniques that provide narrow, conditional uplift on a subset of queries (safer to remove, cost/benefit permitting).
</details>

---

**G8.** Explain why "faithfulness" (Module 7) and "over-reliance on parametric knowledge" (Day 17) are measuring almost the opposite failure direction, even though both are usually discussed under the umbrella of "hallucination."

<details>
<summary>Show answer</summary>
Generic hallucination/low faithfulness is the model ADDING content beyond what the context supports — fabrication, going beyond the evidence. Over-reliance on parametric knowledge is subtly different: the model IGNORING or overriding correct evidence that WAS present, substituting its own prior instead — a failure of not using available information, rather than inventing information that isn't there. They can even show up as opposite symptoms in a faithfulness score: pure hallucination lowers faithfulness because claims aren't traceable to context at all; over-reliance on parametric knowledge might in some cases still score as "faithful" by a naive check if the parametric answer happens to overlap enough with context in an unrelated way, or might show as unfaithful if the parametric answer directly contradicts a specific retrieved claim — meaning the two failure modes require genuinely different measurement approaches (standard claim-decomposition faithfulness checks vs. deliberately-constructed counterfactual eval slices), not just one metric catching both.
</details>

---

**G9.** In a live interview, you're asked to design a RAG system, and the interviewer deliberately gives you almost no requirements, then interrupts your clarifying questions after just one with "let's just say it's for a general use case, keep going." How do you proceed without losing the structured methodology from Day 22?

<details>
<summary>Show answer</summary>
I'd state a small set of reasonable default assumptions explicitly and out loud (e.g., "given a general use case, I'll assume a moderate-scale corpus — a few million documents — a latency target around 1-2 seconds, and moderate accuracy stakes, and I'll flag if any part of my design would change significantly under different assumptions"), then proceed directly into Phase 2's high-level architecture rather than stalling or repeatedly pushing for clarification the interviewer has signaled they don't want to give right now. This demonstrates the SAME underlying skill the clarification phase was meant to test — recognizing what assumptions matter and being explicit about them — just compressed into a faster form, rather than abandoning the methodology because the interviewer short-circuited one part of it. I'd also make a mental note to revisit and explicitly flag trade-offs in Phase 4 tied to the assumptions I stated upfront, showing the assumption-stating wasn't just a formality but is actually driving specific downstream design choices.
</details>

---

**G10.** Final synthesis: in under 90 seconds, explain the ENTIRE RAG pipeline end to end, naming every major stage and the ONE most important trade-off at each stage, as if explaining to a technically strong engineer who has never worked on RAG specifically. This is the "can you actually teach this" test.

<details>
<summary>Show answer</summary>
A strong 90-second answer touches: (1) Ingestion — documents get chunked (trade-off: chunk size balances coherence vs. granularity) and embedded (trade-off: embedding model accuracy vs. footprint/cost) into a vector index (trade-off: HNSW vs. IVF balances update-friendliness vs. memory). (2) Retrieval — a query gets embedded and searched, often combined with keyword/BM25 search via RRF (trade-off: sparse catches exact matches dense embeddings miss). (3) Reranking — a smaller candidate set gets more precisely reordered by a more expensive model (trade-off: accuracy vs. latency/cost, only feasible because the set is small). (4) Context construction — reranked chunks get assembled into a prompt (trade-off: more chunks helps recall but risks lost-in-the-middle and dilution). (5) Generation — the LLM produces an answer conditioned on that context (trade-off: faithfulness enforcement and refusal calibration trade helpfulness against hallucination risk). (6) Evaluation/monitoring — retrieval and generation quality are measured SEPARATELY, because a wrong answer could originate at any stage and conflating them into one score loses the ability to diagnose and fix the actual problem. This full-pipeline fluency, delivered smoothly without notes, is genuinely the single best pre-interview readiness check in this entire curriculum — if you can do this comfortably, you're ready.
</details>

---

## Part B — If You Have Time Left

Pick **one** of the following, based on what still feels least automatic:

- Redo Day 22's worked cost model from scratch, with different made-up numbers, without looking.
- Redo Day 20's diagnostic decision tree from memory, sketching it out by hand.
- Pick any three "Golden interview line" quotes from across the daily files and say them out loud verbatim, then explain why each is phrased the way it is (what specific misconception it's preempting).

---

*Next up — Day 30: final buffer day. If today's gauntlet went well, Day 30 is genuinely optional rest. If it didn't, Day 30 repeats whichever section of today felt weakest, one more time.*
