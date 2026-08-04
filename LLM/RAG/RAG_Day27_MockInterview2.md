# RAG Interview Prep — Day 27
## Mock Interview #2 — Full 45-Minute Simulation

---

## 📋 How to run this mock interview

Same rules as Day 25: **45-minute timer, no peeking, answer out loud.** Every question below is new — none overlap with Mock #1 — specifically so you can't pattern-match memorized answers and this actually measures whether Day 26's repairs held, not just whether you remember last time's exam.

**Structure (identical shape to Mock #1):**
- Part 1 — Rapid-fire conceptual (10 min, 6 questions)
- Part 2 — Calculations (10 min, 4 questions)
- Part 3 — Diagnostic scenario (10 min, 1 extended question)
- Part 4 — System design (15 min, 1 extended question)

Answers and rubric are walled off at the bottom. Do not scroll past "END OF EXAM" early.

---

# PART 1 — Rapid-Fire Conceptual (10 min)

**1.1** Why is a cross-encoder architecturally incapable of serving as a first-stage retriever over a large corpus, regardless of how much faster the hardware gets?

**1.2** What is term frequency saturation in BM25, and why does plain TF-IDF lack it?

**1.3** Explain the MaxSim operation in ColBERT-style late interaction, and why it sits architecturally between a bi-encoder and a cross-encoder.

**1.4** Why does increasing k (number of retrieved chunks passed to the generator) not monotonically improve answer quality, even when it monotonically improves Recall@k?

**1.5** What's the difference between chunk-level embedding drift and index staleness — are they the same problem?

**1.6** Why should access control for a RAG system be enforced at the retrieval layer rather than the UI layer?

---

# PART 2 — Calculations (10 min)

**2.1** A term appears in 45 of 2,200 documents. Compute its IDF.

**2.2** Using BM25's saturation term with k1=1.4, compute the contribution `f/(f+k1)` at f=3 and at f=15. What does the gap tell you?

**2.3** A semantic cache has similarity threshold 0.88. A new query has cosine similarity 0.91 to a cached query whose underlying source document was updated 2 hours ago. Is this a cache hit by the threshold logic, and what's the risk?

**2.4** You need to index 400 million vectors at 768 dimensions. Compute memory for (a) raw float32, and (b) PQ-compressed with m=8 sub-vectors and 256-entry codebooks.

---

# PART 3 — Diagnostic Scenario (10 min)

**3.1** A team reports: "Our multi-hop agentic RAG system performs great on 2-hop questions (95% accuracy on our eval set) but drops to 60% accuracy on 4+ hop questions. Retrieval Recall@k, measured per-hop, looks healthy at every individual hop when we spot-check manually. Latency is within budget."

Walk through your full diagnostic reasoning: what's the most likely failure category here, what specific mechanism would you investigate first, and how would you confirm it without just re-reading the final wrong answers?

---

# PART 4 — System Design (15 min)

**4.1** "Design a RAG-based feature that helps users troubleshoot hardware issues by searching product manuals, known-issue databases, and community forum threads — deployed as part of a support app used by millions of consumers, with a requirement that the system gracefully handles both simple factual questions ('what does this blinking light mean') and complex multi-symptom troubleshooting ('my device does X, Y, and sometimes Z, what's wrong')."

Structure your answer: clarify requirements, sketch architecture, go deep on at least two components, close with trade-offs. Pay specific attention to how you'd handle the two very different query complexity types mentioned in the prompt.

---

# ⏱️ END OF EXAM — Stop here until your 45 minutes are up.

---
---
---

# ANSWER KEY & SCORING RUBRIC

---

## Part 1 Answers

**1.1** A cross-encoder requires the query and document to be present together during a joint forward pass (for self-attention across both), so its score cannot be computed until a specific query is known — nothing about it can be precomputed offline. This isn't a speed limitation that faster hardware fixes; it's an architectural property. Even with infinitely fast hardware, you'd still need one full model inference per (query, document) pair at query time, which doesn't change the fundamental need for a fast first-stage narrowing step before a cross-encoder can be applied — the issue is the *number* of required forward passes at query time, not their individual speed.

**1.2** Saturation means additional occurrences of a term contribute diminishing score rather than growing linearly forever — implemented via `f/(f+k1)`, which flattens as f grows. Plain TF-IDF's term frequency component has no such ceiling; it grows roughly linearly (modulo length normalization), meaning a term appearing 100 times scores proportionally higher than one appearing 10 times with no diminishing-returns behavior, which can over-reward raw repetition or keyword stuffing.

**1.3** MaxSim computes, for each query token, its maximum similarity against any document token, then sums these across query tokens. It sits between bi-encoder and cross-encoder because document token embeddings (not just one collapsed vector) can be precomputed offline like a bi-encoder, but the fine-grained token-level matching at query time captures genuine token-to-token interaction more like a cross-encoder — without requiring the full joint forward pass a cross-encoder needs.

**1.4** More chunks means more content pushed into the vulnerable "middle" of the context (lost-in-the-middle risk) and more dilution from marginal/irrelevant content, both of which can hurt context relevance and faithfulness even though Recall@k (which only checks presence anywhere in top-k, not usage quality) technically improves. The optimal k for generation quality should be tuned against downstream generation metrics, not retrieval recall alone.

**1.5** No, they're different problems. Embedding drift is about vector-space incompatibility between different embedding model versions — mixing vectors from an old and new model produces meaningless similarity scores. Index staleness is about the index (e.g., IVF cluster centroids) not reflecting the current data distribution or corpus content because it hasn't been updated/re-clustered/re-indexed recently — a problem that can occur even with a single, unchanged embedding model, simply due to insufficient index maintenance as data changes over time.

**1.6** If retrieval queries the full corpus regardless of permissions and access control is only enforced at the display/UI layer, unauthorized content can still be retrieved and fed into the generator's context — producing an answer that's conditioned on, and can reveal, content the user was never authorized to see, silently, through fluent generated text rather than a visibly-hidden document. Enforcement must happen where the actual data access occurs (retrieval), not just where it's displayed.

---

## Part 2 Answers

**2.1**
```
IDF = log(2200/45) = log(48.9) ≈ 3.89
```

**2.2**
```
f=3:  3/(3+1.4) = 3/4.4 ≈ 0.682
f=15: 15/(15+1.4) = 15/16.4 ≈ 0.915
```
Going from 3 to 15 occurrences (12 more) only adds about 0.233 to the score contribution, and the curve is clearly flattening — demonstrating the diminishing-returns behavior BM25's saturation is designed to produce.

**2.3** Yes, this is a cache hit by the threshold logic (0.91 > 0.88). The risk: the underlying source document was updated 2 hours ago, meaning the cached answer may now be stale/incorrect, and unless cache invalidation is explicitly tied to document update events, the system will confidently serve outdated information despite the similarity-matching logic working "correctly" by its own rules.

**2.4**
```
(a) 400,000,000 × 768 × 4 bytes = 1,228,800,000,000 bytes ≈ 1.14 TB
(b) 400,000,000 × 8 bytes (8 sub-vectors × 1 byte each) = 3,200,000,000 bytes ≈ 2.98 GB
```

---

## Part 3 Answer

**Most likely failure category:** error propagation in multi-hop retrieval (Day 16/17) — a wrong or subtly-off intermediate fact from an early hop compounding through later hops, rather than a retrieval-quality problem, since per-hop Recall@k spot-checks look healthy in isolation.

**Why per-hop Recall@k looking "healthy" doesn't rule this out:** Recall@k only checks whether relevant documents were retrieved for that hop's specific query — it says nothing about whether the *fact extracted/reasoned from* that hop's retrieved content was correctly interpreted or whether the *next hop's query* was correctly formulated based on that fact. A hop can retrieve perfectly relevant documents and still have the reasoning step built on top of them go subtly wrong (e.g., misreading a specific number, conflating two similar entities mentioned in the same chunk), and that error then propagates forward even though every individual hop's retrieval was technically fine.

**What to investigate first:** trace the full hop-by-hop reasoning chain (not just final answers) for a sample of failing 4+ hop questions specifically, checking at each hop whether the *intermediate conclusion drawn* (not just the retrieved documents) was actually correct — looking specifically for the point where a hop's stated "fact" first becomes wrong, even if its retrieval was accurate. This is different from re-reading final wrong answers, because the final answer alone doesn't reveal *which* hop's reasoning step first introduced the error — you need the full intermediate trace (Day 20's observability requirement, applied specifically to multi-hop chains).

**How to confirm without just re-reading final answers:** compare each hop's stated intermediate conclusion against ground truth (if available) or against a careful manual re-derivation from that hop's actual retrieved content, isolating the specific hop number where accuracy first drops for 4+ hop chains vs. 2-hop chains — if this shows errors clustering at a consistent hop position (e.g., hop 3 specifically), that's strong evidence of a specific weak link (e.g., a particular type of reasoning step, or a stopping-criteria issue causing premature synthesis) rather than generic accumulated noise.

*(Full credit requires: naming error propagation specifically, explaining why per-hop recall doesn't rule it out, AND proposing hop-by-hop intermediate-conclusion tracing rather than just "look at more examples.")*

---

## Part 4 — Sample Strong Answer Outline

**Phase 1 (clarify):** Roughly how large is the combined corpus (manuals + known-issues + forum threads)? Forum threads likely update very frequently and vary wildly in quality/reliability compared to official manuals — should these be weighted or filtered differently? What's the latency target given this is consumer-facing at scale (likely need sub-1-2 second responses)? Is there a way to distinguish simple vs. complex queries upfront, or should the system detect this itself?

**Phase 2 (architecture):** Standard pipeline, but with an explicit **query complexity routing step** early in the pipeline (Day 9/11's routing principle, now as a primary design feature rather than an optimization): classify incoming queries as simple (single symptom/fact lookup) vs. complex (multi-symptom troubleshooting) before deciding the retrieval strategy.

**Deep dive candidates (pick 2):**
- **Query complexity routing (Day 9/11/16):** simple queries ("what does this blinking light mean") route to standard single-shot hybrid retrieval + rerank — fast, cheap, sufficient. Complex multi-symptom queries ("does X, Y, and sometimes Z") route to query decomposition (Day 11) at minimum — treating each symptom as a sub-question — and potentially agentic multi-hop retrieval (Day 16) if symptoms need to be cross-referenced against known-issue combinations rather than looked up independently (e.g., "X+Y+Z together specifically indicates known issue #482" requires synthesis across sub-question results, not just independent lookups). This directly addresses the prompt's explicit requirement to handle both complexity types gracefully.
- **Source reliability weighting:** official manuals and a curated known-issues database are higher-trust sources than open community forum threads — I'd consider this in ranking/reranking (e.g., a reliability-aware component in the reranking score, Day 10) and definitely in citation/faithfulness enforcement (Day 15) — the system should probably be more conservative (more willing to hedge/refuse) when its best supporting evidence for a complex multi-symptom answer comes primarily from unverified forum content vs. official documentation, since the cost of a wrong hardware-troubleshooting answer at consumer scale is non-trivial (potentially leading to unnecessary returns, unsafe DIY fixes, etc.).

**Phase 4 (trade-offs):** given consumer scale (millions of users), I'd push hard on caching (Day 14) for the most common simple queries — troubleshooting questions likely follow a long-tail distribution where a small set of common issues account for a large fraction of volume, making semantic caching a strong lever here specifically, with cache invalidation tied to known-issue-database updates. I'd explicitly flag that complex multi-symptom queries (rarer, but higher latency due to decomposition/multi-hop) shouldn't be optimized at the expense of the much higher-volume simple-query path — the routing step exists precisely to avoid paying multi-hop latency/cost on every query by default.

*(Full credit requires: explicit query-complexity routing as the central design idea addressing the prompt's stated requirement, source-reliability reasoning tied to faithfulness/citation behavior, and a caching strategy scoped to consumer-scale volume patterns.)*

---

## 📊 Scoring Interpretation

| Score | What it means |
|---|---|
| 26-30 | You're ready. Spend remaining days on light review and rest, not new drilling. |
| 20-25 | Close — one more focused repair pass (repeat Day 26's process with these new results) before the real interview. |
| 14-19 | Compare this score to Mock #1's — if it's meaningfully lower, something in Day 26's repair didn't stick; if it's similar or higher, you're improving but need another full repair cycle. |
| Below 14 | Don't panic, but be honest: revisit full weeks via the Weak Spot Trackers (Days 6, 12, 18, 21, 24), not just individual questions, and consider whether you need more time than originally planned. |

**Compare this score directly against Mock #1.** The delta matters more than the absolute number — it's your actual measure of whether the last two days of repair worked.

---

*Next up — Day 28: Final Cheat-Sheet Consolidation — a single-page reference pulling from all 27 prior days, built specifically for a last skim the morning of the interview.*
