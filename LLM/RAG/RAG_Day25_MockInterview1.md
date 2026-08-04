# RAG Interview Prep — Day 25
## Mock Interview #1 — Full 45-Minute Simulation

---

## 📋 How to run this mock interview

**This is a real exam, not a study guide.** Do not scroll ahead to check answers. Set a timer for 45 minutes. Answer out loud if possible — speaking your answer is a different skill than recognizing a correct answer on a page, and it's the actual skill being tested in a real interview.

**Structure (mirrors a real Apple MLE loop):**
- Part 1 — Rapid-fire conceptual (10 min, 6 questions, ~90 sec each)
- Part 2 — Calculations (10 min, 4 questions, ~2-3 min each)
- Part 3 — Diagnostic scenario (10 min, 1 extended question)
- Part 4 — System design (15 min, 1 extended question)

All answers and a scoring rubric are in a separate section at the very end — do not read past "END OF EXAM" until your 45 minutes are up.

---

# PART 1 — Rapid-Fire Conceptual (10 min)

**1.1** Why does cosine similarity remain the standard comparison metric for text embeddings instead of raw dot product?

**1.2** What's the single biggest reason a two-stage retrieve-then-rerank pipeline outperforms a single-stage bi-encoder-only pipeline?

**1.3** Explain, in one or two sentences, why BM25's term-frequency saturation exists.

**1.4** What problem does HyDE solve, and why doesn't the hypothetical document's factual accuracy matter?

**1.5** Why can't retrieval and generation quality be captured by a single end-to-end correctness metric?

**1.6** What's the practical difference between pre-filtering and post-filtering a vector search, and when does the choice actually matter?

---

# PART 2 — Calculations (10 min)

**2.1** A document has "warranty" appearing 6 times in a 180-word document. The corpus has 4,000 documents, and "warranty" appears in 80 of them. Compute the IDF, and the TF-IDF score (using TF = raw count / document length).

**2.2** Using RRF with k=60, a document ranks 4th in a sparse retriever's results and 2nd in a dense retriever's results. Compute its RRF score.

**2.3** A cross-encoder reranker takes 14ms per candidate. Your latency budget for the reranking stage is 350ms. What's the maximum number of candidates you can rerank sequentially within budget?

**2.4** A context window is 10,000 tokens. You reserve 2,000 for generation, 500 for system instructions, and 1,500 for conversation history. Average chunk size is 450 tokens. How many chunks fit in the remaining budget?

---

# PART 3 — Diagnostic Scenario (10 min)

**3.1** A team reports: "Our RAG system's aggregate faithfulness score (measured on our standard eval set) is 0.94 — very healthy. But we're getting user complaints about occasional confidently wrong answers when customers ask about pricing that changed last month."

Walk through your full diagnostic reasoning: what's the most likely failure mode, why does the healthy aggregate score not rule it out, and what would you actually do to confirm your hypothesis and fix it? Be specific — name the exact mechanism, not just "it's probably a data issue."

---

# PART 4 — System Design (15 min)

**4.1** "Design a RAG system that lets internal engineers at a company search across 500,000 internal design documents, RFCs, and Slack threads, with a requirement that different teams' confidential documents remain isolated from engineers outside that team. Latency target is under 2 seconds. Walk me through your design."

Structure your answer using the four-phase approach: clarify requirements, sketch high-level architecture, go deep on at least two components you think matter most for this specific prompt, and close with trade-offs.

---

# ⏱️ END OF EXAM — Stop here until your 45 minutes are up.

---
---
---

# ANSWER KEY & SCORING RUBRIC

*(Score yourself honestly. For each question: 2 points = nailed it with correct reasoning, 1 point = right idea but incomplete/imprecise, 0 points = wrong or blank. Max score: 30. Interpretation guide is at the very bottom.)*

---

## Part 1 Answers

**1.1** Cosine similarity normalizes by both vectors' magnitudes, isolating pure directional/angular similarity — this matters because embedding magnitude often reflects incidental factors (text length, model confidence) rather than meaning, so comparing direction alone better reflects semantic similarity. Raw dot product is magnitude-sensitive and can be skewed by vector length differences unless embeddings are pre-normalized (in which case dot product and cosine similarity become identical).

**1.2** A bi-encoder alone can't model fine-grained token-to-token interactions between query and document (each is encoded independently into a single vector before ever seeing the other) — a cross-encoder reranker adds that joint-attention accuracy on top, catching cases the bi-encoder's coarser first-pass ranking got wrong, specifically at the positions (top of the list) that matter most for generation.

**1.3** Without saturation, a term appearing many times would keep contributing linearly-increasing score forever, over-rewarding raw repetition (and potentially rewarding keyword stuffing) rather than reflecting a genuine ceiling on how much more relevant a document becomes after enough occurrences already establish the topic.

**1.4** HyDE addresses the query-document style/vocabulary asymmetry — a hypothetical LLM-generated answer is stylistically similar to real answer documents even if factually imperfect, so embedding it (instead of the raw query) turns retrieval into a document-to-document style match. Factual accuracy doesn't matter because the hypothetical document is discarded after producing its embedding — only used to generate a better search vector, never shown to the user.

**1.5** A single end-to-end score conflates two independently-failing stages — a wrong answer could stem from bad retrieval, bad generation on top of good retrieval, or even correct retrieval/generation on top of wrong source data. Separate metrics (Recall@k/nDCG for retrieval, faithfulness/answer relevance for generation) let you localize which stage to actually fix, since the fixes are completely different.

**1.6** Post-filtering runs similarity search first and discards non-matching results afterward — simple, but risks returning far fewer than k results if the filter is highly selective. Pre-filtering restricts the candidate pool before/during search — better recall for selective filters, but can degrade toward near-brute-force performance within the filtered subset unless the ANN index itself is filter-aware. The choice matters most when the filter is highly selective (e.g., a single tenant in a huge shared index); it matters much less for low-selectivity filters where most documents match anyway.

---

## Part 2 Answers

**2.1**
```
IDF = log(4000/80) = log(50) ≈ 3.91
TF = 6/180 ≈ 0.0333
TF-IDF = 0.0333 × 3.91 ≈ 0.130
```

**2.2**
```
1/(60+4) + 1/(60+2) = 1/64 + 1/62 ≈ 0.01563 + 0.01613 = 0.03176
```

**2.3**
```
350ms / 14ms ≈ 25 candidates
```

**2.4**
```
10,000 - 2,000 - 500 - 1,500 = 6,000 tokens remaining
6,000 / 450 ≈ 13.3 → 13 chunks
```

---

## Part 3 Answer

**Most likely failure mode:** over-reliance on parametric knowledge — specifically, a knowledge-conflict failure triggered by recently-changed pricing information. The pattern ("recently changed," "occasional," "confidently wrong" despite retrieval presumably working) is the signature case: the model's strong pretrained prior about "typical" or historical pricing competes with and sometimes overrides correctly-retrieved, updated context.

**Why the healthy 0.94 aggregate score doesn't rule this out:** a standard golden eval set is very unlikely to contain deliberately-constructed examples where retrieved context contradicts a strong parametric prior — that's not how synthetic or typical eval questions get generated. This failure mode only manifests under that specific adversarial condition, so a system can score 0.94 in aggregate on typical questions while having a real, undetected blind spot specifically for recently-changed facts.

**What to actually do:**
1. Build a dedicated counterfactual eval slice — deliberately construct examples where retrieved context states a changed/updated fact that contradicts commonly-known or previously-true information, and measure the rate at which the model follows the context vs. reverts to the old default.
2. Before assuming it's purely a generation-stage issue, rule out a much simpler and equally likely alternative explanation: check whether this is actually a stale semantic cache or stale index issue instead (Day 14/4) — i.e., confirm the updated pricing document was actually re-embedded/re-indexed and that no cached answer from before the price change is being served. This has the same visible symptom but a completely different root cause and fix.
3. If confirmed as a genuine generation-stage knowledge-conflict issue (not caching/staleness), fix via strengthened prompt instructions prioritizing provided context explicitly, surfacing recency metadata (e.g., "as of [date]") directly in the prompt, and/or fine-tuning on knowledge-conflict examples if prompting alone doesn't sufficiently fix it.
4. Validate the fix against the new counterfactual eval slice AND the full existing eval set (regression check), not just the originally reported complaint.

*(Full credit requires: naming the specific failure mode by name, explaining why aggregate eval missed it, AND raising the caching/staleness alternative explanation as a first-check before jumping to a generation-stage fix. Partial credit for naming the failure mode without the staleness-ruling-out step.)*

---

## Part 4 — Sample Strong Answer Outline

**Phase 1 (clarify):** How many teams, roughly, and how granular does isolation need to be (team-level, or finer)? Is 500,000 documents relatively static or does it grow significantly day-to-day (Slack threads suggest high-frequency updates)? Is "confidential" a hard compliance requirement or an internal best-practice preference — this affects how much isolation-cost is justified? What's an acceptable staleness window for newly-posted Slack threads to become searchable?

**Phase 2 (architecture):** Standard reference pipeline — ingestion (chunking documents/RFCs structurally, treating Slack threads as a different, more frequently-updated content type possibly needing different ingestion cadence) → hybrid retrieval (BM25 + bi-encoder, since RFCs/internal docs likely contain exact identifiers like ticket numbers or project codenames alongside natural language) → reranking → context construction → generation with citation enforcement.

**Deep dive candidates (pick 2):**
- **Isolation strategy (Day 5):** given confidentiality is a hard requirement across teams, lean toward per-team namespaces/partitions rather than a single shared filtered index — logical isolation without the operational cost of fully separate indexes per team, and avoids the filtering-correctness risk of a shared index with a `team_id` filter for something as sensitive as confidential internal documents.
- **Freshness/ingestion design (Day 4/22):** Slack threads need much more frequent (near-real-time or frequent-batch) ingestion than RFCs/design docs, which change rarely — differentiated ingestion cadence per content source, similar to Day 22's HR-vs-engineering-wiki example, rather than one uniform batch schedule.
- **Latency budget (Day 8/10):** a 2-second budget is generous enough to comfortably support hybrid retrieval + a full cross-encoder reranking stage without needing to cut corners — worth explicitly stating that this scale/latency profile does NOT require billion-scale infrastructure treatment.

**Phase 4 (trade-offs):** If document volume were 100x larger or latency requirements were much tighter, would reconsider more aggressive index compression and possibly consolidate to a shared-index-with-filter model for cost efficiency, accepting more isolation risk in exchange — but at this stated scale, per-team namespacing is the safer default given the explicit confidentiality requirement.

*(Full credit requires: asking genuine clarifying questions before architecture, correctly identifying per-team isolation as the right call given the confidentiality requirement stated in the prompt, and explicitly stating that this scale doesn't require heavy infrastructure — not just describing a generic pipeline.)*

---

## 📊 Scoring Interpretation

| Score | What it means |
|---|---|
| 26-30 | Strong — you're interview-ready on breadth; focus remaining time on speaking fluency and speed under real time pressure |
| 20-25 | Solid — some real gaps exist; use Day 26 to target specific weak questions, not a general re-review |
| 14-19 | Concerning — significant gaps in either depth or synthesis; revisit whole weeks, not just individual days |
| Below 14 | Go back through the Weak Spot Trackers from Days 6, 12, 18, 21, 24 before attempting Mock #2 |

**Log every question you scored below 2 points into Day 26's repair template — that document only works if you're specific about exactly which questions and why.**

---

*Next up — Day 26: Targeted Repair, using your actual results from this mock interview.*
