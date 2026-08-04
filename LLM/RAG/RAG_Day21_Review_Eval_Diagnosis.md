# RAG Interview Prep — Day 21
## Review Day: Evaluation & Diagnosis (Days 19–20) — Closed Book

---

## 📋 How to run this review

No notes. This review leans heavily on **applying** the diagnostic workflow to *new* bug scenarios you haven't seen before, rather than recalling Day 20's specific worked example — that's the actual skill being tested, not memorization of one walkthrough. Target 60–75 minutes.

---

## Section A — Evaluation Fundamentals (quick recap, fresh numbers)

**A1 (calculation).** 4 relevant docs exist for a query; top-8 retrieval finds 3 of them. Compute Recall@8.

<details>
<summary>Show answer</summary>

```
Recall@8 = 3/4 = 0.75
```
</details>

**A2.** High context relevance, low faithfulness. What's the diagnosis?

<details>
<summary>Show answer</summary>
The right evidence was retrieved cleanly (context relevance high — not diluted with noise), but the generator produced claims not well-supported by that good context — a generation-stage failure (hallucination or over-reliance on parametric knowledge), not a retrieval or context-assembly problem.
</details>

**A3.** Why can't a purely synthetic golden eval set alone reliably catch refusal miscalibration's false-answer side?

<details>
<summary>Show answer</summary>
Synthetic QA generation produces questions directly answerable from a source chunk by construction — it doesn't naturally include genuinely-unanswerable "no good answer exists" examples, so the false-answer error type (confidently answering when context is insufficient) has no opportunity to be tested unless that slice is deliberately, separately constructed.
</details>

---

## Section B — Diagnostic Workflow Application (new scenarios)

**B1.** A user reports a customer support RAG bot answers "AirPods Max" questions with information about "AirPods Pro" instead. Walk through Step 1 of the diagnostic workflow for this specific case — what exactly would you check, and what result at Step 1 would tell you to stop there vs. continue to Step 2?

<details>
<summary>Show answer</summary>
I'd inspect the raw, pre-reranking top-k retrieval results for several failing "AirPods Max" queries, specifically checking whether AirPods Max-specific chunks are present at all, and whether AirPods Pro chunks are also present and ranking competitively or higher. If AirPods Max chunks are absent or ranking poorly while Pro chunks dominate, that's a clear retrieval-stage problem (likely an embedding/chunking issue insufficiently distinguishing the two product lines, or a query-vocabulary issue) — I'd stop at Step 1 and address it there. If the correct Max-specific chunks ARE present and ranking well in raw retrieval, Step 1 is ruled out and I'd move to Step 2 to check whether context construction or generation is where the mix-up actually happens.
</details>

**B2.** A team's logging captures raw retrieval results and the final generated answer, but not the actual constructed prompt sent to the generator. Which specific diagnostic step becomes impossible, and why?

<details>
<summary>Show answer</summary>
Step 2 (context-assembly diagnosis) becomes impossible to execute properly — you cannot verify chunk ordering (lost-in-the-middle), dilution, or truncation without inspecting the literal text actually sent to the generator; inferring it from "which chunks were retrieved" isn't sufficient, since the actual ordering, formatting, and whether anything got cut off by budget constraints are all things that happen between retrieval and generation and require their own dedicated log.
</details>

**B3.** A bug only manifests on 3+ hop agentic queries (Day 16), not 1-2 hop queries. Why is per-hop tracing specifically necessary here, rather than just applying the standard 3-step workflow to the final answer alone?

<details>
<summary>Show answer</summary>
The standard 3-step workflow applied only to the final answer would only tell you whether the LAST hop's retrieval/context/generation worked correctly — but Day 16/17's error propagation means a wrong intermediate fact from an EARLIER hop can produce a final answer that looks like a clean "generation-stage failure" when actually the final hop behaved perfectly correctly given the (already-corrupted) information it was working from. Per-hop tracing applies the 3-step workflow recursively to each individual hop, which is the only way to localize whether the problem originated early (and propagated) or genuinely only at the final synthesis step.
</details>

**B4.** After implementing a fix for a chunking issue, a team confirms the original failing query now returns the correct answer and ships the fix. What step did they skip, and what specific risk does skipping it introduce?

<details>
<summary>Show answer</summary>
They skipped regression testing against the full golden eval set. The risk: a chunking change made to fix one query type can have unintended effects on other, previously-working query types (e.g., a larger chunk size that fixes one product's fragmented instructions might dilute embeddings for a different, more narrowly-scoped product's chunks) — without re-running the full eval set, there's no way to confirm the fix didn't introduce new regressions elsewhere, only that it fixed the one reported symptom.
</details>

---

## Section C — Cross-Pipeline Synthesis (the hardest section)

**C1.** A system shows healthy Recall@k and nDCG (Day 4/Module 7) on your dashboard, but users report answers are frequently incomplete for comparative questions ("which is better, X or Y"). Using concepts from Days 11, 16, and 20, diagnose the likely issue.

<details>
<summary>Show answer</summary>
Healthy Recall@k/nDCG measured on a standard eval set likely reflects performance on single-hop, non-comparative queries — if the eval set doesn't specifically include comparative/multi-part questions (Day 11's decomposition scenario), it wouldn't surface a failure specific to that query type. The likely root cause: comparative queries aren't being decomposed (Day 11) or handled via multi-hop retrieval (Day 16) at all — a single embedding of "which is better, X or Y" is being used for one-shot retrieval, which structurally can't match a chunk covering both X and Y simultaneously, since real documents typically describe one product at a time. Diagnostic workflow (Day 20) applied here: Step 1 would reveal that raw retrieval for these specific queries returns chunks about only ONE of the two products being compared, not both — pointing squarely at a missing decomposition/multi-hop step rather than a general retrieval quality problem, which is exactly why the aggregate Recall@k dashboard (likely dominated by simpler queries) looks healthy while this specific query pattern silently fails.
</details>

**C2.** Connect Day 14 (caching), Day 17 (failure modes), and Day 20 (diagnosis) — a user reports getting a wrong answer, but when you try to reproduce it with the exact same query, you get the correct answer instead. What would you check first, and why might standard diagnostic Step 1-3 tracing on your NEW reproduction attempt be misleading?

<details>
<summary>Show answer</summary>
I'd check whether semantic caching (Day 14) served a stale cached answer to the original user — if the underlying knowledge base was updated between the user's original query and your reproduction attempt, a cache invalidation gap could mean the user's request hit a stale cached response while your later reproduction attempt (post-update, or post-cache-expiry) naturally retrieves fresh, correct results. This is misleading for standard diagnosis because tracing your reproduction attempt through Steps 1-3 would show everything working correctly — the bug isn't reproducible via the normal pipeline at all, since the actual root cause is a caching-layer artifact from a specific point in time, not a persistent flaw in retrieval, context assembly, or generation. This is exactly why cache hit/miss status and cache timestamp should be part of the required logging (Day 20), not just the four pipeline-stage logs — otherwise this entire failure category is invisible to standard diagnosis.
</details>

**C3.** A RAG system passes its counterfactual eval slice (Day 17/19 — testing over-reliance on parametric knowledge) with a 95% correct-context-following rate, yet production still shows occasional confidently-wrong answers on recently-changed information. Give two possible explanations that don't contradict the 95% eval result, using concepts from across the curriculum.

<details>
<summary>Show answer</summary>
(1) The counterfactual eval slice, however well-constructed, is necessarily a finite sample of possible knowledge-conflict scenarios — a 95% pass rate still leaves a real 5% failure rate, and if recently-changed-policy queries specifically are a disproportionate fraction of a particular high-traffic query pattern in production, even a "good" 95% aggregate score can produce a non-trivial absolute number of visible production failures, especially concentrated in one query type not evenly represented in the eval slice's composition. (2) The failure could actually originate upstream of generation entirely — e.g., a stale semantic cache (Day 14) serving a pre-update cached answer, or an embedding-drift/stale-index issue (Day 2/4/5) meaning the "recently-changed" document hasn't actually been re-embedded/re-indexed yet — which would look identical in symptom ("wrong answer about recently-changed info") to a genuine generation-stage over-reliance failure, but would NOT be caught or reflected by a generation-stage counterfactual eval slice at all, since that eval slice tests generation behavior given correct retrieved context, not whether the corpus/index itself is actually up to date.
</details>

---

## 📊 Weak Spot Tracker

| Section | Topic | Got it cold? | Needs repair? |
|---|---|---|---|
| A1–A3 | Evaluation fundamentals | ☐ | ☐ |
| B1–B4 | Diagnostic workflow application (new scenarios) | ☐ | ☐ |
| C1–C3 | Cross-pipeline synthesis | ☐ | ☐ |

**This is the last pure-review day before System Design week (Days 22-24).** If Section C felt genuinely hard rather than just unfamiliar, that's worth an extra pass through Days 17 and 20 before moving on — system design questions will assume this diagnostic fluency as a baseline, not build up to it again from scratch.

---

*Days 1–21 complete: Foundations, Retrieval, Generation, Evaluation, and Diagnosis all covered and reviewed. Next up — Day 22: System Design for RAG at Scale.*
