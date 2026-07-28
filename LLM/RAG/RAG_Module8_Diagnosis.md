# RAG Module 8 — Diagnosis & Debugging

---

## 8.1 The failure taxonomy

Every bad RAG answer traces back to a failure in one (or more) of four distinct places. Naming this taxonomy explicitly, unprompted, is one of the highest-signal things you can do in a system-design or debugging interview.

1. **Retrieval miss** — the relevant chunk was never fetched at all (didn't make it into top-k). Root causes live in Modules 1-4: embedding domain mismatch, bad chunking, wrong index/metric, insufficient k, dense-only retrieval missing an exact-term query.
2. **Retrieval-but-ignored** — the relevant chunk *was* fetched (present in the context sent to the generator) but the generator didn't use it. Root cause lives in Module 6: "lost in the middle" positioning, context overloaded with noise/redundant chunks crowding it out, poor prompt instructions.
3. **Hallucination-despite-context** — the generator fabricates or distorts claims not supported by the retrieved context, even though good context was present and positioned well. A pure generation-stage failure — model overriding provided context with parametric "knowledge," or extrapolating beyond what's actually stated.
4. **Stale-index** — the retrieved content itself is outdated or has been superseded, but the index hasn't been updated (Module 3.6) — the system is faithfully grounding its answer in context that is simply wrong because it's old. This is neither a retrieval bug nor a generation bug in the usual sense — everything "worked correctly" against a stale snapshot of truth.

**Why this ordering matters as a debugging sequence**: each failure type requires ruling out the ones before it. You cannot productively debug generation (#2/#3) until you've confirmed retrieval (#1) succeeded, and you cannot conclude "hallucination" (#3) until you've confirmed the correct chunk was both retrieved *and* well-positioned (ruling out #1 and #2). Jumping straight to "let's fix the prompt" without checking retrieval first is the single most common debugging mistake candidates describe.

---

## 8.2 The debugging workflow: isolate retrieval before blaming generation

Given a bad answer, work strictly in this order:

**Step 1 — Inspect retrieved chunks directly.** Before looking at the generated answer at all, log and manually inspect exactly which chunks were retrieved (and in what order/rank) for the failing query. This single step resolves failure type #1 immediately: is the relevant chunk present anywhere in the retrieved set, yes or no?

**Step 2 — If the chunk is absent (failure type #1):** the problem is upstream, in embedding/chunking/indexing/retrieval strategy (Modules 1-4), not generation. Check:
- Does the query use rare terms/acronyms/exact IDs that dense retrieval would miss (Module 4.1)? → test with BM25 alone, see if sparse retrieval finds it when dense doesn't
- Is the relevant information split across multiple chunks awkwardly, such that no single chunk fully captures it (chunking issue, Module 2)?
- Is k too small (Module 5.6's retrieval-vs-reranking cutoff tradeoff)?

**Step 3 — If the chunk is present but the answer is still wrong (failure types #2 or #3):** the problem is downstream, in augmentation/generation (Module 6). Check:
- Where was the correct chunk positioned in the final assembled prompt — buried in the middle ("lost in the middle," #2)?
- Was the context cluttered with redundant/irrelevant chunks competing for attention (Module 6.3)?
- Does the generated claim actually trace back to *any* content in the context (run a faithfulness check, Module 7.3) — if not, this is #3, pure hallucination despite good context, and the fix is prompt-level (stronger grounding instructions) or model-level (a model less prone to overriding context with parametric knowledge).

**Step 4 — If the chunk is present, well-positioned, faithfully used, and *still* wrong:** check whether the source content itself is outdated (failure type #4, stale index) — verify against the current ground truth outside the system. If so, this is an ingestion-pipeline/index-freshness problem (Module 3.6), not a retrieval or generation bug at all.

**Interview framing**: this workflow is valuable to state explicitly because it demonstrates you don't randomly poke at prompts when something goes wrong — you have a systematic elimination process that narrows the failure to a specific pipeline stage before proposing a fix.

---

## 8.3 Common root causes, mapped to symptoms

| Symptom | Likely root cause | Module |
|---|---|---|
| Fails specifically on queries with acronyms/IDs/exact terms | Dense-only retrieval, no sparse/hybrid component | 4.1-4.3 |
| Fails specifically on paraphrased queries, works on queries that echo source wording | Embedding domain mismatch, or corpus never covered paraphrase variety in training | 1.7 |
| Answer references content but gets facts subtly wrong | Chunk boundary split a fact/table awkwardly, or context ordering buried the precise numeric detail | 2, 6.1 |
| Correct chunk retrieved (verified in logs) but ignored in the answer | "Lost in the middle" positioning, or context overloaded with irrelevant chunks | 6.1, 6.3 |
| Answer confidently states something no retrieved chunk supports | Hallucination despite context — weak grounding instructions, or a model that leans on parametric knowledge | 6.2, 6.5 (Self-RAG/CRAG as mitigations) |
| Answer was correct last month, wrong now, nothing else changed | Stale index — source document updated but not re-ingested/re-embedded | 3.6 |
| Retrieval quality degraded gradually over weeks with no code changes | IVF cluster centroids drifting from data distribution (Module 3.2/3.6), or query distribution shift not represented in original eval set | 3, 7.7 |
| Works fine on eval set, users report bad answers in production | Synthetic eval set doesn't represent real query distribution (Module 7.6's weakness) | 7.6, 7.7 |

---

## 8.4 Monitoring in production

Debugging a single bad case is reactive; production monitoring is proactive — catching degradation before it accumulates into a pattern of user complaints.

- **Retrieval latency** (p50/p95/p99, broken down by stage — embedding, ANN search, reranking, generation): a latency regression in one stage specifically points to that stage's infra (e.g. an index that's grown past its efficient operating range and needs resharding, Module 3.7)
- **Cache hit rate**: many production RAG systems cache embeddings for repeated/similar queries — a dropping cache hit rate can signal a shift in query diversity/distribution worth investigating on its own
- **Drift in query distribution**: track the statistical shape of incoming queries over time (e.g. topic clustering of queries, or embedding-space centroid drift) — a meaningful shift signals that your original golden eval set (Module 7.6) may no longer represent current real usage, and retrieval/chunking choices tuned against the old distribution may be quietly underperforming on the new one, even with zero code changes (directly connects to the "gradual degradation, no code changes" row in 8.3's table)
- **Faithfulness/relevance sampling in production**: periodically run the Module 7 LLM-judge metrics on a sample of live production traffic (not just the static offline eval set) to catch regressions the offline set wouldn't surface — this is the online evaluation practice from Module 7.7 applied specifically as an ongoing monitoring signal rather than a one-time A/B test

---

## Interview Q&A drill

**Q: A user reports a wrong answer. Walk me through exactly how you'd debug it, in order.**
A: First, pull the logged retrieved chunks for that exact query before looking at the generated answer at all — this immediately tells me whether the relevant information was fetched. If it's absent, the problem is upstream in retrieval (check whether it's a dense-embedding blind spot on rare terms, a chunking issue splitting the needed fact across chunks, or k being too small) — I wouldn't touch the prompt or generation config at this point, since the answer was never going to be correct regardless of generation quality. If the relevant chunk is present, I'd next check where it was positioned in the assembled context (lost-in-the-middle risk) and whether it was crowded out by redundant/irrelevant chunks. If it was well-positioned and still ignored or contradicted, I'd run a faithfulness check on the specific claim to confirm it's a genuine hallucination rather than a subtle misreading, and only then look at prompt-level grounding instructions or model choice. Finally, if everything upstream checks out and the answer is still wrong, I'd verify the source content itself isn't simply stale relative to current ground truth.

**Q: Retrieval quality has degraded gradually over the past month with no code or model changes. What are your top hypotheses?**
A: Two leading hypotheses, both consistent with "no code changes but gradual drift": first, if using an IVF-based index, the cluster centroids were trained on a data snapshot and haven't been retrained — as new documents are added over time, the actual data distribution drifts from those original centroids, degrading recall gradually without any explicit failure. Second, the incoming query distribution itself may have shifted (new topics, new phrasing patterns, a new user segment) in a way the original golden eval set doesn't represent — meaning the system's actual real-world performance was degrading even though it would still score fine against the now-outdated offline eval set. I'd check both: inspect index staleness/retraining schedule, and compare recent production query samples against the original eval set's topic/embedding distribution to look for drift.

**Q: How do you distinguish "hallucination despite good context" from "retrieval fetched the wrong context and the model just repeated it faithfully"?**
A: Both would look identical from the final answer alone — wrong information stated confidently. The distinguishing step is checking faithfulness specifically: does the wrong claim in the answer actually trace back to something stated in the retrieved context? If yes, this is a retrieval/content problem — the model was faithful to bad or stale evidence (points to stale-index or a chunking/retrieval issue, not a generation flaw). If no — the retrieved context doesn't contain or support the claim at all — that's pure hallucination despite good context, a generation-stage failure. This is exactly why faithfulness is measured as its own metric independent of overall answer correctness (Module 7.3): it isolates precisely this distinction.

**Q: What production signal would tell you your offline eval set has become stale, before users start complaining?**
A: Monitoring drift in the live query distribution — e.g. clustering incoming query embeddings over time and watching for the centroid or topic mix shifting away from what the original golden eval set represents. A meaningful shift means the system might be silently underperforming on a growing segment of real traffic even while continuing to score well on the (now unrepresentative) static offline eval set, since offline eval only measures performance against the queries it happens to contain. Combined with periodic LLM-judge faithfulness/relevance sampling directly on production traffic — not just the offline set — this surfaces regressions before they show up as user complaints.

---

**Next up: Module 9 — System design & interview synthesis (end-to-end walkthrough, scaling, security, advanced architectures, practice question bank).** Say the word when ready.
