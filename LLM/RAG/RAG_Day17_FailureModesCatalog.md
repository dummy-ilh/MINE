# RAG Interview Prep — Day 17
## Failure Modes Catalog

---

## 🚀 Quick Summary

This is a capstone day: a systematic catalog of everything that can go wrong across the RAG pipeline, organized by *where* in the pipeline the failure originates — because correctly localizing a failure is the entire prerequisite for fixing it (this is the exact principle Module 7's evaluation triad and Day 6/12's cross-week synthesis questions have been building toward). Most of these failure modes were touched on individually across Days 1–16; today's job is to organize them into one coherent map, and to give full treatment to two failure modes that deserve deeper focus than they've gotten so far: **over-reliance on parametric knowledge** (the model ignoring good retrieved context in favor of its own pretrained "knowledge") and **refusal miscalibration**, framed properly as a two-sided error problem.

**Think of it like a doctor's differential diagnosis checklist.** A patient with a symptom ("the RAG system gave a wrong answer") could have many different underlying causes, and guessing wrong wastes time treating the wrong thing. A good differential diagnosis process — systematically ruling stages in or out — is exactly what a failure mode catalog gives you: a structured way to ask "is this a retrieval problem, a context-assembly problem, or a generation problem" before reaching for a fix.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Parametric knowledge** | What a model "knows" from pretraining, baked into its weights (Day 1) |
| **Knowledge conflict** | When retrieved context contradicts the model's parametric knowledge, and the model must choose which to trust |
| **Over-reliance on parametric knowledge** | The failure mode where a model answers from its pretrained knowledge instead of the provided (and correct) retrieved context |
| **Context dilution** | Retrieved context containing enough irrelevant/noisy content to degrade generation quality, even when relevant content is technically present |
| **Refusal miscalibration** | Either over-refusing (declining answerable questions) or under-refusing (confidently answering unanswerable ones) |
| **Error propagation** | A failure originating early in a pipeline (e.g., one bad hop in multi-hop retrieval) that corrupts everything built on top of it |

---

# PHASE 1 — The Master Failure Taxonomy

## Organizing principle: which stage, and what specifically broke

```
   RETRIEVAL STAGE          CONTEXT ASSEMBLY STAGE        GENERATION STAGE
   (did we find the          (did we present it            (did the model use
    right evidence?)          well?)                        it correctly?)
        │                         │                              │
        ▼                         ▼                              ▼
  • Low recall              • Lost-in-the-middle          • Hallucination despite
    (chunking/embedding)      (Day 13)                       good context
  • Vocabulary mismatch     • Context dilution/noise      • Over-reliance on
    (Day 7/11)                (too many marginal            parametric knowledge
  • Redundant/duplicate       chunks)                        (today's deep dive)
    candidates (Day 14)     • Truncation from budget      • Citation fabrication
                               overrun (Day 13/14)            (Day 15)
                                                            • Refusal miscalibration
                                                              (Day 15, deep dive today)

              CROSS-CUTTING / INFRASTRUCTURE FAILURES (can affect any stage)
   • Embedding drift (Day 2)     • Stale index / centroid drift (Day 4/5)
   • Cache staleness (Day 14)    • Error propagation in multi-hop (Day 16)
```

**Why organizing it this way matters in an interview:** when asked "your RAG system gave a wrong answer, how do you debug it," the strong response doesn't jump straight to a fix — it walks through this taxonomy systematically, which is exactly the diagnostic instinct Module 7's Q&A drill (§"high Recall@k but poor faithfulness") was training. This taxonomy is essentially that diagnostic workflow, generalized and made comprehensive.

---

# PHASE 2 — Deep Dive on the Two Under-Covered Failure Modes

## 1. Over-Reliance on Parametric Knowledge (Knowledge Conflict)

**The mechanism:** an LLM's pretrained weights encode a huge amount of general knowledge — and when retrieved context is provided, the model is supposed to prioritize that context over its own pretrained "beliefs." But this doesn't always happen reliably, especially when the model's parametric knowledge about a topic is *strong and confident* (a well-known, widely-repeated fact from training data) and the retrieved context contradicts it (e.g., because the real-world fact has genuinely changed since the model's training, or because the retrieved context describes a specific, non-default case that differs from the "common" case the model learned during pretraining).

**Worked example:**
```
Retrieved context (accurate, current): "As of the policy update in
March 2024, AirPods Pro returns are now accepted within 30 days of
purchase, extended from the previous 14-day window."

Model's parametric knowledge (from pretraining data, now outdated):
strongly associates "Apple return policy" with "14 days" because
that fact appeared far more frequently and consistently across the
model's training data than this specific, more recent policy change.

Generated answer (a knowledge-conflict failure): "AirPods Pro can
be returned within 14 days of purchase." ← WRONG — the model
reverted to its strong parametric prior instead of using the
correctly-retrieved, current context that was right there in the
prompt.
```
**Why this is a distinct failure mode from generic hallucination:** generic hallucination (Day 15) is the model inventing content not present in the context *at all*. Over-reliance on parametric knowledge is subtler and arguably more dangerous — the model *had* the correct information available in context, but chose (implicitly, as a byproduct of how strongly it learned the "default" fact during training) to answer from memory instead. This is a failure of *prioritization*, not just of *information availability* — it can happen even with a perfect retrieval and context construction pipeline, which is exactly why it deserves separate treatment from retrieval-stage or context-assembly-stage failures.

**How this is detected in practice — counterfactual/perturbation testing:** deliberately construct eval examples where retrieved context intentionally *contradicts* well-known parametric facts (e.g., artificially stating a changed policy, a different numeric spec, or a corrected historical detail), and check whether the model's generated answer follows the provided context or reverts to the well-known "default" answer. This is a specific, deliberate eval design pattern worth naming — a generic golden eval set (Module 7 §7.6) built only from "normal" questions likely won't surface this failure mode at all, since it only manifests specifically when context contradicts a *strong* prior, which needs to be deliberately engineered into the eval set to test for.

**Mitigation approaches:**
- **Stronger prompt instructions** emphasizing recency/specificity of provided context over general knowledge (e.g., "the provided context reflects the most current information and should always be prioritized over general knowledge you may have").
- **Fine-tuning** specifically on knowledge-conflict examples (again, Day 1's "fine-tune for behavior/skill" pattern) — teaching the model, as a learned behavior, to consistently defer to provided context even when it contradicts a strong parametric prior.
- **Explicit context-recency signaling** — if retrieved chunks carry metadata like publication/update date (Day 5's metadata filtering infrastructure), surfacing that date explicitly in the prompt can help the model recognize "this is more current than what I might otherwise assume."

> **Why This Matters callout:** This failure mode is a favorite "do you actually understand RAG deeply, or just the textbook version" interview probe, because it's genuinely counter-intuitive — most people assume providing correct context in the prompt is sufficient to guarantee a correct answer, and this failure mode is the concrete counterexample: correct context isn't sufficient if the model's parametric prior is strong enough to compete with it.

---

## 2. Refusal Miscalibration — The Full Two-Sided Error Matrix

Day 15 introduced refusal calibration; today's treatment frames it properly as a **binary classification problem with two distinct error types**, which is the more rigorous way to reason about it in an interview.

```
                            ACTUAL GROUND TRUTH
                     Context IS sufficient    Context is NOT sufficient
                     ────────────────────────────────────────────────────
System ANSWERS   │   ✓ Correct                │   ✗ FALSE ANSWER
                 │   (helpful, correct)        │   (hallucination risk —
                 │                             │    the worst failure mode)
System REFUSES   │   ✗ FALSE REFUSAL          │   ✓ Correct
                 │   (unhelpful — the system   │   (appropriately declines)
                 │    could have answered)     │
```

**Why this framing matters:** this is exactly a confusion-matrix / precision-recall framing, and articulating it this way in an interview signals you're thinking about refusal calibration as a genuine tunable decision boundary (like any classifier threshold), not just a vague "be more careful" instruction. The two error types have very different costs depending on the domain:
- **False refusal** (declining an actually-answerable question) — an unhelpfulness cost, generally lower-stakes.
- **False answer** (confidently answering when context is insufficient) — a correctness/trust cost, generally higher-stakes, especially in domains like medical/legal/financial (echoing Day 15's medical RAG system design answer).

**The threshold-tuning implication:** just as a classifier's decision threshold can be moved to trade recall against precision, a RAG system's refusal-confidence threshold can be tuned to trade false-refusals against false-answers — and the *correct* operating point depends on the relative cost of each error type for your specific domain, not a universal default. A medical RAG system should sit at a very different point on this trade-off curve than a casual internal FAQ bot, deliberately accepting more false refusals to avoid false answers.

**Worked example connecting this to Day 15's eval-set point:** to actually measure where your system sits on this trade-off (not just assume it), you need a golden eval set (Module 7 §7.6) with *known* ground truth for both quadrants — a set of queries with genuinely sufficient context (to measure false-refusal rate) AND a set of queries with genuinely insufficient context (to measure false-answer rate) — measuring only one type of query gives you visibility into only one of the two error types, leaving you blind to the other.

---

# PHASE 3 — Full Failure Mode Reference Table (Master Summary)

| Failure mode | Stage | Symptom | Root cause | Primary detection signal | Primary fix |
|---|---|---|---|---|---|
| **Low recall** | Retrieval | Relevant doc never surfaces in top-k | Bad chunking, wrong embedding model, k too small | Recall@k | Chunk-size sweep (Day 3), domain-specific embeddings (Day 2), increase k with rerank |
| **Vocabulary mismatch** | Retrieval | Relevant doc exists but isn't retrieved due to phrasing gap | Query and document phrased very differently | Recall@k on paraphrased queries specifically | Hybrid search (Day 9), query transformation/HyDE (Day 11) |
| **Redundant candidates** | Retrieval | Context wastes budget on near-duplicate chunks | Chunk overlap (Day 3), multi-query overlap (Day 11) | Manual inspection, embedding-similarity clustering of candidates | Deduplication (Day 14) |
| **Lost-in-the-middle** | Context assembly | Right evidence retrieved but not used correctly | Poor chunk ordering in the final prompt | nDCG improving but faithfulness not | Sandwiching / reorder by relevance (Day 13) |
| **Context dilution** | Context assembly | Generator struggles despite relevant doc being present | Too many marginal/irrelevant chunks included | Context relevance (Module 7) | Lower k, better reranking (Day 10), extractive compression (Day 14) |
| **Truncation** | Context assembly | Critical content silently cut off | Budget mismanagement, no reserved generation headroom | Manual inspection of actual prompts sent | Explicit budget allocation (Day 13) |
| **Hallucination (pure)** | Generation | Answer contains claims absent from any retrieved chunk | Weak grounding instructions, no runtime check | Faithfulness (Module 7) / groundedness guardrail failing | Runtime groundedness guardrail (Day 15), stronger prompt instructions |
| **Over-reliance on parametric knowledge** | Generation | Answer contradicts correct retrieved context, matches a "default"/well-known fact instead | Strong pretrained prior competing with context | Counterfactual/perturbation eval (today) | Fine-tuning on conflict examples, recency signaling (today) |
| **Citation fabrication** | Generation | Citation marker present but doesn't actually support the claim | Model self-reports citations without verification | Post-hoc attribution mismatch rate | Post-hoc attribution instead of inline citation (Day 15) |
| **Refusal miscalibration** | Generation | Either unhelpful over-refusal or confident under-refusal | Poorly calibrated confidence threshold | False-refusal rate AND false-answer rate, measured separately | Threshold tuning against a two-sided golden eval set (today) |
| **Error propagation** | Multi-hop/agentic | Confidently wrong multi-hop answer built on an early bad fact | No validation of intermediate hop results | Manual trace inspection of hop-by-hop reasoning | Reflection/Self-RAG-style validation per hop (Day 16) |
| **Embedding drift** | Cross-cutting | Similarity search behaves inconsistently/degraded after a model change | Mixing vector spaces from different embedding model versions | Sudden unexplained recall drop after a deployment | Full corpus re-embedding with migration strategy (Day 2) |
| **Stale index/cache** | Cross-cutting | System returns outdated info despite a corpus update | Centroid drift (IVF), un-invalidated semantic cache | Discrepancy between source-of-truth and served answer | Re-clustering (Day 4), cache invalidation tied to updates (Day 14) |

---

# PHASE 4 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What's the difference between generic hallucination and over-reliance on parametric knowledge?

<details>
<summary>Show answer</summary>

Generic hallucination is the model inventing content that isn't present in the retrieved context at all — pure fabrication. Over-reliance on parametric knowledge is subtler: the correct information *was* available in the retrieved context, but the model answered from its own pretrained "default" knowledge instead, typically because that parametric fact was learned very strongly and consistently during training, competing with and overriding the (correct, but perhaps less frequently-seen-in-training-data) retrieved context.
</details>

---

**Q2 (Easy — conceptual).** Why can't a standard golden eval set (built from typical questions) reliably surface over-reliance on parametric knowledge as a failure mode?

<details>
<summary>Show answer</summary>

This failure mode only manifests specifically when retrieved context contradicts a *strong* parametric prior — a normal eval set built from typical questions, where context and parametric knowledge usually agree, gives the model no opportunity to reveal this failure. Detecting it requires deliberately constructing counterfactual/perturbation test cases where context is engineered to contradict well-known facts, specifically to check whether the model follows the context or reverts to its prior.
</details>

---

**Q3 (Medium — conceptual, confusion-matrix framing).** Frame refusal calibration as a two-error-type problem, and explain why the "correct" threshold isn't universal across domains.

<details>
<summary>Show answer</summary>

Refusal decisions have two possible error types: false refusal (declining a question that was actually answerable from context — an unhelpfulness cost) and false answer (confidently answering when context was actually insufficient — a correctness/trust cost, generally more severe). Because these two error types have different costs, and that cost trade-off varies by domain (e.g., a wrong medical answer is far more costly than an unhelpful refusal, while a casual internal tool might tolerate more false answers in exchange for fewer annoying refusals), the correct operating point on this trade-off is domain-specific, not a universal default — analogous to how a classifier's decision threshold should be set based on the relative cost of false positives vs. false negatives for the specific application.
</details>

---

**Q4 (Medium — conceptual).** A RAG system exhibits high faithfulness scores on your evaluation set but users still occasionally report confidently wrong answers about recently-changed policies. What failure mode does this suggest, and why might it not show up in standard faithfulness metrics?

<details>
<summary>Show answer</summary>

This is a strong signal of over-reliance on parametric knowledge specifically around recently-changed information — a knowledge-conflict failure. It might not show up in standard faithfulness metrics if the eval set doesn't specifically include recently-changed-policy examples that contradict the model's strong pretrained prior; faithfulness scores measured on a general eval set can look high overall while this specific, narrow but high-impact failure slips through, since it only manifests under the specific condition of a context-vs-parametric-prior conflict, not under typical eval conditions.
</details>

---

**Q5 (Medium — diagnostic reasoning).** Retrieval Recall@k is high, context relevance is high, but faithfulness is low. Using today's taxonomy, which stage does this point to, and what specific generation-stage failure modes would you investigate first?

<details>
<summary>Show answer</summary>

High recall and high context relevance rule out retrieval-stage and most context-assembly-stage problems (the right, relevant evidence is present and not diluted with noise) — this points squarely at the generation stage. I'd investigate, in order: (1) pure hallucination — is the model inventing claims unsupported by the good context it was given; (2) over-reliance on parametric knowledge — is the context perhaps being contradicted by a strong prior on specific claims; (3) citation fabrication if citations are part of the output — is the model citing sources that don't actually support the claim next to them. All three are generation-stage failures that can coexist with excellent retrieval and context quality, which is exactly the diagnostic signature this metric combination points to.
</details>

---

**Q6 (Hard — system design synthesis).** Design an evaluation and monitoring strategy specifically to catch both over-reliance on parametric knowledge and refusal miscalibration in production, given that standard eval sets don't reliably surface either.

<details>
<summary>Show answer</summary>

For over-reliance on parametric knowledge: build a dedicated counterfactual eval slice — deliberately constructed examples where retrieved context is engineered to contradict well-known, high-confidence facts (recently-changed policies, updated specs, corrected historical details), and measure the rate at which generated answers correctly follow the provided context vs. revert to the parametric default. This needs to be a separate, deliberately-constructed eval slice, not something a general golden set will surface incidentally. For refusal miscalibration: build a two-sided eval set with a known-sufficient-context slice (to measure false-refusal rate) and a known-insufficient-context slice (to measure false-answer rate), tracking both rates separately rather than a single aggregate "refusal accuracy" number that could hide an imbalance between the two error types. In production, I'd also implement lightweight ongoing monitoring — sampling live traffic for cases where context appears to contain contradicting-to-common-knowledge information (a heuristic trigger) and periodically auditing model behavior on those specific samples, since this failure mode's low base rate in typical eval sets means production monitoring is a necessary supplement to offline evaluation, not a redundant afterthought.
</details>

---

**Q7 (Hard — full pipeline diagnosis).** A RAG system gives a wrong answer to a multi-hop question. Walk through, using today's full taxonomy, the systematic order in which you'd investigate potential causes.

<details>
<summary>Show answer</summary>

I'd start at the earliest pipeline stage and work forward, since an error at an early stage can masquerade as a failure at a later stage (and fixing a later stage first would be wasted effort if the real problem is upstream): (1) Retrieval stage — check whether each individual hop's retrieval actually surfaced relevant documents (Recall@k per hop, if traceable) — a vocabulary mismatch or chunking issue at any single hop could be the root cause; (2) Context assembly — check whether each hop's retrieved content was well-constructed into that hop's reasoning context, watching for lost-in-the-middle or dilution within any individual hop's context; (3) Error propagation specifically (Day 16's multi-hop-specific failure) — trace through the hop-by-hop reasoning chain to identify whether an early hop's retrieved fact was actually correct, since a correct final-hop retrieval built on an incorrect earlier-hop fact will still produce a wrong final answer despite the last hop "working correctly" in isolation; (4) Generation-stage failures at the final synthesis step — hallucination, over-reliance on parametric knowledge, or refusal miscalibration in how the final answer was assembled from all the gathered hop observations. This ordered, stage-by-stage walk-through — rather than jumping straight to "the model hallucinated" — is what a systematic failure-mode taxonomy actually buys you in a live debugging conversation.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Assuming correct retrieved context guarantees a correct answer — over-reliance on parametric knowledge is the concrete counterexample.
- ❌ Testing refusal calibration with only "should answer" or only "should refuse" examples — you need both to see both error types.
- ❌ Treating "the model hallucinated" as a catch-all diagnosis without distinguishing pure fabrication from parametric-knowledge override, citation fabrication, or upstream error propagation — these have different fixes.
- ❌ Debugging a multi-hop failure by only inspecting the final generation step, missing that an early hop's bad fact is the actual root cause.
- ❌ Assuming a single aggregate "accuracy" or "faithfulness" number is sufficient monitoring — it can hide an imbalance between distinct error types (false refusal vs. false answer) that need separate tracking.
- ❌ Not maintaining a deliberately-constructed counterfactual eval slice — standard eval sets systematically fail to surface knowledge-conflict failures.

---

# 📌 Cheat Sheet (Day 17)

**The taxonomy:** Retrieval (did we find it?) → Context assembly (did we present it well?) → Generation (did the model use it correctly?) → Cross-cutting infrastructure issues (drift, staleness) that can corrupt any stage.

**Over-reliance on parametric knowledge:** correct context present, model still answers from a strong pretrained default instead — a prioritization failure, not an information-availability failure. Detect via deliberately-constructed counterfactual eval slices; fix via fine-tuning on conflict examples and recency signaling.

**Refusal miscalibration:** a two-error-type classification problem — false refusal (unhelpful) vs. false answer (worse, especially in high-stakes domains). Needs a two-sided eval set to measure both error rates separately; correct threshold is domain-specific, not universal.

**Diagnostic discipline:** always work stage-by-stage from earliest to latest when debugging a failure — an early-stage error can masquerade as a later-stage symptom, and fixing downstream first wastes effort if the root cause is upstream (especially critical in multi-hop error propagation).

**Golden interview line:** *"A wrong answer could originate at any stage — retrieval, context assembly, or generation — and two of the most easily-missed failure modes, over-reliance on parametric knowledge and refusal miscalibration, both require deliberately-constructed eval slices to even detect, because they don't reliably show up in a standard golden eval set built from typical questions."*

---

*Generation week's teaching content complete. Next up — Day 18: Review Day (cold Q&A across Days 13-17, integrated with Retrieval and Foundations weeks).*
