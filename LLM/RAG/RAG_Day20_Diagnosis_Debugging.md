# RAG Interview Prep — Day 20
## Diagnosis & Debugging

---

## 🚀 Quick Summary

Every prior day has built toward this one: Module 7 opened by explaining *why* retrieval and generation need separate metrics, Day 17 cataloged every failure mode by pipeline stage, and today assembles both into an actual **systematic debugging workflow** — a repeatable process for going from "a user reported a wrong answer" to "here's the specific stage and specific root cause" without guessing. This is the single most practical, most interview-relevant day in the curriculum, because "walk me through how you'd debug this" is one of the most common ways Apple (and most companies) actually test RAG systems knowledge in an interview — not "define BM25," but "here's a bug, find it."

**Think of it like an ER doctor's triage protocol, not a general checkup.** You don't run every possible test on every patient — you follow a structured sequence: check the most likely, cheapest-to-rule-out things first, narrow down based on what you find, and only escalate to expensive/detailed tests once cheaper checks have narrowed the search space. A good RAG debugging workflow works the same way: cheap, high-information checks first (did we even retrieve the right chunk?), before expensive ones (do we need to re-examine the embedding model?).

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Diagnostic workflow** | A structured, ordered sequence of checks used to localize a failure to a specific pipeline stage |
| **Observability / tracing** | Logging what actually happened at each pipeline stage for a given query, so it can be inspected after the fact |
| **Differential diagnosis** | Systematically ruling stages/causes in or out based on symptoms, rather than jumping to the first plausible explanation |
| **Regression testing** | Re-running the golden eval set after a fix to confirm the problem is resolved and nothing else broke |
| **Root cause vs. symptom** | The actual originating failure vs. where its effects become visible (often several stages downstream) |

---

# PHASE 1 — The Diagnostic Workflow (The Core Algorithm)

## The master decision tree

```
                    "The RAG system gave a wrong/bad answer"
                                    │
                                    ▼
              STEP 1: Was the RIGHT EVIDENCE ever retrieved?
              (inspect the raw top-k retrieval output, before
               reranking/context construction — Day 4's Recall@k
               question, applied to this ONE specific query)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                     ▼
                 NO                                    YES
       (retrieval-stage problem)              STEP 2: Was the evidence
                  │                            ORDERED/PRESENTED well?
                  ▼                            (inspect the actual
     Check: chunking (Day 3), embedding        constructed prompt sent
     model fit (Day 2), query-vocabulary       to the generator)
     mismatch (Day 7/11), k too small                    │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                             NO                      YES
                                    (context-assembly           STEP 3: Did the
                                     problem)                    GENERATOR use
                                              │                  the evidence
                                              ▼                  correctly?
                                Check: lost-in-the-middle       (compare the
                                (Day 13), context dilution,     answer's claims
                                truncation (Day 13/14)          against the
                                                                 actual context
                                                                 it was given)
                                                                          │
                                                          ┌───────────────┴───────────────┐
                                                          ▼                                 ▼
                                                         NO                                YES
                                              (generation-stage problem)          System is working —
                                                          │                        the "problem" may be
                                                          ▼                        a data/corpus issue
                                    Check: hallucination, over-reliance           (the source content
                                    on parametric knowledge, citation              itself was wrong/
                                    fabrication, refusal miscalibration            outdated) — a
                                    (all Day 17)                                   different fix entirely
```

**Why this specific ordering (retrieval → context assembly → generation) matters:** this is a deliberate cheapest-and-most-diagnostic-first ordering. Checking "was the right chunk even retrieved" is a fast, cheap, unambiguous check (just look at the raw top-k list) — and if it fails here, there's no point spending time inspecting generation behavior, since the model was never given a fair chance to succeed. Working forward through the pipeline in order avoids the common mistake of jumping straight to "the model hallucinated" (a generation-stage assumption) when the actual root cause might be sitting one or two stages earlier.

---

## Step 1 in Detail — Retrieval-Stage Diagnostic Checklist

**The core question:** for this specific failing query, does the raw (pre-reranking) top-k retrieval output actually contain a chunk that could answer the question?

| If... | Likely cause | Which day's fix applies |
|---|---|---|
| The relevant chunk is **nowhere** in top-k, even with a generous k | Embedding model mismatch, bad chunking, or genuinely rare vocabulary | Day 2 (domain-specific embeddings), Day 3 (chunk-size sweep) |
| The relevant chunk exists in the corpus but retrieval consistently misses it for **paraphrased** versions of the query | Vocabulary/style mismatch between query and document | Day 7 (add sparse/hybrid), Day 11 (HyDE, multi-query) |
| The relevant chunk is split awkwardly, with only half the needed info in any single chunk | Chunking boundary problem | Day 3 (overlap tuning, structure-aware/small-to-big chunking) |
| An **exact identifier** (SKU, error code) query fails specifically | Dense-only retrieval structurally weak on exact match | Day 7 (ensure hybrid search includes BM25) |
| The relevant chunk WAS in the corpus a week ago but seems to have vanished from retrieval now | Embedding drift after a model change, or index staleness | Day 2 (embedding drift), Day 4/5 (stale index/centroids) |

---

## Step 2 in Detail — Context-Assembly Diagnostic Checklist

**The core question:** given that the right chunk WAS retrieved, was it actually presented to the generator in a way that gives it a fair chance of being used?

| If... | Likely cause | Which day's fix applies |
|---|---|---|
| The relevant chunk was retrieved but placed in the **middle** of a long context | Lost-in-the-middle, poor chunk ordering | Day 13 (sandwiching, reorder by rerank score) |
| The relevant chunk is present but surrounded by many irrelevant chunks | Context dilution — k too high, or reranking not being applied/working | Day 10 (reranking), Day 13 (lower k, tune against generation metrics) |
| The relevant chunk seems to have been **cut off mid-sentence** in the actual prompt | Context window budget mismanagement, no reserved generation headroom | Day 13 (explicit budget allocation) |
| Multiple near-identical copies of the same chunk are crowding out other evidence | Chunk overlap redundancy, or multi-query fusion surfacing duplicates | Day 3 (overlap tuning), Day 14 (deduplication) |

---

## Step 3 in Detail — Generation-Stage Diagnostic Checklist

**The core question:** given that the right evidence WAS retrieved and WAS well-presented, did the model actually use it correctly?

| If... | Likely cause | Which day's fix applies |
|---|---|---|
| The answer contains claims **absent from any retrieved chunk** entirely | Pure hallucination | Day 15 (runtime groundedness guardrail, stronger instructions) |
| The answer **contradicts** correct retrieved context, matching a well-known "default" fact instead | Over-reliance on parametric knowledge | Day 17 (fine-tune on conflict examples, recency signaling) |
| A citation marker is present but doesn't actually support the claim next to it | Citation fabrication (inline self-reported citation) | Day 15 (switch to post-hoc attribution) |
| The system confidently answered a question it should have declined | False-answer refusal miscalibration | Day 15/17 (recalibrate threshold using a two-sided eval set) |
| The system declined a question it could have reasonably answered | False-refusal miscalibration | Day 15/17 (recalibrate threshold — likely set too conservatively) |

---

## The "None of the Above" Case — Corpus/Data Problems

**A critical, easy-to-miss branch:** sometimes every pipeline stage worked *correctly* — the right chunk was retrieved, well-presented, and faithfully used by the generator — but the answer is still wrong because **the source document itself was wrong or outdated**. This isn't a RAG pipeline bug at all — it's a data quality / corpus freshness problem, and the fix is entirely different (content team correction, source document update, ingestion pipeline audit) rather than anything covered in Days 1-19's technical toolkit.

> **Why This Matters callout:** Explicitly naming this branch in an interview is a strong signal — it shows you're not treating "the RAG system" as an isolated black box, but understanding it as one component of a larger system that includes the underlying content itself, which can be wrong independent of anything the ML system did.

---

# PHASE 2 — Observability: What to Actually Log

**The prerequisite for all of the above:** none of this diagnostic workflow is possible without adequate logging/tracing at each stage. A production RAG system needs to log, per query, at minimum:

1. **The raw user query** (and any query transformation applied — Day 11's rewritten variants, HyDE's hypothetical document, decomposition sub-questions).
2. **Raw first-stage retrieval results** — the actual top-k chunks and their scores, *before* reranking, so Step 1's diagnostic question can even be answered.
3. **Post-reranking order** — the final chunk ordering after Day 10's reranking step, so you can distinguish "reranking helped/hurt" from "first-stage retrieval was already wrong."
4. **The actual constructed prompt** sent to the generator — not a reconstruction or approximation, the literal text, so lost-in-the-middle/truncation/budget issues (Step 2) can be directly inspected rather than inferred.
5. **The generated answer**, plus any citation markers or attribution metadata.
6. **Any runtime guardrail results** (Day 15's groundedness check pass/fail, refusal decision and its confidence score).

**Why this matters in practice:** without logging item #4 specifically (the actual constructed prompt, not just "which chunks were retrieved"), you cannot diagnose lost-in-the-middle or truncation issues at all — you'd be guessing at what the model actually saw, rather than inspecting it directly. This is a common, costly gap in real production systems — teams often log retrieval results but not the final assembled prompt, making Step 2 of the diagnostic workflow impossible to execute properly.

---

# PHASE 3 — Worked Full Diagnostic Walkthrough

**Bug report:** "Users ask about AirPods Pro battery life and sometimes get told '4 hours' instead of the correct '6 hours.'"

```
STEP 1 — Was the right evidence retrieved?
  → Inspect raw top-k for a sample of failing queries.
  → FINDING: the correct chunk ("6 hours...") IS present in the
    top-3 raw retrieval results for every failing query sampled.
  → Retrieval stage: RULED OUT. Move to Step 2.

STEP 2 — Was it presented well?
  → Inspect the actual constructed prompt for failing queries.
  → FINDING: the correct chunk is present, positioned 2nd out of 5
    chunks — not buried in an extreme middle position, and total
    context is well within budget (no truncation).
  → Context-assembly stage: RULED OUT (or at least, not the primary
    driver). Move to Step 3.

STEP 3 — Did the generator use it correctly?
  → Compare the actual generated answer's claim against the
    provided context, sampled across failing queries.
  → FINDING: an OLDER support document, also present in the corpus
    (from an earlier product generation), states "4 hours" for a
    DIFFERENT/older AirPods model — and it's ALSO being retrieved
    in some of the failing cases, sitting in the context alongside
    the correct "6 hours" chunk.
  → FURTHER FINDING: on samples where BOTH chunks are present, the
    model sometimes picks the wrong ("4 hours") one — suggesting
    the model isn't reliably distinguishing WHICH product each
    chunk actually refers to when both are present in context
    together.

DIAGNOSIS: This is actually a hybrid retrieval-stage AND generation-
stage issue: (a) the retriever is surfacing a chunk about a
DIFFERENT product as if relevant (an under-specified retrieval
issue — the older-model chunk shouldn't rank so closely for an
"AirPods Pro" specific query, suggesting either a chunking issue
where product-identifying context isn't well-preserved per chunk,
or an embedding-model issue insufficiently distinguishing between
product generations), and (b) even when both chunks are present,
the generator isn't reliably disambiguating which one is actually
about the queried product (a generation-stage attribution problem).

FIX: (1) Improve chunk metadata/structure (Day 3, Day 5) to ensure
product-identifying context (e.g., "AirPods Pro" vs "AirPods 2nd
gen") is explicitly attached to every chunk, improving retrieval
precision for product-specific queries; (2) strengthen prompt
instructions (Day 15) to explicitly require the model to verify
which specific product each piece of context refers to before
using it, rather than assuming the first battery-life-shaped
number found is the right one.
```

**Why this worked example matters:** it demonstrates that real bugs often aren't cleanly localized to exactly one stage — this case genuinely spans a retrieval-precision issue (Step 1 didn't cleanly rule retrieval out on closer inspection — the *presence* of the right chunk isn't the same as the *absence* of a misleadingly similar wrong chunk) and a generation-stage disambiguation issue (Step 3). The diagnostic workflow's value isn't that every bug resolves in exactly one step — it's that working through the steps in order surfaces this compound picture systematically, rather than stopping at the first plausible-looking explanation.

---

## Regression Testing — Confirming a Fix Actually Worked

**After implementing a fix, don't just re-test the original failing query** — run the **full golden eval set** (Module 7 §7.6, expanded with Day 19's additional slices) to confirm two things:
1. The specific failure mode is actually resolved (not just for the one reported example, but across the relevant eval slice).
2. **No regression** was introduced elsewhere — e.g., the chunk-metadata change from the worked example above should be checked against the *entire* eval set, not just AirPods Pro battery queries, since metadata/chunking changes can have unintended effects on unrelated query types.

This is the same offline-eval-as-pre-filter principle from Module 7 §7.7, now applied specifically to fix-validation rather than new-feature validation — and it's worth stating explicitly in an interview that "fixing a bug" isn't complete until you've confirmed it via the eval set, not just anecdotally re-checking the original complaint.

---

# PHASE 4 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What's the logical justification for diagnosing retrieval before context assembly, and context assembly before generation, rather than checking generation behavior first?

<details>
<summary>Show answer</summary>

This is a cheapest-and-most-diagnostic-first ordering: checking whether the right chunk was even retrieved is a fast, unambiguous check (just inspect the raw top-k list), and if it fails here, there's no point spending time analyzing generation behavior — the model was never given a fair chance to succeed regardless of how well it might have used good evidence. Working forward through the pipeline in order avoids the common mistake of assuming "the model hallucinated" when the actual root cause is one or two stages earlier.
</details>

---

**Q2 (Easy — conceptual).** Why is logging the actual constructed prompt (not just which chunks were retrieved) essential for diagnosing lost-in-the-middle or truncation issues?

<details>
<summary>Show answer</summary>

Lost-in-the-middle and truncation are about the *specific position and completeness* of content within the literal text sent to the generator — you cannot verify chunk ordering or check whether content was cut off by inspecting only "which chunks were retrieved," since that doesn't capture the actual final arrangement or whether budget constraints truncated anything. Without logging the literal constructed prompt, you'd be guessing at what the model actually saw rather than directly inspecting it.
</details>

---

**Q3 (Medium — diagnostic reasoning).** A failing query has the correct chunk present in raw top-k retrieval AND well-positioned in the constructed prompt, but the generated answer is still wrong, containing a claim not present in any retrieved chunk. Which stage, and which specific failure mode from Day 17?

<details>
<summary>Show answer</summary>

Generation stage — and specifically, since the claim is absent from any retrieved chunk entirely (not contradicting correct context, just fabricated), this points to pure hallucination rather than over-reliance on parametric knowledge (which would specifically involve the answer contradicting correct context that WAS present) or citation fabrication (which is about mismatched citations, not necessarily fabricated content itself). Fix: Day 15's runtime groundedness guardrail and stronger grounding instructions.
</details>

---

**Q4 (Medium — conceptual).** Why should a fix be validated against the full golden eval set rather than just the originally-reported failing query?

<details>
<summary>Show answer</summary>

Re-testing only the original failing query confirms the specific symptom is gone, but doesn't confirm the fix generalizes correctly to the broader relevant eval slice, and critically doesn't check for regressions — unintended negative effects on other, previously-working query types that the fix might have inadvertently broken (e.g., a chunking/metadata change aimed at one product category could affect retrieval behavior for unrelated categories). Running the full golden eval set catches both the intended fix's effectiveness and any unintended side effects in one pass.
</details>

---

**Q5 (Medium — conceptual).** What's the "none of the above" diagnostic branch, and why is it easy to miss?

<details>
<summary>Show answer</summary>

Sometimes every pipeline stage — retrieval, context assembly, generation — worked correctly, but the answer is still wrong because the underlying source document itself was wrong or outdated. It's easy to miss because the natural instinct when debugging a "RAG system" is to assume the bug lives somewhere in the RAG pipeline's technical stages, rather than considering that the pipeline may be faithfully and correctly surfacing genuinely bad source content — a data quality/corpus problem requiring a completely different fix (content correction, not model/retrieval tuning).
</details>

---

**Q6 (Hard — full walkthrough synthesis).** Walk through, in order, how you would debug a report that a multi-hop agentic RAG system (Day 16) gives wrong answers specifically on 3+ hop questions, but performs fine on 1-2 hop questions.

<details>
<summary>Show answer</summary>

I'd first isolate whether the failure is uniform across all 3+ hop questions or specific to certain hop patterns, then trace individual failing examples hop-by-hop rather than only inspecting the final answer: for each hop, check (a) was the right evidence retrieved for that specific hop's sub-query (Step 1's retrieval check, applied per-hop), (b) was that hop's retrieved context well-constructed (Step 2, per-hop), and (c) was the intermediate fact/conclusion drawn from that hop actually correct (Step 3, per-hop) — since Day 16/17's error propagation means a wrong fact at hop 2 can produce a "generation stage looks fine" but ultimately wrong final answer, even though the final hop's own retrieval and generation behaved correctly in isolation. Given the specific pattern of "fine at 1-2 hops, breaks at 3+," I'd hypothesize this points toward either (a) accumulating error propagation that only becomes visible once enough hops compound, or (b) a stopping-criteria issue (Day 16) — the system might be terminating the loop prematurely once hop count grows, self-assessing "sufficient information" too early on longer chains specifically. I'd validate by checking whether 3+ hop failures show a pattern of correctly-retrieved-but-prematurely-synthesized answers (pointing to stopping criteria) vs. a specific wrong intermediate fact appearing consistently (pointing to error propagation from a specific weak hop).
</details>

---

**Q7 (Hard — meta-synthesis, capstone question).** An interviewer says: "Our RAG system's users report it sometimes gives confidently wrong answers about recently-changed company policies, but only occasionally, and our evaluation dashboard shows healthy faithfulness and recall scores. Walk me through your full diagnostic and fix process." Answer as you would in the actual interview.

<details>
<summary>Show answer</summary>

I'd start by noting that healthy aggregate faithfulness/recall scores don't rule out a narrow, high-impact failure mode that a standard eval set wouldn't surface — specifically, this symptom pattern ("recently-changed" information, "occasional," "confidently wrong" despite good context) is the classic signature of over-reliance on parametric knowledge (Day 17), not a generic retrieval or faithfulness problem, since it implies the correct updated information is likely being retrieved (consistent with healthy recall) but sometimes overridden by a strong pretrained prior at generation time. I'd walk the diagnostic workflow to confirm: Step 1, verify the updated-policy chunk is indeed present in top-k for the failing queries (likely yes, given healthy recall); Step 2, verify it's well-positioned in the constructed prompt (rule out lost-in-the-middle as a confound); Step 3, directly compare the generated answer against the actual provided context for failing samples — if the answer matches a well-known "old" fact instead of the provided updated one, that confirms knowledge conflict as the root cause. To fix: build a dedicated counterfactual eval slice (deliberately contradicting well-known facts) to reliably measure this specific failure rate going forward, since it's currently invisible to the existing eval dashboard; then apply mitigations — strengthened prompt instructions prioritizing provided context explicitly, recency metadata surfaced in the prompt (Day 5's metadata infrastructure), and potentially fine-tuning on knowledge-conflict examples if the issue persists after prompt-level fixes. Finally, I'd validate the fix against both the new counterfactual eval slice and the full existing eval set, to confirm the fix works without introducing regressions elsewhere.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Jumping straight to "the model hallucinated" without first ruling out retrieval and context-assembly stages — the single most common shortcut mistake in RAG debugging.
- ❌ Debugging without adequate logging of the actual constructed prompt — you cannot diagnose lost-in-the-middle or truncation from retrieval logs alone.
- ❌ Treating "the right chunk was retrieved" as sufficient to rule out retrieval-stage problems — a misleadingly similar *wrong* chunk being co-retrieved is also a retrieval-stage issue, even if the right chunk is also present.
- ❌ Fixing a bug and only re-testing the original failing example, without running the full golden eval set to check for regressions.
- ❌ Forgetting the "none of the above" branch — sometimes the pipeline is working correctly and the source data itself is simply wrong.
- ❌ Treating multi-hop failures as a single generation-stage problem rather than tracing hop-by-hop for error propagation.

---

# 📌 Cheat Sheet (Day 20)

**The workflow, in order:** (1) Was the right evidence retrieved? (raw top-k, pre-reranking) → (2) Was it presented well? (actual constructed prompt — check ordering, dilution, truncation) → (3) Did the generator use it correctly? (compare answer's claims against actual provided context) → (4) "None of the above" — is the source data itself wrong?

**Required logging:** raw query + transformations, raw first-stage retrieval + scores, post-rerank order, the literal constructed prompt (not a reconstruction), the generated answer + citations, runtime guardrail results.

**Compound failures are common:** real bugs often span more than one stage (worked example: a retrieval-precision issue AND a generation-disambiguation issue together) — don't stop at the first plausible explanation.

**Fix validation:** always re-run the full golden eval set after a fix, not just the originally-reported query — checks both effectiveness and regressions.

**Golden interview line:** *"I'd work forward through the pipeline — retrieval, then context assembly, then generation — because each stage is cheaper and more unambiguous to check than the next, and jumping straight to 'the model hallucinated' skips exactly the checks that would tell you if the model was ever given a fair chance to succeed in the first place."*

---

*Day 20 complete — the full technical curriculum (Days 1-20) is now covered. Next up — Day 21: Review Day (Evaluation + Diagnosis, Days 19-20), then Days 22-24 shift into System Design and Apple-specific framing.*
