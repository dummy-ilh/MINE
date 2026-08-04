# RAG Interview Prep — Day 22
## System Design for RAG at Scale

---

## 🚀 Quick Summary

This day doesn't introduce much genuinely new technical content — it synthesizes nearly everything from Days 1–21 (chunking, embeddings, indexing, retrieval, reranking, context construction, caching, evaluation, diagnosis) into a **structured methodology for answering a live "design a RAG system" interview question**. The actual skill being tested in a system design interview isn't whether you know individual facts — you've spent three weeks proving that — it's whether you can **structure an open-ended design conversation**: clarify requirements first, sketch a high-level architecture, then go deep on the 2-3 components the interviewer actually cares about, all while reasoning explicitly about trade-offs rather than presenting one "correct" answer.

**Think of it like an architect designing a building versus a contractor who already knows how to pour concrete.** You've spent three weeks becoming the contractor — you know how each individual piece (chunking, indexing, reranking) actually works and what its trade-offs are. Today is about becoming the architect: given a vague brief ("design me a building"), asking the right clarifying questions (how many floors, what's the budget, where's the site), sketching the overall structure first, and only then diving deep into the specific rooms that matter most for this particular building.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Requirements clarification** | The opening phase of a system design interview — asking questions to pin down scale, latency, freshness, and cost constraints before proposing any architecture |
| **High-level architecture** | A first-pass end-to-end sketch of all major components, before any deep dive |
| **Deep dive** | Detailed exploration of 1-3 specific components, usually interviewer-directed, where most of the technical signal is actually assessed |
| **Capacity planning** | Translating scale/throughput requirements into concrete infrastructure sizing (Day 4, now applied system-wide) |
| **Ingestion pipeline** | The offline/background process that chunks, embeds, and indexes new/updated documents |
| **SLA (Service Level Agreement)** | A committed target for latency, availability, or freshness that the system design must satisfy |

---

# PHASE 1 — The System Design Interview Methodology

## The four-phase structure (use this every time, regardless of the specific prompt)

```
   PHASE 1: REQUIREMENTS CLARIFICATION  (~5 min)
        │  Ask before designing anything — a design built on wrong
        │  assumptions wastes the rest of the interview.
        ▼
   PHASE 2: HIGH-LEVEL ARCHITECTURE      (~10 min)
        │  One end-to-end sketch covering ingestion + query path +
        │  monitoring. Breadth first, not depth.
        ▼
   PHASE 3: DEEP DIVES                    (~20-25 min)
        │  Interviewer will steer you toward 2-3 components —
        │  follow their lead, this is where signal is actually assessed.
        ▼
   PHASE 4: TRADE-OFFS & WRAP-UP          (~5 min)
        │  Explicitly name what you'd reconsider under different
        │  constraints, and any open questions/risks.
```

**Why this structure matters more than any individual fact:** an interviewer forms their impression largely from *how* you navigate ambiguity, not from reciting the most facts fastest. Jumping straight into a detailed architecture without clarifying requirements is a common, costly mistake — it signals you'll build the wrong thing quickly rather than the right thing deliberately.

---

## Phase 1 — Requirements Clarification Checklist

**Always ask about these dimensions before proposing an architecture (adapt based on the prompt, but don't skip this phase):**

| Dimension | Example clarifying questions | Why it matters (which day it maps to) |
|---|---|---|
| **Scale** | How many documents/vectors? Expected QPS? | Determines ANN algorithm choice, sharding needs (Day 4) |
| **Latency** | What's the target p95/p99 end-to-end latency? | Determines how many retrieval/reranking stages fit in budget (Day 8, 10) |
| **Freshness** | How often does content change? Real-time or batch-acceptable? | Determines ingestion pipeline design, index update strategy (Day 4's incremental-update discussion) |
| **Consistency** | Is slightly stale data acceptable, or does every write need to be immediately searchable? | Determines replication/consistency model (Day 5) |
| **Multi-tenancy** | Single customer or multiple? Isolation requirements? | Determines namespace/sharding strategy (Day 5) |
| **Budget/cost sensitivity** | Is this cost-constrained or accuracy-maximizing? | Determines model size choices, caching aggressiveness (Day 2, 14) |
| **Accuracy/stakes** | What's the cost of a wrong answer? (casual FAQ vs. medical/legal) | Determines how much runtime faithfulness enforcement to build (Day 15) |

**Worked example of why this matters:** if the prompt is "design a RAG system for customer support," an interviewer might have a very different system in mind depending on unstated assumptions — is this 10,000 documents or 10 million? Does it need sub-100ms latency or is 2 seconds fine? Answering these upfront prevents building an elaborate sharded, cached, multi-stage pipeline for a problem that a single well-tuned bi-encoder + small vector index would have solved perfectly well — over-engineering in an interview is graded just as harshly as under-engineering.

---

## Phase 2 — High-Level Reference Architecture (The One Diagram to Have Ready)

```
                          ┌─────────────────────────────────────┐
                          │         INGESTION PIPELINE            │
                          │      (offline / background process)   │
                          │                                         │
   New/updated  ────────▶ │  1. Chunking (Day 3)                  │
   documents               │  2. Embedding (Day 2)                 │
                          │  3. Index write (Day 4) +              │
                          │     metadata attach (Day 5)             │
                          └─────────────────────────────────────┘
                                          │
                                          ▼
                          ┌─────────────────────────────────────┐
                          │           VECTOR INDEX / DB            │
                          │   (HNSW/IVF-PQ, sharded + replicated,  │
                          │    namespaced per tenant — Day 4/5)     │
                          └─────────────────────────────────────┘
                                          ▲
                                          │
   User query ──▶ ┌─────────────────┐    │
                  │ Query transform  │────┘
                  │ (Day 11, if      │
                  │  triggered)      │
                  └─────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  FIRST-STAGE RETRIEVAL │  Hybrid: BM25 (Day 7) +
              │                         │  bi-encoder/ANN (Day 8),
              │                         │  fused via RRF (Day 9)
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │      RERANKING          │  Cross-encoder / ColBERT
              │                         │  (Day 10)
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  CONTEXT CONSTRUCTION   │  Sandwiching, dedup,
              │                         │  compression, budget
              │                         │  allocation (Day 13/14)
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │      GENERATION         │  Citation enforcement,
              │                         │  groundedness guardrail,
              │                         │  refusal calibration (Day 15)
              └───────────────────────┘
                          │
                          ▼
                   Response to user

              ┌─────────────────────────────────────┐
              │     MONITORING / OBSERVABILITY         │  Logging every
              │  (cuts across every stage above)        │  stage (Day 20),
              │                                          │  faithfulness
              │                                          │  sampling, drift
              │                                          │  detection, alerting
              └─────────────────────────────────────┘
```

**Why presenting it this way scores well:** this single diagram demonstrates fluency across the *entire* curriculum in one shot — an interviewer can immediately see you understand this as one coherent system with a clear data flow, not a disconnected bag of techniques. From here, you'd verbally walk through it left-to-right/top-to-bottom once at a high level, then wait for the interviewer to direct you toward specific deep dives.

---

## Phase 3 — Deep Dive Areas (Be Ready for Any of These)

The interviewer will typically pick 2-3 of these to go deep on — you won't cover all of them in one interview, but you should be ready for any:

1. **Capacity planning** — given scale numbers, walk through the memory/sharding/replication math (Day 4's worked example, now applied to whatever specific numbers this interview's prompt gives you).
2. **Latency budget breakdown** — allocate the end-to-end latency target across retrieval, reranking, and generation stages (Day 8/10's worked latency tables, now built into one full budget).
3. **Ingestion/freshness design** — batch vs. streaming ingestion, how updates propagate to the index, staleness guarantees (Day 4/5's incremental update discussion, now framed as a pipeline design decision).
4. **Cost modeling** — full end-to-end cost breakdown (see worked example below).
5. **Multi-tenancy/isolation** — namespace vs. shared-index trade-offs for the specific tenant profile in this prompt (Day 5).
6. **Monitoring/evaluation strategy** — what gets logged, what gets sampled for faithfulness checks, how drift gets detected (Day 20, now as ongoing production monitoring rather than one-off debugging).

---

# PHASE 2 — Worked Full Cost Model (A Common Deep-Dive Request)

**Scenario:** 5,000 queries/day, average 6 retrieved chunks per query (300 tokens each), reranking top-30 candidates, average generated answer 200 tokens.

```
── EMBEDDING COST (query-time only; corpus embedding is a one-time/
   incremental ingestion cost, computed separately) ──
Query embedding: 1 call per query, ~20 tokens average
  5,000 queries/day × 20 tokens = 100,000 tokens/day
  (embedding APIs are typically priced much lower than generation —
   assume $0.02/million tokens for this estimate)
  Cost: negligible, ~$0.002/day

── RERANKING COST ──
Assume a cross-encoder API priced per document scored, ~$0.001/doc
  5,000 queries/day × 30 candidates reranked = 150,000 rerank calls/day
  Cost: 150,000 × $0.001 = $150/day

── GENERATION COST (the dominant cost, almost always) ──
Input tokens per query: 6 chunks × 300 tokens + 100 tokens
  (instructions/query) = 1,900 tokens
Output tokens per query: 200 tokens
Assume $3/million input tokens, $15/million output tokens
  (a realistic ballpark spread for input vs. output pricing)

  Input cost:  5,000 × 1,900 = 9,500,000 tokens/day
               9.5M/1M × $3 = $28.50/day
  Output cost: 5,000 × 200 = 1,000,000 tokens/day
               1M/1M × $15 = $15.00/day
  Generation subtotal: $43.50/day

── VECTOR DATABASE / INFRASTRUCTURE COST ──
(Highly variable — depends on managed vs. self-hosted, corpus size,
 QPS. For this example, assume a managed vector DB tier costs a
 flat ~$200/month regardless of query volume at this modest scale.)
  Cost: ~$6.67/day amortized

── TOTAL DAILY COST ──
  Embedding:     ~$0.002/day   (negligible)
  Reranking:     $150.00/day   ← actually the LARGEST line item here
  Generation:    $43.50/day
  Infrastructure: $6.67/day
  ─────────────────────────
  TOTAL: ~$200.17/day ≈ $6,005/month
```

**The counter-intuitive insight worth stating explicitly in an interview:** in this worked example, **reranking is the single largest cost driver**, not generation — because reranking cost scales with the *number of candidates scored* (30 per query here), while generation cost, though processing more total tokens, is priced more cheaply per token in this example. This is exactly the kind of concrete, numbers-driven insight that separates "I know reranking exists" from "I understand its actual cost profile at scale" — and it directly motivates a real optimization: reducing the reranking candidate set size (e.g., from 30 to 15) would materially cut cost, which is a genuine trade-off against retrieval recall worth discussing (tying back to Day 10's candidate-set-size latency/cost lever, now framed as a cost lever too, not just a latency one).

> **Why This Matters callout:** Cost modeling questions are as much about the *reasoning process* (identify every stage that costs money, get real per-unit pricing assumptions, multiply by volume, sum) as about landing on an exact number — interviewers care far more that you systematically accounted for every cost-bearing stage (including the easy-to-forget reranking cost) than that your final dollar figure is precise.

---

# PHASE 3 — Monitoring & Production Observability (Building on Day 20)

Day 20 covered what to log to **debug a specific reported bug**. In a system design context, this needs to be framed as **ongoing production monitoring**, not just ad-hoc investigation tooling:

| Monitoring category | What to track | Why |
|---|---|---|
| **Latency** | p50/p95/p99 per pipeline stage (retrieval, rerank, generation) | Catches degradation before it becomes a user-facing SLA breach; per-stage breakdown localizes *which* stage regressed |
| **Error rates** | Retrieval failures, generation timeouts, guardrail trigger rate | Operational health signal, distinct from quality signal |
| **Faithfulness sampling** | Periodically sample live production responses and run them through Module 7's faithfulness check | Since running faithfulness checks on 100% of traffic is often too costly, sampling gives an ongoing quality signal cheaper than full coverage |
| **Refusal rate tracking** | Track false-refusal and false-answer proxies over time (Day 17's two-sided framing) | Detects calibration drift — a system that was well-calibrated at launch can drift as query distribution or corpus content shifts |
| **Drift detection** | Monitor for embedding drift (Day 2) after any model update, and corpus/index staleness (Day 4/5) | Catches infrastructure-level degradation that wouldn't necessarily show up immediately in user complaints |
| **Cache hit rates** | Prefix cache hit rate, semantic cache hit rate (Day 14) | Cost/performance health signal; a dropping cache hit rate might indicate a shift in query patterns worth investigating |

**Why sampling (not 100% coverage) is the standard approach for faithfulness monitoring:** running a full faithfulness check (claim decomposition + NLI/LLM-judge verification) on every single production response would itself add significant cost and latency at scale — mirroring Day 15's runtime-guardrail cost trade-off, but now at the *aggregate monitoring* level rather than the per-response gating level. A representative sample (e.g., 2-5% of traffic, or oversampling specific high-risk query categories) gives an ongoing quality signal at a fraction of the cost of checking everything.

---

# PHASE 4 — Worked Full System Design Answer (Sample Interview Prompt)

**Prompt:** "Design a RAG-based internal knowledge assistant for a company with 50,000 employees, covering HR policies, IT documentation, and engineering wikis."

**A strong structured answer, condensed:**

```
PHASE 1 (clarify): I'd ask: roughly how many total documents across
all three domains? What's the acceptable latency (this feels like an
internal tool, so probably 1-3 seconds is fine, not sub-100ms)? How
often does content change (HR policies rarely, engineering wikis
possibly daily)? Any access-control requirements (should all 50,000
employees see all content, or does HR content need role-based
restriction)?

[Assume answers: ~200,000 total documents, 2-second latency
acceptable, engineering wikis update daily, HR content requires
role-based access restriction.]

PHASE 2 (high-level architecture): I'd sketch the standard reference
architecture from earlier — ingestion pipeline (chunking + embedding
+ indexing), hybrid retrieval (BM25 + bi-encoder, since HR/IT/eng
content will have both natural-language questions AND exact
terminology/ticket-number-style queries — Day 7's exact-match case),
reranking, context construction, generation with citation
enforcement, and monitoring throughout.

PHASE 3 (deep dive, likely interviewer-directed toward):
- Access control: given HR content needs role-based restriction,
  I'd apply Day 5's retrieval-layer filtering principle explicitly —
  every query includes the requesting employee's role/permission
  scope as a mandatory filter applied during retrieval, not just
  hidden in the UI, to prevent a scenario where an unauthorized
  employee's query could surface restricted HR content in generated
  answers.
- Freshness: given engineering wikis update daily vs. HR policies
  rarely, I'd design the ingestion pipeline with differentiated
  update cadences per content source rather than one uniform batch
  schedule — near-real-time or frequent-batch ingestion for eng
  wikis, much less frequent (even manual-triggered) reindexing for
  stable HR content, avoiding unnecessary reprocessing cost for
  content that rarely changes.
- Given the modest scale (200,000 documents) and generous 2-second
  latency budget, I'd note this does NOT require the aggressive
  sharding/IVF-PQ compression treatment from Day 4's billion-scale
  example — a single well-configured HNSW index comfortably handles
  this scale, which is worth explicitly stating rather than
  over-engineering for a scale this system doesn't actually have.

PHASE 4 (trade-offs): If this were instead customer-facing (not
internal) with much higher QPS and stricter latency requirements,
I'd reconsider several choices — more aggressive caching, sharding
even at this modest document count if QPS were very high, and a
tighter reranking candidate set to control cost/latency. I'd flag
role-based access control as the single highest-risk area to get
right given the HR content sensitivity, and recommend the
counterfactual/access-control eval slice (Day 17/21) be a
non-negotiable part of the launch evaluation criteria.
```

**Why this example matters:** notice the explicit statement that this scale does *not* need billion-scale infrastructure — a common mistake in system design interviews is reflexively reaching for the most sophisticated solution (sharding, PQ compression, aggressive caching) regardless of whether the actual requirements call for it. Correctly scoping the solution to the stated requirements, and saying so explicitly, is itself a signal of good engineering judgment.

---

# PHASE 5 — Interview Q&A Practice Set

*(These are meta-questions about the system design process itself, plus a few applied scenarios.)*

---

**Q1 (Easy — process).** What's the biggest risk of skipping the requirements clarification phase in a system design interview?

<details>
<summary>Show answer</summary>

Building an elaborate, technically sophisticated architecture that solves the wrong problem — e.g., designing for billion-vector scale with aggressive sharding and compression when the actual use case only has 50,000 documents and generous latency tolerance. This signals poor engineering judgment (over- or under-engineering relative to actual needs) more than it signals technical depth, and wastes interview time building on wrong assumptions that could have been corrected in the first two minutes.
</details>

---

**Q2 (Medium — cost reasoning).** In a cost model where reranking scores 40 candidates per query and generation processes 2,000 input tokens per query, why might reranking end up being the dominant cost despite generation processing far more total tokens?

<details>
<summary>Show answer</summary>

Reranking cost typically scales with the *number of candidates scored* (a per-document pricing model), while generation cost scales with *total tokens processed*, but is often priced at a different rate. If reranking's per-document cost is high relative to generation's per-token cost, a large candidate set can dominate total cost even though generation handles more raw tokens — this is exactly the kind of counter-intuitive result that only shows up when you actually work through a full cost model rather than assuming generation (the most "visible" LLM cost) is always the largest line item.
</details>

---

**Q3 (Medium — monitoring).** Why is 100% faithfulness monitoring on production traffic usually impractical, and what's the standard alternative?

<details>
<summary>Show answer</summary>

Running a full faithfulness check (claim decomposition + NLI/LLM-judge verification) on every single response adds meaningful cost and latency at scale, mirroring the same trade-off as Day 15's runtime groundedness guardrail. The standard alternative is sampling — checking a representative percentage of traffic (or oversampling specific high-risk query categories) to get an ongoing quality signal at a fraction of the cost of full coverage, accepting reduced (but still statistically useful) visibility in exchange for sustainable operating cost.
</details>

---

**Q4 (Hard — scoping judgment).** An interviewer gives you a prompt with no scale numbers at all ("design a RAG system for search"). How do you avoid either wasting time on an over-generic answer or guessing wrong and over/under-engineering?

<details>
<summary>Show answer</summary>

I'd explicitly use the requirements clarification phase to ask for concrete numbers before proposing architecture specifics — scale (document count, QPS), latency target, freshness needs, and stakes/accuracy requirements — rather than either assuming a specific scale unprompted or giving a maximally generic answer that hedges on every design decision. If the interviewer declines to give specifics (sometimes intentional, to see how you handle ambiguity), I'd state a reasonable assumption explicitly ("I'll assume a mid-size corpus, on the order of a few million documents, and a latency budget of 1-2 seconds, since that's a common profile for this type of system — let me know if you'd like me to design for a different scale") and proceed, rather than stalling — this shows both that I know to ask, and that I can make and state a reasonable assumption when an interviewer wants me to move forward.
</details>

---

**Q5 (Hard — full synthesis).** Design the monitoring and alerting strategy for a production RAG system, specifically covering how you'd distinguish a retrieval-stage regression from a generation-stage regression using only aggregate dashboard metrics, without manually re-running Day 20's full diagnostic workflow on every incident.

<details>
<summary>Show answer</summary>

I'd track Recall@k (or a proxy, if ground-truth labels aren't available in production — e.g., a sampled/labeled subset) and nDCG as retrieval-stage health signals, and faithfulness/answer-relevance (sampled, per the discussion above) as generation-stage health signals, dashboarded separately rather than only as one aggregate "answer quality" number — mirroring Module 7's foundational argument for why retrieval and generation need separate metrics in the first place, now applied to production monitoring rather than offline evaluation. A sudden drop in retrieval-side metrics (Recall@k/nDCG) with generation-side metrics holding steady would immediately point toward a retrieval-stage regression (e.g., following a recent embedding model change, or an index update issue) without needing to manually trace individual failing queries first — the aggregate metric split itself does the initial stage-localization that Day 20's workflow does at the individual-query level, giving an early, cheap signal for *which* stage to investigate further before diving into per-query tracing.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Jumping straight into a detailed architecture without clarifying requirements first — the single most common system-design-interview mistake.
- ❌ Reflexively proposing billion-scale infrastructure (heavy sharding, PQ compression, aggressive multi-tier caching) regardless of the actual stated scale — over-engineering is graded as harshly as under-engineering.
- ❌ Assuming generation is always the dominant cost line item without actually working through a full cost model — reranking and infrastructure costs can dominate depending on configuration.
- ❌ Proposing 100% faithfulness checking on all production traffic without acknowledging the cost/latency trade-off that makes sampling the standard practical approach.
- ❌ Treating monitoring as a single aggregate "quality" dashboard rather than splitting retrieval-stage and generation-stage metrics, losing the stage-localization value that split gives you.
- ❌ Not explicitly stating trade-offs at the end — a system design answer that never says "here's what I'd reconsider under different constraints" reads as less mature than one that does.

---

# 📌 Cheat Sheet (Day 22)

**The four-phase structure:** Requirements clarification → High-level architecture (one full-pipeline diagram) → Deep dives (interviewer-directed, 2-3 areas) → Trade-offs/wrap-up.

**Clarification checklist:** scale, latency, freshness, consistency, multi-tenancy, budget, accuracy/stakes — ask before designing.

**Reference architecture:** ingestion (chunk→embed→index) → vector DB (sharded/replicated/namespaced) → query path (transform→hybrid retrieval→rerank→context construction→generation with guardrails) → monitoring cutting across everything.

**Cost modeling:** account for embedding, reranking (often the surprise dominant cost), generation, and infrastructure separately — don't assume generation always dominates.

**Monitoring:** split retrieval-stage and generation-stage metrics on the dashboard (mirrors Module 7's core argument, now at production scale); sample faithfulness checks rather than running them on 100% of traffic; track cache hit rates and drift signals alongside standard latency/error metrics.

**Golden interview line:** *"I'd clarify scale, latency, and freshness requirements first, sketch the full pipeline once at a high level, and then go as deep as you'd like on any specific component — and I'd rather explicitly scope the design to the stated requirements than default to the most sophisticated architecture regardless of whether this system actually needs it."*

---

*End of Day 22. Next up — Day 23: Apple-Specific RAG Framing (on-device/privacy constraints, Siri/Spotlight-style latency, small-model + retrieval tradeoffs).*
