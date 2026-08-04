# RAG Mastery Curriculum — Apple MLE Interview
### 4 Weeks | 1–2 hrs/day | Beginner → Interview-Ready

---

## How this is structured

Each week = one "act" of the RAG pipeline story: **build it → retrieve well → generate well → evaluate & operate it**. Every day ends with a concrete output (not just reading) so progress is visible. Every Sunday is a **no-new-content review day** — this is where retention actually happens, don't skip it.

For each module, we'll produce a **master notes doc** in the same format as your Module 7 notes (Quick Summary → Key Concepts → Deep Dive → Interview Q&A). By the end of the month you'll have a complete, self-contained reference stack.

**Daily rhythm (60–90 min):**
- 40–50 min: new concept (I build the module notes with you)
- 15–20 min: worked numerical example / small calculation by hand
- 10–15 min: 2-3 Q&A drill questions, self-tested

---

## Week 1 — Foundations: How RAG Actually Works Under the Hood

*Goal: you can whiteboard the full pipeline and explain every component's job before we touch evaluation or optimization.*

| Day | Topic | Why it matters for the interview |
|---|---|---|
| **Mon** | RAG vs. fine-tuning vs. long-context — when to use which | Almost every RAG interview opens with "why RAG instead of X" — this is your first 5 minutes |
| **Tue** | Embeddings: how text becomes vectors, cosine similarity, embedding model choices | Foundation for literally everything downstream; expect a "explain embeddings to a non-technical PM" style question |
| **Wed** | Chunking strategies: fixed-size, semantic, recursive, sliding window + overlap | Chunking mistakes are the #1 root cause of retrieval failures — high interview weight |
| **Thu** | Vector databases & indexing: HNSW, IVF, exact vs. approximate NN search | System-design-adjacent; Apple cares about latency/memory tradeoffs here |
| **Fri** | Metadata filtering, hybrid storage, multi-tenancy basics | Common "how would you scale this" follow-up |
| **Sat** | **Module Build Day**: consolidate Mon–Fri into Module 1 master notes | Deliverable: `RAG_Module1_Foundations.md` |
| **Sun** | **Review**: redo Week 1 Q&A cold, no notes | Spaced repetition checkpoint |

---

## Week 2 — Retrieval: Getting the Right Evidence

*Goal: you can defend retrieval design choices with trade-offs, not just definitions.*

| Day | Topic | Why it matters for the interview |
|---|---|---|
| **Mon** | Sparse retrieval: BM25, TF-IDF, when keyword search still wins | Interviewers probe "why not just use embeddings for everything" |
| **Tue** | Dense retrieval deep dive: bi-encoders vs. cross-encoders | Core architecture question, very commonly asked |
| **Wed** | Hybrid search: combining sparse + dense, reciprocal rank fusion | High-frequency real-world design question |
| **Thu** | Reranking: cross-encoder rerankers, why a 2-stage pipeline beats 1-stage | Directly feeds into your nDCG/position-sensitivity knowledge from Module 7 |
| **Fri** | Query transformation: query expansion, HyDE, multi-query, query decomposition for multi-hop | This is where "intermediate" candidates separate from "advanced" ones |
| **Sat** | **Module Build Day**: Module 2 master notes (Retrieval) | Deliverable: `RAG_Module2_Retrieval.md` |
| **Sun** | **Review + mini mock**: I quiz you cold on Weeks 1–2 combined | Checks integration, not just isolated recall |

---

## Week 3 — Generation: Turning Evidence into a Good Answer

*Goal: you understand what happens after retrieval, including the failure modes that only show up at generation time.*

| Day | Topic | Why it matters for the interview |
|---|---|---|
| **Mon** | Context construction: prompt templates, context ordering, "lost in the middle" | Ties directly to nDCG/position discussion in Module 7 — shows you can connect modules |
| **Tue** | Context window management: what to do when retrieved content exceeds budget, compression/summarization strategies | Practical engineering question Apple likes to probe (on-device / memory constraints) |
| **Wed** | Citation & attribution, faithfulness enforcement at generation time (not just eval-time) | Directly the runtime version of Module 7's faithfulness metric |
| **Thu** | Multi-hop RAG & agentic RAG: iterative retrieval, self-querying, tool-calling loops | "Advanced" tier topic — good differentiator if asked |
| **Fri** | Failure modes catalog: hallucination despite good context, over-reliance on parametric knowledge, "I don't know" calibration | Sets up Module 8 (Diagnosis) cleanly |
| **Sat** | **Module Build Day**: Module 3 master notes (Generation) | Deliverable: `RAG_Module3_Generation.md` |
| **Sun** | **Review**: Weeks 1–3 combined cold quiz | Last full-content review before eval/ops week |

---

## Week 4 — Evaluation, Diagnosis, Systems, and Interview Simulation

*Goal: full pipeline fluency + you've said the answers out loud under light pressure at least twice before the real thing.*

| Day | Topic | Why it matters for the interview |
|---|---|---|
| **Mon** | **Module 7 review** (you already have this) — re-read cold, redo all 8 Q&A without looking | Consolidation, not new content |
| **Tue** | Diagnosis & Debugging (Module 8): using the retrieval/generation split + RAG-triad triangulation table to root-cause failures | Direct continuation you were promised — ties the whole month together |
| **Wed** | System design: scaling RAG in production — latency budgets, caching, index refresh/freshness, cost tradeoffs, monitoring/observability | Apple MLE interviews often include a system-design component — this is the highest-leverage day of the month |
| **Thu** | Apple-specific framing: on-device/privacy-constrained RAG, latency-sensitive UX (Siri/Spotlight-style), small-model + retrieval tradeoffs | Tailors generic RAG knowledge to Apple's actual constraints — do NOT skip this day |
| **Fri** | **Mock interview #1**: full 45-min simulation, mixed conceptual + calculation + system design | Real signal on where you're weak with a week still to fix it |
| **Sat** | Targeted repair: revisit only your weak spots from Friday's mock | Efficient — no wasted review time on things you already know |
| **Sun** | **Mock interview #2** + final cheat-sheet consolidation across all modules | Confidence + a single-page artifact to skim the morning of |

---

## Deliverables by end of month

- `RAG_Module1_Foundations.md`
- `RAG_Module2_Retrieval.md`
- `RAG_Module3_Generation.md`
- `RAG_Module7_Evaluation_MasterNotes.md` ✅ *(already done)*
- `RAG_Module8_Diagnosis.md`
- `RAG_Module9_SystemDesign.md`
- `RAG_Apple_Specific_Notes.md`
- `RAG_Final_Cheat_Sheet.md` (built Week 4, Sunday)

---

## Ground rules to actually hit "mastery"

1. **No new content on Sundays.** Review-only. This is non-negotiable — spaced repetition is what turns "read it once" into "can say it under pressure."
2. **Every formula gets a hand-worked number.** If you can't compute a toy example without looking, you don't know it yet.
3. **Say answers out loud, not just in your head.** Interview performance is a speaking skill, not just a knowledge skill — the two mock interviews in Week 4 exist for exactly this reason.
4. **If a day runs long, cut breadth, not depth.** Better to deeply know 80% of the curriculum than shallowly know 100%.

---

**Ready to start Day 1 (RAG vs. fine-tuning vs. long-context)?** Say the word and we'll build Module 1 the same way we did Module 7.
