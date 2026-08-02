# RAG Interview Prep — Day 1
## RAG vs. Fine-Tuning vs. Long-Context

---

## 🚀 Quick Summary

There are three fundamentally different ways to get a language model to "know" something it wasn't born knowing: **bake it into the weights** (fine-tuning), **paste it into the prompt every time** (long-context), or **fetch just the relevant piece at query time** (RAG). They solve different problems — fine-tuning changes *behavior*, long-context and RAG both supply *knowledge*, but differ sharply in cost, freshness, and scale. Nearly every RAG interview opens with some version of "why not just do X instead" — this is your first 5 minutes, so it needs to be automatic.

**Think of it like three ways to make an employee good at their job.** Fine-tuning is sending them to a training course that changes their instincts permanently. Long-context is handing them the entire company handbook every time they answer a question and trusting them to skim the right page under time pressure. RAG is giving them a fast assistant who, the moment a question comes in, pulls out *just* the relevant pages and hands them over.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Fine-tuning** | Updating a model's internal weights on a custom dataset so knowledge/behavior becomes baked into the parameters |
| **Long-context** | Feeding an entire knowledge base (or a large chunk of it) directly into the prompt instead of retrieving a subset |
| **RAG** | Retrieval-Augmented Generation — fetch relevant external evidence at query time, generate an answer conditioned on it |
| **Parametric knowledge** | What a model "knows" because it's encoded in its trained weights |
| **Non-parametric knowledge** | What a model "knows" only because it was handed to it at inference time (via prompt or retrieval) |
| **Knowledge cutoff** | The point after which a model has no parametric knowledge of new events/facts, absent retrieval |
| **Lost in the middle** | The empirically observed tendency of LLMs to attend less reliably to information buried in the middle of a long context, vs. the start or end |

---

# PHASE 1 — Intuition & Visual Map

## The core question each approach answers

```
                    ┌─────────────────────────────────────────┐
                    │      IS THIS A "KNOWLEDGE" PROBLEM         │
                    │           OR A "SKILL" PROBLEM?            │
                    └─────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                                 ▼
      "The model doesn't know                        "The model doesn't know
       HOW to respond"                                 WHAT to respond with"
      (tone, format, style,                            (facts, documents,
       domain conventions)                              current knowledge)
              │                                                 │
              ▼                                  ┌──────────────┴──────────────┐
        FINE-TUNING                              ▼                             ▼
                                          Small, static,                Large, changing,
                                          fits in a prompt               or per-user/
                                                  │                      per-permission
                                                  ▼                             │
                                          LONG-CONTEXT                         ▼
                                                                              RAG
```

This diagnostic tree is the single most useful mental model to carry into the interview. When someone describes a use case, your first move should be classifying it: is this fundamentally a *behavior* gap or a *knowledge* gap? That one question eliminates half the wrong answers immediately.

## When to use each — with reasoning, not just labels

**Fine-tuning — use when the problem is *how* the model responds:**
- Teaching a specific output format (e.g., always respond in structured JSON matching a schema)
- Teaching domain jargon/tone (e.g., sounding like a specific brand's support voice)
- Teaching a skill, like reliably refusing to answer when evidence is missing, or following a citation format consistently
- **NOT** a good fit for teaching new facts that change over time — this is the #1 fine-tuning misuse to flag in an interview

**Long-context — use when the knowledge base is small and mostly static:**
- A single 50-page policy PDF that rarely changes
- A single codebase small enough to fit in the window
- Situations where you genuinely don't want retrieval infrastructure at all — sometimes the simplest answer really is "just paste it in"

**RAG — use when the problem is *what* the model needs to know, and that knowledge is large, changes often, or must be scoped per-user:**
- A support knowledge base updated daily
- A product catalog with thousands of SKUs
- Any system needing per-user/per-permission access control over what's even eligible to be seen

**They are not mutually exclusive — this is the answer that separates "good" from "great" in an interview.** A very common and strong real-world pattern: **fine-tune the model to be good at using retrieved context well** (following a citation format, refusing gracefully when evidence is missing, adopting the right tone) — **and use RAG to supply the actual facts**. Fine-tuning teaches the skill; RAG supplies the knowledge. If you're only given one option in an interview question, say so — but volunteer that the real answer is usually "both, for different reasons."

---

# PHASE 2 — Deep Dive: The Trade-offs, Quantified

## Comparison table (memorize this cold)

| Dimension | Fine-tuning | Long-context | RAG |
|---|---|---|---|
| **Knowledge freshness** | Stale until retrained (slow, expensive) | Fresh — just change what you paste in | Fresh — just update the index |
| **Cost per query** | Low (no extra tokens beyond the query itself) | High (paying for a huge context on every single call) | Moderate (retrieval step + a much smaller context) |
| **Latency** | Fast (no retrieval hop, no huge prompt) | Slow (processing a huge prompt takes time) | Moderate (retrieval adds a hop, but the context fed to generation stays small) |
| **Best for** | Teaching style/format/skill | Small, static corpora that fit in the window | Large, frequently-changing, or per-user knowledge |
| **Attribution/citations** | Hard — can't point to a source, it's baked into weights | Possible, but you're citing from one giant blob | Natural — you know exactly which chunk was retrieved |
| **Hallucination risk** | Higher on facts outside training data | Lower if the info is in context, but subject to lost-in-the-middle | Lower, but only as good as retrieval — garbage in, garbage out |
| **Access control / privacy** | Hard to scope per-user (weights are global to all users) | Hard to scope (whole context is one blob shared across the call) | Natural — filter retrieval per user/permission at query time |
| **Infra complexity** | Training pipeline, eval, versioning of model checkpoints | Minimal — just prompt engineering | Vector DB, indexing pipeline, retrieval + reranking infra |

## Worked numerical example — the actual cost math

This is the calculation that makes "why not just use a huge context window" concrete instead of hand-wavy.

**Setup:** A support bot answers customer questions against a 2-million-token knowledge base (think: the entire Apple support documentation corpus). Assume an LLM API charges **$3 per million input tokens** (a realistic ballpark for a mid-tier model).

**Option A — Long-context (stuff the whole knowledge base in every call):**
```
tokens per query = 2,000,000 (entire knowledge base, every single call)
cost per query    = 2,000,000 / 1,000,000 × $3 = $6.00 per query
```
At **10,000 queries/day**, that's **$60,000/day** — and most of those 2 million tokens are completely irrelevant to any single question.

**Option B — RAG (retrieve top-5 chunks of ~500 tokens each):**
```
tokens per query = 5 × 500 = 2,500 tokens (retrieved context) + ~100 tokens (query + instructions)
                  ≈ 2,600 tokens

cost per query    = 2,600 / 1,000,000 × $3 = $0.0078 per query
```
At 10,000 queries/day, that's **~$78/day**.

**The gap:** roughly **770x cheaper** with RAG in this scenario — and this doesn't even account for the *latency* difference (processing 2M tokens per call vs. ~2.6K tokens per call is a massive difference in time-to-first-token, which matters enormously for anything user-facing).

**Why this matters in practice:** This is the number to have ready when someone asks "context windows are huge now, why not skip RAG entirely?" It's not that long-context is technically impossible — it's that you're paying (in both dollars and latency) for the model to re-read the entire haystack on every single query just to answer a question about one needle, and that cost scales linearly with corpus size in a way RAG's retrieval step doesn't.

## The "lost in the middle" problem

Even setting cost aside, **stuffing everything into context doesn't guarantee the model actually uses it well**. Empirical research on long-context LLMs has repeatedly found that models attend less reliably to information placed in the *middle* of a long context, compared to information near the beginning or end — performance on a fact-retrieval task tends to dip when the relevant fact is buried in the middle of a long document, even though the fact is technically "in the prompt."

**What this means practically:** simply having the right information *somewhere* in a 2-million-token context is not the same as the model actually *using* it correctly when generating an answer. This is precisely why RAG's much smaller, curated context (Module 1.3–1.4: good chunking + good retrieval) tends to produce more reliable answers than "just paste everything in and hope" — a smaller, higher-relevance context is easier for the model to attend to correctly, and this exact phenomenon resurfaces later when we discuss context construction and ordering (a later Generation-phase topic) and nDCG's position-sensitivity in evaluation.

---

## Compare & Contrast: Common Follow-Up Scenarios

| Scenario | Best fit | Reasoning |
|---|---|---|
| A legal team wants the model to always cite specific statute numbers in a fixed format | Fine-tuning (behavior) + RAG (facts) | Format compliance is a fine-tuning problem; the actual statute content should be retrieved, not memorized, since laws change |
| A single internal 30-page onboarding doc, rarely updated, low query volume | Long-context | Small, static, low volume — building retrieval infra is overkill |
| A customer support bot over a product catalog that changes daily, needs per-region content | RAG | Large, frequently-changing, needs access-control scoping |
| A model needs to always respond in the company's specific brand voice | Fine-tuning | Pure behavior/style problem, no new facts involved |
| A coding assistant needs awareness of an entire (large) proprietary codebase that changes hourly | RAG (possibly hybrid with long-context for smaller repos) | Large + frequently changing points to RAG; if the repo is small enough to fit affordably, long-context is a legitimate simpler alternative |

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** In one sentence, what's the fundamental difference between what fine-tuning solves and what RAG solves?

<details>
<summary>Show answer</summary>

Fine-tuning changes the model's *behavior* (baking knowledge/skills into its weights), while RAG supplies *external, up-to-date knowledge* at query time without changing the model itself — one is a "how do I respond" fix, the other is a "what do I know" fix.
</details>

---

**Q2 (Easy — conceptual).** Your product catalog updates weekly. Would you recommend fine-tuning or RAG, and why?

<details>
<summary>Show answer</summary>

RAG. Weekly-changing knowledge would require weekly retraining under a fine-tuning approach, which is slow and expensive. RAG only requires updating the vector index (re-embedding and inserting new/changed documents), which is far cheaper and can happen continuously without touching the model itself.
</details>

---

**Q3 (Medium — calculation).** A knowledge base is 500,000 tokens. An LLM API charges $2 per million input tokens. Compare the per-query cost of stuffing the entire knowledge base into context vs. retrieving 4 chunks of 400 tokens each (plus ~100 tokens overhead) via RAG.

<details>
<summary>Show answer</summary>

```
Long-context: 500,000 / 1,000,000 × $2 = $1.00 per query

RAG: (4 × 400) + 100 = 1,700 tokens
     1,700 / 1,000,000 × $2 = $0.0034 per query
```
Long-context costs roughly **294x more per query** than RAG in this scenario.
</details>

---

**Q4 (Medium — conceptual).** Someone argues "context windows are now huge (1M+ tokens), so RAG is becoming obsolete." How would you respond?

<details>
<summary>Show answer</summary>

Context window size solves *capacity*, not the underlying problems RAG addresses: (1) cost — you pay for the full context on every query even if only a tiny fraction is relevant, and that cost scales with corpus size; (2) latency — processing huge prompts is slow, hurting user-facing responsiveness; (3) lost-in-the-middle — empirically, models attend less reliably to information buried in the middle of a long context, so having the right fact "somewhere in there" doesn't guarantee the model uses it correctly; (4) access control — RAG lets you filter what's even eligible to be retrieved per user/permission, which a static long context can't do cleanly. Bigger context windows make long-context a more viable option for medium-sized static corpora, but they don't eliminate RAG's advantages for large, frequently-changing, or access-controlled knowledge.
</details>

---

**Q5 (Hard — synthesis).** Describe a realistic system that would use fine-tuning, long-context, AND RAG together, and justify each piece.

<details>
<summary>Show answer</summary>

Example: A legal research assistant. **Fine-tuning** teaches the model to always respond in a specific structured citation format and to reliably say "insufficient evidence" rather than guess when retrieved context doesn't cover the question — a behavior/skill problem. **RAG** supplies the actual case law and statutes, which are large in volume, updated as new rulings are published, and need per-jurisdiction access scoping — a knowledge problem best solved by retrieval, not memorization. **Long-context** might be used within a single case file review — e.g., once RAG has retrieved the 15 most relevant case documents for a specific matter, all 15 might be small enough to load together into context for the model to reason across them jointly in one pass, rather than retrieving one at a time. Each piece is solving a distinct problem the others don't address.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Treating fine-tuning as a way to keep a model "up to date" on facts — it's the wrong tool for frequently-changing knowledge.
- ❌ Answering "why not skip RAG with big context windows" using only "context isn't big enough yet" — the stronger answer includes cost, latency, lost-in-the-middle, and access control.
- ❌ Presenting these three as mutually exclusive when the strongest real-world (and interview) answer is usually "combine them for different reasons."
- ❌ Forgetting that having information *in* the context ≠ the model reliably *using* it — lost-in-the-middle is a real, measurable effect, not a theoretical footnote.

---

# 📌 Cheat Sheet (Day 1)

**Fine-tuning** = fix *behavior* (tone, format, skill). Bakes into weights. Stale unless retrained. Hard to scope per-user. No natural citations.

**Long-context** = fresh, simple, no infra — but expensive and slow at scale, and subject to lost-in-the-middle even when the answer is technically "in there."

**RAG** = fix *knowledge*. Fresh via index updates, not retraining. Much cheaper per-query at scale (worked example: ~300-800x cheaper depending on corpus size). Natural citations and per-user access control. Only as good as retrieval quality.

**Golden interview line:** *"It's rarely either/or — fine-tuning teaches the skill of using context well, RAG supplies the knowledge, and long-context is a legitimate simpler choice only when the corpus is small and static enough to make retrieval infrastructure unnecessary."*

---

*End of Day 1. Next up — Day 2: Embeddings & Vector Representations.*
