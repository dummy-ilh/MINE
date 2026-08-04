# RAG Module 7 — Evaluation
### Master Interview Prep Notes

---

## 🚀 Quick Summary

RAG (Retrieval-Augmented Generation) has **two independent failure points** — retrieval (did we fetch the right evidence?) and generation (did we write a good answer from that evidence?) — and you need **separate metrics for each**, because a single end-to-end "is the answer correct" score can't tell you which stage broke. This module covers the standard metric toolkit (Recall@k, MRR, nDCG for retrieval; the "RAG triad" of faithfulness/answer relevance/context relevance for generation), the frameworks that implement them (RAGAS, TruLens, DeepEval), the mechanics and pitfalls of using an LLM as a judge, how to build a golden eval set, and when to trust offline eval vs. run a live A/B test.

**Think of it like a restaurant.** A bad meal could be because the kitchen bought bad ingredients (retrieval failure) or because the chef ruined good ingredients (generation failure). If you only taste the final dish, you can't tell which happened — you need to inspect the ingredients *and* taste the dish separately to know what to fix.

---

## 🔑 Key Concepts (Glossary — skim this first, reference it later)

| Term | One-line definition |
|---|---|
| **Recall@k** | Of all relevant docs that exist, what fraction did we find in our top-k? |
| **Precision@k** | Of what we retrieved (top-k), what fraction was actually relevant? |
| **MRR** | How early did the *first* relevant result show up, averaged over queries? |
| **nDCG** | Recall + Precision + ranking-order awareness + graded (not just binary) relevance, all in one number |
| **Faithfulness / groundedness** | Does the answer only say things supported by the retrieved context? (hallucination detector) |
| **Answer relevance** | Does the answer actually address the question asked? |
| **Context relevance** | Of the context we fed the model, how much of it was actually useful vs. noise? |
| **LLM-as-judge** | Using an LLM to grade another LLM's output against a rubric |
| **Golden eval set** | A labeled (query, relevant docs, ideal answer) dataset used as your regression-test suite |
| **Offline eval** | Running your eval set against metrics in a controlled/repeatable setting before shipping |
| **Online eval** | Measuring real user behavior signals (clicks, thumbs, follow-ups) in production |

---

# PHASE 1 — Intuition & Mental Model

## Why RAG needs two separate evaluation surfaces

**Analogy — the restaurant kitchen.** Imagine a restaurant where the waiter takes your order (the query), the kitchen pulls ingredients from the pantry (retrieval), and the chef cooks a dish from those ingredients (generation). If the dish tastes bad, there are two totally different possible root causes:

1. The pantry had spoiled or wrong ingredients → **retrieval failure**
2. The pantry had perfect ingredients, but the chef botched the cooking → **generation failure**

If you only ever taste the finished plate, you genuinely cannot tell which one happened — and worse, *the fix is completely different* depending on which it was. Fixing "wrong ingredients" means changing your supplier relationships (chunking strategy, embedding model, hybrid search). Fixing "bad cooking" means retraining/re-prompting the chef (prompt engineering, context ordering, generation model choice). This is exactly why RAG evaluation is split into two surfaces instead of one end-to-end pass/fail score.

```
   USER QUERY
       │
       ▼
 ┌─────────────┐        ┌──────────────┐
 │  RETRIEVAL  │──docs──▶│  GENERATION  │──▶ FINAL ANSWER
 │  (pantry)   │        │   (chef)     │
 └─────────────┘        └──────────────┘
       │                       │
  Recall@k, MRR,         Faithfulness,
  nDCG, Precision@k      Answer Relevance,
                         Context Relevance
```

**When to evaluate retrieval and generation separately:**
- ✅ Whenever you're debugging *why* a RAG system underperforms (you need to localize the failure)
- ✅ Whenever you're A/B testing a change to only *one* stage (e.g., new chunking strategy) — you don't want generation noise polluting your read on retrieval quality
- ✅ Pre-launch, to catch stage-specific regressions in CI

**When end-to-end metrics are still fine:**
- ✅ Final "is this system good enough to ship" business decision (you still want an end-to-end number too — just not as your *only* number)
- ✅ Online/production signals (users don't care which stage failed, they just abandon the session)

This framing sets up **Module 8 (Diagnosis & Debugging)** — the whole reason you split metrics this way is so that when something goes wrong in production, you have enough instrumentation to know *where* to look first instead of guessing.

---

# PHASE 2 — Math & Formulas

## Section A: Retrieval Metrics

All retrieval metrics require a **labeled eval set**: a list of `(query, set of known-relevant document/chunk IDs)` pairs. Without ground-truth relevance labels, none of these numbers mean anything — this is worth saying explicitly because interviewers will probe "how do you get the labels?" (see §7.6 below).

### Notation table

| Symbol | Meaning |
|---|---|
| `k` | Number of top-ranked results you're looking at (a cutoff you choose, e.g. 5, 10) |
| `relevant` | The full set of ground-truth relevant documents for a given query |
| `top-k` | The set of documents your system actually retrieved and ranked in positions 1..k |
| `rank_i` | The position (1st, 2nd, 3rd...) of the first relevant result for query *i* |
| `rel_i` | The graded relevance score of the document at rank *i* (e.g. 0, 1, 2, 3) |
| `Q` | The set of all queries in your eval set |

---

### 1. Recall@k

```
Recall@k = |relevant ∩ top-k| / |relevant|
```

**Plain English:** "Of everything that *should* have been found, what fraction did we actually surface in our top-k results?" It only cares whether the good stuff showed up *somewhere* in the top-k — not where, not how many irrelevant things came along for the ride.

**Term-by-term:**
- `relevant ∩ top-k` — the overlap: relevant documents that *also* made it into your top-k. This is your "hits."
- `|relevant|` — the total number of documents that were ever relevant to begin with (your denominator — the full size of the "correct answer" set).
- Dividing hits by total-relevant gives you a fraction from 0 (found nothing relevant) to 1 (found everything relevant).

**What happens if each term changes:**
- **k increases** → Recall@k can only go up or stay the same (you're looking at a bigger net, so you can't lose hits you already had). This is why Recall@k is always reported *at a specific k* — Recall@100 is trivially easier than Recall@3.
- **|relevant| increases** (more ground-truth relevant docs exist) → for a fixed number of hits, recall goes *down* — harder queries with many valid relevant docs naturally have lower recall unless retrieval is proportionally better.

**Worked numerical example:**
Suppose for the query *"What is Apple's return policy for AirPods?"* there are **4 truly relevant chunks** in your knowledge base (ground truth, labeled by a human). Your retriever returns the top 5 chunks, and 3 of those 5 happen to be relevant.

```
relevant ∩ top-5 = 3   (3 of the true 4 relevant chunks were found)
|relevant|        = 4   (4 relevant chunks exist in total)

Recall@5 = 3 / 4 = 0.75  →  75% recall
```

**Why it matters in practice:** Recall@k is the single most fundamental retrieval metric because **nothing downstream can fix a document that was never retrieved**. If the answer is in a chunk that never made it into top-k, no amount of clever prompting or reranking will save the final answer. This is why Module 5.6 (k-tuning) and Module 2.8 (chunk-size sweeps) are directly optimizing against Recall@k — it's the ceiling on everything else.

> **Business example:** An e-commerce support bot needs Recall@10 ≥ 0.9 on policy questions. If it's only hitting 0.6, you know immediately: don't bother tuning the LLM prompt yet, the retriever is losing 40% of the necessary evidence before generation even starts.

---

### 2. Precision@k

```
Precision@k = |relevant ∩ top-k| / k
```

**Plain English:** "Of the k things we actually showed, what fraction were worth showing?" Same numerator as Recall@k, but now the denominator is *k* (a fixed number you chose) instead of the total relevant count.

**Term-by-term:**
- `relevant ∩ top-k` — same hits as before.
- `k` — the fixed size of your retrieved set. This is a constant *you* pick, not a property of the query.

**What happens if each term changes:**
- **k increases** without more hits → precision *drops* (you're diluting your hit rate with more retrieved junk).
- This creates the classic **recall/precision trade-off**: raising k almost always helps recall but hurts precision.

**Worked numerical example (same query as above):**
```
relevant ∩ top-5 = 3
k                = 5

Precision@5 = 3 / 5 = 0.60  →  60% precision
```

**Why it matters — and why it matters *less* in RAG than in classic search UX:** In a traditional search engine (think Google results page), every irrelevant result you show costs the user's attention and time — they have to read and discard it themselves. In RAG, the LLM generator + reranker sit *between* retrieval and the human, so some irrelevant chunks in the top-k can be silently absorbed and ignored by the generator as long as the *relevant* chunks are also present. Precision@k is therefore mostly useful for measuring **retrieval noise/waste** (e.g., is your context window getting bloated with junk, driving up cost/latency) rather than being a direct proxy for final answer quality the way it is in web search.

> **Gotcha:** Don't over-index on Precision@k as your primary retrieval metric for RAG — that's a classic mistake carried over from traditional IR intuition. Recall@k matters more here because the generator can filter noise, but it can't invent missing evidence.

---

### 3. MRR (Mean Reciprocal Rank)

```
MRR = (1/|Q|) Σ 1/rank_i
```

**Plain English:** "On average, how close to the #1 spot was the *first* correct result?" You get the reciprocal of the position of the first relevant hit for each query, then average that across all your queries.

**Term-by-term:**
- `rank_i` — the position (1-indexed) of the first relevant result for query *i*. If the first relevant doc is ranked #1, `rank_i = 1`. If it's ranked #4, `rank_i = 4`.
- `1/rank_i` — the reciprocal. A rank of 1 gives a perfect score of 1.0; a rank of 4 gives a much lower 0.25. This reciprocal shape means the *penalty for being late* grows very fast — going from rank 1 to rank 2 costs you 0.5 points, but going from rank 9 to rank 10 barely costs you anything.
- `1/|Q|` and the `Σ` — sum the reciprocal ranks across every query, then divide by the number of queries to get the average.

**What happens if each term changes:**
- If the first relevant result moves from rank 1 → rank 2 → rank 3, its contribution drops 1.0 → 0.5 → 0.33 — a steeply diminishing curve. This means MRR is very sensitive to *early* position and barely distinguishes "found at rank 8" from "found at rank 20."
- MRR only ever looks at the *first* relevant hit per query — it completely ignores whether there's a 2nd, 3rd, or 4th relevant result also present. Two systems that both find the first relevant doc at rank 1 get the *same* MRR contribution for that query, even if one of them also surfaces 5 more relevant docs and the other surfaces zero more.

**Worked numerical example:**
Three queries, ranks of the first relevant result:
```
Query 1: first relevant result at rank 1  → 1/1 = 1.00
Query 2: first relevant result at rank 3  → 1/3 = 0.33
Query 3: first relevant result at rank 2  → 1/2 = 0.50

MRR = (1.00 + 0.33 + 0.50) / 3 = 1.83 / 3 = 0.61
```

**Why it matters in practice:** MRR is the right metric when there's typically **one clearly-best answer** and its position matters a lot — classic single-hop factoid QA ("What's Apple's HQ address?"). It's the wrong metric when a query genuinely has multiple valid relevant documents that should all be surfaced (e.g., "summarize all product recalls in 2024") — for that, use Recall@k or nDCG instead.

> **Business example:** A voice-assistant FAQ lookup ("what's your return window") has one canonical best chunk — MRR is perfect here because you care intensely about that chunk being rank 1 (so it can be read aloud immediately) vs. rank 5 (buried, effectively useless for a voice UI with no scrolling).

---

### 4. nDCG (normalized Discounted Cumulative Gain)

```
DCG@k  = Σ_{i=1}^{k} rel_i / log2(i+1)
nDCG@k = DCG@k / IDCG@k        (IDCG = DCG of the ideal/perfect ranking)
```

**Plain English:** "Give partial credit for graded relevance (not just yes/no), and discount that credit the further down the ranking a good result appears, then normalize against the best-possible ranking so the score is comparable across queries."

**Term-by-term:**
- `rel_i` — the *graded* relevance of the document at position *i*. Unlike Recall/Precision/MRR (which treat relevance as binary — relevant or not), nDCG allows relevance to be a scale, e.g. 0 = irrelevant, 1 = somewhat relevant, 2 = highly relevant, 3 = perfect match.
- `log2(i+1)` — the **position discount**. This is the denominator that shrinks the contribution of a document the further down the list it sits. At position 1, `log2(2) = 1` (no discount). At position 3, `log2(4) = 2` (half credit). At position 7, `log2(8) = 3` (one-third credit). This is a *softer* penalty curve than MRR's reciprocal — it still rewards early position but doesn't crash as violently.
- `DCG@k` — sum up (graded relevance ÷ position discount) across the top-k results. This is your raw, un-normalized score.
- `IDCG@k` — the DCG you'd get if the ranking were *perfect* (all the highest-relevance docs sorted in the best possible order). This is your ceiling.
- `nDCG@k = DCG@k / IDCG@k` — dividing by the ceiling normalizes the score to a 0–1 range, making it comparable across queries that have different numbers of relevant docs or different relevance distributions.

**What happens if each term changes:**
- If a highly-relevant document (`rel_i = 3`) moves from position 1 to position 5, its contribution drops from `3/log2(2) = 3.0` to `3/log2(6) ≈ 1.16` — a big penalty for burying a great result.
- If you only have binary relevance labels (0/1), nDCG degenerates toward something similar to MRR/Recall behavior — the graded-relevance benefit disappears, so there's no point using nDCG over simpler metrics if your labels aren't graded.

**Worked numerical example:**
Say for a query, your system retrieves 3 documents with graded relevance labels `[3, 0, 2]` (position 1 has relevance 3, position 2 has relevance 0, position 3 has relevance 2).

```
DCG@3 = 3/log2(2) + 0/log2(3) + 2/log2(4)
      = 3/1        + 0/1.585   + 2/2
      = 3.0         + 0.0       + 1.0
      = 4.0
```

Now compute the ideal ranking (sort relevances descending: `[3, 2, 0]`):
```
IDCG@3 = 3/log2(2) + 2/log2(3) + 0/log2(4)
       = 3.0         + 2/1.585   + 0
       = 3.0         + 1.26      + 0
       = 4.26
```

```
nDCG@3 = DCG@3 / IDCG@3 = 4.0 / 4.26 ≈ 0.94
```

A near-perfect 0.94 — the system found the right documents, just put the relevance-2 document one slot later than ideal.

**Why it matters in practice:** nDCG is the **most information-rich retrieval metric** — use it whenever (a) relevance isn't purely binary (some docs are "perfect," some are "okay," some are "irrelevant") and (b) ranking order genuinely matters for downstream use, not just "is it in top-k anywhere." This is especially relevant once you have a **reranker** (Module 5) in the pipeline, since the whole point of a reranker is to improve the *order*, not just the top-k membership — Recall@k would be blind to a reranker's contribution if it doesn't change which docs are in top-k, only their order. nDCG is also the metric of choice once you know that document position specifically affects what gets "sandwiched" at the start of the LLM's context window (Module 6.1's lost-in-the-middle effect) — top-1/top-2 position isn't just cosmetic, it materially affects generation quality.

---

### Retrieval metric comparison table (Interview-ready)

| Metric | Handles graded relevance? | Rewards ranking order? | Rewards finding *multiple* relevant docs? | Best for |
|---|---|---|---|---|
| **Recall@k** | No (binary) | No | Yes (implicitly, via the ratio) | "Is the evidence there at all" — the foundational check |
| **Precision@k** | No (binary) | No | No | Measuring retrieval noise/waste |
| **MRR** | No (binary) | Yes, but only the *first* hit | No | Single-hop factoid QA with one best answer |
| **nDCG** | Yes | Yes, with a smooth discount | Yes | Reranked pipelines, graded relevance labels, position-sensitive downstream use |

**Interview framing to memorize:** *"Recall@k answers 'is the evidence there at all' — it's the most fundamental metric because nothing downstream can work if this fails. MRR and nDCG answer 'is the evidence ranked well' — that matters more once you have a reranker in the pipeline and top-1/top-2 position specifically affects what gets sandwiched at the start of the context window."*

---

## Section B: Generation Metrics — "The RAG Triad"

These three metrics are specific to RAG (as opposed to generic text-quality metrics like fluency/grammar that apply to any generated text).

```
        ┌─────────────────────────────────────────┐
        │              THE RAG TRIAD                │
        │                                             │
        │   FAITHFULNESS         ANSWER RELEVANCE     │
        │  (answer vs context)   (answer vs question) │
        │         ╲                    ╱              │
        │          ╲                  ╱               │
        │           ╲                ╱                │
        │         CONTEXT RELEVANCE                   │
        │       (retrieved context vs question)        │
        └─────────────────────────────────────────┘
```

### 1. Faithfulness / Groundedness

**Plain English definition:** Does the generated answer's content follow *only* from the retrieved context, without adding unsupported claims? This is RAG's primary **hallucination detector**.

**Critical nuance (a favorite interview trap):** A faithful answer can still be *wrong*, if the retrieved context itself was wrong, outdated, or incomplete. Faithfulness only checks whether the *generator* introduced fabrication on top of whatever context it was given — it says nothing about whether that context was correct in the first place. Don't conflate "faithful" with "correct."

**How it's measured:**
1. Decompose the generated answer into individual claims/sentences.
2. For each claim, check whether it's entailed by (i.e., logically supported by) the retrieved context — using either an NLI (Natural Language Inference / entailment classification) model, or an LLM-judge prompt.
3. Faithfulness score = fraction of claims that are supported.

This is literally the same mechanism as Module 6.2's post-hoc citation verification — the only difference is that here it's framed as an aggregate *metric* for eval purposes rather than a per-response runtime safety check.

**Worked numerical example:** Suppose the generated answer contains 5 discrete claims. An NLI model checks each claim against the retrieved context chunks:
```
Claim 1: "AirPods Pro have a 6-hour battery life"     → Supported ✓
Claim 2: "They include active noise cancellation"     → Supported ✓
Claim 3: "The return window is 14 days"               → Supported ✓
Claim 4: "AirPods Pro launched in 2019"                → NOT supported ✗ (context never mentions launch year)
Claim 5: "They are water resistant (IPX4)"             → Supported ✓

Faithfulness = 4/5 supported claims = 0.80
```
Claim 4 is a hallucination — the model pulled it from parametric (pretrained) knowledge rather than the retrieved context, even though it happens to be factually true in the real world. Faithfulness penalizes it anyway, because the *system's job* was to ground itself in the retrieved evidence, not to freelance from memory.

---

### 2. Answer Relevance

**Plain English definition:** Does the generated answer actually address the user's question — regardless of whether it's grounded? A perfectly faithful answer can still be *irrelevant* if it accurately summarizes retrieved content that doesn't actually answer what was asked.

**How it's commonly measured:** Generate several *synthetic questions that the produced answer would answer* (i.e., reverse-engineer "what question does this answer look like it's answering?"), embed those synthetic questions, and compare their embedding similarity to the original user query. High similarity implies the answer is on-topic for what was actually asked; low similarity means the answer drifted off-topic.

**Worked numerical example:**
```
Original query: "What is Apple's battery replacement cost for AirPods Pro?"

Generated answer: "AirPods Pro have active noise cancellation and a
transparency mode, along with sweat and water resistance rated IPX4."

→ Generate synthetic questions this answer would address:
  Q1: "What features do AirPods Pro have?"
  Q2: "Are AirPods Pro water resistant?"

→ Embedding similarity between {Q1, Q2} and the original query
  ("battery replacement cost") is LOW — the answer is fully
  faithful to some context chunk about AirPods features, but it
  never touches battery replacement or cost at all.

Answer Relevance ≈ 0.15  (low — the model answered a different
                            question than what was asked)
```
This is a completely faithful answer (every claim is true and grounded) that is nonetheless a *bad* answer — which is exactly why you need both metrics, not just one.

---

### 3. Context Relevance

**Plain English definition:** Of the retrieved context actually passed to the generator, how much of it is relevant to the query? This is retrieval-adjacent (it's about what got retrieved) but is usually grouped with the generation metrics because it's computed via the same LLM-judge tooling as faithfulness/answer relevance, rather than via ground-truth relevance labels like Recall@k.

**How it differs from Recall@k (a common point of confusion — interviewers love this one):**
- **Recall@k** asks: "Of all the relevant documents that exist, did we fetch them?" (measures *completeness* against ground truth)
- **Context relevance** asks: "Of what we actually fetched and fed to the generator, how much of it was useful vs. noise?" (measures *purity/signal-to-noise* of what was retrieved — no ground-truth label list required, since it's usually LLM-judged per response)

You can have **high Recall@k and low context relevance simultaneously**: the single relevant document you needed made it into top-10, but the other 9 chunks in that context window are irrelevant noise, diluting the signal the generator has to work with.

**Worked numerical example:**
```
Retrieved 5 chunks for the query "AirPods Pro return policy":
Chunk 1: About the return policy               → Relevant
Chunk 2: About AirPods Pro charging case specs  → Irrelevant
Chunk 3: About the return policy (different clause) → Relevant
Chunk 4: About AirPods Max (wrong product line) → Irrelevant
Chunk 5: About warranty claims process          → Irrelevant

Context Relevance = 2 relevant chunks / 5 total chunks = 0.40
```
40% signal, 60% noise — even if Recall@k was a perfect 1.0 (both relevant chunks were found), the generator is now working with a context window that's mostly clutter, which raises the risk of both distraction (lost-in-the-middle) and hallucination.

---

### Why these three together, not just one (the actual exam-answer version)

They can fail **independently**, and the specific *combination* of failures tells you exactly where to look:

| Faithfulness | Answer Relevance | Context Relevance | Diagnosis |
|---|---|---|---|
| High | **Low** | Low | Model accurately summarized context that didn't address the question — a **retrieval-context-relevance problem** (wrong stuff was fetched) |
| **Low** | High | High | Right evidence was retrieved, but the model hallucinated on top of it anyway — a **pure generation-stage failure** |
| High | High | High | System is working well end-to-end |
| Low | Low | Low | Both retrieval and generation are failing — start with retrieval, since generation quality can't be assessed fairly on top of bad evidence |

This triangulation table is the actual diagnostic value of measuring all three — not just producing one end-to-end correctness score. It's exactly what Module 8 builds its debugging workflow on top of.

> **Why This Matters callout:** In a real interview, if you're asked "your RAG faithfulness is low, what do you check?" — do NOT just say "improve the prompt." The strong answer is: *check context relevance first*, because low faithfulness with low context relevance often means the model is grasping at straws with noisy context (a retrieval problem masquerading as a generation problem), whereas low faithfulness with high context relevance is a genuine generation-stage hallucination issue.

---

## Section C: Frameworks

| Framework | What it is | Metrics implemented | Primary workflow fit |
|---|---|---|---|
| **RAGAS** | Purpose-built RAG evaluation library | Faithfulness, answer relevance, context relevance (the triad) + context recall/precision, mostly via LLM-judge prompting | Benchmark/notebook analysis — the most commonly cited framework in RAG interviews specifically; know its metric names by heart |
| **TruLens** | Broader LLM-app observability/eval framework (not RAG-exclusive) | Its own naming for the same "RAG triad" concept (groundedness / answer relevance / context relevance) | Production observability — tracing/logging to debug *which specific step* in a chain produced a bad output |
| **DeepEval** | Pytest-style eval framework | RAG metric assertions written as unit tests | CI/CD — regression-tests a RAG pipeline the way you'd regression-test code |

**Interview-relevant distinction to state explicitly:** These tools mostly differ in **workflow integration** (CI-testing vs. production tracing vs. benchmark analysis) — not in the underlying metrics. The actual metrics (faithfulness, relevance, groundedness) are conceptually the same across all three frameworks and are computed via broadly the same LLM-judge mechanism under the hood. Saying this explicitly in an interview signals you understand the space isn't three competing metric philosophies, it's one metric philosophy with three different deployment wrappers.

> **Gotcha:** Don't describe RAGAS/TruLens/DeepEval as if they measure fundamentally different things — a common surface-level mistake. The differentiator is *where in your ML lifecycle* you'd reach for each one, not *what* they measure.

---

## Section D: LLM-as-Judge Evaluation

Most of the generation-side metrics above are, under the hood, implemented by **prompting an LLM to assess another LLM's output**. "Just use GPT-4/Claude to grade it" is not automatically reliable — you need to understand the mechanics and pitfalls.

### Prompt design principles

1. **Give the judge a clear rubric with explicit criteria** — not just "rate this 1–10." Vague criteria produce high-variance, low-reproducibility judgments (ask 10 different graders to "rate this 1-10" with no rubric and you'll get wildly different numbers; give them a specific checklist and agreement jumps).
2. **Ask for reasoning/rationale before the score** — chain-of-thought judging. Same mechanism as CoT improving reasoning generally: forcing the model to articulate *why* before committing to a number improves consistency.
3. **Decompose compound judgments** — don't ask one LLM call to simultaneously judge faithfulness AND relevance AND fluency in one shot. Separate calls per criterion reduce conflation (the judge anchoring on one dimension and letting it bleed into its score for another).

### Known biases (name these proactively in an interview — this is a favorite question)

| Bias | What happens | Mitigation |
|---|---|---|
| **Position bias** | When comparing two candidate answers side by side, the judge tends to favor whichever is presented first (or sometimes second), regardless of true quality | Evaluate both orderings and average, or randomize order across the eval set |
| **Verbosity bias** | LLM judges tend to rate longer answers as higher quality even when the extra length adds no real information | Explicitly instruct the judge to penalize unnecessary length; normalize/control for length when comparing configurations that differ systematically in output length |
| **Self-preference bias** | A judge model tends to rate outputs from *the same model family* it belongs to more favorably (e.g., using GPT-4 to judge GPT-4-generated answers vs. a different model's answers) | Use a different model family as judge than the one being evaluated; calibrate against human labels to catch systematic skew |

### Calibration

Periodically validate the LLM judge against a small **human-labeled sample** — compute agreement (correlation, or exact-match rate) between judge scores and human scores. This confirms the judge is actually tracking the thing you care about before you trust it at scale.

> **Why This Matters callout:** An uncalibrated judge is a plausible-looking number that may not mean what you think it means. This is the single most important sentence to remember from this section — LLM-judge scores *look* rigorous (they're numeric, reproducible-ish, scalable) but without a human-agreement check, you have no idea if a "0.85 faithfulness score" corresponds to what a human would actually call faithful.

---

## Section E: Building a Golden Eval Set

### Synthetic QA generation from your corpus

**The method:** Use an LLM to generate `(question, answer, source chunk)` triples directly from your own document corpus — for each chunk, prompt an LLM to write a question that chunk would answer.

**Why it's valuable:** Fast, scalable way to bootstrap an initial eval set without manual labeling — especially valuable **pre-launch**, when no real query logs exist yet and you have nothing else to evaluate against.

**Known weakness (a very common interview follow-up):** Synthetic questions tend to be **too literal/extractive** — they closely mirror the source chunk's own phrasing and vocabulary, since the LLM generating them is looking directly at the chunk while writing the question. This doesn't reflect how *real users* phrase queries — real queries are more paraphrased, more ambiguous, and sometimes multi-hop (requiring synthesis across multiple chunks). **A system that scores well on purely synthetic eval data can still underperform on real user query distributions** — this gap is the single most important caveat to volunteer whenever synthetic eval sets come up.

### Human-in-the-loop curation

- Have domain experts review/edit synthetic QA pairs
- Add genuinely hard cases: ambiguous questions, questions requiring synthesis across multiple chunks, adversarial phrasing
- Periodically mine real production query logs (once available) to keep the eval set representative of actual usage, not just synthetic/idealized queries

### Practical structure of a good golden set (memorize this checklist)

A well-built golden eval set should be a *mix* of:
1. **Easy single-hop factoid questions** — sanity-check baseline
2. **Multi-hop questions** (Module 4B) — requires combining info across chunks
3. **Questions with no good answer in the corpus** — tests whether the system correctly says "I don't know" rather than hallucinating a confident-sounding wrong answer. This slice is important and frequently neglected — teams often build eval sets entirely out of answerable questions and never test the "graceful refusal" behavior at all.
4. **Paraphrased/adversarially-phrased versions of the same underlying question** — tests robustness to *phrasing*, not just content coverage

> **Gotcha:** If your golden set skips category 3 (no-good-answer questions), you have zero visibility into your system's hallucination behavior on genuinely unanswerable queries — which in production is often where the worst, most embarrassing failures happen (a confident wrong answer is worse than a correct "I don't know").

---

## Section F: Online vs. Offline Evaluation

| | Offline Evaluation | Online Evaluation |
|---|---|---|
| **What it measures** | Full eval set run against retrieval/generation metrics in a controlled, repeatable setting | Real user behavior signals in production |
| **Signals used** | Recall@k, MRR, nDCG, faithfulness, answer relevance, context relevance | Click-through/dwell time on cited sources, thumbs up/down, follow-up-question rate, session abandonment |
| **Speed / cost** | Cheap, fast, repeatable — the standard "regression test" workflow | Noisier, slower to accumulate signal |
| **Coverage** | Limited to whatever queries/labels exist in the (necessarily finite, potentially stale) golden set | Captures the *real* query distribution and real user judgment |
| **Confounds** | None from outside the system (it's a controlled test) | Confounded by factors outside the RAG system itself — UI issues, ambiguous user intent, unrelated product changes |

**A signal worth calling out specifically:** a high rate of immediate rephrased follow-up questions can itself be a proxy metric — it often signals the first answer was unsatisfying, even without an explicit thumbs-down.

### A/B testing retrieval changes

Since offline eval sets can't perfectly predict real-world impact (this is the exact same synthetic-eval weakness from Section E resurfacing at the system level), meaningful retrieval/reranking/chunking changes are typically validated with a **live A/B test**: split traffic, compare online metrics (thumbs-up rate, follow-up rate) between the old and new configuration, before fully rolling out.

**Standard practice / the funnel to describe in an interview:**
```
   Candidate change
         │
         ▼
   ┌─────────────┐
   │ Offline eval │  ← fast, cheap pre-filter to catch regressions
   └─────────────┘
         │  passes?
         ▼
   ┌─────────────┐
   │  A/B test    │  ← real-world confirmation, slower/costlier
   └─────────────┘
         │  wins?
         ▼
   Full rollout
```
Going straight to A/B testing *every* candidate change is too slow and expensive to iterate with — offline eval exists precisely to filter out the obviously-bad candidates cheaply before you spend the time/traffic budget on a live test.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What are the two evaluation surfaces in a RAG system, and why can't you rely on a single end-to-end score?

<details>
<summary>Show answer</summary>

Retrieval (did we fetch the right evidence?) and generation (given good evidence, did we write a good answer?). A single end-to-end correctness score conflates the two — you can't tell from a wrong answer alone whether retrieval fetched the wrong evidence, or fetched the right evidence but generation hallucinated anyway, or fetched the right evidence and used it correctly but the underlying source document itself was simply wrong or outdated. Separate metrics let you localize which stage to fix, and the fix for each stage is completely different (chunking/embedding tuning vs. prompt/context-ordering tuning).
</details>

---

**Q2 (Easy — calculation).** For a query, there are 6 relevant documents in the ground truth. Your system retrieves the top 8 results, and 4 of them are relevant. Compute Recall@8 and Precision@8.

<details>
<summary>Show answer</summary>

```
Recall@8    = 4/6 ≈ 0.67  (67%)
Precision@8 = 4/8 = 0.50  (50%)
```
</details>

---

**Q3 (Medium — conceptual).** Why does Precision@k matter less in RAG than in traditional web search?

<details>
<summary>Show answer</summary>

In traditional search, every shown result costs the user's attention directly — a human has to read and discard irrelevant results themselves. In RAG, a reranker and the LLM generator sit between retrieval and the human, and can tolerate some irrelevant chunks in the top-k as long as the relevant ones are also present — the generator can effectively "skip over" noise. Precision@k is still useful for measuring retrieval noise/waste (context bloat, cost, latency), just not as a direct proxy for final answer quality.
</details>

---

**Q4 (Medium — calculation).** A query's retrieved ranking has graded relevance `[2, 3, 0]` at positions 1, 2, 3. Compute nDCG@3. (Hint: the ideal ranking sorts relevance descending.)

<details>
<summary>Show answer</summary>

```
DCG@3 = 2/log2(2) + 3/log2(3) + 0/log2(4)
      = 2/1 + 3/1.585 + 0
      = 2.0 + 1.89 + 0
      = 3.89

Ideal ranking = [3, 2, 0]
IDCG@3 = 3/log2(2) + 2/log2(3) + 0/log2(4)
       = 3.0 + 1.26 + 0
       = 4.26

nDCG@3 = 3.89 / 4.26 ≈ 0.91
```
The system found the right documents but put the higher-relevance one (3) one position later than ideal, costing about 9% of the possible score.
</details>

---

**Q5 (Medium — diagnostic / "spot the problem").** Your RAG system has high faithfulness (0.95) but low answer relevance (0.30). What's likely going on, and how would you confirm it?

<details>
<summary>Show answer</summary>

High faithfulness means the generated claims are well-grounded in the retrieved context — the model isn't fabricating. Low answer relevance means the answer doesn't address what was actually asked. Put together, this is the classic pattern of the model accurately summarizing context that doesn't actually address the question — usually a **context relevance / retrieval problem** (wrong chunks were fetched and the model faithfully reported on them anyway) rather than a generation-stage hallucination problem. To confirm: check context relevance directly — if it's also low, retrieval is fetching off-topic content; look at the actual retrieved chunks for a few failing queries to see if they're topically adjacent but not actually responsive to the question.
</details>

---

**Q6 (Hard — conceptual + calculation combo).** Your RAG system has high Recall@k on your eval set, but production faithfulness scores are poor. What does this tell you, and what would you check next?

<details>
<summary>Show answer</summary>

High Recall@k means retrieval is doing its job — relevant documents are being fetched. Poor faithfulness despite that points to a **generation-stage failure**: the model is producing claims not well-supported by the retrieved context, independent of whether the context itself was correct. Next steps:
1. Check for **lost-in-the-middle** effects — is the relevant chunk buried in a long context and effectively ignored by the generator? (Module 6.1)
2. Check whether **context relevance** is actually low even though recall is high — recall can be satisfied by having the right document *somewhere* in top-k while it's still surrounded by a lot of irrelevant noise that distracts generation.
3. Inspect whether the prompt is adequately instructing the model to stick to the provided context versus drawing on its own parametric knowledge.
</details>

---

**Q7 (Hard — conceptual).** What's the major weakness of using synthetic LLM-generated QA pairs as your *only* eval set, and how would you address it?

<details>
<summary>Show answer</summary>

Synthetic questions are generated by an LLM looking directly at a source chunk, so they tend to closely mirror that chunk's own phrasing and vocabulary — systematically easier and more literal than real user queries, which are more paraphrased, sometimes ambiguous, and sometimes require synthesis across multiple chunks. A system that scores near-perfectly on a purely synthetic eval set can still underperform badly on real traffic. Address this by supplementing synthetic QA with human-curated hard cases (paraphrased, multi-hop, no-good-answer-exists questions), and once available, continuously mining real production query logs to keep the eval set aligned with the actual query distribution rather than relying solely on the initial synthetic bootstrap.
</details>

---

**Q8 (Hard — "spot the bug" scenario).** You're comparing two RAG configurations using an LLM judge. Configuration B consistently produces longer answers than Configuration A, and the judge rates B higher on every single query. Your teammate concludes "B is strictly better." What's the flaw in this conclusion, and how would you re-run the evaluation properly?

<details>
<summary>Show answer</summary>

This is a textbook case of **verbosity bias** — LLM judges tend to rate longer answers as higher quality even when the extra length doesn't add real information, and since B *systematically* differs in length from A, the judge's preference could be entirely explained by length rather than actual quality. To re-run properly: (1) explicitly instruct the judge to penalize unnecessary length / ignore length as a quality signal in the rubric, (2) have the judge also justify each verdict with reasoning tied to specific rubric criteria (faithfulness, relevance) rather than a bare score, and (3) calibrate a sample against human raters to see if humans agree B is actually better or if the human raters are unswayed by the length difference — this tells you whether the judge's preference reflects real quality or just the verbosity confound.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Treating Precision@k as equally important as Recall@k in RAG (it isn't — the generator can absorb some noise, but it can't invent missing evidence).
- ❌ Using MRR when queries can have multiple valid relevant answers (use Recall@k or nDCG instead — MRR only ever credits the first hit).
- ❌ Assuming "faithful" means "correct" (faithfulness only checks against the *retrieved* context — garbage-in, faithfully-reported-garbage-out is still possible).
- ❌ Describing RAGAS/TruLens/DeepEval as measuring fundamentally different things (they mostly differ in workflow integration, not in the underlying metrics).
- ❌ Trusting an LLM-judge score without ever calibrating it against human labels.
- ❌ Building a golden eval set entirely from synthetic QA with no human-curated hard cases or "no answer exists" questions.
- ❌ A/B testing every single candidate change instead of using offline eval as a cheap pre-filter first.
- ❌ Comparing two model configurations with an LLM judge without controlling for verbosity bias when output lengths differ systematically.

---

# 📌 One-Page Cheat Sheet (for last-minute review)

**Retrieval:** Recall@k (is it there?) → Precision@k (how much noise?) → MRR (how early, single best hit?) → nDCG (graded relevance + ranking order, most complete).

**Generation (RAG triad):** Faithfulness (grounded in context?) + Answer Relevance (addresses the question?) + Context Relevance (was the fetched context itself useful?). These three fail independently — use the combination to diagnose.

**Frameworks:** RAGAS (benchmark/notebook, most commonly cited), TruLens (production tracing/observability), DeepEval (CI/CD unit-test style). Same underlying metrics, different deployment wrapper.

**LLM-as-judge:** Rubric > vague scale. CoT reasoning before score. Decompose criteria into separate calls. Watch for position/verbosity/self-preference bias. Always calibrate against humans.

**Golden set:** Bootstrap with synthetic QA (fast, but too literal) → add human-curated hard cases (paraphrase, multi-hop, no-answer-exists) → refresh with real query logs once available.

**Offline vs. online:** Offline = cheap, repeatable, controlled, limited coverage. Online = real distribution, real judgment, noisy, confounded. Use offline as a pre-filter, A/B test as final confirmation before full rollout.

---

*End of Module 7. Next up in the series: Module 8 — Diagnosis & Debugging, which builds directly on the retrieval/generation split and the RAG-triad triangulation table established here.*
