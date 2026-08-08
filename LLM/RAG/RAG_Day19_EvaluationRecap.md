# RAG Interview Prep — Day 19
# RAG Evaluation — Master Notes

> Two independent failure surfaces. One metric can't tell you which broke.

---

## The Pipeline

```
Query → Retrieval → Context (top-k chunks) → Generation → Answer
          ↓                                      ↓
  Recall@k · MRR · nDCG              Faithfulness · Answer Relevance
                                         · Context Relevance
```

**Core principle:** A wrong final answer could come from retrieval fetching the wrong evidence, or from generation hallucinating on top of correct evidence. A single end-to-end score can't tell you which. You need separate metrics at each stage so you know *where* to fix.

---

## Part 1 — Retrieval Metrics

All retrieval metrics require a **labeled eval set**: `(query, set of known-relevant doc/chunk IDs)` pairs. Without ground-truth labels, none of these numbers mean anything.

### Notation

| Symbol | Meaning |
|---|---|
| `k` | Cutoff — number of top-ranked results you're evaluating |
| `relevant` | Full set of ground-truth relevant documents for a query |
| `top-k` | Documents your system actually retrieved at positions 1..k |
| `rank_i` | Position of the first relevant result for query *i* |
| `rel_i` | Graded relevance score of the doc at rank *i* (e.g. 0, 1, 2, 3) |
| `Q` | Set of all queries in your eval set |

---

### 1. Recall@k

```
Recall@k = |relevant ∩ top-k| / |relevant|
```

**Plain English:** Of everything that *should* have been found, what fraction did we surface in our top-k?

**Term by term:**
- `relevant ∩ top-k` — hits: relevant docs that also made it into your top-k
- `|relevant|` — total relevant docs that exist (the denominator, the full "correct answer" set)

**What changes if parameters shift:**
- k increases → Recall@k can only go up or stay flat (bigger net, can't lose existing hits)
- `|relevant|` increases → for the same number of hits, recall goes *down*

**Worked example:**

Query: *"What is Apple's return policy for AirPods?"*

Ground truth: **4** relevant chunks exist. Your retriever returns top-5; **3** of those 5 are relevant.

```
relevant ∩ top-5 = 3
|relevant|        = 4

Recall@5 = 3 / 4 = 0.75  →  75%
```

**Why it matters most:** Nothing downstream can fix a document that was never retrieved. Recall@k is the ceiling on everything else. If you're hitting 0.60 on policy questions, don't tune the LLM prompt — the retriever is losing 40% of the necessary evidence before generation even starts.

---

### 2. Precision@k

```
Precision@k = |relevant ∩ top-k| / k
```

**Plain English:** Of the k things we retrieved, what fraction were worth retrieving?

**Same worked example:**

```
relevant ∩ top-5 = 3
k                = 5

Precision@5 = 3 / 5 = 0.60  →  60%
```

**Why it matters less in RAG than in web search:** In traditional search, every irrelevant result costs the user's attention directly. In RAG, the LLM and reranker sit between retrieval and the human — some noise in top-k is absorbed silently as long as the relevant chunks are also present. Precision@k is most useful for measuring **retrieval noise and cost** (context bloat, token waste), not as a proxy for answer quality.

> **Common mistake:** Don't over-index on Precision@k as your primary RAG retrieval metric. That's a carry-over from classic IR intuition that doesn't translate.

---

### 3. MRR — Mean Reciprocal Rank

```
MRR = (1/|Q|) · Σ 1/rank_i
```

**Plain English:** On average, how close to position #1 was the *first* relevant result?

**Term by term:**
- `rank_i` — position of the first relevant result for query *i* (1-indexed)
- `1/rank_i` — the reciprocal: rank 1 → 1.00, rank 4 → 0.25. The penalty for being late grows fast (going rank 1→2 costs 0.50 pts; going rank 9→10 costs almost nothing)
- MRR only ever looks at the **first** relevant hit — it completely ignores whether a 2nd or 3rd relevant doc is also present

**Worked example:**

| Query | First relevant rank | Score |
|---|---|---|
| Q1 | 1 | 1/1 = 1.00 |
| Q2 | 3 | 1/3 = 0.33 |
| Q3 | 2 | 1/2 = 0.50 |

```
MRR = (1.00 + 0.33 + 0.50) / 3 = 0.61
```

**When to use it:** Single-hop factoid QA with one clearly-best answer ("What's Apple's HQ address?") where position matters a lot — especially voice UIs where only the top result gets read aloud.

**When not to use it:** Queries that have multiple valid relevant documents that should all be surfaced (e.g. "summarise all product recalls in 2024"). Use Recall@k or nDCG instead.

---

### 4. nDCG — Normalized Discounted Cumulative Gain

```
DCG@k  = Σ_{i=1}^{k}  rel_i / log₂(i+1)
nDCG@k = DCG@k / IDCG@k
```

where `IDCG@k` = DCG of the ideal (perfect) ranking.

**Plain English:** Give partial credit for graded relevance (not just yes/no), discount that credit logarithmically by position, then normalize against the best possible ranking so scores are comparable across queries.

**Term by term:**
- `rel_i` — graded relevance at position *i*: e.g. 0 = irrelevant, 1 = somewhat relevant, 2 = highly relevant, 3 = perfect
- `log₂(i+1)` — the position discount: position 1 → log₂(2) = 1.0 (no discount); position 3 → log₂(4) = 2.0 (half credit); position 7 → log₂(8) = 3.0 (one-third credit)
- `IDCG@k` — the ceiling: DCG you'd get if ranking were perfect. Dividing by it normalizes to 0–1.

**Worked example:**

Retrieved relevances at positions 1, 2, 3: `[3, 0, 2]`

```
DCG@3 = 3/log₂(2) + 0/log₂(3) + 2/log₂(4)
      = 3/1.000   + 0/1.585   + 2/2.000
      = 3.00      + 0.00      + 1.00
      = 4.00

Ideal ranking [3, 2, 0]:
IDCG@3 = 3/1.000 + 2/1.585 + 0/2.000
       = 3.00 + 1.26 + 0.00
       = 4.26

nDCG@3 = 4.00 / 4.26 ≈ 0.94
```

Near-perfect score — the right docs were found, the relevance-2 doc was just one slot later than ideal.

**When to use it:** Use nDCG when (a) relevance isn't purely binary and (b) ranking order genuinely matters downstream. Especially relevant once a **reranker** is in the pipeline — Recall@k would be blind to a reranker's contribution if it doesn't change *which* docs are in top-k, only their *order*. nDCG captures that.

---

### Retrieval Metric Comparison

| Metric | Graded relevance? | Rewards ranking order? | Rewards multiple hits? | Best for |
|---|---|---|---|---|
| Recall@k | No (binary) | No | Yes | "Is the evidence there at all?" — foundational |
| Precision@k | No (binary) | No | No | Retrieval noise / cost |
| MRR | No (binary) | First hit only | No | Single-answer factoid QA |
| nDCG | Yes | Log discount | Yes | Reranked pipelines, graded relevance, position-sensitive use |

**Interview framing:**
> "Recall@k answers 'is the evidence there at all' — the most fundamental metric because nothing downstream works if this fails. MRR and nDCG answer 'is the evidence ranked well' — that matters more once you have a reranker in the pipeline and top-1/top-2 position specifically affects what gets sandwiched at the start of the context window."

---

## Part 2 — Generation Metrics: The RAG Triad

Three metrics specific to RAG (distinct from generic text-quality metrics like fluency or grammar).

```
FAITHFULNESS           ANSWER RELEVANCE
(answer vs context)    (answer vs question)
         \                  /
          \                /
         CONTEXT RELEVANCE
       (context vs question)
```

---

### 1. Faithfulness / Groundedness

**Question it answers:** Does the generated answer's content follow *only* from the retrieved context, without adding unsupported claims?

**This is RAG's primary hallucination detector.**

**Critical nuance (common interview trap):** A faithful answer can still be *wrong* if the retrieved context itself was wrong or outdated. Faithfulness only checks whether the *generator* introduced fabrication on top of whatever context it was given.

**How it's measured:**
1. Decompose the generated answer into individual claims
2. For each claim, check whether it's entailed by the retrieved context (via NLI model or LLM judge)
3. Faithfulness = fraction of claims that are supported

**Worked example:**

Query: *"Describe AirPods Pro."*

| Claim | Verdict |
|---|---|
| "AirPods Pro have 6-hr battery life" | Supported ✓ |
| "Includes active noise cancellation" | Supported ✓ |
| "Return window is 14 days" | Supported ✓ |
| "AirPods Pro launched in 2019" | **Not in context ✗** |
| "Water resistant IPX4" | Supported ✓ |

```
Faithfulness = 4/5 supported = 0.80
```

Claim 4 is a hallucination — pulled from parametric knowledge even though it happens to be factually true. Faithfulness penalises it because the system's job is to ground itself in retrieved evidence.

---

### 2. Answer Relevance

**Question it answers:** Does the generated answer actually address the user's question — regardless of whether it's grounded?

A perfectly faithful answer can be completely *irrelevant* if it accurately summarises retrieved content that doesn't address what was asked.

**Common measurement approach:** Generate several synthetic questions that the produced answer would answer, embed those synthetic questions, and compare embedding similarity to the original query.

**Worked example:**

```
Original query: "What is Apple's battery replacement cost for AirPods Pro?"

Generated answer: "AirPods Pro have active noise cancellation and a
transparency mode, along with sweat and water resistance rated IPX4."

Synthetic questions this answer would answer:
  Q1: "What features do AirPods Pro have?"
  Q2: "Are AirPods Pro water resistant?"

Embedding similarity of {Q1, Q2} to "battery replacement cost" → LOW

Answer Relevance ≈ 0.15
```

This is a fully faithful answer (all claims grounded) that is nonetheless a bad answer — which is exactly why you need both metrics.

---

### 3. Context Relevance

**Question it answers:** Of the retrieved context passed to the generator, how much was actually relevant to the query?

**How it differs from Recall@k:**
- **Recall@k** — of all relevant docs that exist, did we fetch them? (completeness against ground truth)
- **Context relevance** — of what we fetched and fed to the LLM, how much was useful vs. noise? (signal-to-noise, no ground-truth label list required)

You can have **high Recall@k and low context relevance simultaneously**: the one relevant doc made it into top-10, but the other 9 chunks are irrelevant noise diluting the signal.

**Worked example:**

Query: *"AirPods Pro return policy"* — 5 chunks retrieved:

| Chunk | Content | Verdict |
|---|---|---|
| 1 | Return policy details | Relevant ✓ |
| 2 | Charging case specs | Irrelevant ✗ |
| 3 | Return policy (other clause) | Relevant ✓ |
| 4 | AirPods Max specs | Irrelevant ✗ |
| 5 | Warranty claims process | Irrelevant ✗ |

```
Context Relevance = 2/5 = 0.40
```

40% signal, 60% noise — even if Recall@k was 1.0, the generator is working with a context window that's mostly clutter.

---

### Diagnostic Table — Why All Three Together

They fail **independently**. The combination tells you exactly where to look:

| Faithfulness | Answer Relevance | Context Relevance | Diagnosis |
|---|---|---|---|
| High | **Low** | Low | Model accurately summarised off-topic chunks. **Retrieval problem** — wrong content was fetched. |
| **Low** | High | High | Right evidence retrieved, model hallucinated on top. **Pure generation failure** — fix the prompt or context ordering. |
| **Low** | High | Low | Noisy context AND hallucination. Check lost-in-the-middle, reranker, context ordering. |
| High | High | High | System working end-to-end ✓ |
| **Low** | **Low** | **Low** | Everything failing. Start with retrieval — generation quality can't be fairly assessed on top of bad evidence. |

> **Interview answer to "faithfulness is low, what do you check?"** Don't just say "improve the prompt." Check context relevance first — low faithfulness with low context relevance often means the model is grasping at straws with noisy context (a retrieval problem masquerading as a generation problem). Low faithfulness with high context relevance is a genuine generation-stage hallucination issue.

---

## Part 3 — Frameworks

| Framework | What it is | Workflow fit |
|---|---|---|
| **RAGAS** | Purpose-built RAG eval library. Implements faithfulness, answer relevance, context relevance (the triad) plus context recall/precision, via LLM-judge prompting. | Benchmark / notebook analysis. Most commonly cited in RAG interviews — know its metric names. |
| **TruLens** | Broader LLM-app observability framework (not RAG-exclusive). Its own naming for the RAG triad (groundedness / answer relevance / context relevance) with tracing and logging. | Production observability — debug which specific step in a chain produced a bad output. |
| **DeepEval** | Pytest-style eval framework. RAG metric assertions written as unit tests integrated into CI pipelines. | CI/CD — regression-test a RAG pipeline the way you'd regression-test code. |

**Key interview point:** These tools differ in **workflow integration**, not in the underlying metrics. The actual metrics (faithfulness, relevance, groundedness) are conceptually the same across all three, computed via broadly the same LLM-judge mechanism. State this explicitly — describing them as measuring fundamentally different things is a surface-level mistake.

---

## Part 4 — LLM-as-Judge Evaluation

Most generation-side metrics are implemented by prompting an LLM to assess another LLM's output. "Just use GPT-4 to grade it" is not automatically reliable.

### Prompt Design Principles

1. **Explicit rubric, not "rate 1–10"** — vague criteria produce high-variance, low-reproducibility judgements. Give a specific checklist per criterion.
2. **Ask for reasoning before the score** — chain-of-thought judging. Forcing the model to articulate *why* before committing to a number improves consistency.
3. **Separate calls per criterion** — don't ask one LLM call to judge faithfulness AND relevance AND fluency simultaneously. Separate calls reduce conflation.

### Known Biases

Name these proactively in interviews.

| Bias | What happens | Mitigation |
|---|---|---|
| **Position bias** | When comparing two answers side-by-side, the judge favours whichever is presented first regardless of quality. | Evaluate both orderings and average, or randomise order across the eval set. |
| **Verbosity bias** | LLM judges rate longer answers as higher quality even when extra length adds no real information. | Explicitly instruct the judge to penalise unnecessary length; normalise/control for length when comparing configurations that differ systematically in output length. |
| **Self-preference bias** | A judge model favours outputs from its own model family. | Use a different model family as judge than the one being evaluated; calibrate against human labels to catch systematic skew. |

### Calibration

Periodically validate the LLM judge against a small **human-labelled sample** — compute agreement (correlation or exact-match rate) between judge scores and human scores. This confirms the judge is tracking what you care about before trusting it at scale.

> **The most important sentence in this section:** An uncalibrated judge is a plausible-looking number that may not mean what you think it means.

---

## Part 5 — Building a Golden Eval Set

### The Four Slice Types

A well-built golden set must include all four:

| Slice | Purpose | Frequently neglected? |
|---|---|---|
| Easy single-hop factoid | Sanity-check baseline. One chunk answers it directly. | No |
| Multi-hop | Requires synthesising across 2+ chunks. Tests whether retrieval finds all necessary evidence. | Sometimes |
| No-answer-exists | Tests whether the system correctly says "I don't know" vs. hallucinating confidently. | **Yes — often skipped entirely** |
| Paraphrased / adversarial | Same underlying question, different phrasing. Tests robustness to wording, not just content coverage. | Sometimes |

> **If your golden set skips the "no-answer-exists" category**, you have zero visibility into hallucination behaviour on unanswerable queries — which is often where the worst, most embarrassing production failures happen.

### Synthetic QA Generation

**Method:** Use an LLM to generate `(question, answer, source chunk)` triples from your corpus. For each chunk, prompt an LLM to write a question that chunk would answer.

**Strengths:** Fast, scalable, no manual labelling required. Valuable pre-launch when no real query logs exist.

**Critical weakness:** Synthetic questions are generated by an LLM looking directly at the source chunk, so they closely mirror that chunk's own phrasing and vocabulary. Real user queries are more paraphrased, more ambiguous, and sometimes multi-hop. **A system that scores well on purely synthetic eval data can still underperform badly on real traffic.**

### Human-in-the-Loop Curation

- Have domain experts review and edit synthetic QA pairs
- Add genuinely hard cases: ambiguous questions, multi-hop, adversarial phrasing
- Once available, mine real production query logs to keep the eval set representative of actual usage

---

## Part 6 — Offline vs. Online Evaluation

| | Offline Evaluation | Online Evaluation |
|---|---|---|
| What it measures | Full eval set against retrieval / generation metrics in a controlled, repeatable setting | Real user behaviour signals in production |
| Signals | Recall@k, MRR, nDCG, faithfulness, answer relevance, context relevance | Click-through, thumbs up/down, follow-up rephrasing rate, session abandonment |
| Speed / cost | Cheap, fast, repeatable | Noisier, slower to accumulate signal |
| Coverage | Limited to the (finite, potentially stale) golden set | Captures the real query distribution and real user judgement |
| Confounds | None from outside the system | UI issues, user intent ambiguity, unrelated product changes |

**Signal worth calling out:** A high rate of immediate rephrased follow-up questions is a proxy metric — it often signals the first answer was unsatisfying, even without an explicit thumbs-down.

### The Correct Funnel

```
Candidate change
      │
      ▼
 Offline eval   ← fast, cheap pre-filter to catch regressions
      │ passes?
      ▼
  A/B test      ← real-world confirmation on live traffic
      │ wins?
      ▼
 Full rollout
```

Going straight to A/B testing every candidate change is too slow and expensive. Offline eval exists to filter out obviously-bad candidates cheaply before spending live traffic budget.

---

## Part 7 — Interview Q&A Drill

**Q: Why can't you just use one end-to-end correctness score?**

A single score conflates two independently-failing stages. You can't tell from a wrong answer alone whether retrieval fetched the wrong evidence, generation hallucinated on top of correct evidence, or the source document was simply wrong. Separate retrieval and generation metrics let you localise which stage to fix — and the fix for each is completely different (chunking / embedding tuning vs. prompt / context ordering).

---

**Q: Your RAG system has high Recall@k but poor production faithfulness. What does this tell you?**

High Recall@k means retrieval is doing its job — relevant documents are being fetched. Poor faithfulness despite that points to a generation-stage failure: the model is producing claims not supported by the retrieved context. Next steps: (1) check for lost-in-the-middle effects — is the relevant chunk buried and effectively ignored? (2) check whether context relevance is actually low even though recall is high — recall can be satisfied while the context window is still 60% noise, which causes hallucination via distraction. (3) inspect whether the prompt adequately instructs the model to stick to the provided context.

---

**Q: What's the major weakness of synthetic LLM-generated QA pairs as your only eval set?**

Synthetic questions are generated by an LLM looking directly at a source chunk, so they closely mirror that chunk's vocabulary — systematically easier and more literal than real user queries. A system that scores near-perfectly on synthetic data can still underperform badly on real traffic. Fix: supplement with human-curated hard cases (paraphrased, multi-hop, no-good-answer-exists) and, once available, continuously mine real production query logs.

---

**Q: What biases should you watch for when using an LLM as a judge?**

- **Position bias** — favours whichever answer appears first in a comparison. Mitigate by evaluating both orderings and averaging.
- **Verbosity bias** — rates longer answers as better regardless of information added. Mitigate by explicitly instructing the judge to penalise unnecessary length.
- **Self-preference bias** — a judge favours outputs from its own model family. Mitigate by using a different model family as judge, and calibrating against human labels.

---

**Q: High faithfulness, low answer relevance. What happened?**

The model faithfully summarised context that didn't address the question — a retrieval/context relevance problem, not a generation hallucination. Confirm by checking context relevance: if it's also low, the wrong chunks were fetched and the model accurately reported on them. The answer was faithful to the wrong evidence.

---

## Gotchas — Common Mistakes

- **✗** Treating Precision@k as the primary RAG retrieval metric. The LLM can absorb noise; it can't invent missing evidence. Recall@k matters more.
- **✗** Using MRR when queries have multiple valid relevant answers. MRR only credits the first hit — use Recall@k or nDCG instead.
- **✗** Assuming "faithful" means "correct." Faithfulness only checks whether the generator introduced fabrication. Garbage-in, faithfully-reported-garbage-out is still possible.
- **✗** Describing RAGAS / TruLens / DeepEval as measuring fundamentally different things. They differ in workflow integration, not in the underlying metrics.
- **✗** Trusting an LLM-judge score without calibrating it against human labels.
- **✗** Building a golden set with only synthetic QA and no "no-answer-exists" slice. Zero visibility into hallucination on unanswerable queries — often where the worst production failures occur.
- **✗** A/B testing every candidate change instead of using offline eval as a cheap pre-filter first.
- **✗** Comparing two configurations with an LLM judge without controlling for verbosity bias when output lengths differ systematically.
- **✗** Seeing high Recall@k + low faithfulness and blaming generation alone. Check context relevance first — recall can be satisfied while the context window is still mostly noise.

---

## One-Page Cheat Sheet

**Retrieval:**
Recall@k (is it there?) → Precision@k (how much noise?) → MRR (how early, single best hit?) → nDCG (graded relevance + ranking order, most complete)

**Generation (RAG triad):**
Faithfulness (grounded in context?) + Answer Relevance (addresses the question?) + Context Relevance (was the fetched context useful?) — these three fail independently; use the combination to diagnose which stage broke.

**Frameworks:**
RAGAS (benchmark / notebook, most cited), TruLens (production tracing), DeepEval (CI/CD unit tests). Same metrics, different deployment wrapper.

**LLM-as-judge:**
Rubric > vague scale. CoT reasoning before score. Decompose criteria into separate calls. Watch for position / verbosity / self-preference bias. Always calibrate against humans.

**Golden set:**
Bootstrap with synthetic QA (fast, but too literal) → add human-curated hard cases (paraphrase, multi-hop, no-answer-exists) → refresh with real query logs once available.

**Offline vs. online:**
Offline = cheap, repeatable, controlled, limited coverage. Online = real distribution, real judgement, noisy, confounded. Offline as pre-filter → A/B test as final confirmation → full rollout.

---

## 📋 How to run this review

This is a cold recap of your **Module 7 (Evaluation)** notes — the very first document in this curriculum — now viewed through the lens of everything covered in Days 1–18. Close Module 7 before starting. The questions below deliberately use **new numbers** (not the same worked examples from Module 7 itself) so you can't pattern-match from memory of the exact figures — you need to actually know the formulas. The final section connects evaluation concepts to the full pipeline you've now built up across three weeks.

---

## Section A — Retrieval Metrics (fresh numbers)

**A1 (calculation).** For a query, 7 relevant documents exist. Your system retrieves top-10, and 5 are relevant. Compute Recall@10 and Precision@10.

<details>
<summary>Show answer</summary>

```
Recall@10 = 5/7 ≈ 0.714
Precision@10 = 5/10 = 0.5
```
</details>

**A2 (calculation).** Three queries have first-relevant-result ranks of 2, 1, and 5. Compute MRR.

<details>
<summary>Show answer</summary>

```
1/2 + 1/1 + 1/5 = 0.5 + 1.0 + 0.2 = 1.7
MRR = 1.7/3 ≈ 0.567
```
</details>

**A3 (calculation).** A ranking has graded relevance [1, 3, 2] at positions 1-3. Compute nDCG@3.

<details>
<summary>Show answer</summary>

```
DCG@3 = 1/log2(2) + 3/log2(3) + 2/log2(4) = 1 + 1.89 + 1.0 = 3.89
Ideal order = [3,2,1]
IDCG@3 = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26 + 0.5 = 4.76
nDCG@3 = 3.89/4.76 ≈ 0.817
```
</details>

**A4.** Why does Recall@k matter more fundamentally than Precision@k in RAG specifically (not general search)?

<details>
<summary>Show answer</summary>
Nothing downstream (reranking, generation) can recover a document that was never retrieved at all — recall is the ceiling on everything else. Precision matters less because the generator/reranker can tolerate some irrelevant chunks in top-k, as long as the relevant ones are also present; it's mostly a noise/cost signal, not a hard blocker like recall.
</details>

---

## Section B — The RAG Triad

**B1.** A generated answer has 6 claims; an NLI check finds 4 supported by retrieved context. Compute faithfulness, and state the one thing this metric does NOT tell you.

<details>
<summary>Show answer</summary>

```
Faithfulness = 4/6 ≈ 0.667
```
It doesn't tell you whether the *retrieved context itself* was correct — a faithful answer can still be wrong if the context was wrong or outdated (tie-in to Day 17's over-reliance discussion: faithfulness also can't distinguish "correctly used good context" from "ignored good context and got lucky matching a parametric fact").
</details>

**B2.** Faithfulness is high, answer relevance is low. What's the diagnosis, and which earlier-week concept does this typically point back to?

<details>
<summary>Show answer</summary>
The model accurately summarized context that didn't actually address the question — usually a context-relevance/retrieval problem, not a generation problem. Points back to retrieval-stage query-document matching (Days 7-9) or possibly a chunking issue (Day 3) surfacing chunks that are topically adjacent but not actually responsive.
</details>

**B3.** Why is context relevance considered "retrieval-adjacent" but grouped with generation metrics?

<details>
<summary>Show answer</summary>
It's about what got retrieved (a retrieval-side property), but it's typically measured via the same LLM-judge tooling as faithfulness/answer relevance (not via ground-truth relevance labels like Recall@k), which is why it's grouped operationally with the generation-metric triad despite being conceptually about retrieval quality.
</details>

---

## Section C — LLM-as-Judge & Golden Eval Sets

**C1.** Name the three LLM-judge biases and one mitigation for each.

<details>
<summary>Show answer</summary>
Position bias (favors whichever answer is shown first/second) — mitigate by randomizing/averaging both orderings. Verbosity bias (favors longer answers regardless of added value) — mitigate by explicitly instructing the judge to penalize unnecessary length. Self-preference bias (favors outputs from its own model family) — mitigate by using a different model family as judge, or calibrating against human labels.
</details>

**C2.** Why can't a purely synthetic LLM-generated golden eval set alone reliably catch the failure modes from Day 17 (over-reliance on parametric knowledge, refusal miscalibration)?

<details>
<summary>Show answer</summary>
Synthetic QA generation mirrors the source chunk's own phrasing and typically produces "normal," non-adversarial questions where context and parametric knowledge usually agree — it won't naturally include the deliberately-constructed counterfactual (context contradicts a strong prior) or two-sided (both answerable and genuinely-unanswerable) examples needed to surface those specific failure modes. This is Module 7 §7.6's synthetic-eval weakness, now shown to have concrete consequences for two specific Day 17 failure modes, not just an abstract "may not generalize" concern.
</details>

---

## Section D — Frameworks & Online/Offline Eval

**D1.** What's the actual differentiator between RAGAS, TruLens, and DeepEval, if not the underlying metrics?

<details>
<summary>Show answer</summary>
Workflow integration: RAGAS is a benchmark/notebook-analysis tool, TruLens is production observability/tracing, DeepEval is CI/CD-style unit testing. The underlying metrics (faithfulness, relevance, groundedness) are conceptually the same across all three, computed via broadly similar LLM-judge mechanisms.
</details>

**D2.** Why is A/B testing not used for every candidate retrieval change, and what's the standard funnel?

<details>
<summary>Show answer</summary>
A/B testing is slower and more expensive than offline eval, so testing every candidate change live would be too costly to iterate with. Standard funnel: offline eval as a fast, cheap pre-filter to catch regressions → A/B test only for changes that pass offline eval and need real-world confirmation before full rollout.
</details>

---

## Section E — Full-Pipeline Synthesis (Evaluation × Everything Else)

**E1.** You need to evaluate a system using ColBERT-style reranking (Day 10) and agentic multi-hop retrieval (Day 16). Which retrieval metric would best detect a reranking improvement, and why would per-hop evaluation matter for the multi-hop component specifically?

<details>
<summary>Show answer</summary>
nDCG is best for detecting reranking improvement, since reranking reorders rather than discovers new candidates — Recall@k often won't move much, while nDCG's position-sensitivity directly captures the value of better ordering. For multi-hop, evaluating only the final answer's correctness would obscure *where* in the hop chain a problem occurred (Day 16/17's error propagation) — per-hop evaluation (checking each hop's retrieval quality and the correctness of intermediate facts) is needed to localize whether a failure originated early (corrupting everything downstream) or only at the final synthesis step, mirroring the same "measure each stage separately" principle Module 7 opened with, just applied recursively within a multi-hop pipeline.
</details>

**E2.** Design a golden eval set (Module 7 §7.6 + Day 17's additions) comprehensive enough to catch every major failure mode covered across this entire curriculum. List the required slices.

<details>
<summary>Show answer</summary>
Required slices: (1) easy single-hop factoid questions — baseline sanity check; (2) multi-hop/comparative questions — tests decomposition (Day 11) and agentic retrieval (Day 16), ideally with per-hop ground truth to localize error propagation; (3) paraphrased/adversarially-phrased questions — tests vocabulary-mismatch robustness (Day 7/11); (4) no-good-answer-exists questions — tests refusal calibration's false-answer side (Day 15/17); (5) genuinely-answerable-but-easy-to-wrongly-refuse questions — tests refusal calibration's false-refusal side (Day 17's two-sided framing); (6) counterfactual/context-contradicts-parametric-knowledge questions — tests over-reliance on parametric knowledge (Day 17), not covered by any of the above; (7) queries with contextual metadata implying recency (e.g. explicit dates) — tests whether recency signaling actually works (Day 17's mitigation). Most golden eval sets in practice only cover slices 1-4; slices 5-7 are the ones most commonly missing, and missing them is exactly why sophisticated-looking systems can still have real, undetected blind spots.
</details>

---

## 📊 Weak Spot Tracker

| Section | Topic | Got it cold? | Needs repair? |
|---|---|---|---|
| A | Retrieval metrics (fresh calculations) | ☐ | ☐ |
| B | RAG triad | ☐ | ☐ |
| C | LLM-as-judge & golden sets | ☐ | ☐ |
| D | Frameworks & online/offline | ☐ | ☐ |
| E | Full-pipeline synthesis | ☐ | ☐ |

**This is your last pure-recap day before Day 20's Diagnosis day** — if Section E felt hard, that's the actual signal to revisit before moving forward, since Diagnosis day assumes fluent cross-pipeline reasoning as a starting point, not a stretch goal.

---

*Next up — Day 20: Diagnosis & Debugging — using the retrieval/generation split and the full failure-mode taxonomy (Day 17) to systematically root-cause problems.*
