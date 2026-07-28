# RAG Module 7 — Evaluation

---

## 7.1 Why RAG needs two separate evaluation surfaces

RAG has two distinct components that can each fail independently (this framing sets up Module 8's diagnosis workflow): **retrieval quality** (did we fetch the right evidence?) and **generation quality** (given good evidence, did we produce a good answer?). Evaluating only the end-to-end answer conflates these — a wrong final answer could stem from either stage, or both, so evaluation needs metrics at *each* stage independently, not just an end-to-end pass/fail.

---

## 7.2 Retrieval metrics

All of these require a labeled eval set: (query, set of known-relevant documents/chunks).

**Recall@k**: of all truly relevant documents for this query, what fraction appear in the top-k retrieved results?
```
Recall@k = |relevant ∩ top-k| / |relevant|
```
The core metric for "did retrieval even fetch the right thing" — directly what Module 5.6's k-tuning sweep and Module 2.8's chunk-size sweep are optimizing against.

**Precision@k**: of the top-k retrieved results, what fraction are actually relevant?
```
Precision@k = |relevant ∩ top-k| / k
```
Matters less in RAG than in traditional search UX (where every shown result costs user attention) — since reranking (Module 5) and generation (Module 6) can tolerate some irrelevant chunks in the top-k as long as the *relevant* ones are also present. Still useful for measuring retrieval noise/waste.

**MRR (Mean Reciprocal Rank)**: for each query, take the reciprocal of the rank position of the *first* relevant result (1/rank), average across queries.
```
MRR = (1/|Q|) Σ 1/rank_i
```
Sensitive to how *early* the first relevant result appears — useful when there's typically one clearly-best answer and its position matters a lot (e.g. single-hop factoid QA). Doesn't reward finding *multiple* relevant results, only the first one.

**nDCG (normalized Discounted Cumulative Gain)**: accounts for *graded* relevance (not just binary relevant/irrelevant) and discounts the contribution of relevant results further down the ranking (logarithmic discount by rank position), normalized against the ideal possible ranking.
```
DCG@k = Σ_{i=1}^{k} rel_i / log2(i+1)
nDCG@k = DCG@k / IDCG@k   (IDCG = DCG of the ideal/perfect ranking)
```
The most information-rich retrieval metric — use it when relevance isn't purely binary (e.g. a document can be "highly relevant," "somewhat relevant," "irrelevant") and ranking order genuinely matters, not just whether relevant docs appear anywhere in top-k.

**Interview framing for choosing among these**: Recall@k answers "is the evidence there at all" (most fundamental — nothing downstream can work if this fails), MRR/nDCG answer "is the evidence ranked well" (matters more once reranking is in the pipeline and top-1/top-2 position specifically affects what gets sandwiched at the start of context, Module 6.1).

---

## 7.3 Generation metrics — the RAG triad

The three metrics most specific to RAG (distinct from generic text-generation quality metrics like fluency):

**Faithfulness / groundedness**: does the generated answer's content follow *only* from the retrieved context, without adding unsupported claims? This is the primary hallucination-detection metric for RAG specifically — a faithful answer might still be *wrong* if the retrieved context itself was wrong or incomplete, but faithfulness specifically checks generator-introduced fabrication, not retrieval correctness.
- Measured by decomposing the answer into individual claims/sentences and checking each against the retrieved context via NLI (entailment classification) or an LLM judge — same mechanism as Module 6.2's post-hoc citation verification, just framed as a metric rather than a runtime check.

**Answer relevance**: does the generated answer actually address the user's question (regardless of whether it's grounded)? A perfectly faithful answer can still be *irrelevant* if it accurately summarizes retrieved content that doesn't actually address what was asked.
- Common measurement approach: generate several *synthetic questions that the produced answer would answer*, embed and compare those synthetic questions against the original query — high similarity implies the answer is on-topic for what was actually asked.

**Context relevance** (a retrieval-adjacent metric, sometimes grouped with generation metrics because it's usually computed via the same LLM-judge tooling): of the retrieved context actually passed to the generator, how much of it is relevant to the query? Distinct from Recall@k — Recall@k asks "was the relevant document fetched," context relevance asks "how much of what was fetched and fed to the generator was actually useful, vs noise."

**Why these three together, not just one**: they can fail independently and diagnose different problems. High faithfulness + low answer relevance → the model is accurately summarizing context that doesn't address the question (a retrieval-context-relevance problem, or the model answering a different question than what was asked). Low faithfulness + high context relevance → the right evidence was retrieved but the model hallucinated on top of it anyway (a pure generation-stage failure). This triangulation is the actual diagnostic value of measuring all three, not just an end-to-end correctness score.

---

## 7.4 Frameworks

**RAGAS** — purpose-built RAG evaluation library, implements the faithfulness/answer-relevance/context-relevance triad (7.3) plus context recall/precision, largely via LLM-judge prompting under the hood. The most commonly cited framework in RAG interviews specifically — know its metric names.

**TruLens** — broader LLM-app observability/eval framework (not RAG-exclusive), implements a similar "RAG triad" concept (its own naming for groundedness/answer relevance/context relevance) with tracing/logging integration for debugging *which specific step* in a chain produced a bad output — more oriented toward production observability than one-off benchmark scoring.

**DeepEval** — pytest-style eval framework, lets you write RAG metric assertions as unit tests integrated into CI pipelines — most relevant framing: brings RAG evaluation into a standard software-testing workflow (regression testing a RAG pipeline the way you'd regression test code) rather than being just a notebook/analysis tool.

**Interview-relevant distinction**: these tools mostly differ in *workflow integration* (CI-testing vs production tracing vs benchmark analysis), not in the underlying metrics — the actual metrics (faithfulness, relevance, groundedness) are conceptually the same across all three, computed via broadly the same LLM-judge mechanism. Good to state this explicitly rather than listing them as if they measure fundamentally different things.

---

## 7.5 LLM-as-judge evaluation

Most of the generation-side metrics above (7.3) are, under the hood, implemented by prompting an LLM to assess another LLM's output — critical to understand the mechanics and pitfalls, since "just use GPT-4 to grade it" is not automatically reliable.

**Prompt design principles**:
- Give the judge a clear rubric with explicit criteria, not just "rate this 1-10" (vague criteria → high variance, low reproducibility judgments)
- Ask for reasoning/rationale before the score (chain-of-thought judging) — improves consistency, same mechanism as CoT improving reasoning generally
- Decompose compound judgments (don't ask one LLM call to simultaneously judge faithfulness AND relevance AND fluency — separate calls per criterion reduce conflation)

**Known biases to name proactively**:
- **Position bias**: when comparing two candidate answers, the judge tends to favor whichever is presented first (or sometimes second) regardless of true quality — mitigated by evaluating both orderings and averaging, or randomizing order across the eval set
- **Verbosity bias**: LLM judges tend to rate longer answers as higher quality even when the extra length isn't adding real information — a well-known confound when comparing models/configurations that differ systematically in output length
- **Self-preference bias**: an LLM judge tends to rate outputs from *the same model family* it belongs to more favorably — relevant when using, e.g., GPT-4 to judge GPT-4-generated RAG answers vs a different model's answers

**Calibration**: periodically validate the LLM judge against a small human-labeled sample — compute agreement (e.g. correlation or exact-match rate) between judge scores and human scores, to confirm the judge is actually tracking the thing you care about before trusting it at scale. An uncalibrated judge is a plausible-looking number that may not mean what you think it means.

---

## 7.6 Building a golden eval set

**Synthetic QA generation from your corpus**: use an LLM to generate (question, answer, source chunk) triples directly from your own document corpus — for each chunk, prompt an LLM to write a question that chunk would answer. Fast, scalable way to bootstrap an initial eval set without manual labeling, especially valuable pre-launch when no real query logs exist yet.
- **Known weakness**: synthetic questions tend to be *too literal/extractive* — they closely mirror the source chunk's phrasing (since the LLM generating them is looking directly at the chunk), which doesn't reflect how *real users* phrase queries (more paraphrased, more ambiguous, sometimes multi-hop). A system that scores well on purely synthetic eval data can still underperform on real user query distributions.

**Human-in-the-loop curation**: have domain experts review/edit synthetic QA pairs, add genuinely hard cases (ambiguous questions, questions requiring synthesis across multiple chunks, adversarial phrasing), and periodically mine real production query logs (once available) to keep the eval set representative of actual usage rather than only synthetic/idealized queries.

**Practical structure of a good golden set**: mix of easy single-hop factoid questions (sanity-check baseline), multi-hop questions (Module 4B), questions with no good answer in the corpus (tests whether the system correctly says "I don't know" rather than hallucinating — an important and often-neglected eval slice), and paraphrased/adversarially-phrased versions of the same underlying question (tests robustness to phrasing, not just content coverage).

---

## 7.7 Online vs offline evaluation

**Offline evaluation**: run the full eval set (7.6) against retrieval/generation metrics (7.2/7.3) in a controlled, repeatable setting before deploying a change — the standard "regression test" workflow, cheap to run frequently, but limited to whatever queries and relevance judgments are in the (necessarily finite, potentially stale) golden set.

**Online evaluation**: measure real user behavior signals in production — click-through/dwell time on cited sources, explicit thumbs-up/down feedback, follow-up-question rate (a high rate of immediate rephrased follow-ups can signal the first answer was unsatisfying), session abandonment. Captures real query distribution and real user judgment of quality, but is noisier, slower to accumulate signal, and confounded by factors outside the RAG system itself (UI issues, user intent ambiguity).

**A/B testing retrieval changes**: since offline eval sets can't perfectly predict real-world impact (synthetic eval weakness from 7.6), meaningful retrieval/reranking/chunking changes are typically validated with a live A/B test — split traffic, compare online metrics (thumbs-up rate, follow-up rate) between the old and new configuration, before fully rolling out. Standard practice: use offline eval as a fast pre-filter to catch regressions cheaply, reserve A/B testing for changes that pass offline eval and need real-world confirmation before full rollout — going straight to A/B testing every candidate change is too slow and expensive to iterate with.

---

## Interview Q&A drill

**Q: Your RAG system has high Recall@k on your eval set, but production faithfulness scores are poor. What does this tell you, and what would you check next?**
A: High Recall@k means the retrieval stage is doing its job — relevant documents are being fetched. Poor faithfulness despite that points to a generation-stage failure: the model is producing claims not well-supported by the retrieved context, independent of whether the context itself was correct. Next steps: check "lost in the middle" effects (is the relevant chunk buried in a long context and effectively ignored, Module 6.1), check whether context relevance is actually high even though recall is (recall can be satisfied by having the *right document* somewhere in top-k while still surrounding it with a lot of irrelevant noise that distracts generation), and inspect whether the prompt is adequately instructing the model to stick to provided context versus drawing on parametric knowledge.

**Q: Why can't you just use one end-to-end "is the answer correct" metric instead of separate retrieval and generation metrics?**
A: A single end-to-end correctness score conflates two independently-failing stages — you can't tell from a wrong answer alone whether retrieval fetched the wrong evidence, or fetched the right evidence but generation hallucinated anyway, or fetched the right evidence and generation used it correctly but the underlying source document itself was simply wrong/outdated. Separate retrieval metrics (Recall@k, MRR, nDCG) and generation metrics (faithfulness, answer relevance) let you localize which stage to actually fix — critical for the diagnosis workflow in Module 8, since the fix for a retrieval failure (chunking, hybrid search tuning) is completely different from the fix for a generation failure (prompt structure, context ordering).

**Q: What's a major weakness of using synthetic LLM-generated QA pairs as your only eval set, and how would you address it?**
A: Synthetic questions are generated by an LLM looking directly at a source chunk, so they tend to closely mirror that chunk's own phrasing and vocabulary — they're systematically easier and more literal than real user queries, which are more paraphrased, sometimes ambiguous, and sometimes require synthesis across multiple chunks. A system that scores near-perfectly on a purely synthetic eval set can still underperform badly on real traffic. Address this by supplementing synthetic QA with human-curated hard cases (paraphrased, multi-hop, no-good-answer-exists questions) and, once available, mining real production query logs to continuously refresh the eval set toward the actual query distribution rather than relying solely on the initial synthetic bootstrap.

**Q: What are the main biases to watch for when using an LLM as a judge, and how do you mitigate each?**
A: Position bias — the judge favors whichever answer appears first/second in a comparison regardless of quality; mitigate by evaluating both orderings and averaging, or randomizing presentation order across the eval set. Verbosity bias — judges tend to rate longer answers as better even without added informational value; mitigate by explicitly instructing the judge to penalize unnecessary length, or normalizing/controlling for length when comparing configurations. Self-preference bias — a judge model tends to favor outputs from its own model family; mitigate by using a different model family as judge than the one being evaluated, or by periodically calibrating judge scores against a human-labeled sample to catch systematic skew before trusting the judge at scale.

---

**Next up: Module 8 — Diagnosis & debugging.** Say the word when ready.
