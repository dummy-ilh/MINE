# RAG Interview Prep — Day 19
## Evaluation Deep Recap — Closed Book

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
