# Chapter 5 — Evaluation Metrics
### Mastery Edition

---

## The Big Picture First

Every metric in this chapter answers a version of the same question:

> **Did the right documents appear near the top of the list?**

But "right" and "near the top" mean different things depending on the product. Before you compute anything, you need to know:

```
1. Does relevance come in degrees, or is it binary (good/bad)?
2. Does the user look at the whole list, or stop at the first good result?
3. Is missing a relevant document a catastrophe or an inconvenience?
4. Are queries independent, or do users refine and iterate?
```

These four questions map directly to which metric you should use. The framework at the end of this chapter makes those choices mechanical.

---

## Setup — the running example

All metrics use the same query throughout so you can compare their behavior directly.

```
query: "deep learning optimization"

Corpus has 5 relevant documents total:
  D1 — "Adam optimizer for deep neural networks"         (highly relevant)
  D3 — "SGD and momentum in deep learning"               (relevant)
  D5 — "Gradient descent variants and learning rates"    (highly relevant)
  D7 — "Backpropagation and automatic differentiation"   (relevant)
  D9 — "Learning rate schedules and warmup strategies"   (somewhat relevant)

System A returned this ranked list:
  rank 1: D1  (highly relevant)     ✓
  rank 2: D2  (not relevant)        ✗
  rank 3: D3  (relevant)            ✓
  rank 4: D4  (not relevant)        ✗
  rank 5: D5  (highly relevant)     ✓

  D7 and D9 were not retrieved at all.
```

Binary relevance labels for AP/MRR: `rel(D1)=1, rel(D3)=1, rel(D5)=1, rel(D7)=1, rel(D9)=1`
Graded labels for NDCG: `rel(D1)=3, rel(D3)=2, rel(D5)=3, rel(D7)=2, rel(D9)=1, all others=0`

---

## Metric 1 — Precision@k

### What it measures

Of the top-k results you returned, what fraction are relevant?

```
P@k = (relevant documents in top k) / k
```

### Full calculation

```
Retrieved list: D1(✓), D2(✗), D3(✓), D4(✗), D5(✓)

P@1 = 1/1 = 1.000   ← D1 is relevant
P@2 = 1/2 = 0.500   ← D1 relevant, D2 not
P@3 = 2/3 = 0.667   ← D1, D3 relevant; D2 not
P@4 = 2/4 = 0.500
P@5 = 3/5 = 0.600
```

### Real product examples

**Where P@k is the right metric:**

E-commerce grid view — Amazon shows 48 products on a page. Users scan all of them. The fraction of relevant products on the page (P@48) matters more than whether the single best product is at position 1 vs 3.

Ad systems — an advertiser wants 3 of their ads to appear in the top 10 results. They care about P@10, not about the ordering within those 10.

**Where P@k misleads:**

```
System A ranking: D2(✗), D3(✓), D1(✓)   →  P@3 = 2/3 = 0.667
System B ranking: D1(✓), D3(✓), D2(✗)   →  P@3 = 2/3 = 0.667

Identical P@3. But System B is clearly better — it puts the relevant
results first. P@k can't see this. That's what MAP fixes.
```

### The precision-recall tradeoff — visualized

```
Imagine lowering a score threshold to return more results:

Threshold high (return 5 docs):   P=0.80, R=0.20  ← few retrieved, mostly relevant
Threshold medium (return 20 docs): P=0.50, R=0.60  ← more retrieved, some junk
Threshold low (return 100 docs):  P=0.10, R=0.95  ← almost everything retrieved, lots of junk

Return EVERYTHING → R=1.00, P≈0  (useless)
Return NOTHING    → R=0.00, P is undefined
```

The Precision-Recall curve plots P against R as threshold varies. A better system has higher precision at every recall level — its curve is "pushed up and to the right."

---

## Metric 2 — Recall@k

### What it measures

Of all relevant documents in the corpus, what fraction appeared in your top-k?

```
R@k = (relevant documents in top k) / (total relevant in corpus)
```

### Full calculation

```
Total relevant in corpus = 5 (D1, D3, D5, D7, D9)
Retrieved top 5: D1(✓), D2(✗), D3(✓), D4(✗), D5(✓)

R@1 = 1/5 = 0.200   ← found D1 only
R@3 = 2/5 = 0.400   ← found D1, D3
R@5 = 3/5 = 0.600   ← found D1, D3, D5  (D7 and D9 never retrieved)
```

### Real product examples

**Where recall is the right metric:**

Legal discovery — a law firm searches for all documents relevant to a lawsuit. Missing even one could lose the case. They'll tolerate returning 1,000 documents if needed (a human reviews them all) as long as nothing important is missed. Recall@1000 matters more than P@1000.

Medical literature search — a researcher surveying all studies on a drug interaction needs high recall. Missing a contradictory study is a scientific error.

Patent prior-art search — an inventor must find all patents that could invalidate their application. Missing one is expensive.

**Where recall misleads:**

```
The recall cheat: return every document in the corpus.
  R@N = 5/5 = 1.000  ← perfect recall
  P@N = 5/1000000 ≈ 0.000  ← completely useless

Recall without precision tells you nothing about whether the system is useful.
A random document generator can achieve recall=1 trivially.
```

**The right way to use recall:** as a lower bound. Set a minimum recall requirement (e.g., R@100 ≥ 0.95) for your retrieval stage, then optimize precision separately in the ranking stage.

---

## Metric 3 — AP and MAP

### What it measures

AP rewards systems where **relevant documents appear early in the list**. It computes precision at every rank where a relevant document appears, then averages those precisions.

```
AP = (1/R) × Σ_{k: rel(k)=1} P@k

where R = total relevant documents in corpus (not just retrieved)
```

MAP = mean of AP over all queries.

### Why AP sees what P@k misses

The key insight: precision is **sampled only at ranks where relevant docs appear.** Each relevant document "earns" a precision sample equal to the fraction of relevant docs found so far.

```
If a relevant doc appears at rank 1 (after only 1 doc retrieved) →
  P@1 = high — we've been very selective

If a relevant doc appears at rank 10 (after 10 docs retrieved) →
  P@10 = likely lower — means we retrieved 9 other docs before this one,
  some of which were probably irrelevant

AP punishes you for making the user wait.
```

### Full calculation

```
Retrieved: D1(✓) D2(✗) D3(✓) D4(✗) D5(✓)
Total relevant R = 5 (D1, D3, D5, D7, D9 — D7 and D9 never retrieved)

Relevant docs found at:
  rank 1 (D1): P@1 = 1/1 = 1.000
  rank 3 (D3): P@3 = 2/3 = 0.667
  rank 5 (D5): P@5 = 3/5 = 0.600
  D7: never retrieved → contributes 0
  D9: never retrieved → contributes 0

AP = (1/5) × (1.000 + 0.667 + 0.600 + 0 + 0)
   = (1/5) × 2.267
   = 0.453
```

### What happens if we reorder the same results

**System A (current):** D1(✓) D2(✗) D3(✓) D4(✗) D5(✓)
**System B (better ordering):** D1(✓) D3(✓) D5(✓) D2(✗) D4(✗)

```
System B AP:
  rank 1 (D1): P@1 = 1/1 = 1.000
  rank 2 (D3): P@2 = 2/2 = 1.000
  rank 3 (D5): P@3 = 3/3 = 1.000

AP_B = (1/5) × (1.000 + 1.000 + 1.000) = 3.0/5 = 0.600

AP_A = 0.453   (same docs retrieved, worse order)
AP_B = 0.600   (same docs retrieved, better order)

AP captured the ordering difference. P@5 for both = 3/5 = 0.600 (identical).
This is exactly what P@k misses and AP catches.
```

### MAP over multiple queries

```
query 1: "deep learning optimization"  → AP = 0.453
query 2: "transformer architecture"    → AP = 0.820
query 3: "reinforcement learning"      → AP = 0.310

MAP = (0.453 + 0.820 + 0.310) / 3 = 0.528
```

A system with MAP=0.60 beats one with MAP=0.52 if the difference is statistically significant across a large query set.

### The D6-never-retrieved penalty — visualized

```
Corpus has 5 relevant docs. You retrieve 3 of them perfectly at ranks 1, 2, 3.

AP = (1/5) × (P@1 + P@2 + P@3)
   = (1/5) × (1.0 + 1.0 + 1.0) = 0.600

Even a perfect system that retrieves 3 out of 5 relevant docs with perfect
precision scoring only 0.600. The two missing docs cap you at 0.600 no
matter how well you rank what you found.

This is the recall component baked into AP.
```

### Where MAP fails

AP treats all relevant documents equally. A perfect answer and a tangentially relevant answer get the same weight. Searching for "what is the capital of France?" — a document saying "Paris is the capital of France" and a document saying "France is a country in Europe with many cities including Paris" are both "relevant" under binary labels. AP can't distinguish them. That's what NDCG is for.

---

## Metric 4 — NDCG

### What it measures

NDCG handles the two things AP can't:

1. **Graded relevance** — not just relevant/not relevant, but how relevant
2. **Exponential position discount** — rank 1 is worth far more than rank 5

```
DCG@k = Σᵢ₌₁ᵏ  (2^relᵢ - 1) / log₂(i + 1)

NDCG@k = DCG@k / IDCG@k
```

The numerator `(2^rel - 1)` is the **gain** — it explodes with relevance:

```
rel=0 → gain = 2⁰-1 = 0
rel=1 → gain = 2¹-1 = 1
rel=2 → gain = 2²-1 = 3
rel=3 → gain = 2³-1 = 7

Going from rel=2 to rel=3 (good → perfect) is 7× vs 3× — more than doubling.
This reflects reality: the difference between a good result and a perfect result
is much bigger than the difference between a mediocre result and a good one.
```

The denominator `log₂(i+1)` is the **discount**:

```
rank 1: log₂(2) = 1.000  ← no discount
rank 2: log₂(3) = 1.585  ← 37% less weight than rank 1
rank 3: log₂(4) = 2.000  ← 50% less weight than rank 1
rank 5: log₂(6) = 2.585  ← 61% less weight than rank 1
rank 10: log₂(11) = 3.459 ← 71% less weight than rank 1
```

### Full calculation

```
Graded labels: D1=3, D2=0, D3=2, D4=0, D5=3 (D7=2, D9=1 but not retrieved)

DCG@5:
  rank 1 (D1, rel=3): (2³-1)/log₂(2) = 7/1.000 = 7.000
  rank 2 (D2, rel=0): (2⁰-1)/log₂(3) = 0/1.585 = 0.000
  rank 3 (D3, rel=2): (2²-1)/log₂(4) = 3/2.000 = 1.500
  rank 4 (D4, rel=0): (2¹-1)/log₂(5) = 0/2.322 = 0.000
  rank 5 (D5, rel=3): (2³-1)/log₂(6) = 7/2.585 = 2.708

DCG@5 = 7.000 + 0.000 + 1.500 + 0.000 + 2.708 = 11.208

Ideal ranking — sort all relevant docs by relevance: 3, 3, 2, 2, 1
  (D1=3, D5=3, D3=2, D7=2, D9=1)

IDCG@5:
  rank 1 (rel=3): 7/1.000 = 7.000
  rank 2 (rel=3): 7/1.585 = 4.416
  rank 3 (rel=2): 3/2.000 = 1.500
  rank 4 (rel=2): 3/2.322 = 1.292
  rank 5 (rel=1): 1/2.585 = 0.387

IDCG@5 = 7.000 + 4.416 + 1.500 + 1.292 + 0.387 = 14.595

NDCG@5 = 11.208 / 14.595 = 0.768
```

### The cost of burying a highly relevant document — quantified

The irrelevant D2 at rank 2 cost us `4.416 - 0.000 = 4.416 DCG points`. If D5 (rel=3) had been there instead:

```
Hypothetical rank 2 (D5, rel=3): 7/1.585 = 4.416 gained instead of 0.000
New DCG@5 = 7.000 + 4.416 + 1.500 + 0.000 + 0.000 = 12.916
NDCG@5 = 12.916 / 14.595 = 0.885
```

Moving D5 from rank 5 to rank 2 is worth 0.117 NDCG points. That's a huge improvement in IR terms, where 0.01 NDCG gains are considered meaningful. **NDCG quantifies exactly how much each swap is worth.**

### Why NDCG is the standard at Google/Bing/Meta

Web search has deeply graded relevance. Users searching "symptoms of appendicitis" distinguish between:
- The Mayo Clinic article listing every symptom clearly (rel=3 — perfect)
- A health blog with general abdominal pain information (rel=1 — tangential)
- A Wikipedia article on the appendix with a symptoms section (rel=2 — good)

Treating these as identically "relevant" would make it impossible to tell whether a model change actually improved results for the user. Graded NDCG captures this.

### When to use linear gain instead of exponential

The standard gain formula is `2^rel - 1`. But sometimes linear gain (`rel` directly) makes more sense:

```
Exponential gain penalizes low-relevance docs very softly (0, 1, 3, 7)
and rewards high-relevance docs aggressively.

If your relevance scale is already calibrated (0 to 5 by trained raters),
linear gain might better reflect the actual quality differences.

Most production systems use exponential — it's the default in academic benchmarks
and aligns with the intuition that "perfect" results matter much more than "okay" ones.
```

---

## Metric 5 — MRR (Mean Reciprocal Rank)

### What it measures

How quickly does the first relevant document appear?

```
RR(q) = 1 / rank_of_first_relevant_document_for_query_q

MRR = (1/|Q|) × Σ RR(q)
```

### Full calculation

```
Three queries for our system:

query 1: "deep learning optimization"
  ranked: D1(✓), D2(✗), D3(✓), D4(✗), D5(✓)
  first relevant at rank 1 → RR = 1/1 = 1.000

query 2: "convolutional neural networks"
  ranked: D6(✗), D7(✓), D8(✗), D9(✓), D10(✗)
  first relevant at rank 2 → RR = 1/2 = 0.500

query 3: "natural language processing transformers"
  ranked: D11(✗), D12(✗), D13(✓), D14(✗), D15(✗)
  first relevant at rank 3 → RR = 1/3 = 0.333

MRR = (1/3) × (1.000 + 0.500 + 0.333) = 1.833/3 = 0.611
```

### The sharp drop-off — why MRR is ruthless

```
first relevant at rank 1: RR = 1.000
first relevant at rank 2: RR = 0.500  ← loses half its value immediately
first relevant at rank 3: RR = 0.333
first relevant at rank 4: RR = 0.250
first relevant at rank 5: RR = 0.200
first relevant at rank 10: RR = 0.100
```

By rank 5, you've already lost 80% of the possible score. This matches the reality of voice search and QA: if Siri doesn't surface the right answer first or second, the user gives up or rephrases.

### Real product examples

**Where MRR is the right metric:**

Siri / Google Assistant — "Hey Siri, what year was the Eiffel Tower built?" The answer is either first or the system failed. There's no browsing.

Stack Overflow answer search — a developer searching for a specific error message wants the working solution at rank 1. They don't want to scroll through 10 partially relevant answers.

Autocomplete / query suggestion — which suggestion to show first when a user types three characters. Only one suggestion matters.

**Where MRR misleads:**

```
System A: returns one perfect result at rank 1, then 9 irrelevant results.
System B: returns 10 relevant results, with the first at rank 1.

MRR for both: 1.000  (identical — MRR can't see past the first hit)

But System B is obviously better for a user who browses beyond rank 1.
Use NDCG or MAP here.
```

---

## Metric 6 — F1 Score (and when it appears in IR)

You'll see F1 mentioned in IR contexts — it's the harmonic mean of precision and recall.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Why harmonic mean, not arithmetic?

```
System returns everything in corpus:
  Precision = 0.001  (tiny fraction is relevant)
  Recall = 1.000     (found everything)

Arithmetic mean: (0.001 + 1.000) / 2 = 0.500  ← looks decent, wrong
Harmonic mean:   2 × (0.001 × 1.000) / (0.001 + 1.000) = 0.002  ← correctly terrible
```

The harmonic mean punishes extreme imbalances. If either precision or recall is near zero, F1 collapses. It forces both to be high simultaneously.

### F1 vs MAP/NDCG — when to use F1

F1 is **not** a ranking metric. It doesn't care about order. It's appropriate when:
- You have a fixed retrieval set (not a ranked list) — e.g., a classifier that outputs "relevant" or "not relevant"
- You want a single number balancing precision and recall for a binary classification
- NLP tasks: named entity recognition, question answering extraction, document classification

Use MAP/NDCG when you have a ranked list. Use F1 when you have a set of binary predictions.

### Fβ — controlling the tradeoff

```
Fβ = (1+β²) × (Precision × Recall) / (β²×Precision + Recall)

β=1: equal weight on precision and recall (standard F1)
β=2: recall weighted 2× more (use when missing relevant docs is costly)
β=0.5: precision weighted 2× more (use when returning junk is costly)
```

Example: Medical document filtering. Missing a relevant paper (low recall) is much worse than including an irrelevant one (low precision). Use F2 to penalize missed recalls more.

---

## Choosing the Right Metric — The Full Decision Framework

This is the exam question every FAANG interviewer is building toward. Learn this cold.

### Step 1 — Is relevance binary or graded?

```
Binary (good/bad):
  → MAP, MRR, P@k, R@k, F1

Graded (excellent/good/okay/irrelevant):
  → NDCG

If unsure: use NDCG. It reduces to MAP-like behavior with binary labels
and adds power with graded labels. No downside to graded.
```

### Step 2 — Does the user scan the whole list, or stop at the first good result?

```
User scans the whole list (browsing behavior):
  → MAP (binary), NDCG (graded)

User wants exactly one answer and stops:
  → MRR

User sees only a grid/page and doesn't care about within-page order:
  → P@k
```

### Step 3 — Is missing a relevant document a catastrophe?

```
Yes — missing one is costly (legal, medical, patent):
  → R@k, MAP (AP penalizes for missed docs)

No — a few relevant docs missed is fine:
  → P@k, MRR (both ignore docs not retrieved)
```

### Step 4 — Are you in the retrieval stage or the ranking stage?

```
Retrieval stage (candidate generation):
  Primary metric: Recall@k  (did the right docs make it into the candidate set?)
  Goal: k should be large enough that true top-10 is almost certainly in candidates

Ranking stage (reordering candidates):
  Primary metric: NDCG@10 or MAP@10  (is the best stuff at the top?)
  Goal: minimize position errors that hurt users
```

### The complete decision table

| Product context | Relevance | User behavior | Metric |
|---|---|---|---|
| Web search (Google) | Graded | Scans top 10 | NDCG@10 |
| Voice assistant (Siri) | Binary | First answer only | MRR |
| E-commerce grid (Amazon) | Graded | Scans whole page | NDCG@48 or P@48 |
| Legal discovery | Binary | Must find everything | R@1000 |
| Medical literature review | Binary | Must find everything | R@1000, MAP |
| Q&A (Stack Overflow) | Binary | First answer | MRR |
| Academic IR benchmarks | Binary | Full list | MAP |
| News recommendation | Graded | Scans feed | NDCG@10 |
| App Store search (Apple) | Graded | Scans top ~10 | NDCG@10 |
| Ad ranking | Binary | P@k (slot-based) | P@3 |
| Document retrieval stage | Binary | Does doc exist in candidates? | Recall@k |
| Candidate reranking stage | Graded | Full ranked list | NDCG@10 |

### The tricky cases

**Case: NDCG went up, MRR went down. Which matters?**

Find out what the user actually does. If the product is voice search or a chatbot → MRR is primary, investigate the regression. If it's a web search results page where users browse 5-10 results → NDCG is primary and the MRR drop may be acceptable. Run an A/B test with user engagement metrics (CTR on first result, session abandonment) to confirm which offline metric predicts real user behavior.

**Case: Two systems have the same MAP. Which do you deploy?**

Run a paired t-test on per-query AP scores. Same mean AP doesn't mean same per-query distribution — one system might consistently give AP=0.55, the other might give AP=0.90 on half the queries and AP=0.20 on the other half (high variance). Also check whether the improvement is concentrated on head queries (high volume, already well-served) or tail queries (where users actually suffer). A gain on tail queries with same MAP is often more valuable. Then A/B test — offline MAP improvement ≠ online improvement.

**Case: Recall@100 = 0.95. Is that good enough for the retrieval stage?**

Depends on what happens at the re-ranking stage and your latency budget. If your cross-encoder can re-rank 100 docs in time, 0.95 means 1 in 20 queries will have the correct answer missing from candidates — the re-ranker can never recover it. For most products, 0.95 is acceptable. For high-stakes applications (medical, legal), 0.99+ is required. Plot the recall@k curve and find the elbow point: the k where recall plateaus. That's your sweet spot.

---

## Common Confusions — Cleared Up

### "Higher AP means better ranking, right?"

Almost. AP measures both retrieval quality (recall component — did you find the relevant docs?) AND ranking quality (precision component — did you rank them first?). A system can have high AP by improving either. In the ranking stage (where candidate set is fixed), AP purely measures ordering. In the retrieval stage, it measures what you chose to retrieve.

### "Can I use NDCG with binary labels?"

Yes — set rel ∈ {0, 1}. The gain formula `2^1-1=1` and `2^0-1=0` makes NDCG equivalent to a ranked precision metric. You lose the benefit of graded relevance but keep the position weighting. It's a perfectly valid choice if you only have binary labels.

### "Why does AP divide by total relevant R, not retrieved relevant?"

Intentionally. It penalizes you for failing to retrieve relevant documents. If 10 docs are relevant and you only retrieve 3 of them (even perfectly ranked), your AP is capped at 3/10 = 0.30. Missing relevant documents is a retrieval failure — AP wants the metric to reflect that.

### "When is MAP misleading?"

When query difficulty varies enormously. A query with 100 relevant documents and a query with 1 relevant document both contribute one AP score each to MAP. The rare-document query is much harder — a miss there is catastrophic. Consider stratifying your evaluation by query type (head/torso/tail, easy/hard) and reporting MAP separately per stratum rather than as one global number.

---

## Worked Comparison — Same System, All Five Metrics

```
Retrieved: D1(✓) D2(✗) D3(✓) D4(✗) D5(✓)
5 relevant docs in corpus. Graded: D1=3, D3=2, D5=3, D7=2(not retrieved), D9=1(not retrieved)

P@5   = 3/5 = 0.600
R@5   = 3/5 = 0.600   (coincidence — same numbers here, different meaning)
AP    = (1/5)×(1.000 + 0.667 + 0.600) = 0.453
NDCG@5 = 11.208/14.595 = 0.768
MRR   = 1/1 = 1.000   (D1 is first result, and it's relevant)

Interpretation:
  MRR=1.000  ← looks perfect (first result is always relevant)
  NDCG=0.768 ← decent but loses points for irrelevant D2 at rank 2 and
                for not retrieving D7 and D9
  AP=0.453   ← lower because two relevant docs (D7, D9) were never found
  P@5=0.600  ← 40% of returned results are irrelevant
  R@5=0.600  ← missed 40% of all relevant documents

Same system. Five different numbers. Five different angles on quality.
An interviewer who asks "how good is this system?" wants you to say:
"it depends on which aspect matters for the product."
```

---

## Statistical Significance — The Part Everyone Skips

Computing MAP=0.723 vs MAP=0.689 means nothing without significance testing. You need to know if the difference is real or random variation across queries.

```
Test: paired t-test on per-query AP scores

null hypothesis: both systems have the same mean AP per query

Data:
  query 1: AP_A=0.80, AP_B=0.75  → difference = +0.05
  query 2: AP_A=0.50, AP_B=0.60  → difference = -0.10
  query 3: AP_A=0.90, AP_B=0.85  → difference = +0.05
  ...
  (over N=1000 queries)

Compute: mean difference, standard deviation of differences, t-statistic
If p < 0.05 → difference is statistically significant

Rule of thumb: you need at least 100-200 queries for reliable significance.
TREC benchmarks typically use 50 queries — sometimes considered too few.

Also check: Wilcoxon signed-rank test (non-parametric, if AP distribution is skewed)
```

**The bottom line:** Before declaring a model improvement, run significance tests. In production, NDCG differences of 0.5-1% are considered meaningful *if* statistically significant over a large query set.

---

## Summary — What to Remember

```
P@k    → fraction of returned results that are relevant (ignores order within k)
R@k    → fraction of all relevant docs that were returned (recall)
AP     → precision averaged at each relevant doc rank (both precision + recall + order)
MAP    → AP averaged across all queries (system-level summary)
NDCG   → position-weighted graded precision (best for most production systems)
MRR    → speed to first relevant result (voice, QA, one-answer tasks)
F1     → harmonic mean of P and R (for binary classification, not ranked lists)

Decision framework:
  Graded relevance?          → NDCG
  Binary relevance, browsing → MAP
  First result only?         → MRR
  Must find everything?      → Recall@k
  Fixed-set classification?  → F1 or Fβ

Always:
  Report metrics at multiple k values (NDCG@1, @5, @10)
  Test statistical significance before calling a result an improvement
  Validate offline metric gains with online A/B tests
  Stratify by query type — head vs tail can tell very different stories
```

---

## Quick Reference

| Formula | Meaning |
|---|---|
| `P@k = rel_in_top_k / k` | Fraction of top-k that are relevant |
| `R@k = rel_in_top_k / total_relevant` | Fraction of all relevant docs retrieved |
| `AP = (1/R) × Σ P@k × rel(k)` | Avg precision, sampled at relevant ranks |
| `MAP = (1/\|Q\|) × Σ AP(q)` | Mean AP over query set |
| `DCG@k = Σ (2^relᵢ-1) / log₂(i+1)` | Discounted cumulative gain |
| `NDCG@k = DCG@k / IDCG@k` | Normalized DCG, range [0,1] |
| `RR = 1 / rank_of_first_relevant` | Reciprocal rank for one query |
| `MRR = (1/\|Q\|) × Σ RR(q)` | Mean reciprocal rank over query set |
| `F1 = 2PR/(P+R)` | Harmonic mean of P and R |
| `Fβ = (1+β²)PR/(β²P+R)` | Weighted F-score (β>1 weights recall more) |
