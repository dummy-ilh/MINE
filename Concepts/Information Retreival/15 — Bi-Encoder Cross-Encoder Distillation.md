# Chapter 15 — Bi-Encoder ↔ Cross-Encoder Distillation

## What is it?

Chapter 14 introduced two architectures in passing: the **bi-encoder** (encodes query and document separately, so document embeddings can be precomputed and searched with ANN indexing) and the **cross-encoder** (encodes query and document *together*, with full attention across both, so it's far more accurate but can't precompute anything and must run at inference time per pair).

**Distillation** here means: train a cross-encoder to be an excellent (but slow) relevance judge, then use its outputs — not just hard labels, but its *scores* — as a training signal to make the bi-encoder better, transferring some of the cross-encoder's accuracy into an architecture that's actually cheap enough to serve as the first-stage retriever. This is the standard production pattern that resolves the fundamental tension in dense retrieval: **you need cross-encoder-level accuracy but bi-encoder-level speed, and you can't have an architecture that's both.**

---

## The intuition

**Observation 1 — the two architectures aren't a choice, they're a pipeline.** A production search stack almost never picks one or the other. It uses a cheap bi-encoder (or BM25) to retrieve a candidate set of ~100-1000 documents from a billion-document corpus, then a cross-encoder re-ranks just those candidates, because 100-1000 pairwise cross-encoder calls is affordable per query but a billion is not.

**Observation 2 — this pipeline has a gap.** The bi-encoder is what determines *which* 100-1000 documents even make it to the cross-encoder. If the bi-encoder's notion of "similar" is cruder than the cross-encoder's, the cross-encoder never even gets a chance to see the true best documents — they were filtered out at stage 1. **Recall at the retrieval stage caps the ceiling of what re-ranking can ever achieve.**

**Observation 3 — distillation closes that gap without paying the cross-encoder's cost at retrieval time.** Instead of just training the bi-encoder on binary (relevant/irrelevant) labels, you train it to *match the cross-encoder's full score distribution* over a candidate set — including matching its relative ranking of the negatives, not just its verdict on the positive. This transfers the cross-encoder's finer-grained judgment into the bi-encoder's embedding space, so the cheap first-stage retriever gets smarter without becoming slow.

---

## The mechanics

### Step 1 — train (or obtain) a strong cross-encoder teacher

Train a cross-encoder on labeled (query, document, relevance) triples using standard cross-entropy or pairwise ranking loss. Because it jointly attends over query+document tokens, it typically achieves meaningfully higher accuracy than any bi-encoder on the same data — this gap is exactly what gets distilled away.

### Step 2 — generate soft labels from the teacher

For each training query, take a candidate pool (e.g., BM25 or bi-encoder top-100) and score every candidate with the cross-encoder to get a full score distribution, not just a binary label:

```
Query q, candidates {d1, d2, d3, d4, d5}
Cross-encoder scores: [9.2, 7.8, 3.1, 2.9, 0.4]
```

### Step 3 — train the student bi-encoder to match this distribution

Convert both teacher and student scores to probability distributions (softmax with temperature `T`) and minimize KL divergence between them:

```
P_teacher(dᵢ) = softmax(teacher_score(q, dᵢ) / T)
P_student(dᵢ) = softmax(sim(q, dᵢ) / T)

L_distill = KL(P_teacher || P_student) = Σᵢ P_teacher(dᵢ) × log[P_teacher(dᵢ) / P_student(dᵢ)]
```

Often combined with the original hard-label contrastive loss from Chapter 14:

```
L_total = α × L_distill + (1 − α) × L_contrastive
```

where `α` balances how much to trust the teacher's soft judgments vs. the ground-truth hard labels.

---

## Worked numeric example

```
Query: "best hiking boots for wide feet"
Candidate pool: {d1, d2, d3, d4}

Cross-encoder (teacher) raw scores:
  d1 = 8.5   (review specifically addressing wide-foot hiking boots)
  d2 = 6.0   (general hiking boots review, mentions fit briefly)
  d3 = 2.0   (running shoes for wide feet — wrong category)
  d4 = 0.5   (unrelated hiking backpack review)
```

**Step 1 — teacher distribution (temperature T = 2.0):**

```
scores / T = [4.25, 3.0, 1.0, 0.25]
exp(scores/T) = [70.1, 20.1, 2.72, 1.28]
sum = 94.2

P_teacher = [0.744, 0.213, 0.029, 0.014]
```

**Step 2 — suppose the student bi-encoder currently gives:**

```
sim(q, d1) = 0.70, sim(q, d2) = 0.68, sim(q, d3) = 0.30, sim(q, d4) = 0.10
   (student barely distinguishes d1 from d2 — this is the gap we want to close)

scores / T = [0.35, 0.34, 0.15, 0.05]
exp(scores/T) = [1.419, 1.405, 1.162, 1.051]
sum = 5.037

P_student = [0.282, 0.279, 0.231, 0.209]
```

**Step 3 — KL divergence (the loss driving the gradient):**

```
KL = Σ P_teacher(i) × log(P_teacher(i) / P_student(i))

term(d1) = 0.744 × log(0.744/0.282) = 0.744 × log(2.638) = 0.744 × 0.970 = 0.722
term(d2) = 0.213 × log(0.213/0.279) = 0.213 × log(0.763) = 0.213 × (-0.271) = -0.058
term(d3) = 0.029 × log(0.029/0.231) = 0.029 × log(0.126) = 0.029 × (-2.073) = -0.060
term(d4) = 0.014 × log(0.014/0.209) = 0.014 × log(0.067) = 0.014 × (-2.703) = -0.038

KL ≈ 0.722 − 0.058 − 0.060 − 0.038 ≈ 0.566
```

The largest term by far is `d1` (0.722) — the loss is dominated by the fact that the teacher is confident d1 is the best match (74.4% of its probability mass) while the student is barely distinguishing it from d2 (28.2% vs 27.9%, nearly a coin flip). The gradient will push hardest on **separating d1 from d2** — exactly the discrimination the student most needs to learn, and exactly the kind of signal a binary hard label ("d1 is relevant, d2 is not, tie broken arbitrarily") would never have communicated with this much nuance.

---

## Why it works / why it fails

**Why it works:**
- Soft labels carry far more information per training example than hard labels. A hard label says "d1 good, d2/d3/d4 bad" (1 bit-ish of signal per candidate); the teacher's full score distribution says *how much better* d1 is than d2, and that d2 is meaningfully closer to correct than d3, which is closer than d4 — an ordinal, graded signal that directly shapes the geometry the bi-encoder learns.
- It decouples training-time cost from serving-time cost: the expensive cross-encoder only runs during offline training (as a teacher, scoring training candidates once), while the deployed bi-encoder remains a single cheap forward pass per document, fully precomputable.
- It naturally combines with hard negative mining (Chapter 14): the candidates you distill over are often exactly the bi-encoder's own current hard negatives, so you're simultaneously teaching the model what to push away from (contrastive loss) and by how much, relative to other confusable candidates (distillation loss).

**Why it fails / risks:**
- **Distillation is bounded by the teacher's own quality.** If the cross-encoder teacher has systematic blind spots or biases (e.g., over-weighting lexical overlap), the student inherits them — distillation transfers whatever the teacher knows, good or bad.
- **Temperature sensitivity.** Too low a temperature makes the teacher distribution nearly one-hot, discarding the nuanced ranking information among negatives that's the whole point of distillation; too high flattens it toward uniform, diluting the signal about which document is actually best.
- **Candidate pool coverage.** Distillation only teaches the student about the candidates it's shown. If the training candidate pool (e.g., from BM25 top-100) systematically misses a class of relevant documents, the student never learns to value that class — this is why iterative re-mining (Chapter 14, ANCE-style) is often layered with distillation, so the candidate pool keeps improving as the student improves.
- **Compute cost is real, just moved.** You still need to run the expensive cross-encoder over every (query, candidate) training pair at least once — this is a large one-time (or periodically-repeated) offline cost, not a free lunch; it's just paid at training time instead of serving time.

---

## The one thing to remember

Distillation transfers the *shape of the teacher's judgment* — not just its verdicts — into the student, and the richest signal comes specifically from cases where the teacher is confident but the student is still confused (as the worked example shows: the KL loss concentrated almost entirely on the one pair the student couldn't yet separate).

---

## Formulas used in this chapter

| Formula | Meaning |
|---|---|
| `P(dᵢ) = softmax(score(q,dᵢ) / T)` | Convert raw scores to a probability distribution, temperature T controls sharpness |
| `L_distill = KL(P_teacher \|\| P_student)` | Loss pulling the student's score distribution toward the teacher's |
| `L_total = α·L_distill + (1−α)·L_contrastive` | Combined loss mixing soft-label distillation with hard-label contrastive training |

---

## Interview Q&A

**Q1. Why not just deploy the cross-encoder as your retriever directly, if it's so much more accurate?**

Cross-encoders require both the query and document to be present simultaneously to compute a score (full cross-attention between them), so nothing can be precomputed offline — every query would require running the full model against every candidate document in the corpus, which is computationally infeasible at billion-document scale and real-time latency budgets. Bi-encoders precompute document embeddings once, offline, and reduce query-time work to a single forward pass on the query plus fast vector similarity search (ANN indexing, Chapter 13). Distillation exists specifically to import as much of the cross-encoder's accuracy as possible into the bi-encoder, since the bi-encoder is the only one of the two architecturally capable of serving as first-stage retrieval at scale.

**Q2. What's the practical difference between training a bi-encoder with hard labels versus with distillation from a cross-encoder?**

Hard labels give a binary or coarse graded signal (relevant/irrelevant, or maybe a 1-5 relevance scale from human raters) — informative, but sparse, and identical in strength regardless of how "close" a negative actually is to being relevant. Distillation from a cross-encoder's full score distribution gives continuous, relative signal: exactly how much more relevant d1 is than d2, and how d2 compares to d3, etc. As the worked example shows, this lets the loss concentrate gradient exactly where the student is most confused relative to the teacher, rather than treating all mistakes as equally important the way a hard-label loss would.

**Q3. How would you decide the temperature T for the distillation loss?**

Temperature controls how much of the teacher's ranking nuance survives into the training signal. Too low, and the teacher's softmax becomes nearly one-hot on its top choice — you lose the graded information about how the remaining candidates compare to each other, degenerating toward something close to a hard label. Too high, and the distribution flattens toward uniform, diluting the signal about which candidate is genuinely best. In practice, T is a hyperparameter tuned empirically on a held-out validation set by tracking downstream retrieval metrics (recall@k, NDCG) at several T values, often alongside checking that the teacher's own distribution at that T still meaningfully separates known-good from known-bad candidates.

**Q4. If your cross-encoder teacher has a known bias (say, it over-rewards lexical overlap between query and document), what happens to the student bi-encoder after distillation, and how would you address it?**

The student inherits the bias, because distillation is explicitly training the student to reproduce the teacher's score distribution — whatever systematic error the teacher has becomes a systematic error in the student too, and may even become more consistent/entrenched in the student since it's now baked into its embedding geometry rather than being one judge's occasional mistake. Mitigations include mixing in independent hard-label supervision alongside distillation (the `α` term in `L_total`) so ground truth can correct teacher errors, using multiple teachers/ensembling to average out individual biases, or explicitly auditing and re-training the teacher on cases where lexical overlap and true relevance diverge before using it for distillation at scale.

**Q5. How does distillation interact with the hard negative mining pipeline from the previous chapter — are they competing techniques or complementary?**

They're complementary and typically run together: hard negative mining (Chapter 14) determines *which* candidates the student trains on (finding the documents the current bi-encoder is currently confused about), while distillation determines *what signal* it receives about those candidates (a graded relative score from the cross-encoder rather than a binary label). In practice, a common production loop is: retrieve hard negatives with the current bi-encoder (or BM25), score the full candidate set with the cross-encoder to get soft labels, train the bi-encoder on the combined distillation + contrastive loss, then periodically re-mine negatives with the now-improved bi-encoder and repeat — each component improving the data the other operates on.
