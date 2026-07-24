# Chapter 14 — Hard Negative Mining

## What is it?

Dense retrieval (Chapter 4) trains a bi-encoder: one tower embeds queries, another embeds documents, and relevance is a dot product or cosine similarity in that shared space. But that training needs *negative* examples — documents that should score low for a given query — and **which negatives you choose is the single biggest lever on how good the resulting retriever actually is.**

**Hard negative mining** is the practice of selecting negatives that are *semantically close but actually irrelevant*, instead of negatives that are trivially unrelated. This is arguably the most-asked "how do you actually make dense retrieval work" question at the senior level, because it's where most of the real engineering effort in training a retriever goes.

---

## The intuition

**Observation 1 — the loss function needs contrast, not just positives.** Contrastive/triplet-style losses (used to train bi-encoders) pull the query embedding toward its true positive document and push it away from negative documents. If you never show the model a negative, it has no signal about what to push away from.

**Observation 2 — random negatives are too easy.** If you sample a random document from the corpus as a negative for the query "python list comprehension syntax," you'll likely get something like a cooking recipe — the model doesn't need to learn anything subtle to push that away; the embeddings are probably already far apart before training even starts. Training almost entirely on random negatives produces a model that's good at coarse topic separation but bad at fine-grained relevance — exactly the discrimination a real search ranker needs.

**Observation 3 — the negatives that matter are the ones the model currently gets wrong.** A document about "python: a large snake species" is lexically similar (shares the token "python") but semantically wrong for a coding query — this is a **hard negative**: close enough in embedding space (or lexical space) to be genuinely confusing, but actually irrelevant. Forcing the model to separate these teaches it the fine-grained boundary that a search system actually needs at inference time, because at inference time *all* the candidates it retrieves will already have passed a coarse relevance bar (they came from the same index) — the hard cases are exactly what it needs to resolve.

---

## The three main mining strategies

### 1. In-batch negatives (the baseline)

For a training batch of `B` (query, positive-document) pairs, treat every *other* document in the batch as a negative for each query.

```
Batch of B=4 (query, positive) pairs:
  (q1, d1+), (q2, d2+), (q3, d3+), (q4, d4+)

For q1: positive = d1+, negatives = {d2+, d3+, d4+}
For q2: positive = d2+, negatives = {d1+, d3+, d4+}
...
```

**Loss (softmax over in-batch negatives, "in-batch contrastive"):**

```
L(qi) = -log[ exp(sim(qi, di+)/τ) / Σⱼ exp(sim(qi, dj)/τ) ]
```

where `τ` is a temperature parameter and the sum in the denominator runs over the positive plus all in-batch negatives.

This is essentially free — no extra retrieval calls needed — and larger batch sizes give more (and often somewhat harder) negatives for free. But it's still fundamentally random with respect to any individual query, since the "other documents in this batch" are just whatever happened to be sampled together.

### 2. ANCE-style negatives (mining from the model itself)

**A**pproximate **N**earest neighbor **C**ontrastive **E**stimation. The idea: periodically re-index the *entire corpus* using the current (in-training) model's embeddings, then for each training query, retrieve its current top-k results using ANN search (Chapter 13) and pick negatives from documents that are *retrieved but not actually relevant*.

```
Every N training steps:
  1. Freeze current bi-encoder checkpoint
  2. Re-embed the full document corpus with it
  3. Rebuild the ANN index
  4. For each training query q, retrieve top-k candidates
  5. Negatives = candidates in top-k that are NOT labeled positive
```

This directly targets the model's *current* blind spots — by construction, these are documents the model currently thinks are relevant but aren't, which is exactly the error you want to correct next.

### 3. Cross-encoder-mined negatives (distillation-adjacent)

Use a separately-trained, more powerful but slower **cross-encoder** (which jointly encodes query+document together, so it can model token-level interactions the bi-encoder architecturally cannot) to score a large candidate pool, then select negatives the cross-encoder confidently rates low but that a cheap first-pass retriever (BM25 or an earlier bi-encoder) ranked high.

```
Candidate pool for query q: retrieved via BM25 top-100
Cross-encoder scores all 100 → sorts them
Negatives = candidates ranked 40-100 by cross-encoder
           (BM25 thought they were plausible; cross-encoder disagrees)
```

This produces the highest-quality hard negatives because the cross-encoder is a much stronger relevance judge — but it's expensive to run at the scale needed to mine negatives for millions of training queries, which is exactly why it's usually run periodically/offline rather than every training step.

---

## Worked numeric example

```
Query: "jaguar top speed"
True positive: doc about the Jaguar car's top speed spec

Candidate pool (BM25 top-5, before hard negative mining):
  d1: "Jaguar (car) 0-60 mph and top speed specifications"    ← TRUE POSITIVE
  d2: "Jaguar (animal) hunting speed and behavior"             ← hard negative (topic overlap: "jaguar" + "speed")
  d3: "Top speed records for Formula 1 cars"                   ← hard negative (topic overlap: "top speed" + cars)
  d4: "Jaguar Land Rover company history"                      ← medium negative (shares brand, not the query intent)
  d5: "Recipe for jaguar shark ceviche"                        ← easy/random negative (barely related)
```

**In-batch negative loss, before hard mining (temperature τ = 0.1):**

```
Suppose current model similarity scores:
  sim(q, d1+) = 0.82   (correct positive)
  sim(q, d5)  = 0.10   (random in-batch negative, model already separates this easily)

L = -log[ exp(0.82/0.1) / (exp(0.82/0.1) + exp(0.10/0.1)) ]
  = -log[ exp(8.2) / (exp(8.2) + exp(1.0)) ]
  = -log[ 3641 / (3641 + 2.72) ]
  = -log(0.99925)
  ≈ 0.00075   ← tiny gradient, model barely learns anything from this pair
```

**Same loss, using the hard negative d2 instead:**

```
Suppose sim(q, d2) = 0.61   (model is currently confused — "jaguar" + "speed" overlap fools it)

L = -log[ exp(0.82/0.1) / (exp(0.82/0.1) + exp(0.61/0.1)) ]
  = -log[ exp(8.2) / (exp(8.2) + exp(6.1)) ]
  = -log[ 3641 / (3641 + 445.9) ]
  = -log(0.8909)
  ≈ 0.1156   ← ~150× larger gradient signal than the random-negative case
```

This is the entire point of hard negative mining in one number: **the hard negative produces a ~150× larger loss (and gradient) than the easy negative**, because the model is actually uncertain about it. Training on easy negatives wastes most of your compute on examples the model already gets right.

---

## Why it works / why it fails

**Why it works:**
- Loss/gradient magnitude is driven by how *confused* the model currently is — hard negatives maximize that confusion by construction, so every training step teaches the model something it doesn't already know.
- Because hard negatives are mined from the model's own current retrieved candidates (ANCE-style) or from a stronger judge (cross-encoder-mined), the training distribution matches the actual *inference-time* distribution: at serving time, all candidates the model must rank came from a similar coarse-relevance bar, so training on similarly-confusable examples directly optimizes for the deployment scenario.

**Why it fails / risks:**
- **False negatives.** The biggest failure mode: a document mined as a "hard negative" because it's semantically close might actually *also* be relevant but simply unlabeled (common with sparse, incomplete relevance judgments) — training against it as a hard negative actively teaches the model to push away a document it should pull toward. This is why cross-encoder-mined negatives (a stronger judge double-checking) are often preferred over pure ANN-based mining alone.
- **Training instability from re-indexing.** ANCE-style mining requires periodically freezing and re-embedding the whole corpus — expensive, and if done too frequently, training becomes unstable as the negative distribution shifts under the model; if done too infrequently, you're training against a stale, no-longer-representative set of hard negatives.
- **Diminishing returns / over-hardening.** Mining negatives that are *too* hard (e.g., near-duplicates of the true positive) can hurt more than help — the model may start conflating genuinely relevant near-duplicates with negatives, hurting recall on legitimately similar-but-relevant documents. Most production pipelines mix hard-mined negatives with some random/in-batch negatives to keep training stable.

---

## The one thing to remember

The gradient signal from a negative example is proportional to how *confused the model currently is about it* — random negatives the model already separates easily contribute almost nothing to learning, so hard negative mining is really about spending your training compute on the examples that actually teach the model something.

---

## Formulas used in this chapter

| Formula | Meaning |
|---|---|
| `L(qi) = -log[exp(sim(qi,di+)/τ) / Σⱼ exp(sim(qi,dj)/τ)]` | In-batch contrastive loss (InfoNCE-style) over one positive and all in-batch negatives |
| Gradient magnitude ∝ model confusion | Larger loss (and gradient) comes from negatives the model currently scores close to the positive |

---

## Interview Q&A

**Q1. Why do random in-batch negatives alone produce a weaker retriever than one trained with hard negative mining, even with a very large batch size?**

Even with a large batch, in-batch negatives are sampled independently of the specific query — they're whatever other positives happened to land in the same batch. Most of them will be topically unrelated to any given query, so the model separates them almost immediately in training, and the resulting loss/gradient is near zero for most pairs (as the worked example shows — ~150x smaller gradient for an easy vs. hard negative). Larger batches give you more *chances* at an accidentally-hard negative, but they don't guarantee it, and you pay for all those wasted, uninformative comparisons in compute. Hard negative mining directly targets confusing examples instead of hoping to stumble onto them.

**Q2. What is the biggest risk in ANCE-style hard negative mining, and how would you mitigate it?**

The biggest risk is false negatives: since negatives come from the model's own current top-k retrievals, and relevance labels are typically sparse (only a few documents per query are labeled positive, not exhaustively), some "negatives" mined this way may actually be relevant but simply unlabeled — training against them teaches the model exactly the wrong lesson. Mitigations include: using a stronger cross-encoder to re-score and filter the mined negative pool before use, de-duplicating near-identical documents to the positive, and/or using denoised or multi-annotator labels where available so borderline cases are excluded rather than misused as hard negatives.

**Q3. Why would you use a cross-encoder to mine negatives instead of just training the cross-encoder as your retriever directly?**

Cross-encoders jointly encode the query and document together (full token-to-token attention across both), which lets them model relevance far more precisely than a bi-encoder's late dot-product — but that also means you cannot precompute document embeddings, so scoring is `O(1)` per query-document pair only at inference time with both present, making it far too slow to run over an entire corpus per query. The standard pattern is therefore to keep the bi-encoder as the actual retriever (fast, precomputable) but use the cross-encoder offline, as a teacher, to identify which bi-encoder-retrieved candidates are truly hard negatives (or, relatedly, to distill its judgments into the bi-encoder — the topic of the next chapter).

**Q4. How does the temperature parameter τ in the contrastive loss interact with hard negative mining?**

Temperature controls how sharply the softmax distinguishes similarity scores — lower τ makes the loss more sensitive to small differences in similarity (sharper distribution), while higher τ smooths it out. When you're training predominantly on random/easy negatives, a lower τ can help amplify a weak signal into a usable gradient. But once you're training with hard negatives, similarities between positive and negative are already close (as in the worked example: 0.82 vs 0.61), so an overly low τ can cause the loss to swing sharply for small, possibly noisy differences in similarity, destabilizing training — practitioners often need to re-tune τ (typically raise it slightly) when switching from random to hard-negative training regimes.

**Q5. In a production search system, how often would you re-mine hard negatives during training, and what factors drive that choice?**

This is a tradeoff between staleness and cost: re-mining requires freezing the current model, re-embedding the full corpus, and rebuilding the ANN index — all expensive at scale. Re-mining too rarely means you keep training against negatives that reflect an outdated version of the model's blind spots (by the time you've made progress, those "hard" negatives may no longer even be confusing to the current model, wasting compute again). Re-mining too frequently adds substantial infrastructure and compute overhead and can destabilize training as the negative distribution shifts under the model mid-training. In practice, teams pick a re-mining cadence (e.g., every few thousand steps or at each training epoch boundary) based on corpus size, embedding/indexing cost, and empirically monitoring whether retrieval metrics on a held-out set are still improving between re-mining rounds.
