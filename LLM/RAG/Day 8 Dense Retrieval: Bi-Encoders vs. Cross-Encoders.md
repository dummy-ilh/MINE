# Day 8 — Dense Retrieval: Bi-Encoders vs. Cross-Encoders

## The one-line mental model

A **bi-encoder** embeds query and document independently, then compares vectors with cosine similarity. A **cross-encoder** processes both jointly in a single forward pass, letting every token attend to every other token. This single architectural difference is the entire source of the speed/accuracy trade-off — and the reason every production RAG system uses both.

---

## Architecture

### Bi-encoder

```
Query ──▶ f(query) ──▶ vec_q ──┐
                                ├──▶ cosine_sim(vec_q, vec_d) ──▶ score
Doc   ──▶ g(doc)  ──▶ vec_d ──┘
```

- `g(doc)` is computed **once, offline**, for every document in the corpus
- At query time: only `f(query)` runs live → then an ANN lookup against precomputed vectors
- Total query-time latency: ~15–45 ms

### Cross-encoder

```
[Query + SEP + Document] ──▶ [JOINT encoder, full self-attention] ──▶ scalar score
```

- **Nothing can be precomputed.** The model needs both inputs together to produce any score
- A full forward pass runs per `(query, document)` pair, at query time, live
- Accurately models token-level interactions (e.g. "battery" in the query attending to "battery life" three sentences into a document) — something a bi-encoder's query-agnostic vector structurally cannot do

### Why cross-encoders are more accurate — the actual mechanism

A bi-encoder compresses an entire document into a fixed-length vector *before ever seeing the query*. It has no idea which aspects of the document will matter for some future query. This is an inherent information bottleneck.

A cross-encoder processes both inputs through self-attention together, so query tokens can attend directly to specific document tokens and vice versa — capturing fine-grained interactions specific to that exact `(query, document)` pairing.

> **Interview line:** Don't say "cross-encoders are more complex." Say: *"the joint attention is what a bi-encoder's independently precomputed, query-agnostic vector structurally cannot represent."*

---

## Symmetric vs. Asymmetric Bi-Encoders

| | Symmetric | Asymmetric (e.g. DPR) |
|---|---|---|
| Encoders | Same weights for query and doc | Separate encoder per input |
| Use when | Queries and docs are similar in style/length | Short queries vs. long passages (typical search) |
| Why asymmetric wins for search | Queries ("how do I reset my Apple ID?") are structurally very different from passages (a full paragraph of steps) — separate encoders let each specialize |

Most production retrieval setups — and most embedding models on the MTEB retrieval leaderboard — use asymmetric or asymmetrically-trained setups for exactly this reason.

---

## Latency math

One cross-encoder forward pass ≈ 20 ms.

| Task | Time |
|---|---|
| Bi-encoder: encode query + ANN over 10M docs | ~30 ms |
| Cross-encoder: rerank 50 candidates | 50 × 20 ms = **1 second** |
| Cross-encoder: rerank 500 candidates | 500 × 20 ms = **10 seconds** |
| Cross-encoder: score 10M docs (full corpus) | 10M × 20 ms = **55+ hours** |

The jump from "rerank 50" to "score full corpus" is the clearest argument for why a cross-encoder can never be a first-stage retriever over any real corpus.

---

## Two-Stage Retrieval (the production pattern)

```
Query
  │
  ▼
[Bi-encoder] ──▶ ANN index ──▶ top-100 candidates    (~15–45 ms)
                                        │
                                        ▼
                               [Cross-encoder reranker] ──▶ top-5 reranked   (~2 s for 100)
                                        │
                                        ▼
                                   LLM generates answer
```

Neither alone is sufficient:
- **Bi-encoder only** — fast, scalable, but misses nuanced query-document interactions
- **Cross-encoder only** — impossible over a large corpus (55+ hours for 10M docs)
- **Both together** — bi-encoder makes search possible at scale; cross-encoder makes the final ranking accurate

---

## Training: Contrastive Learning

### InfoNCE loss

$$L = -\log\frac{\exp(\text{sim}(q, d^+) / \tau)}{\sum_i \exp(\text{sim}(q, d_i) / \tau)}$$

**Plain English:** "Out of this positive document and a set of negatives, correctly assign the highest similarity to the positive one." Implemented as a softmax over similarity scores with cross-entropy against the known-correct positive.

**Temperature τ:**
- Low τ → amplifies small similarity differences → aggressive training signal, higher risk of overfitting
- High τ → softens the distribution → weaker gradient, slower learning

**Worked example** (τ = 1, 1 positive, 3 negatives):

```
sim(q, d+) = 0.8   negatives: 0.3, 0.1, 0.5

numerator   = exp(0.8) ≈ 2.226
denominator = exp(0.8) + exp(0.3) + exp(0.1) + exp(0.5) ≈ 6.330

P(positive) = 2.226 / 6.330 ≈ 0.352
loss = -log(0.352) ≈ 1.044
```

The positive only got 35% of the probability mass — loss is high, gradients push `sim(q, d+)` up relative to negatives.

### In-batch negatives vs. hard negatives

| | In-batch negatives | Hard negatives |
|---|---|---|
| What | Other examples' positive docs in the same batch | Docs retrieved as high-scoring but actually wrong (e.g. mined by BM25 or a weaker retriever) |
| Cost | Free | Requires a mining step |
| Quality | "Easy" — usually unrelated, low signal | High signal — forces fine-grained distinctions |
| When to use | Always (as baseline) | When you want to materially improve retrieval quality |

> **Interview tip:** "How would you improve a dense retriever?" → **mine hard negatives**. Not just "more data" or "bigger model." The *type* of negatives determines how discriminative the model becomes.

---

## Master comparison table

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Encoding | Query and doc independently | Query and doc jointly (concatenated) |
| Precomputable? | Yes — doc vectors computed once offline | No — must run per `(query, doc)` pair live |
| Query-time cost | ~O(1) encode + sub-linear ANN | O(candidates) × full forward pass |
| Accuracy | Good | Better — joint attention models token-level interactions |
| Scales to full corpus? | Yes | No — architecturally infeasible |
| Role in RAG pipeline | First-stage retrieval | Second-stage reranking (small candidate set only) |

---

## Common mistakes to avoid

- Saying cross-encoders are "just more accurate" without explaining the mechanism (joint attention vs. query-agnostic precomputed vector)
- Suggesting a cross-encoder could work as a first-stage retriever — the latency math makes this infeasible, not just suboptimal
- Assuming query and doc should always share the same encoder — asymmetric is usually the stronger default
- Improving a retriever by only adding more data/model size, without considering hard negative mining
- Forgetting that bi-encoder doc vectors are precomputed **offline** — this is what makes ANN search possible at all

---

## Interview cheat sheet

**The golden line:**
> "Bi-encoders make large-scale search possible at all because document vectors are precomputed offline. Cross-encoders are more accurate because they let query and document tokens attend to each other jointly — but that same joint attention is exactly what makes them impossible to precompute. Which is why every serious production system uses bi-encoders for first-stage retrieval and cross-encoders only to rerank a small shortlist."

**Key numbers to have ready:**
- Bi-encoder query latency: ~15–45 ms
- Cross-encoder per pass: ~15–20 ms
- Cross-encoder on 10M docs: ~55 hours
- Typical rerank set size: 50–100 docs → ~1–2 s

**If asked "why not just use a cross-encoder for everything":**
Do the arithmetic live. 10M docs × 20 ms = 200,000 seconds ≈ 55 hours. Then contrast with reranking 50 docs = 1 second.

**If asked "how would you improve retrieval quality":**
Hard negative mining — not just more data or a bigger model.

---

*Next: Day 9 — Hybrid Search & Reciprocal Rank Fusion*
