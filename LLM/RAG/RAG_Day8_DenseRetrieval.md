# RAG Interview Prep — Day 8
## Dense Retrieval: Bi-Encoders vs. Cross-Encoders

---

## 🚀 Quick Summary

Dense retrieval (Day 2's embeddings, put into practice) has two fundamentally different architectural approaches: **bi-encoders**, which embed the query and each document *independently* into the same vector space and compare them with simple math (fast, scalable, what makes searching millions of documents possible at all), and **cross-encoders**, which feed the query and a document *together* into one model so they can attend to each other directly (much more accurate, but far too slow to run against an entire corpus). Understanding exactly why each has its speed/accuracy profile — not just that one is "fast" and one is "accurate" — is one of the most consistently asked architecture questions in RAG interviews, because it's the direct justification for the two-stage retrieve-then-rerank pattern that essentially every production system uses.

**Think of it like two ways to find your best match at a speed-dating event with 10,000 attendees.** A bi-encoder is like having every attendee fill out a personality profile *in advance*, independently, then you compare your own profile against everyone else's using simple math to find the closest matches — fast, because the profiles were prepared ahead of time and comparing profiles is cheap. A cross-encoder is like actually sitting down and having a real conversation with each candidate — far more accurate, because you can react to their specific answers and they to yours, but you obviously can't have a real conversation with all 10,000 people. The practical answer: use the fast profile-matching (bi-encoder) to narrow 10,000 down to 50 promising candidates, *then* have real conversations (cross-encoder) with just those 50.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Bi-encoder** | Encodes query and document independently into the same vector space; similarity computed afterward via cosine/dot product |
| **Cross-encoder** | Encodes query and document *jointly* in one forward pass, letting them attend to each other; outputs a relevance score directly |
| **Symmetric bi-encoder** | Query and document share the same encoder — used when queries and documents are similar in style/length |
| **Asymmetric bi-encoder** | Query and document use separate, specialized encoders (e.g., DPR) — used when queries and documents differ structurally (short question vs. long passage) |
| **Contrastive learning** | Training approach that pulls matching (query, document) pairs together in vector space while pushing non-matching pairs apart |
| **In-batch negatives** | Using other examples' documents within the same training batch as "free" negative examples for contrastive loss |
| **Hard negatives** | Deliberately-selected negative examples that are superficially similar but actually irrelevant — much more useful for training than random negatives |
| **Two-stage retrieval** | Using a fast bi-encoder to narrow a huge corpus to a small candidate set, then a slow-but-accurate cross-encoder to rerank just that small set |

---

# PHASE 1 — Intuition & Visual Map

## The architectural difference, visually

```
  BI-ENCODER                                CROSS-ENCODER

  Query ──▶ [Encoder] ──▶ vector_q          Query + Document ──▶ [Encoder,
                                                                    JOINTLY]
  Doc   ──▶ [Encoder] ──▶ vector_d                    │
                                                        ▼
              │                              relevance_score (single number,
              ▼                              e.g. 0.87)
       cosine_similarity(vector_q, vector_d)

  Document vectors can be computed          Nothing can be precomputed — the
  ONCE, offline, in advance, and            model needs BOTH the query and
  stored in a vector index (Day 4).         the specific document together,
  At query time: only the QUERY needs       every single time, to produce
  to be embedded — then it's a fast         a score. Must run a full model
  nearest-neighbor lookup against           forward pass per (query, document)
  millions of precomputed document          pair, at query time, live.
  vectors.
```

**This is the entire reason the speed/accuracy trade-off exists.** A bi-encoder's document vectors are computed *once, offline, ahead of any specific query* — all the expensive model inference happens at indexing time, not query time. A cross-encoder cannot do this at all, because its whole value comes from letting the query and document tokens attend to each other *jointly* — there's no way to "pre-compute" that interaction without already knowing the query, which defeats the purpose.

## When to use each

- ✅ **Bi-encoder** — first-stage retrieval over a large corpus, where you need to search millions/billions of documents and only a nearest-neighbor lookup is fast enough.
- ✅ **Cross-encoder** — reranking a small candidate set (typically 10-100 documents) that a first-stage retriever already narrowed down — never as the sole retrieval mechanism over a full corpus.
- ❌ Never use a cross-encoder as your only/first-stage retrieval mechanism over a large corpus — the latency math (below) makes this infeasible past a small number of candidates.

---

# PHASE 2 — Math & Mechanics

## 1. Bi-Encoder Scoring

```
score(q, d) = similarity( f(q), g(d) )
```
where `f` and `g` are the query encoder and document encoder (often, but not always, the same model — see symmetric vs. asymmetric below), and `similarity` is typically cosine similarity or dot product (Day 2).

**Complexity at query time:** encode the query once (`O(1)` model forward pass), then compare against `N` precomputed document vectors — and crucially, that comparison step is exactly what Day 4's ANN indexes (HNSW, IVF) are built to make fast, so it's not even a full `O(N)` brute-force comparison in practice, it's sub-linear.

**Worked latency example:**
```
Query encoding: one forward pass through the encoder, ~10-30ms
                 depending on model size and hardware
ANN search over 10M precomputed document vectors: ~5-15ms (Day 4)

Total: roughly 15-45ms per query
```
This is fast enough for interactive, user-facing search.

---

## 2. Cross-Encoder Scoring

```
score(q, d) = Model( concat(q, d) )  →  a single relevance score
```
The query and document are concatenated (typically with a separator token) and passed through the model *together* as one input, so every token can attend to every other token — query tokens to document tokens and vice versa — via self-attention. The model outputs a single scalar relevance score directly (often trained as a classification or regression head on top of the encoder).

**Complexity at query time:** a full model forward pass is required **for every single (query, document) pair** — there's no precomputation possible, because the model needs to see the actual query alongside the actual document to produce its joint attention-based score.

**Worked latency example — why cross-encoders can't scale to full-corpus search:**
```
Assume one cross-encoder forward pass takes ~20ms (a reasonable ballpark
for a moderately-sized cross-encoder model on typical inference hardware).

Reranking 50 candidates:     50 × 20ms = 1,000ms = 1 second
Reranking 500 candidates:    500 × 20ms = 10,000ms = 10 seconds
"Reranking" 10,000,000 docs: 10,000,000 × 20ms = 200,000 seconds ≈ 55.5 HOURS
```
The jump from "reranking 50 candidates" (a perfectly reasonable ~1 second) to "scoring the full 10-million-document corpus" (55+ hours) is the single clearest, most concrete illustration of why cross-encoders are architecturally incapable of being a first-stage retriever over any real-world corpus size — this is the calculation to have ready if asked "why not just use a cross-encoder for everything, since it's more accurate?"

---

## Why Cross-Encoders Are More Accurate — The Actual Mechanism

**Plain English:** A bi-encoder has to compress an entire document's meaning into a single fixed-length vector *before ever seeing the query* — it has no idea, at encoding time, which aspects of the document will end up mattering for some future, unknown query. This is an inherent information bottleneck: some nuance about *how* a query and document relate to each other specifically can only be captured by looking at both together.

A cross-encoder, by processing the query and document jointly through self-attention, can directly model fine-grained token-to-token interactions — e.g., noticing that the specific word "battery" in the query attends strongly to a specific mention of "battery life" three sentences into the document, in a way that's contextually relevant to *this particular query's phrasing* — a level of interaction a bi-encoder's independent, pre-computed, query-agnostic document vector simply cannot represent.

**Why this matters in practice:** This is the actual "why" behind the accuracy gap, and it's the correct thing to say in an interview instead of just "cross-encoders are more accurate because they're more complex" — the mechanism is specifically the *joint attention over both inputs together*, which a bi-encoder's architecture structurally cannot do.

---

## Symmetric vs. Asymmetric Bi-Encoders

**Symmetric:** query and document are encoded with the *same* encoder (shared weights). Works well when queries and documents are structurally similar — e.g., semantic textual similarity tasks, or search where the query itself looks like a mini-document (e.g., "find me articles similar to this one").

**Asymmetric (e.g., DPR — Dense Passage Retrieval):** query and document use *separate* encoders, each specialized for its own input distribution. This matters because a typical search query ("how do I reset my Apple ID password") is structurally very different from the passage that answers it (a full paragraph of instructions) — short vs. long, question-form vs. declarative-form. Training separate encoders lets each one specialize for its own input type, rather than forcing one shared encoder to represent both a short question and a long answer passage equally well in the same way.

> **Why This Matters callout:** If asked "would you use the same encoder for queries and documents," the strong answer references this asymmetry directly — most production RAG retrieval setups (and most modern sentence-embedding models built specifically for retrieval, like many on the MTEB retrieval leaderboard from Day 2) use asymmetric or at least asymmetrically-*trained* setups (sometimes sharing weights but with different input prefixes/instructions signaling "this is a query" vs. "this is a document") precisely because queries and documents aren't structurally interchangeable.

---

## Training Dense Retrievers — Contrastive Learning

**The core training objective:** pull the embeddings of a matching (query, relevant-document) pair *closer together* in vector space, while pushing the embeddings of non-matching (query, irrelevant-document) pairs *further apart*.

### InfoNCE-style Contrastive Loss (the standard approach)

```
L = -log( exp(sim(q, d+) / τ) / Σ_{i} exp(sim(q, d_i) / τ) )
```

**Plain English breakdown:**
- `sim(q, d+)` — the similarity score between the query and its **correct/positive** matching document.
- `Σ_i exp(sim(q, d_i)/τ)` — sum over the positive document *and* a set of negative (non-matching) documents `d_i`, exponentiated.
- `τ` (temperature) — a scaling hyperparameter controlling how "sharp" or "soft" the resulting probability distribution is. Lower `τ` makes the model more aggressively confident in distinguishing the positive from negatives; higher `τ` softens the distinction.
- The overall loss is essentially a **classification objective**: "out of this positive document and a set of negative documents, correctly assign the highest similarity score to the positive one" — implemented as a softmax over similarity scores, with cross-entropy loss against the known-correct positive.

**Worked conceptual example (small scale, illustrative):** Say a training batch has 1 positive document and 3 negatives for a given query, with these similarity scores (τ=1 for simplicity): `sim(q, d+) = 0.8`, negatives: `0.3, 0.1, 0.5`.
```
numerator   = exp(0.8) ≈ 2.226
denominator = exp(0.8) + exp(0.3) + exp(0.1) + exp(0.5)
            ≈ 2.226 + 1.350 + 1.105 + 1.649
            = 6.330

probability assigned to the positive = 2.226 / 6.330 ≈ 0.352
loss = -log(0.352) ≈ 1.044
```
The loss is high when the model doesn't confidently separate the positive from the negatives (here, the positive only got ~35% of the "probability mass" despite being the correct match); training pushes the model to increase `sim(q,d+)` relative to the negatives, driving this loss down and improving separation.

### In-Batch Negatives vs. Hard Negatives

- **In-batch negatives:** a computationally cheap trick — for a training batch containing many (query, positive document) pairs, treat *every other example's positive document* in the same batch as a negative for this example's query. Free negatives with no extra data collection needed, but they're often "easy" negatives (randomly unrelated), which limits how discriminative the model becomes.
- **Hard negatives:** documents that are *superficially* similar to the correct answer (e.g., retrieved by an earlier, weaker retriever as a high-scoring-but-wrong candidate) but are actually irrelevant. Training on hard negatives forces the model to learn much finer-grained distinctions than random in-batch negatives can teach — a well-known, high-leverage technique for materially improving retrieval quality, often mined using a previous version of the retrieval system itself (or BM25) to surface "confusable" wrong answers.

> **Why This Matters callout:** If asked "how would you improve a dense retriever's quality," a strong, specific answer is "mine hard negatives" — not just "use more training data" or "use a bigger model." This shows understanding that *what kind* of negative examples the model sees during training directly determines how fine-grained its learned similarity function becomes, which is a frequently underappreciated lever.

---

## Bi-Encoder vs. Cross-Encoder — Master Comparison Table

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **Encoding** | Query and document independently | Query and document jointly (concatenated) |
| **Precomputable?** | Yes — document vectors computed once, offline | No — must run per (query, document) pair, live |
| **Query-time complexity** | ~O(1) query encode + sub-linear ANN search | O(candidates) full model forward passes |
| **Accuracy** | Good | Better — captures fine-grained token-level query-document interaction |
| **Scalability to full corpus** | Yes — this is what makes large-scale search possible at all | No — architecturally infeasible past a small candidate set |
| **Typical role in a RAG pipeline** | First-stage retrieval (Day 4's ANN search operates on these vectors) | Second-stage reranking (Day 10) on a small candidate set |

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** In one sentence, what architectural difference makes bi-encoders fast and cross-encoders slow?

<details>
<summary>Show answer</summary>

Bi-encoders encode the query and document independently, so document vectors can be precomputed once offline and only the query needs encoding at search time; cross-encoders process the query and document jointly in a single forward pass, so nothing can be precomputed and a full model inference is required for every single query-document pair at search time.
</details>

---

**Q2 (Easy — calculation).** A cross-encoder takes 15ms per forward pass. How long would it take to score 300 candidate documents, and why does this make it unsuitable as a first-stage retriever over a 5-million-document corpus?

<details>
<summary>Show answer</summary>

```
300 × 15ms = 4,500ms = 4.5 seconds  (feasible as a reranking step on a small candidate set)

5,000,000 × 15ms = 75,000,000ms ≈ 20.8 hours  (completely infeasible for live queries)
```
The gap between reranking a small shortlist and scoring an entire large corpus is exactly why cross-encoders are only used as a second-stage reranker on top of a fast first-stage retriever, never as the sole retrieval mechanism.
</details>

---

**Q3 (Medium — conceptual).** Why are cross-encoders more accurate than bi-encoders — describe the actual mechanism, not just "they're more complex."

<details>
<summary>Show answer</summary>

A bi-encoder must compress an entire document into a single fixed-length vector before ever seeing the query, with no way to know in advance which aspects of the document will matter for some future query — an inherent information bottleneck. A cross-encoder processes the query and document together through self-attention, letting individual tokens from the query and document attend directly to each other, capturing fine-grained token-to-token interactions specific to that exact query-document pairing. This joint, query-aware attention is something a bi-encoder's independently-precomputed, query-agnostic document vector structurally cannot represent.
</details>

---

**Q4 (Medium — conceptual).** What's the difference between a symmetric and asymmetric bi-encoder, and why would you choose asymmetric for a typical search retrieval setup?

<details>
<summary>Show answer</summary>

A symmetric bi-encoder uses the same shared encoder for both queries and documents; an asymmetric bi-encoder (e.g., DPR) uses separate, specialized encoders for each. Asymmetric setups are preferred for typical search retrieval because queries and documents are often structurally very different — a short question-form query vs. a long declarative passage — and forcing one shared encoder to represent both input types equally well can be suboptimal. Separate encoders let each specialize for its own input distribution, generally improving retrieval quality for this common query-vs-passage asymmetry.
</details>

---

**Q5 (Medium — conceptual).** What are hard negatives in dense retriever training, and why are they more valuable than in-batch (random) negatives?

<details>
<summary>Show answer</summary>

Hard negatives are documents that are superficially similar to the correct answer — often ones that a weaker retriever (or BM25) scored highly despite being actually irrelevant — while in-batch negatives are simply other examples' positive documents within the same training batch, which are usually randomly/easily distinguishable from the correct answer. Training on hard negatives forces the model to learn much finer-grained distinctions between genuinely relevant and merely superficially-similar content, directly improving the model's discriminative power in exactly the cases that matter most at inference time (when a query has several plausible-looking but wrong candidates), whereas random in-batch negatives mostly teach the easier, less useful distinction between totally unrelated content.
</details>

---

**Q6 (Hard — calculation + reasoning).** Explain, using the InfoNCE loss structure, why increasing the temperature `τ` would make training less aggressive about separating positives from negatives. (Conceptual reasoning is fine — full derivation not required.)

<details>
<summary>Show answer</summary>

In `L = -log(exp(sim(q,d+)/τ) / Σ_i exp(sim(q,d_i)/τ))`, dividing similarity scores by a larger `τ` shrinks the differences between them before exponentiating — e.g., similarities of 0.8 and 0.3 differ by 0.5 raw, but divided by a large τ (say 10), they become 0.08 and 0.03, a much smaller effective gap once exponentiated. This "softens" the resulting distribution — the positive's probability mass relative to the negatives becomes less dominant even if the raw similarity gap is the same, meaning the loss provides a weaker gradient signal pushing the model to sharply separate positive from negative. A smaller τ has the opposite effect — it amplifies small similarity differences, making the model's training signal more aggressive about achieving a large, confident separation between positives and negatives.
</details>

---

**Q7 (Hard — system design synthesis).** Design a retrieval pipeline for a RAG system that needs both low latency (under 200ms end-to-end) and high accuracy over a 50-million-document corpus. Explain how bi-encoders and cross-encoders would each be used, and why neither alone is sufficient.

<details>
<summary>Show answer</summary>

Using only a bi-encoder would be fast enough (encode the query, ANN search via Day 4's HNSW/IVF over precomputed document vectors) but would leave accuracy on the table, since bi-encoders can't model fine-grained query-document token interactions. Using only a cross-encoder is architecturally infeasible at this scale — scoring even a fraction of 50 million documents at ~15-20ms per forward pass would take many hours, far outside any latency budget. The standard solution is a two-stage pipeline: a bi-encoder (paired with an ANN index) performs fast first-stage retrieval, narrowing 50 million documents down to a small candidate set (e.g., top 50-100) within a small latency budget (tens of ms); a cross-encoder then reranks just that small candidate set (Day 10), taking on the order of 1-2 seconds worst case for 100 candidates at ~15-20ms each — or fewer candidates if the latency budget is tight — producing much more accurate final rankings than the bi-encoder alone could, while keeping total end-to-end latency within the 200ms target by controlling how many candidates get passed to the expensive reranking stage.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Saying cross-encoders are "just more accurate" without explaining the actual mechanism (joint attention over both inputs vs. independent precomputed vectors).
- ❌ Suggesting a cross-encoder could ever be used as a first-stage retriever over a large corpus — the latency math makes this infeasible, not just suboptimal.
- ❌ Assuming query and document should always share the same encoder — asymmetric setups are often the better default given how structurally different queries and documents usually are.
- ❌ Improving a dense retriever by only adding more training data or a bigger model, without considering hard negative mining — often the higher-leverage lever.
- ❌ Forgetting that bi-encoder document vectors are precomputed *offline*, which is the entire reason ANN search (Day 4) is even possible — conflating this with cross-encoders, which have no equivalent precomputation step.

---

# 📌 Cheat Sheet (Day 8)

**Bi-encoder:** encode query & document independently → compare via cosine/dot product. Document vectors precomputed offline. Fast, scalable — the only viable option for first-stage retrieval over large corpora.

**Cross-encoder:** encode query & document jointly → single relevance score via self-attention across both. Can't precompute. More accurate (models fine-grained token interactions) but architecturally can't scale past a small candidate set (worked example: 5M docs × 15ms ≈ 20+ hours).

**Symmetric vs. asymmetric:** shared encoder (symmetric) vs. separate specialized query/document encoders (asymmetric, e.g. DPR) — asymmetric usually wins for typical short-query/long-passage retrieval.

**Training:** contrastive loss (InfoNCE) pulls matching pairs together, pushes non-matching apart, via a softmax-style objective over similarity scores. In-batch negatives = cheap but easy; hard negatives = more expensive to mine but much higher-leverage for retrieval quality.

**Golden interview line:** *"Bi-encoders make large-scale search possible at all because their document vectors are precomputed offline; cross-encoders are more accurate because they let query and document tokens attend to each other jointly, but that same joint attention is exactly what makes them impossible to precompute — which is why every serious production system uses bi-encoders for first-stage retrieval and cross-encoders only to rerank a small shortlist."*

---

*End of Day 8. Next up — Day 9: Hybrid Search & Reciprocal Rank Fusion.*
