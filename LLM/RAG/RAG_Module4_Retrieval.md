# RAG Module 4 — Retrieval Strategies

---

## 4.1 Dense retrieval — strengths and blind spots

Dense retrieval = embed query and corpus with the same (or asymmetric-paired) embedding model, rank by vector similarity (Module 1/3).

**Strengths**: captures semantic similarity even with zero lexical overlap — "automobile" retrieves documents about "car," a paraphrased question retrieves the right passage even if no words match literally. This is the entire value proposition over pure keyword search.

**Blind spots** (know these cold — this is the setup for why hybrid retrieval exists):
- **Exact terms / identifiers**: product SKUs, part numbers, legal citation numbers, error codes — dense embeddings have no strong signal for exact-string matching; "SKU-4471829" and "SKU-4471830" embed nearly identically despite being completely different products
- **Rare/out-of-distribution tokens**: acronyms, proper nouns, domain jargon underrepresented in the embedding model's training data (directly connects to Module 1.7 — domain adaptation)
- **Negation and fine-grained lexical distinctions**: "symptoms that are NOT caused by X" can embed close to "symptoms caused by X" — dense embeddings are better at topical similarity than logical/semantic precision
- **Numerical/quantitative specificity**: "revenue above $10M" vs "revenue above $1M" can embed very close together despite being materially different queries

---

## 4.2 Sparse retrieval — BM25/TF-IDF mechanics

**TF-IDF** intuition: a term is important to a document if it appears frequently *in that document* (TF) but rarely *across the whole corpus* (IDF, inverse document frequency) — common words like "the" get down-weighted, distinctive words get up-weighted.

**BM25** (the actual production standard, refines TF-IDF):
```
score(D,Q) = Σ IDF(q_i) · [f(q_i,D)·(k1+1)] / [f(q_i,D) + k1·(1-b+b·|D|/avgdl)]
```
- `f(q_i,D)`: term frequency of query term in document D
- `k1`: controls term-frequency saturation — without it, a document repeating a term 100 times would score proportionally higher than one repeating it 10 times, which isn't a meaningful 10x relevance signal in practice. `k1` caps the marginal benefit of repeated occurrences.
- `b`: controls document-length normalization — without it, longer documents win purely by containing more words/more term repetitions, not because they're more relevant. `b` penalizes length inflation.

**Why sparse still wins on keyword-heavy queries**: BM25 gives *exact* lexical match credit — no representation collapse, no domain-adaptation gap, no OOV problem for tokens the way dense embeddings have. It directly solves every blind spot listed in 4.1.

**Sparse's own blind spot**: zero semantic understanding — a query and a relevant document sharing zero exact terms (pure paraphrase, synonym, cross-lingual) gets zero BM25 score regardless of true relevance. This is the mirror image of dense retrieval's weakness — which is exactly why they're combined rather than one replacing the other.

---

## 4.3 Hybrid retrieval — combining dense + sparse

Since dense and sparse have *complementary* failure modes (4.1 vs 4.2), combine both scores into a single ranking.

### Fusion methods

**Reciprocal Rank Fusion (RRF)** — the most common production choice, rank-based rather than score-based:
```
RRF_score(d) = Σ_systems 1 / (k + rank_system(d))
```
where `rank_system(d)` is the document's rank position (1st, 2nd, 3rd...) within *that* system's result list, and `k` is a constant (commonly 60) that dampens the impact of very high ranks.

**Why rank-based instead of raw score combination**: BM25 scores and cosine similarity scores live on **completely different, incomparable scales** (BM25 is unbounded and corpus-dependent; cosine similarity is bounded [-1,1]) — naively summing raw scores requires careful normalization that's fragile and corpus-specific. RRF sidesteps this entirely by only using *rank position*, which is directly comparable across any two systems regardless of their internal scoring scale. This is the single most important reason RRF is the default in practice.

**Weighted sum (with normalization)** — normalize each system's scores (e.g. min-max scaling per query) then combine as `α·dense_score + (1-α)·sparse_score`. More tunable (the α weight can be learned or tuned per domain) but more fragile — normalization choices and the weighting coefficient both need empirical tuning per corpus, and can behave unpredictably if a query returns very few sparse hits (sparse score distribution skews for that specific query).

**Interview-ready comparison**: RRF is simpler, more robust, no tuning required — a strong default. Weighted sum can outperform RRF *if* you invest in proper tuning per domain/corpus, since it lets you explicitly express "trust dense retrieval more for this corpus." Good answer: start with RRF, move to a tuned weighted combination only if eval metrics show a specific, addressable gap.

---

## 4.4 Query transformation techniques

The query as typed by the user is often not the ideal string to embed/search with. Several techniques reshape the query before retrieval:

### HyDE (Hypothetical Document Embeddings)
1. Given the user's query, prompt an LLM to generate a **hypothetical answer document** (even if it might be factually wrong/hallucinated)
2. Embed *that hypothetical document*, not the original query
3. Search the index using the hypothetical document's embedding

**Why this works**: this directly attacks the query-document asymmetry problem (Module 1.4) — a short question and a long informative passage don't naturally embed close together even when the passage answers the question, because they're structurally different text. A generated hypothetical *answer* is structurally similar to the real documents in the corpus (same length, same style, same information density), so its embedding lands in a much more comparable region of the vector space to the true relevant document, even though the hypothetical document's actual factual content might be wrong.

**Interview trap**: candidates sometimes worry "isn't this circular, since the hypothetical document is hallucinated?" — the answer is that HyDE never uses the hypothetical document's *content* as fact, only its *embedding* as a better-shaped search vector. It's a retrieval-shaping trick, not a generation step.

### Query expansion
Add related terms/synonyms to the original query (via an LLM, a thesaurus, or pseudo-relevance feedback from an initial retrieval pass) before searching — improves sparse retrieval recall in particular, since BM25 has zero tolerance for missing exact terms.

### Multi-query retrieval
Prompt an LLM to generate several *reformulations* of the original query (different phrasings, different angles/sub-aspects), run retrieval for each independently, then merge/deduplicate the combined result set. Reduces sensitivity to any single query phrasing's blind spots — increases recall at the cost of more retrieval calls (latency/cost tradeoff, same shape as the multi-hop cost tradeoff in Module 4B).

### Step-back prompting
Prompt the LLM to first generate a more general/abstract version of the query ("what's the mechanism behind X" → "how does the general category of mechanism that X belongs to work"), retrieve using the step-back query to get broader grounding context, *then* answer the original specific question using that broader context. Useful when the specific question requires background/principles that aren't directly stated near the specific answer in the corpus.

---

## 4.5 Multi-hop / iterative retrieval — pointer

Covered in full depth in **Module 4B** (already delivered) — iterative retrieval, decomposition, IRCoT, Self-Ask, agentic/ReAct retrieval, graph-based multi-hop, and the associated failure modes (error propagation, stopping criteria, cost explosion, query drift).

---

## 4.6 Metadata/structured filtering combined with vector search

Real production retrieval is almost never pure vector similarity — it's vector similarity *within* a filtered subset (e.g. "similar documents, but only ones the current user has permission to see," or "only from the last 90 days"). This connects directly back to Module 3.5's pre-filter/post-filter/hybrid discussion — the filtering *mechanism* lives in the index layer, but the *decision* of what to filter on (permissions, recency, document type, source) is a retrieval-strategy design choice made here, at the query-construction layer.

**Interview-relevant framing**: this is where retrieval strategy and system design (Module 9) meet — access-control-aware retrieval isn't a nice-to-have, it's often a hard security requirement (a RAG system must never surface a chunk the requesting user isn't authorized to see, regardless of how semantically relevant it is).

---

## Interview Q&A drill

**Q: Why not just always use hybrid retrieval — is there ever a reason to use dense-only or sparse-only?**
A: Hybrid adds latency and infra complexity (maintaining two indexes, a fusion step) for a quality gain that isn't always worth it. Dense-only is sufficient when queries are natural-language and paraphrase-heavy with low reliance on exact terms/IDs (e.g. general customer support Q&A). Sparse-only (or sparse-dominant) is preferable when the corpus is dominated by exact-match needs — legal citation lookup, product SKU search, code symbol search — where dense retrieval's semantic fuzziness is actively unhelpful. Hybrid earns its complexity when the query distribution genuinely mixes both needs, which is common but not universal.

**Q: Explain why RRF uses rank instead of raw similarity/BM25 scores.**
A: BM25 and cosine similarity scores are on different, incomparable numeric scales — BM25 is unbounded and depends on corpus statistics (IDF, average doc length), cosine similarity is bounded between -1 and 1. Combining raw scores requires a normalization step that's fragile and needs to be re-tuned per corpus/query distribution. RRF sidesteps the entire problem by using each document's *rank position* within each system's individual result list — rank position is directly comparable across any two systems regardless of their internal scoring, making RRF simple and robust with no corpus-specific tuning required.

**Q: A user asks "what are the side effects that are NOT related to the liver?" and dense retrieval keeps returning liver-related content. Diagnose and propose fixes.**
A: This is a dense-retrieval blind spot around negation (4.1) — embeddings capture topical similarity ("side effects," "liver") much more strongly than logical polarity ("not related to"), so a query about non-liver side effects embeds close to documents that are heavily about liver side effects. Fixes: (1) add sparse/BM25 retrieval to the hybrid mix so exact-term matching can help surface documents that don't contain "liver" prominently, (2) query rewriting — have an LLM reformulate the query to make the actual information need more explicit for retrieval (e.g. decompose into "list all side effects" + a post-retrieval filtering/generation step that excludes liver-related ones, rather than expecting a single embedding-based retrieval call to encode a negation), (3) rely on the generation stage rather than retrieval to handle the logical exclusion — retrieve broadly, let the LLM filter by the "not liver" constraint from the full retrieved set instead of asking retrieval itself to be negation-aware.

**Q: When would you reach for HyDE versus multi-query retrieval?**
A: HyDE targets the query-document *structural* mismatch — short question vs long informative passage — and is most useful when the corpus consists of long-form, answer-shaped documents (articles, reports, documentation) where a real query rarely resembles the phrasing of the matching passage. Multi-query retrieval targets *phrasing ambiguity/coverage* — when a single query formulation might miss relevant documents due to vocabulary mismatch or the question having multiple valid sub-interpretations — and is more useful for broad/exploratory queries or corpora with high lexical diversity. They're not mutually exclusive; some pipelines generate multiple hypothetical documents (combining both ideas) for even broader retrieval coverage, at a proportionally higher cost.

---

**Next up: Module 5 — Reranking.** Say the word when ready.
