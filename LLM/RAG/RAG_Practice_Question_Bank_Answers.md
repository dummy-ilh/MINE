# RAG Practice Question Bank — Answer Key

Answers for all 55 questions from the practice bank, at interview-ready depth. Organized by the same categories.

---

## Category A: Core Concepts

**1. What problem does RAG solve that fine-tuning and long-context alone don't?**
Fine-tuning bakes knowledge into model weights — expensive to update, doesn't scale to fast-changing or very large corpora, and gives no attribution/citation. Long-context stuffs everything into the prompt — doesn't scale past context limits, is expensive per query (you pay for every token every time), and suffers "lost in the middle" degradation on large contexts. RAG retrieves only the relevant slice at query time: cheap to keep current (just re-index), scales to arbitrarily large corpora, and naturally supports citation since you know exactly which source informed the answer.

**2. Explain the difference between a bi-encoder and a cross-encoder.**
A bi-encoder encodes query and document independently into fixed vectors, compared via cosine/dot product — no cross-attention between them, but document embeddings are precomputable, enabling fast ANN search over millions of documents. A cross-encoder concatenates query and document and passes them jointly through the model with full cross-attention — much more accurate (captures fine-grained interaction) but requires a full forward pass per candidate pair, no precomputation possible. Bi-encoders handle first-stage retrieval at scale; cross-encoders rerank a small shortlist.

**3. What is the purpose of chunking, and what's the core tradeoff in choosing chunk size?**
Chunking decides the retrievable unit of text. Larger chunks capture more complete context (higher recall — more likely to contain the answer) but dilute the embedding and feed more irrelevant text to the generator (lower precision). Smaller chunks embed more precisely but risk losing context needed to interpret them standalone and can fragment a single fact across multiple chunks. It's a precision/recall and context-completeness tradeoff, usually resolved practically via parent-child/small-to-big chunking rather than picking one fixed size.

**4. What's the difference between dense and sparse retrieval, and what does each miss?**
Dense retrieval uses embedding similarity — captures semantic/paraphrase similarity but is weak on exact terms, IDs, rare tokens, and negation. Sparse retrieval (BM25) uses lexical term matching — captures exact matches precisely but has zero signal for paraphrases or synonyms with no shared vocabulary. Their blind spots are complementary, which is why hybrid retrieval combines both.

**5. What is BM25, and what do the k1 and b parameters control?**
BM25 is a lexical scoring function refining TF-IDF: it weights a document by query-term frequency, dampened by term-frequency saturation, discounted by inverse document frequency, and normalized by document length. `k1` controls term-frequency saturation — caps the marginal benefit of a term appearing many times, so 100 repetitions doesn't score 10x higher than 10 repetitions. `b` controls document-length normalization — prevents longer documents from winning purely by containing more words/repetitions.

**6. What is an ANN algorithm, and why is it needed instead of exact kNN?**
Exact kNN computes distance to every vector in the index — O(N·d) per query, too slow for millions/billions of vectors at interactive latency. Approximate Nearest Neighbor algorithms (HNSW, IVF, etc.) trade a small amount of recall for large speedups by avoiding a full linear scan — e.g. navigating a graph structure or searching only a subset of clusters — making large-scale vector search feasible at low latency.

**7. Name three vector database options and one differentiator for each.**
Pinecone — fully managed, proprietary, zero-ops fast path to production. Weaviate — open-source with rich schema/GraphQL-based filtering and hybrid search built in. pgvector — a Postgres extension, best when the team already runs Postgres and wants to avoid a new database dependency rather than optimizing for maximum ANN performance.

**8. What is reranking, and why is it a separate stage from initial retrieval?**
Reranking is a second-stage pass that re-scores a small candidate set (e.g. top-50) using a more accurate but more expensive model (typically a cross-encoder) before final generation. It's separate because cross-encoders can't be run over the full corpus (no precomputation, too slow at scale) — the funnel pattern is cheap broad recall (bi-encoder + ANN) followed by expensive narrow precision (cross-encoder) on a small shortlist.

**9. What are the three metrics in the "RAG triad" and what does each measure?**
Faithfulness/groundedness — does the answer's content follow only from the retrieved context, without unsupported additions? Answer relevance — does the answer actually address what was asked, regardless of grounding? Context relevance — how much of what was retrieved and fed to the generator was actually useful versus noise? They're measured separately because they can fail independently and point to different root causes.

**10. What is hallucination in the context of RAG specifically, and how is it different from generic LLM hallucination?**
Generic hallucination is a model generating false content from its parametric knowledge with no external grounding at all. RAG-specific hallucination is more subtle: it's the model fabricating or distorting claims *despite having correct, relevant retrieved context available* — either overriding the context with parametric knowledge, extrapolating beyond what's actually stated, or citing a source that doesn't actually support the claim. This is why faithfulness is measured as its own metric rather than assuming retrieval quality alone guarantees grounded output.

---

## Category B: Deep-Dive / Mechanism-Level

**11. Derive or explain the InfoNCE contrastive loss used to train embedding models.**
`L = -log( exp(sim(a,p)/τ) / Σᵢ exp(sim(a,nᵢ)/τ) )` — a softmax over similarity scores between an anchor and a positive versus a set of negatives, trained to push the positive's similarity higher relative to all negatives. τ (temperature) controls how sharply the loss penalizes near-miss negatives — lower τ demands finer discrimination between the positive and its closest negatives, producing a harder training signal; higher τ is more forgiving and produces a smoother loss landscape.

**12. Explain hard negative mining and why in-batch negatives alone are insufficient.**
In-batch negatives (treating other examples' positives in the same batch as negatives) are free but tend to be "easy" — trivially unrelated to the anchor, so the model plateaus after learning coarse topical separation. Hard negatives are lexically or topically close but actually wrong, forcing the model to learn finer-grained distinctions closer to where real retrieval errors occur. Mined either statically (via BM25 or a weaker model, precomputed) or dynamically (using the model's own current embeddings during training, refreshed periodically) — dynamic mining is what most top-performing modern embedders use.

**13. Walk through how HNSW search works, layer by layer.**
HNSW builds a multi-layer graph — top layers are sparse with long-range connections ("highways"), the bottom layer is dense with fine-grained local connections. Search starts at the top layer, greedily navigates toward the query vector until no closer neighbor is found at that layer, then drops down one layer and repeats, progressively refining toward the true nearest neighbors as it descends to the bottom layer, where the final candidate set is collected. `efSearch` controls how much of each layer is explored — the main recall/speed knob at query time.

**14. Explain IVF-PQ and why product quantization trades accuracy for memory.**
IVF partitions vectors into clusters via k-means; at query time, only the `nprobe` closest clusters are searched, avoiding a full scan. Product Quantization compresses each vector by splitting it into sub-vectors and quantizing each to a small codebook index, drastically shrinking memory footprint (e.g. 32x smaller). The tradeoff: distances computed on quantized vectors are approximate, not exact, so recall drops — commonly mitigated by using IVF-PQ for a fast initial shortlist, then re-scoring that shortlist with full-precision vectors to recover lost accuracy.

**15. What is ColBERT's MaxSim operator, and why is late interaction a genuine middle ground?**
For each query token, MaxSim finds its single most-similar document token (max similarity), then sums these per-token maxes across all query tokens to get the final relevance score. It's a genuine middle ground because document token embeddings are precomputed and stored like a bi-encoder (no query-time document encoding), but scoring still involves token-level interaction like a cross-encoder — capturing much of the fine-grained precision benefit without paying the cost of a full joint forward pass per candidate.

**16. Explain Reciprocal Rank Fusion and why it uses rank instead of raw scores.**
RRF combines multiple ranked lists by summing `1/(k + rank)` for each document across systems, where `rank` is the document's position within each individual system's result list. It uses rank instead of raw scores because BM25 and cosine similarity live on incomparable scales (BM25 unbounded and corpus-dependent, cosine bounded [-1,1]) — combining raw scores requires fragile, corpus-specific normalization, while rank position is directly comparable across any two systems regardless of their internal scoring.

**17. What is HyDE, and why does embedding a hallucinated document improve retrieval instead of hurting it?**
HyDE prompts an LLM to generate a hypothetical answer document for the query, then embeds and searches using *that* document's embedding instead of the raw query's. It works because it solves a structural mismatch: a short question and a long, information-dense answer passage don't naturally embed close together even when the passage answers the question. A generated hypothetical answer is structurally similar to real corpus documents (same length, style, density), so its embedding lands in a much more comparable region of the vector space — the hypothetical document's factual content doesn't need to be correct, only its embedding needs to be well-shaped for search.

**18. Explain "lost in the middle" and its implications for how you order retrieved context.**
LLMs show a U-shaped attention/recall curve over long contexts — information at the very start or end is used more reliably than information buried in the middle, even when it's technically present. Implication: don't concatenate retrieved chunks in arbitrary order — surface the highest-confidence chunks at the beginning and/or end of the context ("sandwiching"), and keep total context length as tight as possible (tighter top-n from reranking) to reduce the chance relevant content gets buried and ignored.

**19. Walk through pointwise, pairwise, and listwise LLM-as-reranker approaches and their cost/quality tradeoffs.**
Pointwise scores each document independently against the query — cheapest, fully parallelizable, but absolute scores across separate calls aren't reliably calibrated against each other. Pairwise compares two documents at a time — generally the most reliable per-comparison signal (LLMs are better at relative judgments), but needs many comparisons to build a full ranking, scaling poorly. Listwise ranks the whole candidate set in one call — most token-efficient with full comparative context, but bounded by context window for large sets and more prone to position bias.

**20. Explain nDCG and when you'd prefer it over Recall@k or MRR.**
nDCG accounts for graded relevance (not just binary relevant/irrelevant) and discounts a result's contribution based on its rank position (logarithmic discount), normalized against the ideal possible ranking. Prefer it when relevance isn't binary (documents can be "highly relevant" vs "somewhat relevant") and *ranking order* genuinely matters beyond just whether relevant documents appear anywhere in top-k — Recall@k only asks if the evidence is present at all, MRR only cares about the first relevant hit's position.

**21. What are the known biases in LLM-as-judge evaluation, and how do you mitigate each?**
Position bias — judges favor whichever answer is shown first/second regardless of quality; mitigate by evaluating both orderings and averaging or randomizing order. Verbosity bias — judges rate longer answers as better even without added information; mitigate by explicitly instructing the judge to penalize unnecessary length or normalizing for length. Self-preference bias — a judge favors outputs from its own model family; mitigate by using a different model family as judge, and periodically calibrating judge scores against a human-labeled sample.

**22. Explain the pre-filter vs post-filter vs integrated-filter tradeoff for metadata-constrained vector search.**
Post-filtering runs ANN search first then discards non-matching results — breaks when the filter is highly selective, since you can end up with too few results even though matching documents exist elsewhere in the index. Pre-filtering restricts the candidate set to matching vectors first, then searches — breaks on high-cardinality filters (many distinct values, few matches each), since you lose ANN benefits over tiny filtered subsets. Integrated filtering pushes the filter into the graph/cluster traversal itself, avoiding both failure modes, but requires index structures that support filter-aware traversal natively.

**23. Walk through Self-RAG and Corrective RAG and how they differ from standard single-shot RAG.**
Standard RAG performs one fixed retrieval step regardless of query difficulty. Self-RAG trains the model (via special reflection tokens) to critique its own retrieval and generation at each step — e.g. flagging "this retrieved passage isn't relevant, retrieve again" or "this sentence isn't well-supported, revise" — baking faithfulness self-checking into the generation loop. Corrective RAG adds an explicit retrieval-quality evaluator right after retrieval: if retrieved docs score as correct, proceed normally; if incorrect, discard and fall back to an alternate strategy (e.g. web search); if ambiguous, combine both. Both add iteration/self-correction at the cost of more calls and latency, reserved for cases where fixed single-shot retrieval demonstrably fails often enough to justify it.

**24. Explain why embeddings from two different model versions can't be mixed in the same index, and what a safe migration looks like.**
Different embedding models produce vectors in different, incompatible geometric spaces with no shared coordinate system — cosine similarity between an old-model query vector and a new-model document vector is meaningless, not just degraded. A safe migration is a full re-embed of the entire corpus with the new model plus a full index rebuild, typically done as a blue-green swap: build the new index fully offline, validate recall against an eval set, then cut over — never an incremental migration where only new documents get the new model.

**25. What's the difference between context relevance and Recall@k as metrics — don't they measure the same thing?**
No — Recall@k asks whether the truly relevant document(s) appear anywhere in the top-k retrieved set at all. Context relevance asks, of what was actually retrieved and passed to the generator, how much of it was useful signal versus noise. A retrieval can satisfy high Recall@k (the right document is in there) while still having low context relevance (surrounded by a lot of irrelevant chunks diluting/distracting the generator) — they measure presence-of-evidence versus purity-of-evidence, and can diverge.

---

## Category C: Multi-Hop Specific

**26. Why does single-hop retrieval fail on compositional/bridge-entity questions?**
No single chunk contains the full chain of facts needed to answer — e.g. "who directed the film that won Best Picture the year X was born" requires first resolving X's birth year, then the Best Picture winner that year, then that film's director. A single embedding-similarity search against the original question won't retrieve a chunk containing all three linked facts, since they likely don't co-occur in one passage.

**27. Compare IRCoT, Self-Ask, and decomposition-based multi-hop approaches.**
Decomposition breaks the complex query into sub-questions upfront, retrieves per sub-question, then composes the final answer — risk: a wrong upfront decomposition dooms the whole chain with no self-correction. IRCoT interleaves chain-of-thought reasoning steps with retrieval calls, where each reasoning step conditions the next retrieval — more adaptive than upfront decomposition. Self-Ask has the model explicitly emit "follow-up question needed" steps that trigger retrieval per follow-up, similar in spirit to IRCoT but structured around explicit follow-up-question emission rather than free-form reasoning interleaving.

**28. What is error propagation in multi-hop retrieval, and what mitigates it?**
A bad early hop (wrong retrieval or wrong intermediate conclusion) poisons every downstream hop, since later retrievals are conditioned on earlier results — there's often no easy recovery once an early hop goes wrong. Mitigations: verification/self-consistency checks between hops (does this hop's answer contradict the previous hop's evidence?), and adaptive retrieval that only triggers multi-hop when a cheaper single-hop pass demonstrably fails, reducing the number of opportunities for propagation in the first place.

**29. How do you decide when to stop iterating in an iterative multi-hop retrieval loop?**
Options: a fixed hop-count cap (simple, but can stop too early or waste calls on already-sufficient context), LLM self-judged "sufficient context" (the model decides it has enough to answer — more adaptive but relies on the model's own calibration, which can be wrong), or confidence thresholding (stop once retrieval/answer confidence crosses a threshold). In practice, a fixed cap combined with a self-judged sufficiency check is common — cap bounds worst-case cost, sufficiency check avoids always running to the cap unnecessarily.

**30. When would you use graph-based multi-hop instead of iterative dense retrieval?**
When the question depends on explicit relational structure between entities (e.g. "what companies did this person's former colleagues go on to found") where the relevant entities may not be textually or semantically similar to the query at all, only *connected* to it through relationships a vector search can't capture. Graph traversal directly follows those relationships; dense multi-hop retrieval would need to stumble onto the right chain purely through semantic similarity at each step, which is unreliable for this question shape.

**31. What evaluation datasets are standard for multi-hop RAG, and what does each specifically stress?**
HotpotQA — general multi-hop QA with supporting-fact annotations. 2WikiMultihopQA — stresses bridge-entity and comparison-style reasoning with structured, verifiable reasoning chains. MuSiQue — specifically designed to stress distractor robustness and prevent shortcut/single-hop solutions to nominally multi-hop questions.

**32. Why can a multi-hop system get the final answer right for the wrong reason, and how do you catch that?**
A model can sometimes guess the correct final answer via a shortcut or spurious correlation without actually retrieving or reasoning through the correct intermediate chain of evidence — the final-answer accuracy alone doesn't reveal this. Catch it with supporting-fact evaluation: check whether the system retrieved and used the *correct intermediate evidence* at each hop, not just whether the final answer string matches — this is exactly why multi-hop benchmarks like HotpotQA include supporting-fact annotations rather than only final-answer labels.

---

## Category D: Diagnosis / Debugging

**33. A user reports a wrong answer. Walk through your debugging process in order.**
First, inspect the logged retrieved chunks for that exact query, before looking at the generated answer at all — this determines if the relevant information was fetched (retrieval miss) or not. If absent, debug upstream: embedding domain gaps on rare terms, chunking splitting the needed fact awkwardly, or k too small. If present, check where it was positioned in the assembled context (lost-in-the-middle risk) and whether it was crowded out by redundant/irrelevant chunks. If well-positioned and still ignored or contradicted, run a faithfulness check to confirm genuine hallucination versus a subtle misreading. If everything upstream checks out and the answer is still wrong, verify the source content itself isn't simply stale relative to current ground truth.

**34. Retrieval quality has degraded gradually over a month with no code changes. What are your hypotheses?**
Two leading hypotheses: IVF-based index cluster centroids trained on a data snapshot have drifted from the actual (now-larger/different) data distribution as new documents were added, degrading recall without any explicit failure signal; or the incoming query distribution itself has shifted (new topics, phrasing patterns) in a way the original golden eval set doesn't represent, so real-world performance degraded even though the system would still score fine against the now-outdated offline eval set. Check both: inspect index retraining schedule/staleness, and compare recent production query samples against the original eval set's distribution.

**35. The correct chunk was retrieved (verified in logs) but the model still got the answer wrong. What do you check next?**
Since retrieval succeeded, this is a generation-stage failure. Check where the correct chunk was positioned in the final prompt (buried mid-context = lost in the middle), whether the context was cluttered with redundant or irrelevant chunks competing for attention, and run a faithfulness check to confirm whether the wrong claim actually traces back to anything in the retrieved context — if not, it's pure hallucination despite good context, pointing to prompt-level grounding instructions or model choice as the fix.

**36. A generated answer cites a real document, but the citation doesn't actually support the claim. How would you catch this systematically, not just in this one case?**
Self-reported citations from the generator aren't proof of faithfulness — a model can cite a real chunk ID while fabricating content not actually supported by it. Catch this systematically with a post-hoc verification pass: for each generated claim, run an NLI model or LLM-judge check asking whether the cited source actually entails the claim, independent of what the generator itself claims to be citing. Run this as an ongoing sampled check on production traffic, not just as a one-off investigation.

**37. Your offline eval metrics look great but production user satisfaction is dropping. What's your first hypothesis and how do you investigate?**
First hypothesis: the offline golden eval set no longer represents the real production query distribution — synthetic/curated eval questions tend to be more literal and idealized than real user phrasing, and query patterns shift over time. Investigate by comparing recent production query samples (topic clustering, embedding-space distribution) against the eval set's distribution to look for drift, and by running the Module 7 LLM-judge metrics (faithfulness, relevance) directly on sampled production traffic rather than only the static offline set.

**38. You suspect your embedding model has a domain-adaptation gap. What's the fastest way to confirm or rule this out?**
Compare retrieval hit rate on domain-specific/jargon-heavy queries specifically versus natural-language queries in your eval set. If domain-heavy queries underperform disproportionately, and BM25/sparse retrieval alone outperforms the dense retriever on that same query subset, that's a strong signal of a dense-embedding domain-coverage gap rather than a chunking or indexing bug — the sparse-vs-dense comparison on the same slice isolates the hypothesis quickly.

**39. Answer quality was fine for months and suddenly dropped after a routine content update. What's your leading hypothesis?**
Leading hypothesis: the content update introduced a stale-index or ingestion-pipeline issue — either new/updated documents weren't fully re-embedded and re-indexed, or the update introduced content that broke a chunking assumption (e.g. a new document format with tables or structure the existing chunker mishandles). Since the drop is sudden and correlates with a specific event rather than gradual drift, check the ingestion pipeline logs from that update specifically before looking at anything else.

---

## Category E: System Design

**40. Design a RAG system for customer support over product documentation, sub-second latency requirement.**
Scope first: query volume, corpus size/update frequency, whether queries are mostly simple factoid lookups. Given sub-second latency: favor HNSW indexing (fast ANN, incremental updates), hybrid retrieval via RRF (support queries often mix natural language and exact product/error codes), a lightweight/fast reranker or possibly skip reranking if latency-constrained (or use ColBERT-style late interaction to fold precision into retrieval itself), tight top-n into generation to minimize lost-in-the-middle risk and token cost, and semantic caching for the high-repetition FAQ-style query pattern typical of support traffic.

**41. Design a RAG system for enterprise search with per-user/per-team access control.**
Access control must be enforced at retrieval time via integrated/pre-filtering on ACL metadata tagged at ingestion — never post-hoc filtering after content has already entered the model's context. ACL metadata must stay synced with the source-of-truth permission system (e.g. the document platform's actual sharing settings) as a real-time sync problem, not a one-time ingestion tag. Add audit logging of what was retrieved for whom. Given heterogeneous content sources (Confluence, Slack, tickets), consider federated/multi-index retrieval so each source type uses its optimal chunking strategy, merged via rank fusion at query time.

**42. Design a RAG system that must handle both trivial factoid queries and complex multi-hop queries efficiently — how do you avoid paying multi-hop cost on every request?**
Use adaptive/agentic retrieval: run a cheap single-shot retrieval pass by default, and only escalate to iterative multi-hop (Module 4B) when a lightweight check indicates the first pass was insufficient — e.g. low retrieval confidence, or the LLM itself signaling it can't answer from the given context (Self-RAG/CRAG-style gating, Module 6.5). This keeps simple queries cheap and fast while reserving the expensive multi-hop path for queries that actually need it, rather than uniformly running every query through the expensive path.

**43. How would you scale a RAG system from 100K to 100M documents? What breaks first, and what would you change?**
What breaks first: a flat/exact index or naive IVFFlat becomes too slow and memory-heavy; an HNSW graph held fully in memory may exceed a single machine's RAM (rough napkin math: 100M × 768-dim float32 ≈ 300GB). Changes: move to IVF-PQ or quantized HNSW for memory efficiency, shard the index across machines (semantic/cluster-based sharding to reduce query fan-out versus naive hash sharding), and reconsider the ingestion pipeline for incremental/streaming updates with change-detection rather than full reprocessing at this scale.

**44. Design a RAG system where source documents update multiple times per day and answers must reflect edits within minutes.**
Favor HNSW (natively incremental insertion, no periodic retraining requirement) over IVF-based indexing. Build a streaming/event-driven ingestion pipeline (embed and upsert on document-change events) rather than batch/nightly reprocessing, with change-detection via content hashing to avoid re-embedding unchanged documents. Any caching layer (semantic cache) needs an invalidation strategy tied to document update timestamps, not just a TTL, to avoid serving stale cached answers within that minutes-level freshness window.

**45. How would you design cost monitoring for a RAG system in production — what are the line items and which typically dominates?**
Line items: embedding calls (ingestion volume + query volume, plus query-transformation calls if using HyDE/multi-query), storage (vector storage scales with corpus size × embedding dimension, reducible via quantization/Matryoshka truncation), reranker calls (scale with query volume × k), and generation calls (scale with total context tokens × query volume). Generation typically dominates per-query cost in most systems, since it's priced per token and context tokens (retrieved chunks) directly drive that cost — though at very high k, reranking can become the largest non-generation cost. Monitor each line item separately so cost regressions can be attributed to a specific stage.

**46. Design a multi-tenant RAG system — shared index or per-tenant index, and why?**
Default to a shared index with metadata-based tenant filtering (Module 3.5/9.2) for resource efficiency and simpler operations — but this requires the filtering mechanism to be airtight, since a filtering bug leaking one tenant's data into another's results is a severe security failure, not a minor bug. Switch to fully separate per-tenant indexes when tenants have very different scale/access patterns, when hard isolation is a compliance requirement, or when a single large tenant's query load would degrade performance for smaller tenants sharing the same index.

**47. Walk through migrating a production RAG system to a new embedding model with zero downtime.**
Build the new index fully offline by re-embedding the entire corpus with the new model — never mix old and new model vectors in one index, since they're incompatible geometric spaces. Validate retrieval recall on the new index against a golden eval set before any user traffic touches it. Use a blue-green cutover: keep the old index serving live traffic while the new index is built and validated, then switch traffic over atomically (or via a canary rollout to a small percentage first, Module 11.6) rather than an in-place migration that could serve inconsistent results mid-transition.

---

## Category F: Judgment / Tradeoff

**48. When is hybrid retrieval not worth the added complexity?**
When the query distribution is dominated by natural-language, paraphrase-heavy queries with low reliance on exact terms/IDs (e.g. general FAQ-style support), where dense retrieval alone already performs well — the added infra (maintaining a sparse index, a fusion step) and latency cost of hybrid retrieval may not be justified by the marginal quality gain. Worth revisiting if error analysis later shows a specific class of queries (exact terms, codes) failing under dense-only.

**49. When would you choose pgvector over a managed vector DB like Pinecone, even at meaningful scale?**
When the team already runs Postgres as the system of record and wants to avoid introducing a new database dependency, a new data-sync problem (keeping metadata and vectors consistent across two separate systems), and the operational surface of a new service — and when the scale doesn't yet demand Pinecone-grade horizontal ANN scaling or advanced tuning. Transactional consistency between metadata and vectors in a single system can also matter more than raw ANN performance for some use cases.

**50. When is GraphRAG worth its overhead, and when is it a poor fit?**
Worth it when questions require explicit relational reasoning across entities where relevant information is connected via relationships rather than textual/topical similarity — vector search alone struggles here since related entities may not be semantically similar to the query. Poor fit when the corpus/query distribution is mostly topical/semantic lookup, since building and maintaining the knowledge graph (itself an error-prone LLM extraction pipeline) adds significant upfront and ongoing cost that isn't repaid by a real gain in that case — usually deployed as a complement to vector search, routing only relationally-structured queries to the graph.

**51. Your reranker is your single biggest latency cost after generation. What are your options, and what does each cost you?**
Lower k into the reranker (cheaper, but risks cutting off the true relevant document if k was tuned too aggressively — validate against eval recall first). Use a smaller/distilled cross-encoder model (faster, some accuracy loss). Substitute a ColBERT-style late-interaction model that shifts cost to index-time precomputation rather than query-time joint encoding (larger index footprint as the tradeoff). Batch reranking calls more efficiently. Each option trades some combination of accuracy, index size, or engineering complexity for latency.

**52. When would you use Corrective RAG versus accepting some retrieval failures as a base rate?**
Use CRAG when retrieval failures are a known, non-trivial, and consequential source of bad answers — e.g. a corpus with real coverage gaps, or queries that legitimately fall outside the corpus's scope — and the cost of confidently generating from irrelevant context is high enough to justify an added evaluation/fallback step on every query. Accept a base rate of failures when the failure rate is already low and the added latency/cost of gating every request isn't justified by the marginal quality improvement — a judgment call that should be grounded in actual error-analysis data, not assumed upfront.

**53. When is context compression worth its own risk of information loss?**
Worth it at high query volume where token cost meaningfully affects unit economics, or when source documents are long and low-density (e.g. full legal contracts where the relevant content is a small fraction of the text) such that compression's noise-reduction benefit outweighs its information-loss risk. Not worth it for high-stakes, low-volume queries where preserving every nuance of the retrieved context matters more than cost/latency savings — compression stacks a lossy step on top of an already-lossy retrieval pipeline, and errors compound.

**54. Weighted-sum fusion vs RRF for combining dense and sparse retrieval — when would you actually invest in the former?**
RRF is a strong default — simple, robust, no corpus-specific tuning required, since it only uses rank position rather than raw incomparable scores. Invest in a tuned weighted-sum combination when you have the eval infrastructure to properly tune the weighting coefficient and score normalization per domain, and when eval metrics show RRF's rank-only approach is leaving quality on the table — e.g. a domain where you specifically want to express "trust dense retrieval more than sparse for this corpus" in a way RRF's rank-agnostic-to-magnitude approach can't capture.

**55. Your corpus has adversarially many near-duplicate documents. How does this affect retrieval, reranking, and generation, and what would you change at each stage?**
Retrieval: near-duplicates can flood the top-k with redundant content, crowding out genuinely distinct relevant documents and wasting recall budget — mitigate with deduplication/near-duplicate detection at ingestion or post-retrieval (embedding-similarity clustering). Reranking: redundant candidates waste reranker calls (cost scales with k) without adding new information — dedup before reranking to make better use of the k budget. Generation: even after retrieval-stage dedup, near-identical chunks reaching the generator waste context budget and increase "lost in the middle" risk by padding the context with repetitive content instead of diverse evidence — dedup or diversity-aware selection (e.g. maximal marginal relevance) should be applied before final context assembly.
