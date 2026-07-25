# Recommendation Systems — Master Interview Cheat Sheet

A condensed, one-pass review of all 25 chapters. Read top to bottom the night before an interview. Each section: **the one thing to remember**, key formula (if any), and the line to say out loud if asked.

---

## MODULE 1: FOUNDATIONS

**Ch1 — Problem Framing**
- Explicit feedback (ratings) = rare, direct. Implicit feedback (clicks/watch time) = abundant, ambiguous (no click ≠ dislike — could mean "never shown").
- Most production systems = **implicit + ranking-based**, not explicit rating prediction.
- Cold-start has 3 types: **user** (new user), **item** (new item), **system** (new product). All three get fixed by falling back to content-based signals.
- 🗣️ *"Before I pick a model, I need to know: is this explicit or implicit feedback? That decision changes everything downstream."*

**Ch2 — Evaluation Metrics**
- Precision@K / Recall@K: ignore order within top K.
- MRR: only cares about rank of the *first* relevant item.
- MAP: precision averaged at each relevant hit's rank; binary relevance only.
- **NDCG**: the industry standard — handles graded relevance + discounts by position.
$$NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG@K=\sum\frac{2^{rel_k}-1}{\log_2(k+1)}$$
- 🗣️ *"Offline metric improvement is necessary but not sufficient — I'd still want an A/B test with guardrail metrics before shipping."*

**Ch3 — Baselines**
- Popularity = zero personalization, but never fully retired (fallback for cold users).
- Content-based filtering = recommend similar-feature items; solves item cold-start, causes filter bubbles.
- 🗣️ *"I'd always benchmark against popularity first — if my fancy model can't beat 'just show trending,' something's wrong."*

---

## MODULE 2: CLASSICAL COLLABORATIVE FILTERING

**Ch4 — User/Item-Based CF**
- Similarity: cosine or **Pearson correlation** (Pearson corrects for individual rating bias — a harsh critic vs. generous rater).
- Item-based CF won in industry: item-item similarity is more **stable over time**, catalogs are smaller/more stable than user bases, precomputable offline.
- 🗣️ *"Item-based CF scales better because items change less than users, and there are usually far fewer items than users."*

**Ch5 — Matrix Factorization (SVD/ALS)**
$$\hat{R}_{ui} = \mu + b_u + b_i + p_u^Tq_i$$
- "SVD" in industry ≠ real linear-algebra SVD — it's regularized SGD/ALS on the *sparse observed* matrix.
- ALS parallelizes better (fix one matrix → closed-form solve the other, independent per row) → standard for distributed systems (Spark).
- 🗣️ *"When people say 'SVD' in production they usually mean regularized matrix factorization via SGD or ALS, not textbook SVD — true SVD needs a complete matrix."*

**Ch6 — Implicit Feedback MF (Hu-Koren-Volinsky)**
- Split raw signal into: **preference** $p_{ui}\in\{0,1\}$ + **confidence** $c_{ui}=1+\alpha r_{ui}$.
- Trains on the FULL matrix (not just observed entries) — every unobserved pair is a weak negative.
- 🗣️ *"HKV doesn't regress on raw counts — it converts counts into confidence weights around a binary preference label."*

**Ch7 — Bias Terms & Regularization ("Global Mean Trap")**
- Without $\mu, b_u, b_i$: latent vectors get polluted by absolute rating-scale differences, not just taste.
- New item with 1 rating → shrink toward global mean, don't trust the raw number.
$$b_i = \frac{n_i(\bar{r}_i-\mu)}{n_i+\lambda}$$
- 🗣️ *"For a brand-new item with one 5-star rating, I wouldn't predict 5 — I'd shrink toward the global mean since one data point is weak evidence."*

---

## MODULE 3: LEARNING-TO-RANK

**Ch8 — Pointwise vs Pairwise vs Listwise**
- Pointwise: scores items independently (MSE-style) — good MSE ≠ good ranking.
- Pairwise: learns "is A > B" — order-aware but position-blind (treats rank1-2 swap same as rank9-10 swap).
- Listwise: optimizes the whole list/metric directly — best but expensive.
- 🗣️ *"A model can have great RMSE and terrible ranking — pointwise loss has no notion of relative order."*

**Ch9 — BPR (Bayesian Personalized Ranking)**
- Trains on **triples**: (user, positive item, sampled negative item).
$$\text{BPR-Opt} = \sum \ln\sigma(\hat{y}_{ui}-\hat{y}_{uj})$$
- = pairwise LTR applied specifically to implicit-feedback MF.
- 🗣️ *"BPR is pairwise ranking for implicit MF — every training example is a (user, positive, sampled negative) triple, not a single labeled pair."*

**Ch10 — LambdaRank / LambdaMART**
- NDCG is non-differentiable (sorting) → can't optimize directly.
- Trick: scale the pairwise gradient by $|\Delta NDCG|$ if you swapped that pair — top-of-list errors get way more gradient signal than bottom-of-list errors.
- LambdaMART = same idea + gradient boosted trees (industry standard — handles tabular features well, used in LightGBM/XGBoost).
- 🗣️ *"LambdaRank reweights pairwise gradients by how much fixing that pair would change NDCG — position-1 mistakes matter way more than position-10 mistakes."*

---

## MODULE 4: DEEP LEARNING

**Ch11 — Neural CF**
- Replace MF's dot product with a learned MLP over concatenated $[p_u; q_i]$.
- NeuMF = GMF (generalized dot product) + MLP path, fused; **separate embedding tables** for each path.
- ⚠️ Caveat: Rendle et al. (2019) showed well-tuned MF often matches/beats NCF — cite this for depth.
- 🗣️ *"NCF's real legacy is architectural — it seeded two-tower and Wide&Deep — but a well-tuned MF baseline often matches it in practice."*

**Ch12 — Two-Tower (THE workhorse — know this cold)**
- User tower and item tower compute embeddings **independently**, only meet at final dot product.
- This lets item embeddings be **precomputed offline** → ANN search at serving time.
- Training: in-batch softmax with **in-batch negatives** (other items in the batch = free negatives). Watch for popularity bias in in-batch negatives → correct with log-uniform correction.
- MF (Ch5) = degenerate two-tower (towers = single embedding lookup, no MLP).
- 🗣️ *"Two-tower works for retrieval because item embeddings never depend on the user — they're precomputed and searched via ANN. NCF can't do this because it requires joint computation."*

**Ch13 — Wide & Deep / DeepFM (feature crossing)**
- Wide & Deep: wide=linear+manual crosses (memorization), deep=MLP (generalization). Jointly trained.
- DeepFM: replaces manual crosses with **Factorization Machine** — models ALL pairwise interactions automatically, in $O(kn)$ not $O(n^2)$.
- DeepFM shares embeddings between FM+deep; Wide&Deep uses separate representations.
- 🗣️ *"Wide component memorizes specific known patterns, deep generalizes to new combinations — DeepFM automates the wide part instead of hand-engineering crosses."*

**Ch14 — Sequence Models (GRU4Rec, SASRec)**
- Recast as next-item prediction (like language modeling).
- GRU4Rec: RNN/GRU over session; session-based → naturally cold-start-robust (no persistent user ID needed).
- SASRec: self-attention instead of RNN → parallelizable + direct long-range attention (no info bottleneck through sequential hidden states). Needs **causal masking**.
- 🗣️ *"SASRec beats GRU4Rec because any position can attend directly to any earlier position — no need to pass signal through many sequential hidden-state updates."*

**Ch15 — GNNs (PinSage)**
- Item embedding = aggregate of neighbors' embeddings (message passing) → captures multi-hop "friends of friends" collaborative signal.
- PinSage: random-walk importance weighting (not uniform averaging) + **neighbor sampling** (popular nodes can have millions of neighbors — must cap).
- 🗣️ *"A 2-layer GNN lets an item's embedding reflect items liked by people who liked similar items — multi-hop signal a flat embedding lookup can't capture."*

---

## MODULE 5: SYSTEM ARCHITECTURE (heaviest interview weight)

**Ch16 — The Funnel**
- **Candidate Gen** (millions→thousands, optimize recall, cheap: two-tower+ANN) → **Ranking** (thousands→hundreds, optimize NDCG, expensive: DeepFM/LambdaMART) → **Re-Ranking** (hundreds→final list, business rules/diversity).
- Multiple candidate sources unioned (two-tower + sequence model + GNN + popularity).
- 🗣️ *"You can't run an expensive ranker against the full catalog — the funnel exists so cheap models filter first, expensive models only see what's already promising."*

**Ch17 — ANN / Retrieval at Scale**
- Brute force = infeasible (500M items × 128-dim = ~10 seconds/query). HNSW/FAISS = milliseconds.
- Trade-off: latency vs. recall vs. memory. Index refresh is separate (slower) cadence from model retraining.
- 🗣️ *"Brute force dot-product search doesn't scale — HNSW-style graph search examines a tiny fraction of nodes, cutting query cost by orders of magnitude."*

**Ch18 — Feature Stores & Serving**
- Feature store solves **training-serving skew**: same feature logic for offline training and online serving, avoiding two-implementation drift.
- **Point-in-time correctness**: training data must use feature values as they existed historically, not current values (data leakage otherwise).
- Batch features → early funnel stages; real-time features → later stages (afford freshness at smaller scale).
- 🗣️ *"A feature store's core job is making sure training and serving compute the same feature the same way — otherwise you get silent train-serve skew."*

**Ch19 — System Design (capstone — memorize this skeleton)**
1. **Clarify**: scale, latency budget, feedback type, business objective, cold-start needs.
2. **Frame**: implicit + ranking, composite label.
3. **Funnel**: candidate gen (multi-source) → ranking → re-ranking.
4. **Cold-start**: name all 3 types + fix for each.
5. **Evaluate**: offline (NDCG) + online (A/B + guardrails).
6. **Trade-offs**: feedback loops, engagement-vs-long-term tension, latency-vs-freshness.
- 🗣️ *Always start with clarifying questions before naming a single model — jumping straight to "I'll use two-tower" is the #1 way people lose points.*

---

## MODULE 6: ADVANCED / BUSINESS TOPICS

**Ch20 — Cold-Start Solutions**
- 3 complementary tools: **content-based fallback** (item tower uses content features), **meta-learning** (train for fast adaptation from few examples — optimizes loss *after* a few gradient steps), **exploration** (deliberately show under-explored items to gather data).
- 🗣️ *"These aren't competing — mature systems layer all three: content signal for a starting point, exploration to gather real data, meta-learning to adapt fast once data arrives."*

**Ch21 — Diversity, Serendipity, Filter Bubbles**
- Diversity = within-list variety. Serendipity = unexpected + liked. Filter bubble = longitudinal narrowing over time.
- Fix: **MMR re-ranking** at the final stage — trades relevance for penalty-on-similarity-to-already-selected-items.
$$MMR(i) = \lambda\cdot relevance(i) - (1-\lambda)\max_j sim(i,j)$$
- 🗣️ *"Filter bubbles happen because the model trains on engagement shaped by its own past recommendations — a self-reinforcing loop. MMR re-ranking is the standard fix."*

**Ch22 — Contextual Bandits (LinUCB, Thompson Sampling)**
- The core tension: exploit (best-known item) vs explore (uncertain item, to learn its value).
- LinUCB: score = point estimate + uncertainty bonus (shrinks as more data collected).
- Thompson Sampling: sample from each item's posterior distribution, pick highest sample — naturally explores wide-uncertainty items sometimes.
- 🗣️ *"Bandits formalize cold-start exploration — an item with little data gets a real, calibrated chance to be shown, proportional to how plausible it is that it's actually great."*

**Ch23 — Counterfactual Evaluation (IPS, Doubly Robust)**
- Can't naively evaluate a new policy on logs generated by an old policy — missing data for actions the old policy never took (exposure bias, again).
- **IPS**: reweight by $\pi_1(a|x)/\pi_0(a|x)$ — unbiased but high variance (small propensities blow up).
- **Doubly Robust**: direct reward-model estimate + IPS-weighted correction — unbiased if *either* the propensity model *or* the reward model is right.
- 🗣️ *"Testing a new model against old logs isn't a normal supervised eval — you never observed rewards for actions the old policy didn't take. That's why we need propensity-weighted estimators."*

**Ch24 — A/B Testing Pitfalls**
- **Position bias**: raw CTR comparison confounds true relevance with which slot items landed in.
- **SUTVA violation / network effects**: marketplaces & social platforms — one user's outcome depends on another's treatment (shared supply pool, friend feeds).
- **Novelty effects** (fade over time) vs **feedback-loop compounding** (grows over time) — both mean short tests can mislead.
- Fixes: cluster/switchback randomization, holdback groups, guardrail metrics beyond CTR.
- 🗣️ *"A short CTR-only A/B test can show a clickbait model as a 'win' — you need longer windows and guardrails like watch-time-percentage to catch the real cost."*

**Ch25 — Multi-Objective Optimization**
- True objective (retention) is sparse & delayed — hard to train on directly. Proxy signals (engagement) are dense but risk exactly the failure modes above.
- 3 approaches: **composite weighted label** (fixed at training time), **multi-task learning** (flexible — combine at serving time, shared representations help sparse tasks), **constrained optimization** (e.g. "maximize engagement subject to diversity floor" — ties to MMR).
- 🗣️ *"You train on dense short-term proxies but validate against sparse long-term guardrails — the tension is permanent, not something you solve once."*

---

## TOP 10 THINGS TO SAY IN ANY RECSYS INTERVIEW (if in doubt)

1. Ask: explicit or implicit feedback? (Ch1)
2. Name the baseline before proposing a fancy model. (Ch3)
3. NDCG > raw accuracy for ranking; offline metrics aren't the final word. (Ch2, Ch19)
4. Two-tower for retrieval (precomputed + ANN), richer models only at final ranking. (Ch12, Ch16)
5. The funnel exists because expensive models can't scale to the full catalog. (Ch16)
6. Name all 3 cold-start types and a fix for each. (Ch1, Ch20)
7. Exposure bias shows up everywhere: implicit feedback (Ch6), BPR negatives (Ch9), off-policy eval (Ch23).
8. Feedback loops / rich-get-richer is a named risk — mention diversity/exploration as the fix. (Ch3, Ch21, Ch22)
9. A/B tests need guardrail metrics and awareness of novelty effects, feedback loops, and (for marketplaces/social) SUTVA violations. (Ch24)
10. There's always a business-objective tension (engagement vs. revenue vs. retention) — name it explicitly. (Ch25)
