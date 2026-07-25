# Chapter 2: Evaluation Metrics — Precision@K, Recall@K, MAP, NDCG, MRR

## 1. Intuition

Once you've framed the problem as ranking (Ch. 1), you need metrics that evaluate **ordered lists**, not point predictions. RMSE is meaningless here — nobody cares if your predicted score is 4.3 vs true 4.5, they care whether the top 10 items shown are the right 10, in a good order.

All ranking metrics answer variations of one question: **"of the items I showed, and the order I showed them in, how good was that?"** They differ in whether they care about order (rank position), whether they're binary (relevant/not) or graded (relevance levels), and whether they evaluate one list or average across users.

## 2. Precision@K and Recall@K

**Precision@K**: of the top K items you recommended, what fraction were actually relevant?

$$P@K = \frac{\text{# relevant items in top K}}{K}$$

**Recall@K**: of all relevant items that exist for this user, what fraction did you capture in your top K?

$$R@K = \frac{\text{# relevant items in top K}}{\text{total # relevant items}}$$

Key distinction interviewers probe: Precision@K penalizes showing irrelevant items; Recall@K penalizes missing relevant items. Neither accounts for **order within the top K** — a relevant item at rank 1 and at rank 10 count identically. This is the core limitation motivating MAP/NDCG/MRR.

## 3. Mean Reciprocal Rank (MRR)

Used when there's typically **one** correct/best answer per query (e.g., "what's the first relevant search result?").

$$RR = \frac{1}{\text{rank of first relevant item}}$$

$$MRR = \frac{1}{|U|}\sum_{u=1}^{|U|} RR_u$$

If the first relevant item is at rank 1, RR = 1. At rank 4, RR = 0.25. MRR only cares about the **first** hit — it ignores everything after. Good for tasks like "did we get the right answer near the top" (autocomplete, QA), bad for tasks where multiple relevant items matter (feed recommendations), because it's blind to everything past the first hit.

## 4. Mean Average Precision (MAP)

Fixes the "order-blindness" of Precision@K by computing precision **at each position where a relevant item appears**, then averaging.

$$AP = \frac{\sum_{k=1}^{K} P@k \cdot \text{rel}(k)}{\text{# relevant items}}$$

where $\text{rel}(k) = 1$ if the item at rank $k$ is relevant, else 0.

$$MAP = \frac{1}{|U|}\sum_{u=1}^{|U|} AP_u$$

Intuition: you get credit for relevant items, but *more* credit if they appear early (because early precision values are computed over fewer items and get "locked in" at that position). MAP assumes **binary relevance** — an item is relevant or not, no partial credit. This is its main limitation vs. NDCG.

## 5. NDCG (Normalized Discounted Cumulative Gain)

The industry-standard metric because it handles **graded relevance** (not just binary) and **explicitly discounts** items by rank position.

$$DCG@K = \sum_{k=1}^{K} \frac{2^{rel_k} - 1}{\log_2(k+1)}$$

- $rel_k$ = relevance grade of item at rank $k$ (e.g., 0=irrelevant, 1=relevant, 2=highly relevant)
- $\log_2(k+1)$ = position discount — the deeper the rank, the more the gain is discounted
- $2^{rel_k}-1$ = gain — exponential so highly-relevant items (grade 2) contribute disproportionately more than merely-relevant ones (grade 1)

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

$IDCG@K$ = DCG of the **ideal** ranking (relevant items sorted in perfect order). Dividing by IDCG normalizes the score to [0,1], making it comparable across users/queries who have different numbers of relevant items — this normalization is what MAP lacks, and it's why NDCG is preferred when relevant-item counts vary a lot across users.

## 6. Worked Numerical Example

User has 4 candidate items shown, ranked in this order, with true relevance grades (0=irrelevant, 1=relevant, 2=highly relevant):

| Rank | Item | Relevance |
|---|---|---|
| 1 | A | 2 |
| 2 | B | 0 |
| 3 | C | 1 |
| 4 | D | 0 |

**Precision@3** = relevant items in top 3 (A, C are relevant → 2) / 3 = **0.667**

**Recall@3**: assume 2 total relevant items exist (A, C) → 2/2 = **1.0**

**MRR**: first relevant item is at rank 1 → RR = **1.0**

**MAP**:
- At rank 1 (relevant): P@1 = 1/1 = 1
- At rank 3 (relevant): P@3 = 2/3 = 0.667
- AP = (1 + 0.667) / 2 relevant items = **0.833**

**NDCG@4**:
- $DCG = \frac{2^2-1}{\log_2 2} + \frac{2^0-1}{\log_2 3} + \frac{2^1-1}{\log_2 4} + \frac{2^0-1}{\log_2 5}$
- $= \frac{3}{1} + \frac{0}{1.585} + \frac{1}{2} + \frac{0}{2.322}$
- $= 3 + 0 + 0.5 + 0 = 3.5$
- Ideal order would be A(2), C(1), B(0), D(0): $IDCG = \frac{3}{1} + \frac{1}{\log_2 3} + 0 + 0 = 3 + 0.631 = 3.631$
- $NDCG@4 = 3.5 / 3.631 = \mathbf{0.964}$

Notice: MAP and NDCG both reward A being at rank 1, but NDCG additionally rewards the *magnitude* of relevance (grade 2 vs grade 1) — MAP would treat A and C as equally valuable "relevant" hits since it's binary.

## 7. Production Considerations

- **Offline metrics ≠ online success.** A model with better offline NDCG can still perform worse in an A/B test — this is the single most important caveat and the reason production teams treat offline metrics as a *filter*, not a *decision-maker*. Causes: distributional shift between logged data and live traffic, position bias in the logged data itself (Ch. 24), and offline metrics not capturing business metrics (revenue, retention).
- Relevance labels for NDCG are often themselves noisy proxies (e.g., using click as relevance=1, no click as relevance=0), which reintroduces the implicit-feedback ambiguity problem from Chapter 1.
- Metric choice should match the actual UI: if the surface only shows one result (voice assistant), MRR is right; if it's a scrollable feed, NDCG/MAP over a larger K matters more.

## 8. Interview Traps

- Saying "NDCG is just better than MAP" without explaining *why* — the real answer is graded relevance + normalization, not "it's more modern."
- Forgetting that Precision@K and Recall@K are blind to *within-list order*.
- Computing DCG without normalizing by IDCG and calling it NDCG.
- Not mentioning offline-online metric divergence when asked "how would you evaluate this system" — this is one of the most common L5 differentiators interviewers listen for.

## 9. L5-Differentiating Talking Points

- Proactively state that **offline metric improvement is necessary but not sufficient** — you'd still want an A/B test with guardrail metrics (revenue, session length, long-term retention) before shipping.
- Note that metric choice itself signals product understanding: MRR for single-answer surfaces, NDCG for graded-relevance ranked feeds, Recall@K when the business goal is "don't miss good content" (e.g., ads eligibility) vs. Precision@K when "don't show garbage" matters more (e.g., first row of a homepage).
- Mention that relevance labels used to compute these metrics are frequently derived from implicit signals, which reintroduces bias (exposure bias, position bias) into your evaluation itself — a rare, senior-level observation that offline eval isn't "ground truth," it's another biased dataset.

## 10. Comprehension Check

1. Why can two rankings have identical Precision@K but different NDCG@K?
2. What does IDCG normalize for, and why does that matter when comparing NDCG across different users?
3. When would you prefer MRR over NDCG?
4. Why is MAP limited to binary relevance, and what does that cost you compared to NDCG?
5. Why might a model with better offline NDCG perform worse in a live A/B test?
