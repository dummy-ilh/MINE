# Chapter 4: User-Based & Item-Based Collaborative Filtering

## 1. Intuition

Collaborative filtering (CF) drops the requirement of item features entirely (unlike Ch. 3's content-based approach) and instead exploits a single idea: **patterns in the user-item interaction matrix itself are the signal.** No genre tags, no text embeddings — just "who interacted with what, and how."

Two flavors:
- **User-based CF**: "find users similar to you, recommend what they liked."
- **Item-based CF**: "find items similar to items you've liked, recommend those."

Both rely on a **similarity function** over rows (users) or columns (items) of the interaction matrix. The core mechanical difference is which axis you compute similarity over — everything else follows from that choice.

## 2. The Interaction Matrix

Let $R$ be a matrix where $R_{ui}$ = rating (or interaction strength) of user $u$ for item $i$. Most entries are missing (sparsity is the defining challenge of CF — a typical matrix is >99% empty).

**User-based CF** treats each row (a user's rating vector across all items) as the unit of similarity comparison.
**Item-based CF** treats each column (an item's rating vector across all users) as the unit of similarity comparison.

## 3. Similarity Metrics

**Cosine similarity** between two users $u, v$ (or items):

$$\text{sim}(u,v) = \frac{\sum_{i \in I_{uv}} R_{ui} R_{vi}}{\sqrt{\sum_{i \in I_{uv}} R_{ui}^2}\sqrt{\sum_{i \in I_{uv}} R_{vi}^2}}$$

where $I_{uv}$ = items both users rated.

**Pearson correlation** — the more common choice in classical CF because it corrects for individual rating bias (some users rate everything a 4, others reserve 5s for masterpieces only):

$$\text{sim}(u,v) = \frac{\sum_{i \in I_{uv}} (R_{ui}-\bar{R}_u)(R_{vi}-\bar{R}_v)}{\sqrt{\sum_{i \in I_{uv}}(R_{ui}-\bar{R}_u)^2}\sqrt{\sum_{i \in I_{uv}}(R_{vi}-\bar{R}_v)^2}}$$

where $\bar{R}_u$ is user $u$'s average rating. Subtracting the mean is exactly the mean-centering step that removes each user's personal rating scale before comparing — this is the single biggest practical improvement over raw cosine similarity for explicit ratings.

## 4. Prediction Formula (User-Based)

Once you have similarities, predict a missing rating as a similarity-weighted average of what similar users rated, mean-centered to correct for the target user's own bias:

$$\hat{R}_{ui} = \bar{R}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v)(R_{vi}-\bar{R}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}$$

$N(u)$ = the $k$ nearest neighbors of user $u$ (hence "neighborhood methods").

Item-based CF uses the identical formula with roles swapped — similarity computed between items, neighbors are similar items the target user has already rated.

## 5. Why Item-Based CF Won Out in Practice

This is a frequently-tested interview point. Item-based CF became the dominant approach (famously, Amazon's original recommender) for structural reasons, not accuracy alone:

- **Item-item similarities are far more stable over time** than user-user similarities — an item's relationship to other items barely changes day to day, while users' tastes and active-item-sets shift constantly, requiring frequent recomputation.
- **Scalability**: the number of items is typically far smaller and more stable than the number of users (millions of users vs. thousands-to-millions of items, and items grow much more slowly than users at most companies). Precomputing an item-item similarity matrix is far more tractable and cacheable than a user-user matrix, since it can be computed offline and refreshed periodically rather than in real time.
- **Cold-start asymmetry**: a new user has no rating vector at all — can't compute user-user similarity for them immediately. But you can still use item-based CF for that new user the moment they rate even one item, since you just look up that item's precomputed neighbors.

## 6. Worked Numerical Example

Rating matrix (rows = users, columns = items, blank = unrated):

| | Item A | Item B | Item C | Item D |
|---|---|---|---|---|
| User 1 | 5 | 3 | — | 1 |
| User 2 | 4 | — | 2 | 1 |
| User 3 | 1 | 1 | 4 | 5 |

**Goal**: predict User 1's rating for Item C using user-based CF with User 2 and User 3 as potential neighbors.

Co-rated items between User 1 and User 2: A, D. Between User 1 and User 3: A, B, D.

$\bar{R}_1 = (5+3+1)/3 = 3.0$, $\bar{R}_2 = (4+2+1)/3 = 2.33$, $\bar{R}_3=(1+1+4+5)/4=2.75$

**Pearson sim(1,2)** using items A, D:
- Deviations U1: A: 5-3=2, D: 1-3=-2
- Deviations U2: A: 4-2.33=1.67, D: 1-2.33=-1.33
- Numerator: (2)(1.67) + (-2)(-1.33) = 3.34 + 2.66 = 6.0
- Denom: $\sqrt{2^2+(-2)^2}\sqrt{1.67^2+(-1.33)^2} = \sqrt{8}\sqrt{4.56} = 2.828 \times 2.135 = 6.038$
- sim(1,2) = 6.0/6.038 = **0.994**

**Pearson sim(1,3)** using items A, B, D:
- Deviations U1: A: 2, B: 0, D: -2
- Deviations U3: A: 1-2.75=-1.75, B: 1-2.75=-1.75, D: 5-2.75=2.25
- Numerator: (2)(-1.75)+(0)(-1.75)+(-2)(2.25) = -3.5+0-4.5 = -8.0
- Denom: $\sqrt{4+0+4}\sqrt{3.0625+3.0625+5.0625} = \sqrt{8}\sqrt{11.1875} = 2.828\times3.345=9.459$
- sim(1,3) = -8.0/9.459 = **-0.846**

Interpretation: User 1 and User 2 have near-identical taste (0.994); User 1 and User 3 have nearly opposite taste (-0.846). This makes intuitive sense — User 3 loves items User 1 dislikes and vice versa.

**Predict $\hat{R}_{1,C}$** using both as neighbors:

$$\hat{R}_{1,C} = \bar{R}_1 + \frac{\text{sim}(1,2)(R_{2,C}-\bar{R}_2) + \text{sim}(1,3)(R_{3,C}-\bar{R}_3)}{|\text{sim}(1,2)|+|\text{sim}(1,3)|}$$

$$= 3.0 + \frac{0.994(2-2.33) + (-0.846)(4-2.75)}{0.994+0.846}$$

$$= 3.0 + \frac{0.994(-0.33) + (-0.846)(1.25)}{1.84} = 3.0 + \frac{-0.328 - 1.058}{1.84} = 3.0 + \frac{-1.386}{1.84} = 3.0 - 0.753 = \mathbf{2.25}$$

So User 1 is predicted to rate Item C around 2.25 — low, driven mainly by User 3's strong negative correlation dragging the prediction down (User 3 loved item C, but since User 3 has opposite taste, that's evidence User 1 will *not* like it).

## 7. Production Considerations

- Neighborhood-based CF is rarely deployed at scale as the primary production model today — it's been largely superseded by matrix factorization (Ch. 5) and neural methods (Module 4) because computing full pairwise similarities is $O(n^2)$ in the number of users or items, which doesn't scale to Google/YouTube-sized catalogs.
- It's still extremely relevant as **teaching intuition** and shows up as a component in some real systems (e.g., "similar items" widgets are often literally item-based CF or a close descendant of it, since it's cheap, interpretable, and doesn't require training a model).
- Sparsity is the dominant practical failure mode: with 99%+ missing entries, co-rated item counts between two users/items can be tiny (or zero), making similarity estimates noisy or undefined. Minimum co-rating thresholds (e.g., require ≥5 common items) are a standard mitigation.

## 8. Interview Traps

- Using raw cosine similarity on ratings without mean-centering, then wondering why a user who rates everything 5 looks "similar" to a user who rates everything 4 despite very different absolute preferences — Pearson correlation exists specifically to fix this.
- Not being able to explain *why* item-based CF scales better than user-based CF — this is one of the most commonly asked follow-ups and needs the catalog-size-stability argument, not just "it's just better."
- Ignoring the sparsity problem — assuming similarity is reliably computable when co-rated item counts are tiny.
- Confusing "item-based CF" with "content-based filtering" (Ch. 3) — item-based CF uses **behavioral** co-occurrence in ratings, not item metadata/features at all. This is a very common and heavily-tested confusion.

## 9. L5-Differentiating Talking Points

- Explain the item-based-over-user-based preference using the **catalog-size and stability argument**, not just accuracy — this shows systems thinking, since interviewers are really probing whether you understand scalability trade-offs, not the algorithm in isolation.
- Bring up minimum co-rating thresholds and shrinkage/regularization of similarity estimates when few co-ratings exist — a classic sign of hands-on production experience with sparse data.
- Note that neighborhood CF is largely a **historical stepping stone** to matrix factorization in modern systems, but still surfaces in explainable "similar items" UI components — showing you know where this technique fits in the broader evolution of the field (ties directly into Ch. 5).
- Mention that both user-based and item-based CF suffer from the same cold-start limitations from Chapter 1 — reinforcing that these are complementary techniques on a spectrum, not a solved problem in isolation.

## 10. Comprehension Check

1. Why does Pearson correlation outperform raw cosine similarity for CF on explicit ratings?
2. Give the two structural (not accuracy-based) reasons item-based CF scales better than user-based CF in production.
3. In the worked example, why did User 3's strong negative correlation pull the predicted rating for Item C *down* even though User 3 rated Item C very highly?
4. What's the difference between item-based collaborative filtering and content-based filtering, and why do candidates often confuse them?
5. What practical mitigation would you apply when two users/items have very few co-rated items in common?
