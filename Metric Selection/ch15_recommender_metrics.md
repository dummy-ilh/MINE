# Chapter 15: Recommender System Metrics

> *"A recommender system that only gives users what they already know they want is not recommending — it's confirming. The best recommenders expand the user's world. Measuring that expansion is the hard part."*

---

## 15.1 The Recommender Evaluation Problem

Recommender systems sit at the intersection of ranking (Chapter 3), business metrics (Chapter 5), and Goodhart's Law (Chapter 6). They are among the hardest systems to evaluate correctly because:

1. **The feedback loop problem**: Users only interact with what the system shows them. You never observe what they would have done with different recommendations.
2. **The long-tail problem**: Popular items get clicked regardless of recommendation quality. Rare items need good recommendations to get any exposure at all.
3. **The beyond-accuracy problem**: A system that recommends the same 10 popular items to everyone is accurate but useless. Diversity, novelty, and serendipity matter — but are hard to measure.
4. **The temporal problem**: User preferences change. Yesterday's good recommendation is today's stale one.

```
Accuracy alone is not enough:
  System A recommends: [Titanic, Avatar, Avengers, Star Wars, Jurassic Park]
  System B recommends: [Titanic, A Hidden Life, Parasite, Moonlight, Shoplifters]

  System A has higher predicted CTR.
  System B has higher user satisfaction (if the user hasn't seen these).

  Accuracy metrics choose A. Users who try B discover new favorites.
```

---

## 15.2 Accuracy Metrics (Revisited for Recommendation)

The metrics from Chapter 3 apply directly. Here we add recommendation-specific context.

### Precision@K and Recall@K

```
Precision@K = # relevant items in top K / K
Recall@K    = # relevant items in top K / # total relevant items for user
```

**Recommendation-specific considerations:**
- "Relevant" typically means: item the user interacted with in the test period
- K is usually 5, 10, or 20 (the number of recommendations shown)
- Must be computed **per user** and averaged (macro average over users)

**Hit Rate (HR@K):**
```
HR@K = fraction of users for whom at least one relevant item appears in top K
```

Binary version of Recall@K at the user level. "Did we get at least one right for this user?"

### NDCG@K

As derived in Chapter 3, now with recommendation context:

```
NDCG@K = DCG@K / IDCG@K

In recommendation, relevance labels often come from:
  - Explicit: star ratings (1–5)
  - Implicit: binary (clicked=1, not clicked=0)
  - Engagement-weighted: time spent, repeated views, purchases
```

**Position matters more in recommendation than in search.** On a mobile app showing 5 cards, position 1 has 10× the visibility of position 5. NDCG captures this correctly.

### Mean Reciprocal Rank (MRR)

```
MRR = mean over users of (1 / rank of first relevant item)
```

Particularly useful for **next-item prediction** — predicting the single next item a user will interact with. The system succeeds if it ranks that item first.

---

## 15.3 Beyond Accuracy: The Diversity-Accuracy Trade-off

### Why Accuracy Alone Is Insufficient

```
Recommender optimized purely for accuracy:
  User A profile: liked 10 action movies
  Recommendation: [action1, action2, action3, action4, action5]

  Accuracy: high (user will likely click)
  User experience: "I've seen all these"

Recommender with diversity:
  Recommendation: [action1, thriller2, drama3, action4, documentary5]

  Accuracy: slightly lower
  User experience: discovers thriller2, becomes a thriller fan
  Long-term retention: higher
```

Accuracy and diversity trade off — but the trade-off is not fixed. The goal is to find the Pareto frontier: the set of systems where you cannot improve diversity without reducing accuracy, and vice versa.

### Intra-List Diversity (ILD)

Measures how different the recommended items are from each other:

```
ILD@K = (2 / K(K-1)) × Σᵢ Σⱼ>ᵢ dist(itemᵢ, itemⱼ)

dist(i, j) = 1 - cosine_similarity(item_embedding_i, item_embedding_j)
```

ILD is the average pairwise distance between items in the recommendation list. Higher = more diverse.

**Example:**

```
Items with genre embeddings:
  Action1:      [1.0, 0.0, 0.0]
  Action2:      [0.9, 0.1, 0.0]
  Thriller1:    [0.3, 0.7, 0.0]
  Documentary1: [0.0, 0.1, 0.9]

List [Action1, Action2]:         ILD ≈ 0.10  (very similar)
List [Action1, Documentary1]:    ILD ≈ 0.82  (very diverse)
List [Action1, Thriller1, Doc1]: ILD ≈ 0.71  (diverse)
```

### Aggregate Diversity (Catalog Coverage)

What fraction of the item catalog is ever recommended to any user?

```
Catalog Coverage = |∪ᵤ recommended_items(u)| / |item catalog|
```

A system with high catalog coverage exposes users to a wide range of items. A system with low catalog coverage creates a **filter bubble** — most of the catalog is never surfaced.

**Long-tail coverage:** Specifically track what fraction of long-tail items (bottom 80% by popularity) appear in recommendations. A system can have high catalog coverage driven by popular items while completely ignoring the long tail.

### Binomial Diversity (Stricter Catalog Coverage)

Measures whether the distribution of recommended items matches the distribution of available items:

```
Binomial Diversity = 1 - (# items recommended by at least X% of users / |catalog|)
```

Captures concentration: even if 90% of items are recommended, if 10% of items capture 90% of recommendations, diversity is low.

---

## 15.4 Novelty

Novelty measures whether the recommended items are **new to the user** — not what they already know about.

### Self-Information Novelty

Items that are rarely popular carry more information when recommended:

```
Novelty(item) = -log₂ P(item interacted)
             = -log₂ (# users who interacted with item / # total users)

Novelty of recommendation list:
  Nov@K = (1/K) × Σᵢ -log₂ P(itemᵢ)
```

Popular items have low novelty (low self-information). Rare items have high novelty.

**Interpretation:** If you recommend an item that 50% of users have seen (P=0.5), novelty = 1 bit. If you recommend an item 0.1% of users have seen (P=0.001), novelty ≈ 10 bits.

### Expected Popularity Complement (EPC)

Accounts for both item popularity and user history:

```
EPC = (1/K) × Σᵢ (1 - P(item known to user))
```

Higher EPC = more items are new to this specific user. Lower EPC = mostly recommending what the user already knows.

### Novelty vs. Accuracy Trade-off

```
High accuracy, low novelty:   "You liked Titanic, here's Avatar"
High novelty, low accuracy:   "You might enjoy Mongolian throat singing"
Sweet spot:                   "You liked Titanic, here's Parasite" (novel but likely relevant)
```

The sweet spot requires models that understand latent user preferences beyond surface-level similarity.

---

## 15.5 Serendipity

The hardest recommender metric to define and measure. Serendipity captures **unexpected but relevant** recommendations.

```
Serendipitous item = unexpected AND relevant

Not serendipitous:
  Expected + Relevant:     "You liked Harry Potter 1–6, here's HP7"
  Unexpected + Irrelevant: "You liked sci-fi, here's a cooking show"

Serendipitous:
  Unexpected + Relevant:   "You liked sci-fi, you'll love this documentary
                            about quantum physics you didn't know existed"
```

### Formal Serendipity Metrics

**Serendipity = Novelty × Relevance:**

```
Serendipity@K = (1/K) × Σᵢ unexpected(itemᵢ) × relevant(itemᵢ)

unexpected(item) = 1 - similarity(item, user's history)
relevant(item)   = 1 if user engages with item, else 0
```

**Unexpectedness relative to a primitive model:**

```
unexpected(item) = 1 if item NOT in recommendations of a popularity-based baseline
                   0 if item IS in popularity baseline recommendations
```

An item is "serendipitous" if it's both outside what a naive recommender would suggest AND the user actually likes it.

### Why Serendipity Is Hard to Measure Offline

The fundamental problem: **you can only observe serendipity if the item was recommended**. Items you didn't recommend — which might have been serendipitous — are never observed.

This is the **missing counterfactual** problem. True serendipity evaluation requires online A/B tests where you explicitly compare serendipitous recommendation strategies against baselines and measure long-term user satisfaction.

---

## 15.6 Fairness in Recommendation

Recommenders can systematically disadvantage certain users or items. Fairness metrics are critical for consumer-facing systems.

### Two-Sided Fairness

Recommenders have two sets of stakeholders with fairness concerns:

```
User-side fairness:
  Does the system give equally good recommendations to all user groups?
  (demographic groups, new vs. established users, geographic regions)

Item-side fairness:
  Does the system give all items / item providers equal exposure opportunity?
  (small creators, minority producers, new entrants)
```

### User-Side Fairness Metrics

**Equal recommendation quality across groups:**
```
NDCG_groupA ≈ NDCG_groupB   (for all demographic groups)
```

Disparate impact: if NDCG for Group A is significantly lower than Group B, the system disadvantages Group A users.

**Calibration across groups:**
```
For each user group g:
  Expected CTR(g) should match Observed CTR(g)
```

### Item-Side Fairness: Provider Fairness

Particularly important in two-sided markets (Airbnb, Etsy, Spotify for artists):

**Exposure fairness:**
```
Exposure(item i) = Σᵤ Σₖ 1/log₂(k+1) × 𝟙[item i at position k for user u]
```

A fair system ensures items of equal quality receive proportional exposure, regardless of the item provider's size or popularity.

**Proportional fairness:**
```
Exposure(provider) / Relevance(provider) = constant across providers
```

Small providers with relevant items should receive exposure proportional to their relevance, not disadvantaged by network effects.

### Calibration of Recommendation Distributions

User preferences are multi-faceted:
```
User watch history: 60% action, 30% comedy, 10% drama

Well-calibrated recommendation:
  6 action, 3 comedy, 1 drama (in top 10)

Poorly calibrated:
  10 action (exploiting dominant preference; ignores other tastes)
```

Calibration metric:
```
Calibration = Σ_genre |p_history(genre) - p_recommended(genre)|
```

Lower = recommendations better reflect the breadth of user preferences.

---

## 15.7 Coverage and Cold Start Metrics

### User Coverage

```
User Coverage = fraction of users who receive at least K recommendations
```

Some users have no interaction history. A recommender must handle them — or explicitly flag them for fallback strategies.

### Item Coverage and the Long Tail

```
Item Coverage@K = |unique items recommended| / |total items|

Long-tail Coverage = # long-tail items recommended / # long-tail items total

Gini Coefficient of Recommendations:
  Measures inequality in recommendation frequency across items
  Gini = 0: all items recommended equally
  Gini = 1: one item gets all recommendations
```

### Cold Start Metrics

**User cold start:** New user, no history.
**Item cold start:** New item, no interactions.

Metrics:
```
Cold-start Recall@K:
  Evaluate recall separately for new users (< 5 interactions)
  vs. established users (≥ 50 interactions)

Expected traffic fraction:
  What % of recommendations go to new items vs. established items?
  (A system that never recommends new items has 0 item cold-start handling)
```

---

## 15.8 Temporal Metrics

User preferences and item relevance change over time. Static evaluation misses this.

### Temporal Train-Test Split

**Never use random splits for recommender evaluation.** Always split by time:

```
Historical data: Jan–Oct  → Training
Recent data:     Nov      → Validation
Future data:     Dec      → Test

At test time, the model knows nothing about December interactions.
```

Random splits let the model "see the future" — interactions that happen after the recommendation time are used as training labels.

### Temporal Diversity

Does the system recommend different items over time, or does it repeat the same recommendations?

```
Temporal Diversity = diversity of recommendations to user u over time T

If temporal diversity is low: user sees the same items repeatedly → fatigue
```

### Freshness

For time-sensitive domains (news, social media, trending products):

```
Freshness = mean age of recommended items
          = mean(current_time - item_creation_time)
```

A news recommender showing 3-day-old articles has low freshness. Balance freshness against relevance — a fresh but irrelevant article is worthless.

---

## 15.9 Offline vs. Online Evaluation in Recommendation

The offline-online gap (Chapter 2) is especially severe in recommendation.

### The Feedback Loop Problem

Your offline evaluation data was collected under a previous recommendation policy. This creates two problems:

**1. Popularity bias in training data:**
Popular items were recommended more → clicked more → appear more in training data → trained model recommends them more → circular. Offline metrics overestimate the value of popular items.

**2. Missing data problem:**
A truly novel recommendation (item never recommended before) has no clicks in historical data — so offline metrics assign it zero relevance, even if users would have loved it.

### Inverse Propensity Scoring (IPS)

Correct for position and popularity bias in offline evaluation:

```
IPS-corrected metric = Σᵢ (relevance(item, user) / P(item was shown to user)) × metric_contribution

P(item shown) = propensity score: probability the previous policy showed this item
```

Requires knowing (or estimating) the previous policy's propensity scores. Variance can be high — use clipped IPS (cap maximum weight) to stabilize.

### Online A/B Metrics for Recommendation

| Metric | Definition | Latency |
|---|---|---|
| CTR | Clicks / impressions | Hours |
| Dwell time | Time spent on recommended item | Hours |
| Conversion rate | Purchases / impressions | Days |
| D7 retention | Users active 7 days after recommendation | Week |
| Long-tail CTR | CTR restricted to non-popular items | Hours |
| Diversity of clicked items | ILD of items the user actually clicked | Days |
| Regret rate | % users who rate recommendation as "not for me" | Days |

---

## 15.10 The Full Evaluation Framework for Recommenders

```
Accuracy metrics (necessary but not sufficient):
  ├── NDCG@K
  ├── Precision@K, Recall@K, HR@K
  └── MRR (for next-item prediction)

Beyond accuracy (evaluate always):
  ├── Intra-List Diversity (ILD)
  ├── Catalog Coverage
  ├── Long-tail Coverage
  ├── Novelty (self-information)
  └── Serendipity (offline proxy; validate online)

Fairness (required for consumer-facing systems):
  ├── NDCG gap across user groups
  ├── Exposure equity across item providers
  └── Recommendation calibration

Cold start and coverage:
  ├── User Coverage
  ├── Cold-start Recall (new users, new items)
  └── Gini Coefficient

Temporal:
  ├── Always use temporal train-test split
  ├── Freshness (if domain is time-sensitive)
  └── Temporal diversity (repeat rate over sessions)

Online validation:
  ├── A/B test primary metric (CTR, dwell time, or conversion)
  ├── Guardrails: diversity, long-tail CTR, D7 retention
  └── Business metric: revenue, LTV, churn
```

---

## 15.11 Worked Example: Music Recommendation System

**System:** Music streaming app. 50M users, 80M tracks. Goal: improve daily listening session quality.

```
Current system: Collaborative filtering baseline
New system:     Transformer-based sequential model with diversity penalty

Offline evaluation (temporal split: train Jan-Oct, test Nov):

                      Baseline    New System   Delta
NDCG@10:              0.142       0.158        +11.3% ✓
Recall@20:            0.231       0.247        +6.9%  ✓
ILD@10:               0.41        0.58         +41%   ✓ (much more diverse)
Catalog Coverage:     12.3%       19.7%        +60%   ✓
Long-tail Coverage:   4.1%        9.8%         +139%  ✓
Novelty:              2.31 bits   3.47 bits    +50%   ✓
User Coverage:        97.3%       98.1%        +0.8%  ✓
Cold-start Recall@10: 0.041       0.089        +117%  ✓

Fairness:
  NDCG gap (power vs non-power users): 0.12 → 0.07  ✓ (gap reduced)
  Small artist exposure: 8.3% → 14.2%              ✓

Temporal split check:
  Used temporal split ✓ (no random split leakage)
  IPS correction applied to CTR estimates ✓
```

**Online A/B test (2 weeks, 10% of users):**

```
Primary: Session length:       +4.2%   ✓
         Track skip rate:      -3.1%   ✓ (fewer skips = better fit)
         D7 retention:         +1.8%   ✓

Guardrails: 
         Revenue:              +0.9%   ✓ (neutral to positive)
         Load time:            +2ms    ✓ (acceptable)
         Error rate:           flat    ✓

Qualitative: 
         User surveys (n=500): "I discovered new artists" rated 4.3/5
```

**Decision:** Roll out. Diversity gains confirmed online. Novelty and long-tail improvements drive real user satisfaction.

**Lesson:** Offline NDCG improvement of +11% was meaningful, but the more compelling story was diversity, catalog coverage, and cold-start — metrics that pure accuracy evaluation would have missed entirely.

---

## Summary

| Metric | What It Measures | Use When |
|---|---|---|
| NDCG@K | Ranking quality with position discount | Always; primary accuracy metric |
| HR@K | At least one hit per user | Mobile/constrained list settings |
| MRR | Speed of first relevant item | Next-item prediction |
| ILD | Diversity within a recommendation list | Always alongside accuracy |
| Catalog Coverage | Breadth of catalog exposed | Always; filter bubble detection |
| Long-tail Coverage | Exposure to non-popular items | Content creator fairness |
| Novelty | How new items are to users | Combating over-familiarity |
| Serendipity | Unexpected + relevant | Long-term engagement; needs online validation |
| Calibration | Matches user's taste distribution | Breadth of taste satisfaction |
| Provider Exposure | Fair exposure to all item providers | Two-sided marketplace fairness |
| Temporal split | Evaluation integrity | Always; never use random split |
| IPS correction | Debiasing offline CTR estimates | When propensity scores available |

---

## Further Reading

- Herlocker et al. — *Evaluating Collaborative Filtering Recommender Systems* (ACM TOIS, 2004) — foundational
- Vargas & Castells — *Rank and Relevance in Novelty and Diversity Metrics for RS* (RecSys 2011)
- Abdollahpouri et al. — *The Unfairness of Popularity Bias in Recommendation* (2019)
- Steck — *Calibrated Recommendations* (RecSys 2018)
- Joachims et al. — *Unbiased Learning-to-Rank with Biased Feedback* (WSDM 2017) — IPS for ranking
- Ekstrand et al. — *All the Cool Kids, How Do They Fit In? Popularity and Demographic Biases in RS* (FAccT 2018)

---

*Next: Chapter 16 — Fairness & Bias Metrics*
