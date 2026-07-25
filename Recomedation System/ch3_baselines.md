# Chapter 3: Baselines — Popularity, Content-Based Filtering, and Why They Matter

## 1. Intuition

Before touching collaborative filtering or deep learning, every serious recsys build starts with baselines. Not because they're expected to win, but because they define the **floor** — if your fancy matrix factorization model can't beat "just show the most popular items," something is broken, and you need to know that before you spend three weeks tuning embeddings.

Baselines also matter enormously in interviews: when asked to design a recommender, jumping straight to a neural architecture without mentioning what you'd compare it against is a signal of inexperience. L5 candidates state the baseline first, then justify why something more complex is needed.

## 2. Popularity-Based Recommendations

The simplest possible recommender: rank items by a global popularity signal (total interactions, purchases, views) and show the same list to everyone (or everyone within a coarse segment like country/age-bracket).

$$\text{score}(i) = \sum_{u} \mathbb{1}[\text{interaction}(u, i)]$$

Often adjusted with **time-decay** so stale-but-historically-popular items don't dominate forever:

$$\text{score}(i) = \sum_{u,t} \mathbb{1}[\text{interaction}(u,i,t)] \cdot e^{-\lambda(T - t)}$$

where $T$ is current time and $\lambda$ controls how fast popularity decays.

**Strengths**: zero cold-start problem for users (works even with no user history), trivial to compute and serve, surprisingly hard to beat in aggregate metrics because popular items are popular for a reason.

**Weaknesses**: zero personalization — every user gets the same list. Reinforces a rich-get-richer feedback loop (Ch. 24 — popular items get shown more, therefore get more clicks, therefore stay popular). No mechanism for niche or long-tail content to ever surface.

## 3. Content-Based Filtering

Recommend items similar to what a user has liked before, based on **item features/metadata** rather than other users' behavior.

Mechanism: represent items as feature vectors (genre, text embeddings, tags, price, brand), build a user profile as an aggregate (often a weighted average) of the features of items that user liked, then recommend items whose feature vector is most similar (cosine similarity) to the user profile.

$$\text{user\_profile}(u) = \frac{\sum_{i \in I_u} w_i \cdot \vec{v}_i}{\sum_{i \in I_u} w_i}$$

$$\text{score}(u, i) = \cos(\text{user\_profile}(u), \vec{v}_i)$$

where $I_u$ = items user $u$ has interacted with, $w_i$ = interaction weight (e.g., rating or recency), $\vec{v}_i$ = feature vector of item $i$.

**Strengths**: solves item cold-start (Ch. 1) — a brand-new item has a feature vector the moment it's created, no interaction history needed. Explainable ("recommended because you liked X, which is similar in genre/style").

**Weaknesses**: over-specializes — tends to recommend items very similar to what's already been consumed, limiting discovery (the "filter bubble" problem, Ch. 21). Requires good item features to exist, which isn't free — someone has to engineer or embed them. Doesn't leverage the wisdom-of-crowds signal that collaborative filtering exploits (i.e., "users like you also liked Y" is invisible to content-based methods).

## 4. Worked Numerical Example

Three items with feature vectors (genre indicators: [Action, Comedy, Drama]):

| Item | Vector |
|---|---|
| A | [1, 0, 0] |
| B | [0.8, 0.2, 0] |
| C | [0, 0, 1] |

User liked A (rating 5) and B (rating 3). Build user profile as rating-weighted average:

$$\text{profile} = \frac{5 \cdot [1,0,0] + 3 \cdot [0.8, 0.2, 0]}{5+3} = \frac{[5,0,0] + [2.4, 0.6, 0]}{8} = \frac{[7.4, 0.6, 0]}{8} = [0.925, 0.075, 0]$$

Now score a new candidate item D = [0.9, 0.1, 0] (another action-leaning item) via cosine similarity:

$$\cos(\text{profile}, D) = \frac{(0.925)(0.9) + (0.075)(0.1) + 0}{\|\text{profile}\| \cdot \|D\|}$$

Numerator = 0.8325 + 0.0075 = 0.84
$\|\text{profile}\| = \sqrt{0.925^2 + 0.075^2} = \sqrt{0.8556+0.0056}=\sqrt{0.8612}=0.928$
$\|D\| = \sqrt{0.9^2+0.1^2}=\sqrt{0.82}=0.9055$

$$\cos = \frac{0.84}{0.928 \times 0.9055} = \frac{0.84}{0.8403} \approx \mathbf{0.9996}$$

Near-perfect similarity — D gets recommended strongly, correctly, since the user profile leans heavily action. Compare this to item C = [0,0,1] (pure drama): cosine similarity would be ~0, correctly filtered out. Notice how content-based filtering makes this decision using **zero information about other users** — purely from item features and this one user's own history.

## 5. Production Considerations

- Popularity baselines are frequently kept in production permanently as a **fallback tier** — when personalized models have no signal (brand-new user with zero history), the system gracefully degrades to "trending now" rather than showing nothing or random items.
- Real systems blend popularity into personalized rankings even for established users, often as one input feature into a learned ranking model (Module 5) rather than a standalone recommender — pure popularity is rarely the final production system, but it's almost always present *somewhere* in the pipeline (e.g., candidate generation stage).
- Content-based filtering is the default answer to item cold-start (Ch. 1) precisely because it needs no interaction data — new items are typically injected into the recommendation pool via content similarity until enough behavioral data accumulates to let collaborative signals take over. This handoff (content-based → collaborative as data accumulates) is itself a designed system behavior, not automatic.

## 6. Interview Traps

- Dismissing popularity as "too simple to mention" — interviewers explicitly want to hear it named as the baseline/floor, and as the cold-start fallback.
- Confusing content-based filtering with collaborative filtering — content-based uses item features and a single user's history; it never uses other users' behavior at all. This is a common terminology slip.
- Not mentioning that content-based filtering causes over-specialization/filter bubbles — interviewers listening for awareness of this specific limitation.
- Proposing content-based filtering as if it fully solves cold-start "for free" — it still requires item features to exist and be meaningful, which is itself an engineering cost.

## 7. L5-Differentiating Talking Points

- State explicitly that baselines exist to establish a **floor**, and that any proposed complex model should be justified against measurable lift over that floor — signals engineering discipline, not just modeling enthusiasm.
- Point out the popularity-feedback-loop problem (rich-get-richer) unprompted — this connects directly to Module 6 fairness/diversity discussions and shows systems-level thinking.
- Frame content-based filtering and collaborative filtering as complementary, not competing — most production systems are hybrid, using content-based signals specifically to patch the collaborative approach's cold-start blind spot.
- Mention that popularity baselines are rarely fully retired — they persist as a serving-time fallback tier, which is a subtle but important production detail interviewers value.

## 8. Comprehension Check

1. Why is a popularity-based recommender considered zero-personalization, and in what scenario would it still be used in a fully mature, personalized production system?
2. Why does content-based filtering solve item cold-start but not user cold-start?
3. What does the "filter bubble" limitation of content-based filtering mean concretely, and how would you detect it happening in production?
4. In the worked example, why would the cosine similarity between the user profile and item C be near zero, and what does that imply practically?
5. Why do most production systems combine popularity, content-based, and collaborative signals rather than picking just one?
