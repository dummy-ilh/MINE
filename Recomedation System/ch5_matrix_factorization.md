# Chapter 5: Matrix Factorization (SVD, ALS) — Full Numerical Walkthrough

## 1. Intuition

Neighborhood CF (Ch. 4) compares users or items directly against each other using raw rating vectors. Matrix factorization (MF) takes a different bet: instead of comparing raw rows/columns, it assumes there's a small number of **latent factors** (hidden dimensions like "how much action," "how arthouse," "how much for kids") that explain most of the rating variance, and it learns compact vector representations of users and items in that latent space.

The interaction matrix $R$ (users × items) is approximated as the product of two much smaller matrices:

$$R \approx P Q^T$$

where $P$ is (users × k) and $Q$ is (items × k), $k \ll$ number of users or items. A user's row in $P$ and an item's row in $Q$ are learned embeddings — their dot product reconstructs the predicted rating. This is the direct conceptual ancestor of every embedding-based recsys architecture in Module 4.

## 2. The Model

$$\hat{R}_{ui} = p_u^T q_i = \sum_{f=1}^{k} p_{uf} q_{if}$$

$p_u \in \mathbb{R}^k$ = latent vector for user $u$. $q_i \in \mathbb{R}^k$ = latent vector for item $i$. The dot product is high when a user's factor weights align with an item's factor weights (e.g., both have high weight on the "action" latent dimension).

In practice, bias terms are added because raw dot products alone don't capture systematic effects (some users just rate everything higher, some items are just universally better-received):

$$\hat{R}_{ui} = \mu + b_u + b_i + p_u^T q_i$$

- $\mu$ = global average rating
- $b_u$ = user bias (how much higher/lower this user rates vs. average)
- $b_i$ = item bias (how much higher/lower this item is rated vs. average)

This bias decomposition is a classic interview trap point (Ch. 1's "global mean trap") — omitting it makes the model implicitly conflate genuine preference with rating-scale idiosyncrasy.

## 3. Learning via SGD (the Simon Funk / Netflix Prize approach)

Minimize regularized squared error over observed ratings only:

$$\min_{P,Q,b} \sum_{(u,i) \in \text{observed}} \left(R_{ui} - \mu - b_u - b_i - p_u^T q_i\right)^2 + \lambda\left(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2\right)$$

$\lambda$ = regularization strength, preventing overfitting to sparse observed entries. SGD update rule, per observed rating, with error $e_{ui} = R_{ui} - \hat{R}_{ui}$:

$$p_u \leftarrow p_u + \gamma(e_{ui} \cdot q_i - \lambda p_u)$$
$$q_i \leftarrow q_i + \gamma(e_{ui} \cdot p_u - \lambda q_i)$$
$$b_u \leftarrow b_u + \gamma(e_{ui} - \lambda b_u), \quad b_i \leftarrow b_i + \gamma(e_{ui} - \lambda b_i)$$

$\gamma$ = learning rate. Note the elegant symmetry: the update to $p_u$ uses $q_i$ (and vice versa) — each factor gets nudged in the direction that reduces error, scaled by the other side's current vector.

## 4. Learning via ALS (Alternating Least Squares)

SGD updates one rating at a time; ALS instead fixes one matrix and solves for the other **exactly** (closed-form least squares), then alternates:

1. Fix $Q$, solve for optimal $P$ (each $p_u$ independently, closed-form)
2. Fix $P$, solve for optimal $Q$ (each $q_i$ independently, closed-form)
3. Repeat until convergence

Why ALS matters in production, specifically: when $Q$ is fixed, solving for each $p_u$ is an independent least-squares problem — this **parallelizes trivially across users** (and symmetrically across items in the other step), which is why ALS is the standard choice for distributed systems (Spark's `ALS` implementation exists for exactly this reason). SGD is inherently sequential/harder to parallelize cleanly, though minibatch/distributed variants exist.

ALS is also the natural fit for **implicit feedback** (Ch. 6) because the confidence-weighted loss used there has a closed-form per-user/per-item solution that SGD handles less cleanly.

## 5. Worked Numerical Example (One SGD Update Step)

Say $k=2$ latent factors, $\mu = 3.0$, $\gamma=0.1$, $\lambda=0.02$.

Current state for user $u$ and item $i$:
- $b_u = 0.2$, $b_i = -0.1$
- $p_u = [0.6, 0.2]$, $q_i = [0.4, 0.5]$
- Observed rating $R_{ui} = 5$

**Step 1 — Predict:**
$$\hat{R}_{ui} = \mu + b_u + b_i + p_u \cdot q_i = 3.0 + 0.2 - 0.1 + (0.6\times0.4 + 0.2\times0.5)$$
$$= 3.1 + (0.24+0.10) = 3.1+0.34 = 3.44$$

**Step 2 — Error:**
$$e_{ui} = 5 - 3.44 = 1.56$$

**Step 3 — Update biases:**
$$b_u \leftarrow 0.2 + 0.1(1.56 - 0.02\times0.2) = 0.2+0.1(1.56-0.004)=0.2+0.1(1.556)=0.2+0.1556=0.3556$$
$$b_i \leftarrow -0.1+0.1(1.56-0.02\times(-0.1))=-0.1+0.1(1.56+0.002)=-0.1+0.1562=0.0562$$

**Step 4 — Update latent vectors** (using the *old* $p_u, q_i$ values on the right-hand side, since both updates happen simultaneously using pre-update values):
$$p_u \leftarrow [0.6,0.2] + 0.1\big(1.56\times[0.4,0.5] - 0.02\times[0.6,0.2]\big)$$
$$= [0.6,0.2] + 0.1\big([0.624,0.78] - [0.012,0.004]\big) = [0.6,0.2]+0.1[0.612,0.776]$$
$$= [0.6,0.2]+[0.0612,0.0776] = \mathbf{[0.6612, 0.2776]}$$

$$q_i \leftarrow [0.4,0.5] + 0.1\big(1.56\times[0.6,0.2] - 0.02\times[0.4,0.5]\big)$$
$$= [0.4,0.5]+0.1\big([0.936,0.312]-[0.008,0.01]\big) = [0.4,0.5]+0.1[0.928,0.302]$$
$$= [0.4,0.5]+[0.0928,0.0302] = \mathbf{[0.4928, 0.5302]}$$

**Step 5 — Verify improvement:** recompute prediction with updated values:
$$\hat{R}_{ui}^{new} = 3.0+0.3556+0.0562+(0.6612\times0.4928+0.2776\times0.5302)$$
$$= 3.4118+(0.3259+0.1472)=3.4118+0.4731=3.885$$

Prediction moved from 3.44 → 3.885, closer to the true rating of 5 — one gradient step correctly nudges the model in the right direction. Repeating this over all observed ratings, many epochs, converges toward a good factorization.

## 6. Why $k$ (Number of Latent Factors) Matters

Small $k$ → underfitting, can't capture nuanced taste dimensions. Large $k$ → overfitting risk (especially with sparse data) and higher serving cost (every embedding lookup and dot product scales with $k$). $k$ is a hyperparameter tuned via validation performance (Ch. 2 metrics) — typically tens to low hundreds in classical MF, though modern deep two-tower models (Ch. 12) push into the hundreds of dimensions.

## 7. Production Considerations

- Pure SVD in the strict linear-algebra sense (full decomposition) requires a **complete** matrix — doesn't work directly on sparse data with missing entries. What's actually used in practice is **regularized SGD/ALS factorization**, which is often loosely called "SVD" (this terminology looseness, inherited from the Netflix Prize era, is itself a common interview confusion point worth clarifying explicitly.)
- Cold-start remains unsolved by MF alone — a new user/item has no learned latent vector until they accumulate interactions, which is exactly why MF is usually paired with content-based fallbacks (Ch. 3) or hybrid architectures.
- At serving time, once trained, recommending for a user reduces to computing $p_u^T q_i$ for all candidate items — this is a **nearest-neighbor search problem** in latent space, foreshadowing the ANN/retrieval infrastructure in Module 5 (Ch. 17).
- ALS is embarrassingly parallel and is the standard choice in distributed big-data settings (Spark MLlib); SGD variants (including deep-learning-style minibatch SGD) dominate when the model is embedded in a larger neural architecture that's trained end-to-end (Module 4).

## 8. Interview Traps

- Calling this "SVD" without acknowledging that true SVD requires a complete matrix, while production MF uses regularized gradient-based factorization on the sparse observed set only — interviewers often probe this exact nuance.
- Forgetting the bias terms ($\mu, b_u, b_i$) and only writing $\hat{R}_{ui}=p_u^Tq_i$ — this ties back to the Chapter 1 "global mean trap" and is heavily tested.
- Not being able to explain *why* ALS parallelizes better than SGD — the answer is specifically that fixing one matrix turns the problem into independent, closed-form least-squares problems per user/item row.
- Assuming MF solves cold-start — it explicitly does not; that's the entire reason hybrid systems exist.

## 9. L5-Differentiating Talking Points

- Proactively distinguish "SVD" as colloquially used in industry (regularized factorization via SGD/ALS on sparse data) from textbook SVD (exact decomposition of a complete matrix) — this single clarification signals genuine depth versus memorized terminology.
- Explain the ALS-vs-SGD choice as a **systems/infrastructure decision** (parallelizability, whether the model is standalone or embedded in a larger end-to-end neural pipeline), not just an optimization detail.
- Connect MF's latent vectors directly to embeddings used in Module 4's two-tower architecture (Ch. 12) — MF is literally a linear, shallow special case of the two-tower model, which is a powerful unifying insight interviewers reward.
- Mention $k$ as a concrete tunable hyperparameter with a real accuracy/latency/overfitting trade-off, not just an abstract "hyperparameter to tune."

## 10. Comprehension Check

1. Why do production "SVD-style" recommenders actually use regularized SGD/ALS rather than true singular value decomposition?
2. What role do the bias terms $b_u, b_i, \mu$ play, and what goes wrong if you omit them?
3. Why does ALS parallelize more naturally than SGD in a distributed setting?
4. What happens to model quality if $k$ is set too small? Too large?
5. How does matrix factorization relate conceptually to the two-tower architecture covered later in the curriculum?
