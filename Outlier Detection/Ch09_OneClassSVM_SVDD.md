# Chapter 9: One-Class SVM & SVDD

## 9.1 Motivation

Chapters 6–8 all assumed a roughly elliptical/Gaussian shape for "normal" data (covariance-based methods) or a linear subspace (PCA). What if the normal region is an arbitrary, non-elliptical, possibly non-convex shape? One-Class SVM and Support Vector Data Description (SVDD) use the **kernel trick** to fit a flexible boundary around normal data **without assuming any particular parametric distribution** — trading interpretability and closed-form thresholds for shape flexibility.

## 9.2 SVDD — The More Geometrically Intuitive of the Two

**Intuition:** find the **smallest hypersphere** (center $a$, radius $R$) that encloses (almost all of) the data. Points outside the sphere are outliers.

**Formal (primal) objective:**
$$
\min_{a,\,R,\,\xi}\ R^2 + C\sum_{i=1}^n \xi_i
$$
subject to:
$$
\|x_i - a\|^2 \le R^2 + \xi_i, \qquad \xi_i \ge 0 \ \ \forall i
$$

- $R^2$: minimize the sphere's volume (tightest possible fit)
- $\xi_i$: slack variables allowing some points to fall outside the sphere (soft margin, same idea as soft-margin SVM in supervised classification)
- $C$: regularization constant controlling the tradeoff between a tight sphere and tolerating outliers in the training set — larger $C$ penalizes violations more, forcing a tighter fit around *all* training points (risk of overfitting to training-set noise); smaller $C$ allows more training points to be excluded as slack, yielding a more conservative, smaller sphere.

**Dual form** (what's actually solved in practice, and where the kernel trick enters):
$$
\max_\alpha \sum_i \alpha_i (x_i\cdot x_i) - \sum_{i,j}\alpha_i\alpha_j (x_i \cdot x_j)
$$
subject to $0\le\alpha_i\le C$, $\sum_i \alpha_i = 1$.

Replacing every dot product $x_i\cdot x_j$ with a **kernel function** $K(x_i,x_j)$ lets the sphere become an arbitrarily flexible shape in the *original* feature space, while still being a simple sphere in a higher-dimensional (implicit) feature space — this is exactly the kernel trick from supervised SVMs, reused here.

**Decision function** (test if a new point $x$ is inside the learned boundary):
$$
\|x-a\|^2 = K(x,x) - 2\sum_i \alpha_i K(x,x_i) + \sum_{i,j}\alpha_i\alpha_j K(x_i,x_j) \ \lessgtr\ R^2
$$
If this exceeds $R^2$, $x$ is flagged as an outlier.

## 9.3 One-Class SVM (Schölkopf formulation)

**Intuition:** instead of a sphere, separate the data from the **origin** with maximum margin in kernel space — conceptually, "push a hyperplane as far from the origin as possible while keeping (almost all) data on the far side."

**Primal objective:**
$$
\min_{w,\,\rho,\,\xi}\ \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_i \xi_i
$$
subject to:
$$
w\cdot\phi(x_i) \ge \rho - \xi_i, \qquad \xi_i\ge0
$$

**Decision function:**
$$
f(x) = \text{sign}\big(w\cdot\phi(x) - \rho\big)
$$
Negative → outlier.

**The $\nu$ parameter (crucial, and a favorite interview detail):** $\nu\in(0,1]$ has a dual, provable interpretation:
- $\nu$ is an **upper bound** on the fraction of training points allowed to be outliers (margin errors).
- $\nu$ is simultaneously a **lower bound** on the fraction of points that end up as support vectors.

So setting $\nu=0.05$ tells the model directly: "expect roughly at most 5% contamination in this training set." This is a uniquely direct, interpretable knob compared to $C$ in SVDD, and interviewers often ask specifically about $\nu$'s dual meaning.

**Equivalence note:** with the RBF kernel specifically, One-Class SVM and SVDD produce **identical decision boundaries** — the RBF kernel satisfies $K(x,x)=1$ for all $x$ (constant), which collapses the "separate from origin" and "smallest enclosing sphere" formulations into the same optimization problem. With other kernels, they can differ. This equivalence is a well-known but often-missed detail worth having ready.

## 9.4 Worked Numerical (Conceptual, RBF Kernel)

**RBF kernel:**
$$
K(x_i,x_j) = \exp\left(-\gamma\|x_i-x_j\|^2\right)
$$

Suppose training data are tightly clustered around $(0,0)$ with typical pairwise distances small, so $K(x_i,x_j)\approx 1$ for in-cluster pairs (since $\|x_i-x_j\|^2\approx0$). A test point far away, say at distance $d=5$ from the cluster with $\gamma=0.5$:
$$
K(x_{test}, x_i) = \exp(-0.5\times25) = \exp(-12.5) \approx 3.7\times10^{-6}
$$

This kernel value is essentially **zero** — meaning in the decision function (§9.2), the test point contributes almost nothing to the "similarity to training data" sum, so $\|x-a\|^2$ comes out large (close to $K(x,x)=1$ alone, with almost no offsetting similarity term), pushing it well past $R^2$ → **flagged as an outlier.**

**Key numerical intuition:** the RBF kernel's exponential decay means similarity collapses extremely fast with distance — this is *why* One-Class SVM/SVDD with RBF can carve out tight, non-elliptical boundaries: points just slightly outside the training data's support get almost zero kernel similarity to everything, making them easy to separate, while covariance-based methods (Ch.6–8) would only flag them once they cross a smooth quadratic (elliptical) distance threshold.

## 9.5 Diagnosis: When to Use One-Class SVM/SVDD

| Condition | Recommendation |
|---|---|
| Normal region is non-convex / arbitrarily shaped (e.g., a crescent, ring, or multi-lobed cluster) | Strong fit — this is the primary advantage over Ch.6–8 |
| You have a rough estimate of expected contamination fraction | Use One-Class SVM, set $\nu$ directly to that fraction |
| Large training sets (tens of thousands+ points) | Caution — SVM training is $O(n^2)$ to $O(n^3)$ depending on solver; scales poorly compared to Isolation Forest (Ch.12) or LOF (Ch.11) |
| High-dimensional sparse data (e.g., text/TF-IDF features) | Can work well with linear kernel; RBF kernel struggles as distances become less meaningful in very high dimensions (curse of dimensionality, revisited in Ch.13) |
| Need probabilistic/interpretable output (a calibrated anomaly probability) | Poor fit — output is a hard boundary decision or an uncalibrated distance-to-boundary score, not a probability |

## 9.6 Production Considerations
- Training cost scales poorly with $n$ (quadratic programming over all pairwise kernel evaluations) — rarely used as-is on full-scale production data; more common to train on a representative sample or a reduced "prototype" set.
- Kernel and hyperparameter choice ($\gamma$ for RBF, $\nu$ or $C$) require careful tuning via cross-validation on a validation set that itself must be free of too much contamination — a similar chicken-and-egg concern to the circularity problem from Ch.6-7, though less severe since these are hyperparameters, not the model's core estimate of normality.
- Because there's no simple closed-form retraining update (unlike, say, updating a running mean/covariance), online/streaming environments with high refresh-rate requirements often prefer Isolation Forest or streaming-friendly density methods instead.

## 9.7 Interview Traps
- Not knowing $\nu$'s dual interpretation (upper bound on outlier fraction / lower bound on support vector fraction) — this is probably the single most commonly asked follow-up question for this topic.
- Confusing One-Class SVM's "separate from the origin" framing with SVDD's "smallest enclosing sphere" framing as if they're unrelated — being unable to state the RBF-kernel equivalence between them is a missed opportunity to show deeper understanding.
- Assuming the kernel trick makes these methods scale-free with respect to dimensionality — very high-dimensional data still suffers from distance concentration effects (all pairwise distances becoming similar), just like any distance-based method.
- Treating the decision function's raw output as a calibrated anomaly probability — it's a signed distance to a boundary, not a probability, and shouldn't be reported as one without further calibration.

## 9.8 L5-Differentiating Talking Points
- Stating the RBF-kernel equivalence between One-Class SVM and SVDD unprompted — a detail that immediately signals depth beyond textbook familiarity.
- Explicitly contrasting this chapter's kernel/boundary-based philosophy against Ch.6–8's density/distribution-based philosophy: "these don't try to model $f(x)$ or its inverse density at all — they directly learn a decision boundary that separates dense regions from sparse ones, sidestepping distributional assumptions entirely." This ties back cleanly to the Ch.1 unifying framework (§1.2) by explicitly naming what makes this family *different* from the rest.
- Being explicit about scalability limitations and naming Isolation Forest (Ch.12) as the practical production alternative when $n$ is large — shows calibrated judgment about tool selection under real constraints, not just theoretical knowledge.

## 9.9 Comprehension Check
1. Explain the dual meaning of the $\nu$ parameter in One-Class SVM, and why both interpretations must hold simultaneously.
2. Under what specific kernel condition do One-Class SVM and SVDD produce identical decision boundaries, and why?
3. Why does the RBF kernel's exponential decay make it especially effective at carving tight, non-elliptical decision boundaries compared to a linear kernel?
4. Give one concrete reason why One-Class SVM/SVDD would be a poor choice for a production system needing to score millions of new points per hour against a frequently-updated reference set.

---
*Next: Chapter 10 — Autoencoder-Based Outlier Detection (reconstruction loss thresholding).*
