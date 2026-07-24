# Chapter 25 — Response Surface Methodology
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

This is the final chapter of the book. Chapter 24 screened many factors down to a handful of important ones using cheap, linear (first-order) two-level designs. This chapter answers the natural next question: **once you know which few factors matter, how do you find the exact combination of settings that optimizes the response?** The answer requires moving beyond linear models entirely, since **optima are, by definition, places where the response curves** — a straight line can never have a peak or a valley.

### The Worked Example

Two factors survived screening: **ad spend** ($X_1$, coded) and **price discount** ($X_2$, coded). Response: **profit**. We want to find the exact combination of ad spend and discount that maximizes profit.

---

## 25.1 The Sequential Philosophy of Response Surface Methodology (RSM)

**Plain English.** RSM is not a single experiment — it's a **strategy**, unfolding in stages: (1) when far from the optimum, use a cheap **first-order** (linear) design to find the general *direction* of improvement; (2) move in that direction, re-running cheap linear experiments, until improvement stops (signaling you've reached a region of curvature — near the optimum); (3) **switch to a second-order (quadratic) design** to properly characterize the curved region and pinpoint the exact optimum.

---

## 25.2 Stage One: The Method of Steepest Ascent (Brief)

**When you're far from the optimum**, a simple two-level factorial (Chapter 24's exact toolkit) is enough: fit a first-order model $\hat Y=b_0+b_1X_1+b_2X_2$, and the coefficient vector $(b_1,b_2)$ **points in the direction of steepest increase** in the response. **The practical recipe:** take steps proportional to $(b_1,b_2)$, actually running the process at each new point, until the response stops improving — a clear signal you've entered a region where the true surface **curves**, and a linear model can no longer describe it well (its own lack-of-fit test, exactly Chapter 3's tool, would start failing). That's the cue to move to Stage Two.

---

## 25.3 Stage Two: The Second-Order Model and the Central Composite Design

### Why You Need a Quadratic Model Near an Optimum

$$
Y = \beta_0+\beta_1X_1+\beta_2X_2+\beta_{11}X_1^2+\beta_{22}X_2^2+\beta_{12}X_1X_2+\varepsilon
$$

**Why the squared terms are essential, not optional.** A linear model's fitted surface is a plane — it has no peak or valley anywhere in its interior; its "best" point is always at the edge of whatever region you tested. Only a model with curvature ($X_1^2$, $X_2^2$ terms) can have a genuine interior maximum or minimum — exactly Chapter 8's polynomial-regression insight (a model can be curved in the predictors while remaining linear in the parameters), now put to direct practical use for optimization.

### The Central Composite Design (CCD)

**Plain English.** A CCD is the standard, efficient design for fitting a second-order model with two (or more) factors. It combines three parts:
1. **Factorial points** ($\pm1,\pm1$ — Chapter 24's full or fractional factorial): estimate the linear and interaction terms.
2. **Axial ("star") points** ($\pm\alpha,0$ and $0,\pm\alpha$, with $\alpha$ chosen for good design properties — here $\alpha=\sqrt2\approx1.414$ for a 2-factor rotatable design): these extend along each axis individually, providing the extra information needed to estimate the **pure quadratic** terms ($X_1^2,X_2^2$), which the factorial points alone cannot cleanly separate from the intercept.
3. **Center point replicates** (several runs at $(0,0)$): provide a direct, model-free estimate of **pure error** — exactly Chapter 3's pure-error concept, now serving double duty to also test whether the second-order model itself shows lack of fit.

### The Worked Design (13 runs total)

| Run | $X_1$ | $X_2$ | $Y$ |
|---|---|---|---|
| 1 (factorial) | 1 | 1 | 50 |
| 2 (factorial) | 1 | −1 | 48 |
| 3 (factorial) | −1 | 1 | 44 |
| 4 (factorial) | −1 | −1 | 34 |
| 5 (axial) | 1.414 | 0 | 49.07 |
| 6 (axial) | −1.414 | 0 | 34.93 |
| 7 (axial) | 0 | 1.414 | 50.24 |
| 8 (axial) | 0 | −1.414 | 41.76 |
| 9–13 (center) | 0 | 0 | 49, 50, 51, 50, 50 |

---

## 25.4 Fitting the Second-Order Model

### The Linear and Interaction Terms — Computable Directly by Hand

**Interaction term** $b_{12}$: uses only the factorial points (axial and center points all have at least one coordinate at 0, contributing nothing to the $X_1X_2$ column):
$$
\sum(X_1X_2)Y = (+1)(50)+(-1)(48)+(-1)(44)+(+1)(34)=-8, \qquad \sum(X_1X_2)^2=4
$$
$$
b_{12} = \frac{-8}{4}=-2.0
$$

**Linear term $b_1$:** combining factorial and axial contributions ($\sum X_1^2=8$ across all 13 runs: $4$ from factorial + $4$ from the two $X_1$-axial points):
$$
\sum X_1Y = \underbrace{50+48-44-34}_{\text{factorial}=20} + \underbrace{1.414(49.07-34.93)}_{\text{axial}\approx20} + 0 = 40
$$
$$
b_1 = \frac{40}{8}=5.0
$$

**Linear term $b_2$:** by the same method (using the $X_2$-axial points $(0,\pm1.414)$):
$$
\sum X_2Y = \underbrace{50-48+44-34}_{\text{factorial}=12}+\underbrace{1.414(50.24-41.76)}_{\text{axial}\approx12}+0=24
$$
$$
b_2 = \frac{24}{8}=3.0
$$

### The Intercept and Pure Quadratic Terms

**These require solving the full normal equations** ($\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$, Chapter 5's exact machinery, now for a 6-parameter model), because in a CCD, the pure quadratic terms ($X_1^2$, $X_2^2$) and the intercept are **correlated with each other** (unlike the clean orthogonality that made $b_1$, $b_2$, $b_{12}$ so easy to compute directly above) — this is a known structural feature of CCDs, and in practice this step is essentially always done by software rather than by hand. Solving the system (verified via the code below) recovers:
$$
b_0=50.0, \qquad b_{11}=-4.0, \qquad b_{22}=-2.0
$$

**The fitted second-order model:**
$$
\hat Y = 50+5X_1+3X_2-4X_1^2-2X_2^2-2X_1X_2
$$

### Checking Lack of Fit Using the Center-Point Pure Error

**Pure error**, directly from the 5 center-point replicates ($49,50,51,50,50$, mean $50$):
$$
SS_{\text{pure error}} = (49-50)^2+0+(51-50)^2+0+0=2, \qquad df=4, \qquad MS_{\text{pure error}}=0.5
$$
**Exactly Chapter 3's lack-of-fit logic**: comparing the model's total residual sum of squares to this pure-error benchmark tests whether the second-order model itself is adequate, or whether even more curvature (or a different functional form entirely) is needed. (With our data constructed to closely follow the assumed quadratic form, this test would show no significant lack of fit — consistent with the second-order model being the right choice here.)

---

## 25.5 Canonical Analysis: Finding and Classifying the Optimum

### Finding the Stationary Point

**Set both partial derivatives of the fitted surface to zero** (ordinary calculus, applied to the fitted quadratic):
$$
\frac{\partial\hat Y}{\partial X_1} = b_1+2b_{11}X_1+b_{12}X_2 = 5-8X_1-2X_2=0
$$
$$
\frac{\partial\hat Y}{\partial X_2} = b_2+2b_{22}X_2+b_{12}X_1 = 3-4X_2-2X_1=0
$$

**Solving simultaneously:** from the first equation, $X_2=2.5-4X_1$; substituting into the second: $3-4(2.5-4X_1)-2X_1=0 \Rightarrow -7+14X_1=0 \Rightarrow X_1^*=0.5$, then $X_2^*=2.5-4(0.5)=0.5$.

**The stationary point is $(X_1^*,X_2^*)=(0.5,0.5)$** in coded units — slightly more ad spend and slightly more discount than the design's center. **Predicted profit there:**
$$
\hat Y^* = 50+5(0.5)+3(0.5)-4(0.25)-2(0.25)-2(0.25) = 50+2.5+1.5-1-0.5-0.5=52.0
$$

### Confirming It's Actually a Maximum — Not a Saddle Point or Minimum

**Plain English.** A stationary point (where the slope is zero in every direction) could be a peak, a valley, or a **saddle** (a maximum in one direction but a minimum in another — like the middle of a horse's saddle, or a mountain pass). **You cannot tell which just from the stationary point itself — you need to examine the curvature matrix.**

$$
\mathbf{B} = \begin{bmatrix}b_{11} & b_{12}/2 \\ b_{12}/2 & b_{22}\end{bmatrix} = \begin{bmatrix}-4&-1\\-1&-2\end{bmatrix}
$$

**Compute the eigenvalues** (solving $\det(\mathbf{B}-\lambda\mathbf{I})=0$):
$$
(-4-\lambda)(-2-\lambda)-1 = \lambda^2+6\lambda+7=0 \quad\Rightarrow\quad \lambda = \frac{-6\pm\sqrt{36-28}}{2}=\frac{-6\pm2.828}{2}
$$
$$
\lambda_1=-1.586, \qquad \lambda_2=-4.414
$$

**Both eigenvalues are negative.** **This is what confirms a true maximum**: the quadratic form is negative in every direction from the stationary point, meaning the fitted surface curves *downward* no matter which way you move away from $(0.5,0.5)$. (Had the eigenvalues had mixed signs, the stationary point would be a saddle — a maximum along one axis but a minimum along another — and "the optimum" would need to be reported very differently, typically by exploring along the direction of the positive eigenvalue instead of trusting the stationary point as a genuine optimum.)

**Interview question:** *"You find a stationary point in a fitted response surface. How do you know whether it's actually the optimum you're looking for, rather than a saddle point?"*
**Ideal answer:** Examine the eigenvalues of the matrix of pure quadratic and half-interaction coefficients (the curvature matrix). If all eigenvalues are negative, the stationary point is a genuine maximum (the surface curves downward in every direction); if all are positive, it's a genuine minimum; if the eigenvalues have mixed signs, the point is a saddle — a maximum in some directions and a minimum in others — and cannot be reported as "the optimum" without further exploration along the specific directions (given by the eigenvectors) where the surface is still increasing.

---

## 25.6 The Direct Connection to Modern Machine Learning: Bayesian Optimization

**Response Surface Methodology is, in a very real sense, the direct classical ancestor of Bayesian Optimization**, now standard for expensive hyperparameter tuning and experimental design in ML:

| RSM Concept | Bayesian Optimization Equivalent |
|---|---|
| Second-order polynomial surrogate model | Gaussian Process (or other flexible) surrogate model |
| Steepest ascent direction | Acquisition function (e.g., Expected Improvement) guiding the next query point |
| Central Composite Design (fixed experimental plan) | Sequential, adaptively-chosen query points |
| Canonical analysis (stationary point, eigenvalues) | Optimizing the acquisition function / surrogate posterior mean |
| Lack-of-fit test (is quadratic enough?) | Surrogate model uncertainty / kernel choice |

**Why this connection matters, practically and for an interview.** Both frameworks solve the identical underlying problem: **finding the optimum of an expensive-to-evaluate function using as few actual evaluations as possible**, by fitting a cheap approximate model (surrogate) to guide where to look next. RSM's polynomial surrogate is simpler and less flexible than a Gaussian Process, but the core logic — fit an approximate model, use it to decide where to sample next, iterate — is exactly the same idea Bayesian Optimization applies to hyperparameter tuning, neural architecture search, and materials/drug discovery today.

**Interview question:** *"How does classical Response Surface Methodology relate to Bayesian Optimization used in modern ML hyperparameter tuning?"*
**Ideal answer:** Both are sequential strategies for optimizing an expensive-to-evaluate function using a cheap surrogate model to decide where to sample next. RSM classically uses a low-order polynomial (linear, then quadratic) surrogate fit via a structured design like a Central Composite Design, moving via steepest ascent and then canonical analysis of the fitted quadratic. Bayesian Optimization generalizes this with a much more flexible surrogate (typically a Gaussian Process) and an acquisition function (like Expected Improvement) that formally balances exploring uncertain regions against exploiting the currently-best-known region — but the fundamental strategy (approximate, sample strategically, refine, repeat) is a direct, continuous lineage from RSM's classical roots.

---

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize

# --- Central Composite Design data ---
X1 = np.array([1,1,-1,-1, 1.414,-1.414,0,0, 0,0,0,0,0])
X2 = np.array([1,-1,1,-1, 0,0,1.414,-1.414, 0,0,0,0,0])
Y  = np.array([50,48,44,34, 49.07,34.93,50.24,41.76, 49,50,51,50,50])

# --- Fit full second-order model via least squares ---
X_design = np.column_stack([
    np.ones(len(X1)), X1, X2, X1**2, X2**2, X1*X2
])
b = np.linalg.lstsq(X_design, Y, rcond=None)[0]
print("Coefficients [b0, b1, b2, b11, b22, b12]:", np.round(b, 3))

b0, b1, b2, b11, b22, b12 = b

# --- Canonical analysis: find stationary point ---
# Solve [2b11, b12; b12, 2b22] @ [X1*, X2*] = -[b1, b2]
A = np.array([[2*b11, b12],[b12, 2*b22]])
rhs = -np.array([b1, b2])
stationary_point = np.linalg.solve(A, rhs)
print("Stationary point:", stationary_point)

Y_at_stationary = (b0 + b1*stationary_point[0] + b2*stationary_point[1]
                    + b11*stationary_point[0]**2 + b22*stationary_point[1]**2
                    + b12*stationary_point[0]*stationary_point[1])
print("Predicted response at stationary point:", Y_at_stationary)

# --- Eigenvalues to classify the stationary point ---
B = np.array([[b11, b12/2],[b12/2, b22]])
eigenvalues = np.linalg.eigvals(B)
print("Eigenvalues:", eigenvalues)
print("Classification:", "Maximum" if all(eigenvalues < 0) else
                          "Minimum" if all(eigenvalues > 0) else "Saddle point")

# --- Pure error / lack of fit ---
center_Y = Y[8:]
SS_pure_error = np.sum((center_Y - center_Y.mean())**2)
print(f"Pure error SS: {SS_pure_error}, df={len(center_Y)-1}")
```

---

## Interview Question Bank — Chapter 25

**Conceptual:**
1. Why can't a first-order (linear) model ever have a genuine interior optimum?
2. What are the three components of a Central Composite Design, and what does each contribute to estimating the second-order model?
3. What does it mean for a stationary point to be a "saddle point," and why is this possible with two or more factors?

**Derivation:**
4. Derive the stationary point equations from the fitted second-order model via partial derivatives.
5. Explain why the eigenvalues of the curvature matrix determine whether a stationary point is a maximum, minimum, or saddle.

**ML/Statistics:**
6. Compare Response Surface Methodology to Bayesian Optimization — what's the same, and what's genuinely different (surrogate model flexibility, sampling strategy)?
7. Why does RSM use a sequential strategy (steepest ascent, then a local quadratic model) rather than fitting one large quadratic model over the entire possible factor range from the start?
8. How does the center-point-based lack-of-fit test in RSM connect to Chapter 3's original lack-of-fit F-test?

**Coding:**
9. Implement the full second-order model fit and canonical analysis (stationary point + eigenvalue classification) from scratch in NumPy.
10. Construct a Central Composite Design for 3 factors and verify the factorial, axial, and center point counts match the standard formula.

**Traps:**
11. "We found a stationary point in our fitted response surface, so we've found the optimal settings." — what must be checked before this conclusion is safe?
12. "A quadratic model always fits better than a linear one, so you should always use a full CCD from the very start of any optimization." — what's the practical cost this ignores?
13. Someone treats Bayesian Optimization as a completely novel ML technique unrelated to classical statistics. What's the more accurate historical/conceptual framing?

---

*This file covers Kutner Ch. 25 — the sequential RSM philosophy (steepest ascent, then second-order modeling), the Central Composite Design and its three components, fitting the full second-order model (with linear and interaction terms derived by hand, and quadratic terms via the normal equations), the lack-of-fit test using center-point pure error, canonical analysis via the stationary point and eigenvalue classification (worked completely by hand, confirming a genuine maximum), and the direct conceptual lineage from RSM to modern Bayesian Optimization.*

---

## The Complete Course: Final Summary

This closes out the full arc of Kutner, Nachtsheim, Neter, and Li's *Applied Linear Statistical Models*, covered chapter by chapter with hand-worked numerical examples, from-scratch Python/NumPy implementations, and interview-oriented explanations throughout:

**Part I–II — Regression (Chapters 1–14):** simple linear regression, inference, diagnostics, simultaneous inference, the matrix approach, multiple regression estimation and inference, extra sums of squares, qualitative predictors and interactions, model building and selection, diagnostics for multiple regression, remedial measures (WLS, ridge, robust regression, bootstrapping), autocorrelation, nonlinear regression and neural networks, and the Generalized Linear Model framework unifying linear, logistic, and Poisson regression.

**Part III–V — Design of Experiments (Chapters 15–25, plus ANCOVA):** the causal/observational distinction and randomization, single-factor ANOVA and its identity with indicator-variable regression, multiple comparison procedures (Tukey, Scheffé, Bonferroni), two-factor and multi-factor ANOVA with interactions, ANCOVA and its direct connection to modern CUPED-style variance reduction, Randomized Complete Block Designs and the relative efficiency of blocking, nested designs and the pseudo-replication trap, repeated measures designs and sphericity, two-level factorial and fractional factorial screening designs, and Response Surface Methodology culminating in its direct lineage to modern Bayesian Optimization.

Every chapter file stands as a self-contained reference in your outputs folder — together, they form a comprehensive foundation spanning the statistical theory and applied machinery expected at the senior Data Scientist / ML Engineer level, from first-principles derivation through the exact algorithms (IRLS, MLE, bias-variance tradeoffs, hierarchical modeling) underlying modern ML tooling and industry experimentation practice.
