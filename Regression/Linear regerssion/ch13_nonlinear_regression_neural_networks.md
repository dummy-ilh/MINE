# Chapter 13 — Introduction to Nonlinear Regression and Neural Networks
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Every model so far has been **linear in the parameters** — even polynomial and interaction models (Chapter 8) satisfy this, since $\beta$'s always enter as simple multiplicative coefficients. This chapter covers genuinely **nonlinear** models, where the mean function's *shape* with respect to the parameters themselves is curved, and closed-form least squares (Chapter 5's $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$) no longer applies at all.

---

## 13.1 Linear vs. Nonlinear Regression Models

### The Precise Distinction (Not About Curved Data — About Curved Parameters)

**Plain English, and a very common point of confusion to clear up first.** "Nonlinear regression" does **not** mean "the relationship between X and Y is curved" — polynomial regression ($Y=\beta_0+\beta_1X+\beta_2X^2$) has a curved relationship with X but is still a **linear model**, because it's linear in $\beta_0,\beta_1,\beta_2$. **"Nonlinear" refers to the model being nonlinear in its *parameters*.**

**Formal test:** take the partial derivative of the mean function with respect to each parameter. If none of those derivatives themselves contain any of the parameters, the model is linear (in parameters). Consider:
$$
f(X;\gamma) = \gamma_0 e^{\gamma_1 X}
$$
$$
\frac{\partial f}{\partial \gamma_0} = e^{\gamma_1 X} \quad(\text{still contains } \gamma_1!), \qquad \frac{\partial f}{\partial \gamma_1}=\gamma_0Xe^{\gamma_1X}\quad(\text{contains both } \gamma_0,\gamma_1)
$$
Since these derivatives depend on the parameters themselves, this is a genuinely **nonlinear** model — a fundamentally different estimation problem from anything in Chapters 1–12.

### Intrinsically Linear vs. Intrinsically Nonlinear Models

Some nonlinear-looking models can be transformed into linear ones. E.g., $Y=\gamma_0e^{\gamma_1X}\cdot\varepsilon^*$ (multiplicative error) becomes, after taking logs: $\ln Y = \ln\gamma_0+\gamma_1X+\ln\varepsilon^*$ — **linear** in $\ln\gamma_0$ and $\gamma_1$. Kutner calls this **intrinsically linear**: nonlinear in its original form, but linearizable via a transformation, letting you fall back on everything from Chapters 1–12.

**But many practically important models cannot be linearized by any transformation** — e.g., the logistic growth curve $Y=\dfrac{\gamma_0}{1+\gamma_1e^{\gamma_2X}}$ has no transformation that makes it linear in $\gamma_0,\gamma_1,\gamma_2$ simultaneously. These are **intrinsically nonlinear**, and require the iterative estimation approach developed in the rest of this chapter.

**Interview trap:** people sometimes reach immediately for "just log-transform it" whenever they see an exponential or power-law-looking relationship. That works for *additive-error-after-transformation* cases, but changes what error structure you're assuming (multiplicative vs. additive noise) — worth checking explicitly whether the transformed model's error assumptions (constant variance of $\ln\varepsilon^*$, say) are actually more reasonable than the original's, not just algebraically convenient.

---

## 13.2 Least Squares Estimation in Nonlinear Regression: the Gauss-Newton Method

### Why Chapter 1's Calculus Trick No Longer Works

**Plain English.** In linear regression, setting $\partial Q/\partial b=0$ gave a *linear* system of equations (the normal equations), solvable in one closed-form step. For a nonlinear mean function, setting the analogous derivative to zero produces equations where the unknown parameters appear **inside nonlinear functions** (exponentials, ratios, etc.) — there's generally no algebraic way to isolate them. **We need an iterative numerical method instead.**

### The Gauss-Newton Idea

**Core idea, in plain English:** at your current best guess for the parameters, **locally approximate** the nonlinear function with a linear one (its first-order Taylor expansion), solve the resulting *linear* least-squares problem for an *update* to the parameters, take a step in that direction, and repeat until the updates become negligible.

**The math, one step at a time.** Let $f(X_i;\boldsymbol\gamma)$ denote the nonlinear mean function, and let $\boldsymbol\gamma^{(0)}$ be your current guess. The first-order Taylor expansion around $\boldsymbol\gamma^{(0)}$:
$$
f(X_i;\boldsymbol\gamma) \approx f(X_i;\boldsymbol\gamma^{(0)}) + \sum_k \frac{\partial f}{\partial\gamma_k}\bigg|_{\boldsymbol\gamma^{(0)}}(\gamma_k-\gamma_k^{(0)})
$$

Define the **residual at the current guess** $r_i = Y_i - f(X_i;\boldsymbol\gamma^{(0)})$ and the **Jacobian matrix** $\mathbf{J}$, whose $(i,k)$ entry is $\partial f(X_i;\boldsymbol\gamma)/\partial\gamma_k$ evaluated at $\boldsymbol\gamma^{(0)}$. Substituting the Taylor approximation into the sum-of-squares objective and minimizing over the *update* $\boldsymbol\delta=\boldsymbol\gamma-\boldsymbol\gamma^{(0)}$ gives — remarkably — **exactly Chapter 5's linear normal equations, but for the update $\boldsymbol\delta$ instead of the parameters themselves:**
$$
\boldsymbol\delta = (\mathbf{J}'\mathbf{J})^{-1}\mathbf{J}'\mathbf{r}
$$

**Why this is elegant and worth appreciating explicitly:** every single tool built in Chapters 5–7 (the normal equations, the role of $(\mathbf{X}'\mathbf{X})^{-1}$, even the variance-covariance-matrix logic) reappears here almost unchanged — just applied *locally*, to the Jacobian instead of the design matrix, and applied *repeatedly* rather than once. Update: $\boldsymbol\gamma^{(1)}=\boldsymbol\gamma^{(0)}+\boldsymbol\delta$, recompute the Jacobian and residuals at the new point, and repeat until $\boldsymbol\delta$ is negligibly small (convergence).

### Worked Example: One Gauss-Newton Iteration by Hand

**Model:** $f(X;\gamma_0,\gamma_1)=\gamma_0e^{\gamma_1X}$. **True generating values** (unknown to the algorithm): $\gamma_0=2,\ \gamma_1=0.5$.

**Data** ($n=4$):

| X | Y |
|---|---|
| 0 | 2.1 |
| 1 | 3.2 |
| 2 | 5.5 |
| 3 | 9.0 |

**Starting guess (deliberately poor):** $\gamma_0^{(0)}=1,\ \gamma_1^{(0)}=1$.

**Step 1 — evaluate $f$, the residuals, and the Jacobian at the starting guess.** Recall $\partial f/\partial\gamma_0=e^{\gamma_1X}$ and $\partial f/\partial\gamma_1=\gamma_0Xe^{\gamma_1X}$:

| X | $f(X;1,1)$ | $r_i=Y_i-f_i$ | $\partial f/\partial\gamma_0$ | $\partial f/\partial\gamma_1$ |
|---|---|---|---|---|
| 0 | 1.000 | 1.100 | 1.000 | 0.000 |
| 1 | 2.718 | 0.482 | 2.718 | 2.718 |
| 2 | 7.389 | -1.889 | 7.389 | 14.778 |
| 3 | 20.086 | -11.086 | 20.086 | 60.257 |

**Step 2 — form $\mathbf{J}'\mathbf{J}$ and $\mathbf{J}'\mathbf{r}$** (identical mechanics to building $\mathbf{X}'\mathbf{X}$ and $\mathbf{X}'\mathbf{Y}$ in Chapter 5, just using the Jacobian columns instead of raw X columns):

$$
\mathbf{J}'\mathbf{J} = \begin{bmatrix}466.41 & 1326.87\\1326.87&3856.64\end{bmatrix}, \qquad \mathbf{J}'\mathbf{r}=\begin{bmatrix}-234.21\\-694.58\end{bmatrix}
$$

**Step 3 — solve for the update $\boldsymbol\delta$** (same 2×2 solve method as Chapters 6–10):
$$
\det(\mathbf{J}'\mathbf{J}) = 466.41(3856.64)-1326.87^2 \approx 38{,}209
$$
$$
\delta_0 = \frac{(-234.21)(3856.64)-1326.87(-694.58)}{38{,}209}\approx 0.480
$$
$$
\delta_1 = \frac{466.41(-694.58)-(-234.21)(1326.87)}{38{,}209}\approx -0.346
$$

**Step 4 — update the parameters:**
$$
\gamma_0^{(1)}=1+0.480=1.480, \qquad \gamma_1^{(1)}=1-0.346=0.654
$$

**In a single iteration, starting from a poor guess $(1, 1)$, Gauss-Newton has already moved to $(1.48, 0.65)$** — visibly closer to the true $(2, 0.5)$. A few more iterations (each repeating exactly this process — re-evaluate $f$, the residuals, and the Jacobian at the new point, solve again for $\boldsymbol\delta$) would continue converging toward the true values, with the update size shrinking toward zero as the algorithm settles into the least-squares solution.

**Interview question:** *"How does the Gauss-Newton method differ from a generic gradient descent step, even though both are iterative?"*
**Ideal answer:** Gradient descent takes a step directly opposite the gradient of the loss, scaled by a learning rate — a first-order method using no curvature information. Gauss-Newton instead locally linearizes the *model* itself (not just the loss) via its Jacobian, and solves the resulting linear least-squares problem exactly at each step — effectively using an approximation to the loss function's curvature (the Hessian is approximated as $\mathbf{J}'\mathbf{J}$, dropping a term involving the residuals and second derivatives of $f$) rather than just its slope. This typically gives faster, more direct convergence for well-behaved nonlinear least-squares problems, at the cost of needing to form and invert a (small) matrix at every step — which is exactly why it doesn't scale to models with millions of parameters, unlike plain gradient-based methods.

**Practical caveats Kutner flags:** Gauss-Newton can diverge or converge very slowly if the starting guess is poor or the model is highly nonlinear near the solution; **Levenberg-Marquardt** (a very standard, widely-used refinement) blends Gauss-Newton with a gradient-descent-like damping term, taking smaller, more conservative steps when the pure Gauss-Newton step would be unreliable — this is the actual default algorithm behind most off-the-shelf nonlinear least squares solvers (e.g., `scipy.optimize.curve_fit`'s default method).

---

## 13.3 Model Building and Diagnostics for Nonlinear Regression (Brief)

**What carries over from linear regression:** residual plots (Chapter 3) are still your primary diagnostic tool for checking constant variance, normality, and functional form adequacy — nothing about nonlinearity in the parameters changes what a "good" residual plot should look like.

**What changes, and needs care:** the elegant, *exact* small-sample inference machinery from Chapter 2 (exact t-distributions, F-distributions) **no longer applies exactly**. Standard errors and confidence intervals for nonlinear parameter estimates are typically based on the **asymptotic** (large-sample) normal approximation of the maximum likelihood estimator, using the *final* Jacobian (evaluated at the converged solution) in place of $\mathbf{X}$ in the familiar formula:
$$
s^2(\hat{\boldsymbol\gamma}) \approx MSE\cdot(\mathbf{J}'\mathbf{J})^{-1}
$$
**Why this is only an approximation, not exact:** this formula relies on the local linear approximation being a good stand-in for the true nonlinear curvature near the solution — reliable with reasonably large samples and mild nonlinearity, but potentially misleading with small samples or highly curved mean functions, where the true sampling distribution of $\hat{\boldsymbol\gamma}$ can be noticeably non-normal and skewed even asymptotically-motivated formulas would suggest otherwise. **Bootstrapping (Chapter 11.5) is a commonly recommended alternative** for nonlinear-regression confidence intervals precisely because it doesn't rely on this local-linearization approximation being accurate.

---

## 13.4 Introduction to Neural Network Modeling — Kutner's Framing

### Neural Networks as a (Very) Flexible Nonlinear Regression Model

Kutner introduces neural networks explicitly as a **generalization of everything in this chapter**, not a separate topic. A single-hidden-layer network for a scalar output is, structurally, just another nonlinear mean function:
$$
f(\mathbf{X};\boldsymbol\theta) = \beta_0+\sum_{h=1}^H\beta_h\cdot\sigma\left(w_{h0}+\sum_j w_{hj}X_j\right)
$$
where $\sigma(\cdot)$ is an activation function (e.g., sigmoid), and $\boldsymbol\theta$ collects every weight and bias — exactly the same "nonlinear-in-parameters mean function $f(X;\boldsymbol\gamma)$" object this entire chapter has been about, just with (potentially) thousands or millions of parameters $\boldsymbol\gamma$ instead of two.

**What's genuinely different at neural-network scale, connecting directly to Gauss-Newton above:**
- **Scale makes $(\mathbf{J}'\mathbf{J})^{-1}$ computationally infeasible.** Forming and inverting a matrix with dimensions in the millions, every iteration, simply isn't tractable — this is precisely why deep learning abandoned second-order/Gauss-Newton-style methods in favor of **stochastic gradient descent** and its refinements (momentum, Adam, etc.) — first-order methods needing only the gradient, never a matrix inversion.
- **Backpropagation is just an efficient algorithm for computing that gradient** (equivalently, one column of the Jacobian at a time) via the chain rule through a composed function — mechanically the same derivative-taking this chapter's Jacobian required, just organized for computational efficiency at massive scale.
- **The loss function generalizes beyond squared error** (cross-entropy for classification, etc.), but the core estimation philosophy — iteratively improving parameter guesses using local derivative information to reduce a loss — is identical in spirit to Gauss-Newton's iterative refinement.
- Some advanced optimization methods (K-FAC, natural gradient methods) explicitly attempt to reintroduce curvature information resembling $\mathbf{J}'\mathbf{J}$ in an approximated, tractable form — a direct, ongoing echo of exactly the Gauss-Newton idea, adapted for scale.

**Interview question:** *"In what precise sense is a neural network 'just' a nonlinear regression model, and where does that analogy break down?"*
**Ideal answer:** Structurally, a neural network is exactly a nonlinear-in-parameters mean function being fit by minimizing a loss via iterative, gradient-based updates — the same estimation philosophy as Gauss-Newton for classical nonlinear regression, and backpropagation is mechanically just efficient computation of that gradient via the chain rule. The analogy starts to break down at scale and in practice: the sheer number of parameters makes second-order methods (which classical nonlinear regression relies on for both fitting and inference) computationally infeasible, so training relies on first-order stochastic methods instead; and classical nonlinear regression's asymptotic inference machinery (confidence intervals via $(\mathbf{J}'\mathbf{J})^{-1}$) isn't practically used for neural networks at all — uncertainty quantification for deep models instead relies on entirely different tools (ensembling, dropout-as-approximate-Bayesian-inference, conformal prediction), since classical asymptotic normality assumptions are far less trustworthy at that scale and with that much non-convexity in the loss surface.

---

## Python Implementation — Nonlinear Least Squares

```python
import numpy as np
from scipy.optimize import curve_fit

X = np.array([0, 1, 2, 3], dtype=float)
Y = np.array([2.1, 3.2, 5.5, 9.0], dtype=float)

def model(X, gamma0, gamma1):
    return gamma0 * np.exp(gamma1 * X)

# scipy's curve_fit uses Levenberg-Marquardt (a damped Gauss-Newton) by default
popt, pcov = curve_fit(model, X, Y, p0=[1, 1])
print("Fitted gamma0, gamma1:", popt)
print("Approximate covariance matrix (MSE * (J'J)^-1 equivalent):\n", pcov)
```

```python
# Manual Gauss-Newton implementation, showing the iteration explicitly
def gauss_newton(X, Y, gamma0_init, gamma1_init, n_iter=10):
    g0, g1 = gamma0_init, gamma1_init
    for it in range(n_iter):
        f = g0 * np.exp(g1 * X)
        r = Y - f
        J = np.column_stack([np.exp(g1*X), g0*X*np.exp(g1*X)])
        delta = np.linalg.inv(J.T @ J) @ J.T @ r
        g0, g1 = g0 + delta[0], g1 + delta[1]
        print(f"Iter {it+1}: gamma0={g0:.4f}, gamma1={g1:.4f}, SSE={np.sum(r**2):.4f}")
    return g0, g1

gauss_newton(X, Y, 1.0, 1.0)
```

---

## Interview Question Bank — Chapter 13

**Conceptual:**
1. What precisely makes a model "nonlinear," as opposed to just having a curved relationship with X?
2. What's the difference between an intrinsically linear and an intrinsically nonlinear model?
3. Why do confidence intervals for nonlinear regression parameters rely on an asymptotic approximation rather than exact small-sample distributions?

**Derivation:**
4. Derive the Gauss-Newton update equation $\boldsymbol\delta=(\mathbf{J}'\mathbf{J})^{-1}\mathbf{J}'\mathbf{r}$ from the first-order Taylor expansion of the nonlinear mean function.
5. Explain, precisely, what part of the true Hessian Gauss-Newton's $\mathbf{J}'\mathbf{J}$ approximation ignores, and why that approximation is often reasonable near a good fit.

**ML/Statistics:**
6. Why doesn't Gauss-Newton scale to training modern neural networks, and what replaced it?
7. In what specific technical sense is backpropagation related to the Jacobian computation used in Gauss-Newton?
8. Name a modern optimization technique that tries to reintroduce curvature information into large-scale training, and explain how it relates to the classical Gauss-Newton idea.

**Coding:**
9. Implement Gauss-Newton from scratch in NumPy for a two-parameter nonlinear model, and verify it matches `scipy.optimize.curve_fit`.
10. Fit an intrinsically linear model (e.g., $Y=\gamma_0e^{\gamma_1X}$ with multiplicative error) both via log-transformation-then-OLS and via direct nonlinear least squares, and compare the resulting parameter estimates.

**Traps:**
11. "Since the relationship between X and Y is exponential, I should always log-transform and use ordinary least squares." — when is this actually appropriate, and when might it not be?
12. "Gauss-Newton is just gradient descent with a different name." — what's the precise distinction?
13. Someone reports a "confidence interval" for a neural network's weight using the classical $MSE\cdot(\mathbf{J}'\mathbf{J})^{-1}$ formula. What practical issues make this approach unreliable at that scale?

---

*This file covers Kutner Ch. 13 — the distinction between linear and nonlinear (and intrinsically linear vs. intrinsically nonlinear) models, the Gauss-Newton method derived and worked by hand through one full iteration, the necessarily-approximate nature of nonlinear regression inference, and Kutner's explicit framing of neural networks as large-scale nonlinear regression — connecting Gauss-Newton directly to backpropagation and modern optimization. Chapter 14 (Logistic Regression, Poisson Regression, and Generalized Linear Models) is next, and is likely the single highest-value remaining chapter for ML interviews — it's the direct bridge from everything built so far to classification and count-data modeling.*
