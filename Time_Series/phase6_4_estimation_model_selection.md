# Phase 6, Part 4 of 5: Estimation (Yule-Walker, MLE) & Model Selection (AIC/BIC)

Roadmap: 6.1 AR(p) [done] → 6.2 MA(q) [done] → 6.3 ARMA/ARIMA/SARIMA [done] → **6.4 Estimation & model selection (this file)** → 6.5 Diagnostics & forecasting.

Everything so far has been "here's the SHAPE of the model, and here's what its ACF/PACF looks like." Now: **given real numbers, how do we actually find the specific $\phi$ and $\theta$ values that fit best, and how do we choose $p$, $d$, $q$ in the first place?** Same commitment as last time — every symbol explained plainly, the moment it appears.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $\hat{\phi}, \hat{\theta}$ | the ESTIMATED (from data) versions of $\phi,\theta$ — the hat means "our best guess," not the true unknown value |
| $n$ | number of data points we have |
| $L(\cdot)$ | the **likelihood function** — a formula that scores "how probable is this data, GIVEN a candidate set of parameters" |
| $\ln L$ | the **log-likelihood** — the same score, but on a log scale (explained in section 4) |
| $k$ | the number of parameters a model uses (counting $\phi$'s, $\theta$'s, and a couple of extras — explained in section 6) |
| AIC, BIC | two competing "scorecards" for comparing different candidate models |
| $\text{argmax}$ | "the value of the input that makes this expression as LARGE as possible" |
| $\text{argmin}$ | "the value of the input that makes this expression as SMALL as possible" |

---

## 2. The simplest estimation method: Yule-Walker equations (for AR models)

**Where this comes from — you already derived the core idea in Phase 6, Part 1, section 4!** Recall we derived, for AR(1): $\rho(k) = \phi\,\rho(k-1)$. **The Yule-Walker method's entire idea is simply: run that relationship BACKWARDS.** Instead of "if I know $\phi$, I can compute the ACF," we flip it around: "if I can MEASURE the sample ACF from real data (Phase 3, section 4 — you already know how to compute $\hat\rho(k)$ by hand), I can SOLVE for $\phi$."

**For AR(1), this is almost embarrassingly simple.** We derived $\rho(1) = \phi$ directly (since $\rho(k)=\phi^k$, so $\rho(1)=\phi^1=\phi$). So the Yule-Walker ESTIMATE is just:
$$
\hat\phi = \hat\rho(1)
$$
**Plain English: for AR(1), your best estimate of $\phi$ is LITERALLY just the sample lag-1 autocorrelation you already know how to compute from Phase 3.** No new machinery needed at all for this simplest case.

**For AR(2), it takes two equations instead of one.** Using the same "multiply by $x_{t-k}$ and take expectations" trick from Part 1 section 4, but now with TWO $\phi$ terms in the model ($x_t = \phi_1 x_{t-1}+\phi_2 x_{t-2}+\varepsilon_t$), you get a small SYSTEM of two equations:
$$
\rho(1) = \phi_1 + \phi_2\,\rho(1)
$$
$$
\rho(2) = \phi_1\,\rho(1) + \phi_2
$$
**Plain English reading:** each equation just says "the correlation at some lag equals a weighted combination of the two $\phi$'s, using the ACF's OWN values as the weights." This might look intimidating, but it's just TWO linear equations with TWO unknowns ($\phi_1,\phi_2$) — exactly the kind of system you'd solve in basic algebra (substitution or matrix methods). Plug in the sample values $\hat\rho(1)$ and $\hat\rho(2)$ (computed from real data, Phase 3 style), solve the two equations, and out come $\hat\phi_1,\hat\phi_2$.

**General AR(p): same idea, just $p$ equations instead of 2.** These are called the **Yule-Walker equations**, and in matrix form they're solved using standard linear algebra (matrix inversion) — software does this instantly; you just need to recognize the NAME and the CORE IDEA ("turn the known ACF-formula relationship around, and solve for the coefficients that would have produced the ACF you actually measured").

**The catch — why Yule-Walker ISN'T used for MA or ARMA models:** recall from Part 2, MA models don't have a simple closed-form relationship this clean between $\rho(k)$ and $\theta$ for higher lags (well, they do, but it involves solving a genuinely messier, NON-linear system — MA's $\rho(1) = \theta/(1+\theta^2)$ formula from Part 2 has $\theta$ appearing in a more tangled way, not as a simple linear multiplier). **This is exactly why we need a completely different, more powerful, more general estimation method for anything involving an MA piece** — which brings us to Maximum Likelihood.

---

## 3. Building intuition for "likelihood" from scratch (before any formula)

**Plain English, zero formulas first:** Imagine you have a coin, and you don't know if it's fair. You flip it 10 times and get 7 heads, 3 tails. Now ask: "if the TRUE probability of heads were 0.5, how surprising/likely would THIS SPECIFIC outcome (7 heads out of 10) have been? What if the true probability were 0.7 instead? Or 0.9?" **The likelihood function is exactly this idea, generalized: for each CANDIDATE value of an unknown parameter, compute how probable the data you ACTUALLY OBSERVED would have been, if that candidate value were the true one.** Then, **Maximum Likelihood Estimation (MLE)** simply says: **pick whichever candidate parameter value makes your ACTUALLY OBSERVED data look the MOST probable/least surprising.** In the coin example, MLE would pick $\hat{p}=0.7$ — because a true coin-bias of exactly 0.7 makes "7 heads out of 10" the single most probable outcome, more probable than any other candidate bias value would.

**Now transplant this exact idea to ARMA models:** we have observed data $x_1,\ldots,x_n$. We have candidate parameter values ($\phi$'s, $\theta$'s, and $\sigma^2$, the noise variance). We ask: "for THIS candidate set of parameters, how probable was the data sequence we actually saw?" We then search over ALL possible candidate parameter values and pick the ones that make our real, observed data as probable as possible.

---

## 4. The likelihood function for ARMA — built up piece by piece

**Step 1 — recall the noise is Gaussian (Phase 2):** we typically assume $\varepsilon_t \sim N(0,\sigma^2)$ (Normal/bell-curve distributed, mean 0, variance $\sigma^2$ — this notation "$\sim N(\cdot,\cdot)$" was introduced back in Phase 2, section 4). The Normal distribution has a known formula for "how probable is this specific value" — its **probability density function (a new term: this just means 'the formula that tells you the relative likelihood of any specific numeric outcome for a continuous random variable')**:
$$
f(\varepsilon_t) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{\varepsilon_t^2}{2\sigma^2}\right)
$$
**Plain English, no need to memorize this exact formula, just understand its SHAPE:** values of $\varepsilon_t$ close to 0 get a HIGH score (probable, since noise is usually small); values far from 0 get a rapidly SHRINKING score (improbable, since large shocks should be rare) — exactly matching how a bell curve looks: tall in the middle, thin at the edges.

**Step 2 — chain the individual probabilities together across the WHOLE dataset.** Because each $\varepsilon_t$ is independent of the others (white noise, Phase 2), the probability of the WHOLE sequence of shocks happening together is just the PRODUCT of each individual probability (this is the basic probability rule: for independent events, multiply their individual probabilities to get the joint probability of all of them happening together):
$$
L(\phi,\theta,\sigma^2) = \prod_{t=1}^n f(\varepsilon_t)
$$
The symbol $\prod$ (capital Greek pi) just means "multiply all of these together," the multiplication equivalent of the $\sum$ (sum) symbol you already know from Phase 3. **Crucial connecting point: the $\varepsilon_t$ values inside this formula aren't directly observed — they're BACK-CALCULATED from the candidate $\phi,\theta$ values and the actual observed $x_t$ data, by rearranging the ARMA formula from Part 3 to solve for $\varepsilon_t$ at each time step.** So different candidate $(\phi,\theta)$ values produce DIFFERENT implied $\varepsilon_t$ sequences, and therefore different likelihood scores.

**Step 3 — why we take the LOG of this (the "log-likelihood"), a genuinely practical trick, not just decoration:** multiplying together hundreds or thousands of small probability numbers (each less than 1) produces an ASTRONOMICALLY tiny final number that computers struggle to represent accurately (a real numerical problem, not just theoretical). Taking the logarithm turns that giant PRODUCT into a much more manageable SUM instead (a basic log rule: $\log(a\times b) = \log(a)+\log(b)$), and since the logarithm function is always increasing (bigger inputs always give bigger log-outputs), **whichever parameter values maximize the log-likelihood are GUARANTEED to be the exact same parameter values that would have maximized the original likelihood** — so nothing is lost, and everything becomes numerically and algebraically easier:
$$
\ln L = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^n \varepsilon_t^2
$$
**You do not need to memorize or re-derive this exact expanded formula — the important takeaway is structural:** notice the term $\sum_{t=1}^n \varepsilon_t^2$ — **this is literally just the sum of squared residuals/errors, the exact same quantity you'd minimize in ordinary least-squares regression!** This reveals a genuinely important, practical connection: **maximizing this log-likelihood (for a fixed $\sigma^2$) is mathematically EQUIVALENT to MINIMIZING the sum of squared errors** — meaning, in practice, fitting an ARMA model by Maximum Likelihood ends up being very closely related to (and in some simplified cases, IDENTICAL to) simple least-squares curve fitting, a technique you likely already have intuition for from ordinary regression. This connection is a common, genuinely useful interview talking point: "MLE for Gaussian-noise models reduces to least-squares."

**Step 4 — how do we actually find the maximizing values in practice?** For AR-only models, section 2's Yule-Walker approach gives a fast, direct (closed-form, meaning "solvable with plain algebra, no trial-and-error needed") answer. Once ANY MA component is involved, there's generally no clean closed-form algebra solution — instead, software uses **numerical optimization**: starting from some reasonable initial guess for $(\phi,\theta,\sigma^2)$, it repeatedly nudges the values in the direction that increases the log-likelihood (conceptually similar to walking uphill on a landscape shaped by the log-likelihood function, taking small steps toward the peak) until it can't improve further — this converged point is the MLE estimate. You don't need to hand-implement this optimization algorithm — just understand WHAT it's searching for (the peak of the log-likelihood "landscape") and WHY it's needed (no simple algebra shortcut exists once MA terms are involved).

---

## 5. A very small, honest numerical taste of MLE (kept deliberately simple)

Full MLE optimization genuinely requires calculus/numerical search that isn't practical by hand for real ARMA models — but here's a simplified, concrete taste of the CORE IDEA using just 2 data points and ONE candidate AR(1) parameter comparison, to make section 3's coin-flip intuition fully concrete in a time-series setting.

Suppose we observe just two points, $x_1=2, x_2=5$ (pretend $x_0=0$ known), and we're comparing two CANDIDATE values: $\phi_A = 0.5$ vs $\phi_B=0.9$, with $\sigma^2=1$ assumed known for simplicity.

**Back-calculate the implied noise for each candidate**, using $\varepsilon_t = x_t - \phi\,x_{t-1}$ (rearranging the AR(1) formula from Part 1):

For $\phi_A=0.5$: $\varepsilon_1 = 2-0.5(0)=2$; $\varepsilon_2=5-0.5(2)=5-1=4$
For $\phi_B=0.9$: $\varepsilon_1=2-0.9(0)=2$; $\varepsilon_2=5-0.9(2)=5-1.8=3.2$

**Compare the sum of squared implied noise** (recall from step 3 above: bigger squared errors = lower likelihood, since large $\varepsilon_t$ values are "surprising"/improbable under our bell-curve noise assumption):

$\phi_A$: $2^2+4^2 = 4+16=20$
$\phi_B$: $2^2+3.2^2=4+10.24=14.24$

**Conclusion: $\phi_B=0.9$ produces SMALLER implied noise/errors, meaning the observed data $[2,5]$ is MORE probable/less surprising under $\phi_B$ than under $\phi_A$ — so MLE would prefer $\phi_B$ over $\phi_A$, exactly matching the "minimize sum of squared errors" shortcut we identified in step 3.** A real MLE search would continue trying MANY more candidate $\phi$ values (not just these two), using calculus/numerical optimization to efficiently zero in on the single best one, rather than manually comparing a small handful like we just did — but the comparison logic itself (smaller squared implied errors = more likely = preferred) is EXACTLY the same idea scaled up.

---

## 6. Model selection: choosing $p$, $d$, $q$ using AIC and BIC

**The problem this solves:** Yule-Walker and MLE (sections 2 and 3-4) tell you HOW to estimate the BEST $\phi,\theta$ values ONCE you've already picked a specific $p$ and $q$. But how do you choose $p$ and $q$ THEMSELVES in the first place? You could use the ACF/PACF signature table (Phase 3, and derived in Parts 1-2) as a starting GUESS — but often several candidate $(p,q)$ combinations look plausible, and you need a formal, numeric way to compare them.

**The naive (and WRONG) approach: just pick whichever model has the best log-likelihood.** Here's the problem: **adding MORE parameters to ANY model will ALWAYS make the log-likelihood look at least as good, often better — even if those extra parameters are just fitting random noise in your specific sample, not real underlying structure.** This is the exact same **overfitting** problem you may have encountered in general machine learning: a model with enough free parameters can always bend itself to fit the training data more closely, without that extra flexibility representing anything real or generalizable to new data.

**The fix — penalize complexity.** Both AIC and BIC take the log-likelihood and SUBTRACT a penalty that grows with the number of parameters $k$ used (for ARMA(p,q), $k$ counts $p$ (AR coefficients) $+$ $q$ (MA coefficients) $+ 1$ (the noise variance $\sigma^2$, which also has to be estimated) $+$ possibly $1$ more for a constant term $c$ if included).

**AIC (Akaike Information Criterion):**
$$
\text{AIC} = -2\ln L + 2k
$$
**Plain English:** take the (negative, doubled) log-likelihood score, then ADD a penalty of exactly "2 points per parameter used." **Lower AIC is better** (since we're working with NEGATIVE log-likelihood here — a technical sign-flip convention purely so that "smaller number = better model" consistently, matching the everyday intuition of minimizing a score, similar to minimizing an error). A model with a slightly better fit but 3 more parameters than another model needs to improve its raw log-likelihood by MORE than the extra $2\times 3=6$-point penalty to still "win" on AIC — this forces genuinely worthwhile complexity, not complexity for its own sake.

**BIC (Bayesian Information Criterion):**
$$
\text{BIC} = -2\ln L + k\ln(n)
$$
**Plain English:** almost identical structure to AIC, EXCEPT the penalty per parameter is $\ln(n)$ (the natural log of your sample size) instead of a flat "2." **Key practical consequence: for any reasonably large dataset, $\ln(n) > 2$** (e.g., for $n=100$, $\ln(100)\approx 4.6$, already more than double AIC's flat penalty of 2) — **meaning BIC penalizes extra parameters MORE HARSHLY than AIC does, and this penalty grows even stronger as you get more data.** Practical consequence: **BIC tends to prefer smaller, simpler models than AIC does, especially with large datasets.**

**Which one should you actually use? (a genuine, common interview question, with a real, defensible answer):** AIC is theoretically aimed at finding the model that will forecast/PREDICT best on NEW, unseen future data (even if the "true" real-world process is more complex than any finite model could capture — AIC doesn't assume the true model is even in your candidate list). BIC is theoretically aimed at finding the actual TRUE underlying model, assuming the true model genuinely IS one of your candidates (a stronger, sometimes unrealistic assumption) — and BIC is proven to correctly identify that true model as your data grows toward infinity ("consistency," a real statistical property), while AIC does not have this same guarantee. **Practical, defensible summary for an interview: "if forecasting accuracy is my main goal, I lean AIC; if I care more about correctly identifying a parsimonious, truly correct underlying structure, I lean BIC — and in practice, I usually compute both, and if they agree, I'm confident; if they disagree, I typically favor the simpler model BIC selected, since overfit models tend to forecast poorly out-of-sample despite fitting the training data well."**

**The actual practical workflow (this is what software like `auto.arima()` in R, or `pmdarima` in Python, automates for you):** try a reasonable RANGE of candidate $(p,d,q)$ combinations (guided by your ACF/PACF reading as a sensible starting range, not brute-force testing every possible number), fit each one via MLE (section 4), compute AIC (and/or BIC) for each, and pick the combination with the lowest score. This is a genuinely standard, real production workflow — not just a textbook exercise.

---

## 7. Quick self-check questions

1. For an AR(1) model, what IS the Yule-Walker estimate of $\phi$, in terms of something you already know how to compute from Phase 3?
   *(Answer: $\hat\phi = \hat\rho(1)$ — literally just the sample lag-1 autocorrelation.)*
2. In plain English, what question does the likelihood function answer?
   *(Answer: "for this specific candidate set of parameter values, how probable would the data I actually observed have been?" — MLE then picks whichever candidate parameters make the observed data look most probable.)*
3. Why do we take the LOG of the likelihood before maximizing, rather than maximizing the raw likelihood directly?
   *(Answer: multiplying many small probabilities together produces a number too tiny for computers to handle accurately; the log turns that product into a sum, which is numerically stable and algebraically easier, and since log is an increasing function, the same parameter values maximize both the log-likelihood and the original likelihood.)*
4. If two candidate models have nearly identical log-likelihood, but Model A has 3 parameters and Model B has 6 parameters, which will AIC and BIC tend to prefer, and why?
   *(Answer: both will prefer Model A (fewer parameters), because with essentially tied fit quality, the extra parameter penalty (2k for AIC, k·ln(n) for BIC) makes the more complex Model B score worse without a large enough likelihood improvement to offset it — this is the mechanism that discourages overfitting.)*

---

## What's next
**Part 5 of Phase 6** (the final part) covers **residual diagnostics** (using the Ljung-Box test from Phase 3, section 8, to formally check "do the leftover residuals look like white noise?"), and **forecasting** — deriving actual point forecasts AND prediction intervals from a fitted ARIMA model, completing the full Box-Jenkins cycle end-to-end.

Say "next" for Part 5, or ask me to slow down more on the likelihood/MLE material first (it's genuinely the most abstract material in this phase, and worth re-explaining differently if anything felt shaky).
