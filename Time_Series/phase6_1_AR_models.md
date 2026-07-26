# Phase 6, Part 1 of 5: Autoregressive AR(p) Models

Phase 6 is the largest phase in this course, so we split it into five well-paced parts:
**6.1 AR(p) models (this file) → 6.2 MA(q) models → 6.3 ARMA/ARIMA/SARIMA → 6.4 Estimation (Yule-Walker/MLE) & model selection (AIC/BIC) → 6.5 Diagnostics & forecasting.**

Everything here builds directly on Phase 2 (white noise, the random walk as a special AR(1) case), Phase 3 (ACF/PACF, and the identification table you memorized but didn't yet derive), and Phase 4 (stationarity, unit roots).

---

## 1. Building AR(1) from something you already fully understand

Recall from Phase 2, section 5.3, we previewed:
$$
x_t = \phi\, x_{t-1} + \varepsilon_t
$$
This IS the **AR(1) model** — "AR" stands for **AutoRegressive**, meaning literally "a regression of the series ON ITSELF" (auto = self). Compare this to ordinary linear regression, where you'd predict $y$ from some OTHER variable $x$: $y = \beta x + \varepsilon$. Here, we're doing the exact same regression logic, except the "predictor" is just the series' OWN past value. That's the entire concept — nothing more exotic than that.

**Defining every symbol again, precisely, now as a full model rather than a preview:**
- $x_t$ = the value of the series at time $t$ (what we're trying to explain/predict).
- $x_{t-1}$ = the value one step in the past (the "predictor").
- $\phi$ (phi) = the **AR coefficient** — a fixed number telling us how strongly, and in which direction, yesterday's value influences today's value.
- $\varepsilon_t$ = white noise (Phase 2) — the unpredictable fresh shock at time $t$, assumed independent of all past $x$ values and past $\varepsilon$ values.

Often you'll also see a constant added: $x_t = c + \phi x_{t-1} + \varepsilon_t$, where $c$ just shifts the overall mean level up or down (we'll come back to exactly how in section 3). For now, to keep the algebra clean, we assume $c=0$ (equivalently, assume the series has already been centered around zero).

---

## 2. The stationarity condition for AR(1) — derived, not just stated

You already know from Phase 2 that $\phi=1$ gives a random walk (non-stationary, variance grows forever) and $|\phi|<1$ gives something stationary (mean-reverting). Let's now DERIVE precisely why $|\phi|<1$ is the exact boundary, using the same "unroll the recursion" technique from Phase 2.

**Unroll the recursion:**
$$
x_t = \phi x_{t-1} + \varepsilon_t = \phi(\phi x_{t-2}+\varepsilon_{t-1}) + \varepsilon_t = \phi^2 x_{t-2} + \phi\varepsilon_{t-1}+\varepsilon_t
$$
Continuing this substitution pattern indefinitely (going back infinitely far into the past):
$$
x_t = \varepsilon_t + \phi\varepsilon_{t-1} + \phi^2\varepsilon_{t-2} + \phi^3\varepsilon_{t-3} + \dots = \sum_{j=0}^{\infty}\phi^j \varepsilon_{t-j}
$$
This form — expressing $x_t$ as an infinite weighted sum of ALL past noise shocks — is called the **MA($\infty$) representation** (moving-average representation of infinite order; you'll see WHY this name makes sense once we cover actual MA models in Part 2 — for now just recognize this is expressing $x_t$ purely in terms of past shocks, similar in spirit to what we did for the random walk in Phase 2).

**Now compute the variance of this infinite sum**, using the same rule as Phase 2 (variance of a sum of independent terms = sum of variances, and $\text{Var}(\phi^j \varepsilon_{t-j}) = \phi^{2j}\text{Var}(\varepsilon_{t-j}) = \phi^{2j}\sigma^2$):
$$
\text{Var}(x_t) = \sum_{j=0}^{\infty}\phi^{2j}\sigma^2 = \sigma^2\sum_{j=0}^{\infty}(\phi^2)^j
$$
**This is exactly the geometric series we summed in Phase 5, section 4!** Recall: $\sum_{j=0}^\infty r^j = \frac{1}{1-r}$, but CRUCIALLY only when $|r|<1$ — if $|r|\geq 1$, the sum diverges to infinity (doesn't converge to any finite number at all). Here $r=\phi^2$. So:

$$
\text{Var}(x_t) = \frac{\sigma^2}{1-\phi^2} \qquad \text{valid ONLY if } |\phi| < 1
$$

**This is the actual proof, from the ground up: if $|\phi|<1$, the variance is a finite, fixed constant — not depending on $t$ at all — satisfying stationarity Condition 2 from Phase 4. If $|\phi|\geq 1$ (including the random walk's $\phi=1$ boundary case), the geometric series never converges — the variance is literally infinite/undefined, which is the formal version of "variance grows forever" that we computed a different way (via $t\sigma^2$) directly for the random walk in Phase 2.** Both derivations are describing the same underlying fact from two different angles — good to notice, since interviewers sometimes ask for either version.

---

## 3. The mean of a stationary AR(1) process

Take the expectation of both sides of $x_t = c + \phi x_{t-1}+\varepsilon_t$. Since the process is stationary, $E[x_t]=E[x_{t-1}]=\mu$ (same constant mean at every time point — that's literally what stationarity Condition 1 requires), and $E[\varepsilon_t]=0$ (white noise property from Phase 2):
$$
\mu = c + \phi\mu + 0 \quad\Rightarrow\quad \mu - \phi\mu = c \quad\Rightarrow\quad \mu(1-\phi) = c \quad\Rightarrow\quad \mu = \frac{c}{1-\phi}
$$
Plain English: the long-run average level the process settles around is $c/(1-\phi)$. Notice this formula also breaks down (divides by zero) exactly when $\phi=1$ — one more angle confirming that $\phi=1$ is the precise breaking point where the whole stationary framework stops making sense. This is also why we could safely assume $c=0$ above: any nonzero $c$ just shifts the process to a nonzero mean $\mu$, without changing any of the variance/stationarity logic — you can always mentally work with the mean-centered series $x_t - \mu$ instead.

---

## 4. Deriving the ACF of AR(1) — finally proving the "tails off" shape from Phase 3

This is one of the most important derivations in classical time series — it PROVES the ACF signature table you memorized in Phase 3.

**Start from the defining equation, multiply both sides by $x_{t-k}$ (for some lag $k>0$), and take expectations:**
$$
x_t = \phi x_{t-1} + \varepsilon_t \quad\Rightarrow\quad E[x_t x_{t-k}] = \phi\, E[x_{t-1}x_{t-k}] + E[\varepsilon_t x_{t-k}]
$$
(Working with the mean-centered version so these expectations directly correspond to (co)variances — i.e., $E[x_t x_{t-k}] = \gamma(k)$, the autocovariance from Phase 3.)

**Key fact used here:** $E[\varepsilon_t x_{t-k}] = 0$ for $k>0$, because $\varepsilon_t$ is a FRESH shock happening at time $t$, and $x_{t-k}$ (for $k>0$) only depends on shocks from time $t-k$ and earlier — i.e., $\varepsilon_t$ hasn't happened yet as far as $x_{t-k}$ is concerned, so there's no way they can be correlated (this is really just the white noise "no memory, independent of everything before it" property from Phase 2, applied carefully).

So we get:
$$
\gamma(k) = \phi\, \gamma(k-1) \qquad \text{for } k \geq 1
$$
**This is called the Yule-Walker recursion for AR(1)** (we will generalize this to AR(p) with a full system of equations in Part 4, when we cover estimation — for now, just this simple version). Dividing both sides by $\gamma(0)$ (converting autocovariance into the normalized ACF, exactly as defined in Phase 3):
$$
\rho(k) = \phi\, \rho(k-1)
$$
**This single-step recursion, unrolled starting from $\rho(0)=1$ (always true, Phase 3), immediately gives:**
$$
\rho(k) = \phi^k
$$

**This is the complete, exact formula for the ACF of an AR(1) process, fully derived from first principles.** Let's read what it tells us: since $|\phi|<1$ for stationarity, $\phi^k$ shrinks toward 0 as $k$ grows — but it NEVER hits exactly zero at any finite lag; it just keeps getting smaller and smaller, geometrically. **This is the literal mathematical proof of "the ACF tails off gradually" from the Phase 3 table** — now you don't just have to memorize that AR processes "tail off," you've derived the EXACT formula showing precisely how ($\phi^k$, geometric decay), and you can even predict the SHAPE: if $\phi>0$, the ACF decays smoothly, all positive; if $\phi<0$, the ACF ALTERNATES sign while decaying ($\phi^k$ flips sign each time $k$ increases by 1 when $\phi$ is negative) — producing a classic "oscillating, shrinking zig-zag" correlogram shape, a specific pattern you should now be able to recognize and explain on sight.

---

## 5. Generalizing to AR(p): the full model

$$
x_t = c + \phi_1 x_{t-1} + \phi_2 x_{t-2} + \dots + \phi_p x_{t-p} + \varepsilon_t
$$

Plain English: now today's value depends on the last $p$ values, each with its OWN coefficient $\phi_1,\ldots,\phi_p$ measuring its own individual direct contribution, plus fresh noise. $p$ is called the **order** of the model — "how far back does the direct memory reach."

**Using the lag operator (introduced in Phase 4, section 5) to write this compactly:** recall $L x_t = x_{t-1}$ (applying $L$ once shifts back one step; applying it $j$ times, $L^j x_t = x_{t-j}$). We can rewrite the AR(p) equation as:
$$
(1 - \phi_1 L - \phi_2 L^2 - \dots - \phi_p L^p)\,x_t = c + \varepsilon_t
$$
Define $\Phi(L) = 1-\phi_1 L - \dots - \phi_p L^p$ — the **AR characteristic polynomial** in the lag operator. This compact notation, $\Phi(L)x_t = c+\varepsilon_t$, is the standard way AR models are written in more advanced textbooks (Hamilton, Shumway & Stoffer) and interview settings — recognize it even if you personally prefer working with the expanded form.

### 5.1 The general stationarity condition (generalizing section 2's $|\phi|<1$)
**For AR(p), the formal condition is: all roots of the characteristic equation $\Phi(z) = 1-\phi_1 z - \phi_2 z^2 - \dots - \phi_p z^p = 0$ must lie OUTSIDE the unit circle** (meaning: if you treat $z$ as a complex number and solve for where this polynomial equals zero, every solution must have magnitude $|z|>1$).

**Why does this match the AR(1) case?** For AR(1), $\Phi(z) = 1-\phi z = 0 \Rightarrow z = 1/\phi$. The condition "$|z|>1$" becomes "$|1/\phi|>1$," which algebraically means $|\phi|<1$ — EXACTLY the condition we derived directly in section 2. **The general AR(p) rule is a direct generalization of the simple AR(1) case** — same underlying idea (don't let $\phi=1$ or its generalizations happen), just expressed through polynomial roots once there's more than one coefficient to juggle simultaneously. You are not expected to hand-solve high-order polynomial roots in an interview — the point is understanding WHAT the condition means and WHY it generalizes the simple $|\phi|<1$ case, and recognizing that software (e.g., checking the "roots" output after fitting an AR model) does this check for you in practice.

---

## 6. Numerical worked example: simulate and analyze an AR(1) process by hand

Let $\phi = 0.6$, $\sigma^2=1$ (white noise variance), starting at $x_0=0$, with these noise draws: $\varepsilon_1=1.0, \varepsilon_2=-0.5, \varepsilon_3=0.8, \varepsilon_4=-1.2, \varepsilon_5=0.3$.

**Step 1 — Generate the series using $x_t = 0.6\,x_{t-1}+\varepsilon_t$:**
$x_1 = 0.6(0) + 1.0 = 1.0$
$x_2 = 0.6(1.0) + (-0.5) = 0.6-0.5=0.1$
$x_3 = 0.6(0.1)+0.8 = 0.06+0.8=0.86$
$x_4 = 0.6(0.86)+(-1.2)=0.516-1.2=-0.684$
$x_5 = 0.6(-0.684)+0.3=-0.4104+0.3=-0.1104$

Series: $[1.0, 0.1, 0.86, -0.684, -0.1104]$ — notice it wanders around zero, pulled back by the $\phi=0.6$ "memory," never drifting away permanently the way a random walk would (contrast this behavior directly against the random walk numerical example from Phase 2, section 6, where the series wandered further from zero as it accumulated shocks with NO pull-back).

**Step 2 — Use our derived formulas to predict the theoretical variance and ACF (not computed from THIS tiny sample, but from the underlying model parameters):**
$$
\text{Var}(x_t) = \frac{\sigma^2}{1-\phi^2} = \frac{1}{1-0.36} = \frac{1}{0.64} = 1.5625
$$
$$
\rho(1) = \phi^1 = 0.6, \quad \rho(2)=\phi^2=0.36, \quad \rho(3)=\phi^3=0.216, \quad \rho(4) = \phi^4 = 0.1296
$$
**Interpretation:** notice how quickly this decays — by lag 4, the correlation has dropped to about 0.13, a bit over a fifth of where it started at lag 1. If $\phi$ had instead been something closer to 1 (say 0.95), the decay would be MUCH slower ($0.95^4 \approx 0.81$, still very strongly correlated at lag 4) — **the SPEED of ACF decay is a direct, readable signal of how close $\phi$ is to the non-stationary boundary**, a genuinely useful practical intuition: a slowly-decaying ACF in real data is a visual warning sign that you might be close to a unit root, motivating a formal ADF test (Phase 4) to check.

---

## 7. Quick self-check questions

1. Derive (in your own words, following section 4's logic) why $E[\varepsilon_t x_{t-k}]=0$ for $k>0$ in the Yule-Walker derivation.
   *(Answer: $\varepsilon_t$ is a fresh, independent shock occurring exactly at time $t$; $x_{t-k}$ for $k>0$ is built entirely from information available strictly BEFORE time $t$ (past $x$ values and past noise terms up through $t-k$) — since white noise has no memory/correlation with anything before it occurred (Phase 2), a shock that hasn't happened yet cannot be correlated with something that already happened.)*
2. If $\phi = -0.7$, describe the shape of the theoretical ACF without computing every value.
   *(Answer: it will oscillate/alternate in sign at every lag (since $(-0.7)^k$ flips sign each time $k$ increases by 1) while shrinking in magnitude toward zero — a decaying zig-zag correlogram pattern.)*
3. Why does the AR(p) stationarity condition ("roots outside the unit circle") reduce to exactly $|\phi|<1$ when $p=1$?
   *(Answer: for AR(1), the characteristic equation is $1-\phi z=0$, giving the single root $z=1/\phi$; requiring $|z|>1$ is algebraically equivalent to requiring $|\phi|<1$ — the general polynomial-root condition and the simple AR(1) condition are the exact same statement, just expressed differently.)*
4. Using $\text{Var}(x_t)=\sigma^2/(1-\phi^2)$, what happens to the variance as $\phi \to 1$ (approaches 1 from below), and why does this make intuitive sense given Phase 2?
   *(Answer: the denominator $(1-\phi^2)\to 0$, so the variance blows up toward infinity — this matches Phase 2's random walk result that variance grows without bound as $\phi$ reaches exactly 1 (the unit root case), confirming AR(1) smoothly "becomes" a random walk as $\phi$ approaches 1.)*

---

## What's next
**Part 2 of Phase 6** covers **MA(q) models** — built the same way, from first principles: the formula, WHY they always have a sharp ACF cutoff (the mirror image of what we just proved for AR), the **invertibility** condition (the MA equivalent of AR's stationarity condition), and a full numerical example.

Say "next" for Part 2, or ask for more AR(p) drilling first (e.g., deriving the ACF of an AR(2) process, which involves solving a small system of two Yule-Walker equations — a common interview whiteboard exercise).
