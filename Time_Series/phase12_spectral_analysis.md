# Phase 12: Spectral Analysis — Periodogram, Spectral Density, Wiener-Khinchin

Every phase so far has looked at a time series through the **time domain lens**: how does today relate to yesterday, how does a value depend on past values (ACF, AR, MA...). This phase introduces a genuinely different lens — the **frequency domain**: instead of asking "what's the relationship between consecutive points," we ask "what hidden, repeating CYCLES (fast ones, slow ones) is this series secretly built from, and how strong is each one?" By the end, a beautiful result (Wiener-Khinchin) will show these two lenses are actually two views of the exact same underlying information.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $f$ | **frequency** — how many complete up-and-down cycles happen per time step (a new core concept, defined carefully in section 2) |
| $I(f)$ | the **periodogram** — a function measuring "how much of the series' total variation is explained by cycles at frequency $f$" |
| $S(f)$ | the **spectral density** — the theoretical/true version of $I(f)$ (an ESTIMATE vs. TRUE-VALUE relationship, exactly like $\hat\rho(k)$ vs $\rho(k)$ back in Phase 3) |
| $n$ | number of observations (same as always) |
| $\gamma(k)$ | autocovariance at lag $k$ (Phase 3 — will reappear here in a new role) |

---

## 2. Building "frequency" from scratch: what does it even mean for data to "have a frequency"?

**Plain English, starting from something familiar (Phase 8's Fourier waves):** recall from Phase 8, section 3, a wave like $\sin(2\pi t/m)$ completes exactly ONE full cycle every $m$ time steps. **Frequency is simply the FLIP SIDE of that period: frequency $f = 1/m$ measures "what FRACTION of a full cycle happens per single time step."** If $m=12$ (a cycle repeats every 12 months), the frequency is $f=1/12\approx0.083$ — meaning roughly 8.3% of a full cycle occurs each month. A HIGH frequency ($f$ close to 0.5, the fastest possible frequency for evenly-spaced data) means a FAST, jittery, rapidly-oscillating pattern (completing a cycle in just 2 time steps). A LOW frequency (close to 0) means a SLOW, gradual pattern, unfolding over many, many time steps.

**Why this "different lens" is genuinely useful:** in the time domain (ACF, Phase 3), we asked "how correlated is today with $k$ steps ago" — a single number per lag. **In the frequency domain, we instead ask "how much of the series' total UP-AND-DOWN VARIATION is caused by fast wiggles, versus slow wiggles, versus medium-speed wiggles?"** — decomposing the total variance of the series into contributions from different SPEEDS of oscillation, rather than into contributions from different TIME LAGS. Both are valid, complete descriptions of the same series' behavior — just organized along a different axis (lag vs. speed-of-cycle).

---

## 3. The Periodogram: measuring how much "cycle strength" exists at each frequency

**Building the intuition first, no formula yet:** imagine you had a big toolbox of pure sine/cosine waves at every possible frequency (exactly like Phase 8, section 3's Fourier building blocks) — one wave that cycles very slowly, one that cycles a bit faster, one faster still, all the way up to the fastest possible wave. **The periodogram asks, for EACH of these candidate frequencies: "if I tried to fit JUST this one wave to my data (via ordinary regression, Phase 7's machinery, exactly like Phase 8's Fourier regression), how much of the series' total variance would that single wave manage to explain?"** A frequency where the answer is "a lot" reveals a genuine, strong cyclical pattern at that speed hiding in the data. A frequency where the answer is "almost none" means there's no meaningful cyclical structure happening at that particular speed.

**The formula (built directly from Fourier regression coefficients, Phase 8, section 3):**
$$
I(f) = \frac{n}{2}\left(\hat\beta_f^2 + \hat\gamma_f^2\right)
$$
Where $\hat\beta_f, \hat\gamma_f$ are EXACTLY the same kind of Fourier regression coefficients from Phase 8, section 3 (the sine and cosine coefficients), but now estimated separately for EVERY candidate frequency $f$, one at a time, rather than choosing just a couple of seasonal frequencies as Phase 8 did. **Plain English: $I(f)$ is basically "how big are the fitted sine/cosine wave coefficients at this particular frequency, squared and combined" — a big value means a strong, genuine cyclical presence at that speed; near-zero means essentially nothing happening at that speed.**

**A genuinely important structural connection worth pointing out explicitly:** this is EXACTLY the same underlying computational machinery as Phase 8's Fourier regression, just applied SYSTEMATICALLY across every possible frequency (not just a couple of pre-chosen seasonal ones) to build a complete PICTURE of where all the cyclical strength in the series is concentrated, across the entire range of possible speeds.

**Plotting $I(f)$ against $f$ (a periodogram plot):** sharp SPIKES at specific frequencies reveal genuine, strong periodic/cyclical components hiding in the data — e.g., if you had disguised daily data with a strong weekly pattern but hadn't been told the period, a periodogram would show an obvious spike right at $f=1/7$, REVEALING the hidden 7-day cycle directly from the data, without you needing to already know to look for it. **This is a genuinely practical, real use case: detecting UNKNOWN or UNEXPECTED periodicities that you wouldn't have thought to check for using Phase 5/Phase 8's methods, which require you to already suspect/specify a particular seasonal period in advance.**

---

## 4. The Spectral Density: the theoretical, "true" version

**Exactly parallel to the ACF/sample-ACF relationship from Phase 3:** the periodogram $I(f)$ is an ESTIMATE, computed from your actual finite dataset — just like $\hat\rho(k)$ was an ESTIMATE of the true, unobservable $\rho(k)$. The **spectral density** $S(f)$ is the TRUE, underlying theoretical quantity the periodogram is trying to estimate — a property of the actual stochastic process (Phase 2) generating the data, not of your particular finite sample.

---

## 5. The Wiener-Khinchin Theorem: the elegant bridge between the two lenses

**This is genuinely one of the most beautiful, unifying results in all of time series analysis, and it directly ties together the ENTIRE time-domain toolkit you've spent 11 phases building (ACF) with this brand-new frequency-domain toolkit.**

**The theorem, stated in plain English first:** *the spectral density $S(f)$ and the autocovariance function $\gamma(k)$ (Phase 3!) contain EXACTLY THE SAME INFORMATION about a process — you can compute one directly from the other, with nothing lost or gained either way.* Formally, $S(f)$ is built by combining ALL the autocovariances, at every lag, weighted by cosine waves at frequency $f$:
$$
S(f) = \gamma(0) + 2\sum_{k=1}^{\infty}\gamma(k)\cos(2\pi f k)
$$
**Plain English, piece by piece:** start with $\gamma(0)$ (the plain variance, Phase 3). Then, for each lag $k=1,2,3,\ldots$, take the autocovariance AT that lag ($\gamma(k)$ — exactly the quantity you learned to hand-compute back in Phase 3, section 7) and weight it by a cosine wave oscillating at frequency $f$, then ADD all these weighted terms together (doubled, for a technical symmetry reason involving negative frequencies that we won't dwell on). **The genuinely deep takeaway: the ENTIRE autocorrelation structure of a series (every single $\gamma(k)$ value, for every lag, all the memory/dependency information from Phase 3) gets folded together into this ONE formula to produce the spectral density at EACH frequency.**

**Why does this make intuitive sense, connecting back to what you already derived?** Recall Phase 6, Part 1, section 4: an AR(1) process's ACF is $\rho(k)=\phi^k$ — SLOWLY decaying if $\phi$ is close to 1. **A slowly-decaying ACF means strong dependency persists over LONG lags — which, translated into frequency-domain language via this theorem, means the spectral density is concentrated at LOW frequencies (slow cycles), since long-lasting memory/dependency corresponds to smooth, slow-moving behavior rather than fast jitter.** Conversely, an ACF that decays to zero almost immediately (weak, short-lived memory) translates into spectral density spread out more evenly across ALL frequencies (including high/fast ones) — genuinely no particular "preferred speed" of variation. **This is precisely why WHITE NOISE (Phase 2, ZERO autocorrelation at every lag beyond 0) has a spectral density that is COMPLETELY FLAT across every frequency** — plug $\gamma(k)=0$ for all $k\geq1$ directly into the Wiener-Khinchin formula above, and every term in the sum vanishes, leaving $S(f)=\gamma(0)$, a CONSTANT, unchanging across every frequency $f$. **This is the literal mathematical origin of the name "white noise" from Phase 2, section 4, finally fully explained: just like white LIGHT contains all colors/frequencies EQUALLY (no color dominates), white NOISE contains all cyclical frequencies EQUALLY — a flat spectral density, with no particular speed of oscillation more prominent than any other.** You were told to just accept this name back in Phase 2; now you have the actual mathematical justification.

---

## 6. Smoothing the periodogram: a genuinely necessary practical fix

**The problem: the raw periodogram $I(f)$, despite being a reasonable-sounding ESTIMATE of $S(f)$, is actually a surprisingly POOR, erratic, noisy estimator — a real, well-known statistical fact.** Specifically, the periodogram's own variance does NOT shrink as you collect more data (a genuinely counter-intuitive fact, unlike almost every other estimator you've met in this course, where more data generally means a more precise/reliable estimate) — adding more data points just gives you a periodogram evaluated at MORE frequency points, not a smoother/more reliable estimate at each individual point. **The raw periodogram plot tends to look jagged and spiky everywhere, making it genuinely hard to distinguish real, meaningful cyclical spikes from pure random noise-driven jaggedness.**

**The fix: smooth the periodogram by averaging together NEARBY frequency values, exactly the same "trade off responsiveness for stability" idea you've now seen repeatedly (moving averages in Phase 1/5, exponential smoothing's $\alpha$ in Phase 5, the Kalman gain in Phase 9).** A **Daniell filter** (the standard named technique here) is simply a specific, systematic moving-average SMOOTHING applied directly to the raw periodogram values across neighboring frequencies — literally Phase 1, section 1's moving-average concept, just applied along the FREQUENCY axis instead of along the TIME axis. **Plain English: instead of trusting each individual raw periodogram value $I(f)$ on its own (too noisy/unreliable), average together several NEIGHBORING frequency values to get a smoother, more trustworthy, still-recognizable curve — trading a bit of sharp frequency-resolution for a large gain in reliability**, precisely the same fundamental tradeoff you've encountered repeatedly throughout this entire course, just showing up again in a new setting.

---

## 7. A small numerical taste: computing $I(f)$ at one frequency by hand

Reuse the tiny 4-point Fourier example from Phase 8, section 7: $m=4$, and suppose at $f=1/4$ (one full cycle every 4 steps — the fundamental frequency for this data length), we'd already fit (via the same Fourier regression logic) $\hat\beta_f=3, \hat\gamma_f=1$ (identical numbers to Phase 8's worked example, deliberately reused so you can trace the connection).

$$
I(1/4) = \frac{n}{2}(\hat\beta_f^2+\hat\gamma_f^2) = \frac{4}{2}(3^2+1^2) = 2(9+1)=2(10)=20
$$

**Interpretation: this single number, 20, represents how much of the series' total variance is attributable to a cyclical pattern completing exactly one cycle every 4 time steps.** If you computed $I(f)$ at OTHER candidate frequencies too (e.g., $f=1/2$, the fastest possible pattern for this data) and found much SMALLER values there, that would confirm the $f=1/4$ cyclical pattern is the genuinely dominant one — exactly matching what you already know from Phase 8's worked example, where the seasonal pattern (period 4) was clearly the real signal built into that toy dataset. **This is the same underlying fact, now viewed and quantified through the frequency-domain lens instead of the seasonal-decomposition lens.**

---

## 8. Quick self-check questions

1. In plain English, what does a sharp SPIKE in a periodogram plot at a specific frequency $f_0$ tell you about the data?
   *(Answer: it indicates a genuine, strong cyclical/periodic pattern in the data that completes a full cycle roughly every 1/f₀ time steps — a real, dominant repeating component at that specific speed of oscillation.)*
2. Why does white noise have a perfectly FLAT spectral density across all frequencies, derived directly from the Wiener-Khinchin formula?
   *(Answer: white noise has zero autocovariance at every lag beyond 0 (Phase 2); plugging γ(k)=0 for all k≥1 into the Wiener-Khinchin formula S(f)=γ(0)+2Σγ(k)cos(2πfk) makes every term in the sum vanish, leaving S(f)=γ(0), a constant that doesn't depend on f at all — hence flat across every frequency.)*
3. Why is the raw, unsmoothed periodogram considered a poor/unreliable estimator, and what is the standard fix?
   *(Answer: the raw periodogram's variance doesn't shrink as more data is collected — more data just gives more noisy frequency points rather than more reliable estimates at each point — making it look jagged and spiky even when there's no real signal there; the standard fix is smoothing by averaging neighboring frequency values together, e.g., using a Daniell filter, trading some frequency resolution for greater reliability.)*
4. How does an AR(1) process with φ close to 1 (strong, slowly-decaying autocorrelation) translate into a statement about its spectral density's shape?
   *(Answer: slow ACF decay means strong dependency across long lags, which translates (via Wiener-Khinchin) into spectral density concentrated at LOW frequencies — smooth, slow-moving cyclical behavior dominates, rather than fast/high-frequency jitter.)*

---

## What's next
Phase 13 shifts gears from classical statistical theory into **time series cross-validation and evaluation** — why ordinary k-fold cross-validation is actually WRONG for time series (a genuine, common mistake), the correct rolling-origin alternative, and a full derivation of every major forecast accuracy metric (MAE, RMSE, MAPE and its flaws, MASE, and the pinball/quantile loss used for probabilistic forecasts) — the practical toolkit for correctly judging whether any of the models from Phases 1-12 are actually any good.

Say "next" for Phase 13, or ask for more periodogram/spectral density drilling first.
