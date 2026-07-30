# Chapter 6: Propensity Score Matching (PSM) and Inverse Probability Weighting (IPW)

## 1. Explanation

### The problem this chapter solves

Chapter 5 showed regression adjustment's Achilles' heel: it silently extrapolates when treated/control units don't overlap well on X, and it depends on getting the functional form right. Chapter 4 showed that with many confounders, finding literal exact matches on all of them is often impossible (the "curse of dimensionality"). This chapter introduces a tool that sidesteps both problems: instead of matching on many raw covariates, match on a *single number* that summarizes them.

### The propensity score, and the theorem that makes it powerful

Define the propensity score as:
```
e(X) = P(D=1 | X)
```
— literally, "given everything we know about this unit, how likely were they to receive treatment?"

**Rosenbaum & Rubin's theorem (1983)** states: if adjusting for the full covariate vector X would achieve ignorability, then adjusting for the *scalar* $e(X)$ achieves the same thing:
```
(Y(1),Y(0)) ⊥ D | X   ⟹   (Y(1),Y(0)) ⊥ D | e(X)
```
This is a genuinely surprising and powerful result. It means you don't need to find a control unit that matches a treated unit on age AND tenure AND device type AND ten other variables simultaneously — you just need a control unit with a similar *propensity score*, because that single number already encodes the combined "how likely to be treated" signal from all those variables. This converts an often-intractable high-dimensional matching problem into a much more manageable one-dimensional problem.

### Two ways to use the propensity score

**Matching:** for each treated unit, find one (or several) control unit(s) with the closest propensity score — often within a maximum allowed distance, called a "caliper," beyond which no match is accepted — then compare outcomes directly within matched pairs. The estimated ATT:
```
ATT_hat = (1/n_1) Σ_{i:D=1} [ Y_i − Y_{j(i)} ]
```
where $j(i)$ is unit $i$'s matched control. Intuitively: "find a control unit who *looked* just as likely to get treated as this treated unit did, but who, by chance, didn't — use their outcome as this treated unit's counterfactual stand-in."

**Inverse Probability Weighting (IPW):** rather than discarding unmatched units, keep everyone but *reweight* them by the inverse of the probability of the treatment they actually received:
```
w_i = D_i/e(X_i) + (1-D_i)/(1-e(X_i))
```
A treated unit with a *low* propensity score (surprising that they got treated) is upweighted heavily — because they're effectively standing in for many other low-propensity units who, in the vast majority, did *not* get treated. Symmetrically, a control unit with a *high* propensity score (surprising they didn't get treated) is upweighted. This reweighting constructs a "pseudo-population" in which the weighted distribution of X looks the same in the treated and control groups — as if treatment had been randomly assigned. The Horvitz-Thompson-style ATE estimator:
```
ATE_hat = (1/n)Σ [ D_i Y_i / e(X_i) ]  −  (1/n)Σ [ (1-D_i) Y_i / (1-e(X_i)) ]
```

### The critical fragility: positivity / overlap

The weights involve dividing by $e(X)$ or $1-e(X)$. If $e(X)$ is very close to 0 or 1 for some units — meaning some covariate profiles make treatment almost certain or almost impossible — the corresponding weight explodes toward infinity. A tiny handful of extreme-weight units can then dominate the entire weighted average, making the estimate wildly unstable (huge variance; the point estimate can swing dramatically from small perturbations in the data). This is why the **positivity (or overlap) assumption** — requiring $0 < e(X) < 1$ for every X value in the population you care about — isn't a technicality; it's very often the practical crux of whether PSM/IPW can work at all.

### The diagnostic you must always run: covariate balance

After matching or weighting, check whether treated and control groups now look similar on their *observed* covariates (e.g., compute the standardized mean difference for each covariate, before vs. after adjustment). This is the PSM/IPW analogue of checking Sample Ratio Mismatch in an RCT (Chapter 3) — it's your "did this actually work" diagnostic. If balance doesn't visibly improve after weighting/matching, the propensity model itself is likely misspecified (e.g., missing an important nonlinearity or interaction), and you shouldn't trust the resulting causal estimate.

## 2. Example

### Example A — Basic IPW calculation, five users

Five users, covariate X=1 (engaged) or X=0 (not engaged), D = received a push notification.

| User | X | D | Y (sessions) |
|---|---|---|---|
| 1 | 1 | 1 | 10 |
| 2 | 1 | 1 | 12 |
| 3 | 1 | 0 | 8 |
| 4 | 0 | 1 | 4 |
| 5 | 0 | 0 | 3 |

Suppose from a larger population we've estimated propensity scores: $e(X=1)=0.7$ (engaged users more likely to be targeted), $e(X=0)=0.3$.

IPW weights:
- User 1 (X=1, D=1): $w = 1/0.7 = 1.4286$
- User 2 (X=1, D=1): $w = 1/0.7 = 1.4286$
- User 3 (X=1, D=0): $w = 1/(1-0.7) = 1/0.3 = 3.333$
- User 4 (X=0, D=1): $w = 1/0.3 = 3.333$
- User 5 (X=0, D=0): $w = 1/(1-0.3) = 1/0.7 = 1.4286$

Weighted treated mean:
```
Σ(D_i · Y_i · w_i) / Σ(D_i · w_i)
= (10×1.4286 + 12×1.4286 + 4×3.333) / (1.4286+1.4286+3.333)
= (14.286+17.143+13.333) / 6.190
= 44.762 / 6.190 = 7.23
```
Weighted control mean:
```
= (8×3.333 + 3×1.4286) / (3.333+1.4286)
= (26.667+4.286) / 4.762 = 30.952 / 4.762 = 6.50
```
IPW ATE_hat ≈ 7.23 − 6.50 = **0.73** sessions.

Compare naive difference-in-means: mean(Y|D=1) = (10+12+4)/3 = 8.67; mean(Y|D=0) = (8+3)/2 = 5.5; naive diff = **3.17**. The naive comparison is much larger, because it doesn't account for the fact that engaged users (X=1) are both more likely to be treated AND have higher baseline sessions regardless of treatment — IPW correctly nets out this confounding, revealing a much smaller true effect.

### Example B — Watching weights explode (illustrating the positivity problem)

Same setup conceptually, but now imagine the targeting algorithm is much more deterministic — engaged users (X=1) almost always get the notification, disengaged users (X=0) almost never do. Suppose the estimated propensity scores are now $e(X=1) = 0.98$, $e(X=0) = 0.02$.

- A treated user with X=1: $w = 1/0.98 = 1.02$ (barely upweighted — unsurprising they were treated)
- A **control** user with X=1 (if one even exists in the data — rare, since almost everyone with X=1 is treated): $w = 1/(1-0.98) = 1/0.02 = 50$ — a massive weight
- A treated user with X=0 (rare): $w = 1/0.02 = 50$ — also massive

Suppose in your actual dataset there happens to be exactly **one** control user with X=1 (an unusual case who wasn't targeted despite being engaged), and their observed session count happens to be unusually low, say Y=1 (maybe a fluke bad day). Because their weight is 50 — vastly larger than everyone else's — this single unusual data point can single-handedly drag the entire weighted control-group average down, distorting the whole ATE estimate based on one atypical observation. This is exactly the instability the positivity assumption is meant to warn you about: **the correct response is not to trust this IPW estimate**, but to trim units with extreme propensity scores (e.g., drop anyone with $e(X)>0.95$ or $<0.05$) and report the estimate as applying only to the "overlap population" that remains, being explicit that you can no longer speak to the near-deterministic-treatment subgroup you excluded.

## 3. Interview Q&A

**Q: Why does the Rosenbaum-Rubin propensity score theorem matter practically, beyond being "a nice math result"?**
A: It converts an intractable high-dimensional matching problem (matching on many covariates simultaneously, which suffers from the curse of dimensionality) into a one-dimensional matching problem (matching on a single score). This makes adjustment feasible with realistic sample sizes and enables straightforward diagnostics like balance plots that would be far harder to interpret across many covariates at once.

**Q: You compute propensity scores and find several units with e(X) > 0.98 or < 0.02. What do you do?**
A: Trim (drop) these units from the analysis — they violate the positivity/overlap assumption for practical purposes, since they have almost no realistic comparison group in the data. Including them in IPW would give them enormous, destabilizing weights (as shown in Example B), where a single atypical observation can dominate the entire estimate. Report the resulting estimate as applying to the "overlap population" that remains, and be explicit that the conclusion doesn't generalize to the near-deterministic-treatment subgroup that was trimmed.

**Q: What's a "covariate balance check" and when do you run it?**
A: After matching or weighting, compare the distribution (e.g., standardized mean difference) of each covariate between the treated and control groups — it should look much more similar than in the raw, unadjusted data, analogous to what you'd expect under true randomization. Run it immediately after fitting the propensity model, *before* ever looking at the outcome difference, to avoid the temptation to keep tweaking the propensity model specification until you get a causal answer you like ("fishing").

**Q: Contrast the sensitivity of matching vs. IPW to a poorly-specified propensity model.**
A: Matching is somewhat self-protecting, because if there's no good match for a unit within the caliper, that unit is simply dropped — a visible, honest failure mode you can inspect (e.g., "we lost 30% of treated units to lack of matches"). IPW is more dangerous because a poorly-fit propensity model producing a few extreme values can silently blow up variance and destabilize the point estimate without it being obvious from the headline number alone — you must specifically inspect the distribution of weights (not just the final estimate) to catch this.

**Q: When would you choose a doubly-robust (augmented IPW) estimator over plain PSM?**
A: When you want protection against misspecifying either the propensity model or the outcome model individually — augmented IPW (AIPW) / TMLE-style estimators remain consistent if *at least one* of the two models (propensity or outcome) is correctly specified, giving you effectively "two chances" to get it right. This is valuable when you're not fully confident in either model's functional form, which is the normal state of affairs with messy, real-world product data.

**Q: In Example B, why was it wrong to just "trust the math" and report the IPW estimate as-is?**
A: Because the estimate was effectively determined by one unusual, high-weight observation rather than by a broad, stable pattern in the data — this is exactly what the positivity assumption warns against. An estimator can be mathematically well-defined (the formula computes a number) while being statistically worthless (that number reflects noise from a single extreme data point rather than genuine signal). Always inspect the weight distribution, not just whether the formula "runs."

**Q: How is checking covariate balance after PSM/IPW conceptually similar to checking Sample Ratio Mismatch (SRM) after an RCT?**
A: Both are diagnostic checks that verify the core mechanism actually worked as intended, run *before* trusting any treatment-effect number. SRM checks whether the randomization mechanism produced the expected group sizes (a sanity check on the "coin flip" itself); balance checks whether the reweighting/matching mechanism produced groups that look statistically similar on observed covariates (a sanity check on whether you've successfully approximated the "as good as random" condition). In both cases, failing the check should stop you from trusting downstream results, regardless of how clean the treatment-effect number looks.

---
**Previous: Chapter 5 — Regression Adjustment**
**Next: Chapter 7 — Instrumental Variables**
