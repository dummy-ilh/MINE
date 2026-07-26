# Phase 18: Causal Inference in Time Series — CausalImpact, Synthetic Control, A/B Testing Pitfalls

This phase answers a genuinely common business question that's DIFFERENT from forecasting: not "what will happen next," but **"what EFFECT did a specific intervention (a marketing campaign, a price change, a product launch) actually HAVE?"** This is precisely the Google-specific topic flagged in the original syllabus (CausalImpact), and it builds directly on Phase 9's state space/Kalman filter machinery.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| intervention / treatment | the specific event/action whose EFFECT we want to measure (e.g., a marketing campaign launch) |
| treatment period | the time AFTER the intervention happened |
| pre-period | the time BEFORE the intervention, used to establish a baseline relationship |
| counterfactual | "what WOULD have happened, if the intervention had NOT occurred" — a genuinely unobservable, hypothetical quantity we have to ESTIMATE |
| control series/unit | a comparison series NOT affected by the intervention, used to help estimate the counterfactual |
| ATE | Average Treatment Effect — the standard, general causal-inference term for "the average size of the true causal effect" |

---

## 2. The core problem: you can never observe the counterfactual directly

**Plain English, the single foundational idea of ALL causal inference, time series or otherwise:** suppose Google runs an ad campaign for two weeks, and search-related revenue goes up by 8% during that period. **The genuinely hard, fundamental problem: you CANNOT simply attribute that entire 8% increase to the campaign** — some of it might have happened ANYWAY (general growth, a seasonal effect, Phase 1's trend/seasonality, a competitor's product failure driving customers your way, pure noise) **— to know the TRUE causal effect of the campaign specifically, you'd need to compare what ACTUALLY happened (with the campaign running) against what WOULD HAVE happened in that EXACT SAME period, IF the campaign had NOT run — this second, hypothetical quantity is called the counterfactual, and it is, by definition, something you can NEVER directly observe** (you can't rerun history twice, once with and once without the campaign, and compare). **This is precisely, structurally, the SAME "we only observe ONE realization of the random process" problem from Phase 2, section 2 — just now applied to a hypothetical INTERVENED-vs-NOT-intervened comparison, rather than to a single stochastic process's random path.**

**The entire practical challenge of causal inference in time series: since we cannot directly observe the counterfactual, we must CONSTRUCT a credible ESTIMATE of it — and the quality of any causal inference method comes down entirely to how CREDIBLE that estimated counterfactual actually is.**

---

## 3. CausalImpact: Google's method, built directly on Phase 9's Kalman filter

**The core strategy, in plain English before any formula:** use the PRE-INTERVENTION period (before the campaign started) to learn a reliable STATISTICAL RELATIONSHIP between the series you care about (e.g., Google's search revenue) and one or more CONTROL SERIES that were NOT affected by the intervention (e.g., search revenue in a DIFFERENT country where the campaign didn't run, or a related metric genuinely unaffected by this specific campaign). **Then, PROJECT that learned relationship FORWARD into the treatment period, using the (unaffected) control series' ACTUAL values during the treatment period to predict what the target series WOULD HAVE looked like, had the intervention not happened — this projection IS the estimated counterfactual.** **Finally, the estimated CAUSAL EFFECT is simply the GAP between what ACTUALLY happened and this estimated counterfactual: $\text{effect}_t = y_t^{\text{actual}} - y_t^{\text{counterfactual}}$.**

**Why is this built on Phase 9's state space machinery, specifically?** **CausalImpact uses a Bayesian Structural Time Series model (BSTS) — literally EXACTLY the state space framework from Phase 9, section 3 (a hidden LEVEL/TREND state, PLUS a regression relationship to the control series) — fit using the Kalman filter (Phase 9, section 4) on the PRE-intervention data, and then used to GENERATE FORECASTS (Phase 6, Part 5's forecasting logic, and Phase 9's predict-step) forward through the treatment period, WITH their full prediction intervals (also directly Phase 6, Part 5, section 5's machinery).**

**Concretely, the model structure (directly assembling pieces you already fully know):**
$$
y_t = \mu_t + \beta\, x_t + \varepsilon_t
$$
where $\mu_t$ is a Phase 9-style local level/trend STATE component (capturing the target series' own organic trend), $x_t$ is the CONTROL series (assumed unaffected by the intervention), and $\beta$ is a learned regression coefficient (Phase 7's ordinary regression machinery) — **the WHOLE model is fit via the Kalman filter using ONLY pre-intervention data, then used to forecast $y_t$ forward through the treatment period using the ACTUAL, REAL values of $x_t$ during that period (since the control series, by assumption, is genuinely unaffected and continues behaving normally) — this forecast IS the estimated counterfactual, complete with a full Bayesian prediction interval (directly Phase 6, Part 5, section 5, and Phase 9's uncertainty-propagation logic) quantifying how confident we should be in that counterfactual estimate.**

**The critical, genuinely important ASSUMPTION this entire method rests on, worth stating explicitly (a real interview "what could go wrong" question): the control series must be GENUINELY unaffected by the intervention itself, and the PRE-intervention relationship between the target and control series must remain STABLE/valid throughout the treatment period.** **If the "control" series was actually ALSO affected by the intervention (e.g., some spillover effect), or if the underlying relationship between target and control genuinely changes for some OTHER, unrelated reason right around the same time as the intervention (a confounding coincidence), the entire estimated counterfactual — and therefore the entire estimated causal effect — becomes unreliable, precisely because it's built on top of a broken assumption.** **This directly echoes Phase 7, section 6's spurious regression warning: a plausible-looking statistical relationship (here, between target and control) is not automatically a trustworthy one, and the validity of the WHOLE causal estimate hinges on this relationship genuinely holding for the RIGHT reasons.**

---

## 4. Synthetic Control: a closely related method, for when a SINGLE clean control series isn't available

**The motivating scenario, slightly different from CausalImpact's setup:** suppose you want to measure the effect of a new state-level policy on that state's economy — but no SINGLE other state is a genuinely perfect, clean comparison (every other individual state differs from the treated one in various ways). **Synthetic control's clever idea: instead of relying on ONE control series, construct a WEIGHTED COMBINATION of SEVERAL other, untreated units (e.g., a blend of several other states) specifically chosen so that this WEIGHTED BLEND closely matches the treated unit's behavior during the PRE-intervention period.** **This weighted blend — the "synthetic control" — becomes the estimated counterfactual, exactly playing the same role as CausalImpact's control series $x_t$, section 3, just constructed as an optimized COMBINATION of several real comparison units rather than relying on a single one.**

**Why this can be MORE credible than picking just one control unit: by construction, the synthetic control is specifically OPTIMIZED to match the treated unit's PRE-intervention behavior as closely as possible (directly analogous, in spirit, to fitting ANY regression model to minimize error, Phase 7's OLS machinery, just here optimizing a set of WEIGHTS across several comparison units rather than ordinary regression coefficients on predictor variables) — giving you more confidence that, absent the intervention, the treated unit would likely have continued tracking this well-matched synthetic blend, rather than relying on the hope that one single, arbitrarily-chosen comparison unit happens to be a good match.**

---

## 5. The genuinely important A/B testing pitfall: autocorrelation inflates false positives

**This is a distinct, but closely related, and genuinely very commonly tested practical warning.** **Recall Phase 7, section 4's precise finding: autocorrelated regression errors cause STANDARD ERRORS to be UNDERSTATED, making things look "statistically significant" when they may not genuinely be.** **This EXACT SAME mathematical mechanism causes a serious, real, well-documented problem in ordinary A/B testing when the underlying data is autocorrelated over TIME** (which time series data, almost by definition, tends to be — directly recalling Phase 2, section 2's foundational point about why time series needs special tools at all).

**Concretely, how this plays out in a real A/B test:** suppose you're testing whether a website redesign increases daily conversion rate, running the test for, say, 14 days, and comparing the average conversion rate in the "treatment" group against the "control" group using an ORDINARY statistical test (e.g., a standard t-test) that ASSUMES each day's observation is an independent, fresh data point. **But daily conversion rates are typically genuinely AUTOCORRELATED (a good day tends to be followed by another good day, due to persistent underlying factors — recall Phase 2, section 2's core point about why time series data isn't independent) — meaning your ACTUAL number of genuinely INDEPENDENT pieces of information is SUBSTANTIALLY LESS than the raw count of 14 days might suggest (directly Phase 7, section 4's "effective sample size" warning, now showing up in an A/B testing context).** **A standard t-test, unaware of this, computes standard errors AS IF you had 14 fully independent observations — systematically UNDERSTATING the true uncertainty, and correspondingly INFLATING your apparent statistical significance, potentially leading you to confidently declare a "winning" treatment that's actually just riding on ordinary, persistent, autocorrelated noise.**

**The practical, genuine fixes, directly connecting to tools you already have:** (1) explicitly MODEL the autocorrelation structure (e.g., using Phase 7, section 5's Cochrane-Orcutt-style GLS correction, or fitting an explicit ARIMA-error structure, Phase 7, section 8) rather than assuming independence; (2) use a BLOCK-based or CLUSTER-based resampling/bootstrap approach that respects the time-dependency structure, rather than treating each day as independent; (3) directly apply a CausalImpact/synthetic-control-style approach (sections 3-4 above) instead of a naive t-test, since these methods are SPECIFICALLY built, from the ground up, to properly account for a series' own autocorrelation structure via the underlying state space/BSTS machinery (Phase 9), rather than naively assuming independence the way an ordinary t-test does.

---

## 6. Difference-in-Differences (DiD): briefly, connecting to a slightly different but related setup

**New term, briefly: Difference-in-Differences.** Plain English: a genuinely common, classic causal-inference technique for when you have BOTH a treated group AND a control group, observed BOTH before AND after an intervention (a 2×2 structure: treated/control crossed with before/after). **The core idea: compute the treated group's CHANGE (before → after), separately compute the control group's CHANGE (before → after) over that SAME period, and then take the DIFFERENCE BETWEEN these two changes** — plain English, "how much MORE did the treated group change, compared to how much the control group ALSO naturally changed over that same period (capturing whatever GENERAL, non-treatment-related drift/trend was happening anyway)." **This is genuinely, structurally very similar to CausalImpact's core logic (section 3) — both are fundamentally trying to NET OUT whatever "would have happened anyway" using a comparison/control group — DiD is typically applied with just TWO time points (before/after) and assumes a SIMPLE, constant CONTROL-GROUP-VS-TREATED-GROUP relationship, whereas CausalImpact explicitly models the FULL TIME SERIES DYNAMICS (via the Kalman filter/BSTS machinery) throughout BOTH periods, generally a MORE FLEXIBLE, richer approach when you have GENUINE, granular time series data available (rather than just two snapshot time points) — directly the reason a syllabus focused on TIME SERIES specifically emphasizes CausalImpact-style methods over classic two-point DiD.**

**The key assumption DiD requires, worth knowing by name (a genuinely common interview follow-up question): the "parallel trends" assumption — the treated and control groups must have been on genuinely PARALLEL/similar trajectories BEFORE the intervention, for the DiD estimate to be credible** (directly analogous, in spirit, to CausalImpact's requirement that the pre-intervention target-control relationship be stable and genuine, section 3's critical assumption, and Synthetic Control's explicit optimization to ensure exactly this kind of close pre-period matching, section 4).

---

## 7. A concise, numerical illustration of the CausalImpact logic

Suppose (pre-intervention) daily search revenue $y_t$ and a control-country's revenue $x_t$ have historically tracked closely, with a learned relationship $\hat y_t \approx 1.0\times x_t$ (a simple 1-to-1 relationship, for illustration).

**Pre-intervention (baseline check):** $x=[100,102,98,101]$, $y=[101,103,99,100]$ — genuinely close, tracking well (consistent with the assumed $\beta\approx1.0$ relationship).

**Treatment period (campaign now running in the target country only):** control country continues normally: $x=[103, 105]$. **Estimated counterfactual (what target country's revenue WOULD have been, absent the campaign, using the learned relationship):** $\hat y^{\text{counterfactual}} = [103, 105]$ (applying the learned $\beta\approx1.0$ relationship directly to the control's actual values).

**Actual observed target country revenue during treatment:** $y^{\text{actual}}=[115,119]$.

**Estimated causal effect at each time point:**
$$
\text{effect}_1 = 115-103=12, \qquad \text{effect}_2=119-105=14
$$
**Plain English interpretation: the campaign appears to have caused an estimated lift of roughly +12 to +14 units per day — NOT the raw, naive "115 minus the historical average of about 100" (‌which would overstate things by ignoring genuine, ongoing organic movement, captured here by the control's own rise from 100s to 103-105) — but specifically the gap ABOVE what the control-series relationship would have predicted, properly netting out whatever was happening ANYWAY (the shared organic drift both series were experiencing, echoed in the control's own rise from ~100 to ~104).** **This concretely demonstrates section 2's core principle: the causal effect isn't "before vs. after," it's "actual vs. a properly estimated counterfactual" — a genuinely different, more rigorous, and more defensible quantity.**

---

## 8. Quick self-check questions

1. Why is "the counterfactual" fundamentally unobservable, and what is the general strategy every method in this phase uses to deal with that?
   *(Answer: the counterfactual is "what would have happened without the intervention," which requires observing the same unit/period both with and without the treatment simultaneously — impossible, since you can't rerun history twice. Every method in this phase deals with this by ESTIMATING the counterfactual using some form of control/comparison data (an unaffected control series, a weighted synthetic blend of comparison units, or a control group) that is believed to behave as the treated unit WOULD have, absent the intervention.)*
2. What is the critical assumption CausalImpact depends on regarding its chosen control series, and what happens if that assumption is violated?
   *(Answer: the control series must be genuinely unaffected by the intervention itself, and the pre-intervention statistical relationship between the target and control series must remain stable through the treatment period; if the control series is actually also affected (spillover) or the relationship breaks for some unrelated reason around the same time, the estimated counterfactual — and therefore the entire estimated causal effect — becomes unreliable.)*
3. Precisely why can naive A/B testing on autocorrelated daily data produce inflated false-positive rates?
   *(Answer: standard statistical tests like a t-test assume each observation is independent; if daily data is autocorrelated (nearby days share persistent underlying factors), the true amount of independent information is much less than the raw count of days suggests, causing standard errors to be understated and making a treatment look statistically significant more often than is actually warranted — the exact same mechanism as Phase 7's autocorrelated-regression-errors warning, applied here to A/B testing.)*
4. What is the "parallel trends" assumption in Difference-in-Differences, and how does it relate to assumptions made by CausalImpact and Synthetic Control?
   *(Answer: it requires that the treated and control groups were following genuinely similar/parallel trajectories before the intervention, for the DiD estimate to be credible; this directly parallels CausalImpact's requirement that the pre-intervention target-control relationship be stable and genuine, and Synthetic Control's explicit optimization to closely match the treated unit's pre-period behavior — all three methods fundamentally depend on the pre-intervention comparison being a trustworthy, valid baseline.)*

---

## What's next
Phase 19 covers **Probabilistic and Bayesian Forecasting** more broadly — extending Phase 9's Bayesian Structural Time Series and Phase 16's DeepAR into a fuller treatment of why point forecasts are often insufficient for real business decisions, quantile regression forecasting, and **conformal prediction** — a genuinely elegant, distribution-free method for building prediction intervals with formal coverage guarantees, regardless of what model produced the original forecast.

Say "next" for Phase 19, or ask for more CausalImpact/synthetic-control drilling first.
