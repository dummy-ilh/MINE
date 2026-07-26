# Phase 21: Interview Drill Bank

This phase is different again: no new concepts, pure consolidation. Four sections: rapid-fire conceptual Q&A, whiteboard-derivation prompts, product-sense case studies, and a coding-round simulation. Use this as a final pass before an actual interview, not as a first-pass learning tool — everything here assumes you've been through Phases 1-20.

---

## 1. Rapid-Fire Conceptual Q&A

*Answer each in one or two sentences, out loud, without looking anything up. If you hesitate on any of these, that's a signal to revisit the corresponding phase.*

1. What's the difference between strict and weak stationarity? *(Phase 4)*
2. Why does a random walk have infinite unconditional variance? *(Phase 2, Phase 6 Part 1)*
3. AR process: does the ACF tail off or cut off? What about PACF? Why? *(Phase 3, Phase 6 Part 1)*
4. What's the null hypothesis of the ADF test? What's the null hypothesis of KPSS? Why are they opposite? *(Phase 4)*
5. What does the "I" in ARIMA stand for, and what does it actually do to the data? *(Phase 6 Part 3)*
6. Why is AIC's penalty different from BIC's, and which one tends to pick simpler models with more data? *(Phase 6 Part 4)*
7. What's the difference between a confidence interval and a prediction interval? *(Phase 13)*
8. Why is MAPE unreliable for series with values near zero? What's the fix? *(Phase 13)*
9. What does it mean for a model to have MASE > 1? *(Phase 13)*
10. Why can't tree-based models extrapolate a trend? *(Phase 15)*
11. What earlier classical method is mathematically equivalent to the Kalman filter applied to a local level model? *(Phase 9)*
12. Why does GARCH need far fewer parameters than a high-order ARCH to capture similar persistence? *(Phase 10)*
13. Is Granger causality real causality? Why or why not? *(Phase 11)*
14. What's the difference between cointegration and spurious regression? *(Phase 7, Phase 11)*
15. Why do vanilla RNNs suffer from vanishing gradients, and what specific structural feature of LSTM fixes it? *(Phase 16 Parts 1-2)*
16. Why does self-attention need positional encoding? *(Phase 16 Part 5)*
17. What's the key assumption CausalImpact's counterfactual estimate depends on? *(Phase 18)*
18. What does conformal prediction's coverage guarantee actually rely on, and why is it shaky for time series? *(Phase 19)*
19. Why is ordinary k-fold cross-validation invalid for time series? *(Phase 13)*
20. What's the difference between anomaly detection and change point detection? *(Phase 17)*

---

## 2. "Derive This on the Whiteboard" Set

*These are the derivations most likely to actually get asked, verbatim, in a rigorous interview. Practice writing each one out from memory, not just recognizing it when you see it.*

**D1. Derive the variance of a random walk, $\text{Var}(x_t)=t\sigma^2$, by unrolling the recursion.** *(Phase 2, section 5.1)*

**D2. Derive the stationarity condition for AR(1), $|\phi|<1$, from the MA(∞) representation's variance.** *(Phase 6, Part 1, section 2)*

**D3. Derive $\rho(k)=\phi^k$ for AR(1) using the Yule-Walker approach (multiply by $x_{t-k}$, take expectations).** *(Phase 6, Part 1, section 4)*

**D4. Derive the ACF of MA(1), showing $\rho(1)=\theta/(1+\theta^2)$ and $\rho(k)=0$ for $k\geq2$.** *(Phase 6, Part 2, section 2)*

**D5. Set up the ADF test regression from the AR(1) equation, and explain precisely why an ordinary t-table can't be used.** *(Phase 4, section 6.2)*

**D6. Derive the Cochrane-Orcutt transformation, showing why $u_t-\rho u_{t-1}$ is white noise.** *(Phase 7, section 5)*

**D7. Derive the Kalman gain formula's two limiting cases (R→0 and P→0), and explain each in plain English.** *(Phase 9, section 4.2)*

**D8. Derive why GARCH(1,1) requires $\alpha_1+\beta_1<1$ for stationarity, using the same technique as AR(1)'s mean derivation.** *(Phase 10, section 6)*

**D9. Derive the forecast error variance for AR(1) at horizon $h$, and show it converges to the unconditional variance as $h\to\infty$.** *(Phase 6, Part 5, section 5.1)*

**D10. Write out the LSTM cell state update equation and explain, step by step, why it doesn't suffer the same gradient decay as the vanilla RNN's hidden state update.** *(Phase 16, Part 2, section 3)*

---

## 3. Product-Sense / Case-Study Prompts

*These have no single correct answer — what's being evaluated is whether you can structure an approach, ask the right clarifying questions, and connect your reasoning to specific tools from this course. Practice talking through each one out loud for 3-5 minutes.*

**C1. "Forecast daily active users for a new Apple Watch app that launched two weeks ago."** *(Think: cold start, Phase 15 section 4; global model with pooled category data; wide, honest prediction intervals given limited history, Phase 19.)*

**C2. "Google wants to know if a new search ranking algorithm change increased ad revenue. How would you measure that?"** *(Think: CausalImpact/synthetic control, Phase 18; the counterfactual problem; autocorrelation's effect on naive significance testing.)*

**C3. "You're forecasting demand for 50,000 different products across 500 stores. Walk me through your approach."** *(Think: hierarchical reconciliation, Phase 15 section 5; global vs. segmented models, Phase 20 section 4; system design pipeline, Phase 20.)*

**C4. "Your forecasting model's accuracy suddenly got much worse last week. How do you investigate?"** *(Think: drift/change point detection, Phase 17 section 7; check for a genuine structural break vs. a data pipeline bug vs. lag leakage newly introduced, Phase 14 section 8; residual diagnostics, Phase 6 Part 5.)*

**C5. "How would you decide between a classical ARIMA approach and a deep learning approach for a given forecasting problem?"** *(Think: data volume/number of series, Phase 15 section 4 and Phase 20 section 7's latency table; interpretability needs; whether the pattern is genuinely non-linear; computational budget.)*

**C6. "A stakeholder says your forecast was 'wrong' because actual sales were outside your 95% interval three times last month. How do you respond?"** *(Think: what 95% coverage actually means statistically — you'd EXPECT roughly 1 in 20 misses; distinguish this from genuinely miscalibrated intervals, Phase 19 section 4's coverage concept; how you'd actually check calibration.)*

---

## 4. Coding-Round Simulation

*Under realistic time pressure (aim for 20-30 minutes each), implement the following. The point is fluency, not perfection — you should be able to write these without hesitating on the core logic, even if exact syntax needs a lookup.*

**P1. Implement rolling-origin cross-validation** for a simple forecasting function, computing MAE and MASE across folds. *(Phase 13, sections 3 and 7 — pay particular attention to correctly shifting the naive baseline for MASE's denominator.)*

**P2. Implement a lag-feature and rolling-mean feature pipeline** for a raw time series, with a deliberate unit test checking that no feature at time $t$ ever uses information from time $t$ or later. *(Phase 14, section 8 — this is literally testing whether you'd catch your own lag leakage.)*

**P3. Implement Simple Exponential Smoothing from scratch** (the recursive update, not a library call), and verify your hand-derived forecast matches library output on a small dataset. *(Phase 5, section 4.)*

**P4. Implement a basic Kalman filter** (predict + update steps) for a 1D local level model, and verify it converges to a sensible estimate on simulated noisy data. *(Phase 9, section 4 — this is a genuinely common "show me you understand recursive estimation" prompt.)*

**P5. Implement CUSUM anomaly detection** on a synthetic series with an injected small, sustained shift, and show it fires while a naive 3-sigma control chart doesn't. *(Phase 17, section 4 and section 8 — literally reproduce that phase's numerical example in code.)*

---

## 5. A note on how to actually use this phase

**This drill bank is not meant to be worked through once and set aside.** The genuinely effective way to use it: pick a handful of items from each section, attempt them cold, then go back to the CITED phase for anything shaky — not to re-read the whole phase, but to re-derive specifically the piece you stumbled on. **Repeat this a few times over the days before an actual interview, rotating which items you drill, rather than doing the whole bank once and considering it "done."** The self-check questions embedded at the end of every phase throughout this course serve the same purpose at a smaller grain — this phase is the same idea, zoomed out to the whole syllabus.

---

## What's next
Phase 22 is the final phase: **Capstone Projects** — five concrete, hands-on projects (a full classical Box-Jenkins cycle on real seasonal data, an ML project using M5-style multi-series data, a deep learning project on multivariate sensor data, a CausalImpact-style analysis, and a system-design write-up) that combine everything from Phases 1-21 into complete, portfolio-worthy deliverables — genuinely useful both for interview preparation and for demonstrating applied competence beyond just answering questions.

Say "next" for Phase 22, or pick specific items from this drill bank and work through them here — I can review your derivations, your code, or your case-study reasoning against what each phase actually covered.
