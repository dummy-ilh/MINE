# ML Problem Formulation — Consolidated Interview Q&A

All comprehension questions from the 12-chapter curriculum, gathered into one reference. Organized by chapter so you can cross-reference the corresponding chapter file for full worked derivations.

---

## Chapter 1: From Business Objective to ML Task

**Q1.** A fintech company says "reduce fraud losses." Walk through the four-link chain: what's a plausible decision, prediction target, and model task? What data-latency issue would you flag immediately?

**Q2.** Why is "predict whether the user clicks" often a worse target than "predict expected watch time," even though click labels are far easier to obtain? Use the reward-decomposition formula to justify your answer.

**Q3.** Give an example where optimizing a well-chosen proxy metric offline produced a *worse* outcome on the true business metric online. What does this imply about how you should treat offline metric improvements during model development?

**Q4.** A stakeholder asks you to "just build a model to predict user churn." List three clarifying questions you'd ask before writing a single line of feature engineering code, and explain what a bad answer to each would derail.

---

## Chapter 2: Task Taxonomy

**Q5.** A ride-sharing app wants to formulate ETA prediction. Is this classification, regression, or ranking? Justify using the decision the ETA feeds into.

**Q6.** Using the pairwise ranking loss formula, compute the loss for the pair (A, B) under Model 1's predictions ($\hat y_A=10,\hat y_B=9$) and Model 2's predictions ($\hat y_A=2.0,\hat y_B=2.1$), given true relevances $y_A=3>y_B=2$. Which model incurs loss on this pair, and why does that match intuition?

**Q7.** A search engine wants to go from "10 billion web pages" to "10 blue links shown to the user." Describe the two-stage architecture this implies and why a single end-to-end ranking model over all 10 billion pages is not viable.

**Q8.** Give an example of a target that's naturally continuous but where you'd deliberately choose to formulate it as classification anyway, and explain the business reason that justifies the information loss.

---

## Chapter 3: Defining the Prediction Target

**Q9.** For a subscription business, define "churn" three different ways using different label windows (7-day, 30-day, explicit cancellation). For each, state one business decision it would and would not support well.

**Q10.** In the worked churn example (logins at days 5,12,40,41,42,100, $t=45$), if the label window were extended to 60 days instead of 30, would this user be labeled churned or not? Show the reasoning.

**Q11.** Explain why "total refunds issued for this transaction" is a leakage risk for a fraud model but "average refund rate for this merchant over the last 6 months" is not, using the serving-time-availability test.

**Q12.** A stakeholder wants daily retraining but also wants labels defined over a 45-day forward window. Explain why these are in tension, and propose one way to reconcile them.

---

## Chapter 4: Label Design & Ground Truth

**Q13.** For a video platform relying on "watch time" as an implicit quality label, describe one plausible selection-bias mechanism and one plausible noise mechanism that could each corrupt this label, and explain why they'd need different fixes.

**Q14.** Using the noise-rate formula, if $\alpha=0.05$ and $\beta=0.10$ with true base rate $P(y^*=1)=0.10$, compute the observed positive rate $P(y=1)$. Is it close to the true rate, and does that closeness tell you anything about whether the noise rates are acceptable?

**Q15.** A team proposes using "items added to cart" as ground truth for "purchase intent." Identify one way this implicit label could diverge from true purchase intent.

**Q16.** Design a dual-cadence labeling scheme (fast proxy + slow ground truth) for a content-moderation model where the true label takes 5 days but daily refresh is wanted. What fast proxy would you use, and what's the risk of relying on it alone long-term?

---

## Chapter 5: Offline Metrics

**Q17.** A medical screening test has FN cost (missed disease) far higher than FP cost (unnecessary follow-up). Explain why $F_1$ is a poor headline metric here and propose an alternative.

**Q18.** Explain AUC-PR's advantage over AUC-ROC using a 0.5% positive-rate scenario — why does AUC-ROC's false-positive-rate axis understate the real-world false-positive burden?

**Q19.** For a housing price regression model, argue for MSE vs. MAE using one scenario where large errors carry disproportionate cost, and one where they don't.

**Q20.** Using the NDCG@3 formula, compute NDCG@3 for relevances $[0,3,2]$ (worst item first) instead of $[3,0,2]$, and explain why the score drops even though the same items/relevances are returned, just reordered.

---

## Chapter 6: Online & Proxy Metrics

**Q21.** Construct a scenario where a search-ranking model wins offline on NDCG but loses on an online guardrail metric. Identify which divergence mechanism(s) (policy bias, proxy-gaming, behavioral adaptation) are at play.

**Q22.** A team wants "average session length" as a fast proxy for "monthly active users." Describe what correlation-study analysis you'd run before trusting this proxy, and what correlation value would make you skeptical.

**Q23.** Explain why an offline metric computed on the same biased labels the model was trained to predict is structurally unable to detect certain classes of failure, even with a perfectly held-out test set.

**Q24.** A guardrail metric (chargeback rate, 0.4% base rate) is tested with $n=50{,}000$/arm, well-powered for the primary metric. Explain why the guardrail might still be underpowered, and what you'd do before launch.

---

## Chapter 7: Objective Function Design

**Q25.** If the true FN:FP cost ratio is 20:1 (not 100:1), what weight values would you propose for weighted cross-entropy, and compute the loss for a false negative ($y=1,\hat p=0.2$) under this weighting.

**Q26.** For a demand-forecasting model where underestimating demand is far costlier than overestimating, propose a quantile-loss $\tau$ value and justify the direction of your choice.

**Q27.** Explain why pushing an extreme class weight (e.g., 500x) entirely into the loss function, rather than splitting the correction with threshold tuning, risks training instability. What symptom would you look for?

**Q28.** A team combines a relevance loss (magnitude ~1) and a fairness-penalty loss (magnitude ~50) with $\lambda=1$. Diagnose the likely problem and propose a corrected approach.

---

## Chapter 8: Multi-Objective & Constrained Formulations

**Q29.** Explain, using the convex-hull argument, why a company wanting a point deep inside a *non-convex* region of a relevance/diversity tradeoff curve cannot reach it via any weighted-sum loss, regardless of weight choices tried.

**Q30.** Using the Lagrangian formulation, explain what it means if a diversity-floor constraint doesn't bind at the optimum. What does this suggest about the constraint's business value at that threshold?

**Q31.** Using a Pareto-frontier table with diminishing marginal relevance gains, if a stakeholder insists on the loosest diversity floor purely to maximize relevance, what quantitative argument would you make for reconsidering, without an outright refusal?

**Q32.** Propose a business scenario where you'd strongly prefer a hard-constrained formulation over a weighted-sum formulation, justified by interpretability/auditability.

---

## Chapter 9: Data Availability & Cold-Start Framing

**Q33.** A grocery delivery app launches in a new city with zero local order history but has mature models in 50 other cities. Propose a transfer strategy and state what local-data-volume milestone would change your recommendation.

**Q34.** Using the UCB formula, compute the exploration bonus for an item with $n=20$ impressions at $t=50{,}000$, $\alpha=1$, and explain how the bonus changes if $\alpha$ is doubled.

**Q35.** A rare catastrophic failure event has only 8 historical positive examples across 5 years. Propose two reformulation strategies (anomaly detection, correlated proxy, synthetic augmentation), and state one validity risk for each.

**Q36.** Explain why "always rank by highest point-estimate relevance" creates systemic bias against new items even when some are genuinely excellent, and connect this to why exploration bonuses shrink as $n$ grows.

---

## Chapter 10: Baseline Design

**Q37.** A team reports "our model achieves 85% accuracy, beating the 50% random-guess baseline by 35 points." What's wrong with this comparison, and what baseline should have been reported instead?

**Q38.** A +0.02 AUC-PR improvement over the current production model — explain why this might or might not justify a full model migration, and list the three factors (significance, cost-adjustment, complexity) you'd want quantified before deciding.

**Q39.** Describe a scenario where the honest conclusion from baseline comparison is "don't build the ML model," justified by both the accuracy gap and a non-accuracy factor (interpretability, maintenance cost, latency).

**Q40.** Why does re-running an old baseline on stale data invalidate a comparison against a newly-evaluated proposed model, even if both individual numbers seem reasonable?

---

## Chapter 11: Feedback Loops & Selection Bias

**Q41.** Using the numeric exposure-spiral example (items X and Y, equal true relevance), explain why more data collected under the same biased logging policy doesn't fix the underlying bias, and can make it worse over successive retraining rounds.

**Q42.** Using the IPS formula, compute the re-weighting factor for an observation where $\pi_{\text{old}}=0.02$ and $\pi_{\text{new}}=0.10$, and explain why raw IPS becomes high-variance as $\pi_{\text{old}}$ shrinks toward zero.

**Q43.** A company discovers, three years in, that certain content categories have been persistently under-shown despite no evidence they're actually lower quality. What upfront infrastructure decision, made three years earlier, would let them test this today — and why can't they fully answer it retroactively?

**Q44.** Explain why a held-out slice of near-random exposure traffic can catch a feedback-loop bias that standard offline evaluation (train/test split from the same logged data) would miss entirely.

---

## Chapter 12: Case Studies (Integrative)

**Q45.** Walk through the 11-point formulation checklist for a new prompt: "formulate a system to detect and de-rank low-quality product listings on an e-commerce marketplace." Give at least one sentence per point.

**Q46.** Explain precisely why "predict churn probability" and "predict retention uplift from intervention" are different targets, and construct a numeric example (two hypothetical users) where targeting by raw churn probability would waste an intervention budget that targeting by uplift would not.

**Q47.** Explain why the fraud feedback loop is described as "adversarial" rather than merely "passive," and why this distinction changes the recommended retraining/monitoring cadence.

**Q48.** For the news-feed case study, identify which single design choice most directly prevents the clickbait failure mode from re-emerging, and explain the causal chain from that choice back to the original problem.

---

*48 questions across 12 chapters. Each is answerable directly from its corresponding chapter file's worked examples, formulas, and numeric illustrations — use this file as a self-test index, and the chapter files as the answer key.*
