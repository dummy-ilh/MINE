# Chapter 5: Causal Inference vs. Correlation — Counterfactual Framing

## 1. Definition

Correlation measures how two variables move together — it says nothing about whether one causes the other. Causal inference is the discipline of establishing whether a change in X actually *causes* a change in Y, distinct from X and Y merely co-occurring due to a third factor or reverse causation.

The core idea underlying all causal inference is the **counterfactual**: for a single unit (a user, a store, a market), the causal effect of a treatment is the difference between what *did* happen under treatment and what *would have* happened under control — for the *same* unit, at the *same* time. Since we can never observe both outcomes for the same unit simultaneously (the "Fundamental Problem of Causal Inference"), every causal inference method is really a strategy for approximating the missing counterfactual.

## 2. Layman Explanation

Imagine you take a painkiller and your headache goes away 30 minutes later. Did the pill cause the relief, or would the headache have gone away on its own? You can't rewind time and *not* take the pill to check — you only get one version of reality for yourself.

This is exactly the problem in product analytics: if a user sees a new feature and then converts, did the feature *cause* the conversion, or would they have converted anyway? Correlation just tells you the two things happened together. Causal inference is the toolkit for approximating "what would have happened without the feature" — the missing, unobservable alternate reality — using clever comparisons (like a similar group who didn't see the feature).

A/B testing is the cleanest tool for this: by randomly assigning users to see the feature or not, the "control" group becomes your best stand-in for what "would have happened" to the treatment group had they not seen it — because randomization makes the two groups statistically identical in expectation on every other dimension.

## 3. Formal Explanation

**Potential outcomes framework (Rubin Causal Model):**

For each unit i, define:
- Yᵢ(1) = outcome if unit i receives treatment
- Yᵢ(0) = outcome if unit i does not receive treatment

The individual treatment effect is Yᵢ(1) - Yᵢ(0) — but we only ever observe one of these two for any given i. This is the **Fundamental Problem of Causal Inference.**

**Average Treatment Effect (ATE):**
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]

Randomized experiments let us estimate ATE validly because randomization ensures:
E[Y(0) | Treatment group] = E[Y(0) | Control group]

i.e., in expectation, the control group's observed outcome IS a valid stand-in for what the treatment group *would have* experienced without treatment — because randomization removes systematic differences between groups on all observed and unobserved confounders.

**Why correlation ≠ causation — three classic failure modes:**
1. **Confounding:** a third variable Z causes both X and Y (e.g., users who are already highly engaged are both more likely to seek out a new feature AND more likely to convert — the feature didn't cause the conversion, prior engagement did).
2. **Reverse causation:** Y actually causes X, not the other way around (e.g., users who are about to churn browse the help center more — help-center visits don't cause churn, impending churn causes help-center visits).
3. **Selection bias:** the sample itself is non-random in a way tied to the outcome (e.g., only power users opt into a beta feature — comparing beta users to non-beta users conflates the feature's effect with pre-existing differences between power users and everyone else).

## 4. Levers — What Controls It, What Moves It

**Randomization**
- The single strongest lever for valid causal inference — when done correctly, it breaks the link between treatment assignment and any confounder, observed or unobserved, by construction.
- Randomization must be truly random and enforced at the right level (see Chapter 7 — unit of randomization) or its guarantees break down.

**Sample size for balance**
- Randomization guarantees balance *in expectation*, but with small samples, imbalance on some confounder can occur by chance. Larger n makes empirical balance more likely to match the theoretical guarantee — this is why we still check for balance on pre-treatment covariates even in randomized experiments.

**When randomization isn't possible**
- This is where quasi-experimental methods step in as substitutes for the missing counterfactual: difference-in-differences, regression discontinuity, instrumental variables, propensity score matching (all covered later in this curriculum, Part 3). Each is a different strategy for constructing a believable stand-in control group when you can't randomize.

**Confounder control**
- In observational settings, controlling for known confounders (via regression, matching, stratification) can partially address bias — but only for confounders you've measured. Unmeasured confounding remains a threat, which is exactly why randomized experiments are preferred whenever feasible.

## 5. Famous Q&A (Google / Apple style)

**Q: Users who enable a new "dark mode" setting show 15% higher retention than users who don't. Does dark mode cause retention?**
A: Not necessarily — this is a classic confounding/selection setup. Users who proactively enable a new setting are likely more engaged, more technically savvy, or more invested in the product to begin with — that prior engagement could be driving both the decision to enable dark mode AND the higher retention, with no causal link between the two. To make a causal claim, you'd want to randomly assign dark mode (or default it on for a random subset) rather than let users self-select, so the "enabled" and "not enabled" groups are comparable on everything except the setting itself.

**Q: Why is a randomized controlled experiment considered the "gold standard" for causal inference?**
A: Because randomization solves the Fundamental Problem of Causal Inference in expectation — it ensures the control group is, on average, identical to the treatment group on every observed and unobserved characteristic, making the control group's outcome a valid proxy for what the treatment group would have experienced absent treatment. No observational method can fully guarantee this, because you can never be certain you've measured and controlled for every possible confounder.

**Q: A team can't randomize a pricing change (regulatory/business constraints) but still needs to estimate its causal effect. What's your approach?**
A: I'd reach for quasi-experimental designs that try to approximate the missing counterfactual without randomization. For example, difference-in-differences if there's a comparable market/region that didn't get the price change, using it as a proxy control group and comparing the *change* over time in both groups (not just the levels) to net out any pre-existing trend differences. Alternatively, regression discontinuity if the price change was rolled out based on some threshold (e.g., account tier), or an instrumental variable if there's a variable that affects treatment assignment but not the outcome directly. Each is an imperfect substitute for randomization, so I'd also be explicit about the assumptions each relies on and where they could fail.

**Q: A metrics dashboard shows that customer support response time and customer satisfaction are negatively correlated (faster responses, happier customers). A VP wants to conclude "cutting response time causes higher satisfaction" and mandate a response-time SLA. What do you flag?**
A: I'd flag that this is observational correlation, not a randomized comparison, so reverse causation and confounding are both live possibilities. For instance, easier/simpler issues might both get resolved faster AND naturally produce happier customers — the issue complexity is a confounder driving both variables, not the response time itself. Before mandating an SLA on the strength of this correlation, I'd recommend either a randomized test (e.g., randomly prioritizing some tickets for faster response, holding issue type roughly constant) or at minimum a regression controlling for issue complexity/type as an observed confounder, while being upfront that unmeasured confounders could still bias the observational estimate.

---
*Next: Chapter 6 — Metrics: North Star, Guardrails, and the Overall Evaluation Criterion (OEC).*
