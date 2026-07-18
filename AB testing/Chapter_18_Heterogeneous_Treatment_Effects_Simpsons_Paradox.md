# Chapter 18: Heterogeneous Treatment Effects — Segment-Level Analysis, Simpson's Paradox Traps

## 1. Definition

**Heterogeneous Treatment Effects (HTE):** the treatment effect of a feature is not constant across all users — it varies by subgroup (e.g., new vs. existing users, mobile vs. desktop, different geographies). The overall average treatment effect (ATE) can mask meaningfully different — sometimes opposite-signed — effects within subgroups.

**Simpson's Paradox:** a specific, striking manifestation of this problem where a trend that appears in several different subgroups reverses or disappears when the subgroups are combined — i.e., the aggregate result actively misleads you about what's happening in every individual subgroup.

## 2. Layman Explanation

Imagine a new onboarding flow that works great for new users (boosts their conversion) but actually confuses returning users (hurts their conversion, since they're used to the old flow). If you only look at the overall average across everyone, these two effects could partially cancel out, making the feature look like it does "nothing much" — when in reality it's a clear win for one group and a clear loss for another. Averaging masked a genuinely important, actionable story.

Simpson's Paradox is the more extreme, counterintuitive version: imagine a drug that helps recovery rates in BOTH young patients and old patients when you look at each group separately — but when you combine both groups together, the drug appears to *hurt* overall recovery. This isn't a contradiction; it happens because the mix of patients (say, more old patients, who have lower baseline recovery rates regardless of drug) getting the drug versus not getting it differs between the groups, and that mix difference distorts the combined number in a way that reverses the true, consistent within-group pattern.

## 3. Formal Explanation

**Why aggregation can mask or reverse effects:**

The overall ATE is a weighted average of subgroup-specific effects, weighted by subgroup size:

ATE_overall = Σ (nₕ/N) × ATE_h

where ATE_h is the treatment effect in subgroup h and nₕ/N is that subgroup's share of the total population. If ATE_h varies in sign or magnitude across subgroups, and especially if the *proportion* of each subgroup differs between arms (which shouldn't happen under proper randomization, but can happen due to differential attrition or non-random subgroup definitions), the combined number can misrepresent every individual subgroup's true story.

**Simpson's Paradox mechanism (formal):**

Simpson's Paradox specifically arises when a confounding variable is unevenly distributed across the comparison groups in a way that's correlated with both the "grouping" variable and the outcome. In experimental settings, this is usually NOT from broken randomization (properly randomized treatment/control should have similar subgroup mixes by design) — it more commonly arises when analyzing *observational* segments post-hoc (e.g., segmenting by a variable that's itself affected by treatment, like "did the user engage with the feature at all," which is a post-treatment variable and can differ in composition between arms in a confounded way).

**Detecting HTE — formal approaches:**
- Pre-specified subgroup analysis: decide in advance which subgroups to check (e.g., new vs. existing users, platform), avoiding the multiple-testing/fishing concerns from Chapter 15 if done post-hoc without correction.
- Interaction terms in regression: include Treatment × Subgroup interaction terms in a regression model; a significant interaction coefficient formally indicates the treatment effect differs by subgroup.
- CATE (Conditional Average Treatment Effect) estimation: more advanced methods (causal forests, uplift modeling) estimate treatment effect as a function of many covariates simultaneously, rather than pre-defined discrete subgroups — useful for exploratory HTE discovery but requires care to avoid overfitting/false discoveries.

**Key statistical caution:**
Subgroup analyses have less statistical power than the overall test (smaller n per subgroup), so a subgroup showing "no significant effect" while another shows "significant effect" doesn't necessarily mean the true effects differ — this itself needs a formal interaction test, not just eyeballing which subgroup's p-value crossed 0.05.

## 4. Levers — What Controls It, What Moves It

**Pre-specification of subgroups**
- Subgroups decided upfront (as part of pre-registration, Chapter 8) protect against fishing for a flattering story after the fact; post-hoc subgroup discovery should be explicitly labeled exploratory/hypothesis-generating, not decision-driving.

**Choice of segmenting variable (pre- vs. post-treatment)**
- Segmenting by a pre-treatment characteristic (e.g., account tenure at the start of the experiment) is safe and interpretable. Segmenting by a post-treatment behavior (e.g., "users who clicked the new feature" vs. "users who didn't") is dangerous — this variable is itself potentially affected by treatment, and comparing these groups reintroduces the exact confounding/selection problems causal inference is meant to avoid (Chapter 5).

**Sample size per subgroup**
- Smaller subgroups have less power to detect true HTE, and formal interaction tests specifically often require considerably larger total sample sizes than the main-effect test alone — a frequently underestimated planning requirement.

**Business relevance of segments**
- Not all statistically detectable HTE is actionable — a real interview-differentiating skill is connecting HTE findings to a plausible mechanism and a concrete action (e.g., "ship for new users, exclude existing users") rather than just reporting "the effect differs by segment" without a clear next step.

## 5. Worked Example

Suppose an experiment shows an overall ATE of +0.5% on conversion rate (not statistically significant, small effect). But segmenting (pre-specified) by user tenure:

| Segment | n | Conversion lift | p-value |
|---|---|---|---|
| New users (<30 days) | 40,000 | +4.2% | p=0.001 |
| Existing users (30+ days) | 160,000 | -0.8% | p=0.02 |

Weighted combination: (40,000/200,000)×4.2% + (160,000/200,000)×(-0.8%) = 0.2×4.2 + 0.8×(-0.8) = 0.84 - 0.64 = 0.20%, roughly matching the small overall +0.5% (illustrative rounding) — the aggregate number completely obscures that this feature is a strong, significant win for new users and a real, significant harm for existing users.

The correct action here isn't "ship" or "kill" based on the overall number — it's to recommend targeting the feature specifically at new users (where there's a clear, significant, mechanistically sensible win, e.g., new users benefit from simplified onboarding) while excluding existing users (who are likely experiencing friction from an unfamiliar change to a flow they'd already learned) from the launch. This is exactly the kind of finding that separates an L5-level answer ("segment and target the launch") from a lower-level answer ("the overall effect isn't significant, so we shouldn't ship").

## 6. Famous Q&A (Google / Apple style)

**Q: Your experiment shows no significant overall effect, but you suspect the treatment might help new users while hurting existing users. How would you investigate this properly?**
A: I'd check whether this subgroup comparison was pre-specified before running the test — if so, I'd run a formal interaction test (Treatment × Segment term in a regression) rather than just comparing p-values within each subgroup separately, since subgroups naturally have less power and eyeballing "one is significant, one isn't" doesn't itself prove the effects differ. If this wasn't pre-specified, I'd still investigate it, but explicitly frame it as exploratory/hypothesis-generating, and recommend a follow-up experiment specifically designed and powered to test this interaction, rather than making a ship decision on an unplanned post-hoc subgroup finding.

**Q: Explain how Simpson's Paradox could occur even in a properly randomized experiment.**
A: In a properly randomized experiment, treatment and control should have similar subgroup composition by design — so classic Simpson's Paradox (driven by differing group mixes) is less likely to arise from the randomization itself. It's more likely to show up if you segment post-hoc using a variable that's affected by treatment (e.g., "engaged with the new feature vs. didn't") — this segmenting variable isn't a fixed pre-treatment characteristic, so its distribution can differ meaningfully between arms in ways correlated with the outcome, reintroducing exactly the kind of confounding that randomization was supposed to eliminate. This is why segmenting only by pre-treatment characteristics is the safe practice.

**Q: A subgroup analysis shows the treatment effect is +5% for iOS users and -3% for Android users, both individually significant. What would you check before recommending a platform-specific launch?**
A: First, I'd verify this was either pre-specified or, if discovered post-hoc, run a formal interaction test to confirm the difference between platforms is itself statistically significant (not just that one subgroup happens to cross 0.05 and the other doesn't — smaller subgroup sizes make this a common false pattern). I'd also look for a plausible mechanism — is there something about the iOS vs. Android implementation, UI rendering, or user base composition that would sensibly explain opposite-direction effects? If the interaction is confirmed and there's a sensible mechanism, a platform-specific launch (ship on iOS, hold or iterate on Android) would be a strong, defensible recommendation rather than defaulting to the muddled overall average.

**Q: Why is segmenting by a post-treatment variable (like "did the user open the notification") considered dangerous for causal claims, even within a randomized experiment?**
A: Because whether a user opens a notification is itself potentially influenced by treatment (e.g., a redesigned notification might have different open rates than the old one), so comparing "openers" between treatment and control isn't comparing like-for-like groups anymore — the set of people who open notifications in the treatment arm may be systematically different (in ways related to the outcome) from the set who open them in the control arm. This reintroduces selection bias into what was otherwise a clean, randomized comparison — segment only by variables fixed before treatment assignment to preserve the causal validity that randomization was designed to provide.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Segmenting by a post-treatment variable (reintroduces confounding/selection bias, undermining the randomization's validity)
- ❌ Concluding two subgroups have "different" effects just because one p-value is <0.05 and the other isn't, without a formal interaction test
- ❌ Running many post-hoc subgroup comparisons without correction or without labeling them clearly as exploratory (echoes Chapter 15's multiple-testing concerns)
- ❌ Reporting only the overall ATE when a business-relevant, mechanistically sensible HTE story exists — a missed opportunity for a targeted launch recommendation
- ✅ Do: pre-specify key subgroups (tenure, platform, geography) as part of experiment design, not after seeing results
- ✅ Do: use a formal interaction term/test to confirm subgroup effects genuinely differ, rather than comparing significance status across subgroups informally

---
*Next: Chapter 19 — When A/B Testing Fails: Long-Term Holdouts & Quasi-Experiments as Backup.*
