# Chapter 2: Causal Estimands — ATE, ATT, ATC, CATE, LATE

## 1. Explanation

Once you accept (from Chapter 1) that you can only ever estimate *averages* of individual causal effects, the next question is: **average over whom?** This matters enormously and is one of the most common places candidates lose points in interviews — computing a perfectly correct number, for the wrong population.

Think of it as a menu of possible business questions, each with its own matching estimand:

- **"What if we treated everyone in the population?"** → **ATE** (Average Treatment Effect)
```
ATE = E[Y(1) − Y(0)]
```
averaged over the *entire* population of interest, regardless of who actually got treated.

- **"Was it worth treating the people who actually got treated?"** → **ATT** (Average Treatment Effect on the Treated)
```
ATT = E[Y(1) − Y(0) | D=1]
```
averaged only over the subgroup that *actually received* treatment.

- **"What if we extended treatment to the people who currently don't have it?"** → **ATC** (Average Treatment Effect on the Controls)
```
ATC = E[Y(1) − Y(0) | D=0]
```
averaged only over the subgroup that *did not* receive treatment — this asks what would happen if we flipped their status.

- **"What's the effect for a specific subgroup (e.g., new users, mobile users, high-spenders)?"** → **CATE** (Conditional Average Treatment Effect)
```
CATE(x) = E[Y(1) − Y(0) | X=x]
```
This is the estimand behind personalization and targeting — it lets treatment effect vary by covariate profile rather than assuming one number fits everyone.

- **"What's the effect for people whose treatment status was actually moved by our instrument or nudge?"** → **LATE** (Local Average Treatment Effect)
This one only makes sense in an IV or fuzzy-RDD context (Chapters 7 and 9) — it's the average effect specifically among "compliers," the subpopulation whose treatment take-up responds to the instrument. It says nothing about "always-takers" (who'd take treatment regardless) or "never-takers" (who'd refuse regardless).

### Why these numbers can differ — the concept of "essential heterogeneity"

If treatment effects were the same size for everyone, all these estimands would collapse into one number and none of this would matter. The reason they diverge in the real world is **selection on gains**: people (or an algorithm) often choose treatment, or are chosen for treatment, precisely *because* they're expected to benefit more. A marketing team targets a promo at customers predicted to respond well; a doctor prescribes a drug to patients likely to benefit; a targeting model shows a UI change to users predicted to like it. In all these cases, the treated group isn't a random slice of the population — it's a slice **selected partly based on expected treatment effect itself**, so ATT (measured on that select group) will systematically differ from ATC (the effect on those left out) and from ATE (the population-wide average).

### The algebraic relationship connecting them

```
ATE = π · ATT + (1 − π) · ATC
```
where $\pi = P(D=1)$ is the share of the population actually treated. This says ATE is nothing but the treatment-share-weighted blend of ATT and ATC — it's a bookkeeping identity, not a new assumption, and it's extremely useful for catching arithmetic/conceptual errors quickly in an interview (if your ATT, ATC, and treated-share don't reproduce your stated ATE, something is wrong).

### Which estimand does randomization give you?

Under full randomization, $D \perp (Y(0), Y(1))$, which implies:
```
E[Y(1)-Y(0) | D=1] = E[Y(1)-Y(0) | D=0] = E[Y(1)-Y(0)]
```
i.e., **ATT = ATC = ATE**. This is one of the cleanest, most underrated reasons randomization is so valuable: it's not just that it removes selection bias in levels, it removes the *distinction between these estimands entirely* — you don't have to ask "average over whom?" because under randomization, every subgroup's average effect is (in expectation) the same as every other subgroup's.

## 2. Example

**Scenario:** An online retailer wants to know the effect of a "free shipping" banner (D) on purchase probability (Y — here shown as a continuous "purchase likelihood score" for clarity), for 10 users. Users who are already price-sensitive/deal-seeking were more likely to have *seen* the banner (it was more prominently shown to users flagged as price-sensitive by an internal targeting model) — so this is observational, not randomized, and treatment was deliberately targeted at users predicted to respond.

| User | Y(0) | Y(1) | D (actual) |
|---|---|---|---|
| 1 | 0.3 | 0.5 | 1 |
| 2 | 0.4 | 0.7 | 1 |
| 3 | 0.2 | 0.6 | 1 |
| 4 | 0.5 | 0.6 | 1 |
| 5 | 0.6 | 0.65 | 0 |
| 6 | 0.7 | 0.72 | 0 |
| 7 | 0.55 | 0.6 | 0 |
| 8 | 0.65 | 0.68 | 0 |
| 9 | 0.6 | 0.63 | 0 |
| 10 | 0.5 | 0.55 | 0 |

Individual effects, treated users (1–4): 0.2, 0.3, 0.4, 0.1 → **ATT** = (0.2+0.3+0.4+0.1)/4 = 1.0/4 = **0.25**

Individual effects, control users (5–10): 0.05, 0.02, 0.05, 0.03, 0.03, 0.05 → **ATC** = 0.23/6 ≈ **0.038**

**ATE** = π·ATT + (1−π)·ATC, with π = 4/10 = 0.4:
```
ATE = 0.4×0.25 + 0.6×0.038 = 0.10 + 0.023 = 0.123
```
Check by direct computation: sum all 10 individual effects = (0.2+0.3+0.4+0.1)+(0.05+0.02+0.05+0.03+0.03+0.05) = 1.0+0.23 = 1.23, divided by 10 = **0.123** ✓ matches.

**Interpretation:** the banner has a *much* bigger effect (0.25) on the price-sensitive users it was actually shown to (ATT) than it would on the broader population (ATC ≈ 0.038, ATE ≈ 0.123). If leadership naively assumed "let's show this banner to 100% of users and expect a 0.25 lift," they'd overstate the impact by roughly 2x — the targeting algorithm had already found the users for whom this banner matters most, and the remaining 60% of users (who weren't targeted) are mostly people who'd buy anyway regardless of shipping cost, so extending the banner to them yields little extra.

**A second mini-example to isolate "which estimand does a randomized experiment give you":** Take just users 1, 2, 5, 6 and imagine flipping a coin: suppose the coin assigns 1 and 5 to treatment, 2 and 6 to control (pure random draw, ignoring their price-sensitivity). Now:
- Treated-by-coin (1, 5): effects 0.2, 0.05 → ATT_random = 0.125
- Control-by-coin (2, 6): effects 0.3, 0.02 → ATC_random = 0.16

These are much closer to each other than the ATT=0.25 vs ATC=0.038 gap in the real (targeted) scenario — with a large enough random sample, ATT_random and ATC_random converge to the same number (the true ATE for this subpopulation), illustrating concretely why randomization collapses the ATT/ATC distinction.

## 3. Interview Q&A

**Q: Marketing ran a promo only for high-value customers and found ATT = $50 extra spend. They want to extend it to everyone and multiply $50 by the full customer base to forecast revenue. What's wrong with this forecast?**
A: This conflates ATT with ATE. The $50 effect is specific to the customers who were *selected* for the promo (likely because they were predicted to respond well) — the ATC for previously-excluded customers could be much smaller (or even negative, e.g., if it just gives away margin to customers who'd have purchased anyway). Forecasting off ATT × full population overstates expected revenue.

**Q: What estimand does a standard, fully-randomized A/B test give you, and why?**
A: ATE — because random assignment ensures the treated and control groups are, in expectation, statistically identical in composition (including in their potential outcomes), so ATT = ATC = ATE. This equivalence is one of the main *reasons* to prefer randomization when feasible, beyond just avoiding level-bias.

**Q: What's CATE and why does it matter for personalization/targeting problems?**
A: CATE = E[Y(1)-Y(0) | X=x], the treatment effect for a specific covariate profile. It matters because business decisions are often about *whom* to target (e.g., "which users should get this discount") rather than "should we deploy to everyone" — CATE estimation (via causal forests, meta-learners like S/T/X-learners, etc.) directly supports that targeting decision, whereas a single ATE number cannot.

**Q: Can ATT be negative while ATE is positive?**
A: Yes, mathematically and substantively — imagine the people who self-select into treatment are, unusually, the ones it hurts (maybe due to some behavioral or informational quirk), while it helps everyone else more; ATT (average over the treated, negative) and ATE (weighted average over everyone, could still be positive if the untreated group is large and ATC is strongly positive) can point in different directions. This is a good "trick" question to test whether someone actually understands the weighting formula rather than just memorizing definitions.

**Q: How would you go about estimating CATE in practice at Google scale?**
A: Meta-learner approaches (S-learner: single model with treatment as a feature; T-learner: separate models per treatment arm; X-learner: cross-fits residuals, especially useful with imbalanced treatment/control sizes) or tree-based causal effect estimators (causal forests / generalized random forests) that directly split on subgroups with different treatment effects, validated via held-out uplift/qini curves.

**Q: Derive, in words, why ATE = π·ATT + (1−π)·ATC must hold as an identity, not an assumption.**
A: ATE is defined as the average of τ_i over the *entire* population. You can split that population into the treated subgroup (fraction π) and the control subgroup (fraction 1−π). The average of τ_i over the treated subgroup is, by definition, ATT; over the control subgroup, ATC. The overall average of any quantity over a population is just the size-weighted average of its subgroup averages — this is basic weighted-averaging, true by construction, with no causal assumption needed at all (it would hold even if τ_i were some arbitrary non-causal quantity).

**Q: If you're told "ATT ≈ ATC" for a given observational dataset, what does that suggest about how treatment was assigned?**
A: It suggests treatment assignment was close to independent of the treatment effect size (i.e., not much "selection on gains") — even if assignment isn't literally random, whatever process determined who got treated didn't systematically favor people who'd benefit more or less than everyone else. This is a useful diagnostic: if ATT and ATC come out very close in an observational analysis, it's at least consistent with (though not proof of) limited selection-on-effect bias.

---
**Previous: Chapter 1 — The Potential Outcomes Framework**
**Next: Chapter 3 — Randomized Experiments (RCTs): Why They Work, and Where They Break**
