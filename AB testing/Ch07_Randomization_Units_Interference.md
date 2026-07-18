# Chapter 7: Randomization Units & Interference

## 1. Intuition

So far, every chapter has quietly assumed you can randomize at the user level and that one user's assignment doesn't affect another user's outcome. This chapter is about what happens when that assumption breaks — which, at a company like Google with social products, marketplaces, ranking systems, and shared infrastructure, is often.

The core question this chapter answers: **what is the "unit" you randomize (user? session? query? geographic region? cluster of connected users?), and what happens to your causal estimate when units interact with each other?**

This is one of the highest-value chapters for an L5 interview, because junior candidates almost always default to "just randomize by user ID" without considering when that's wrong — and getting this wrong in production doesn't just add noise, it introduces **systematic bias in a specific, predictable direction.**

## 2. SUTVA and Why It Matters

Recall from Chapter 1: **SUTVA (Stable Unit Treatment Value Assumption)** requires that unit $i$'s potential outcome depends only on unit $i$'s own treatment assignment, not on anyone else's:

$$Y_i(T_1, T_2, ..., T_n) = Y_i(T_i)$$

i.e., the potential outcome for user $i$ shouldn't depend on the whole vector of everyone's treatment assignments — only on their own. When this fails, we say there's **interference** (also called a network effect or spillover effect).

**Why this matters mathematically**: all the formulas from Chapters 2-6 (the ATE estimator, the SE formulas, the sample size formula) implicitly assume SUTVA. If it's violated, your point estimate can be **biased**, not just noisy — more data doesn't fix it, because the bias doesn't shrink with $n$.

## 3. Common Sources of Interference

- **Social/network effects**: a referral or invite feature — if I'm in treatment and invite you, your outcome is affected by my treatment status even if you're in control. Classic in messaging apps, social networks, marketplaces.
- **Shared/limited inventory (marketplace effects)**: in a two-sided marketplace (e.g., ride-sharing, ads), if treatment increases one rider's booking rate, it can reduce available driver supply for control riders — the two arms are now competing for the same finite pool.
- **Search/ranking effects**: if a new ranking algorithm is tested and it changes what content surfaces, and content has a fixed "budget" of relevance/attention, treatment users' improved results might come at the expense of what's shown to control users if there's any shared exposure budget (e.g., ad auction dynamics).
- **General equilibrium / market-level effects**: pricing experiments where a price change in the treatment group affects overall market supply/demand, indirectly affecting control users too.

## 4. The Direction and Sign of the Bias

Interference doesn't just add noise — it typically biases your estimate in a *predictable direction*, and the direction depends on the mechanism:

- **Substitution/cannibalization** (e.g., marketplace, ranking, shared inventory): treatment "steals" positive outcomes from control, making control look artificially worse and the treatment effect look artificially inflated (**overestimate** of ATE).
- **Positive spillovers** (e.g., referrals, viral features, content sharing): control users indirectly benefit from treatment users' behavior (e.g., a control user receives a referral from a treatment friend), making control look artificially better than it "should," which **underestimates** the true treatment effect (since some of the treatment's effect leaked into control).

Knowing this directional intuition — and being able to reason about which direction the bias goes for a given interference mechanism — is a sharper signal than just saying "there might be interference here."

## 5. Solutions: Alternative Randomization Units

**Cluster randomization**: instead of randomizing individual users, randomize entire clusters (e.g., geographic markets, social network communities, or entire marketplaces) into treatment or control. This contains the interference *within* a cluster rather than letting it leak across the treatment/control boundary.

- Tradeoff: cluster randomization typically has **much lower effective sample size** for the same number of raw users, because outcomes within a cluster are correlated (not independent), inflating variance. The design effect formula:

$$\text{Design Effect (DEFF)} = 1 + (m-1)\rho$$

where $m$ = average cluster size and $\rho$ = intra-cluster correlation coefficient (ICC). Your **effective sample size** is $n_{eff} = n / DEFF$ — so if clusters are large or highly correlated internally, your effective power drops sharply even with the same raw $n$.

**Geo-based randomization (switchback / geo experiments)**: randomize by region (city, DMA) rather than user — common at Google/Uber/Lyft for marketplace and ranking experiments where interference is geographically bounded.

**Switchback designs (time-based randomization)**: instead of splitting users, alternate the *entire system* between treatment and control across time blocks (e.g., odd hours = treatment, even hours = control), useful for marketplace-wide interventions where you can't cleanly split users or geography. Requires care about carryover effects between time blocks.

**Ego-network / graph cluster randomization**: for social networks specifically, cluster users into densely-connected communities (using graph clustering algorithms) and randomize whole clusters together, minimizing edges that cross the treatment/control boundary.

## 6. Worked Example

Suppose you're testing a "invite a friend, get $5 credit" referral feature on a ride-sharing app. You randomize at the **user level**: 50% treatment, 50% control.

**The interference problem**: a treatment user invites their friend, who happens to be in the control group. That friend now uses the app (redeeming their own signup bonus) because of the treatment user's action — but this activity gets counted in the **control group's** metrics, since the friend themselves was randomized to control.

**Consequence**: control group's engagement is inflated by spillover from treatment, making the *measured* treatment effect (Treatment mean − Control mean) **smaller than the true effect** — this is a "contamination" bias that underestimates impact, and it gets worse the larger the true effect actually is (since bigger effects create more spillover).

**Fix**: randomize by **social cluster** (e.g., friend groups or connected components in the social graph) instead of by individual user, so an entire friend group is assigned to the same arm — this contains referral spillovers within-arm rather than across arms. You'd need a graph clustering step in your experiment pipeline before assignment, and you'd expect to need a larger raw $n$ to hit the same power due to the design effect from within-cluster correlation.

## 7. Production Considerations

- **Always ask "could this feature plausibly affect users outside the treatment group?" before choosing a randomization unit** — this single question catches most interference issues before they contaminate a result.
- **Cluster/geo randomization costs statistical power** — be ready to defend the tradeoff (bias reduction vs. variance increase) as an explicit design decision, not a free win.
- **A/A tests can help detect interference** — if a true A/A test (both arms identical) shows systematic non-null differences correlated with cluster membership, that's a sign of contamination in your existing randomization scheme.
- **Some interference can be measured, not just avoided**: e.g., running a "budget-split" experiment (treatment and control literally compete for a shared, artificially fixed resource pool) can let you estimate market-level effects that would otherwise be masked in a naive user-level split.

## 8. Interview Traps

- **Trap #1**: Defaulting to "randomize by user ID" without pausing to ask whether the feature could create interference — this is the single most common way candidates lose points in this section.
- **Trap #2**: Treating interference purely as "extra noise" rather than recognizing it introduces **bias** that doesn't average out with more data.
- **Trap #3**: Proposing cluster randomization without acknowledging the power/variance cost (design effect) — proposing the fix without its tradeoff signals incomplete understanding.
- **Trap #4**: Not being able to reason about the *direction* of bias (over- vs under-estimate) for a given interference mechanism when asked "which way would this bias your result?"

## 9. L5-Differentiating Talking Points

- Spontaneously asking "is there any way this treatment could affect control users indirectly?" before proposing a randomization scheme is exactly the instinct L5 interviewers are probing for — this is usually asked as an open-ended product scenario specifically to see if you raise interference unprompted.
- Being able to state the design effect formula and explain that cluster randomization is a **bias-variance tradeoff**, not a strictly better choice, shows quantitative maturity.
- Naming Google-relevant techniques by category (geo experiments, switchback designs for marketplace/ranking problems) signals familiarity with how this actually plays out at Google's scale and product surface area, even if you don't know Google's exact internal tooling.

## 10. Comprehension Check

1. State SUTVA formally and explain, in your own words, what it means for a treatment effect estimate to be "biased" (not just noisy) when SUTVA is violated.
2. Give an example of an interference mechanism that would cause you to *overestimate* the treatment effect, and one that would cause you to *underestimate* it.
3. What is the design effect, and why does cluster randomization typically require a larger raw sample size to achieve the same power as user-level randomization?
4. Describe a switchback design and when you'd choose it over cluster randomization.
5. A PM proposes testing a new "group chat" feature by randomizing individual users 50/50. What's your first question, and what randomization scheme would you propose instead?

---
*Next: Chapter 8 — One-sided vs Two-sided Testing*
