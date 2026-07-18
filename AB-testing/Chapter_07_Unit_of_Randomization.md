# Chapter 7: Unit of Randomization — User vs. Session vs. Device, and Interference Risks

---

## 1. Intuition

So far, most of experimentation quietly assumes you can randomize at the user level and that one user's assignment doesn't affect another user's outcome. This chapter is about what happens when that assumption breaks — which, at a company with social products, marketplaces, ranking systems, and shared infrastructure, is often.

The core question: **what is the "unit" you randomize (user? session? query? device? geographic region? cluster of connected users?), and what happens to your causal estimate when units interact with each other?**

This is one of the highest-value topics for an L5 interview, because junior candidates almost always default to "just randomize by user ID" without considering when that's wrong — and getting this wrong in production doesn't just add noise, it introduces **systematic bias in a specific, predictable direction.**

### Layman analogy
Imagine testing a new "invite a friend" feature. If you randomize at the *session* level, the same user could see the feature in one session and not in another — muddying what you're even measuring, since their behavior isn't cleanly split into "always treated" vs. "always control." Randomizing at the *user* level fixes that — each user consistently gets one experience.

But now imagine the feature IS about inviting friends. If User A (treatment) invites User B (control), and User B ends up affected by that invite, you've violated the core assumption that each user's outcome only depends on their *own* treatment assignment. This is like testing whether a specific person exercising affects their health, but their treated gym partner keeps dragging the "control" partner to the gym too — the two aren't really independent anymore, and your comparison gets contaminated.

The right unit of randomization is the smallest, most stable "bubble" you can draw around a treatment such that what happens inside the bubble doesn't leak out and contaminate other bubbles.

---

## 2. Core Definitions

- **Unit of randomization**: the entity you randomly assign to treatment or control — most commonly user, device, session, or a cluster (e.g., geographic market, marketplace listing, social-graph community). This choice determines what "independent observation" means for your entire analysis, and must align with (1) the level at which the treatment is actually experienced, and (2) the level at which the outcome metric is measured.

- **SUTVA (Stable Unit Treatment Value Assumption)**: requires that unit $i$'s potential outcome depends only on unit $i$'s own treatment assignment, not on anyone else's:

$$Y_i(T_1, T_2, ..., T_n) = Y_i(T_i)$$

Formally, SUTVA requires two things:
1. **No interference**: unit i's outcome depends only on unit i's own treatment assignment, not on anyone else's.
2. **No hidden variations of treatment**: "treatment" means the same thing for every unit that receives it.

- **Interference** (a.k.a. network effect, spillover effect, SUTVA violation): occurs when one unit's treatment assignment affects another unit's outcome — breaking the independence assumption that the entire statistical framework (CLT, standard variance formulas, hypothesis testing, sample-size formulas) relies on.

**Why this matters mathematically**: all the standard formulas (the ATE estimator, SE formulas, sample-size formula) implicitly assume SUTVA. If it's violated, your point estimate can be **biased**, not just noisy — more data doesn't fix it, because the bias doesn't shrink with n.

---

## 3. Common Units of Randomization — When to Use Each

- **User-level**: default choice for most product experiments — treatment persists across sessions/devices (via login), giving each user a single consistent experience. Requires stable user identification (logged-in ID, not just cookie).
- **Device-level**: used when login isn't available/reliable (logged-out flows), or when the feature itself is device-specific (e.g., a UI change only relevant to mobile). Risk: the same user across multiple devices could get inconsistent experiences.
- **Session-level**: appropriate only when the treatment effect is expected to be fully contained within a single session and doesn't need consistency across visits — rare for most product features, more common for things like search-ranking experiments where each query/session is independent-ish.
- **Cluster-level** (geo, marketplace, school, hospital, social-graph community): required when interference is expected to be severe within a cluster but negligible across clusters — e.g., testing a driver-pricing algorithm in a rideshare marketplace, where treating some drivers in a city and not others in the *same* city creates market-level spillovers, so you randomize whole cities instead.
- **Switchback (time-based cluster) design**: the same unit (e.g., a whole city/market) alternates between treatment and control across different time windows, rather than splitting units in space.

---

## 4. 🧭 Flowchart: Choosing Your Unit of Randomization

```
                         START: Designing a new experiment
                                      │
                                      ▼
              Is the treatment experienced consistently
              only when the user is logged in, on any device?
                                      │
                    ┌─────────────Yes─┴─No──────────────┐
                    ▼                                    ▼
        Could this feature plausibly            Login unreliable/unavailable,
        affect OTHER users indirectly            or feature is device-specific?
        (referrals, shared feed, chat,                     │
         marketplace supply/demand,                        ▼
         shared ranking/ad inventory)?              Use DEVICE-level
                    │                                randomization
        ┌────────No─┴─Yes───────────┐               (watch for same user,
        ▼                            ▼                multiple devices →
 Use USER-level              What is the DOMINANT       inconsistent UX)
 randomization                mechanism of interference?
 (the safe default)                   │
                     ┌─────────────────┼──────────────────────┐
                     ▼                 ▼                      ▼
           Social graph /      Shared marketplace /    Session/query is
           referrals / chat     shared finite supply     independent-ish
           (contamination        (pricing, matching,     (e.g., search
            spreads via           ad auctions)            ranking test)
            direct connections)         │                      │
                     │                  ▼                      ▼
                     ▼          Is interference          Use SESSION-level
        Use GRAPH-CLUSTER      geographically bound?       randomization
        randomization                   │                (rare — confirm no
        (community detection    ┌───Yes─┴──No────┐        cross-session
         on the social graph,   ▼                 ▼        consistency need)
         assign whole        Use GEO/CLUSTER   Use SWITCHBACK
         clusters together)   randomization      design (same unit
                              (whole cities/      alternates treatment/
                               markets/DMAs)       control over TIME
                                                    instead of space)
                     │                 │                      │
                     └────────┬────────┴──────────┬───────────┘
                               ▼                    ▼
                    Estimate the Design Effect (DEFF)
                    and re-run your sample-size calc —
                    cluster/switchback designs need a
                    LARGER raw n for the same power.
                               │
                               ▼
                    Run an A/A test at the chosen unit
                    before launching — unexpected
                    "significant" differences with no
                    real treatment = bug or leftover
                    interference. Also check SRM.
                               │
                               ▼
                         LAUNCH EXPERIMENT
```

---

## 5. Sources & Mechanisms of Interference

- **Social/network effects**: a referral or invite feature — if I'm in treatment and invite you, your outcome is affected by my treatment status even if you're in control. Classic in messaging apps, social networks, marketplaces.
- **Shared/limited inventory (marketplace effects)**: in a two-sided marketplace (e.g., ride-sharing, ads), if treatment increases one rider's booking rate, it can reduce available driver supply for control riders — the two arms are now competing for the same finite pool.
- **Search/ranking effects**: if a new ranking algorithm changes what content surfaces, and content has a fixed "budget" of relevance/attention, treatment users' improved results might come at the expense of what's shown to control users if there's any shared exposure budget (e.g., ad auction dynamics).
- **General equilibrium / market-level effects**: pricing experiments where a price change in the treatment group affects overall market supply/demand, indirectly affecting control users too.

---

## 6. 🔀 Flow Chart: How Interference Contaminates Your Estimate

### A) Referral / social spillover (biases the estimate **down** — underestimate)

```
 TREATMENT USER                                    CONTROL USER
 (gets "invite a friend,                          (randomly assigned
  get $5 credit" feature)                          to NO referral feature)
        │                                                  ▲
        │  invites their friend                            │
        ▼                                                  │
 Friend clicks invite link,                                │
 signs up, redeems bonus  ─────── friend happens ──────────┘
                                   to be a CONTROL
                                   user in THIS test
                                          │
                                          ▼
                     Control group's engagement/signup metric
                     is inflated by spillover from treatment
                                          │
                                          ▼
        Measured effect = (Treatment mean − Control mean)
        is SMALLER than the TRUE effect, because some of the
        treatment's real impact leaked into the control group.
        → Bias DIRECTION: underestimate of ATE
        → Gets WORSE as the true effect gets bigger (more spillover)
```

### B) Marketplace / shared-inventory cannibalization (biases the estimate **up** — overestimate)

```
 TREATMENT RIDER                                   CONTROL RIDER
 (sees new pricing/matching                        (sees old pricing/
  algorithm, books more rides)                      matching algorithm)
        │                                                  ▲
        │  books available drivers                        │
        ▼                                                  │
 Shared, FINITE pool of drivers ───── fewer drivers ───────┘
 in the same city right now           remain available
                                       for control riders
                                          │
                                          ▼
              Control group's booking rate is artificially
              SUPPRESSED — not because the old algorithm is
              worse, but because treatment "stole" the supply
                                          │
                                          ▼
        Measured effect = (Treatment mean − Control mean)
        is LARGER than the TRUE effect, because control was
        artificially starved of a shared resource.
        → Bias DIRECTION: overestimate of ATE
```

### The general pattern

```
     Does treatment CONSUME a shared,          Does treatment CREATE
     finite resource that control              a positive spillover that
     also needs (drivers, ad slots,            reaches control users
     ranking real estate)?                     (referrals, shared content,
              │                                 viral loops)?
              ▼                                          ▼
     Control is artificially                   Control is artificially
     STARVED / made to look WORSE               BOOSTED / made to look BETTER
              │                                          │
              ▼                                          ▼
     Treatment effect is                        Treatment effect is
     OVER-estimated                              UNDER-estimated
```

Knowing this directional intuition — and being able to reason about which direction the bias goes for a given interference mechanism — is a sharper interview signal than just saying "there might be interference here."

---

## 7. Detecting Interference

- **Look for known network structure**: social features, marketplaces, shared infrastructure/caching are all red flags worth checking proactively.
- **Run an A/A test at the candidate unit of randomization**: unexpected significant "effects" with no real treatment difference can signal interference or a broken randomization mechanism.
- **Sample Ratio Mismatch (SRM) checks**: verify actual traffic split matches intended split (e.g., 50/50); large deviations often indicate a bug in randomization/bucketing logic, sometimes correlated with an underlying interference issue.
- **Budget-split / shared-resource experiments**: some interference can be measured, not just avoided — e.g., running an experiment where treatment and control literally compete for a shared, artificially fixed resource pool can let you estimate market-level effects that would otherwise be masked in a naive user-level split.

---

## 8. Solutions: Alternative Randomization Designs

**Cluster randomization**: instead of randomizing individual units, randomize entire clusters (geographic markets, social-network communities, entire marketplaces) into treatment or control. This contains the interference *within* a cluster rather than letting it leak across the treatment/control boundary.

**Design effect (the statistical cost of clustering)**:

$$\text{Design Effect (DEFF)} = 1 + (m-1)\rho$$

where $m$ = average cluster size and $\rho$ = intra-cluster correlation coefficient (ICC). Your **effective sample size** is:

$$n_{eff} = \frac{n}{DEFF}$$

So if clusters are large or highly correlated internally, effective power drops sharply even with the same raw n. Larger clusters (e.g., whole countries) reduce interference risk further but shrink effective sample size dramatically (few independent clusters → wide confidence intervals, sometimes only feasible with weeks/months of data). Smaller clusters (e.g., zip codes) preserve more statistical power but may not fully contain the interference if network effects span cluster boundaries.

**Geo-based randomization (switchback / geo experiments)**: randomize by region (city, DMA) rather than user — common for marketplace and ranking experiments where interference is geographically bounded.

**Switchback designs (time-based randomization)**: instead of splitting users, alternate the *entire system* between treatment and control across time blocks (e.g., odd hours = treatment, even hours = control). Useful for marketplace-wide interventions where you can't cleanly split users or geography. Requires care about **carryover effects** between time blocks (does treatment's effect linger into the next control block?).

**Ego-network / graph cluster randomization**: for social networks specifically, cluster users into densely-connected communities (using graph clustering algorithms) and randomize whole clusters together, minimizing edges that cross the treatment/control boundary.

---

## 9. Levers — What Controls Interference Risk

**Nature of the feature**
- Purely individual features (e.g., a UI color change) → user-level randomization is safe.
- Social/network features (referrals, shared content, messaging) → interference risk is high; consider cluster-level randomization (e.g., by friend-graph community) or specialized designs.
- Marketplace/two-sided features (pricing, matching algorithms) → supply and demand interact; consider switchback tests or geo-based randomization.

**Login/identification infrastructure**
- Availability of stable, cross-device user IDs determines whether user-level randomization is even feasible; logged-out experiences often force device or cookie-level randomization, with known risks of ID churn.

**Granularity of the cluster (for cluster randomization)**
- Larger clusters reduce interference risk but shrink effective sample size dramatically. Smaller clusters preserve power but may not fully contain interference if network effects span cluster boundaries.

---

## 10. Worked Example — Referral Feature on a Ride-Sharing App

You're testing a "invite a friend, get $5 credit" referral feature. You randomize at the **user level**: 50% treatment, 50% control.

**The interference problem**: a treatment user invites their friend, who happens to be in the control group. That friend now uses the app (redeeming their own signup bonus) because of the treatment user's action — but this activity gets counted in the **control group's** metrics, since the friend themselves was randomized to control.

**Consequence**: control group's engagement is inflated by spillover from treatment, making the *measured* treatment effect (Treatment mean − Control mean) **smaller than the true effect** — a "contamination" bias that underestimates impact, and it gets worse the larger the true effect actually is (bigger effects create more spillover). This matches the general referral/spillover flow diagrammed in Section 6A.

**Fix**: randomize by **social cluster** (friend groups or connected components in the social graph) instead of by individual user, so an entire friend group is assigned to the same arm — this contains referral spillovers within-arm rather than across arms. You'd need a graph-clustering step in your experiment pipeline before assignment, and you'd expect to need a larger raw n to hit the same power due to the design effect from within-cluster correlation.

---

## 11. Production Considerations

- **Always ask "could this feature plausibly affect users outside the treatment group?" before choosing a randomization unit** — this single question catches most interference issues before they contaminate a result.
- **Cluster/geo randomization costs statistical power** — be ready to defend the tradeoff (bias reduction vs. variance increase) as an explicit design decision, not a free win.
- **A/A tests can help detect interference** — if a true A/A test (both arms identical) shows systematic non-null differences correlated with cluster membership, that's a sign of contamination in your existing randomization scheme.
- **Some interference can be measured, not just avoided**: e.g., a "budget-split" experiment (treatment and control literally compete for a shared, artificially fixed resource pool) can let you estimate market-level effects otherwise masked in a naive user-level split.

---

## 12. Interview Traps

1. **Defaulting to "randomize by user ID"** without pausing to ask whether the feature could create interference — the single most common way candidates lose points in this topic.
2. **Treating interference purely as "extra noise"** rather than recognizing it introduces **bias** that doesn't average out with more data.
3. **Proposing cluster randomization without acknowledging the power/variance cost** (design effect) — proposing the fix without its tradeoff signals incomplete understanding.
4. **Not being able to reason about the *direction* of bias** (over- vs. under-estimate) for a given interference mechanism when asked "which way would this bias your result?" (see Section 6's general pattern).

---

## 13. Famous Interview Q&A

**Q: You're testing a new "watch party" feature where users can invite friends to co-view content. Why is user-level randomization risky here?**
A: Because the feature is inherently social — if a treatment user invites a friend who's in the control group, the control user's behavior (and possibly their measured outcome, like watch time) gets influenced by their friend's treatment status. This breaks SUTVA/interference-free randomization: the control group is no longer a clean counterfactual because it's been "contaminated" by exposure to the treatment through the social graph. I'd consider randomizing at a cluster level — e.g., by friend groups or communities — so interactions largely stay within a single treatment condition, or use a graph-cluster randomization approach specifically designed for social network experiments.

**Q: Your A/A test (no real treatment difference, but split traffic into two "arms" anyway) shows a statistically significant difference in a marketplace matching experiment. What would you suspect?**
A: A major red flag. It suggests either a bug in the randomization/bucketing mechanism (SRM — traffic isn't actually split as intended) or, in a marketplace context, interference: even though the treatment is identical for both "arms," if the marketplace is shared (same pool of drivers/listings serving both arms), whatever's actually driving the metric may be contaminated by cross-arm interactions, or there's a system-level issue like caching or logging that differs between arms unintentionally. I'd start by checking for SRM, then dig into whether the two "arms" are actually operating on shared, interacting resources.

**Q: A rideshare company wants to test a new pricing algorithm. Why would they use a switchback design instead of standard user-level randomization?**
A: Pricing in a marketplace is inherently interference-prone — drivers and riders in the same city interact through shared supply and demand, so treating some riders with new pricing while others in the same city get old pricing distorts the market for everyone (e.g., driver availability shifts based on where treatment riders are requesting). A switchback design randomizes at the market-time level instead — the same city alternates between the old and new pricing algorithm across different time windows — avoiding the split of a single, interconnected market into two artificially separate conditions at the same time, while still generating enough independent time-blocks across markets for statistical power.

**Q: What's the tradeoff of choosing a very large cluster size (e.g., whole countries) for a cluster-randomized experiment?**
A: Large clusters minimize interference risk since spillovers are unlikely to cross national boundaries, but they drastically shrink effective sample size — with, say, only 20-30 countries available, degrees of freedom for a cluster-level test are tiny, producing very wide confidence intervals and requiring a much longer test duration (weeks to months) to detect anything but large effects. It's a direct tradeoff between minimizing interference bias and preserving statistical power — the right cluster granularity depends on how far the interference actually propagates in practice, worth investigating (e.g., via network analysis) rather than assuming.

**Q: A PM proposes testing a new "group chat" feature by randomizing individual users 50/50. What's your first question, and what randomization scheme would you propose instead?**
A: My first question: "could a treatment user's group-chat experience affect the experience of a control user, e.g., through shared group membership?" Almost certainly yes — a group chat is inherently multi-user, so a mixed group (some treatment, some control members) would contaminate both arms. I'd propose randomizing at the **group** or **social-cluster** level instead — either the whole chat group is treatment or the whole group is control — accepting the resulting reduction in effective sample size (via the design effect) as the cost of an unbiased estimate.

---

## 14. L5-Differentiating Talking Points

- Spontaneously asking "is there any way this treatment could affect control users indirectly?" before proposing a randomization scheme is exactly the instinct L5 interviewers are probing for — often asked as an open-ended product scenario specifically to see if you raise interference unprompted.
- Being able to state the design effect formula and explain that cluster randomization is a **bias-variance tradeoff**, not a strictly better choice, shows quantitative maturity.
- Naming techniques by category (geo experiments, switchback designs for marketplace/ranking problems, graph-cluster randomization for social products) signals familiarity with how this plays out at a large company's scale and product surface area.
- Reasoning about the **direction** of interference bias (Section 6) — not just flagging that bias exists — is a sharper signal than a generic "there might be interference" answer.

---

## 15. Comprehension Check (Self-Test)

1. State SUTVA formally and explain, in your own words, what it means for a treatment effect estimate to be "biased" (not just noisy) when SUTVA is violated.
2. Give an example of an interference mechanism that would cause you to *overestimate* the treatment effect, and one that would cause you to *underestimate* it.
3. What is the design effect, and why does cluster randomization typically require a larger raw sample size to achieve the same power as user-level randomization?
4. Describe a switchback design and when you'd choose it over cluster randomization.
5. A PM proposes testing a new "group chat" feature by randomizing individual users 50/50. What's your first question, and what randomization scheme would you propose instead?
6. Using the flowchart in Section 4, walk through how you'd choose a randomization unit for a new "shared shopping cart" feature where friends can co-build an order together.
7. In the referral worked example (Section 10), which direction is the bias, and why does it get worse as the true effect size grows?
8. What's the difference between an A/A test flagging a bug (SRM) versus an A/A test flagging genuine interference? How would you start to tell them apart?

---
*This tutorial merges two chapters on unit of randomization and interference — one framed around definitions, common units, and product-scenario Q&A; the other framed around SUTVA formalism, the design-effect formula, and directional-bias reasoning with a referral worked example. Two flowcharts were added: (1) a decision flowchart for choosing the right unit of randomization given a feature's characteristics, and (2) interference/contamination flow diagrams showing exactly how spillover (referral) and cannibalization (marketplace) mechanisms bias the treatment effect in opposite directions.*
*Next: Chapter 8 — Hypothesis Formulation & Pre-Registration.*
