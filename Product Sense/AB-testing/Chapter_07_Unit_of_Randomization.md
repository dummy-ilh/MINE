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
# Chapter 7 Supplement: Complete Table of Randomization Units

A reference table of every unit of randomization you might propose in an interview, what it's good for, and — most importantly — the specific failure mode that makes it break in a given scenario. Interviewers care less about you naming a unit and more about you catching *why it fails* before they have to point it out.

---

## Master Table

| Unit | What it means | Best suited for | Why it fails / breaks down |
|---|---|---|---|
| **User ID (logged-in)** | Randomize on a stable account identifier; persists across devices/sessions | Default choice — purely individual features (UI tweaks, personal recs, personal settings) | Fails when the feature is inherently multi-user (referrals, chat, shared carts) — one user's treatment leaks into another user's outcome via direct interaction. Also fails if login isn't universal (logged-out traffic gets no consistent assignment). |
| **Device ID** | Randomize on device/cookie identifier | Logged-out flows, device-specific UI features (e.g., mobile-only redesign) | Same user on multiple devices (phone + laptop) can land in *both* arms — inconsistent experience contaminates the "one clean treatment per unit" assumption. Cookie churn (clearing cookies, browser resets) also causes unit identity to drift mid-experiment. |
| **Session ID** | Randomize per visit/session | Search/ranking experiments where each query is close to independent | Almost never safe for anything with memory or persistence — same user flips between treatment and control across sessions, so you can't attribute a stable "user-level" effect to either arm. Learning effects (users adapting to a UI) get muddled. |
| **Cluster — geographic (city/DMA/country)** | Whole region assigned to one arm | Marketplace or pricing experiments where interference is geographically bounded (rideshare, delivery) | Fails if interference actually crosses the cluster boundary you drew (a driver working two adjacent cities). Also fails statistically if you pick too few clusters — degrees of freedom collapse and effective sample size shrinks (design effect), sometimes to the point the test can't detect anything but huge effects. |
| **Cluster — social graph community** | Graph-clustering algorithm groups tightly-connected users, whole community assigned together | Social/network features — referrals, group chat, co-viewing, shared feeds | Fails if the graph-clustering step doesn't fully contain the interaction (edges still cross cluster boundaries — "leakage" at the seams). Also expensive/slow to compute at scale, and cluster sizes are uneven, which complicates the design-effect correction. |
| **Cluster — marketplace/listing level** | Randomize whole marketplaces, listings, or supply pools | Two-sided marketplace features (matching, pricing) where a shared finite resource (drivers, ad inventory) is contested | Fails when the "marketplace" isn't cleanly separable — e.g., drivers or ad inventory pools already overlap across the clusters you defined, so contamination still occurs across the treatment/control boundary. |
| **Switchback (time-based, same unit alternates)** | The *entire system* (e.g., one city) alternates between treatment/control across time blocks | Marketplace-wide interventions you can't cleanly split by user or geography (global pricing algorithm change) | Fails if there's **carryover** — the treatment's effect lingers into the next control block (e.g., driver repositioning takes hours to reverse), contaminating the "control" period. Also vulnerable to time-based confounds (day-of-week, time-of-day effects) if blocks aren't randomized carefully. |
| **Query/impression-level** | Randomize per individual search query or ad impression, not per user | High-volume, stateless ranking/ads experiments where no persistent identity is needed | Fails the moment there's any session memory, personalization, or user-level learning — the same user can get inconsistent treatment across near-simultaneous queries, making user-level metrics (retention, satisfaction) impossible to attribute cleanly. |
| **Content/item-level** | Randomize which *pieces of content* get a treatment (e.g., some videos get new thumbnails), not which users | Content-side experiments (thumbnail tests, item ranking boosts) where the same user sees both treated and untreated items | Fails when items compete for the same limited attention/slot budget — boosting some items necessarily displaces others, so "control" items are starved by the presence of "treatment" items in the same feed (shared exposure budget interference). |
| **Budget-split (artificial shared-resource split)** | Deliberately split a shared finite resource pool itself, and measure market-level effects directly | When you actually *want* to measure the interference/market effect rather than avoid it | Not really a fix — it's a different research question (measuring the spillover, not eliminating it). Fails if used as a substitute for cluster randomization when your goal was actually a clean, interference-free point estimate. |

---

## Quick Diagnostic: Why a Naive Choice Fails, by Scenario

| Scenario | Naive unit picked | Why it fails |
|---|---|---|
| "Invite a friend" referral feature | User ID | Treatment user's invite reaches a control user → control outcome contaminated → **underestimates** true effect |
| New rideshare pricing algorithm | User ID | Shared driver supply pool → treatment riders "steal" drivers from control riders in the same city → **overestimates** true effect |
| Group chat redesign | User ID | Mixed treatment/control membership in the *same* group → both arms' experience is contaminated by the other |
| New search ranking model | User ID (if session-based product) | If usage is stateless/logged-out, no persistent ID exists — falls back to device/session, which risks the same-user-multiple-arms problem |
| Global feed ranking change | Geo-cluster (too small, e.g., zip code) | Network effects (shared trending content, viral loops) span far beyond a zip code — cluster too small to contain interference |
| Global feed ranking change | Geo-cluster (too large, e.g., country) | Contains interference well, but only ~20–30 independent clusters exist worldwide → tiny effective sample size → can't detect anything but massive effects, needs months of data |
| Ad auction / ranking real estate test | User ID | Fixed inventory of ad slots/relevance budget — boosting treatment users' results can come at the direct expense of what's shown to control users sharing the same exposure budget |

---

## The One Filter Question

Before picking any unit, ask:

> **"Could this treatment plausibly affect the outcome of someone assigned to the *other* arm?"**

- **No** → user-level (or device-level, if login is unavailable) is safe.
- **Yes, through direct connections (referrals, chat, shared content)** → social-cluster randomization.
- **Yes, through a shared finite resource (supply, inventory, ad slots)** → geographic cluster or switchback.
- **Yes, but you actually want to measure the spillover itself** → budget-split design.

Every cluster/switchback choice costs you sample size via the **design effect**:

$$DEFF = 1 + (m-1)\rho, \qquad n_{eff} = \frac{n}{DEFF}$$
This is the **most fundamental technical trap** in A/B testing. 

You can have a perfect hypothesis, perfect pre-registration, and run the test for a month—but if you randomize at the wrong unit, **your p-values are completely invalid**. 

At Google (Search/Ads), Meta (Social Graphs), and Apple (iCloud/Devices), the randomization unit determines *who* gets the treatment and *how* you analyze the data. Interviewers use this topic to test if you understand **variance**, **network interference**, and the **Unit of Analysis fallacy**. Here is your definitive interview playbook.

---

### Part 1: The "Cold Call" Opening Question
**Interviewer:** *"You are testing a new checkout button color on an e-commerce app. You randomize at the User-ID level. You analyze the data at the Pageview level (because you have millions of pageviews). You get a highly significant p-value of 0.001. The PM wants to ship. What do you say?"*

**Your Instant Answer:**
**"Stop the presses. We have committed the Unit of Analysis Fallacy.** 
We randomized at the **User level** (independent observations), but we analyzed at the **Pageview level** (dependent observations). One user who visits 100 times contributes 100 data points to the treatment group. These are not independent; they are correlated within the user. By treating them as independent, we have artificially inflated our sample size, drastically narrowed our standard errors, and created a **false positive**. Our effective sample size is the number of *users*, not the number of *pageviews*. We need to aggregate the metric to the user level (e.g., Average Click-Through-Rate per User) and re-run the analysis."

---

### Part 2: The Core Framework (The 4 Standard Units)

To get a "Hire" signal, you must articulate the trade-offs of each randomization unit. Draw this table on the whiteboard:

| Randomization Unit | Definition | Best Used For | The Fatal Flaw |
| :--- | :--- | :--- | :--- |
| **User-ID (or Cookie)** | Every user gets a consistent treatment across all their sessions. | **UI changes, Feed ranking, Onboarding flows.** (Standard for 90% of tests). | **Network Interference.** If users interact with each other (social), the control user gets contaminated by the treatment user. |
| **Pageview / Session** | The treatment flips on/off each time a user visits a page. | **Low-risk changes** (e.g., testing 2 different ad placements on a travel search page). | **Carryover Effects.** A user sees Version A on Page 1, clicks 'Back', and sees Version B on Page 2. Their behavior on Page 2 is *contaminated* by their experience on Page 1. |
| **Device-ID** | The treatment applies to a specific physical device. | **Push notifications, OS-level features, Hardware changes.** | **Multiple users per device** (family iPad) dilutes the treatment effect. |
| **Cluster (Geographic / Social Graph)** | Entire cities or friend groups get the same treatment. | **Network products** (Messenger, Uber supply/demand, Marketplace liquidity). | **Massively reduced power.** You have far fewer independent clusters than you have users, so you need a huge test to detect small effects. |

---

### Part 3: The "Meta" Nuance (The Social Network Interference Problem)

Meta *loves* this question because their entire product is a social graph.

**Interviewer:** *"You are testing a new feature that encourages users to send more messages to their friends. You randomize by User-ID. The Treatment group sends 15% more messages. You ship it. Three weeks later, overall company-wide messaging drops. Why did this happen?"*

**Your Advanced Answer:**
"Because I violated the **Stable Unit Treatment Value Assumption (SUTVA)**. 

- My Treatment users sent more messages, which means **Control users received more messages** (from their Treatment friends). 
- Control users, annoyed by the spam, stopped sending messages themselves.
- When I shipped to 100% of users, the 'positive' effect of sending more messages was completely offset by the 'negative' effect of *receiving* more messages. The net effect was zero (or negative).

**My solution:** I should have run a **Cluster-Randomized Test** (randomizing by friend groups or geographic regions). The analysis metric would be **Messages Sent per Capita *within the cluster***. By keeping entire clusters in Treatment or Control, I eliminate the contamination between arms, and I measure the *true global equilibrium* effect."

---

### Part 4: The "Google" Nuance (The Cookie vs. User-ID vs. Device Hell)

Google has users logging in/out across Chrome, Android, and Search. They will push you on identity resolution.

**Interviewer:** *"We are testing a new Search feature. We randomize by Cookie. A user searches on their work laptop (Cookie A), gets Treatment. They go home, search on their personal phone (Cookie B), gets Control. They are the same human. How does this destroy our test?"*

**Your Answer:**
"This introduces **massive noise** and **dilution of the treatment effect**.

1. **The Noise:** The same human is experiencing both Treatment and Control. Their behavior on Cookie B is influenced by what they saw on Cookie A (Carryover). We are comparing two experiences of the *same person*, which violates independence.
2. **The Dilution:** If 30% of users have multiple cookies, our treatment effect is diluted by 30%. We are measuring the effect of the feature *plus* the effect of inconsistent user experience. 

**My solution:** 
- For critical tests, we randomize at the **Logged-in User-ID** level. Users who are logged out are excluded from the analysis.
- If we *must* use cookies (for logged-out search), we use a **Device-Graph** (a statistical mapping of cookies belonging to the same user) and randomize at the *Device-Graph level*, ensuring all cookies for that human get the same treatment."

---

### Part 5: The "Apple" Nuance (The Hardware/OS Constraint)

Apple cares about privacy and device performance. They will throw a practical, engineering-driven curveball.

**Interviewer:** *"We are testing a new battery optimization algorithm. We can only apply the treatment at the **Device-ID** level, and the update is pushed via an OTA (Over-the-Air) software update. But families share iPads. How do we handle the analysis?"*

**Your Answer:**
"This is a classic **Device-level randomization with User-level outcomes**.

- If a family shares an iPad (Device-ID), but we are measuring 'User Engagement' (e.g., App Store purchases), we have **multiple users per cluster**.
- We cannot treat each user on that iPad as independent because they share the same treatment environment.
- **My solution:** The analysis metric must be **aggregated to the Device-ID level** (e.g., 'Total App Store Spend per iPad per Week'). We lose the ability to segment by individual user behavior, but we maintain statistical validity.
- **Alternative:** If we absolutely need user-level metrics, we must use a **Hierarchical Model (Mixed Effects Model)** with Device-ID as a random intercept to account for the within-device correlation. But frankly, in a fast-paced interview, aggregating to the device is the safest, cleanest answer."

---

### Part 6: The "Analysis Unit MUST Equal Randomization Unit" Rule

This is the single most important sentence you will say in this interview. Memorize it.

**Interviewer:** *"Okay, but what if I randomize by User-ID but analyze by Session, and I just use 'Robust Standard Errors' or 'Clustered Standard Errors' to fix it? Is that acceptable?"*

**Your Answer:**
"Yes, **Clustered Standard Errors (clustering by User-ID)** is the statistically rigorous way to fix this without aggregating your data. 

- By clustering at the User-ID level, I tell the variance-covariance matrix that all sessions belonging to the same user are correlated (they share the same error term). 
- This widens my confidence intervals appropriately. 
- However, I must be careful: if I have 1 million sessions but only 10,000 users, clustering is *essential*; otherwise, my p-values will be wildly optimistic. 

**The Golden Rule:** The **Unit of Diversion** (how you assign treatment) and the **Unit of Analysis** (how you aggregate your standard errors) must always be the same, **OR** you must explicitly account for the hierarchy via clustering or mixed-effects models. You can never treat lower-level units (sessions) as independent when randomization happened at a higher level (users)."

---

### Part 7: The "Carryover Effect" Scenario (The Time Dimension)

**Interviewer:** *"You are testing a new recommendation algorithm on a streaming service. You randomize at the Session level. Session 1 gets Algorithm A; Session 2 gets Algorithm B. You see a huge lift in Session 2. Why might this be a false positive?"*

**Your Answer:**
"This is a **Carryover Effect**. 
What if Algorithm A (in Session 1) recommended a highly engaging movie that takes 3 hours to watch? The user is still watching that movie during Session 2. When I measure Session 2, I'm actually measuring the *residual* engagement from Session 1's recommendation. 

**My solution:** 
- If I must randomize at the Session level (to test rapid iterations), I must build in a **Washout Period** (e.g., 24 hours) between sessions. I only compare Session 1 of User A to Session 1 of User B. 
- Even better, I pre-register that I will **only analyze the first session of the day** for each user, completely eliminating the carryover from previous sessions."

---

### Part 8: The "Statistical Power" Trade-off (The Senior DS question)

**Interviewer:** *"User-ID randomization gives me clean independent observations, but it takes 4 weeks to reach power. Session-level randomization gives me 10x more observations and reaches power in 2 days. Why don't we just always use Session-level randomization?"*

**Your Answer:**
"Because **Speed is useless if the answer is wrong**. 

- Session-level randomization is incredibly noisy due to carryover and user-specific daily moods. 
- While the sample size (sessions) is huge, the **effective sample size** is much smaller because of intra-user correlation. The intra-class correlation (ICC) for sessions within a user is often 0.3 to 0.5. 
- Running a session-level test without clustering gives you a false sense of precision. When you cluster your standard errors, you'll find the 2-day test actually has very low power.
- **My Rule:** I only use Session-level randomization for **very low-risk, highly symmetrical** tests (e.g., A/B testing two different ad banners that disappear after 1 second). For any feature that changes user behavior over time, I *always* use User-ID and accept the longer duration."

---

### Part 9: Summary Cheat Sheet for the Interview

| Interviewer's Trap | Your Response Framework |
| :--- | :--- |
| *"We have 10M pageviews but only 50k users."* | "Effective sample size is **50k users**. Aggregate or cluster standard errors by User-ID." |
| *"Treatment users talk to Control users."* | "**SUTVA violation.** Need Cluster-randomization (by geography or graph)." |
| *"Same human, different cookies."* | "Stick to **Logged-in User-ID**. For logged-out, use a **Device-Graph**." |
| *"Can I randomize by Session to go faster?"* | "Yes, but only if I analyze **First Session of the Day** to remove carryover." |
| *"The PM aggregated to User level, but they randomized by Device."* | "Invalid. A shared iPad has 3 users. Analyze at the **Device level**." |
| *"How do I fix correlated data without aggregating?"* | "Use **Clustered Standard Errors** at the randomization unit (e.g., User-ID)." |
| *"What if I have a hierarchical test?"* | "**Mixed-Effects Model** with random intercepts for the randomization unit." |
