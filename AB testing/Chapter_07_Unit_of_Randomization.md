# Chapter 7: Unit of Randomization — User vs. Session vs. Device, and Interference Risks

## 1. Definition

The unit of randomization is the entity you randomly assign to treatment or control — most commonly user, device, session, or a cluster (e.g., geographic market, marketplace listing). This choice determines what "independent observation" means for your entire analysis, and it must align with:
1. The level at which the treatment is actually experienced, and
2. The level at which the outcome metric is measured.

**Interference** (also called SUTVA violation — Stable Unit Treatment Value Assumption) occurs when one unit's treatment assignment affects another unit's outcome — breaking the independence assumption that the entire statistical framework (CLT, standard variance formulas, hypothesis testing) relies on.

## 2. Layman Explanation

Imagine testing a new "invite a friend" feature. If you randomize at the *session* level, the same user could see the feature in one session and not in another — muddying what you're even measuring, since their behavior isn't cleanly split into "always treated" vs. "always control." Randomizing at the *user* level fixes that — each user consistently gets one experience.

But now imagine the feature IS about inviting friends. If User A (treatment) invites User B (control), and User B ends up affected by that invite, you've violated the core assumption that each user's outcome only depends on their *own* treatment assignment. This is like trying to test whether a specific person exercising affects their health, but their treated gym partner keeps dragging the "control" partner to the gym too — the two aren't really independent anymore, and your comparison gets contaminated.

The right unit of randomization is the smallest, most stable "bubble" you can draw around a treatment such that what happens inside the bubble doesn't leak out and contaminate other bubbles.

## 3. Formal Explanation

**Common units and when to use them:**

- **User-level:** default choice for most product experiments — treatment persists across sessions/devices (via login), giving each user a single consistent experience. Requires stable user identification (logged-in ID, not just cookie).
- **Device-level:** used when login isn't available/reliable (logged-out flows), or when the feature itself is device-specific (e.g., a UI change only relevant to mobile). Risk: same user across multiple devices could get inconsistent experiences.
- **Session-level:** appropriate only when the treatment effect is expected to be fully contained within a single session and doesn't need consistency across visits — rare for most product features, more common for things like search ranking experiments where each query/session is independent-ish.
- **Cluster-level (geo, marketplace, school, hospital):** required when interference is expected to be severe within a cluster but negligible across clusters — e.g., testing a driver-pricing algorithm in a rideshare marketplace, where treating some drivers in a city and not others in the *same* city creates market-level spillovers, so you randomize whole cities instead.

**SUTVA (Stable Unit Treatment Value Assumption):**
Formally requires:
1. No interference: unit i's outcome depends only on unit i's own treatment assignment, not on anyone else's.
2. No hidden variations of treatment: "treatment" means the same thing for every unit that receives it.

When SUTVA is violated, naive variance/effect estimates become biased — often underestimating true variance (making you overconfident) because units you assumed were independent are actually correlated.

**Detecting interference:**
- Look for known network structure (social features, marketplaces, shared infrastructure/caching).
- Run an A/A test at the candidate unit of randomization — unexpected significant "effects" with no real treatment difference can signal interference or a broken randomization mechanism.
- Sample Ratio Mismatch (SRM) checks — verify that actual traffic split matches intended split (e.g., 50/50); large deviations often indicate a bug in randomization/bucketing logic, sometimes correlated with an underlying interference issue.

## 4. Levers — What Controls It, What Moves It

**Nature of the feature**
- Purely individual features (e.g., a UI color change) → user-level randomization is safe.
- Social/network features (referrals, shared content, messaging) → interference risk is high; consider cluster-level randomization (e.g., by friend-graph community) or specialized designs.
- Marketplace/two-sided features (pricing, matching algorithms) → supply and demand interact; consider switchback tests (same market alternates between treatment/control over time) or geo-based randomization.

**Login/identification infrastructure**
- Availability of stable, cross-device user IDs determines whether user-level randomization is even feasible; logged-out experiences often force device or cookie-level randomization, with known risks of ID churn.

**Granularity of the cluster (for cluster randomization)**
- Larger clusters (e.g., whole countries) reduce interference risk further but shrink your effective sample size dramatically (few independent clusters → wide confidence intervals, sometimes only feasible with weeks/months of data).
- Smaller clusters (e.g., zip codes) preserve more statistical power but may not fully contain the interference if network effects span cluster boundaries.

**Switchback designs (time-based cluster randomization)**
- Used heavily in marketplace/logistics experiments (Uber/Lyft/DoorDash-style): the same geographic unit alternates between treatment and control over different time windows, controlling for market-level confounds while still allowing many independent "cluster-time" observations to preserve power.

## 5. Famous Q&A (Google / Apple style)

**Q: You're testing a new "watch party" feature where users can invite friends to co-view content. Why is user-level randomization risky here?**
A: Because the feature is inherently social — if a treatment user invites a friend who's in the control group, the control user's behavior (and possibly their measured outcome, like watch time) gets influenced by their friend's treatment status. This breaks SUTVA/interference-free randomization: the control group is no longer a clean counterfactual because it's been "contaminated" by exposure to the treatment through the social graph. I'd consider randomizing at a cluster level — e.g., by friend groups or communities — so that interactions largely stay within a single treatment condition, or use a graph-cluster randomization approach that's specifically designed for social network experiments.

**Q: Your A/A test (no real treatment difference, but split traffic into two "arms" anyway) shows a statistically significant difference in a marketplace matching experiment. What would you suspect?**
A: An A/A test showing significance with no real treatment applied is a major red flag — it suggests either a bug in the randomization/bucketing mechanism (e.g., SRM — traffic isn't actually split as intended) or, in a marketplace context, interference: even though the treatment is identical for both "arms," if the marketplace is shared (same pool of drivers/listings serving both arms), then whatever's actually driving the metric may be contaminated by cross-arm interactions, or there's a system-level issue like caching or logging that differs between arms unintentionally. I'd start by checking for SRM, then dig into whether the two "arms" are actually operating on shared, interacting resources.

**Q: A rideshare company wants to test a new pricing algorithm. Why would they use a switchback design instead of standard user-level randomization?**
A: Pricing in a marketplace is inherently interference-prone — drivers and riders in the same city interact through shared supply and demand, so treating some riders with new pricing while others in the same city get old pricing distorts the market for everyone (e.g., driver availability shifts based on where treatment riders are requesting). A switchback design randomizes at the market-time level instead — the same city alternates between the old and new pricing algorithm across different time windows — which avoids splitting a single, interconnected market into two artificially separate conditions at the same time, while still generating enough independent time-blocks across markets to have statistical power.

**Q: What's the tradeoff of choosing a very large cluster size (e.g., whole countries) for a cluster-randomized experiment?**
A: Large clusters minimize interference risk since spillovers are unlikely to cross national boundaries, but they drastically shrink your effective sample size — with, say, only 20-30 countries available, your degrees of freedom for a cluster-level test are tiny, producing very wide confidence intervals and requiring a much longer test duration (weeks to months) to detect anything but large effects. It's a direct tradeoff between minimizing interference bias and preserving statistical power — the right cluster granularity depends on how far the interference actually propagates in practice, which is worth investigating (e.g., via network analysis) rather than assuming.

---
*Next: Chapter 8 — Hypothesis Formulation & Pre-Registration.*
