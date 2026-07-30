# Chapter 11: Interference and SUTVA Violations

## 1. Explanation

### Why this chapter matters more than any other for a company like Google

Every method covered so far — RCT, regression, PSM/IPW, IV, DiD, RDD, synthetic control — silently assumes **SUTVA**: my outcome depends only on *my own* treatment, not on what treatment anyone else received. This assumption is so foundational that introductory courses often don't even name it explicitly. But at the scale and interconnectedness of products like Google Search, Ads, YouTube, or Maps, it's frequently **the single biggest threat to whether a clean-looking experiment's conclusions actually hold up** at full deployment. You can run a textbook-perfect randomized A/B test, pass every check from Chapter 3, and still get a badly wrong answer to "what happens if we launch to everyone" — because the world at 100% rollout is structurally different from the world during a 50/50 test.

### Recall where SUTVA was quietly built in

Back in Chapter 1, the switching equation was written as $Y_i = D_iY_i(1) + (1-D_i)Y_i(0)$ — notice this only has *one* treatment index feeding into unit $i$'s outcome. If interference existed, you'd technically need $Y_i(D_1, D_2, ..., D_n)$ — a potential outcome depending on *everyone's* treatment assignment, not just your own. That's a vastly harder object to reason about, which is exactly why the simplifying SUTVA assumption is made by default — and exactly why it's worth stopping to ask, explicitly, whether it's actually reasonable in your setting before trusting a result.

### Working through the mechanisms of failure, one by one

**Shared, finite resources (zero-sum-ish channels).** An ad auction has a fixed number of slots at any given moment. If treatment users are shown more ads, those extra ad impressions are (at least partly) *taken from* the shared inventory pool — meaning control users end up seeing a *different* (typically worse, since some auction "wins" that would have gone to them instead went to treatment users) ad experience than they would have seen if there were no treatment group at all. The "control" arm is silently contaminated; it is not the clean, unaffected baseline the analysis assumes it to be.

**Social/network spillovers.** On any social product, treating user A can change what user A's friends or followers experience or receive — a triggered notification, newly shared content, a recommendation influenced by A's interactions — regardless of what treatment *those friends themselves* were assigned. Being in the "control" group doesn't insulate a user from a directly-connected treated friend's spillover effects.

**Marketplace/two-sided equilibrium effects.** Changing incentives for one side of a marketplace (e.g., offering riders lower prices) changes the *behavior of the other side* (e.g., drivers respond to the resulting shift in aggregate demand) — and that changed driver behavior then affects **all** riders, including those in the "control" arm, because drivers don't know or care which experimental arm any individual rider happens to be in; they simply respond to the overall incentive landscape they observe.

### Why ignoring interference produces a *biased* estimate of a full rollout — not just extra noise

This is the conceptual heart of the chapter, worth sitting with carefully. A partial (e.g., 50/50) experiment creates an artificial scarcity-or-abundance dynamic that simply will not exist once everyone is treated:

- If treatment "steals" from control (negative spillover — e.g., shared, finite ad inventory or driver supply), your 50/50 test **overstates** the effect of a full rollout. Why? Because during the test, there's a pool of untreated users/drivers to "steal" resources from — an advantage that vanishes entirely once 100% of the population is treated and there's no untreated pool left.
- If treatment "helps" control via spillover (positive spillover — e.g., viral social content, or beneficial network effects), your test **understates** the true effect of a full rollout. Why? Because your "control" group during the test wasn't a genuinely clean baseline in the first place — it received some partial, contaminated benefit from nearby treated units, making the observed treatment-control gap look smaller than the true, uncontaminated effect of universal treatment would be.

### Design-based fixes, and the specific mechanism by which each one works

- **Cluster/graph-cluster randomization**: instead of randomizing individual users, randomize whole tightly-connected groups (e.g., friend clusters, or graph partitions chosen to minimize cross-cluster edges). This works because it keeps most spillover *within* a cluster, which is entirely treatment or entirely control — so spillover mostly doesn't leak across the treatment/control boundary, preserving a cleaner comparison.
- **Geo-based / market-level randomization**: randomize by city, country, or DMA rather than by individual user. This respects the natural boundary of marketplace/auction dynamics (ad markets, ride-hailing driver supply) that genuinely operate at a geographic level, keeping the treatment and control "worlds" largely separate from each other.
- **Switchback experiments**: for a single, indivisible market (like one city's entire ride-hailing driver supply, which can't meaningfully be split into a treatment half and control half), randomize *over time* instead of across units — the whole city gets policy A on some days/hours and policy B on others. This avoids simultaneous cross-contamination between arms, though it introduces a new problem: carryover effects between adjacent time blocks (discussed below).
- **Explicit exposure/dose-response modeling**: when design-based avoidance isn't feasible, model interference directly — for instance, estimate how a control unit's outcome varies with their *degree of exposure* to treated neighbors (e.g., "what fraction of my friends are in the treatment group"), letting you separately estimate a direct effect and a spillover effect rather than lumping them together into one contaminated number.

## 2. Example

### A worked numerical showing the bias direction concretely

Suppose a rideshare company tests a price cut for riders, randomized 50/50 by individual rider (not by city or time block). For teaching purposes, assume we know the following ground truth: the city has 1,000 total available driver-hours per day, essentially fixed in the short run (drivers don't instantly materialize in response to a rider-side price change).

**What happens in the 50/50 test:** 500 riders get a discount, and — because discounted riders naturally request more rides — they end up capturing 550 driver-hours (up from their "fair," population-proportional share of 500), leaving only 450 driver-hours for the 500 control riders (down from their fair share of 500).

- **Treatment riders**: get more rides, using 550 driver-hours instead of a baseline 500 — an apparent "+50" gain, attributed entirely to the discount.
- **Control riders**: get *fewer* rides than their normal baseline (450 instead of 500), a "−50" loss — even though nothing changed about their own price. This drop is purely because treatment riders "stole" driver-hours that would otherwise have gone to them.
- **Naive per-arm comparison**: treatment (+50 relative to baseline) minus control (−50 relative to baseline) = an apparent **100 driver-hour gap** attributed to the discount.

**But at 100% rollout**, there is no "control" pool of riders left to steal driver-hours from — the *entire* market would need to grow its total supply of driver-hours to serve the higher aggregate demand from everyone simultaneously, which this test never actually examined. The true full-rollout effect depends on how much **total** driver-hour supply responds to a citywide increase in demand (a supply-elasticity question) — something the user-randomized test cannot speak to at all, since it only reallocated a fixed pie between two groups rather than testing whether the pie itself grows in response to system-wide demand.

**The fix**: run a **switchback test** instead — randomize the discount by time-block, city-wide, so that every rider in the city experiences the same policy at the same time. There's no "control" pool to steal supply from within any given time block, so the estimated effect reflects genuine demand/supply response dynamics rather than a zero-sum reshuffling of a fixed resource between two artificially separated groups.

## 3. Interview Q&A

**Q: In your own words, why does a standard 50/50 user-randomized test measure something different from "what happens if we launch to 100%" whenever there's a shared, finite resource involved?**
A: Because in the 50/50 test, treatment users can effectively draw on the *untreated* group's share of the shared resource (ad slots, driver supply, etc.) — a kind of "borrowing" that inflates the apparent per-user effect. At 100% rollout, there's no untreated pool left to borrow from; the whole system has to adjust simultaneously, which the partial test never actually tested, so the measured effect systematically overstates what a full rollout would deliver.

**Q: Name three business verticals at a company like Google/Alphabet where SUTVA violations are a first-order concern, and briefly why.**
A: (1) Ads — a finite auction inventory/set of impressions is shared across users, so treating some users' ad load changes what's available to others; (2) YouTube/social products — recommendations and shared content create direct spillovers between connected users regardless of their individual experimental assignment; (3) Maps/rideshare-adjacent products — any marketplace with a shared, capacity-constrained supply side (drivers, delivery couriers), where one side's incentives affect outcomes for everyone on the other side, not just those in a particular experimental arm.

**Q: What's the difference between a "cluster randomization" fix and a "geo-based randomization" fix, and when would you pick one over the other?**
A: Cluster randomization groups users by their *social/interaction graph* (e.g., friend groups) — appropriate when spillover flows through explicit social connections like sharing, notifications, or feed content. Geo-based randomization groups by *physical/market location* — appropriate when spillover flows through a shared local resource or marketplace (ad auctions within a region, ride-hailing driver supply within a city) that isn't primarily about social connections at all.

**Q: A switchback experiment introduces a new problem that a simple user-randomized test didn't have. What is it?**
A: Carryover/contamination across time blocks — effects from one policy period can persist into the next (e.g., driver behavior patterns or rider habits built up under policy A don't instantly reset the moment policy B begins), which can bias the switchback estimate if not accounted for — common fixes include discarding a "burn-in" period at the start of each new block, or explicitly modeling the carryover dynamics.

**Q: How would you even detect that interference is happening, before deciding to redesign the experiment?**
A: A few practical signals: (1) the aggregate/company-wide topline metric moves much less (or more) than the treatment-arm-only lift would predict if you naively scaled it up to 100%; (2) the control arm's metrics degrade over the course of the experiment in ways inconsistent with historical trends (suggesting contamination, not just noise); (3) directly test for it via a "dose" ramp — run the same experiment at different treatment-allocation shares (e.g., 10/90 vs. 50/50) and check whether the *per-user* treatment effect changes with the allocation share. If there's no interference, the per-user effect should be stable regardless of the allocation fraction; if the effect changes noticeably with allocation share, that's strong direct evidence of interference.

**Q: Suppose a positive-spillover scenario (e.g., a viral sharing feature) is tested user-randomized at 50/50. Would the observed effect over- or under-state the true full-rollout impact, and why?**
A: It would **understate** the true full-rollout impact. In the test, the "control" group isn't a genuinely clean, unaffected baseline — some control users still benefit indirectly from their treated friends' activity (shared content, notifications, etc.), which shrinks the measured treatment-control gap relative to what you'd see if *everyone* got the feature and there were no untreated group diluting the comparison at all.

**Q: An ads team ran a user-randomized experiment showing higher ad load increases revenue per treated user, but the company-wide topline ad revenue barely moved after full rollout. What likely explains this discrepancy?**
A: This is a textbook SUTVA violation via shared, finite ad auction inventory — during the partial test, treatment users' extra ad impressions were partly drawn from the shared pool at control users' expense, inflating the apparent per-user effect through reallocation rather than genuine value creation. At full rollout, there's no "control" pool left to draw from — the whole auction has to reach a new equilibrium, and much of what looked like a per-user revenue gain in the test was really just redistribution, not aggregate growth, explaining why the topline barely moved.

---
**Previous: Chapter 10 — Synthetic Control**
**Next: Chapter 12 — Sensitivity Analysis, Placebo Tests, and Quasi-Experiments**
