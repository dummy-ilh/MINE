# Chapter 17: Network Effects / Interference — Cluster Randomization & Switchback Tests (Quantitative Design)

## 1. Definition

Building on Chapter 7's introduction to interference and SUTVA violations, this chapter covers the quantitative machinery for *designing and sizing* cluster-randomized and switchback experiments correctly:

- **Intra-cluster correlation (ICC, denoted ρ):** a measure of how similar units within the same cluster are to each other relative to units in different clusters, on the outcome metric. Higher ICC means units within a cluster behave more alike (e.g., all drivers in the same city respond similarly to a pricing change), which reduces the effective independent information each additional unit within a cluster provides.
- **Design effect (DEFF):** a multiplier that quantifies how much larger your effective variance is under cluster randomization compared to simple random sampling of individuals, driven by ICC and cluster size.

## 2. Layman Explanation

Imagine you want to know if a new pricing algorithm changes rider behavior, but you can only randomize whole cities (not individual riders) because of marketplace interference (Chapter 7). If you have 10,000 riders spread across just 10 cities, you might think you have 10,000 independent data points — but you don't. Riders within the same city tend to behave similarly (same local market conditions, same driver supply, same local competition), so a lot of those 10,000 observations are somewhat redundant with each other.

The design effect tells you how much "real" independent information you actually have, given that redundancy. If riders within a city are highly similar to each other (high ICC), your 10,000 riders across 10 cities might carry only as much statistical information as, say, 500 truly independent observations — even though you collected data on 10,000 people. This is why cluster-randomized experiments often need way more total units than a simple calculation would suggest — you're really constrained by the *number of clusters*, not the number of individuals inside them.

## 3. Formal Explanation

**Design effect formula:**

DEFF = 1 + (m - 1) × ρ

Where:
- m = average cluster size (units per cluster)
- ρ = intra-cluster correlation coefficient

**Effective sample size:**

n_effective = n_total / DEFF

This effective sample size — not the raw total unit count — is what should be plugged into standard power/sample-size formulas (Chapter 9).

**Interpretation of ρ:**
- ρ = 0: no within-cluster correlation; clusters provide no redundancy, and DEFF = 1 (cluster randomization behaves just like individual randomization in terms of statistical efficiency — though you'd rarely see ρ truly at 0 if you needed cluster randomization to address real interference in the first place).
- ρ = 1: units within a cluster are perfectly correlated (redundant); DEFF = m, meaning your effective sample size collapses to just the *number of clusters*, regardless of how many individuals are inside each one.

**Required sample size under clustering:**

n_total (individuals needed) = n_simple_random × DEFF

where n_simple_random is what standard power calculations (Chapter 9) would say you need under individual-level randomization. This means cluster randomization always requires *at least* as many total units as individual randomization for the same power — often substantially more, depending on ICC and cluster size.

**Switchback test design — a related quantitative concern:**
In a switchback test (same cluster alternates between treatment/control over time), the analogous concern is autocorrelation over time within a cluster — if treatment effects or baseline conditions carry over between adjacent time periods (a "carryover effect"), adjacent time-blocks aren't fully independent either, and a similar effective-sample-size discounting applies. Typical mitigation: include a "burn-in" or buffer period when switching conditions, discarding data right at the transition to let carryover effects dissipate before counting a time-block as clean treatment or control data.

## 4. Levers — What Controls It, What Moves It

**Number of clusters vs. cluster size**
- For a fixed total number of units, having MORE, SMALLER clusters is statistically more efficient than FEWER, LARGER clusters — because DEFF grows with cluster size (m), while the fundamental information really scales with the number of independent clusters. This means, where feasible, preferring more granular clusters (e.g., zip codes instead of entire states) reduces required total sample size, provided the smaller clusters still adequately contain the interference.

**Magnitude of ICC**
- Higher ICC (units within clusters are very similar) inflates DEFF sharply, especially as cluster size grows — this is often the case for tightly-coupled marketplaces (e.g., a single city's rideshare supply/demand pool) where local conditions dominate individual variation.
- ICC can sometimes be reduced by choosing a metric or cluster definition that captures less of the shared local variation, though this is often constrained by the nature of the interference itself.

**Duration and number of switchback periods**
- More, shorter switchback periods (with appropriate buffer/burn-in) generally provide more independent "looks" than fewer, longer periods — but too-short periods risk not fully capturing steady-state treatment effects (echoing the novelty/primacy concerns from Chapter 11) and increase the relative cost of buffer/burn-in time.

## 5. Worked Example

Suppose you're planning a cluster-randomized test across cities, where a standard (non-clustered) power calculation says you'd need n = 2,000 riders total for adequate power. You plan to run this across 20 cities, with an average of 500 riders per city (m = 500), and historical data suggests ICC (ρ) for this outcome metric is 0.02 (a seemingly small number).

DEFF = 1 + (m - 1) × ρ = 1 + (500 - 1) × 0.02 = 1 + 499 × 0.02 = 1 + 9.98 = 10.98

Required total individuals under clustering: n_total = n_simple_random × DEFF = 2,000 × 10.98 ≈ 21,960

Even with a seemingly tiny ICC of 0.02, the design effect nearly **11x's** your required sample size, because the cluster size (500) is large enough to compound even a small per-unit correlation substantially. This is the core lesson of cluster randomization: even weak within-cluster correlation, combined with large cluster sizes, can dramatically inflate the number of individual units you need — which is why the real constraint often becomes "how many independent clusters (cities) can we get," not "how many total riders."

If instead you had only 5 cities available (with the same 500 riders per city, same ICC), your total available units would be 2,500 — far short of the ~21,960 needed. In that scenario, no amount of individual-level data collection fixes the problem; you'd need either more clusters (more cities), a design that reduces ICC, or acceptance of a much larger MDE (detecting only bigger effects) to make the test feasible.

## 6. Famous Q&A (Google / Apple style)

**Q: Your marketplace test needs cluster randomization by city due to interference concerns. A colleague says "we have 50,000 riders across 10 cities, that's plenty of sample size." What's the flaw in this reasoning?**
A: Raw rider count is misleading under cluster randomization — what matters is the *effective* sample size after accounting for the design effect, which depends on both cluster size (5,000 riders/city here) and the intra-cluster correlation. If riders within the same city behave similarly (plausible, since they share local market conditions), the design effect could be substantial, and the effective sample size might be dramatically smaller than 50,000 — potentially closer to what 10 independent "super-observations" would provide, if ICC is high enough. I'd want to estimate ICC from historical data and compute DEFF before concluding the sample size is sufficient, rather than trusting the raw rider count.

**Q: Why does having more, smaller clusters tend to be more statistically efficient than fewer, larger clusters, for the same total number of units?**
A: Because the design effect formula (DEFF = 1 + (m-1)×ρ) shows that variance inflation grows with cluster size m — larger clusters mean more redundant, correlated observations packed into each cluster, while what actually drives independent information is closer to the *number* of clusters, not the total unit count. Spreading the same total units across more, smaller clusters reduces m, which directly reduces DEFF and increases your effective sample size for the same total data collected — as long as the smaller cluster size is still large enough to adequately contain whatever interference motivated cluster randomization in the first place.

**Q: An ICC of 0.02 sounds negligible. Why did it inflate the required sample size by roughly 11x in a cluster of 500?**
A: Because the design effect formula multiplies ICC by (cluster size - 1), not just ICC alone — even a small per-unit correlation compounds substantially when there are hundreds of units sharing that correlation within each cluster. This is a common interview trap: people intuitively dismiss a small ICC as unimportant without realizing its impact scales with cluster size, and large clusters (which are often exactly what you'd want operationally, e.g., big cities) are precisely where this compounding effect bites hardest.

**Q: You only have 5 cities available for a marketplace pricing test, but your DEFF-adjusted sample size calculation says you'd need the equivalent of 20+ independent clusters for adequate power. What are your options?**
A: A few honest options: (1) accept a much larger MDE — only commit to detecting bigger effects, since a small number of clusters simply can't provide enough independent information for fine-grained detection; (2) explore whether a finer-grained cluster definition (e.g., neighborhoods within each city rather than whole cities) can still adequately contain the interference while providing more independent clusters; (3) extend the test duration and use a switchback design within each of the 5 cities, trading spatial clusters for temporal ones to gain more independent "looks" over time; or (4) if none of these are feasible, be transparent with stakeholders that the test as designed is underpowered, rather than running it and either over-interpreting a null result or getting a lucky "significant" result that doesn't actually reflect adequate statistical rigor.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Using raw individual count (not effective sample size) when planning power for a cluster-randomized test
- ❌ Assuming a small ICC (e.g., 0.02) is automatically negligible without checking how it compounds with cluster size via the design effect
- ❌ Preferring fewer, larger clusters for "convenience" without recognizing the statistical efficiency cost
- ❌ Ignoring carryover/autocorrelation effects between adjacent switchback periods — always include a buffer/burn-in when transitioning conditions
- ✅ Do: estimate ICC from historical data before finalizing a cluster-randomized design
- ✅ Do: remember DEFF = 1 + (m-1)ρ, and that required units scale by this multiplier relative to a simple random sample calculation

---
*Next: Chapter 18 — Heterogeneous Treatment Effects (Segment-Level Analysis, Simpson's Paradox Traps).*
