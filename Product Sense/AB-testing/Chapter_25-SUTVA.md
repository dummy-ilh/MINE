# 🧩 SUTVA Violations — Deep Dive
### Variants → Diagnostics → Examples → Fixes

---

## 0. The Formal Claim (so the rest of this makes sense)

In the potential outcomes framework, unit *i*'s outcome is written Y_i(**Z**), where **Z** = (Z₁, Z₂, ..., Zₙ) is the *entire vector* of treatment assignments for every unit in the experiment.

**SUTVA says:** Y_i(**Z**) = Y_i(Zᵢ) — unit *i*'s outcome depends **only on its own assignment**, not on anyone else's.

SUTVA is actually two separate assumptions bundled together:

| Component | What it says | What breaks if violated |
|---|---|---|
| **(1) No interference** | My outcome isn't affected by your treatment assignment | Comparability of control vs treatment (the thing you built this whole experiment to get) |
| **(2) No hidden treatment versions (consistency)** | "Treatment" means the same intervention for everyone assigned to it | The estimand itself becomes undefined — you're not sure what you tested |

Almost everything people call "a SUTVA violation" in tech is component (1). Component (2) is the quieter, easier-to-miss one — worth knowing separately because interviewers sometimes probe it specifically.

**Why you should care, in one line:** when SUTVA holds, ATE = E[Y|T] − E[Y|C] is an unbiased estimate of the causal effect. When it doesn't, that difference-in-means is estimating something else entirely — and you often can't tell from the number alone.

---

## 1. The Core Bias Derivation (do this math cold in an interview)

Suppose the treatment has a true **direct effect** τ on treated units. Suppose it also **spills over** onto control units with magnitude *s* (could be positive or negative).

- E[Y | Treatment] = baseline + τ
- E[Y | Control] = baseline + s  ← contaminated! Control should just be "baseline."

Your observed effect:

**Observed = E[Y|T] − E[Y|C] = (baseline + τ) − (baseline + s) = τ − s**

Three regimes, worth internalizing as numbers, not just words:

| Spillover sign | Meaning | Observed vs. true τ | Numeric example (τ = 10) |
|---|---|---|---|
| s > 0 (positive spillover — control benefits too) | Control users partially "receive" the treatment via the network/market | **Underestimate**: Observed = τ − s < τ | s = 4 → Observed = 6 (looks 40% weaker than reality) |
| s < 0 (negative spillover — crowding out) | Treating some users actively hurts others (e.g., they compete for the same fixed supply) | **Overestimate**: Observed = τ − s > τ | s = −4 → Observed = 14 (looks 40% stronger than reality) |
| s = 0 | SUTVA holds | Observed = τ | Exactly what you designed the test for |

This is the single derivation that answers 80% of SUTVA follow-up questions — "which direction is the bias," "why does dilution happen," "why is Uber's control group contaminated." Always name the *sign* of the spillover, not just that spillover exists.

---

## 2. The Five Variants

### Variant A — Social / Network Interference
**Mechanism:** A user's behavior change (from treatment) directly alters a *connected* user's environment or incentives, and that connected user might be in control.

**Diagnostics:**
- Check the **graph density** of your user base on the feature surface — if users have few in-network connections in the experiment, this risk is low.
- Compare **treatment effect by neighbor-treatment-status**: split control users into "has ≥1 treated neighbor" vs. "no treated neighbors." If these subgroups differ, you have measurable spillover.
- Run **randomization checks at the edge level** — for a sample of connected pairs, check if outcomes correlate more than baseline homophily would predict.

**Examples (2–3):**
1. Testing a new "share to friends" button. Treatment users share more → their friends (who may be in control) receive more content and engage more, contaminating the control baseline.
2. A messaging app tests a new "read receipts" feature. Treated users' contacts (control or not) now see/respond to receipts differently regardless of their own assignment.
3. LinkedIn testing a "who viewed your profile" notification — treated users view more profiles, and the *viewed* users (any assignment) get a notification, changing their behavior too.

**Fixes:**
- **Ego-network / cluster randomization:** assign entire friend groups to the same arm (Louvain or similar community detection to define clusters).
- **Geographic randomization** if the network is geographically clustered.
- If you can't cluster, **explicitly estimate the spillover** with a two-level design: vary treatment fraction *within* neighborhoods and regress on both own-treatment and neighbor-treatment-share.
- **Accept and report a different estimand**: total effect (direct + spillover) via full-graph rollout to a subset of geographies, rather than a "clean" direct effect that isn't achievable.

---

### Variant B — Marketplace / Resource Competition Interference
**Mechanism:** Treatment and control units draw from a **shared, finite resource pool** (supply, inventory, ad slots, driver capacity). Improving outcomes for treatment necessarily *reduces* what's available to control.

**Diagnostics:**
- Check whether **treatment and control assignment shares a constrained backend resource** (same city's driver pool, same warehouse inventory, same ad auction).
- Look for a **negative correlation between treatment allocation share and control-group performance** over time — as treatment % increases, does control's metric degrade?
- Run the experiment at **two different treatment/control ratios** (e.g., 50/50 vs. 90/10) — if the estimated effect size changes with the ratio, that's a strong signal of resource-based interference (a clean SUTVA-compliant effect shouldn't depend on split ratio).

**Examples (2–3):**
1. Uber tests a new dispatch algorithm giving treatment riders faster pickups by pulling from the same driver pool — control riders now wait longer than they "should," inflating the apparent treatment benefit.
2. Airbnb tests a recommendation algorithm that surfaces high-demand listings more to treatment users — those listings get booked out, so control users see fewer good options and convert less (this deflates control artificially, inflating the measured lift).
3. An e-commerce site tests a "flash sale" promo module shown only to treatment users for a limited-stock item — treatment buys it out, control users never even see it available.

**Fixes:**
- **Geographic / market-level randomization**: randomize whole cities or markets so supply/demand equilibrates within a market, not leaked across arms.
- **Switchback experiments**: alternate the entire market between treatment and control over time (e.g., by hour), so there's never a simultaneous split competing for the same resource.
- **Supply-side holdout design**: hold back a fixed slice of supply from ever being exposed to the new algorithm, and measure demand-side effects against that fixed reference.
- Analyze with **market/time fixed-effects regression**: outcome ~ treatment + market_FE + time_FE, on switchback data.

---

### Variant C — Hidden Treatment Versions (Consistency Violation)
**Mechanism:** Not everyone assigned "treatment" actually receives the *same* treatment. This is the quieter half of SUTVA — no interference between units, but the treatment itself isn't a single well-defined thing.

**Diagnostics:**
- Audit **implementation consistency across platforms/devices** — does iOS render the feature differently than Android? Is there a slow rollout where some treatment users still see cached old code?
- Check **treatment "dosage" variance** — e.g., if the treatment is "increase notification frequency," did everyone actually get the same increase, or does it depend on their existing notification settings, timezone, or opt-in status?
- Look at **exposure/trigger logs**: if 20% of "treatment" users never actually saw the change (dilution, Day 27 material), you technically have two hidden versions of "treatment" — exposed and unexposed — bundled into one label.

**Examples (2–3):**
1. A feature is A/B tested but the client app has three versions in the wild; only the newest app version can render the new UI correctly — so "treatment" secretly means "new UI" for some users and "broken/fallback UI" for others.
2. A pricing experiment sets a "10% discount" as treatment, but due to existing loyalty-tier logic, some treatment users' true effective discount stacks to 25% while others get exactly 10% — "treatment" isn't one thing.
3. A notification-frequency experiment increases send *attempts* uniformly, but actual *delivery* depends on carrier/OS-level throttling that differs by region — so realized treatment intensity varies by geography even though assignment was uniform.

**Fixes:**
- **Standardize implementation** before running the test — gate the experiment behind a minimum app version, or block users on outdated clients from being assigned at all.
- **Log and analyze actual exposure, not just assignment** (this connects directly to Intent-to-Treat vs. Complier Average Causal Effect from Day 27 — ITT is still valid here, it just answers "what if we roll this out as-is, warts and included," while CACE isolates the effect on those who got a "clean" version).
- **Narrow the treatment definition** — if you discover multiple hidden versions, consider splitting into separate, well-defined arms (e.g., "10% discount, no stacking" vs. "10% discount, stacks with loyalty") rather than one contaminated arm.
- **Pre-registration of the exact treatment spec**, including which platforms/versions are in scope, closes most of these before they start.

---

### Variant D — Temporal / Carryover Interference (within-unit, across time)
**Mechanism:** Not interference *between* units — interference **within the same unit across time**. A unit's past treatment assignment contaminates its current-period outcome in a sequential or crossover design.

**Diagnostics:**
- Compare **first-exposure vs. repeat-exposure behavior** for the same users across sequential experiments.
- Check whether a **washout period** was respected between an old experiment ending and a new one starting on the same population.
- For switchback/crossover designs specifically: test for an **order effect** — does "treatment-then-control" produce a different control-period outcome than "control-then-treatment"? If yes, treatment carried over.

**Examples (2–3):**
1. A recommendation model's weights get updated for treatment users during Experiment A; when Experiment B launches on the same user base a week later, "control" users in B were actually influenced by the model state left over from A.
2. A switchback pricing experiment alternates surge-pricing on/off by the hour in the same city; drivers who just experienced a surge hour behave differently in the very next "control" hour (they linger nearby expecting another surge) — the control hour's outcome is contaminated by the immediately preceding treatment hour.
3. A habit-forming feature (e.g., streak tracking) is tested, removed at experiment end, then a *different* feature is tested on the same cohort the following month — leftover behavioral habituation from the first feature confounds the second.

**Fixes:**
- **Washout periods** between sequential experiments on the same population, sized to the known decay time of the mechanism (model retraining cycle, habit decay, etc.).
- **Separate, non-overlapping user cohorts** for back-to-back experiments where washout isn't practical.
- **Latin square / Balaam designs** for crossover experiments with multiple treatments, which balance out order effects systematically rather than hoping they wash out.
- Explicitly **model and adjust** for the carryover term in the regression (include a lagged-treatment indicator) if you can't avoid it structurally.

---

### Variant E — Physical / Environmental Spillover (proximity-based, non-social)
**Mechanism:** Interference through shared physical or environmental space rather than a social graph or a market — units are near each other and the treatment "leaks" through the shared environment itself.

**Diagnostics:**
- Check for **treatment/control units sharing a physical space** (same office, same household, same delivery route, same store).
- Look for **outcome correlation by physical proximity** independent of any social connection — e.g., do control users in the same household as a treated user show elevated metrics?
- Compare metrics for **geographically isolated control users** (no treated units nearby) vs. **geographically adjacent control users** (treated units nearby). A gap indicates leakage.

**Examples (2–3):**
1. A grocery delivery app tests a new in-app promo banner; a treated driver on a shared delivery route ends up mentioning the promo to a control customer at the door, or the promo affects overall route timing that a control customer on the same route also experiences.
2. A household-shared streaming account: one household member is randomized into "treatment" (new UI), but other household members sharing the login (who might be logged as separate "control" profiles) see the same new UI because it's applied at the account level, not per-profile.
3. A workplace productivity tool tests a new feature on half the employees at a company; the other half (control), sitting in the same office, hear about it, ask to try it, or have their workflows changed because their *teammates'* workflows changed — even without a formal "friend" relationship in the product's social graph.

**Fixes:**
- **Randomize at the level where the physical/environmental sharing occurs** — household, account, office, or delivery-route level instead of individual-user level.
- **Explicit exclusion rules**: detect and exclude (or separately flag) users who share an account/household/office with users in the opposite arm.
- **Geo/site-level randomization** when the environment itself (a store, a warehouse, an office) is the natural unit of shared exposure.

---

## 3. Summary Table (for quick recall)

| Variant | Interference channel | Fastest diagnostic | Go-to fix |
|---|---|---|---|
| A. Social/network | Social graph edges | Effect differs by neighbor-treatment-status | Ego-network / cluster randomization |
| B. Marketplace | Shared finite resource | Effect size changes with treatment/control split ratio | Geo randomization or switchback |
| C. Hidden treatment versions | Inconsistent implementation | Exposure/dosage logs show variance within "treatment" | Standardize + log exposure; ITT vs. CACE |
| D. Temporal/carryover | Same unit, sequential time | Order effects in crossover/switchback data | Washout periods; Latin square design |
| E. Physical/environmental | Shared physical space | Outcome gap: isolated vs. adjacent control units | Randomize at household/site/account level |

---

## 4. Interview-Ready One-Liners

- **"What's the single question that tells you which variant you're facing?"**
  *"What's the shared thing connecting a treatment unit to a contaminated control unit — a social edge, a resource pool, a version string, a point in time, or a physical space?"* That shared thing tells you both the diagnostic and the fix.

- **"How do you know if ignoring interference actually matters?"**
  Run the experiment at two different treatment allocation ratios (e.g., 10% and 50% treatment). If your effect estimate is stable across ratios, SUTVA-style interference is probably not a big factor. If it shifts meaningfully, you have evidence of real spillover — this is often faster than trying to prove interference exists a priori.

- **"Direct effect vs. total effect — which do you report?"**
  Always report **both**, labeled clearly. Direct effect (user-level randomization) tells you the effect holding the rest of the world fixed. Total effect (cluster/geo randomization) tells you what happens if you actually launch it to everyone — which is usually the number that matters for the launch decision, since a full rollout *is* the "everyone else also treated" world.

---

*This is a standalone deep-dive companion to Day 18 (SUTVA) and Day 19 (Network Effects) of the 100-Day A/B Testing curriculum.*
