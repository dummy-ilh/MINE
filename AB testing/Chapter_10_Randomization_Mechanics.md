# Chapter 10: Randomization Mechanics — Hashing, Bucketing, SRM Checks

## 1. Definition

Randomization mechanics are the engineering methods used to actually assign users to experiment arms in a way that is random, consistent, and independently verifiable. The core technique is **hashing**: applying a deterministic hash function to a user identifier (plus an experiment-specific salt) to produce a pseudo-random number, which is then mapped into treatment/control **buckets**.

**Sample Ratio Mismatch (SRM)** is a diagnostic check: verifying that the actual observed split of users across arms matches the intended split (e.g., 50/50). A significant SRM is a red flag that something in the randomization or data pipeline is broken — and it invalidates any effect estimate from that experiment until resolved.

## 2. Layman Explanation

You want every user to get a "coin flip" that decides which arm they land in — but you also want the *same* user to always get the same coin flip result every time they visit (otherwise their experience keeps flip-flopping, and your analysis can't cleanly attribute outcomes to one arm). Hashing solves this: instead of an actual random coin flip each time, you compute a deterministic scramble of the user's ID that *looks* random but always gives the same output for the same input. Add an experiment-specific "salt" (like a seasoning) so that the same user gets an independent, unrelated bucket assignment in a different experiment — otherwise a user who happens to always land in "treatment" would end up in treatment for every single experiment ever run, which would be a disaster for statistical validity.

SRM checking is like weighing out ingredients before baking: if a recipe calls for a 50/50 split of flour and sugar by weight, and you actually weigh what went into the bowl and find it's 60/40, something went wrong in the process — maybe the sugar container is clogged, maybe your scale is miscalibrated. You don't proceed to bake (analyze results) until you understand why the split is off, because a broken assignment process usually corrupts everything downstream, not just the ratio.

## 3. Formal Explanation

**Hashing-based bucketing:**

bucket(user) = hash(user_id + experiment_salt) mod N

Where N is the number of buckets (commonly 100 or 1000 for fine-grained traffic allocation). Buckets are then mapped to arms (e.g., buckets 0-49 → control, 50-99 → treatment for a 50/50 split).

**Properties required of the hash function:**
- **Deterministic:** same input always produces same output (ensures a user stays in the same arm across sessions).
- **Uniform distribution:** outputs should be evenly spread across the bucket range, so the split is actually balanced.
- **Independence across experiments:** using a different salt per experiment ensures a user's bucket assignment in Experiment A is statistically independent of their assignment in Experiment B — critical for running many concurrent experiments without confounding each other.

**SRM detection (formal test):**
A chi-square goodness-of-fit test compares observed counts per arm against expected counts under the intended ratio:

χ² = Σ (Observedᵢ - Expectedᵢ)² / Expectedᵢ

If this χ² statistic exceeds the critical value for the desired significance level (SRM checks are typically run at a *stricter* threshold than normal metric tests, e.g., p < 0.001, because even a small persistent bias is a serious pipeline problem, not a subtle effect worth debating), you have a Sample Ratio Mismatch and should not trust the experiment's results until root-caused.

**Common root causes of SRM:**
- Bot/crawler traffic that disproportionately triggers one code path (e.g., only hits the control experience due to caching)
- Bucketing logic applied inconsistently across platforms (web vs. mobile) or before vs. after a redirect/filter
- Logging pipeline dropping events differently by arm (e.g., treatment arm's new code path has a subtly different logging call that fails silently more often)
- Non-uniform hash function or insufficient bucket granularity causing clumping

## 4. Levers — What Controls It, What Moves It

**Salt choice**
- A unique salt per experiment is what guarantees independence across concurrent experiments — reusing salts (or using a shared/global bucket assignment for all experiments) creates correlated assignment across experiments, undermining the validity of running them simultaneously.

**Number of buckets (N)**
- Finer granularity (e.g., 1000 buckets instead of 100) allows more precise traffic ramp control (e.g., 1% → 5% → 50% rollouts) without needing to change the underlying hash logic — just remap which buckets belong to which arm.

**Hash function quality**
- A poor-quality or non-uniform hash function can silently create SRM even with technically "correct" logic — this is why standard, well-vetted hash functions (e.g., MD5, MurmurHash) are preferred over ad hoc string manipulation.

**Where in the stack bucketing happens**
- Bucketing done too late in the request pipeline (e.g., after some filtering or redirect logic that differs by platform) risks systematically excluding certain user segments from one arm — a frequent, hard-to-spot cause of SRM.

## 5. Worked Example

Suppose you're running an experiment intended as a 50/50 split, and after a week you observe:
- Control: 49,800 users
- Treatment: 50,200 users
- Total: 100,000 users

**Expected under 50/50:** 50,000 each.

χ² = (49,800 - 50,000)² / 50,000 + (50,200 - 50,000)² / 50,000
χ² = (200²)/50,000 + (200²)/50,000
χ² = 40,000/50,000 + 40,000/50,000
χ² = 0.8 + 0.8 = 1.6

With 1 degree of freedom, the critical value at p=0.05 is 3.84. Since 1.6 < 3.84, this is NOT a significant SRM — this level of imbalance is consistent with normal random variation, and you can proceed with analysis.

Now suppose instead you saw Control: 48,000 / Treatment: 52,000:

χ² = (2,000²)/50,000 + (2,000²)/50,000 = 80 + 80 = 160

160 >> 3.84 — this is a highly significant SRM. Before trusting any lift numbers from this experiment, you'd need to investigate: check platform-level splits (is the imbalance concentrated in iOS vs. Android?), check for bot traffic, check logging completeness per arm.

## 6. Famous Q&A (Google / Apple style)

**Q: Your experiment shows a strong positive lift on the primary metric, but you also notice an SRM (52/48 split instead of 50/50). Should you trust the lift?**
A: No — an SRM invalidates the experiment's results until root-caused, regardless of how compelling the lift looks. A skewed split usually means something is systematically different about which users end up in each arm (not just noise), which means the two arms are no longer comparable — the observed "lift" could be entirely an artifact of whichever biased subpopulation ended up over-represented in one arm, rather than a real treatment effect. The right move is to pause interpretation, investigate the root cause (logging gaps, platform-specific bugs, bot traffic), and rerun or fix before drawing conclusions.

**Q: Why do you need a different salt for every experiment instead of reusing the same user-to-bucket mapping across all experiments?**
A: Reusing the same mapping means a user's treatment/control assignment would be perfectly correlated across every experiment ever run — if User X happens to land in "treatment" for Experiment 1, they'd also always land in "treatment" for Experiments 2, 3, 4, etc. This creates confounding between concurrent experiments: any effect you observe could be entangled with effects from other experiments the same users are simultaneously exposed to. A unique salt per experiment makes bucket assignments independent across experiments, which is what allows running many experiments concurrently without them contaminating each other.

**Q: An SRM check flags significance at p < 0.001, but your team argues "the actual ratio is 50.4/49.6, that's basically nothing — let's ignore it." How do you respond?**
A: I'd push back — SRM tests are typically run at a stricter threshold precisely because with large sample sizes, even a very small but *systematic* bias will trigger significance, and the concern isn't the size of the ratio but what caused it. A small, real bias (as opposed to noise) means there's a genuine, non-random reason certain users end up in one arm over another — and that same underlying mechanism could easily be correlated with the outcome metric too, silently biasing the effect estimate in ways that are hard to detect just by eyeballing the ratio. I'd want to at least identify the root cause and confirm it's unrelated to the outcome metric before dismissing it, rather than assuming a small percentage difference is automatically harmless.

**Q: You're rolling out a new feature gradually — 1% of traffic, then 10%, then 50%. How does your bucketing scheme need to support this without disrupting users already in the test?**
A: Using a fine-grained bucket range (e.g., 1000 buckets via hash(user_id + salt) mod 1000) lets you expand the treatment population by simply reassigning additional buckets to treatment (e.g., buckets 0-9 → treatment at 1%, then 0-99 → treatment at 10%), without changing which bucket any individual user falls into. Since the underlying hash-to-bucket mapping for each user never changes, everyone who was already in treatment stays in treatment as you ramp up — you're just expanding the boundary of which buckets count as "treatment," not re-randomizing anyone.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Reusing the same salt across multiple experiments → creates cross-experiment confounding
- ❌ Ignoring a statistically significant SRM because the ratio "looks close enough" → root cause first, always
- ❌ Bucketing at a point in the pipeline after platform-specific filtering/redirects → risk of systematic exclusion in one arm
- ❌ Using a low-quality/non-uniform hash function → can silently create imbalance even with "correct" logic
- ❌ Running SRM checks at the standard α=0.05 instead of a stricter threshold → SRM checks warrant more sensitivity, not less, given how serious a broken pipeline is
- ✅ Do: use fine-grained buckets (100-1000) to support gradual rollouts without re-randomizing existing users
- ✅ Do: always run an SRM check before interpreting any experiment's primary metric result

---
*Next: Chapter 11 — Novelty/Primacy Effects & Test Duration.*
