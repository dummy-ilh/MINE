# Chapter 2: Offline vs. Online Evaluation

> *"A model that looks great on your test set and fails in production is not a good model. It's a model that passed the wrong exam."*

---

## 2.1 The Core Distinction

Every ML system lives a double life:

- **Offline**: You control the data, the labels, the environment. Fast, cheap, reproducible.
- **Online**: Real users, real traffic, real consequences. Slow, expensive, ground truth.

Neither is sufficient alone. Your offline metrics tell you what *could* work. Your online metrics tell you what *does* work.

```
Offline evaluation  →  "Does this model look good on historical data?"
Online evaluation   →  "Does this model improve the product in the real world?"
```

The gap between these two is where most production ML surprises live.

**Why can't offline eval just be made good enough to replace online eval?** Because offline data is, by definition, a *record of the past under the old system*. It captures what happened when the *previous* model (or no model) was making decisions. It cannot capture how users would behave differently if shown different content, differently ranked results, or a different UI response — because that world never happened and was never logged. This is the single deepest reason the two are not substitutes for each other, and it resurfaces throughout this chapter (see §2.3, "missing counterfactuals").

---

## 2.2 Offline Evaluation

### The Train / Validation / Test Split

The foundational setup:

```
Full Dataset
├── Training set      (~70%)   — model learns from this
├── Validation set    (~15%)   — hyperparameter tuning, model selection
└── Test set          (~15%)   — final, held-out, one-time evaluation
```

**Why three splits and not two?** If you only had train and test, you'd be forced to tune hyperparameters by peeking at test performance — which means your final "test accuracy" is no longer an honest estimate of generalization, because you've implicitly fit your choices to that specific test set. The validation set exists purely to absorb that tuning pressure, keeping the test set pristine for one true, final check.

**Hard rules:**
- The test set is used **exactly once**. The moment you tune on it, it becomes a validation set.
- If you evaluate on test 10 times to pick the best model, your test error is optimistically biased.
- Splits must respect **time** for time-series data (no future leaking into past).

**Worked numerical example — why repeated test-set peeking is dangerous.**
Say you have a test set of 200 examples, and true model accuracy (on the whole population) is exactly 70%. Due to random sampling noise, a *single* evaluation on that 200-example set might report anywhere from ~64% to ~76% (roughly ±2 standard errors, where SE = sqrt(0.7×0.3/200) ≈ 3.2%). If you train 10 slightly different model variants and pick whichever one scores highest on this same test set, you're implicitly doing a "max of 10 noisy draws" — and the expected value of that maximum is higher than the true 70%, purely from luck, not skill. This is why the reported test score after repeated selection is optimistically biased, and why the rule is "test set touched once."

### Cross-Validation

When data is scarce, k-fold CV gives a more reliable estimate:

```
K-Fold (k=5):

Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]

Final score = mean ± std across folds
```

**Why average across folds instead of trusting one split?** A single train/val split gives you one noisy estimate that depends on exactly which examples happened to land in validation. K-fold effectively runs the experiment k times with different validation slices and averages the result, which both reduces the noise in the estimate and — via the standard deviation across folds — tells you *how* noisy your estimate actually is.

**Worked numerical example.** 5-fold CV accuracy scores: [0.81, 0.79, 0.84, 0.77, 0.82].

- Mean = (0.81+0.79+0.84+0.77+0.82) / 5 = **0.806**
- Standard deviation ≈ **0.025**
- Reported result: **80.6% ± 2.5%**

Compare this to a single 80/20 split that happened to report 0.84 — you'd have no way of knowing whether that 0.84 was a lucky fold or a representative one. The ± 2.5% spread tells you the true performance plausibly ranges from about 78% to 83%, which matters a lot if you're comparing two models that differ by only 1-2 points.

**When NOT to use standard k-fold:**
- Time-series data → use **TimeSeriesSplit** (always train on past, validate on future)
- Grouped data → use **GroupKFold** (keep user/patient groups intact across folds)
- Highly imbalanced → use **StratifiedKFold** (preserve class ratios per fold)

### Offline Evaluation Failure Modes

| Failure | What Went Wrong |
|---|---|
| **Data leakage** | Future information in training features |
| **Label leakage** | Target signal embedded in a feature |
| **Distribution mismatch** | Test set doesn't reflect production traffic |
| **Temporal leakage** | Random split on time-series data |
| **Duplicate contamination** | Same sample in train and test |

> **Leakage is the silent killer.** A model with 99% offline accuracy that drops to 60% in production almost always has a leakage problem. Always ask: *at inference time, would this feature actually be available?*

**Worked numerical example — quantifying a leakage-driven drop.**
A churn model reports 96% offline accuracy. Investigation reveals one feature, "account_closed_flag," was accidentally included — a flag that's only set *after* a customer churns, so it's a near-perfect leak of the label itself. Remove that feature and retrain: accuracy falls to 74%, which now matches production performance. The 22-point gap (96% → 74%) is not "the leaky feature made the model 22% better" — it's "22 percentage points of the offline score were fake, borrowed from information that doesn't exist at real prediction time."

---

## 2.3 The Offline-Online Gap

Even a perfectly clean offline evaluation can diverge from online performance. Here's why:

### Reasons the Gap Exists

**1. Distribution shift**
The data you trained and tested on is historical. Production traffic is the future. User behavior, language, and patterns shift over time.

**2. Position bias and feedback loops**
In recommender systems, users only interact with what they're shown. Your logged data reflects the *previous model's decisions*, not user preferences in the abstract. A better model might never get credit in offline eval because its recommendations were never logged.

*Concretely:* if the old model never showed Item X to anyone, your logs contain zero clicks on Item X — not because users don't want it, but because they never had the chance to see it. An offline metric computed only from logged clicks will systematically undervalue any item the previous model suppressed, which is exactly the "credit assignment" problem that makes offline recommender evaluation so tricky.

**3. Missing counterfactuals**
Offline data tells you what happened. It doesn't tell you what *would have happened* if the model had made a different decision. Online evaluation actually runs the counterfactual.

**4. System effects**
Latency, caching, UI changes, and downstream system behavior don't exist in offline eval. They exist in production.

**5. Novelty and primacy effects**
Users sometimes engage more with any change simply because it's new. Online A/B tests need to run long enough for this to wash out.

**Worked numerical example — novelty effect masking the true lift.**
A new UI feature shows +8% engagement in the first 3 days of an A/B test. By day 14, the lift has settled to +2%. If you'd stopped the test on day 3 and shipped based on that number, you'd have overestimated the long-run impact by 4x (8% vs. 2%) — purely due to users clicking on something new, not because the feature is genuinely 4x better than it turned out to be. This is the concrete reason "run duration" is listed as a key design decision below.

---

## 2.4 Online Evaluation Methods

### A/B Testing (Randomized Controlled Experiment)

The gold standard for online evaluation.

```
Incoming traffic
       │
       ▼
  [Randomize users]
       │
   ┌───┴───┐
   ▼       ▼
Control   Treatment
(Model A) (Model B)
   │       │
   └───┬───┘
       ▼
  Measure metric
  Run significance test
  Make decision
```

**Why randomization is the whole point.** Randomly assigning users to control vs. treatment is what lets you claim *causation* rather than mere *correlation*. Without randomization, any difference you see between two groups could be explained by who ended up in each group (e.g., power users self-selecting into a "try the new feature" opt-in) rather than the model itself. Randomization statistically balances every other variable — known and unknown — between the two groups, so the only systematic difference left is the thing you're testing.

**Key design decisions:**

| Decision | Guidance |
|---|---|
| Unit of randomization | User-level (not request-level) to avoid carryover effects |
| Traffic split | Start 90/10 or 95/5; go 50/50 once confident in safety |
| Run duration | At least 1–2 full weekly cycles to capture weekly patterns |
| Primary metric | Pre-register one metric before launch; avoid p-hacking |
| Guardrail metrics | Metrics that must not regress (latency, error rate, revenue) |

**Worked numerical example — is the lift real or noise?**
Control: 10,000 users, 500 conversions (5.0% conversion rate). Treatment: 10,000 users, 550 conversions (5.5% conversion rate). That's a +0.5 percentage point (10% relative) lift. Using a simple two-proportion z-test:

- Pooled proportion p̂ = (500+550)/(10000+10000) = 1050/20000 = 0.0525
- Standard error ≈ sqrt(2 × 0.0525 × 0.9475 / 10000) ≈ 0.00315
- z = (0.055 − 0.050) / 0.00315 ≈ **1.59**

A z-score of 1.59 corresponds to roughly a p-value of 0.11 — *not* below the conventional 0.05 significance threshold. Despite looking like a solid "10% relative lift" on the surface, this result is statistically consistent with noise, and you'd need a larger sample (or a longer run) before confidently shipping.

**When A/B testing is hard:**
- Small user base (underpowered tests)
- Long feedback loops (retention takes months)
- Social/network effects (contamination between control and treatment)
- Cold-start (new users have no history)

### Interleaving

Used heavily in search and recommendation. Instead of splitting users, you interleave results from two rankers and see which items users prefer.

```
Model A ranking:  [Item1_A, Item2_A, Item3_A, ...]
Model B ranking:  [Item1_B, Item2_B, Item3_B, ...]

Interleaved:      [Item1_A, Item1_B, Item2_A, Item2_B, ...]
                   (deduplicated, order balanced)

Winner = model whose items got more clicks
```

**Why interleaving needs far less traffic than A/B testing.** In a standard A/B test, half your users never see Model B at all, so any variance in *those users' general behavior* (unrelated to the model) adds noise to the comparison. In interleaving, the *same user*, in the *same session*, sees a blend of both rankers' results side by side — so you're directly comparing which specific items a specific person clicked, canceling out person-to-person variability almost entirely. This is a within-subject comparison instead of a between-subject one, which is statistically far more efficient.

**Advantage:** Much more statistically efficient than A/B — detects differences with far less traffic. Used by Netflix, Google, Spotify.

**Worked numerical example — traffic efficiency.**
Suppose detecting a genuine 2% relative improvement in ranking quality requires roughly 500,000 users in a standard A/B test (a typical order of magnitude for small effect sizes with binary outcomes), but the same effect can be reliably detected with roughly 20,000 sessions via interleaving — a >20x reduction in required traffic. This is why interleaving is preferred for ranking changes at companies where user volume for a specific surface is limited.

### Bandit-Based Evaluation (Exploration-Exploitation)

Rather than a fixed split, dynamically route more traffic to the better-performing model.

```
ε-greedy:   With probability ε → explore (random model)
            With probability 1-ε → exploit (best known model)

UCB:        Route to model with highest upper confidence bound
            on expected reward

Thompson:   Sample from posterior of each model's reward
            distribution, route to the sample winner
```

**Why bandits instead of a fixed 50/50 A/B split?** A fixed A/B test deliberately keeps sending traffic to the losing model for the entire experiment, in order to get a clean, unbiased comparison. A bandit approach instead treats "traffic sent to the worse model" as a cost (regret) to be minimized in real time — it shifts traffic toward the better-performing option *as it learns*, trading some statistical cleanliness for lower opportunity cost. This matters when the losing option is actually expensive (e.g., showing users a genuinely worse ad or recommendation costs real revenue every time it happens).

**Worked numerical example — ε-greedy regret.**
Two models: True conversion rates 5% (A) and 6% (B), but you don't know this in advance. With ε = 0.1 (explore 10% of the time), after the algorithm has correctly identified B as better, 90% of traffic goes to the 6% model and 10% is still "wasted" exploring, including some of it landing back on A. Compare to a fixed 50/50 A/B test that runs the same duration: the A/B test sends 50% of traffic to the worse model (5%) the entire time, while ε-greedy sends at most 10%×50%=5% of traffic to the worse model in the same period once it has converged — meaning roughly 45% of total traffic's worth of "lost conversions" is avoided relative to a fixed split, at the cost of a less rigorous statistical guarantee.

**Use when:** You can't afford to run a fixed A/B experiment and must learn while minimizing regret (e.g., real-time bidding, personalization).

---

## 2.5 Shadow Mode

Before you run a real A/B test, run your new model in **shadow mode**:

```
Production request
        │
        ├──► Model A (Production) ──► User sees this response
        │
        └──► Model B (Shadow)     ──► Response logged but NOT shown to user
```

**What you learn:**
- Does the model run without errors in production?
- How does latency compare?
- How different are the outputs from the current model?
- Are there edge cases you didn't catch offline?

Shadow mode is **zero-risk** — users are never affected. It's your safety check before committing to a real experiment.

**Worked numerical example.** Over 1 million shadowed requests, Model B's outputs disagree with Model A's on 4% of requests (40,000 cases), and of those disagreements, manual review finds Model B is "clearly better" on 60% and "clearly worse" on 15% (the rest ambiguous). This 60/15 split is exactly the kind of numeric signal that justifies moving forward to a real A/B test — versus, say, a 20/50 split (worse more often than better), which would send you back to the drawing board before ever exposing a real user to Model B.

---

## 2.6 Canary Deployments

A canary release sends a small slice of real traffic to the new model before a full rollout.

```
Rollout stages:

Day 1:   1% of traffic  → monitor for errors, latency, crashes
Day 3:   5% of traffic  → monitor business metrics
Day 7:   20% of traffic → run significance check
Day 14:  50% of traffic → A/B conclusion
Day 21:  100% rollout   or  rollback
```

**Canary vs. A/B test:**
- Canary is about **safety** (catching disasters early)
- A/B test is about **measurement** (detecting metric differences)
- You can do both simultaneously — canary is a subset of A/B traffic

**Why start at 1% instead of jumping straight to 50%?** The cost of a catastrophic bug scales with how much traffic is exposed to it. At 1%, a critical failure (crash loop, wrong outputs, security issue) affects a small, recoverable slice of users and is caught by monitoring before it can compound. At 50%, the same bug is a major incident. The staged ramp-up is essentially a risk-management ladder: each stage only proceeds once the previous stage proves the model is safe, so the "blast radius" of any undiscovered problem grows only as confidence grows.

**Automatic rollback triggers** (define these before launch):
- Error rate increases by > X%
- p99 latency exceeds threshold
- Guardrail metric drops by > Y%

**Worked numerical example.** Baseline error rate is 0.2%. Rollback trigger is set at "error rate increases by more than 3x baseline." During the 5% canary stage, observed error rate is 0.9% — that's 4.5x baseline, which crosses the 3x trigger, so the rollout automatically halts and reverts to Model A before reaching the 20% stage, rather than waiting for a human to notice a dashboard anomaly.

---

## 2.7 Choosing the Right Evaluation Strategy

```
Is your change safe to show to users?
    No  → Shadow mode first
    Yes ↓

Is your user base large enough for an A/B test?
    No  → Offline eval + careful canary rollout
    Yes ↓

Is feedback available quickly (< 1 week)?
    No  → Consider surrogate metrics, longer experiment window
    Yes ↓

Do you have network/social effects?
    Yes → Cluster randomization, holdout cells
    No  → Standard A/B test
```

**Why network effects break simple randomization.** If User A and User B are friends and only one of them gets the new "invite a friend" feature, their behavior isn't independent anymore — the treatment can "leak" across the randomization boundary through their interaction. This violates the core assumption behind a standard A/B test (that each unit's outcome is unaffected by which group other units are in, sometimes called SUTVA). The fix — cluster randomization — randomizes at the level of a whole friend group, city, or network cluster instead of individual users, so the leakage happens *within* a treatment arm rather than *between* arms.

---

## 2.8 The Evaluation Ladder

Think of evaluation as a ladder you climb before deploying:

```
Level 5: Long-term online metrics (retention, LTV)        ← weeks/months
Level 4: Short-term online metrics (CTR, engagement)      ← days
Level 3: A/B test / canary                                ← days
Level 2: Shadow mode                                      ← hours
Level 1: Offline test set evaluation                      ← minutes
Level 0: Validation set / cross-validation                ← seconds
```

You move up the ladder as confidence grows. You never skip levels on high-stakes systems. Each level is more expensive but more trustworthy. Notice the pattern: cost and trustworthiness increase together, and speed decreases — a validation-set check takes seconds but tells you almost nothing about production reality, while a long-term retention read takes months but is about as close to "ground truth on what actually matters" as you can get.

---

## 2.9 Logging for Online Evaluation

Online evaluation is only as good as your logging infrastructure. You need:

| Log Event | Why |
|---|---|
| Request ID + timestamp | Join events across systems |
| Model version + feature hash | Know exactly which model served the request |
| Raw features at inference time | Debug distribution shift later |
| Model output (score/ranking) | Reproduce decisions |
| User action (click, purchase, skip) | The label for online eval |
| Delay between request and action | Account for delayed feedback |

> **Don't log just outputs. Log the full decision context.** Six months from now you'll want to know exactly what the model saw when it made that prediction.

**Worked numerical example — why delayed feedback matters for logging.**
Suppose 70% of "conversion" labels arrive within 24 hours of a request, but the remaining 30% trickle in over the next 13 days (e.g., a user browses today but purchases two weeks later). If your online evaluation pipeline only waits 24 hours before computing the "true" conversion rate, you'll systematically undercount conversions by up to 30% — and if Model A and Model B have different *typical delay distributions* (e.g., one drives more "quick impulse" conversions, the other more "considered" long-delay conversions), a too-short measurement window will bias the comparison in favor of whichever model produces faster feedback, even if the slower model is ultimately just as good or better.

---

## Q&A

**Q1: Why is it fundamentally impossible for offline evaluation to fully replace online evaluation?**
A: Offline data is a historical record of what happened under a specific past system (e.g., the previous model, or a rule-based baseline). It cannot tell you what users *would have done* if shown different content or rankings — that's a counterfactual that was never observed. Only running the new model live (online) actually generates that counterfactual data.

**Q2: You see 99% offline accuracy and 60% production accuracy. What's your first hypothesis and how would you check it?**
A: Data or label leakage — a feature that wouldn't actually be available at real inference time (or that directly encodes the label) probably slipped into training. Check by auditing feature provenance: for every feature, ask "would this value exist at the moment I need to make the prediction, in production?" Removing suspect features and re-measuring offline accuracy (as in the churn example in §2.2) should make the offline number drop toward the production number if leakage was the cause.

**Q3: Why does interleaving require so much less traffic than a standard A/B test to detect the same effect size?**
A: Because interleaving is a within-subject comparison — the same user, in the same session, is exposed to both rankers, so person-to-person variability (which dominates the noise in a between-subject A/B test) is essentially cancelled out. You're directly comparing clicks on Model A's items vs. Model B's items for the same person, not comparing two different groups of people.

**Q4: An A/B test shows a big lift on day 2 that shrinks by day 10. What's happening, and what should you do?**
A: This is very likely a novelty effect — users engaging with something simply because it's new/different, not because it's genuinely better. The fix is to run the test long enough (typically 1-2+ full weekly cycles, sometimes longer) for the novelty to wash out before making a shipping decision, and to prefer the day-10+ steady-state number over the early spike.

**Q5: When would you choose a multi-armed bandit over a standard fixed-split A/B test?**
A: When the cost of sending traffic to the losing arm is high and ongoing (e.g., real-time bidding, ad serving, personalization) and you want to minimize the total "regret" of continuing to serve a known-worse option while you're still learning. The trade-off is a less clean, harder-to-interpret statistical result compared to a fixed A/B test.

**Q6: What's the difference between a canary deployment and shadow mode?**
A: Shadow mode is zero-risk — the new model's outputs are computed and logged but never shown to a real user, so it only tells you about correctness/latency/error rate, not about actual user response. A canary deployment exposes a small real slice of users to the new model's actual outputs, so it can also catch effects that only show up when users genuinely interact with the new behavior — at the cost of some real (small, bounded) risk.

**Q7: Why does network/social contamination break the validity of a standard A/B test, and what's the fix?**
A: Standard A/B testing assumes one user's outcome doesn't depend on which group *other* users were assigned to. If users interact with each other (referrals, shared feeds, messaging), a treatment given to one user can "leak" and affect a control user's behavior, violating that assumption and biasing the measured effect (usually toward zero, since the effect partially "spreads" into the control group). The fix is cluster/graph-level randomization — randomize entire friend groups, geographic regions, or network clusters together, rather than individual users.

**Q8: Why is delayed feedback a logging problem, not just an analysis problem?**
A: If a meaningful fraction of true labels (e.g., conversions) arrive well after the initial request — sometimes days or weeks later — an evaluation pipeline that closes the books too early will systematically undercount outcomes, and will especially bias comparisons between models that differ in *how quickly* they drive action (fast-impulse vs. slow-considered conversions). The logging system needs to capture enough identifying information (request ID, timestamp) to join late-arriving labels back to the original decision, and the evaluation window needs to be long enough to capture the tail of the delay distribution.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Offline eval | Fast, controlled, but can diverge from production |
| Train/val/test | Test set is sacred — use it once |
| Leakage | The #1 cause of optimistic offline metrics |
| Offline-online gap | Caused by shift, bias, feedback loops, system effects |
| A/B testing | Gold standard; pre-register metrics, respect run duration |
| Interleaving | More efficient than A/B for ranking problems |
| Shadow mode | Zero-risk production check before experiments |
| Canary | Safety-first rollout with automatic rollback triggers |
| Evaluation ladder | Climb levels; never skip on high-stakes systems |

---

## Further Reading

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (the A/B testing bible)
- Chapelle et al. — *Offline Evaluation of Contextual-Bandit-Based News Article Recommendation*
- Schölkopf et al. — *Toward Causal Representation Learning* (on counterfactuals in eval)

---

*Next: Chapter 3 — Ranking Metrics (NDCG, MAP, MRR)*
