# Chapter 14 : Variance Reduction — CUPED & Stratified Sampling

---

## 1. Start Here: What Problem Are We Even Solving?

Every A/B test is trying to detect a signal (the treatment effect) buried in noise (natural user-to-user variation). Variance reduction doesn't touch the signal — it removes noise that was never about your treatment in the first place.

```
    Your outcome metric's total variance is made of TWO pieces:

    ┌─────────────────────────────────────────────────────┐
    │   Var(Y)  =  "boring" variance          +  variance  │
    │              you could have predicted       actually │
    │              BEFORE the experiment          caused by│
    │              even started                  treatment│
    │              (e.g., some users are just     + true   │
    │              naturally heavy spenders,      noise    │
    │              always were, always will be)            │
    └─────────────────────────────────────────────────────┘
                        │
                        ▼
        If you could perfectly subtract out the "boring,
        predictable" part, whatever's LEFT is a much
        cleaner signal to test treatment against.
```

That's the whole idea. CUPED and stratified sampling are two different ways of getting at that "boring, predictable" chunk of variance and removing it — one at analysis time, one at design time.

---

## 2. The Running Analogy, Made Concrete

```
  Testing a new training program's effect on 5K time.

  WITHOUT accounting for baseline fitness:

    Group A (program):     28, 22, 31, 19, 25 min  →  wildly spread
    Group B (no program):  30, 20, 33, 18, 24 min  →  wildly spread
                             ↑
        This spread has almost NOTHING to do with the program —
        it's just "some people were already fast, some weren't."
        That spread is drowning out whatever the program actually did.


  WITH baseline fitness known (each runner's PRE-program time):

    Runner   Pre-program   Post-program   Change
    Alice    30 min        28 min         -2 min
    Bob      20 min        19 min         -1 min
    Carol    33 min        31 min         -2 min
    ...

        Now look at the CHANGE, not the raw post-program number.
        The "some people are just fast" variance is gone — because
        it was already baked into both the before AND after number,
        so subtracting it out cancels it.
```

CUPED is a slightly smarter, statistically-optimal version of "look at the change instead of the raw number" — it doesn't subtract the full baseline, it subtracts the *optimal fraction* of it (more on why in Section 4).

---

## 3. Diagnosis: Do You Even Have a Variance-Reduction Opportunity?

Before any formula, run this checklist. This is the "when to use" section.

```
                 Do you have PRE-EXPERIMENT data
                 on your users for a metric that's
                 correlated with your outcome?
                            │
              ┌───────Yes───┴───No────────────┐
              ▼                                 ▼
     CUPED is available.                Can't use CUPED directly.
     Go to Section 3a to check           Options: use a proxy covariate
     if it's WORTH using.                (e.g., signup channel) if one
                            │             correlates with outcome, or
                            │             accept no CUPED benefit for
                            │             this segment (common for
                            │             brand-new users).
                            ▼
   3a. Is that pre-period metric MEANINGFULLY
       correlated with the outcome (ballpark:
       ρ > 0.3, ideally 0.5+)?
                            │
              ┌───────Yes───┴───No────────────┐
              ▼                                 ▼
     Worth applying — meaningful           Not worth the engineering
     variance reduction ahead              effort — variance reduction
     (see the ρ² table, Section 5).        will be negligible (ρ² near 0).
                                            Look for a BETTER covariate
                                            instead, or skip CUPED here.

     ───────────────────────────────────────────────────────

                 SEPARATELY — do you have a known,
                 pre-treatment CATEGORICAL variable
                 that segments users into groups with
                 meaningfully different outcome levels?
                 (platform, geography, spend tier...)
                            │
              ┌───────Yes───┴───No────────────┐
              ▼                                 ▼
     Stratified sampling is available     No natural stratification
     — worth doing at RANDOMIZATION       variable — skip it, rely on
     time (it's nearly free to set up).   CUPED alone if available.

     Note: you can do BOTH at once — stratify at randomization
     time on the categorical variable, THEN apply CUPED at
     analysis time on a different, continuous covariate. The
     two stack as long as they're not redundant with each other.
```

---

## 4. CUPED — How to Actually Plug In the Numbers

**The formula, introduced only after you understand what each piece is doing:**

$$Y_{cuped} = Y - \theta(X - \bar X), \qquad \theta = \frac{Cov(Y,X)}{Var(X)}$$

**Read it piece by piece:**

```
  Y              = the outcome you actually care about
                    (this experiment's revenue per user)

  X               = the SAME (or related) metric, measured
                    BEFORE the experiment started
                    (last month's revenue per user)

  (X - X̄)         = "how far above/below average was THIS
                    user's baseline, compared to everyone else"

  θ               = a scaling factor: "how much of that
                    baseline difference should I subtract off?"
                    NOT always 1 — it's whatever fraction
                    minimizes the leftover variance

  Y - θ(X - X̄)    = your outcome, with the predictable
                    baseline-driven part removed
```

**Step-by-step plug-in recipe, using this chapter's numbers:**

You have: $Var(Y) = 400$, $Var(X) = 350$, $Cov(Y,X) = 280$.

```
STEP 1 — Compute θ (how much to subtract):
    θ = Cov(Y,X) / Var(X) = 280 / 350 = 0.8

STEP 2 — Compute the correlation ρ (how strong the
         relationship is — you'll need this for Step 3):
    ρ = Cov(Y,X) / √(Var(Y)×Var(X))
      = 280 / √(400×350) = 280 / 374.2 ≈ 0.748

STEP 3 — Compute the new, reduced variance:
    Var(Y_cuped) = Var(Y) × (1 - ρ²)
                 = 400 × (1 - 0.748²)
                 = 400 × (1 - 0.560)
                 = 400 × 0.440
                 = 176

STEP 4 — Translate into sample-size savings
         (Chapter 5/9 callback: required n scales
          linearly with variance for a fixed MDE):
    176 / 400 = 0.44
    → you need only ~44% of the sample size you'd
      have needed without CUPED — more than HALF the
      users, or HALF the test duration, saved for free.
```

**For an actual per-user calculation** (not just the variance formula), you'd apply Step 1's $\theta$ directly to every user's own data: if a user's revenue this month is $Y_i = 45$ and their revenue last month was $X_i = 60$ (average $\bar X = 50$), their CUPED-adjusted value is:

$$Y_{i,cuped} = 45 - 0.8 \times (60-50) = 45 - 8 = 37$$

You then run your normal t-test (Welch's, Chapter 6) comparing treatment vs. control on these $Y_{cuped}$ values instead of the raw $Y$ values — same test, cleaner input.

---

## 5. The ρ² Table — Building Intuition for "Is This Worth It?"

Since variance reduction is exactly $\rho^2$, and $\rho$ is something you can just check on a table of historical data before running anything, memorize this shape:

| Correlation ρ (pre-period vs. outcome) | Variance reduction (ρ²) | Worth it? |
|---|---|---|
| 0.1 | 1% | No — negligible, not worth the pipeline complexity |
| 0.3 | 9% | Marginal — small win, do it if it's cheap to add |
| 0.5 | 25% | Solid — meaningful sample-size savings |
| 0.7 | 49% | Strong — roughly half your required sample size, gone |
| 0.9 | 81% | Excellent — rare, usually only for very stable/sticky metrics |

**Why this matters for diagnosis**: before implementing CUPED, just compute $\rho$ between last period's metric and this period's metric on historical (non-experiment) data. If it's below ~0.3, don't bother — you're adding engineering complexity for a rounding error.

---

## 6. Why CUPED Doesn't Bias Your Result — The Part People Get Nervous About

A common worry: "wait, aren't you manipulating the data?" Here's the picture that resolves it.

```
   Timeline:

   ├──── PRE-EXPERIMENT PERIOD ────┤├──── EXPERIMENT PERIOD ────┤
   │                                ││                            │
   │   X measured HERE              ││   treatment assigned HERE  │
   │   (before treatment            ││   Y measured HERE          │
   │    even exists)                ││                            │

   Since X is measured BEFORE treatment assignment, treatment
   literally cannot have influenced X. Whatever you subtract
   using X carries ZERO information about the treatment effect —
   you're only removing "this user was already like this"
   variance, never touching the part of Y that treatment created.

   Mathematically: E[X - X̄] = 0 on average, so subtracting
   θ(X - X̄) doesn't shift the average outcome at all — it only
   tightens the spread around that average.
```

This is why CUPED is considered safe by default — the mean-preserving property isn't a side benefit, it's the entire reason it's trusted in production.

---

## 7. Stratified Sampling — How to Actually Plug In the Numbers

**Formula, same treatment — pieces explained before the equation:**

$$Var(\bar Y_{stratified}) = \sum_h \left(\frac{N_h}{N}\right)^2 \frac{Var(Y_h)}{n_h}$$

```
  Picture your population split into strata (e.g., 3 platforms):

  ┌─────────────┬─────────────┬─────────────┐
  │   iOS       │   Android   │    Web      │
  │  N₁ users   │  N₂ users   │  N₃ users   │
  │  Var(Y₁)    │  Var(Y₂)    │  Var(Y₃)    │
  └─────────────┴─────────────┴─────────────┘

  Instead of one big pooled variance across everyone (which
  includes "platform differences" as noise), you compute
  variance WITHIN each platform separately, then combine —
  weighted by how big each platform's slice of the population is.

  If iOS users are all similar to each other, and Android users
  are all similar to each other, but iOS and Android differ a LOT
  from each other, then splitting them apart removes that
  between-platform difference from your noise budget entirely.
```

**Step-by-step plug-in**, hypothetical numbers:

```
Platform    N_h (size)   Var(Y_h)   n_h (sample)
iOS         500,000      100        5,000
Android     400,000      150        4,000
Web         100,000      80         1,000
Total N = 1,000,000

STEP 1 — Compute each stratum's weight (N_h/N)²:
    iOS:     (0.5)²  = 0.25
    Android: (0.4)²  = 0.16
    Web:     (0.1)²  = 0.01

STEP 2 — Compute each stratum's contribution, weight × Var(Y_h)/n_h:
    iOS:     0.25 × (100/5000)  = 0.25 × 0.02   = 0.0050
    Android: 0.16 × (150/4000)  = 0.16 × 0.0375 = 0.0060
    Web:     0.01 × (80/1000)   = 0.01 × 0.08   = 0.0008

STEP 3 — Sum:
    Var(Ȳ_stratified) ≈ 0.0050 + 0.0060 + 0.0008 = 0.0118
```

You'd compare this to what the naive, unstratified pooled-variance estimate would have given — if platforms genuinely differ a lot in their average outcome, the stratified number comes out meaningfully smaller.

---

## 8. Diagnosis Recap: When to Use Which, in One Table

| Situation | Use | Why |
|---|---|---|
| Rich pre-experiment history exists, strongly correlated with outcome | **CUPED** | Analysis-time fix, no design changes needed, big ρ² payoff |
| Many brand-new users, no history | **Not CUPED** (for that segment) — consider a proxy covariate or accept no benefit | X doesn't exist for these users |
| A known categorical variable splits users into very different outcome levels (platform, geo, spend tier) | **Stratified sampling** | Removes between-group variance from your noise budget by design |
| Both a good pre-period covariate AND a good categorical splitter exist | **Both, combined** | They stack if not redundant — stratify at randomization, CUPED at analysis |
| Weak/no correlation between any available covariate and the outcome (ρ < ~0.3) | **Neither** | Engineering cost isn't justified by the tiny variance reduction |
| Post-treatment variable is the only "covariate" available | **Do NOT use it as X** | Using a post-treatment variable can bias your treatment effect estimate — X must be strictly pre-experiment |

---

## 9. Why NOT to Use It — The Costs and Failure Modes

This is the part most references skip. Variance reduction isn't free.

```
COST 1 — Engineering / pipeline complexity
   You need reliable, correctly-joined pre-experiment data
   per user, computed BEFORE the experiment started. If your
   data pipeline can't reliably attach the right historical
   value to the right user, you introduce bugs, not benefits.

COST 2 — Using the WRONG covariate can silently bias your result
   If X is even slightly influenced by treatment (e.g., you
   accidentally used a metric from the first day of the
   experiment instead of strictly before it), the mean-preserving
   guarantee (Section 6) breaks, and you can introduce real bias —
   not just fail to help.

COST 3 — Uneven benefit across user segments
   New users get nothing from CUPED. If a large fraction of your
   experiment population is new, the AVERAGE variance reduction
   across your whole population is much smaller than the ρ²
   table (Section 5) suggests for your "veteran user" segment
   alone — don't apply a veteran-only ρ to your whole population's
   sample-size math.

COST 4 — Weak covariates aren't worth the complexity
   ρ = 0.15 gives you ~2% variance reduction. That's not
   nothing, but it's easy to spend more engineering time
   building the pipeline than the variance reduction is worth
   in saved test duration.

COST 5 — Stratification with a poorly-chosen variable
   Stratifying by a variable barely related to the outcome
   (e.g., stratifying by signup day-of-week for a metric with
   no weekly pattern) adds design complexity for essentially
   zero variance reduction — worse than doing nothing, because
   now your analysis pipeline is more complex for no benefit.
```

**The honest summary**: skip variance reduction when (a) you don't have a good covariate, (b) your covariate might be contaminated by treatment, or (c) the correlation is too weak to justify the engineering lift. Use it eagerly when you have clean, strongly-correlated, strictly-pre-experiment data — which, at a mature company with historical user data, is very often the case for existing users.

---

## 10. Q&A

**Q: Your team is under pressure to detect smaller effects without extending test duration. How would CUPED help, and what data do you need?**
A: CUPED reduces the variance of your outcome metric by adjusting for a pre-experiment covariate correlated with that outcome — since required sample size scales directly with variance, cutting variance in half (achievable with a covariate correlated around ρ=0.7) effectively cuts required sample size in half too, letting you either detect smaller effects in the same duration or hit your existing MDE faster. The key requirement is having reliable pre-experiment data for the same or a closely related metric, ideally for most of your user base — new users without history won't benefit from this adjustment directly.

**Q: Why doesn't applying CUPED bias your estimated treatment effect?**
A: Because the covariate X is measured strictly in the pre-experiment period, before treatment assignment even happens — treatment couldn't possibly have affected X, so subtracting θ(X - X̄) doesn't remove any part of the true treatment effect, it only removes variance that was already present before the experiment started. The adjustment is mean-preserving by construction, which is exactly why it's trusted in production — but that guarantee depends entirely on X being genuinely pre-treatment, which is why using a contaminated covariate is the single riskiest mistake in this topic.

**Q: 40% of users in your experiment are brand new with no pre-experiment history. What do you do?**
A: Handle new and existing users somewhat separately — apply CUPED for existing users using their own pre-period data, and for new users either fall back to the raw outcome or use a coarser proxy covariate (signup channel, onboarding behavior) if it's meaningfully correlated with the outcome. I'd also report the treatment effect separately for new vs. existing users, since both the dynamics and the achievable variance reduction genuinely differ between the two populations — and I wouldn't apply the existing-users' ρ to the whole population's sample-size math.

**Q: How does stratified sampling differ from CUPED in terms of when each intervention happens, and can you use both together?**
A: Stratification happens at randomization time — you split the population into subgroups on a known pre-treatment characteristic and balance allocation within each subgroup, a design-time intervention. CUPED happens at analysis time — after data collection, adjusting the outcome via a pre-treatment covariate regardless of how randomization was done. You can combine both: stratify on a categorical variable (platform) at randomization, then CUPED-adjust using a continuous covariate (pre-period engagement score) at analysis — as long as the two aren't fully redundant, the variance reductions compound.

**Q: When would you actively decide NOT to use CUPED, even though pre-experiment data technically exists?**
A: Three cases: if the correlation between the available covariate and the outcome is too weak (ballpark ρ < 0.3) to justify the pipeline complexity; if there's any risk the "pre-experiment" data isn't strictly pre-treatment (contamination risk outweighs the benefit); or if the population is dominated by new users for whom the covariate doesn't exist, making the aggregate benefit much smaller than a naive ρ² calculation on veteran users alone would suggest.

---

## 11. Comprehension Check

1. Using Section 3's diagnostic flowchart, decide what you'd do for: (a) a metric with ρ=0.6 pre/post correlation for 90% of users, (b) a metric with ρ=0.15, (c) a metric where the only available covariate is measured on day 1 of the experiment itself.
2. Explain, without the formula, why CUPED doesn't bias the treatment effect estimate — use the timeline picture from Section 6.
3. Compute θ and the variance reduction for $Var(Y)=250$, $Var(X)=200$, $Cov(Y,X)=150$. Is this covariate worth using, per the Section 5 table?
4. Why can't you apply CUPED directly to brand-new users, and what are your two options for handling them?
5. What's the difference between "reduce variance at design time" and "reduce variance at analysis time," and which technique does each?
6. Name three concrete reasons you might choose NOT to use variance reduction on a given experiment, even if some pre-experiment data is technically available.
7. A colleague stratifies by signup day-of-week for a metric that shows no weekly pattern at all. What's the actual cost of this decision?

---
This is the **crown jewel** of experimentation at Microsoft, Uber, and especially **Booking.com** (where it was invented) and **Google/Meta** (where variants are heavily used). 

If the Delta Method fixes your *variance formula*, **CUPED (Controlled-experiment Using Pre-Experiment Data)** fixes your *variance itself* by using machine learning to scrub away the "noise" in your metric. 

Interviewers love this topic because it separates juniors (who know what CUPED stands for) from seniors (who know *when it breaks* and *how to implement it in a live dashboard*). Here is your definitive interview playbook.

---

### Part 1: The "Cold Call" Opening Question
**Interviewer:** *"You run an A/B test on Revenue per User. Your control group has a standard deviation of $100. You need 10,000 users to detect a 5% lift. Your PM says: 'We only have budget for 5,000 users.' What do you do?"*

**Your Instant Answer:**
**"We use CUPED (Controlled-experiment Using Pre-Experiment Data).** 
Since we have historical data on these users (their revenue from the 14 days *before* the test started), we can use that pre-experiment variable as a covariate to explain away the natural user-to-user variance. This reduces our standard deviation, effectively shrinking our sample size requirement without shrinking our user count. We can detect the same 5% lift with only 5,000 users."

---

### Part 2: The Core Intuition (The "Signal vs. Noise" Picture)

You need to explain this visually before you write a single formula.

**The Problem:** Your metric (e.g., 7-Day Revenue) has massive variance. 
- **The Signal:** The 5% lift you are trying to measure.
- **The Noise:** The fact that User A is a billionaire who spends $10,000/week, and User B is a student who spends $10/week. 

**CUPED's Solution:** It finds a **pre-experiment variable (X)** that is highly correlated with your **post-experiment metric (Y)**. 
- If I know a user spent $1,000 last week, I can *predict* they will spend ~$1,000 this week. 
- CUPED adjusts everyone's final metric by subtracting this predictable baseline. The billionaire and the student both get "shrunk" toward the average. 
- **Result:** The massive, noisy gap between the billionaire and the student disappears. The remaining data is just the *unpredictable* noise—and the *treatment effect*. Your variance drops dramatically.

> **The One-Sentence Summary:** CUPED uses the past to predict the present, subtracts that prediction from everyone, and suddenly your A/B test has 40%–80% less statistical noise.

---

### Part 3: The Formula (And How to Explain It in an Interview)

Do not just vomit the formula. Walk through it piece by piece.

**The CUPED Adjusted Metric:**

\[
Y_{cv} = Y - \theta \cdot (X - \mu_X)
\]

Where:
- **\( Y \)** = Your post-experiment metric (e.g., Revenue in the test week).
- **\( X \)** = Your pre-experiment covariate (e.g., Revenue in the 14 days before the test).
- **\( \mu_X \)** = The overall average of \( X \) across all users in the experiment.
- **\( \theta \)** = The correlation coefficient \( Cov(Y, X) / Var(X) \) (i.e., the regression slope).

**The Interview Script for this formula:**
> *"For every user, I look at their pre-experiment revenue (\( X \)). I compare it to the global average pre-experiment revenue (\( \mu_X \)). If the user was above average last week, I subtract a little bit from their post-experiment revenue (\( Y \)) to 'penalize' them. If they were below average last week, I add a little bit to their post-experiment revenue to 'boost' them. This perfectly preserves the average treatment effect (the \( \theta \) cancels out), but it dramatically shrinks the variance."*

---

### Part 4: The "Two Golden Rules" of CUPED (The Traps)

Interviewers will immediately test your practical knowledge. Memorize these two constraints:

**Rule #1: The Covariate (X) MUST be pre-experiment.**
> *"X must be measured **before** the user saw the treatment. If I use post-experiment data (e.g., Revenue from Week 1 of the test to predict Revenue from Week 2), I am using data that could have been *affected by the treatment*. This would absorb the treatment effect itself, and I would incorrectly conclude the feature did nothing."*

**Rule #2: CUPED must be applied to BOTH groups simultaneously.**
> *"I compute \( \theta \) using the **pooled** control AND treatment groups. I do NOT compute a separate \( \theta \) for control and treatment. If I do, I risk introducing bias if the treatment changes the relationship between X and Y. One global \( \theta \), applied to everyone."*

---

### Part 5: The "Meta/Google" Nuance (What if the pre-period has ZERO correlation?)

**Interviewer:** *"You are testing a brand-new feature on brand-new users who just installed the app 5 minutes ago. You have zero pre-experiment data. Can you use CUPED?"*

**Your Answer:**
"No. CUPED is useless if \( Cov(Y,X) \) is zero. 

**My workaround:** If I have no historical data on the *user*, I use **contextual covariates**:
- **Device type** (iOS vs. Android)
- **Country** (US users spend more than India users)
- **Acquisition channel** (Facebook ads users vs. Organic users)

I run a regression with these categorical covariates to predict \( Y \), get the residuals, and use those residuals as my adjusted metric. It's the same mathematical idea (reducing variance via prediction), but I use static attributes instead of time-series data. It won't give me 80% variance reduction, but it might give me 10-15%."

---

### Part 6: The "Booking.com" Nuance (The Multiple Covariates / ML Version)

Booking.com invented CUPED, and they often ask about its evolution.

**Interviewer:** *"Why stop at just one pre-experiment metric? What if I use a Machine Learning model with 50 features to predict Y, and use that prediction as X?"*

**Your Answer (The "ML-CUPED" or "CUPED-2" Approach):**
"That is actually the modern best practice. Instead of using *just* pre-experiment Revenue (\( X \)), I train a gradient-boosted model (XGBoost) using all historical features—past revenue, session counts, device, browser, day-of-week—to predict the user's post-experiment metric.

- The prediction (\( \hat{Y} \)) becomes my covariate \( X \).
- As long as this model is trained **strictly on pre-experiment data**, it is safe.
- The better the model predicts \( Y \), the higher the correlation, and the more variance I reduce.
- **The only risk:** Overfitting. I must use cross-validation to ensure the model generalizes; otherwise, I introduce noise into the covariate, which actually *increases* my variance."

---

### Part 7: The "Delta Method vs. CUPED" Distinction (A common interview double-tap)

**Interviewer:** *"We just discussed the Delta Method for Ratio Metrics. Can you use CUPED on a Ratio Metric like CTR (Clicks/Impressions)?"*

**Your Answer:**
"You can, but **only on the numerator and denominator separately**, and you have to be careful.

- I apply CUPED to **Total Clicks** (using pre-experiment clicks as the covariate).
- I apply CUPED to **Total Impressions** (using pre-experiment impressions as the covariate).
- *Then* I take the ratio of these two CUPED-adjusted totals.

**Crucially:** You cannot just apply CUPED to the pre-calculated ratio (CTR) itself, because CTR is already a division. You must apply CUPED *before* the division. After I have the CUPED-adjusted totals, I use the **Delta Method** to compute the variance of the adjusted ratio. They are complementary tools, not substitutes."

---

### Part 8: The "Stakeholder" Question (How much power did I actually gain?)

**Interviewer:** *"You run CUPED. The variance dropped by 50%. How much smaller can my sample size be?"*

**Your Answer:**
"Because sample size scales with **variance squared**, if I reduce variance by half (0.5x), the required sample size drops by \( 0.5^2 = 0.25 \). So I only need **25%** of my original sample size to maintain the same statistical power.

**Alternatively:** If I keep my original 10,000 users, my Minimum Detectable Effect (MDE) shrinks by half. If I previously could only detect a 5% lift, I can now reliably detect a **2.5% lift** with the same sample size. This is huge for product teams with small user bases."

---

### Part 9: The "A/A Test" Failure Scenario (The Ultimate Senior Check)

**Interviewer:** *"We implemented CUPED. Our A/A tests (where there is no real treatment) are showing a p-value of 0.01—too many false positives. What went wrong?"*

**Your Answer (The "Look-Ahead" Bias Trap):**
"This is a classic sign that your \( X \) variable is **not purely pre-experiment**.

1. Did you compute \( X \) using data that includes the *same day* as the experiment started? If yes, the treatment might have affected Day 1 data, which bleeds into your covariate, creating a false correlation.
2. **The Fix:** My pre-experiment period **must end at least 1 day before the experiment starts** (a "gap" period). I use days [-14, -2] as my covariate period, and Day 0 as the start of the test. This ensures zero leakage.
3. Additionally, I check the correlation between \( X \) and \( Y \) in the **Control group only**. If the correlation is artificially high in the treatment group due to the treatment, my pooled \( \theta \) will be wrong. The A/A test failing is the canary in the coal mine."

---

### Part 10: Summary Cheat Sheet for the Interview

| Interviewer's Trap | Your Response Framework |
| :--- | :--- |
| *"What does CUPED stand for?"* | "Controlled-experiment Using Pre-Experiment Data." |
| *"How does it work in one sentence?"* | "Use pre-test data to predict the post-test metric, subtract that prediction, and shrink the variance." |
| *"When can I NOT use it?"* | "When I have zero pre-experiment data (brand new users) or zero correlation with the post-experiment metric." |
| *"Can I use multiple covariates?"* | "Yes, train an ML model on pre-experiment data and use the prediction as the single covariate." |
| *"Does CUPED change the average treatment effect?"* | "No. The \( \theta \cdot (X - \mu_X) \) term sums to zero across all users, so the average lift remains identical." |
| *"What happens if I use post-experiment data?"* | "I absorb the treatment effect and destroy my test (Look-Ahead Bias)." |
| *"What's the math trade-off?"* | "Variance reduction of \( 1 - \rho^2 \). If correlation (\( \rho \)) is 0.8, variance drops by 36%." |
| *"How do I combine with Delta Method?"* | "Apply CUPED to the *numerator* and *denominator* individually *before* taking the ratio and applying Delta." |
