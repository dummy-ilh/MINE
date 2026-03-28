# Chapter 1: Introduction and Motivation


## The Bing Ad Headline Experiment (Motivating Example)

- A Bing engineer proposed merging the ad title with the first line of body text to create a longer headline.
- The idea was low-priority and sat in the backlog for **6+ months**.
- When finally coded and tested via an A/B experiment, a **revenue-too-high alert** fired — usually a sign of a bug (e.g., double-billing), but here the result was **legitimate**.
- Outcome: **+12% revenue** → over **$100M/year** in the US alone, with no significant harm to user-experience metrics.

### Key Takeaways from This Example

| Theme | Insight |
|---|---|
| Idea assessment | It is very hard to predict the value of any given idea |
| Small changes, big impact | Days of work → $100M/year ROI |
| Rarity of big wins | Bing runs 10,000+ experiments/year; breakthroughs happen once every few years |
| Low overhead matters | Experimentation infrastructure (ExP) made it easy to run the test scientifically |
| OEC must be holistic | Revenue alone is not enough — user experience metrics must also be tracked |

---

## Core Terminology

**A/B Test (Controlled Experiment)**
A comparison of two variants — Control (A) and Treatment (B) — where users are randomly assigned to one variant and their interactions are logged and compared.

Also called: field experiments, split tests, bucket tests, flights, randomized controlled experiments.

**Overall Evaluation Criterion (OEC)**
A single quantitative measure of the experiment's objective. Requirements:
- Must be **measurable in the short term** (duration of the experiment)
- Must **causally predict long-term strategic goals**
- Should ideally be a **single metric** (possibly a weighted combination)
- Must not be easily "gamed" (e.g., raw revenue can be gamed by adding more ads)

In statistics: also called Response Variable, Dependent Variable, Outcome, or Fitness Function.

**Parameter**
A controllable variable believed to influence the OEC. Also called a factor or variable. Parameters take on values called *levels*.

**Variant**
A specific user experience being tested. In a simple A/B test, there are two variants: Control and Treatment. The Control is the existing baseline; Treatments are the new ideas being evaluated.

**Randomization Unit**
The entity (usually a **user**) that is pseudo-randomly assigned to a variant via hashing. Requirements:
- Assignment must be **persistent** (same user → same variant across visits)
- Assignment must be **independent** (knowing one user's assignment reveals nothing about another's)

---

## Why Experiment? Correlation vs. Causality

### The Core Problem
Observational data creates misleading correlations. Two examples:

1. **Netflix churn fallacy**: Users who adopt a feature churn at half the normal rate. But this doesn't mean the feature *causes* lower churn — both outcomes may be driven by a third factor (e.g., general user engagement).
2. **Office 365 paradox**: Heavy users see more errors and crashes, yet churn less. Error messages do *not* reduce churn; **usage level** is the common cause.

> Correlation does not imply causality.

### Hierarchy of Evidence (Greenhalgh / Bailar)
From least to most trustworthy:

```
Anecdotes / Expert Opinion (HiPPO)
         ↓
Observational Studies (cohort, case-control)
         ↓
Other Controlled Experiments (non-randomized, natural)
         ↓
Randomized Controlled Experiments  ← Gold Standard
         ↓
Systematic Reviews / Meta-Analysis  ← Highest evidence
![](./images/1.PNG)
```

**HiPPO** = Highest Paid Person's Opinion — a term used to describe intuition-based decision making that experiments are meant to replace.

### Why Online Controlled Experiments Are the Best Tool
- Establish causality with high probability
- Detect **small changes** that other methods miss (high sensitivity)
- Surface **unexpected effects** — crashes, performance degradation, cannibalization of other features

---

## Necessary Ingredients for Running Useful Controlled Experiments

1. **Experimental units** that can be assigned to variants with no (or minimal) interference between groups.
2. **Enough units** — thousands minimum; more users = ability to detect smaller effects.
3. **Agreed-upon metrics / OEC** that are reliably measurable. Surrogates are acceptable when direct measurement is too hard.
4. **Easy-to-make changes** — software is ideal; server-side changes are faster to deploy and test than client-side.

---

## The Three Organizational Tenets

### Tenet 1: The Organization Wants to Make Data-Driven Decisions
- Must define an OEC measurable over short durations (1–2 weeks)
- "Profit" is a poor OEC — short-term tactics can inflate it while hurting long-term health
- **Customer Lifetime Value** is a strategically strong OEC
- *Data-informed* and *data-driven* are used interchangeably; both mean using data (not just intuition) to guide decisions

### Tenet 2: The Organization Invests in Experimentation Infrastructure
- Reliable randomization, telemetry collection, and easy software deployment are prerequisites
- Controlled experiments pair well with **Agile development**, **Customer Development**, and **MVPs**
- Getting numbers is easy; getting numbers **you can trust** is hard

### Tenet 3: The Organization Acknowledges It Is Poor at Predicting Idea Value
Data points on success rates across companies:
- **Microsoft**: ~1/3 of tested ideas actually improve the metrics they target
- **Bing / Google** (well-optimized domains): success rate ~10–20%
- **Slack**: ~30% of monetization experiments are positive
- **Netflix**: reportedly considers ~90% of ideas to be wrong
- **Quicken Loans** (5 years of testing): correct about outcomes ~33% of the time

The consistent lesson: intuition is a poor guide. Experimentation is how you find out what actually works.

---

## Improvements Are Incremental ("Inch by Inch")

Most metric gains come from many small improvements (0.1%–2%), not single breakthroughs. When an experiment only affects a subset of users, the overall impact is diluted accordingly.

### Examples of Sustained Incremental Progress
- **Google Ads (2011)**: Hundreds of experiments over a year → improved ad quality and lower average prices for advertisers
- **Bing Relevance Team**: Goal of 2% OEC improvement per year, achieved through thousands of small certified experiments
- **Bing Ads**: Grew revenue 15–25%/year through monthly "packages" of many small shipped experiments; some months were even slightly negative due to external constraints

---

## Examples of Surprising, High-Impact Experiments

### 41 Shades of Blue (Google & Bing)
- Google tested 41 color gradations for search result links
- Despite being a trivially small visual change, the winning color significantly improved user engagement
- Bing's color experiments improved task completion rates and generated **$10M+/year** in the US

### Amazon Credit Card Offer Placement (2004)
- A credit card offer on the home page had high profit but low click-through
- Moving it to the **shopping cart page** (where users have clear purchase intent) dramatically increased profit — tens of millions of dollars annually

### Amazon Personalized Cart Recommendations
- An engineer built a prototype for recommendations based on cart contents
- A senior VP opposed it, fearing it would distract users from checking out
- A controlled experiment proved it was highly valuable; the feature launched and is now standard across many e-commerce sites

### Speed / Performance (Bing)
- A JavaScript change that shortened HTML → improved performance → better user metrics
- Every **10ms of improvement** funded the fully loaded annual cost of one engineer
- By 2015, every **4ms of improvement** funded an engineer's annual cost (because revenue had grown so much)
- Amazon: a **100ms slowdown** reduced sales by 1%

### Malware Reduction (Bing)
- Freeware-injected ads polluted Bing pages with irrelevant, low-quality ads
- Experiment with 3.8M users: restricted DOM modifications to trusted sources only
- Result: all key metrics improved — sessions per user, search success rate, time-to-click, and revenue up by several million dollars; page-load time improved by hundreds of milliseconds

### Amazon Search Algorithm — "People Who Searched for X Bought Y"
- Extended the recommendation algorithm to use purchase behavior after ambiguous queries (e.g., "24" → returns the TV show, not random products)
- Despite a known weakness (results sometimes didn't contain query terms), controlled experiment showed a **3% increase in overall Amazon revenue** — hundreds of millions of dollars

---

## Strategy, Tactics, and Experiments

Controlled experiments and business strategy are **synergistic**, not competing.

### Scenario 1: Established Product + Enough Users
Experiments help **hill-climb** toward a local optimum:
- Identify high-ROI areas
- Optimize non-obvious variables (color, spacing, latency)
- Iterate toward better redesigns rather than risky full redesigns (which often fail to beat the original)
- Optimize backend algorithms (recommendations, ranking, infrastructure)

The OEC *encodes* the strategy — experiments are the feedback loop that tells you if your tactics are advancing it.

### Scenario 2: Strategy May Need a Pivot
When hill-climbing isn't enough, you may need to jump to a different region of the idea space:
- Consider **primacy effects** and **change aversion** in short-term results
- Test multiple tactics, not just one — a single failing experiment doesn't invalidate the strategy
- Use **MVPs** to cheaply explore new directions

**Bing + Social Media (failed pivot)**: After $25M+ investment and two years of experiments integrating Facebook/Twitter results, no significant metric improvement was found — the strategy was abandoned.

### Guardrail Metrics
Not everything can be traded off. Guardrail metrics define what the organization is *not* willing to sacrifice. Example: software crashes are a guardrail — no feature improvement justifies increasing crash rates.

### OEC as Strategic Integrity Mechanism
The OEC makes strategy explicit and aligns bottom-up work (features and tactics) with top-down direction (strategic goals). Without a good OEC, you may be optimizing the wrong thing entirely.

---

## Key Concepts Summary

| Term | Definition |
|---|---|
| A/B Test | Experiment comparing Control vs. Treatment via random user assignment |
| OEC | Quantitative goal metric; must be short-term measurable and predict long-term success |
| HiPPO | Highest Paid Person's Opinion; intuition-based decision-making |
| Randomization Unit | Entity (usually user) assigned to a variant; must be persistent and independent |
| Guardrail Metric | A metric the org will not allow to degrade, regardless of other gains |
| Primacy Effect | Users are accustomed to the old design; short-term data may not reflect true long-term impact |
| MVT / Multivariate Test | Experiment testing multiple parameters simultaneously |
| Multi-Armed Bandit | Experiment that dynamically reallocates traffic toward winning variants over time |

---


---

## Chapter 2

This chapter walks through a complete, concrete A/B test from start to finish: hypothesis formation, metric selection, statistical design, execution, and decision-making. The running example is a fictional e-commerce site selling widgets. The principles apply broadly — websites, desktop apps, mobile apps, game consoles, assistants, and more.

---

## Setting Up the Example

### The Business Context
The marketing team wants to send promotional emails with coupon codes. But before building a full coupon system, there is a concern: just *seeing* a coupon code field at checkout might distract users, cause them to search for codes elsewhere, and ultimately **reduce revenue** — even if there are no actual codes available.

### The Fake Door (Painted Door) Approach
Rather than building the full system upfront, the team uses a **fake door** strategy: implement only the coupon code field UI. Whatever the user enters returns "Invalid Coupon Code." The goal is to measure the *impact of the field's presence alone* on revenue, before committing to full implementation.

- **Control**: Original checkout page (no coupon field)
- **Treatment 1**: Coupon/gift code field placed below the credit card section (inline)
- **Treatment 2**: Coupon code entry as a popup triggered by a link

Testing two UI implementations simultaneously is common practice — it separates evaluation of the *idea* (adding a coupon field) from the *implementation* (how it is displayed).

### The Funnel Model
Online shopping follows a funnel (though users rarely move linearly through it):

```
Users
  ↓
Visit Homepage
  ↓
Browse / Search for Widgets
  ↓
Add Widgets to Cart
  ↓
Start Purchase Process
  ↓
Complete Purchase Process
  ↓
Revenue
```
![](./images/2.PNG)
Users skip steps, go back, or repeat. Despite this messiness, the funnel is a useful mental model for identifying *where* in the flow a change takes effect.

### Hypothesis
> "Adding a coupon code field to the checkout page will degrade revenue-per-user for users who start the purchase process."

This is the refined hypothesis — note the specificity: the metric is *revenue-per-user* (not total revenue), and the population is *users who start the purchase process* (not all visitors).

---

## Choosing the Right Metric (OEC)

### Why Not Total Revenue?
Total revenue depends on the raw number of users in each variant. Even with equal traffic allocation, random chance can cause variant sizes to differ slightly. Using total revenue would conflate the *effect of the change* with *random size differences*.

**Solution**: Normalize by actual sample size → use **revenue-per-user** as the OEC.

### Choosing the Right Denominator
Who counts as "a user" in the denominator? Three options:

| Denominator Option | Assessment |
|---|---|
| All users who visited the site | Valid, but noisy — includes users who never reached checkout and could not be affected |
| Only users who completed a purchase | **Wrong** — assumes the change only affects purchase amount, not purchase rate. If fewer users complete checkout, revenue-per-user among completers may actually look unchanged or even rise, masking a real problem |
| **Users who started the purchase process** ✓ | **Best choice** — captures all users exposed to the change, excludes unaffected users who would dilute the signal |

The general principle: include all *potentially affected* users, exclude all *definitely unaffected* users.

---

## Hypothesis Testing: Statistical Foundations

### Null Hypothesis Framework
- The **Null Hypothesis (H₀)**: The Control and Treatment have the same mean (i.e., no effect).
- We compute the **p-value**: the probability of observing a difference *at least as large as the one observed* if H₀ were true.
- If p-value < **0.05** → reject H₀ → the result is **statistically significant**.
- Equivalently: if the **95% confidence interval** for the difference does *not* include zero → statistically significant.

### Confidence Interval
For large sample sizes, the 95% CI is approximately:

```
[Δ − 1.96σ,  Δ + 1.96σ]
```
![](./images/3.PNG)
Where:
- Δ = observed difference between Treatment and Control means
- σ = standard error of that difference

The CI covers the true difference 95% of the time. If zero lies outside this interval, significance is declared.

### Statistical Power
**Statistical power** = the probability of correctly detecting a real effect when one exists (i.e., rejecting H₀ when it is actually false).

- More users → higher power
- Industry standard: design for **80–90% power**
- Power analysis is used *before* running the experiment to determine the required sample size

### Statistical vs. Practical Significance
These are two separate concepts and both matter:

| Concept | Meaning |
|---|---|
| Statistical significance | Confidence that the observed difference is not due to random chance |
| Practical significance | Whether the *size* of the difference is large enough to matter for the business |

Example thresholds:
- Large platform (Google, Bing): a **0.2%** change may be practically significant given billions of dollars at stake
- Early-stage startup: a **2%** change might still be too small; they need **10%+ improvements**
- In the chapter's example: a **≥1% change** in revenue-per-user is defined as practically significant

> A result can be statistically significant (real, not noise) but practically insignificant (too small to act on). Always define practical significance *before* running the experiment.

---

## Designing the Experiment

Four key design decisions:

### 1. Randomization Unit
**Users** — by far the most common choice. Each user is assigned persistently to one variant across all their visits. (Alternatives: pages, sessions, user-days — discussed in Chapter 14.)

### 2. Target Population
Which users should the experiment include? You can restrict by:
- Language / locale (if the change only applies to certain languages)
- Geographic region
- Platform or device type
- In this example: **all users**, analyzed among those who reach the checkout page

### 3. Experiment Size
Size (number of users) directly determines precision. Tradeoffs:

| Change | Effect on Sample Size Needed |
|---|---|
| Use purchase indicator (yes/no) instead of revenue-per-user as OEC | Smaller variance → fewer users needed |
| Raise the practical significance bar (e.g., only care about 5%+ changes) | Easier to detect → fewer users needed |
| Lower the p-value threshold (e.g., 0.01 instead of 0.05) | More certainty required → more users needed |

Other size considerations:
- **Safety**: For risky changes, start with a small fraction of users first (ramp-up), but this doesn't change the *final* target size
- **Traffic sharing**: If multiple experiments run simultaneously, each gets a smaller slice — trade off simultaneity vs. individual experiment speed
- **Overpowering** is generally fine and often recommended — it gives power to detect effects within subpopulations (e.g., users in a specific country) and across secondary metrics

### 4. Experiment Duration
Longer runtime = more users = more power, but with diminishing returns (user accumulation is sub-linear due to repeat visitors).

Factors that determine how long to run:

| Factor | Guidance |
|---|---|
| **Day-of-week effect** | User behavior differs between weekdays and weekends; run for **at least one full week** |
| **Seasonality** | Avoid periods that are anomalous (holidays, sales events) unless that's specifically what you're studying — this is *external validity* |
| **Primacy effect** | Users are accustomed to the old UI; short-term data may understate long-term impact of a new design |
| **Novelty effect** | Users may click a new feature out of curiosity, inflating short-term metrics; monitor for decay over time |
| **Variance growth** | Some metrics (e.g., cumulative session counts) have variance that grows over time, so longer doesn't always mean better — see Chapter 18 |

### Final Design for the Coupon Experiment
1. Randomization unit: **user**
2. Target: **all users**; analyze those who reach checkout
3. Power: **80% power** to detect a **≥1% change** in revenue-per-user (requires power analysis to compute exact sample size)
4. Split: **34% Control / 33% Treatment 1 / 33% Treatment 2**
5. Duration: **minimum 4 days** from power analysis; run for a **full week** to capture day-of-week effects; extend if primacy/novelty effects are detected

> Note: When the number of Treatments increases, consider making the Control group larger than each Treatment to maximize efficiency (see Chapter 18).

---

## Running the Experiment and Getting Data

Two infrastructure requirements to run any experiment:

**Instrumentation**
Logging system that records user interactions and tags each event with the experiment variant the user was assigned to. (See Chapter 13 for details.)

**Experimentation Infrastructure**
The platform that handles experiment configuration, variant assignment (randomization), and traffic allocation. (See Chapter 4 for details.)

Once data is collected, compute summary statistics and visualize results.

---

## Interpreting the Results

### Step 1: Sanity Checks via Guardrail Metrics (Invariants)
Before examining results, verify the experiment was run correctly. Guardrail metrics are metrics that should *not* change between variants. If they do change, the result is likely a bug or data pipeline error, not a real treatment effect.

Two types of invariant metrics:

| Type | Examples |
|---|---|
| **Trust-related guardrails** | Sample sizes match configured split ratios; cache-hit rates are equal across variants |
| **Organizational guardrails** | Latency — should not change just because a coupon field was added; if it does, something is wrong |

If sanity checks fail → investigate experiment design, infrastructure, or data processing before proceeding.

### Step 2: Examine Results

Results from the coupon experiment:

| Comparison | Revenue/User (Treatment) | Revenue/User (Control) | Difference | p-value | 95% CI |
|---|---|---|---|---|---|
| Treatment 1 vs. Control | $3.12 | $3.21 | −$0.09 (−2.8%) | 0.0003 | [−4.3%, −1.3%] |
| Treatment 2 vs. Control | $2.96 | $3.21 | −$0.25 (−7.8%) | 1.5e-23 | [−9.3%, −6.3%] |

Both p-values are well below 0.05 → reject H₀ → statistically significant results.

### Step 3: Interpret the Findings
- The hypothesis was confirmed: adding a coupon code field **reduces revenue**
- The mechanism: fewer users complete the purchase process (checkout abandonment increases)
- Treatment 2 (popup) is substantially worse than Treatment 1 (inline field)
- The original marketing plan's projected small revenue gain from coupon emails cannot overcome the baseline drag of just *having the field*

**Decision**: Abandon the coupon code feature entirely. The painted door approach saved significant engineering investment.

---

## From Results to Decisions

Statistical results alone do not make a decision. Context matters.

### Factors That Inform a Launch Decision

**1. Metric Tradeoffs**
What if metrics move in opposite directions? Examples:
- User engagement increases, but revenue decreases → launch?
- CPU utilization increases → does the operational cost outweigh the benefit?

There is no universal answer; it depends on organizational priorities and the OEC structure.

**2. Cost of Launching**
Two components:
- **Build cost**: Cost to finish the feature from the experimental prototype to full production. If already built, the marginal cost of launching is near zero. If the painted door was cheap but the real system is expensive, that factors heavily into the decision.
- **Maintenance cost**: New code has more bugs, is less tested, adds complexity, and makes future changes harder. High maintenance cost raises the bar for practical significance.

The higher the cost, the higher the practical significance threshold should be set.

**3. Cost of Wrong Decisions**
Not all mistakes are equal:
- If a change has a short lifespan (e.g., a promotional banner up for three days), the cost of a wrong call is low → lower the bar for statistical and practical significance
- If a change affects core infrastructure permanently, the cost of a mistake is high → raise both bars

---

## The Six Decision Cases (Figure 2.4)

Results are summarized as an observed delta (black box) with a confidence interval, compared against the practical significance boundary (dashed lines).

| Case | Statistical Significance | Practical Significance | Recommendation |
|---|---|---|---|
| **1** | Not significant | Not significant | No effect; iterate or abandon |
| **2** | Significant | Significant | Launch — easy decision |
| **3** | Significant | **Not** significant | Confident about the size; size too small to justify launch cost |
| **4** | Not significant | CI extends beyond practical boundary in both directions | Underpowered; cannot draw conclusion; run a larger follow-up experiment |
| **5** | Not significant | Likely significant (best estimate exceeds boundary) | Promising but uncertain; rerun with more power |
| **6** | Significant | Likely significant (but CI overlaps boundary) | Launching is reasonable; consider rerunning with more power for certainty |

### Key Takeaway for Ambiguous Cases
When results don't give a clean answer, be explicit about:
- What factors are being weighed
- How those factors translate to specific statistical and practical significance thresholds

Documenting this reasoning creates a consistent decision framework across experiments, not just a one-off judgment call.

---

## Key Concepts Summary

| Term | Definition |
|---|---|
| Fake Door / Painted Door | Testing user behavior toward a feature without fully building it; measures demand or impact cheaply |
| Funnel | A model of sequential user steps toward conversion; useful for identifying where a change takes effect |
| Revenue-per-user | OEC normalized by actual sample size; preferred over raw total revenue |
| Null Hypothesis (H₀) | Assumption that Control and Treatment have the same mean; rejected when p-value < 0.05 |
| p-value | Probability of observing the measured difference (or more extreme) if H₀ were true |
| 95% Confidence Interval | Range that contains the true difference 95% of the time; ≈ [Δ − 1.96σ, Δ + 1.96σ] |
| Statistical Power | Probability of detecting a real effect; typically designed at 80–90% |
| Statistical Significance | Confidence that the result is not due to chance (p < 0.05) |
| Practical Significance | Whether the magnitude of the effect is large enough to matter for the business |
| Invariant Metric | A metric that should not change between variants; used as a sanity check |
| Trust Guardrail | Invariant verifying the experiment infrastructure worked correctly (e.g., sample size ratios) |
| Organizational Guardrail | Invariant reflecting a business-level concern (e.g., latency, crash rate) |
| Day-of-week Effect | Behavioral differences between weekday and weekend users; reason to run ≥ 1 full week |
| External Validity | Whether results generalize beyond the experiment's time/context/population |
| Primacy Effect | Short-term bias from users being habituated to the old design |
| Novelty Effect | Short-term bias from users exploring a new feature out of curiosity, inflating early metrics |
| Power Analysis | Pre-experiment calculation to determine the minimum sample size needed to detect a given effect |

---


