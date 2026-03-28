

# Chapter 10 & 11 — *Trustworthy Online Controlled Experiments* Study Notes

---

## Chapter 10: Complementary Techniques

### Why This Chapter Matters

A/B experiments don't exist in a vacuum. To run successful experiments you also need:

- **An ideas funnel** — sources of things worth testing
- **Validated metrics** — so you're measuring what actually matters
- **Evidence** — for situations where a controlled experiment is impossible or insufficient
- **Complementary metrics** — that augment what controlled experiments measure

For cheap ideas, just run an experiment directly. For expensive ones, use complementary techniques to prune and evaluate first.

---

### The Two-Axis Framework (Figure 10.1)

All complementary techniques sit somewhere on a 2D plane:

| Axis | Meaning |
|---|---|
| X-axis | Number of users (scale) |
| Y-axis | Depth of information per user |

**Positioning of each technique:**

- **UER (User Experience Research)** — few users, very deep
- **Focus Groups** — few users, moderately deep
- **Surveys** — more users, moderate depth
- **Logs-based Analysis** — massive scale, shallow depth
- **Human Evaluation** — uncertain scale, high depth (marked with `?`)
- **External Data** — uncertain scale, uncertain depth (marked with `?`)

The fundamental tradeoff: **the more users you can reach, the less you know about each one.** Generalizability (external validity) increases with scale, but richness of insight decreases.

---

### Logs-based Analysis (Retrospective Analysis)

This is the foundation. Before you can run good experiments, you need proper **instrumentation** — logging of user views, actions, and interactions. The same infrastructure powers retrospective analysis.

#### What logs-based analysis helps with:

**1. Building Intuition**
Answer fundamental product questions:
- What is the distribution of sessions-per-user or click-through rate?
- How do distributions differ by segment (country, platform)?
- How do distributions shift over time?
- How are users growing over time?

This gives you a **product baseline**: what's the variance, what's happening organically independent of experiments, what size of change would be practically significant.

**2. Characterizing Potential Metrics**
Before committing to a metric, understand:
- Its variance and distribution
- How it correlates with existing metrics
- Whether it performed well on *past* experiments
- Whether it provides genuinely new information over existing metrics

**3. Generating Experiment Ideas**
- Examine conversion rates at each funnel step to find large drop-offs (McClure 2007 — the "AARRR" pirate funnel)
- Sessionize data to find action sequences that take longer than expected
- Use discovery to motivate new feature ideas or UI changes

**4. Upper-Bound Sizing**
Before investing engineering effort, get a rough ceiling on potential impact. Example: before making email attachments easier to use, analyze how many attachments are currently being sent. If the number is tiny, maybe it's not worth the investment.

**5. Natural Experiments**
These occur when something changes in the world — exogenous events (a competitor changes a default) or bugs (a bug logs all users out). Use observational analysis (Chapter 11) to measure the effect.

**6. Observational Causal Studies**
When experiments are impossible, quasi-experimental designs can be used. Combined with experiments, they can improve inference of more general results.

#### Key Limitation of Logs-based Analysis
It can only infer the future from the past. Example: low usage of the email attachment feature *appears* to suggest low demand. But the real cause might be that the feature is hard to use — something the logs alone won't reveal. Combining logs with user research gives a more complete picture.

---

### Human Evaluation

A company pays a **human rater (judge)** to complete a task. Common in search and recommendation systems.

#### Types of tasks:
- Simple preference: "Do you prefer side A or side B?"
- Binary classification: "Is this image pornographic?"
- Labeling: "Please label this image."
- Relevance rating: "How relevant is this result for this query?"

More complex tasks require detailed guidelines to **calibrate ratings**. Multiple raters are assigned the same task to handle disagreement — various voting or aggregation mechanisms are used to produce high-quality labels.

**Quality matters:** Data from crowd platforms like Mechanical Turk varies in quality depending on incentives and payment. Quality control and disagreement resolution are essential (Buhrmester, Kwang and Gosling 2011).

#### Key Limitation:
Raters are **not your actual end users**. They do assigned tasks in bulk; real users encounter your product organically. Raters also miss **local context** — example: the query "5/3" looks like arithmetic to most raters (answer: 1.667), but users near Fifth Third Bank are searching for the bank. This is why evaluating personalized recommendation algorithms via human rating is so hard.

Flip side: raters *can* be trained to detect spam or harmful content that real users might not even notice.

#### Uses in A/B experiments:
- Use human ratings as **additional metrics** alongside online experiment results
- Run **side-by-side experiments**: raters see Control and Treatment results simultaneously and judge which is better
- Bing and Google run scaled human evaluation fast enough to use *alongside* online controlled experiments when deciding whether to launch a change
- **Debugging**: examine poor-rated results to understand *why* an algorithm returned them
- **Pair with logs**: understand which user actions correlate with highly-rated results for a query

---

### User Experience Research (UER)

UER uses **field and lab studies** that go deep with a *small* number of users (typically fewer than tens of users). Researchers observe users doing tasks of interest and ask questions — either in a lab or in the actual environment where the product is used (in situ).

#### What UER is good for:
- **Generating ideas** — observe where users struggle during a purchase flow: Is it slow? Are they hunting for coupon codes?
- **Spotting problems** — catch issues that instrumentation won't surface
- **Gaining insights** — direct observation + ability to ask "why" in real time

#### Special techniques within UER:
- **Eye-tracking equipment** — data you cannot get from standard instrumentation
- **Diary studies** — users self-document behavior longitudinally; captures intent and offline activities that instrumentation can't reach

#### Important constraint:
Ideas generated from UER *must be validated* using methods that scale to more users — observational analyses and controlled experiments. UER generates hypotheses; it doesn't confirm them.

---

### Focus Groups

Guided group discussions with recruited users or potential users. Topics can range from:
- Open-ended attitude questions ("What do your peers typically do or discuss?")
- Specific feedback using screenshots or demo walkthroughs

#### Advantages over UER:
- More scalable than UER
- Can handle similarly open-ended, ambiguous questions
- Good for emotional reactions, branding, marketing changes
- Useful early in design when hypotheses are still ill-formed

#### Key weakness — **group-think**:
The group format means less ground is covered than in a 1-on-1 UER study. Opinions tend to converge. What people *say* in a focus group may not match what they actually *do*.

**Classic example — Philips Electronics boom box study:**
Teenage participants in a focus group expressed a strong preference for yellow boom boxes and dismissed black ones as "conservative." When the session ended and they were offered a free boom box to take home, most chose black. Stated preferences ≠ revealed preferences.

---

### Surveys

Surveys recruit a population to answer questions — multiple choice or open-ended, conducted in-person, by phone, or online (embedded in-product or via platforms like Google Surveys).

Examples:
- Windows OS prompts users with 1–2 short questions about the OS and other Microsoft products
- Google ties quick questions to specific in-product experiences (Mueller and Sedley 2014)

#### Challenges in survey design (Marsden and Wright 2010, Groves et al. 2009):

**1. Question wording is fragile**
- Questions can be misinterpreted
- Phrasing can prime respondents toward certain answers
- Question order can change how respondents answer later questions
- If you change the survey over time, historical comparisons become invalid

**2. Self-reported answers are unreliable**
Users may give incomplete or untruthful answers, even in anonymous surveys.

**3. Response bias is pervasive**
Who responds is not random — e.g., only unhappy users may bother responding. The population of respondents may not represent your actual user population. Because of this, **relative results** (comparing period A to period B) are often more meaningful than absolute results.

#### What surveys are actually useful for:
- Reaching more users than UER or focus groups can
- Getting answers to things you **cannot observe from instrumentation**: what happens offline, user intent, trust, satisfaction levels
- Observing **trends over time** on harder-to-measure things like reputation or trust
- Correlating with aggregate business metrics to guide broad investment areas (not specific ideas — use UER for that)

Surveys are **almost never directly comparable** to instrumented data due to the biases above.

---

### External Data

Data collected and analyzed by a party outside your company. Sources include:

- **Panel-based companies** — recruit large panels of users who agree to full online behavior tracking. Provides per-site granular data (number of visitors, detailed behavior). Question: are panel users representative?
- **Per-user data companies** — provide user segments that can be joined with your logs-based data
- **Survey companies** — run custom surveys on your behalf or publish their own research
- **Academic papers** — e.g., papers comparing eye-tracking (what users look at) with click behavior on search engines (Joachims et al. 2005) help you understand how representative your click data is
- **Published lessons / UI pattern libraries** — crowd-sourced design wisdom (Linowski 2018b)
- **Competitive data** — how your metrics compare to industry benchmarks

#### How to use external data:

**Validation:** Compare your internal computed metrics to external benchmarks (e.g., comScore, Hitwise). Absolute numbers rarely match exactly; instead look at **time series alignment** — do the trends and seasonal patterns align?

**Hierarchy of evidence:** External data lets you stand on existing research. Example: Published work from Microsoft and Google establishes that **latency/performance matters** — a smaller company without the resources to run their own latency experiments can use this evidence to justify investment in performance, then run targeted experiments for product-specific tradeoffs.

**Key caveat:** Because you don't control the external sampling methodology, absolute numbers may be unreliable. **Trends, correlations, and metric validation** are the most defensible use cases.

---

### Putting It All Together

Choice of technique depends on **your goal**:

| Goal | Best Technique(s) |
|---|---|
| No idea what metrics to gather | UER, Focus Groups (qualitative, brainstorming-style) |
| Interactions not on your site | Surveys |
| Validating metrics | External data, observational analyses |
| Early-stage idea pruning | UER, Focus Groups |
| Moving toward quantitative validation | Observational studies, experiments |

#### Tradeoffs to consider:
- **Scale vs. depth**: More users → more generalizable, but less "why"
- **Stage of product cycle**: Early = qualitative, mature = quantitative
- **Multiple methods triangulate better than one**: Use several methods to establish bounds and a hierarchy of evidence — this leads to more robust results

**Example of triangulation for personalized recommendations:**
1. **UER** — observe a small number of users, ask if recommendations feel useful, note behavioral signals (time reading screen, click patterns)
2. **Large-scale observational analysis** — validate metric ideas from UER, check interplay with business metrics
3. **On-screen survey** — reach more users with a simple question about recommendation quality
4. **Learning experiments** — change recommendations and see how user happiness metrics relate to overall business metrics and improve the OEC

---

## Chapter 11: Observational Causal Studies

### Why This Chapter Matters

Randomized controlled experiments are the **gold standard for causality**. But sometimes you can't run one. Observational causal studies are a lower-trust alternative using historical data. Knowing the designs *and* the pitfalls is critical.

---

### The Core Problem: Causal Inference

The **"basic identity of causal inference"** (Varian 2016):

```
Outcome for treated − Outcome for untreated
= [Outcome for treated − Outcome for treated if not treated]
+ [Outcome for treated if not treated − Outcome for untreated]
= Impact of Treatment on treated + Selection bias
```

In a proper randomized experiment, the second term (selection bias) has **expected value of zero** because random assignment makes the treated and untreated groups equivalent in expectation. In observational studies, that's exactly the term you cannot eliminate — you're always fighting selection bias.

---

### When Controlled Experiments Are Not Possible

1. **The causal action isn't under your control.** Example: you want to understand how behavior changes when a user switches from iPhone to Samsung. You can't randomly assign people to switch phones.

2. **Too few units.** Example: Merger & Acquisition scenarios — there's one event and estimating the counterfactual is extremely hard.

3. **Opportunity cost of withholding treatment is too high.** Example: establishing the impact of Super Bowl ads (Stephens-Davidowitz, Varian and Smith 2017), or when the OEC takes years to measure (e.g., a car purchase 5 years later).

4. **The change is expensive relative to perceived value.** Example: what happens to churn if you forcibly sign out all users? What if Google or Bing stopped showing ads? Running that experiment is prohibitively costly.

5. **The randomization unit can't be properly randomized.** Example: TV ad effectiveness — you can't randomize individual viewers. Using Designated Market Areas (DMAs) gives you only ~210 units in the US, resulting in very low statistical power.

6. **The experiment would be unethical or illegal.** Example: withholding medical treatments believed to be beneficial.

---

### Designs for Observational Causal Studies

Two core challenges in all observational causal designs:
- How to construct comparable Control and Treatment groups
- How to model the impact given those groups

---

#### 1. Interrupted Time Series (ITS)

**What it is:** A quasi-experimental design where you control *when* the change happens but can't randomize users into Treatment and Control. Instead, the **same population** serves as its own control: you compare pre-intervention to post-intervention.

**How it works:**
- Take multiple measurements *before* the intervention → build a model
- That model produces a **counterfactual** — what the metric *would* have been without intervention
- After intervention, compare actual observed metric to the predicted counterfactual
- Treatment effect = average difference between actual and predicted

**Extension:** Introduce the treatment, then *reverse* it, optionally repeating. Each reversal strengthens the causal inference. Example: helicopter surveillance of home burglaries was switched on and off multiple times over several months. Each time surveillance started, burglaries decreased; each time it was removed, burglaries increased. The repeating pattern is strong evidence of causality.

**Online application:** Understanding impact of online advertising on search-related site visits. Can use **Bayesian Structural Time Series** (BSTS) as the modeling approach (Charles and Melvin 2004).

**Figure 11.1 interpretation:**
- Panel (a): Solid line = actual observed metric, Dashed line = predicted counterfactual. Fit on pre-intervention period; prediction extends through intervention.
- Panel (b): Delta between actual and prediction. If the model is good, this delta is the Treatment effect estimate. Weekends shaded grey.

**Main pitfalls:**
- **Confounding time-based effects** — seasonality is the obvious one, but any underlying system change can confound. Alternating Treatment multiple times reduces this risk.
- **User experience inconsistency** — if users notice their experience flipping back and forth, the measured effect may reflect irritation/frustration rather than the change itself.

---

#### 2. Interleaved Experiments

**What it is:** A design for evaluating **ranking algorithm changes** (e.g., search engines, recommendation systems). Rather than showing users results from algorithm X *or* Y, you show a mix of both in a single result list.

**How it works:**
- Algorithm X would produce: x₁, x₂, ..., xₙ
- Algorithm Y would produce: y₁, y₂, ..., yₙ
- Interleaved list: x₁, y₁, x₂, y₂, ..., xₙ, yₙ (with duplicates removed)
- Measure click-through rate on results attributed to each algorithm

**Strength:** Very sensitive design for comparing ranking algorithms.

**Key limitation:** Results must be **homogeneous** — if the first result takes up more screen space, or results in different positions have different visual treatment, the design breaks down.

---

#### 3. Regression Discontinuity Design (RDD)

**What it is:** Used when there is a clear **threshold** that determines Treatment. Units just below the threshold serve as the Control; units just above serve as Treatment. The assumption is that these near-threshold populations are very similar, so the comparison is approximately unconfounded.

**Classic example:** Scholarship given for an 80% grade. Students just above 80% (Treatment) are assumed similar to those just below 80% (Control). The scholarship impact can then be estimated.

**Online example — drinking age and mortality (Angrist and Pischke 2014):**
Americans over 21 can drink legally. Mortality data plotted by days from 21st birthday shows a sharp spike on and immediately after the 21st birthday (approximately 100 additional deaths above a baseline of ~150/day). The spike is specific to the 21st birthday — it doesn't appear on the 20th or 22nd — suggesting it reflects legal alcohol access, not generic birthday celebration.

**When the assumption breaks down:**
If participants can manipulate which side of the threshold they're on — e.g., students convincing teachers to "mercy pass" them — the design is compromised (McCrary 2008).

**Common pitfall — confounding at the same threshold:** A study of alcohol's impact using the age-21 threshold might also be picking up the effect of **legal gambling**, which also starts at 21.

**Software applications:** RDD applies naturally when a scoring algorithm triggers something at a threshold score. Note that in software, this scenario also lends itself to a proper A/B experiment or a hybrid approach (Owen and Varian 2018).

---

#### 4. Instrumental Variables (IV) and Natural Experiments

**What it is:** A technique that tries to **approximate random assignment** using an "instrument" — a variable that influences Treatment assignment but is otherwise unrelated to the outcome.

**Classic examples:**
- **Vietnam draft lottery** as an instrument for military service — studying earnings differences between veterans and non-veterans. The lottery approximates random assignment.
- **Charter school lottery** as an instrument for attending a particular school — lottery doesn't guarantee attendance but strongly predicts it.

Both examples use **two-stage least-squares (2SLS) regression** to estimate the causal effect.

**Natural experiments:** Situations where "as-good-as-random" assignment occurs naturally:
- **Monozygotic (identical) twins** in medicine — allow twin studies as natural experiments
- **Notification queues and message delivery order** in social networks — can be leveraged to study the impact of notifications on engagement (Tutterow and Saint-Jacques 2019)

Why natural experiments matter in social networks: Controlled experiments on social platform members are tricky because the effect of treating one user may spill over to their connections (network interference).

---

#### 5. Propensity Score Matching (PSM)

**What it is:** Construct comparable Control and Treatment populations by matching on a **single constructed score** (the propensity score) rather than matching on every individual covariate. The propensity score represents the probability of receiving Treatment given observed covariates.

**Motivation:** If examining the impact of users switching from Windows to iOS, you want to ensure any observed difference isn't actually due to demographic differences between the groups.

**Use in practice:** PSM has been used to evaluate online ad campaigns (Chan et al. 2010).

**Criticism:**
- Only *observed* covariates are accounted for — **hidden biases from unobserved factors remain**
- Rosenbaum and Rubin themselves warned it requires "strong ignorability" conditions — but practitioners often fail to recognize when those conditions are violated (Pearl 2009)
- King and Nielsen (2018) argued PSM "often accomplishes the opposite of its intended goal — increasing imbalance, inefficiency, model dependence, and bias"

---

#### 6. Difference in Differences (DD / DID)

**What it is:** Assumes **common trends** between Treatment and Control groups. The Treatment effect is the *difference* in how the metric changed in Treatment vs. how it changed in Control over the same period.

**Intuition:** Even if Treatment and Control groups differ in level (e.g., one DMA has more users than another), if they would have trended the same way absent the intervention, the parallel trend assumption holds and DID is valid.

**Classic online use case:** Measuring the impact of **TV advertising** on user acquisition and retention. You run ads in one Designated Market Area (DMA) and use another DMA as Control. Measure the metric just before the campaign starts (T1) and at a later point (T2). The Control group's change captures external factors (seasonality, economic conditions). The Treatment effect = (Treatment change) − (Control change).

**Figure 11.3:** Shows Treatment, Control, and the counterfactual. The assumed treatment effect is the vertical gap between the Treatment line at T2 and where the counterfactual (extrapolated from the Control trend) predicts the Treatment would have been.

**Famous non-online example:** Card and Krueger (1994) studied the impact of a minimum wage increase in New Jersey on fast-food employment by comparing to eastern Pennsylvania, which closely matched New Jersey on relevant characteristics.

---

### Pitfalls of Observational Causal Studies

#### The Overarching Problem: Confounding

Every observational design is vulnerable to **unanticipated confounds** — variables that affect both the "treatment" assignment and the outcome, creating a false impression of causality.

**Example 1 — palm size and life expectancy:**
Smaller palm size correlates with longer life expectancy. But the common cause is **gender**: women have smaller palms and live longer (about 6 extra years in the US on average). Palm size doesn't cause longevity.

**Example 2 — errors and churn (Microsoft Office 365):**
Users who see more errors actually churn *less*. Counterintuitive! But the common cause is **usage intensity** — heavy users encounter more errors AND churn at lower rates. The lesson: never interpret a feature-usage/churn correlation as evidence that the feature reduces churn without running a controlled experiment.

#### Spurious / Deceptive Correlations

**Deceptive correlation (Figure 11.5):** A marketing company plots energy drink consumption against athletic performance, showing a strong correlation and implying their drink improves performance. The actual explanation: professional athletes who were paid to drink it are plotted — the correlation is manufactured.

**Spurious correlation (Figure 11.6):** Deaths by venomous spiders correlates strongly (r=0.8057) with word length in the Scripps National Spelling Bee. You would immediately reject the causal claim (shorter words → fewer spider deaths) as absurd. The point: **when we lack the intuition to reject a claim, spurious correlations can be convincing.**

Spurious correlations can almost always be found if you look hard enough (Vigen 2018). Multiple hypothesis testing makes this even worse.

---

### Sidebar: Refuted Observational Causal Studies

- **Ioannidis (2005):** Of six highly-cited observational causal studies, five failed to replicate.
- **Young and Karr (2019):** Of 52 claims from 12 observational medical papers, *none* replicated in subsequent randomized controlled trials. In 5 of the 52 cases, the randomized trial found a statistically significant effect in the **opposite direction** from the observational study.
- Their conclusion: *"Any claim coming from an observational study is most likely to be wrong."*

**Online advertising example (Lewis, Rao and Reiley 2011):** Observational studies are often necessary to measure ad effectiveness because the intervention (the ad) and the outcome (sign-up, engagement) happen on different platforms in different spheres of control — making randomized experiments impractical.

---

### The Bottom Line on Observational Studies

Even with perfect execution, **there is never a guarantee** that some unincluded factor isn't affecting your results. Quasi-experimental methods require making many assumptions — some explicit, some implicit — and any of them can be wrong.

Incorrect assumptions can compromise:
- **Internal validity** — did the Treatment actually cause the observed effect?
- **External validity** — does the effect generalize beyond this specific context?

Building domain intuition (Chapter 1) helps improve the quality of assumptions, but it will not eliminate all risks.

> **The scientific gold standard for establishing causality remains the randomized controlled experiment.**

Observational causal studies are a valuable tool when experiments aren't possible — but they should be treated as evidence on a hierarchy, not as definitive proof.




Here are the complete notes for Chapters 12–14:

---

# Chapter 12 — Client-Side Experiments

## Why This Chapter Matters

Most of the book assumes **thin clients** (web browsers) for simplicity. This chapter is dedicated to **thick clients** — native mobile apps and desktop client applications — which behave fundamentally differently and require extra care to ensure trustworthy experiments.

With the explosive growth of mobile usage, the number of experiments running on mobile apps has grown significantly (Xu and Chen 2016).

---

## Terminology

| Term | Definition |
|---|---|
| **Client-side experiment** | Experiment changes made *within* a thick client (the app itself) |
| **Server-side experiment** | Experiment changes made server side — regardless of whether it impacts a thick or thin client, and regardless of whether it's a UX or backend change |

---

## Two Core Differences Between Server and Client Side

---

### Difference #1: Release Process

**Server side (thin clients / web):**
- New feature releases can happen continuously — sometimes multiple times per day
- Changes are fully controlled by the organization; updating server-side code is relatively easy as part of **continuous integration and deployment**
- When a user visits a site, the server pushes data (e.g., HTML) to the browser without interrupting the end-user experience
- In a controlled experiment, the variant the user sees is **fully managed by the server** — no end user action required
- Changing a button color, showing a revamped homepage — all can happen **instantaneously** after a server-side deployment

**Client apps:**
- Many features can still be controlled server side (e.g., feed content shown in the Facebook app) and follow the same easy release process
- The more you can rely on **services** (server-side code), the easier it is to experiment — both in terms of agility and consistency across clients
- Examples: many changes on Bing, Google, LinkedIn, and Office are made server side and impact all clients, whether web or thick clients like mobile apps

**However**, a significant amount of code is **shipped with the client itself**. Changes to this code must be released differently. The release process for a mobile app involves **three parties**:
1. The **app owner** (e.g., Facebook)
2. The **app store** (e.g., Google Play or Apple App Store)
3. The **end user**

**The release pipeline:**
- App owner submits a build to the app store for review → review can take **days**
- Even after approval, releasing to everyone doesn't mean everyone gets it immediately — getting the new version is a **software upgrade**
- Users can **delay or ignore** the upgrade while continuing to use the old version
- Some users take **weeks** to adopt
- Some enterprise organizations **don't allow updates** for their users
- Some software (e.g., Exchange) runs in **sovereign clouds** restricted from calling unapproved services
- At any given time, there are **multiple versions of the app** out in the wild that the app owner must support

This same challenge applies to **native desktop clients** (Office, Adobe Acrobat, iTunes), even without an app store review process.

**Staged rollout** (now supported by both Google Play and Apple App Store):
- App owners can release the new app to only a **percentage** of users and pause if something problematic is discovered
- Staged rollouts are essentially randomized experiments — eligible users are selected at random
- **But they cannot be analyzed as proper randomized experiments** because app owners only know who has *adopted* the new app, not who was *eligible* to receive it — creating selection bias
- App updates also cost users **network bandwidth** and can be an annoying experience
- Some software (e.g., Windows, iOS) cannot update frequently because some updates require a **reboot**

---

### Difference #2: Data Communication Between Client and Server

Once the new app is in the hands of users, it must communicate with the server — both pulling data from the server and sending back telemetry about what's happening on the client.

#### Data Connection Limitations

**Internet connectivity:**
- Connections may be unreliable or inconsistent
- In some countries, users may be **offline for days**
- Even normally-online users may be temporarily without internet (on a plane, in a dead zone)
- Server-side data changes may not reach these clients; client telemetry may be **delayed** in transmitting back to the server
- These delays **vary by country or demographic** and must be accounted for in instrumentation and downstream processing

**Cellular data bandwidth:**
- Most users have **limited cellular data plans**
- Question: send telemetry over cellular or only over Wi-Fi?
- Most apps choose to send telemetry **only over Wi-Fi**, which delays when data arrives server side
- There is also **heterogeneity across countries** — mobile infrastructure varies in bandwidth, cost, and reliability

#### Device Performance Limitations

Even with a good connection, using the network impacts device performance and ultimately user engagement (Dutta and Vadermeer 2018):

**Battery:**
- More data communication = increased battery consumption
- Waking up the app more regularly to send telemetry drains battery faster
- Mobile devices in **low battery mode** have restrictions on what apps can do (Apple, Inc. 2018)

**CPU, latency, and performance:**
- Lower-end devices are still constrained by CPU power
- Frequent data aggregation on-device and back-and-forth with the server can make the app **less responsive** and hurt overall performance

**Memory and storage:**
- **Caching** reduces data communication but increases app size
- Larger app size hurts performance and increases **app uninstallment** (Reinhardt 2016)
- A larger concern for users with lower-end devices with less memory and storage

**The fundamental tradeoff matrix:**

| To get this... | You pay with this... |
|---|---|
| More consistent internet connection | More cellular data consumption |
| Less data sent to server | More CPU spent computing/aggregating on-device |
| Delayed telemetry upload (wait for Wi-Fi) | More on-device storage |

These tradeoffs impact both **visibility into what is happening client side** and **user engagement and behavior** — making this a rich area for experimentation, but also one requiring extra care for trustworthy results.

---

## Implications for Experiments

---

### Implication #1: Anticipate Changes Early and Parameterize

Because client code can't be shipped to users easily, all experiments — including **all variants** — must be **coded and shipped with the current app build**. Any new variants, including bug fixes on existing variants, must wait for the next release.

Example: In a typical monthly release, Microsoft Office ships with hundreds of features that roll out in a controlled manner for safe deployment.

**Three practices that follow from this:**

**1. Feature flags (dark features):**
- A new app may be released before certain features are completed
- These features are **gated by configuration parameters called feature flags** that turn the features off by default
- Features turned off this way are called **dark features**
- When the feature is finished (sometimes when a server-side service completes), it can be turned on remotely

**2. Server-side configurability:**
- More features are built to be **configurable from the server side**
- This allows A/B testing and provides a **safety net**: if a feature performs poorly, you can instantly revert by shutting down the variant without going through a lengthy client release cycle
- Prevents end users from being stuck with a faulty app for weeks until the next release

**3. Fine-grained parameterization:**
- Even though new *code* can't be pushed easily, new *configurations* can be passed to the client
- If the client understands how to parse configurations, passing new config values **effectively creates new variants without a client release**
- Example: Experiment on the number of feed items to fetch. Instead of hardcoding a number and only testing what was planned at build time, parameterize the number and experiment freely after release
- **Real example:** Windows 10 parameterized the search box text in the task bar, ran experiments *over a year after it shipped*, and the winning variant increased user engagement and Bing revenue by **millions of dollars**
- Another common example: updating machine learning model parameters from the server so a model can be tuned over time

> **Note:** App stores may have policies limiting which features can be shipped dark. Read app store policies carefully and disclose dark features appropriately.

---

### Implication #2: Expect Delayed Logging and Effective Starting Time

Limited or delayed data communication can delay both the **arrival of instrumentation data** and the **effective start time of the experiment** itself.

Even after the experiment implementation is shipped and activated for a percentage of users, the experiment is not fully active because:

- **Devices may not get the new experiment configuration** — either because they're offline or in limited/low bandwidth situations where pushing new configs incurs cost or poor UX
- **Config fetched at app open may not take effect until the next session** — you don't want to change a user's experience mid-session. For heavy users (multiple sessions/day) this delay is small; for light users (once a week), the experiment may effectively not start for a week
- **Many devices still have old versions** right after a new app release — based on experience, the initial adoption phase takes about **a week** to reach a more stable adoption rate (this varies greatly by user population and app type)

**Consequences for analysis:**
- Early experiment signals appear weaker (smaller sample size)
- Early adopters are **heavily biased toward frequent users and Wi-Fi users**
- Experiment **duration may need to be extended** to account for these delays
- **Treatment and Control may have different effective starting times** — especially if Control is a shared variant that was live before Treatment, leading to a different user population due to selection bias
- If Control runs earlier, **caches are warmed up** so responses to service requests are faster, potentially introducing additional bias
- The time period used to compare Treatment and Control must be carefully chosen

---

### Implication #3: Create a Failsafe for Offline or Startup Cases

When users open an app, their device could be offline. For consistency:

- **Cache experiment assignment** so that if the next app open occurs while offline, the user sees the same variant
- If the server is not responding with the configuration needed to decide on assignment, have a **default variant** for the experiment
- Some apps are distributed via **OEM (Original Equipment Manufacturer) agreements** — experiments must be set up properly for a **first-run experience**, including:
  - Retrieving configurations that only impact the *next* startup
  - Ensuring a **stable randomization ID** before and after users sign up or log in (the ID must survive authentication)

---

### Implication #4: Triggered Analysis May Need Client-Side Experiment Assignment Tracking

Triggered analysis (only counting users who were actually exposed to the feature) requires extra care on the client side.

- One approach: send tracking data to the server when an experiment feature is actually *used*
- But: to reduce communications, experiment assignment information is usually **fetched for all active experiments at once** (e.g., at app startup), regardless of whether each experiment is actually triggered (see Chapter 20)
- Relying on that fetch-time tracking for triggered analysis would lead to **over-triggering** — counting users as exposed even when they never encountered the feature
- **Solution:** Send assignment information only when the feature is actually used — this requires instrumentation to be sent from the client
- **Caveat:** If the volume of these tracking events is high, it can cause **latency and performance issues**

---

### Implication #5: Track Important Guardrails on Device and App Level Health

Standard engagement metrics may not capture important device-level side effects:

- **CPU and battery:** A Treatment may consume more CPU and drain battery faster — if you only track engagement, you'll miss this
- **Push notifications:** A Treatment may increase push notifications, leading to users **disabling notifications at the device settings level** — may not show up as engagement drop during the experiment but has **significant long-term impact**

**App-level health metrics to track:**
- **App size** — larger app size reduces downloads and causes uninstalls (Tolomei 2017, Google Developers 2019)
- **Internet bandwidth consumption**
- **Battery usage**
- **Crash rate** — for crashes, log a clean exit so telemetry about the crash can be sent on the **next app start** (since the current session crashed)

---

### Implication #6: Monitor Overall App Release Through Quasi-Experimental Methods

Not all changes in a new app can be put behind an A/B parameter. To run a true randomized controlled experiment on the entire new app:

- Bundle both old and new versions behind the same app and randomly assign users to one — but this is **not practical** for most apps because it can **double the app size**

**Alternative (quasi-experimental):**
- Because not all users adopt the new version simultaneously, there is a window where **both versions are serving real users**
- This effectively offers an A/B comparison — but requires **correcting for adoption bias**
- Xu and Chen (2016) share techniques to remove this bias in the mobile adoption setting

---

### Implication #7: Watch Out for Multiple Devices/Platforms and Interactions Between Them

It is common for users to access the same product via multiple devices and platforms (desktop, mobile app, mobile web). Two implications:

**1. Different IDs across devices:**
- Different identifiers may be available on different devices
- As a result, the **same user may be randomized into different variants on different devices** (Dmitriev et al. 2016)

**2. Interactions between devices:**
- Many browsers (e.g., Edge) now support "Continue on desktop" / "Continue on mobile" sync features
- It is common to shift traffic between the mobile app and mobile web — e.g., clicking an Amazon link in an email may open the Amazon app *or* the mobile website depending on installation status
- If an experiment causes or suffers from these cross-device interactions, you **cannot evaluate app performance in isolation** — you must look at user behavior **holistically across platforms**
- Important: the user experience on one platform (usually the app) is often **better than on the web** — directing traffic from app to web tends to **bring down total engagement**, which can be a confounding effect not caused by the experiment itself

---

## Conclusion

Experimentation on thick clients requires substantially more care than on thin clients. Many differences are subtle but critical. With rapid technological improvement, many of the specific details and implications in this chapter will continue to evolve over time.

---

---

# Chapter 13 — Instrumentation

## Why This Chapter Matters

Before running any experiments, you must have **instrumentation** in place to log what is happening to users and the system. Every business should have a baseline understanding of how the system performs and how users interact with it. When running experiments, having rich data about what users saw, their interactions (clicks, hovers, time-to-click), and system performance (latencies) is **critical**.

> The terms **"instrument," "track,"** and **"log"** are used interchangeably throughout this chapter.

---

## Client-Side vs. Server-Side Instrumentation

### Client-Side Instrumentation

Focuses on **what the user experiences** — what they see and do:

- **User actions:** What activities does the user do — clicks, hovers, scrolls? At what times? What actions are done on the client without a server round-trip? (e.g., hovers generating help text, form field errors, slideshow navigation)
- **Performance:** How long does it take the page to display or become interactive? (Chapter 5 discusses complexities around search query latency measurement)
- **Errors and crashes:** JavaScript errors are common, may be browser-dependent, and must be tracked

**Advantage of client-side instrumentation:**
- Gives a direct view of what the user **actually sees and does**
- Can catch things server-side can't — e.g., **client-side malware** that overwrites what the server sends is only discoverable via client-side instrumentation (Kohavi et al. 2014)

**Drawbacks specific to JavaScript-based clients:**

**1. Resource consumption:**
- Can utilize significant CPU cycles and network bandwidth
- Depletes device batteries
- Large JavaScript snippets impact **load time** — increased latency hurts both current visit engagement and the likelihood of users **returning** (see Chapter 5)

**2. Data lossiness (web beacons):**
Web beacons are often used to track user interactions (e.g., when users click a link to go to a new site). They can be lost when:

- **(a) Race condition:** A new site loads *before* the beacon is successfully sent → beacon is cancelled and lost. Loss rate varies by browser.
- **(b) Synchronous redirect:** Force the beacon to be sent *before* the new site loads. Lossiness decreases, but latency increases → worse UX and higher likelihood of user abandoning the click.
- **(c) Application choice:** You can implement either (a) or (b) depending on the use case. For **ad clicks** — which must be reliably tracked for payments and compliance — option (b) is preferred despite the added latency.

**3. Client clock drift:**
- Client clock can be changed manually or automatically (e.g., time zone changes, NTP sync)
- Actual timing from the client may **not be synchronized with server time**
- **Never subtract client and server timestamps** — they could be significantly off even after adjusting for time zones

---

### Server-Side Instrumentation

Focuses on **what the system does**:

- **Performance:** Time for the server to generate a response; which component takes longest? Performance at the 99th percentile?
- **System response rate:** Number of requests received, pages served, how retries are handled
- **System information:** Number of exceptions/errors, cache hit rate

**Advantages:**
- Suffers less from the accuracy and resource concerns affecting client-side instrumentation
- Not impacted by the network → **lower variance** → more sensitive metrics
- Example: logging internal scores in search engine results (why specific results were returned, their ranking) → useful for debugging and tuning the search algorithm
- Example: logging which servers or data centers handled each request → useful for debugging bad equipment or data centers under stress

**Caveat:** Servers also need to be synchronized often. There can be scenarios where a request is served by one server while the beacon is logged by another, creating a **timestamp mismatch**.

---

## Processing Logs from Multiple Sources

In practice you will have **multiple logs from different instrumentation streams** (Google 2019):

- Logs from different client types (browser, mobile)
- Logs from servers
- Per-user state (e.g., opt-ins and opt-outs)

**Key requirements for multi-source log processing:**

**1. A common join key:**
- There must be a way to join logs across sources
- The ideal case is a **common identifier in all logs** to serve as a join key
- The join key must identify which events are for the same user, or **randomization unit** (see Chapter 14)
- You may also need join keys for **specific events** — e.g., a client-side event indicating a user saw a particular screen and a corresponding server-side event explaining *why* they saw it. The join key links these as two views of the same event for the same user.

**2. A shared format:**
- Shared format makes downstream processing easier
- Includes **common fields** (timestamp, country, language, platform) and customized fields
- Common fields are often the **basis for segments** used in analysis and targeting

---

## Culture of Instrumentation

Instrumentation should be treated as **critical for the live site**.

> *Imagine flying a plane with broken instruments in the panel. It is clearly unsafe, yet teams may claim that there is no user impact to having broken instrumentation. How can they know? Without proper instrumentation, they are flying blind.*

**The core difficulty:** Getting engineers to instrument in the first place. This stems from:
- **Time lag** — time between when the code is written and when results are examined
- **Functional dissociation** — the engineer creating the feature is often *not* the one analyzing the logs to see how it performs

**Tips for improving instrumentation culture:**

- **Establish a cultural norm:** Nothing ships without instrumentation. Include instrumentation in the specification. Treat broken instrumentation with the same priority as a broken feature. It is too risky to fly a plane if the gas gauge or altimeter is broken, even if it can still fly.
- **Invest in testing instrumentation during development:** Engineers creating features add necessary instrumentation and can see the resulting data in tests *before* submitting code. Code reviewers check for it.
- **Monitor raw logs for quality:** Track things like number of events by key dimensions, and invariants that should always be true (e.g., timestamps fall within a particular range). Ensure tools exist to detect outliers on key observations and metrics. When a problem is detected, developers across the organization should fix it **right away**.

---

---

# Chapter 14 — Choosing a Randomization Unit

## Why This Chapter Matters

The choice of **randomization unit** is critical in experiment design. It affects:
- The **user experience** (consistency)
- Which **metrics** can be used to measure experiment impact

When building an experimentation system, you need to think through what options to make available. Understanding the options and the relevant considerations leads to improved experiment design and analysis.

> **Important note:** Identifiers serve double duty — they are the base randomization unit for experiments *and* can be used as a **join key** for downstream log file processing (see Chapters 13 and 16).

This chapter focuses on *which identifier to use*, not on base randomization criteria like ensuring independence of assignment between units, or ensuring independence of assignment across simultaneous experiments (see Chapter 4).

---

## Granularity of Randomization Units

One key axis for choosing a randomization unit is **granularity**. For websites, natural granularities include:

| Level | Definition |
|---|---|
| **Page-level** | Each new web page viewed is a unit |
| **Session-level** | Group of pages viewed on a single visit; session typically ends after 30 minutes of inactivity |
| **User-level** | All events from a single user. Approximated via cookies or login IDs — each has overcounting/undercounting issues |

Additional granularities:
- **Query-level** (search engines) — between page and session, since multiple pageviews can result from a single query
- **User × Day** — events from the same user on different days are treated as different units (Hohnhold, O'Brien and Tang 2015)

---

## Two Main Questions for Choosing Granularity

### Question 1: How Important Is Consistency of User Experience?

The main question: **will the user notice the changes?**

- **Font color experiment at page-level:** The font color could change with every page — a jarring experience
- **New feature experiment at page or session level:** The feature may appear and disappear — potentially bad and inconsistent
- **Rule:** The more the user will notice the Treatment, the more important it is to use a **coarser granularity** to ensure consistency

### Question 2: Which Metrics Matter?

- **Finer granularity** → more units → **smaller variance** of the metric mean → **more statistical power** to detect smaller changes
- However: randomizing by pageviews leads to a tiny underestimation of the variance of the Treatment effect (Deng, Lu and Litz 2017), but this underestimation is very small in practice and **commonly ignored**

---

## Constraints That Limit Fine Granularity

Despite the statistical power benefit of finer granularity, there are hard constraints:

**1. If features act across that level of granularity, you cannot randomize at that level.**
- Example: if personalization or inter-page dependencies exist, randomizing by **pageview** is invalid — what happens on one page affects what the user sees on subsequent pages; pages are no longer independent
- Specific example: if a user's first query is in Treatment and results in poor search results, they may reformulate a second query that ends up in Control — creating contamination

**2. If metrics are computed across that level of granularity, they cannot be used.**
- Example: a page-level randomization experiment cannot measure whether Treatment impacts **total number of user sessions** (sessions span pages)

**3. Exposing users to different variants may violate SUTVA** (Stable Unit Treatment Value Assumption, Chapter 3) (Imbens and Rubin 2015).
- SUTVA states that experiment units should not interfere with one another
- If users notice they're in different variants (e.g., because the feature keeps appearing and disappearing), that knowledge can impact their behavior and create interference (see Chapter 22)

---

## Special Randomization Unit Scenarios

- **Enterprise (e.g., Office):** Tenants want consistent experiences for the entire enterprise → limits ability to randomize at user level; may need to randomize by tenant
- **Advertising auctions:** Randomize by advertiser or by clusters of advertisers who compete in the same auctions
- **Social networks:** Randomize by clusters of friends to minimize interference (Xu et al. 2015, Ugander et al. 2013, Katzir, Liberty and Somekh 2012, Eckles, Karrer and Ugander 2017) — generalizes to network components (Yoon 2018)

---

## Randomization Unit and Analysis Unit

**Core recommendation: the randomization unit should be the same as (or coarser than) the analysis unit.**

| Scenario | Verdict |
|---|---|
| Randomization unit = analysis unit | ✅ Variance computation is straightforward; independence assumption is reasonable |
| Randomization unit coarser than analysis unit | ⚠️ Works but requires more nuanced methods (bootstrap, delta method) |
| Randomization unit finer than analysis unit | ❌ User's experience contains a mix of variants; user-level metrics are not meaningful |

**When randomization unit = analysis unit:**
- Randomizing by **page**: clicks on each pageview are independent → variance of click-through rate (clicks/pageviews) is standard to compute
- Randomizing by **user**: sessions-per-user, clicks-per-user, pageviews-per-user → analysis is relatively straightforward

**When randomization unit is coarser than analysis unit** (e.g., randomize by user, analyze click-through rate by page):
- Requires methods like **bootstrap** or **delta method** (Deng et al. 2017, Deng, Knoblich and Lu 2018, Tang et al. 2010, Deng et al. 2011) — see Chapters 18 and 19
- Results can be **skewed by bots** that reuse a single user ID with 10,000+ pageviews
- Mitigations: bound what any individual user can contribute to the finer-grained metric, or switch to a user-based metric (e.g., average click-through rate per user)

**When randomization unit is finer than analysis unit** (e.g., randomize by page, compute sessions-per-user):
- User's experience likely contains a **mix of variants** → computing user-level metrics is **not meaningful**
- If user-level metrics are part of your OEC, you **cannot use finer granularity for randomization**

---

## User-Level Randomization

User-level randomization is the **most common** because it:
- Avoids inconsistent experiences for the user
- Allows long-term measurement such as **user retention** (Deng et al. 2017)

### Types of User-Level Identifiers

**1. Signed-in User ID / Login**
- Stable **across devices and platforms** and **longitudinally across time**
- Best choice when you need cross-device consistency and it's available
- Example use case: measuring long-term effects of latency or speed changes, or users' learned response to ads (Hohnhold et al. 2015)

**2. Pseudonymous User ID (Cookie)**
- Written by the website to the browser on first visit; on mobile apps, the OS provides an equivalent (Apple's IDFA/IDFV, Android's Advertising ID)
- **Not persistent across platforms** — the same user on desktop browser vs. mobile web would have different IDs
- **Controllable by the user** via browser or OS-level settings → less persistent longitudinally than signed-in IDs
- Better choice when testing processes that cut across the sign-in boundary (e.g., new user onboarding that includes signing in for the first time)

**3. Device ID**
- **Immutable** ID tied to a specific physical device
- Because it is immutable, these IDs are considered **identifiable** (privacy implication)
- No cross-device or cross-platform consistency, but typically **stable longitudinally**

**Key consideration:** When choosing between IDs, consider both **functional** and **ethical** aspects (see Chapter 9).

**Functional scope comparison:**

| ID Type | Cross-device? | Cross-platform? | Longitudinal stability |
|---|---|---|---|
| Signed-in user ID | ✅ Yes | ✅ Yes | ✅ High |
| Cookie / pseudonymous ID | ❌ No | ❌ No | ⚠️ Medium (user-controllable) |
| Device ID | ❌ No | ❌ No | ✅ High (but immutable = identifiable) |

---

### IP Address (Not Recommended)

**Only recommended** when it is the sole option — e.g., infrastructure changes like comparing latency using one hosting service or data center vs. another (often only controllable at the IP level).

**Why not recommended generally:**

- **Highly variable granularity:** A user's IP can change when they move locations (different IP at home vs. work) → inconsistent experiences
- **Shared IPs:** Large companies and ISPs may have many users sharing a small set of IP addresses → leads to:
  - Low statistical power (too few IP addresses, very wide variance)
  - Skew and outlier issues from aggregating large numbers of users into a single unit

---

## Sub-User Level Randomization

Useful **only** when:
- There is no concern about **carryover or leakage** from the same user across units (see Chapter 22)
- Success metrics are **also at the sub-user level** (e.g., clicks-per-page, not clicks-per-user)

Often chosen specifically for the **increased statistical power** that comes from a larger sample size.



Here are the complete notes for Chapters 15–16:

---

# Chapter 15 — Ramping Experiment Exposure: Trading Off Speed, Quality, and Risk

## Why This Chapter Matters

Experimentation accelerates product innovation, but **how fast you can innovate is limited by how you experiment**. To control unknown risks associated with new feature launches, experiments should go through a **ramp process** — gradually increasing traffic to new Treatments.

Without a principled approach, ramping can introduce inefficiency and risk, **decreasing product stability** as experimentation scales.

Effective ramping requires balancing three key considerations: **Speed, Quality, and Risk (SQR).**

---

## What Is Ramping?

We often talk about running experiments with a given traffic allocation that provides enough statistical power. In practice, it is common that an experiment goes through a **ramping process** to control unknown risks associated with new feature launches (also called **controlled exposure**).

**How it works:**
- A new feature starts by exposing Treatment to only a **small percentage** of users
- If metrics look reasonable and the system scales well, **more and more users** are exposed
- Traffic is ramped until Treatment reaches the desired exposure level

**Classic negative example — Healthcare.gov:**
The site collapsed when it was rolled out to **100% of users on day one**, only to reveal it couldn't handle the load. This could have been mitigated by rolling out by geographic area or last names A–Z. Insisting on a ramping process became a key lesson for subsequent launches (Levy 2014).

**The core dilemma:**
- Ramping too **slowly** → wastes time and resources
- Ramping too **quickly** → hurts users, risks suboptimal decisions

A fully self-served experimentation platform (Chapter 4) can democratize experimentation, but you need **principles to guide ramping** and **tooling to automate and enforce** those principles at scale.

> **Note:** This chapter focuses primarily on **ramping up**. Ramping down (when a Treatment is bad) typically means **shutting it down to zero very quickly** to limit user impact. Also, large enterprises usually control their own client-side updates, so they are effectively excluded from some experiments and ramping exposure.

---

## SQR Ramping Framework

Three reasons why we run controlled online experiments — and how they map to the SQR framework:

**1. To measure:** Measure the impact and ROI of the Treatment if it launched to 100%
**2. To reduce risk:** Minimize damage and cost to users and business during an experiment when there is a negative impact
**3. To learn:** Learn about user reactions by segments, identify potential bugs, and inform future plans

### Maximum Power Ramp (MPR)

If the *only* reason to run a controlled experiment were to **measure**, we would run at the **Maximum Power Ramp (MPR)** — typically a **50/50 traffic split** (when ramping to 100%), giving the highest statistical sensitivity and the fastest, most precise measurement.

> **MPR footnote:** If the experiment has the entire 100% traffic with only one Treatment, the variance in the two-sample t-test is proportional to 1/q(1−q), where q is the treatment traffic percentage. The MPR in this case is a 50/50 split. If there is only 20% traffic available to experiment, the MPR has a 10/10 split. If there are four variants splitting 100% traffic, each variant gets 25%.

However, we usually **don't start at MPR** — because *what if something goes wrong?* That is why we start at a small exposure to contain impact and mitigate potential risk.

We may also need **intermediate stages between MPR and 100%** — e.g., for operational reasons, waiting at 75% to ensure new services or endpoints can scale to increasing traffic load.

**Long-term holdout:** Sometimes a small fraction (e.g., 5–10%) of users do not receive the new Treatment for a period of time (e.g., two months) primarily for **learning purposes** — to learn whether the impact measured during MPR is sustainable long-term. See Chapter 23.

---

## Four Ramp Phases

Figure 15.1 illustrates the four phases, each with a primary goal:

```
Percent Assigned 
to Treatment
100% ──────────────────────────────────────── Launch
 50% ───────── MPR (e.g. a week) ────────────── Measure
      Pre-MPR   │  Post-MPR  │  Long-term holdout
      (short    │  (short    │  (as needed)
       ramps)   │   ramps,   │
                │ operational│
                │   only)    │
  0% ──────────────────────────────────────────────────▶ Time
      Phase 1  Phase 2  Phase 3        Phase 4
      Mitigate  Measure  Mitigate Risk  Learn
      Risk               (Operational)
```

- **Phase 1 (Pre-MPR):** Primarily for **risk mitigation** → tradeoff between speed and risk
- **Phase 2 (MPR):** For **precise measurement** → tradeoff between speed and quality
- **Phase 3 (Post-MPR):** Optional — for **operational concerns** only
- **Phase 4 (Long-term holdout):** Optional — for **long-term impact** measurement

---

### Ramp Phase 1: Pre-MPR

**Goal:** Safely determine that risk is small and ramp quickly to MPR.

**Technique 1 — Rings of testing populations:**
Create successive rings, gradually exposing Treatment to each ring. Bugs are commonly identified during early rings. Note that measurements from early rings are **biased** because those users are likely "insiders."

| Ring | Population | Purpose |
|---|---|---|
| **(a) Whitelisted individuals** | Team implementing the feature | Get verbatim feedback from team members |
| **(b) Company employees** | All employees | More forgiving if bugs are bad |
| **(c) Beta users / insiders** | Vocal, loyal early adopters who want new features and give feedback | Qualitative feedback at slightly higher scale |
| **(d) Data centers** | Single data center, small traffic (e.g., 0.5–2% at Bing) | Isolate interactions hard to identify otherwise — memory leaks, inappropriate resource use (heavy disk I/O), etc. (see Chapter 22). Once a single data center is ramped to decent traffic, all data centers can ramp up. |

The first rings provide qualitative feedback — there simply isn't enough traffic for a meaningful quantitative read. The next rings may have some quantitative measurement but remain **uncontrolled** because statistical power is too low.

**Technique 2 — Automatically dialing up traffic:**
Automatically increase traffic until it reaches the desired allocation. Even if the desired allocation is only a small percentage (e.g., 5%), taking an extra hour to ramp to 5% — rather than jumping there instantly — can **limit the impact of bad bugs** without adding much delay.

**Technique 3 — Real-time or near-real-time measurements on key guardrail metrics:**
The sooner you can get a read on whether an experiment is risky, the faster you can decide to move to the next ramp phase.

---

### Ramp Phase 2: MPR

**Goal:** Precise measurement of the experiment's impact.

All the trustworthy-results discussions throughout the book apply directly here.

**Key recommendation:** Keep experiments at MPR for **at least one week** — and longer if novelty or primacy effects are present.

**Why at least one week?**
- An experiment running for only one day will have results **biased toward heavy users**
- Users who visit on weekdays tend to be different from users visiting on weekends
- You need to capture **time-dependent factors**

**Diminishing returns on duration:**
While longer experiments generally produce smaller variance, there is a diminishing return. In practice, the precision gained after a week tends to be small **if there are no novelty or primacy trends** in the Treatment effect.

---

### Ramp Phase 3: Post-MPR

**Goal:** Operational scaling only — not for additional measurement.

By the time the experiment is past MPR, there should be **no concerns about end-user impact**. Ideally, operational concerns are also resolved in earlier ramps.

However, some engineering infrastructures may need incremental ramps before going to 100% due to concerns about increasing traffic load. These ramps should:
- Take **a day or less**
- Cover **peak traffic periods**
- Involve **close monitoring**

---

### Ramp Phase 4: Long-Term Holdout (or Replication)

**Goal:** Learn whether short-term effects are sustainable long-term.

Long-term holdouts (also called **holdbacks**) keep certain users unexposed to the Treatment for an extended period.

> **Caution:** Do NOT make long-term holdout a default step in the ramping process. Besides the cost, it can be **unethical** when you know there is a superior experience but deliberately delay delivering it — especially when customers are paying equally. Only do a long-term holdout if it can be truly useful.

**Three scenarios where long-term holdout is useful:**

**1. When the long-term effect may differ from the short-term effect (see Chapter 23):**
- **(a)** The experiment area is known to have a **novelty or primacy effect**
- **(b)** Short-term impact on key metrics is so large that you must confirm it's **sustainable** (e.g., for financial forecasting)
- **(c)** Short-term impact is small-to-none, but teams believe in a **delayed effect** (e.g., due to adoption or discoverability)

**2. When an early indicator metric shows impact, but the true-north metric is long-term** (e.g., one-month retention)

**3. When there is a benefit of variance reduction for holding longer** (see Chapter 22)

**Important nuance on traffic allocation during holdout:**

There is a misconception that holdouts should always have a majority of traffic in Treatment (90–95%). While this works well in general, **for scenario 1c** (where short-term impact is already too small to detect even at MPR), you should **continue the holdout at MPR if possible**. The statistical sensitivity gained by running longer is usually not enough to offset the sensitivity loss of going from MPR to 90%.

**Types of holdouts:**

| Type | Description |
|---|---|
| **Experiment-level holdout** | Standard holdout for a single experiment |
| **Uber holdout** | Some portion of traffic withheld from *any* feature launch over a long term (often a quarter) to measure **cumulative impact** across experiments |
| **Global holdout (Bing)** | 10% of Bing users withheld from any experiments to measure the overhead of the experimentation platform (Kohavi et al. 2013) |
| **Reverse experiment** | Users are put *back* into Control several weeks or months after Treatment launches to 100% (see Chapter 23) |

---

### Replication

When experiment results are surprising, a good rule of thumb is to **replicate**:
- Rerun the experiment with a **different set of users** or with **orthogonal re-randomization**
- If results remain the same, you can have much more confidence they are trustworthy
- Replication is a simple yet effective way to **eliminate spurious errors**
- When there have been many iterations of an experiment, results from the final iteration may be **biased upwards** — a replication run reduces the multiple-testing concern and provides an **unbiased estimate** (see Chapter 17)

---

## Post Final Ramp Cleanup

After an experiment is ramped to 100%, **cleanup is critical** for keeping the production system healthy:

| Architecture | Cleanup needed |
|---|---|
| **Code fork** (variant assignment creates a code branch) | Remove the **dead code path** after launch — failure to do this can be disastrous if the dead path is accidentally executed, e.g., during an experimentation system outage |
| **Parameter system** | Simply set the **new parameter value as the default** |

This cleanup step is often overlooked in fast-moving development — but it is essential.

---

---

# Chapter 16 — Scaling Experiment Analyses

## Why This Chapter Matters

For a company to move to the later phases of experimentation maturity ("Run" or "Fly"), incorporating **data analysis pipelines** as part of the experimentation platform ensures that:
- Methodology is solid, consistent, and scientifically founded
- Implementation is trustworthy
- Teams are saved from time-consuming ad hoc analysis

Understanding the common infrastructure steps for **data processing, computation, and visualization** is essential for scaling.

---

## Data Processing ("Cooking the Data")

Raw instrumented data must be processed into a state suitable for computation. This is called **cooking the data** and involves three steps:

---

### Step 1: Sort and Group

Information about a user request may be logged by multiple systems (both client and server side). Start by:
- **Sorting and joining** multiple logs (see Chapter 13) — sort by both user ID and timestamp
- This allows joining events to create **sessions or visits** and grouping all activity by a specified time window

**Materialization decision:**
- You may not need to materialize this join — a **virtual join** as a step during processing may suffice
- Materialization is useful if the output serves multiple purposes beyond experiments: debugging, hypothesis generation, etc.

---

### Step 2: Clean the Data

Having data sorted and grouped makes cleaning easier:

**Bot and fraud removal:**
- Use heuristics to remove sessions unlikely to be real users (see Chapter 3)
- Useful heuristics: sessions with too much or too little activity, too little time between events, too many clicks on a page, users who engage in ways that "defy the laws of physics"

**Instrumentation issue fixes:**
- Duplicate event detection
- Incorrect timestamp handling

**Important limitations:**
- Data cleansing **cannot fix missing events** — missing data may result from lossiness in the underlying data collection. Click logging is inherently a **tradeoff between fidelity and speed** (Kohavi, Longbotham and Walker 2010)
- Some filtering may **unintentionally remove more events from one variant than another** → can cause a **Sample Ratio Mismatch (SRM)** (see Chapter 3). This is a serious concern to watch for.

---

### Step 3: Enrich the Data

Some data can be parsed and enriched to provide useful dimensions or measures:

**Per-event enrichment examples:**
- Parse user agent raw string → add **browser family and version**
- Extract **day of week** from dates
- Mark an event as a **duplicate**
- Compute **event duration**

**Per-session enrichment examples:**
- Total number of events during the session
- Total session duration

**Experiment-specific annotations:**
- Annotate whether to **include this session** in the computation of experiment results
- Annotate **experiment transition information** (e.g., starting an experiment, ramping up, changing the version number) to help determine inclusion in results
- These annotations are pieces of business logic often added during enrichment **for performance reasons**

---

## Data Computation

Given processed data, compute segments and metrics, then aggregate to get **summary statistics** for each experiment:
- Estimated Treatment effect (e.g., delta of mean or percentiles)
- Statistical significance information (p-value, confidence interval, etc.)
- Identification of interesting segments (Fabijan, Dmitriev and McFarland et al. 2018)

### Two Common Architectural Approaches

*(Assuming the experimental unit is user, without loss of generality)*

**Approach 1: Materialize per-user statistics first**
- For every user, count pageviews, impressions, clicks, etc.
- Join that table to a table mapping users to experiments
- **Advantage:** Per-user statistics can be reused for **overall business reporting**, not just experiments
- For efficiency: consider a flexible way to compute metrics or segments only needed for one or a few experiments

**Approach 2: Fully integrate per-user metric computation with experiment analysis**
- Per-user metrics are computed on the fly as needed, without being materialized separately
- Typically requires a **shared definitions mechanism** to ensure consistency between the experiments computation pipeline and the overall business reporting pipeline
- **Advantage:** More flexibility per-experiment (may also save machine and storage resources)
- **Disadvantage:** Requires additional work to ensure **consistency across multiple pipelines**

---

### Speed and Efficiency at Scale

Speed and efficiency become critically important as experimentation scales:

- Bing, LinkedIn, and Google all **process terabytes of experiment data daily** (Kohavi et al. 2013)
- As segments and metrics multiply, computation becomes **very resource-intensive**
- Delays in scorecard generation add delays to decision-making, which is increasingly costly as experimentation becomes integral to the innovation cycle

**Historical evolution:**
- **Early days:** Bing, Google, and LinkedIn generated experiment scorecards **daily with ~24-hour delay** (e.g., Monday's data shows up by end-of-day Wednesday)
- **Today:** All have **near real-time (NRT) paths**

**NRT path characteristics:**
- Simpler metrics and computations (sums and counts, no spam filtering, minimal statistical tests)
- Operates directly on raw logs without the full data processing described above (except some real-time spam processing)
- Used to **monitor for egregious problems** — misconfigured or buggy experiments
- Can **trigger alerts and automatic experiment shut-off**

**Batch path:** Handles intra-day computation and updates to data processing and computation to ensure trustworthy experiment results are available in a timely manner.

---

### Platform Recommendations for Speed, Efficiency, Correctness, and Trust

Every experimentation platform should:

**1. Common metric definitions:**
- Establish a standard vocabulary so everyone builds the same data intuition
- Allows teams to discuss interesting product questions rather than relitigating definitions or investigating surprising deltas between similar-looking metrics produced by different systems

**2. Consistency in implementation:**
- Either a common implementation or a testing / ongoing comparison mechanism to ensure definitions are implemented consistently everywhere

**3. Change management:**
- Metrics, OEC, and segments will all evolve over time (see Chapter 4 on experimentation maturity)
- Specifying and propagating changes is a **recurring process**
- **Changing an existing metric definition** is often more challenging than adding or deleting one — key question: do you backfill historical data, and if so, for how long?

---

## Results Summary and Visualization

**Goal:** Visually summarize and highlight key metrics and segments to guide decision makers.

### Core Visualization Requirements

- **Highlight key tests** (e.g., SRM) — clearly indicate whether results are trustworthy. Microsoft's ExP platform **hides the scorecard entirely** if key tests fail.
- **Highlight the OEC and critical metrics**, but also show many other metrics (guardrails, quality, etc.)
- **Present metrics as relative change** with clear indication of statistical significance — use color-coding and filters so significant changes are salient
- **Segment drill-downs** — automatically highlight interesting segments to ensure decisions are correct and identify ways to improve the product for poorly performing segments (Wager and Athey 2018, Fabijan et al. 2018)
- **If triggering conditions exist:** Include the overall population impact in addition to the impact on the triggered population (see Chapter 20)

### Accessibility Across Technical Backgrounds

To truly scale experimentation, scorecards must be **accessible to people with various technical backgrounds** — from Marketers to Data Scientists, Engineers, and Product Managers. This requires:
- Ensuring executives and other decision makers **see and understand** the dashboard
- **Hiding debugging metrics** from less technical audiences to reduce confusion

Information accessibility:
- Establishes a **common language** for definitions
- Cultivates a **culture of transparency and curiosity**
- Encourages employees to run experiments
- Helps Finance **tie A/B test results to business outlook**

### Per-Metric Views Across Experiments

The visualization tool should not only show per-experiment results but also allow pivoting to **per-metric results across experiments**:
- Stakeholders need visibility into the **top experiments impacting the metrics they care about**
- If an experiment is hurting their metrics above some threshold, they may want to be **involved in the launch decision**
- A centralized platform can **unify views of both experiments and metrics**

**Two optional platform features for healthy decision-making:**

**1. Metric subscription with email digest:**
Allow individuals to subscribe to metrics they care about and receive an email digest with the top experiments impacting those metrics.

**2. Approval process for negative impacts:**
If an experiment has a negative impact on a metric, the platform initiates an approval process — forcing the experiment owner to **start a conversation with the metrics owners** before ramping up. This drives:
- Transparency regarding launch decisions
- Discussion that increases overall experimentation knowledge across the company

The visualization tool can also be a **gateway to institutional memory** (see Chapter 8).

---

### Managing Metric Proliferation at Scale

As an organization moves into the Run and Fly phases of experimentation maturity, the number of metrics can grow into the **thousands**. Suggested features:

**1. Metric categorization / tiering:**

| Company | Tier structure |
|---|---|
| **LinkedIn** | 1) Companywide 2) Product Specific 3) Feature Specific (Xu et al. 2015) |
| **Microsoft** | 1) Data quality 2) OEC 3) Guardrail 4) Local features/diagnostic (Dmitriev et al. 2017) |
| **Google** | Similar to LinkedIn |

The visualization tool provides controls to dig into different metric groups.

**2. Multiple testing management:**
As the number of metrics grows, multiple testing becomes more important (Romano et al. 2016). A common experimenter question: *"Why did this metric move significantly when it seems irrelevant?"* Solutions:
- Education helps
- Use **p-value thresholds smaller than the standard 0.05** — allows experimenters to quickly filter to the most significant metrics
- Well-studied approaches like the **Benjamini-Hochberg procedure** address multiple testing concerns (see Chapter 17)

**3. Metrics of interest (automated identification):**
When an experimenter reviews results, they have a set of metrics in mind — but there are always unexpected movements in other metrics worth examining. The platform can automatically identify these by combining:
- Importance of the metric to the company
- Statistical significance
- False positive adjustment

**4. Related metrics:**
A metric's movement (or lack thereof) can often be explained by related metrics. Examples:
- When CTR is up: is it because **clicks are up** or because **page views are down**? The reason leads to different launch decisions.
- High-variance metrics like **revenue**: having a more sensitive, lower-variance version (e.g., trimmed revenue or other indicators) allows more informed decisions


Here are the complete notes for Chapters 17–19:

---

# Chapter 17 — The Statistics Behind Online Controlled Experiments

## Why This Chapter Matters

Statistics are fundamental to designing and analyzing experiments. This chapter goes deeper on the statistics critical to experimentation, including **hypothesis testing** and **statistical power** (Lehmann and Romano 2005, Casella and Berger 2001, Kohavi, Longbotham et al. 2009).

---

## Two-Sample t-Test

Two-sample t-tests are the **most common statistical significance tests** for determining whether the difference we see between Treatment and Control is real or just noise (Student 1908; Wasserman 2004). They look at the **size of the difference between the two means relative to the variance**. The significance of the difference is represented by the **p-value** — the lower the p-value, the stronger the evidence that Treatment is different from Control.

### Setup

For a metric of interest Y (e.g., queries-per-user), assume the observed values for Treatment and Control users are **independent realizations** of random variables Y^t and Y^c.

**Hypotheses (Equation 17.1):**
```
H₀ : mean(Yᵗ) = mean(Yᶜ)     [Null hypothesis]
Hₐ : mean(Yᵗ) ≠ mean(Yᶜ)     [Alternative hypothesis]
```

### The t-Statistic

**Equation 17.2:**
```
T = Δ / √var(Δ)
```

where **Δ = Ȳᵗ − Ȳᶜ** is the difference between the Treatment average and the Control average — an **unbiased estimator** for the shift of the mean.

Because the samples are independent **(Equation 17.3):**
```
var(Δ) = var(Ȳᵗ − Ȳᶜ) = var(Ȳᵗ) + var(Ȳᶜ)
```

The t-statistic T is just a **normalized version of Δ**. Intuitively: the larger T is, the less likely it is that the means are the same — i.e., the more likely you are to reject the Null hypothesis.

---

## p-Value and Confidence Interval

The **p-value** is the probability that T would be at least this extreme if there really is **no difference** between Treatment and Control.

- By convention: p-value < **0.05** → "statistically significant"
- p-value < **0.01** → "very significant"
- There are ongoing debates calling for lower p-values by default (Benjamin et al. 2017)

### Common Misinterpretation of p-value

**Wrong interpretation:** The p-value is the probability that the Null hypothesis is true given the data observed.

**Correct interpretation:** The p-value is the **probability of observing the delta, or a more extreme delta, if the Null hypothesis is true.**

These two are related but different. Using Bayes' rule **(Equation 17.4):**

```
P(H₀ is true | Δ observed) = P(Δ observed | H₀ is true) × P(H₀ is true)
                               ────────────────────────────────────────────
                                           P(Δ observed)

                            = P(H₀ is true) × p-value
                              ──────────────────────────
                                    P(Δ observed)
```

To know whether the Null hypothesis is true based on data collected (posterior probability), you need not only a p-value but also the **prior likelihood that the Null hypothesis is true**.

### Confidence Interval

A **95% confidence interval** covers the true difference 95% of the time. It is equivalent to a p-value threshold of 0.05:
- The delta is statistically significant at 0.05 level if the 95% CI **does not contain zero**
- In most cases, the CI for the delta centers around the observed delta with an extension of about **two standard deviations** on each side
- This holds for any statistic that approximately follows a normal distribution, including percent delta

---

## Normality Assumption

In most cases, p-values are computed assuming the t-statistic T follows a **normal distribution** with mean 0 and variance 1 under the Null hypothesis. The p-value is the area under the normal curve.

### Common Misunderstanding

Many people think the normality assumption is an assumption on the **sample distribution** of metric Y itself — and consider this a poor assumption because almost no practical metrics are normally distributed.

**This is wrong.** The assumption is on the **distribution of the average** Ȳ, not Y itself.

In most online experiments, sample sizes are **at least in the thousands**. Even if Y is not normally distributed, Ȳ usually is — because of the **Central Limit Theorem** (Billingsley 1995).

*Figure 17.1 illustrates this:* As sample size n increases, the distribution of the mean of a beta-distributed Y becomes increasingly normal. n=1 is very skewed; n=100 is clearly bell-shaped.

### Minimum Sample Size Rule of Thumb

For the average Ȳ to have a normal distribution, you need at least **355s²** samples per variant (Kohavi, Deng et al. 2014), where s is the skewness coefficient **(Equation 17.5):**

```
s = E[(Y − E(Y))³] / Var(Y)^(3/2)
```

Some metrics — especially **revenue metrics** — have high skewness coefficients.

**Real example (Bing):** After capping Revenue/User at $10 per user per week:
- Skewness dropped from **18 to 5**
- Minimum sample needed dropped **tenfold** — from **114k to 10k**

> **Note:** This rule-of-thumb provides good guidance when |s| > 1 but does not offer a useful lower bound when the distribution is symmetric or has small skewness. Fewer samples are generally needed when skewness is smaller (Tyurin 2009).

For **two-sample t-tests** specifically, because you're looking at the *difference* of two variables with similar distributions, the number of samples needed for normality tends to be **fewer** — especially when Treatment and Control have equal traffic allocations, since the distribution of the difference is approximately symmetric (perfectly symmetric with zero skewness under H₀).

### How to Test If Your Sample Size Is Large Enough

Test it with **offline simulation**:
- Randomly shuffle samples across Treatment and Control to generate the null distribution
- Compare that distribution with the normal curve using **Kolmogorov–Smirnov** or **Anderson-Darling** tests (Razali and Wah 2011)
- Increase test sensitivity by focusing specifically on whether the Type I error rate is bounded by the preset threshold (e.g., 0.05)

### When Normality Assumption Fails

Do a **permutation test** (Efron and Tibshirani 1994) — see where your observation stands relative to the simulated null distribution. Even though permutation tests are expensive to run at scale, they are most often needed in **small sample size** settings, so the cost is manageable in practice.

---

## Type I/II Errors and Power

| Error Type | Definition |
|---|---|
| **Type I error** | Concluding there IS a significant difference when there is NO real difference (false positive) |
| **Type II error** | Concluding there is NO significant difference when there REALLY IS one (false negative) |

You control Type I error at 0.05 by concluding statistical significance only if p-value < 0.05.

**Tradeoff:** Using a higher p-value threshold → higher Type I error rate, but lower Type II error rate (less likely to miss a real difference).

### Statistical Power

Power is the **probability of detecting a difference between variants (rejecting the Null) when there really is a difference (Equation 17.6):**

```
Power = 1 − Type II error
```

Power is parameterized by **δ**, the minimum delta of practical interest. At a 95% confidence level **(Equation 17.7):**

```
Power_δ = P(|T| ≥ 1.96 | true diff is δ)
```

**Industry standard:** Achieve at least **80% power**.

### Required Sample Size Formula

Assuming equal Treatment and Control sizes, the total number of samples needed for 80% power is approximately **(Equation 17.8):**

```
n ≈ 16σ² / δ²
```

where σ² is the sample variance and δ is the difference between Treatment and Control.

**"But how do I know δ before I run the experiment?"**
You don't know the true δ — that's why you're running the experiment. But you do know the size of δ that would **matter in practice** (practical significance). For example: you could miss detecting a 0.1% revenue difference and that's acceptable, but a 1% revenue drop is not. Use the smallest δ that is practically significant — also called the **minimum detectable effect (MDE)**.

### Complexity in Online Experiments

For online experiments, sample size estimation is more complex because:
- Users visit **over time**, so experiment duration also determines actual sample size
- Depending on the randomization unit, **sample variance σ² can change over time**
- With triggered analysis (Chapter 20), both σ² and δ change as trigger conditions change across experiments

### Critical Misinterpretation of Power

Many people treat power as an **absolute property of a test** and forget it is **relative to the size of the effect you want to detect**.

An experiment with enough power to detect a 10% difference does **not necessarily** have enough power to detect a 1% difference.

*Figure 17.2 — "Spot the Difference" analogy:*
- Solid circle (smaller, harder to spot) = lower power
- Dashed circle (larger lily pad difference, easier to spot) = higher power

### Additional Error Types (Small Sample Settings)

Gelman and Carlin (2014) argue that for small sample size settings, it's also important to calculate:
- **Type S (Sign) error:** The probability that an estimate is in the **wrong direction**
- **Type M (Magnitude) error / Exaggeration ratio:** The factor by which the **magnitude of an effect might be overestimated**

---

## Bias

Bias arises when the estimate and the true value of the mean are **systematically different**. It can be caused by:
- A **platform bug**
- A **flawed experiment design**
- An **unrepresentative sample** (e.g., company employee accounts, test accounts)

See Chapter 3 for examples, prevention, and detection recommendations.

---

## Multiple Testing

With hundreds of metrics computed per experiment, a common question is: *"Why is this irrelevant metric significant?"*

**Simple illustration:** If you compute 100 metrics for an experiment, and your feature does nothing, how many metrics would you still see as statistically significant? With significance level at 5% and assuming independent metrics, the answer is **around five**. The problem worsens when examining **hundreds of experiments** and **multiple iterations** per experiment. This is the **"multiple testing" problem** — the number of false discoveries increases when testing multiple things in parallel.

### Common Approaches

| Approach | Type | Drawback |
|---|---|---|
| **Bonferroni correction** | Simple | Too conservative — uses p-value threshold of 0.05 ÷ number of tests |
| **Benjamini-Hochberg procedure** (Hochberg and Benjamini 1995) | More nuanced | Uses varying p-value thresholds for different tests; less accessible |

### Practical Rule-of-Thumb: Two-Step Approach

**Step 1: Separate all metrics into three groups:**
- **First-order metrics:** Those you *expect* to be impacted by the experiment
- **Second-order metrics:** Those *potentially* impacted (e.g., through cannibalization)
- **Third-order metrics:** Those *unlikely* to be impacted

**Step 2: Apply tiered significance levels to each group:**
- First-order → 0.05
- Second-order → 0.01
- Third-order → 0.001

**Bayesian interpretation:** This reflects how much you believe H₀ is true *before* running the experiment. The stronger the prior belief that the Null is true, the lower the significance level you should apply.

---

## Fisher's Meta-Analysis

When you want to **combine results from multiple experiments** testing the same hypothesis (e.g., an original experiment and its replication), Fisher provides a formal method.

**Setup:** Replication is done using either:
- **Orthogonal randomization** — different hash seed, same population
- **Users not allocated** to the original round of the experiment

Both produce **independent p-values**. Intuitively, if both p-values are below 0.05, that's stronger evidence than if only one is.

### Fisher's Method (Equation 17.9)

```
χ²₂ₖ = −2 × Σᵢ₌₁ᵏ ln(pᵢ)
```

where pᵢ is the p-value for the i-th hypothesis test. If all k Null hypotheses are true, this test statistic follows a **chi-squared distribution with 2k degrees of freedom**.

Extensions and alternatives:
- **Brown (1975):** Extends Fisher's method for cases when p-values are **not independent**
- **Edgington (1972), Mudholkar and George (1979):** Other p-value combination methods
- See Hedges and Olkin (2014) for more discussion

### When to Use Fisher's Method

Fisher's method (or any meta-analysis technique) is great for **increasing power and reducing false positives**. If an experiment is underpowered even after applying power-increasing techniques (maximum power traffic allocation from Chapter 15, variance reduction from Chapter 22), consider running two or more **orthogonal replications** of the same experiment and combining results using Fisher's method.

---

---

# Chapter 18 — Variance Estimation and Improved Sensitivity: Pitfalls and Solutions

## Why This Chapter Matters

Variance is the **core of experiment analysis**. Almost all key statistical concepts introduced in this book are related to variance: statistical significance, p-value, power, and confidence interval.

Two critical topics:
1. **Common pitfalls** (and solutions) in variance estimation
2. **Techniques for reducing variance** that improve sensitivity

> *"With great power comes small effect size."*

---

## Standard Variance Computation (Review)

For i = 1, ..., n independent identically distributed (i.i.d.) samples (where i is usually a user, but can also be a session, page, user-day, etc.):

```
Metric (average):         Ȳ = (1/n) Σᵢ Yᵢ

Sample variance:          var(Y) = σ̂² = 1/(n−1) Σᵢ (Yᵢ − Ȳ)²

Variance of the average:  var(Ȳ) = σ̂²/n
```

---

## Common Pitfalls in Variance Estimation

If you incorrectly estimate variance, the p-value and confidence interval will be wrong, making your hypothesis test conclusions wrong:
- **Overestimated variance** → false negatives (miss real effects)
- **Underestimated variance** → false positives (see effects that aren't there)

---

### Pitfall 1: Delta vs. Delta %

It is very common to use the **relative difference (percent delta)** instead of the absolute difference when reporting results. Decision makers usually understand the magnitude of a "1% session increase" better than "0.01 more sessions."

**Percent delta (Equation 18.1):**
```
Δ% = Δ / Ȳᶜ
```

**Variance of Δ (Equation 18.2):**
```
var(Δ) = var(Ȳᵗ − Ȳᶜ) = var(Ȳᵗ) + var(Ȳᶜ)
```

**Common mistake:** Estimating var(Δ%) by dividing var(Δ) by Ȳᶜ² — i.e., using var(Δ) / Ȳᶜ².

**Why this is wrong:** Ȳᶜ is itself a **random variable**, not a constant.

**Correct formula (Equation 18.3):**
```
var(Δ%) = var((Ȳᵗ − Ȳᶜ) / Ȳᶜ) = var(Ȳᵗ / Ȳᶜ)
```
→ This is a **ratio metric** — requires special treatment (see below).

---

### Pitfall 2: Ratio Metrics When Analysis Unit ≠ Experiment Unit

Many important metrics are **ratios of two metrics**:
- **CTR** = total clicks / total pageviews
- **Revenue-per-click** = total revenue / total clicks

Unlike clicks-per-user or revenue-per-user, these ratio metrics have an **analysis unit different from the user** (e.g., pageview or click). When the experiment is randomized by user, this creates a challenge.

**The violated assumption:**
The standard variance formula `var(Y) = 1/(n−1) Σ(Yᵢ − Ȳ)²` requires that samples (Y₁, ..., Yₙ) be **i.i.d. (or at least uncorrelated)**. This is satisfied when the analysis unit = experiment unit. For user-level metrics, each Yᵢ is one user's measurement — the i.i.d. assumption holds. For page-level metrics, Y₁, Y₂, Y₃ could all be from the **same user**, making them **correlated** ("within-user correlation"). The simple variance formula is then **biased**.

**Correct approach — rewrite as ratio of user-level metrics:**

Write the ratio metric M as **(Equation 18.4):**
```
M = X̄ / Ȳ
```

Because X̄ and Ȳ are jointly bivariate normal in the limit, M (as the ratio of two averages) is also normally distributed. By the **delta method**, variance can be estimated as **(Equation 18.5):**

```
var(M) = (1/Ȳ²)·var(X̄) + (X̄²/Ȳ⁴)·var(Ȳ) − (2X̄/Ȳ³)·cov(X̄, Ȳ)
```

For Δ%, since Ȳᵗ and Ȳᶜ are independent **(Equation 18.6):**
```
var(Δ%) = (1/Ȳᶜ²)·var(Ȳᵗ) + (Ȳᵗ²/Ȳᶜ⁴)·var(Ȳᶜ)
```

Note: When Treatment and Control means differ significantly, this is substantially different from the incorrect estimate var(Δ)/Ȳᶜ².

**For metrics that cannot be written as a ratio of user-level metrics** (e.g., 90th percentile of page load time), use **bootstrap** (Efron and Tibshirani 1994) — simulate randomization by sampling with replacement and estimate variance from many repeated simulations. Bootstrap is computationally expensive but broadly applicable and a good complement to the delta method.

---

### Pitfall 3: Outliers

Outliers come in various forms — most commonly from **bots or spam behaviors** performing many pageviews or clicks. Outliers have a big impact on both the mean and variance. In statistical testing, **the impact on variance tends to outweigh the impact on the mean**.

**Simulation result:**
In a simulation where Treatment has a positive true delta over Control, and a single positive outlier is added to the Treatment group:
- As outlier size increases → Treatment mean increases...
- But variance increases **even more**
- As a result (Figure 18.1): the **t-statistic decreases** as outlier size increases, and eventually the test is **no longer statistically significant**

**Solution:** Cap observations at a reasonable threshold.
- Example: human users are unlikely to perform a search over 500 times or have over 1,000 pageviews in one day
- Many other outlier removal techniques exist (Hodge and Austin 2004)

---

## Improving Sensitivity (Variance Reduction)

When running a controlled experiment, you want to detect the Treatment effect when it exists. **Sensitivity = power = ability to detect effects.** One key way to improve sensitivity is **reducing variance**. Here are the main techniques:

---

### Technique 1: Use a Metric with Smaller Variance

Create an evaluation metric with smaller variance while capturing similar information:
- **Number of searches** (high variance) → **Number of searchers** (lower variance)
- **Purchase amount** (real-valued, high variance) → **Purchase** (Boolean, lower variance)

**Real example (Kohavi et al. 2009):** Using conversion rate instead of purchasing spend reduced the required sample size by a **factor of 3.3**.

---

### Technique 2: Transform the Metric

- **Capping:** Cap extreme values at a reasonable threshold
- **Binarization:** Convert to a binary indicator
  - Netflix uses binary metrics indicating whether a user streamed more than X hours in a specified time period (Xie and Aurisset 2016), instead of using average streaming hours
- **Log transformation:** Effective for heavy long-tailed metrics — especially if interpretability is not a concern
  - However, some metrics like revenue may not be appropriate to optimize as a log-transformed version

---

### Technique 3: Use Triggered Analysis

See Chapter 20. This is a great way to **remove noise introduced by people not affected by the Treatment** — only analyze users who were actually exposed to the feature.

---

### Technique 4: Stratification, Control-Variates, or CUPED

**Stratification:**
- Divide the sampling region into strata, sample within each stratum separately, then combine results
- Common strata: platforms (desktop/mobile), browser types (Chrome, Firefox, Edge), day of week
- Usually results in smaller variance than estimating without stratification
- **Runtime stratification** is most theoretically correct but expensive to implement at scale
- Most applications use **post-stratification** — applying stratification retrospectively during analysis
- When sample size is large, post-stratification performs like stratified sampling; may not reduce variance as well with small samples and high variability

**Control-variates:**
- Similar idea to stratification, but uses covariates as **regression variables** instead of using them to construct strata

**CUPED (Controlled-experiment Using Pre-Experiment Data):**
- An application of control-variates that emphasizes **utilizing pre-experiment data** (Deng et al. 2013)
- Implementations and comparisons: Soriano 2017, Xie and Aurisset 2016, Jackson 2018, Deb et al. 2018
- Xie and Aurisset (2016) compare stratification, post-stratification, and CUPED on Netflix experiments

---

### Technique 5: Randomize at a More Granular Unit

For example, randomizing **per page** (rather than per user) can substantially increase sample size if you care about page-load-time.

**Disadvantages of sub-user randomization:**
- If the experiment makes noticeable UI changes, giving the same user **inconsistent UIs** makes for a bad user experience
- It is **impossible to measure user-level impact over time** (e.g., user retention)

---

### Technique 6: Design a Paired Experiment

Show the **same user both Treatment and Control** in a paired design. This removes between-user variability, achieving smaller variance.

**Popular method for ranked lists:** Interleaving design — interleave two ranked lists and present the joint list to users at the same time, observing which list's results get more clicks (Chapelle et al. 2012, Radlinski and Craswell 2013).

---

### Technique 7: Pool Control Groups

If you have several experiments splitting traffic, each with their own Control, consider **pooling separate Controls** into one larger shared Control group. Comparing each Treatment to this shared Control increases power for all experiments involved.

**Practical considerations:**
- If each experiment has its own trigger condition, it may be hard to **instrument all on the same Control**
- You may want to compare Treatments against each other directly — how much does statistical power matter in those cross-Treatment comparisons vs. against Control?
- There are benefits to having **equal-sized Treatment and Control** even though the pooled Control will likely be larger — balanced variants lead to faster normality convergence (Chapter 17) and less concern about cache sizes

---

## Variance of Other Statistics

Most discussions assume the statistic of interest is the **mean**. What about other statistics like **quantiles**?

For time-based metrics like page-load-time (PLT), it is common to use quantiles:
- **90th or 95th percentile** → user engagement-related load times
- **99th percentile** → server-side latency measurements

**Options:**
- Always use **bootstrap** to find tail probabilities — but gets computationally expensive as data size grows
- If the statistic follows a normal distribution **asymptotically**, estimate variance cheaply using **density estimation** — the asymptotic variance for quantile metrics is a function of the density (Lehmann and Romano 2005)

**Additional complication:** Most time-based metrics are at the **event/page level** while the experiment is randomized at **user level**. In this case, apply a combination of **density estimation and the delta method** (Liu et al. 2018).

---

---

# Chapter 19 — The A/A Test

## Why This Chapter Matters

Running A/A tests is a **critical part of establishing trust** in an experimentation platform. The idea is so useful because **tests fail many times in practice**, leading to re-evaluating assumptions and identifying bugs.

> *"If everything seems under control, you're just not going fast enough."* — Mario Andretti
>
> *"If everything is under Control, then you're running an A/A test."* — Ronny Kohavi

---

## What Is an A/A Test?

Split users into two groups as in a regular A/B test, but make **B identical to A** — hence the name A/A test (also called a **Null test**, Peterson 2004).

If the system is operating correctly, in **repeated trials**:
- About **5% of the time**, a given metric should be statistically significant with p-value < 0.05
- The distribution of p-values from repeated trials should be **close to uniform**

---

## Why A/A Tests?

The theory of controlled experiments is well understood, but practical implementations expose multiple pitfalls. A/A tests are highly useful for:

**1. Ensuring Type I errors are controlled (at ~5%) as expected.**
Standard variance calculations may be incorrect for some metrics, or the normality assumption may not hold. A/A tests failing at an unexpected rate point to issues that must be addressed.

**2. Assessing metrics' variability.**
Data from an A/A test shows how a metric's variance changes over time as more users are admitted — the expected reduction in variance of the mean may not materialize (Kohavi et al. 2012).

**3. Ensuring no bias exists between Treatment and Control users** — especially important if reusing populations from prior experiments.
A/A tests are very effective at identifying biases introduced at the platform level. **Example:** Bing uses continuous A/A testing to identify **carry-over effects (residual effects)**, where previous experiments impact subsequent experiments run on the same users (Kohavi et al. 2012).

**4. Comparing data to the system of record.**
A/A tests are often the first step before an organization starts using controlled experiments. If data is collected using a separate logging system, validate that key metrics (number of users, revenue, CTR) match the system of record. Sanity check: if the system of record shows X users visited the website and you ran Control and Treatment at 20% each, do you see ~20% × X users in each? Are you leaking users?

**5. Estimating variances for statistical power calculations.**
A/A tests provide metric variances that help determine how long to run your A/B tests for a given minimum detectable effect.

> **Strong recommendation:** Run continuous A/A tests in parallel with other experiments to uncover problems, including distribution mismatches and platform anomalies.

---

## Five Illustrative Examples of A/A Test Failures

---

### Example 1: Analysis Unit Differs from Randomization Unit

Randomizing by user but analyzing by pages is often desired — for example, alerting systems look at page-load-time and CTR in near real-time by aggregating every page.

**Two reasonable CTR definitions (setup):**
- n = number of users
- Kᵢ = number of pageviews for user i
- N = total pageviews = Σᵢ Kᵢ
- Xᵢⱼ = number of clicks for user i on their j-th page

**CTR Definition 1 — Count all clicks divided by total pageviews (Equation 19.1):**
```
CTR₁ = (Σᵢ Σⱼ Xᵢⱼ) / N
```

Example: User 1 has 0 clicks and 1 pageview; User 2 has 2 clicks across 2 pageviews:
```
CTR₁ = (0 + 2) / (1 + 2) = 2/3        (Equation 19.2)
```

**CTR Definition 2 — Average each user's CTR, then average all CTRs (Equation 19.3):**
```
CTR₂ = (Σᵢ (Σⱼ Xᵢⱼ / Kᵢ)) / n
```

Same example:
```
CTR₂ = (0/1 + 2/2) / 2 = (0 + 1) / 2 = 1/2        (Equation 19.4)
```

Both are valid definitions, but they yield **different results**. In practice, it is common to expose both metrics in scorecards. The authors **generally recommend Definition 2** as it is **more robust to outliers** (bots with many pageviews or frequent clicking).

**The variance problem:**
If the A/B test is randomized by user, the variance of CTR₁ is commonly (incorrectly) computed as **(Equation 19.5):**
```
VAR(CTR₁) = Σᵢ Σⱼ (Xᵢⱼ − CTR₁)² / N²
```

This is **incorrect** — it assumes the Xᵢⱼ are independent, which they are not (within-user correlation). To compute an unbiased variance estimate, use the **delta method** or **bootstrapping** (Tang et al. 2010, Deng et al. 2011, Deng, Lu and Litz 2017).

**How this was discovered:**
Not by recognizing the independence violation theoretically, but because in A/A tests, **CTR₁ was statistically significant far more often than the expected 5%** — a red flag that led to diagnosing the problem.

*Figure 19.1:* Histogram showing a **non-uniform p-value distribution** from A/A tests when variance is computed incorrectly (too many results near p = 0).
*Figure 19.2:* After applying the **delta method**, the distribution becomes **close to uniform** — the test is fixed.

---

### Example 2: Optimizely Encouraged Stopping When Results Were Significant (Peeking Problem)

The book *A/B Testing: The Most Powerful Way to Turn Clicks into Customers* (Siroker and Koomen 2013) suggested stopping experiments as soon as they reach statistical significance.

**Why this is wrong:** The statistics used assume a **single test at the end** of the experiment. "Peeking" — checking results before the experiment ends and stopping early — violates this assumption and leads to **many more false positives** than the expected 5%.

Early versions of Optimizely encouraged peeking and early stopping, leading to many false successes. When experimenters began running A/A tests, they realized this — leading to articles such as *"How Optimizely (Almost) Got Me Fired"* (Borden 2014).

To their credit, Optimizely worked with experts (Ramesh Johari, Leo Pekelis, David Walsh) and updated their methodology, calling it **"Optimizely's New Stats Engine"** (Pekelis 2015, Pekelis, Walsh and Johari 2015).

---

### Example 3: Browser Redirects

Suppose you want to A/B test an old vs. new website, where users in variant B are **redirected** to the new site.

**Spoiler: B will lose with high probability.** This approach has three fatal flaws (Kohavi and Longbotham 2010):

**1. Performance differences:**
Redirected users suffer an **extra redirect**. In the lab this seems fast, but users in other regions may experience **1–2 second wait times**.

**2. Bots:**
Robots handle redirects differently:
- Some may not redirect at all
- Some treat the redirect as a new area and **crawl deeply**, creating large amounts of non-human traffic that impacts key metrics
- Normally, small-activity bots are distributed uniformly across variants, so they cancel out. But a new or updated site is likely to trigger **different bot behavior** that doesn't cancel out

**3. Bookmarks and shared links cause contamination:**
Users going deep into a website via bookmarks or shared links still need to be redirected. Those redirects must be **symmetric** — you must also redirect Control users to site A. This degrades the Control group.

**Conclusion:** Redirects usually **fail A/A tests**. Solutions:
- Build things so there are **no redirects** (e.g., server-side returns one of two home pages)
- Execute a redirect for **both Control and Treatment** (which degrades the Control group but at least makes it symmetric)

---

### Example 4: Unequal Percentages

Uneven splits (e.g., 10%/90%) may suffer from **shared resources** providing a clear benefit to the larger variant.

**Specifically:** Least Recently Used (LRU) caches shared between Control and Treatment have **more cache entries for the larger variant** — because the larger variant generates more requests, warming the cache more effectively.

> Note: Experiment IDs must always be part of any caching system that could be impacted by the experiment, as experiments may cache different values for the same hash key. See Chapter 18.

**Practical workaround:** Sometimes it's easier to run a **10%/10% experiment** (not utilizing 80% of the data) to avoid LRU caching issues. This must be done at runtime — you cannot run 10%/90% and then throw away data after the fact.

A 50/50 A/A test may pass, but if you run experiments at 90%/10%, you should also run A/A tests with those proportions.

**Additional problem — non-uniform normality convergence:**
If you have a highly skewed metric distribution, the Central Limit Theorem guarantees the average converges to Normal — but the **rate of convergence differs** between unequal groups. In an A/B test, what matters is the delta of the metric, and the delta may be **more Normal when the two constituents have the same distribution** (even if neither is Normal). See Chapter 17 for details.

---

### Example 5: Hardware Differences

Facebook had a service running on a fleet of machines. They built a new V2 of the service and wanted to A/B test it. They ran an A/A test between the new and old fleet. Even though they **thought the hardware was identical, the A/A test failed**. Small hardware differences can lead to unexpected differences (Bakshy and Frachtenberg 2015).

---

## How to Run A/A Tests

**Always run a series of A/A tests before utilizing an A/B testing system.**

Ideal process:
1. **Simulate a thousand A/A tests**
2. Plot the **distribution of p-values**
3. If the distribution is **far from uniform**, you have a problem
4. Do not trust your A/B testing system before resolving this issue

**Theoretical basis:** When the metric of interest is continuous and you have a simple Null hypothesis (e.g., equal means), the distribution of p-values under the Null should be **uniform** (Dickhaus 2014, Blocker et al. 2006).

### Practical Shortcut — "Replay the Last Week"

Running a thousand A/A tests live is expensive. Instead:
- **Replay** the last week of stored raw data
- For each iteration, pick a **new randomization hash seed** for user assignment and replay the week of data, splitting users into two groups
- Generate the p-value for each metric of interest (usually tens to hundreds of metrics)
- **Accumulate p-values into histograms**, one per metric
- Run a **goodness-of-fit test** (Anderson-Darling or Kolmogorov-Smirnov) to assess whether distributions are close to uniform

**Limitations of replay approach:** Will not catch performance issues or shared resource problems (like the LRU cache issue in Example 4) — but it is highly valuable and identifies many issues.

---

## When the A/A Test Fails

Common p-value scenarios indicating failure of the goodness-of-fit for uniform distribution (Mitchell et al. 2018):

**Scenario 1: Distribution is clearly skewed and not close to uniform**
A common problem is incorrect variance estimation (see Chapter 18). Check:
- **(a) Is the independence assumption violated?** (e.g., CTR example — randomization unit ≠ analysis unit) → Deploy the **delta method or bootstrapping** (Chapter 15)
- **(b) Does the metric have a highly skewed distribution?** Normal approximation may fail. In some cases, minimum sample size needs to be **over 100,000 users** (Kohavi et al. 2014). Consider **capped metrics** or setting minimum sample sizes (Chapter 17)

**Scenario 2: Large mass around p-value ≈ 0.32 → outlier problem**
Assume a single very large outlier o in the data. When computing the t-statistic **(Equation 19.6):**
```
T = Δ / √var(Δ)
```

The outlier falls into one of the two variants:
- The delta of means will be close to **o/n** (or its negation) — all other numbers swamped by the outlier
- The variance of the mean for that variant will be close to **o²/n²**
- So T ≈ 1 or T ≈ −1, which maps to a **p-value of ~0.32**

This means the t-test will **rarely produce statistically significant results** (see Chapter 18). Investigate the source of the outlier or cap the data.

**Scenario 3: A few point masses with large gaps**
This happens when data is **single-valued** (e.g., all zeros) with a few rare non-zero instances. The delta of means can only take a few discrete values → p-value can only take a few discrete values. The t-test is not accurate here — but this is less serious than Scenario 2, because if a new Treatment causes the rare event to happen often, the Treatment effect will be large and statistically significant on its own.

---

## Ongoing A/A Testing

Even after an A/A test passes, **regularly run A/A tests concurrently with A/B tests** to:
- Identify **regressions** in the system
- Catch new metrics that are failing because their distribution changed
- Detect when **outliers start showing up** in previously well-behaved metrics


# Trustworthy Online Controlled Experiments — Study Notes
## Chapters 20 & 21: Triggering for Improved Sensitivity & Sample Ratio Mismatch

---

# Chapter 20: Triggering for Improved Sensitivity

> *"Be sure you positively identify your target before you pull the trigger."* — Tom Flynn

---

## 20.1 Core Idea

**Triggering** is a method for improving **sensitivity (statistical power)** by filtering out users who could not have been impacted by the experiment.

> A user is **triggered** into the analysis if there is (potentially) some difference in the system or user behavior between the variant they are in and the counterfactual (any other variant).

### Why It Works

If you make a change that only impacts some users, the Treatment effect of those who are **not** impacted is exactly **zero**. Including them in the analysis only adds **noise** and reduces statistical power.

**Key requirement:** Always perform the analysis step for **at least all triggered users**. Log triggering events at runtime to make identification easier.

---

## 20.2 Examples of Triggering (Increasing Complexity)

### Example 1: Intentional Partial Exposure

- Change is targeted at a **specific segment** (e.g., US users only)
- Only analyze users from that segment — non-US users have zero Treatment effect and add pure noise
- **Important edge case:** Include "mixed" users (those who visited from both the US and other countries) if they could have seen the change
  - Include **all their activity** after exposure, even activity outside the US — they were exposed and residual effects may persist

**This applies to any partial exposure:**
- Users of a specific browser (e.g., Edge)
- Users whose shipping address is in a given zip code
- Heavy users (e.g., visited at least 3 times in the last month)

> **Critical:** The definition of "heavy user" or any trigger criterion must be based on data **prior to the experiment start** — not on data that could be impacted by the Treatment itself.

---

### Example 2: Conditional Exposure

- Change only affects users who **reach a specific part of the product**
- As soon as the user encounters the change, they are triggered into the experiment

**Common examples:**
1. A change to checkout → only trigger users who **started checkout**
2. A change to co-editing (Microsoft Word / Google Docs) → only trigger users **participating in collaboration**
3. A change to the unsubscribe screen → only trigger users who **see that screen**
4. A change to the weather answer on a search results page → only trigger users who **issued a query that produced a weather answer**

---

### Example 3: Coverage Increase

- **Scenario:** Free shipping currently offered for carts ≥ $35. Treatment lowers threshold to $25.
- Only users with a cart **between $25 and $35** are affected:
  - Cart > $35 → same offer in both variants → Treatment effect = 0
  - Cart < $25 → same offer in both variants → Treatment effect = 0
- **Only trigger** users who see the free shipping offer at the $25–$35 range

**Venn diagram intuition:**
- Control = circle of users getting free shipping at $35+
- Treatment = larger circle including users at $25+
- Only users in **T \ C** (Treatment minus Control overlap) are triggered
- Users in the intersection see the **same offer** → Treatment effect = 0, no need to trigger them

> **Assumption:** If free shipping is never advertised elsewhere on the site, there is no earlier trigger point. If it *is* advertised, any user who sees a different advertisement is immediately a trigger point.

---

### Example 4: Coverage Change (More Complex)

- **Scenario:** Control = free shipping for cart ≥ $35. Treatment = free shipping for cart ≥ $25 **except** users who returned an item within the last 60 days.
- The groups are no longer simply nested — there is a more complex overlap

**Solution:** Both Control and Treatment must evaluate the **counterfactual** — i.e., what would have happened under the other variant — and mark users as triggered only if the two variants would have produced a **different outcome** for that user.

---

### Example 5: Counterfactual Triggering for Machine Learning Models

- **Scenario:** You trained a new ML classifier (V2) for promotions or a new recommender model. V2 performed better in offline tests. Now you A/B test it.
- **Key insight:** If the new model makes the **same classification or recommendation** as the old model for a given user, the Treatment effect for that user is zero.

**How to implement:**
- **Control variant:** Runs both the Control model and the Treatment model. Exposes users to Control output. Logs both outputs.
- **Treatment variant:** Runs both models. Exposes users to Treatment output. Logs both outputs.
- A user is triggered **only if the actual and counterfactual outputs differ**.

**Cost considerations:**
- Computational cost rises — model inference effectively doubles with one Treatment variant
- Latency may be impacted if models are run sequentially rather than concurrently
- The controlled experiment **cannot expose differences in execution speed or memory** between the two models, since both are run in all variants

---

## 20.3 Numerical Example — Power Gains from Triggering

*(Kohavi, Longbotham et al. 2009)*

**Minimum sample size formula** for 95% confidence level, 80% power:

$$n = \frac{16\sigma^2}{\Delta^2}$$

Where:
- $\sigma^2$ = variance of the OEC metric
- $\Delta$ = minimum detectable effect (the change you want to detect)

### Without Triggering (All Users):
- E-commerce site: 5% of users make a purchase → $p = 0.05$
- $\sigma^2 = p(1-p) = 0.05 \times 0.95 = 0.0475$
- Want to detect a 5% relative change: $\Delta = 0.05 \times 0.05 = 0.0025$
- $n = \frac{16 \times 0.0475}{0.0025^2} = 121,600$ users

### With Triggering (Checkout Users Only):
- Only 10% of users initiate checkout → conditional purchase rate: $p = 0.5$
- $\sigma^2 = 0.5 \times 0.5 = 0.25$
- Want to detect a 5% relative change: $\Delta = 0.5 \times 0.05 = 0.025$
- $n = \frac{16 \times 0.25}{0.025^2} = 6,400$ checkout users needed

**Implication:**
- 90% of users never initiate checkout → total enrolled users needed = $6,400 / 0.10 = 64,000$
- This is roughly **half** the 121,600 users needed without triggering
- The experiment can achieve the same statistical power in **approximately half the time** (and due to repeat users, reaching half the users typically takes less than half the calendar time)

---

## 20.4 Optimal vs. Conservative Triggering

### Optimal Triggering
- Trigger only users for whom there was **some difference** between the actual variant and the counterfactual
- Requires logging the output of **all variants** for every user — both actual and counterfactual

### Conservative Triggering (Non-Optimal but Acceptable)
- Include **more users** than strictly necessary
- Does **not** invalidate the analysis — it only sacrifices some statistical power
- Acceptable when the simpler trigger does not identify many more users than the optimal trigger

**Examples of conservative triggering:**

1. **Multiple Treatments:** Instead of logging the full output of each variant for every user, just log a Boolean indicating whether the variants differed. This may include users whose Treatment effect vs. a specific other variant is zero (e.g., Control = Treatment1 for some users, but Treatment2 was different). When comparing just Control vs. Treatment1, some zero-effect users get included.

2. **Post-hoc analysis:** If counterfactual logging failed (e.g., the recommendation model at checkout didn't log counterfactuals), use a proxy trigger like "user initiated checkout." This includes more users than those who actually saw a different recommendation, but still removes the 90% of users who never initiated checkout and had zero Treatment effect.

---

## 20.5 Overall Treatment Effect (Dilution)

**Critical concept:** When you compute the Treatment effect on the triggered population, you **must dilute it** to the overall user base. This is called the **diluted impact** or **site-wide impact**.

### Common Pitfall ❌
> "If I improved revenue by 3% for 10% of users, I improved overall revenue by 10% × 3% = 0.3%."

**This is WRONG.** The overall impact could be anywhere from **0% to 3%** depending on how much the triggered users contribute to overall revenue.

### Correct Examples

**Example 1:** Change was made to checkout; triggered users = those who initiated checkout. If the **only** way to generate revenue is to go through checkout, then a 3% improvement in triggered revenue = a 3% improvement in overall revenue. No dilution needed.

**Example 2:** Change was made to very low spenders (spend 10% of the average user). Improved 3% for 10% of users who spend 10% of average:
$$\text{Overall improvement} = 3\% \times 10\% \times 10\% = 0.03\%$$
— a negligible impact.

### Notation

| Symbol | Meaning |
|---|---|
| $\omega$ | Overall user universe |
| $\theta$ | Triggered population |
| $C$ | Control |
| $T$ | Treatment |
| $M_{\omega C}$ | Metric value for untriggered Control |
| $M_{\omega T}$ | Metric value for untriggered Treatment |
| $M_{\theta C}$ | Metric value for triggered Control |
| $M_{\theta T}$ | Metric value for triggered Treatment |
| $\Delta_\theta$ | Absolute effect on triggered population = $M_{\theta T} - M_{\theta C}$ |
| $\delta_\theta$ | Relative effect on triggered population = $\Delta_\theta / M_{\theta C}$ |
| $\tau$ | Triggering rate = $N_{\theta C} / N_{\omega C}$ |

### Correct Formulas for Diluted Impact

**Formula 1 — Absolute Treatment effect divided by the total baseline:**

$$\frac{\Delta_\theta \times N_{\theta C}}{M_{\omega C} \times N_{\omega C}}$$

**Formula 2 — Relative Treatment effect × triggering rate:**

$$\frac{\Delta_\theta}{M_{\omega C}} \times \tau$$

These two are mathematically equivalent (since $\tau = N_{\theta C} / N_{\omega C}$).

### Wrong Formula ❌

$$\frac{\Delta_\theta}{M_{\theta C}} \times \tau$$

This formula is only accurate when the triggered population is a **random sample** of the overall population. If the triggered population is **skewed** (e.g., high spenders), this formula is off by a factor of $M_{\omega C} / M_{\theta C}$.

### Ratio Metrics Note

For ratio metrics, more refined formulas must be used (Deng and Hu 2015). Ratio metrics can cause **Simpson's Paradox** (see Chapter 3) — the ratio in the triggered population improves, but the diluted global impact regresses.

---

## 20.6 Trustworthy Triggering

Two essential checks to validate that triggering is done correctly:

### Check 1: Sample Ratio Mismatch (SRM)
- If the overall experiment has no SRM, but the **triggered analysis shows an SRM**, then bias has been introduced
- Root cause: counterfactual triggering is not implemented correctly
- See Chapter 21 for detailed SRM discussion

### Check 2: Complement Analysis
- Generate a scorecard for **never-triggered users** — users who were never exposed to any difference between variants
- This scorecard should look like an **A/A test** (see Chapter 19) — no significant differences
- If more metrics than expected are statistically significant in the never-triggered group, the trigger condition is **incorrect** — you have influenced users who are supposedly outside the triggered population

---

## 20.7 Common Pitfalls

### Pitfall 1: Experimenting on Tiny Segments That Are Hard to Generalize

- If your goal is to improve a metric for the **overall population**, the **diluted value** is what matters
- Example: 5% improvement for a triggered population that is 0.1% of users → diluted impact is negligible (τ = 0.001)
- **Amdahl's Law analogy** from computer architecture: Speeding up a tiny part of the system yields almost no system-wide benefit

**Important exception — Small ideas that generalize:**
> In August 2008, MSN UK ran an experiment where the link to Hotmail opened in a new tab, increasing homepage engagement by 8.9% for triggered users who clicked the Hotmail link. This was a massive improvement, but a small segment. Over several years, this idea was generalized. By 2011, MSN US ran a large experiment (12 million+ users) opening search results in a new tab/window — engagement (clicks-per-user) increased by 5%. This became one of the best features MSN ever implemented. *(Kohavi et al. 2014; Kohavi and Thomke 2017)*

### Pitfall 2: A Triggered User Is Not Kept Triggered for the Remaining Experiment Duration

- Once a user triggers, the analysis must include them **going forward**, not just for the session or day they triggered
- The Treatment may impact their **future behavior** through residual effects
- **Day-by-day or session-by-session analysis** of triggered users is susceptible to this:
  - Example: If Treatment is so terrible that users drastically reduce their visits, analyzing by day or session will **underestimate the Treatment effect** — because there are fewer sessions to analyze for impacted users
  - If visits-per-user hasn't changed significantly, you can gain power by looking at **triggered visits**

### Pitfall 3: Performance Impact of Counterfactual Logging

- To log the counterfactual, both Control and Treatment execute **each other's model/code**
- If one variant's model is significantly **slower** than the other, this latency difference will **not be visible** in the controlled experiment (both variants are running both models)

**Two mitigations:**

1. **Awareness and logging:** Log the execution timing for each model separately, so they can be directly compared outside the experiment
2. **Run an A/A'/B experiment:**
   - A = original system (Control, no counterfactual logging)
   - A' = original system **with** counterfactual logging
   - B = new Treatment with counterfactual logging
   - If A and A' are significantly different, counterfactual logging is making a measurable performance impact

> **Additional note:** Counterfactual logging makes it hard to use **shared controls** (see Chapters 12 and 18), as shared controls typically run without code changes. Triggering conditions may need to be determined through other means, risking suboptimal or incorrect trigger conditions.

---

## 20.8 Open Questions

### Open Question 1: What Is the Triggering Unit?

When a user triggers, should you:
- Take only the logged activities **after the triggering point** (cleanest causal interpretation, but sessions become partial with abnormal metrics — e.g., clicks before checkout = 0)?
- Take the **whole session**?
- Take the **whole day**?
- Take **all user activities from experiment start** (easiest computationally; causes a small loss of statistical power)?

> There is no single right answer — the choice involves tradeoffs between causal cleanliness, metric normality, and statistical power.

### Open Question 2: Plotting Metrics Over Time

- Plotting a metric over time with **increasing numbers of users** usually leads to **false trends** (Kohavi et al. 2012; Chen, Liu and Xu 2019)
- Best practice for non-triggered experiments: each day shows only **users who visited that day**
- **The triggered version of this problem:** On Day 1, 100% of triggered users are new triggers. On Day 2, some users triggered on Day 1 are just returning — they are not "newly triggered" that day. False trends appear, usually as a **decreasing Treatment effect over time**
- Possible fix: Plot each day with only users who **visited and triggered on that day**
- **Key tension:** The single-day numbers and the overall (cross-day) numbers will not match, since the overall Treatment effect must include all days

---

# Chapter 21: Sample Ratio Mismatch and Other Trust-Related Guardrail Metrics

> *"The major difference between a thing that might go wrong and a thing that cannot possibly go wrong is that when a thing that cannot possibly go wrong goes wrong it usually turns out to be impossible to get at or repair."* — Douglas Adams

---

## 21.1 Background: Why Guardrail Metrics Exist

Experiments fail silently more often than people expect. Many experimenters assume that their experiment executed according to design. When this assumption fails, the analysis is **heavily biased** and many conclusions are **invalid**.

There are two types of guardrail metrics:
- **Organizational guardrails** (Chapter 7): Protect the business (e.g., latency, revenue floors)
- **Trust-related guardrails** (this chapter): Protect the internal validity of the experiment itself

**The SRM guardrail should be included in every experiment.**

Multiple companies have documented SRMs and highlighted their value: Kohavi and Longbotham (2017), Zhao et al. (2016), Chen, Liu and Xu (2019), Fabijan et al. (2019). At **Microsoft, ~6% of experiments exhibited an SRM**.

---

## 21.2 Sample Ratio Mismatch (SRM)

### Definition

The **Sample Ratio Mismatch (SRM)** metric checks whether the **ratio of users between variants** matches the experimental design.

- If you designed the experiment for a 1:1 ratio (50% Control, 50% Treatment), the observed ratio should be close to 1:1
- The decision to expose a user to a variant is **independent of the Treatment** — so the ratio should match by design
- Use a **t-test or chi-squared test** to compute the p-value for the observed ratio

> **Rule:** When the p-value for the Sample Ratio metric is low, **all other metrics are probably invalid**. Do not interpret any other results until the SRM is resolved.

---

### Scenario 1: Clear SRM

- Design: 50/50 split
- Observed: Control = 821,588 users; Treatment = 815,482 users
- Ratio = 0.993 (expected: 1.0)
- p-value = 1.8E-6

> The probability of seeing this ratio or more extreme (given 50/50 design) is less than **1 in 500,000**. It is far more likely that there is a **bug in the experiment implementation**. Do not trust any other metrics.

---

### Scenario 2: SRM with Apparently Good Metrics

- Design: 50/50 split
- Ratio = 0.994
- p-value = 2E-5 (still extremely unlikely)
- All five success metrics showed improvement with very small p-values (all < 0.05, four < 0.0001)

**But:** The right column of the scorecard (Figure 21.1) shows the same metrics for ~96% of users after **removing a segment** — users using an old version of Chrome, which was the root cause of the SRM. A bot was also improperly classified due to changes in the Treatment.

**After removing the problematic segment:** The remaining 96% of users are properly balanced, and **none of the five success metrics show statistically significant movement**.

> **Lesson:** An SRM can make a neutral or even bad experiment **look like a big win**. Never rationalize away an SRM because the metrics "look good."

---

## 21.3 SRM Causes

### 1. Buggy Randomization

- Simple Bernoulli randomization sounds easy but becomes complex with:
  - **Ramp-up procedures** (e.g., starting at 1%, ramping to 50%)
  - **Exclusions** (users in experiment X must not be in experiment Y)
  - **Covariate-balancing adjustments** using historical data (see hash seed in Chapter 19)

**Real example:**
> The Treatment was exposed to the Microsoft Office organization internally at 100%. An external experiment was then started at 10%/10%. The relatively small set of additional internal Office users in Treatment (who are heavy users) was enough to skew results and make the Treatment look artificially good. When these internal users were removed, the strong Treatment effect disappeared.

---

### 2. Data Pipeline Issues

- Bot filtering is a common source of SRMs
- Example from Scenario 2: A bot was misclassified due to changes in the Treatment (the Treatment changed behavior in a way that caused the bot-detection heuristic to classify it differently)
- At **Bing, over 50% of US traffic is filtered as bots**; in **China and Russia, ~90%** is bot-generated
- **Real edge case:** At MSN, a Treatment was so effective at increasing usage that real power-users exceeded a usage-based heuristic threshold and were **classified as bots**. Result: the Treatment looked significantly *worse* because its best users were excluded

---

### 3. Residual Effects After Restarting an Experiment

- If a bug is found and the experiment is restarted, there may be reluctance to re-randomize users (to avoid disrupting the UX)
- Analysis start date is set to after the bug fix
- If the bug was serious enough that some users **abandoned**, those users won't return — causing an SRM

---

### 4. Bad Trigger Condition

- Trigger condition must include **any user that could have been impacted**
- **Classic example — Redirect SRM:** Website A redirects some users to A' (the new site being tested). Some users are lost during the redirect. If you only count users who successfully arrive at A', you'll see an SRM, because some Treatment users were lost in transit and never counted
- The correct trigger is at the point of **redirect initiation**, not arrival at A'

---

### 5. Triggering Based on Attributes Impacted by the Experiment

- **Example:** Running a campaign on "dormant users" (defined by a dormant attribute in the user profile database). If Treatment is effective at re-activating dormant users, then at the end of the experiment, those users are no longer dormant. If you identify the triggered population using the dormant attribute **at the end of the experiment**, previously dormant Treatment users who became active will be **excluded** → SRM.
- **Fix:** Always trigger based on the attribute state **before the experiment started** (or before each user was assigned)
- **Extra caution with ML-based triggers:** Models may be **updated during the experiment** and impacted by the Treatment effect — making them especially unreliable as trigger conditions

---

## 21.4 Debugging SRMs

> When the p-value for the Sample Ratio guardrail is low, **reject the hypothesis that the design is properly implemented** and assume there is a bug somewhere. Do not examine any other metrics (except to help debug).

Debugging an SRM is hard — companies typically build dedicated internal tooling.

### Step-by-Step Debugging Guide

**Step 1: Validate there is no difference upstream of the randomization/trigger point**
- If you changed a checkout feature and are analyzing from the checkout point, confirm there is **no difference between variants upstream of checkout**
- Example: You cannot advertise "50% off vs. two-for-one" on the homepage if you're only analyzing from checkout — the homepage mention would be an earlier trigger point
- Real case: The Bing Image team found that their image search experiments sometimes impacted regular Bing web search results (by serving image results inline), causing upstream contamination and SRMs

**Step 2: Validate that variant assignment is correct**
- Most systems start with hashing-based randomization, but assignment gets complicated with concurrent experiments and isolation groups
- **Real example:** Experiment 1 changes font color from black to dark blue. Concurrent Experiment 2 changes background color but filters to users with font set to black. Experiment 2's code "steals" users from Experiment 1 — but only from the black-font variant. This causes an SRM in Experiment 1.

**Step 3: Follow each stage of the data processing pipeline**
- At each stage, check whether a sample ratio mismatch exists
- **Common culprit: bot filtering** — check whether the Treatment changes behavior in ways that interact with bot-detection heuristics

**Step 4: Exclude the initial period and re-check**
- Did both variants start at exactly the same time?
- In systems with shared controls, starting the Treatment later can cause problems even if the analysis period starts after the Treatment is live:
  - Caches take time to prime
  - App pushes take time to propagate
  - Phones may be offline, causing delays

**Step 5: Segment the Sample Ratio**
- Look at **each day separately** — was there an anomalous event? Did someone ramp experiment percentages on a particular day? Did another experiment start and steal traffic?
- Look at **browser segments** — is one browser showing an imbalanced ratio? (as in Scenario 2)
- Compare **new users vs. returning users** — do they show different ratios?

**Step 6: Check intersection with other concurrent experiments**
- Treatment and Control should have **similar distributions of users across other concurrent experiments**
- If one variant has disproportionately more users from another experiment, cross-experiment contamination has occurred

### Resolution

- In some cases, the root cause of the SRM can be fixed **during analysis** (e.g., excluding bots that were misclassified)
- In other cases — particularly where a segment of users was **not properly exposed to Treatment** (e.g., a browser bug excluded some users from receiving the Treatment) — it is better to **rerun the experiment** from scratch

---

## 21.5 Other Trust-Related Guardrail Metrics

Beyond SRM, additional metrics can detect that something is wrong with the experiment infrastructure (Dmitriev et al. 2017):

### 1. Telemetry Fidelity

- Click tracking via web beacons is **lossy** — less than 100% of clicks are properly recorded (Kohavi, Messner et al. 2010)
- If the Treatment changes the click loss rate, results may appear better or worse than they actually are
- **Detection methods:**
  - Monitor clicks through internal referrers to the website
  - Use **dual logging** for clicks (sometimes used in ad clicks, which require high fidelity)
  - Track the **loss rate itself** as a guardrail metric

### 2. Cache Hit Rates

- Shared caches can violate SUTVA (Chapter 3 and Kohavi and Longbotham 2010)
- If the Treatment changes cache behavior (e.g., generates more cache hits), Control users may benefit — a form of indirect interference
- **Monitoring cache hit rates** as a guardrail can surface unexpected shared-resource effects that threaten trustworthiness

### 3. Cookie Write Rate

- The **rate at which a variant writes permanent (non-session) cookies** should be consistent across variants
- This is called **cookie clobbering** (Dmitriev et al. 2016) and can severely distort other metrics due to browser bugs

**Real example:**
> An experiment at Bing wrote a cookie that wasn't used anywhere and set it to a random number with every search response page. The results showed **massive degradations in all key user metrics** — sessions-per-user, queries-per-user, and revenue-per-user — even though the cookie itself had no functional purpose. The cookie write triggered a browser bug that clobbered other cookies, corrupting user state.

---

## Key Takeaways Across Both Chapters

### Chapter 20 — Triggering
1. **Triggering = filtering to users who could have been affected.** Users with zero Treatment effect add only noise.
2. **The optimal trigger is counterfactual-based.** Log both the actual and counterfactual output for each user; trigger on the difference.
3. **Conservative triggers are fine** — they cost power, not validity.
4. **Never naively multiply trigger rate × triggered effect** to get site-wide impact. Use the correct dilution formulas based on baseline metric values.
5. **Run complement analysis** (A/A on never-triggered users) to validate your trigger condition.

### Chapter 21 — SRM
1. **SRM is a hard stop.** When the sample ratio p-value is low, discard all other metrics immediately.
2. **SRM can make a neutral experiment look like a winner.** The Bing scorecard is a concrete demonstration.
3. **~6% of Microsoft experiments had SRMs** — this is not a rare edge case.
4. **The SRM causes are diverse:** buggy randomization, data pipeline issues, residual effects from restarts, bad triggers, and experiment interaction.
5. **Debug systematically:** check upstream, check assignment, walk the pipeline, check segments, check concurrent experiments.
6. **Complement guardrails** — telemetry fidelity, cache hit rates, cookie write rate — catch infrastructure bugs that SRM alone might miss.


Here are comprehensive notes for Chapters 22 and 23:

---

# Trustworthy Online Controlled Experiments — Study Notes
## Chapters 22 & 23: Leakage/Interference & Long-Term Treatment Effects

---

# Chapter 22: Leakage and Interference between Variants

> *"It doesn't matter how beautiful your theory is, it doesn't matter how smart you are. If it doesn't agree with experiment, it's wrong."* — Richard Feynman

---

## 22.1 The Core Assumption: SUTVA

Most A/B experiment analyses rest on the **Rubin Causal Model** (Imbens & Rubin 2015), which makes a key assumption called **SUTVA**:

### Stable Unit Treatment Value Assumption (SUTVA)

> The behavior of each unit in the experiment is **unaffected by variant assignment to other units**.

Formally:

$$Y_i(z) = Y_i(z_i)$$

Where:
- $z = (z_1, z_2, ..., z_n)$ is the full variant assignment vector for all $n$ units
- The outcome of unit $i$ depends *only* on its own assignment $z_i$, not on what others are assigned

**When SUTVA holds (typical case):** A user who likes a new checkout flow will purchase more — and this behavior is independent of others using the same site.

**When SUTVA fails:** The analysis produces **potentially incorrect conclusions**. This failure is called **interference** (also: *spillover* or *leakage*).

---

## 22.2 Two Types of Interference

### 1. Direct Connections
Two units are directly connected if they are:
- Friends on a social network
- Physically co-located at the same time

### 2. Indirect Connections
Two units share a latent variable or a **shared resource**, e.g.:
- Both Treatment and Control users draw from the same ad campaign budget
- Both use the same server machines

In both cases, there is a **medium** connecting Treatment and Control — either a social graph edge or a shared resource — that allows them to interact and contaminate each other's outcomes.

---

## 22.3 Concrete Examples

### Direct Connection Examples

#### Facebook / LinkedIn (Social Networks)
- User behavior is influenced by their social neighborhood (Eckles, Karrer and Ugander 2017; Gui et al. 2015)
- Users find a social feature *more valuable* as more of their neighbors use it
- Examples:
  - I am more likely to use video chat if my friends use it
  - I am more likely to message on LinkedIn if they message me first
  - I am more likely to post if friends in my network post

**A/B Consequence:** A better "People You May Know" algorithm in Treatment causes more connection invitations. But recipients may be in **Control** — and when they visit to accept, they also send more invitations. **Result:** Both Treatment and Control metrics rise → **delta is biased downward** (underestimates true Treatment effect).

#### Skype Calls
- Every call involves at least two parties
- If Treatment improves call quality → Treatment users call more
- Those calls land on users in both Treatment **and** Control
- **Result:** Control group also increases Skype usage → **delta is underestimated**

---

### Indirect Connection Examples

#### Airbnb (Shared Inventory / Supply)
- Treatment improves conversion → more bookings by Treatment users
- **Less inventory** remains for Control users
- Control revenue falls artificially
- **Result:** Delta **overestimates** the Treatment effect (Holtz 2018)

#### Uber / Lyft (Shared Driver Supply)
- Treatment's better surge pricing causes more Treatment riders to opt in
- Fewer drivers available → Control group faces higher prices → fewer Control riders
- **Result:** Delta **overestimates** Treatment effect (Chamandy 2016)

#### eBay (Shared Auction Inventory)
- Treatment gives buyers a rebate/promotion → higher bids
- Treatment users outbid Control users for the same items
- Control users win fewer auctions
- **Result:** Delta on total transactions **overestimates** Treatment effect (Blake and Coey 2014)

#### Ad Campaigns (Shared Budget)
- Treatment users click more ads → campaign budget depletes faster
- Budget is shared between Treatment and Control
- Control ends up with a smaller effective budget
- **Result:** Delta **overestimates** Treatment effect
- **Additional wrinkle:** Budget-constrained experiments behave differently at the beginning vs. end of month/quarter

#### Relevance Model Training (Shared Training Data)
- Treatment model better predicts what users want to click
- If training data from **all users** is used for both Treatment and Control models, the "good" clicks from Treatment users improve the Control model too
- **Result:** The longer the experiment runs, the more Treatment benefits Control → **delta is underestimated**

#### CPU / Shared Server Resources
- A/B requests from both variants are processed by the **same machines**
- A bug in Treatment that holds up CPU/memory hurts **both** Treatment and Control latency
- **Result:** The negative Treatment effect on latency is **underestimated** when comparing Treatment vs. Control directly

#### Sub-User Experiment Unit (e.g., Pageview)
- When randomizing below the user level (e.g., pageview or session), the **same user** experiences both Treatment and Control
- If Treatment dramatically improves page load time → more clicks and revenue
- The same user's behavior on fast pages is influenced by their experience on slow pages, and vice versa
- **Result:** Treatment effect is **underestimated**
- The latent connection here is the **user** themselves

---

## 22.4 Bias Direction Summary

| Scenario | Interference Mechanism | Bias Direction |
|---|---|---|
| LinkedIn PYMK | Direct social network | Underestimate (↓ delta) |
| Skype calls | Direct communication | Underestimate (↓ delta) |
| Airbnb | Shared inventory | Overestimate (↑ delta) |
| Uber | Shared driver supply | Overestimate (↑ delta) |
| eBay | Shared auction items | Overestimate (↑ delta) |
| Ad campaigns | Shared budget | Overestimate (↑ delta) |
| Relevance model | Shared training data | Underestimate (↓ delta) |
| Shared CPU | Shared server resources | Underestimate (↓ delta) |
| Sub-user experiment | Same user in both variants | Underestimate (↓ delta) |

> **Key intuition:** If Treatment "uses up" a shared resource (budget, inventory, drivers, auction wins), Control gets less of it → **overestimate**. If Treatment "improves" a shared resource (model, server capacity) that benefits Control too → **underestimate**.

---

## 22.5 Practical Solutions

When we run an experiment, we want to estimate the delta between two parallel universes: one where **every unit is in Treatment**, and one where **every unit is in Control**. Leakage biases this estimate.

### Solution 1: Rule-of-Thumb — Ecosystem Value of an Action

Not all actions spill over. First, **identify which actions can potentially spill over** and only worry about interference if those actions are materially impacted.

**Approach:** Measure first-order actions **and** downstream responses together. Examples on a social network:
- Total messages sent **and** messages responded to
- Total posts created **and** likes/comments received
- Total likes/comments **and** number of creators receiving them

> An experiment with positive impact on the first-order action but **no impact on downstream metrics** is unlikely to have a measurable spillover effect.

**Establishing the rule-of-thumb:** Use historical experiments with known downstream impact and extrapolate using the **Instrumental Variable approach** (Tutterow and Saint-Jacques 2019).

**Pros:**
- One-time effort to establish ecosystem value
- Applies to any Bernoulli randomized experiment
- More sensitive than isolation approaches

**Cons:**
- Only an approximation — may not fit all scenarios
- Additional messages from a specific Treatment may have *larger-than-average* ecosystem impact

---

### Solution 2: Isolation

Identify the interference **medium** and **separate** the variants so they cannot interact. Requires departing from standard Bernoulli randomization.

#### 2a. Splitting Shared Resources
- Split the resource proportionally to traffic allocation
- Example: If Treatment gets 20% of traffic, give it 20% of the ad budget; train relevance models only on data from each variant's users

**Watch out for:**
1. **Can the resource be split exactly?** Shared machines, for example, have heterogeneity — serving Treatment and Control on different machines introduces confounds
2. **Does the split size introduce bias?** A 5% / 95% training data split unfairly disadvantages the smaller model. **This is why a 50/50 traffic split is recommended**

#### 2b. Geo-Based Randomization
- Randomize at the **region level**, assuming units from different regions do not interfere
- Works well for: competing hotels, taxis, riders in a city (Vaver and Koehler 2011, 2012)
- **Limitation:** Fewer randomization units → bigger variance → less statistical power (see Chapter 18 on variance reduction)

#### 2c. Time-Based Randomization
- At each time unit $t$, flip a coin to assign **all users** to either Treatment or Control
- Eliminates cross-variant contamination at any moment in time
- Time unit can be seconds or weeks depending on practical needs
- Strong temporal variation (day-of-week, hour-of-day) is manageable via paired t-tests or covariate adjustment
- **Limitation:** Requires that intra-user interference across time is not a concern (i.e., the sub-user unit problem doesn't apply)
- Related technique: **Interrupted Time Series (ITS)** — see Chapter 11

#### 2d. Network-Cluster Randomization
- Construct **clusters** of users close to each other based on likelihood to interfere
- Use each cluster as a "mega unit" and randomize at the cluster level (Gui et al. 2015; Backstrom and Kleinberg 2011; Cox 1958)
- **Limitations:**
  1. **Imperfect isolation in practice:** LinkedIn graph example — attempting to create 10,000 isolated clusters still left >80% of connections *between* clusters (Saint-Jacques et al. 2018)
  2. **Small effective sample size** → variance-bias tradeoff: more clusters = smaller variance but worse isolation (larger bias)

#### 2e. Network Ego-Centric Randomization
- Each cluster has one **"ego"** (focal user) surrounded by their **"alters"** (immediate connections)
- Variant assignment for egos and alters can be set independently
- Example: Give all alters Treatment, give only half the egos Treatment
- Compare egos in Treatment vs. Control to measure both first-order and downstream impact
- Achieves **better isolation and smaller variance** than cluster randomization (Saint-Jacques et al. 2018b)

**Best practice:** Combine isolation methods for larger sample size. For example, apply network-cluster randomization *and* use time as an additional sampling dimension when the interference has short time span and Treatment effect is transactional.

---

### Solution 3: Edge-Level Analysis

When leakage happens in clearly-defined pairwise interactions (e.g., messages, calls, likes):

- Use Bernoulli randomization on users (nodes)
- Label each **edge** by the assignment of both endpoints:
  - Treatment → Treatment (TT)
  - Treatment → Control (TC)
  - Control → Control (CC)
  - Control → Treatment (CT)

**What you can learn from edge labels:**
- Compare TT edges vs. CC edges to estimate an **unbiased delta** (both parties experienced the same variant)
- Check for **Treatment affinity** — do Treatment users prefer to message other Treatment users?
- Check if new actions from Treatment get **higher response rates**

See Saint-Jacques et al. (2018b) for a full treatment.

---

### Solution 4: Detecting and Monitoring Interference

Even when precise measurement isn't practical, build a **strong monitoring and alert system**:

- If all ad revenue during the experiment comes from budget-constrained advertisers, results won't generalize post-launch
- **Ramp phases** can catch severe interference early (e.g., Treatment consuming all CPU) — first ramp to employees or a small datacenter
- See Chapter 15 for details on ramp phases

---

# Chapter 23: Measuring Long-Term Treatment Effects

> *"We tend to overestimate the effect of a technology in the short run and underestimate the effect in the long run."* — Roy Amara

---

## 23.1 What Are Long-Term Effects?

**Short-term effect:** The Treatment effect measured in the typical 1–2 week experiment window.

**Long-term effect:** The **asymptotic** effect of a Treatment. In practice, this means 3+ months, or defined by number of exposures (e.g., users exposed at least 10 times).

For most experiments, short-term effects are stable and generalize to the long-term. But there are important exceptions:

| Example | Short-Term | Long-Term |
|---|---|---|
| Raising prices | ↑ revenue | ↓ revenue (users abandon) |
| Showing poor search results | ↑ query share (users re-search) | ↓ query share (users switch engines) |
| Showing more ads (including low-quality) | ↑ ad clicks and revenue | ↓ revenue (fewer clicks, fewer searches) |

**Scope note:** This chapter is concerned only with **changes where the long-term and short-term effects differ**. Short-lived changes (e.g., specific news headlines) are excluded. The focus is on durable product changes.

**OEC connection:** A key challenge in defining the Overall Evaluation Criterion (Chapter 7) is that it must be **measurable short-term** but **causally predictive of long-term goals**. Understanding long-term effects helps improve and validate short-term proxy metrics.

---

## 23.2 Reasons Short-Term and Long-Term Effects Differ

### 1. User-Learned Effects
Users learn and adapt to changes over time. Their behavior shifts toward a new equilibrium:

- **Product crashes:** First occurrence may not drive users away, but frequent crashes do
- **Ad quality:** Users adjust their click rate once they realize ad quality is poor
- **Discoverability:** A great new feature may take time for users to notice and adopt — but engagement surges once they do
- **Priming:** Users primed to the old feature may explore a new change early on but then settle into a routine

(Huang, Reiley and Raibov 2018; Hohnhold, O'Brien and Tang 2015; Chen, Liu and Xu 2019; Kohavi, Longbotham et al. 2009)

### 2. Network Effects
- User behavior is influenced by peers — when friends start using Live Video on Messenger/WhatsApp/Skype, adoption spreads
- A feature may not reach its **full impact** until it propagates through the network — this takes time
- **Two-sided marketplace complication (Airbnb, eBay, Uber):** A feature drives demand quickly, but supply takes longer to respond → revenue impact is delayed
- **Recommendation algorithm analogy:** A new algorithm may outperform early due to diversity, but hits a supply constraint (limited number of people you actually know) → lower long-term equilibrium

### 3. Delayed Experience and Measurement
- Time gap between the online experience and the actual experience the metric captures
- **Airbnb / Booking.com:** Months between the booking decision and the actual stay; retention metrics depend on the offline experience
- **Annual contracts:** Users decide to renew based on their *entire year's* cumulative experience

### 4. Ecosystem Change
The environment evolves while the experiment is running, altering how users react:

- **New features launching:** If other teams embed Live Video, it becomes more valuable over time
- **Seasonality:** Gift card experiments that work during Christmas may not generalize to non-holiday seasons
- **Competitive landscape:** If a competitor launches the same feature, the value of your feature may decline
- **Government policy:** GDPR changes what data is available for ad targeting (European Commission 2016)
- **Concept drift:** ML models trained on stale data degrade as distributions shift over time
- **Software rot:** Shipped features degrade relative to the changing environment around them unless actively maintained

---

## 23.3 Why Measure Long-Term Effects?

### Reason 1: Attribution
- Data-driven companies use experiment results to track team goals and financial forecasting
- "What would the world look like long-term with vs. without this feature?"
- Challenging because:
  - **Endogenous factors:** User-learned effects
  - **Exogenous factors:** Competitive landscape, policy
  - Future builds are layered on past launches — compounding effects are hard to attribute

### Reason 2: Institutional Learning
- Understanding **why** short-term and long-term effects differ
- Strong novelty effect → may indicate suboptimal UX (feature took too long to discover → consider in-product education)
- High initial adoption but low retention → may indicate click-bait or low quality
- Insights power better subsequent iterations

### Reason 3: Generalization
- Measure long-term effects on some experiments → extrapolate principles to others
- "How much long-term impact does this *type* of change have?"
- Example: Hohnhold et al. (2015) derived general principles for search ads
- Create **short-term proxy metrics predictive of long-term outcomes**
- When generalizing, isolate from exogenous shocks unlikely to recur

---

## 23.4 Methods for Measuring Long-Term Effects

### Method 0: Long-Running Experiments (Baseline Approach)

Keep the experiment running for a long time. Measure:
- $p\Delta_1$ = Treatment effect in week 1 → **short-term effect**
- $p\Delta_T$ = Treatment effect in the last week → **long-term effect**

> Note: This is distinct from measuring the *average* effect over the full period. You are comparing the *first week* to the *last week*.

#### Challenges and Limitations

**For attribution — why $p\Delta_T$ may not reflect the true long-term effect:**

1. **Treatment effect dilution:** Users may use multiple devices or entry points (web and app). An experiment may only capture one. Over time, a larger fraction of a user's total experience falls outside the experiment. What you measure at $p\Delta_T$ is a diluted version of what fully-treated users would show.

2. **Cookie churn:** Randomization based on cookies is unstable. Users may re-randomize to a different variant with a new cookie (Dmitriev et al. 2016). The longer the experiment, the higher this risk.

3. **Network leakage grows over time:** If network effects are present and isolation isn't perfect, the Treatment effect cascades more broadly through the network over time → larger leakage → more biased delta (see Chapter 22)

4. **Survivorship bias:** Users who dislike the Treatment may abandon over time. $p\Delta_T$ only captures users who remain — a biased, self-selected population. This should trigger an **SRM (Sample Ratio Mismatch) alert** (Chapters 3, 21)

5. **Interaction with other features:** New features launched during the long-running experiment may erode the original effect. Example: First push-notification experiment is highly effective, but loses impact as more teams add notifications.

**For time-extrapolated interpretation:**
- Without further study, the difference between $p\Delta_1$ and $p\Delta_T$ cannot be cleanly attributed to the Treatment itself
- Exogenous factors (seasonality, competitive changes) can explain the difference
- If the underlying population or external environment has changed, direct comparison is invalid

---

### Method 1: Cohort Analysis

**Idea:** Define a stable cohort of users *before* the experiment starts (e.g., based on logged-in user IDs) and analyze only this cohort's short-term and long-term effects.

**Benefit:** Addresses dilution and survivorship bias (cohort is tracked consistently throughout)

**Two key considerations:**

1. **Cohort stability is crucial:**
   - If using cookies as IDs and cookie churn is high, the method fails
   - Logged-in user IDs are more stable than cookies

2. **External validity:**
   - Logged-in users may systematically differ from non-logged-in users
   - Mitigation: **Stratification + weighting adjustment** (Park, Gelman and Bafumi 2004)
     - Stratify users into subgroups (e.g., high/medium/low pre-experiment engagement)
     - Compute Treatment effect within each subgroup
     - Weight by population distribution to estimate generalizable effect
   - This approach has the same limitations as observational studies (Chapter 11)

---

### Method 2: Post-Period Analysis

**Idea:** Run the experiment for time $T$, then **turn off the experiment** (or ramp to 100% Treatment). During the post-period $T+1$, both groups experience identical features.

The key is that the two groups differ only in their **history of exposure**, not their current experience.

**What you measure in the post-period ($p\Delta_L$):** The **"learning effect"** — the lasting impact of the Treatment after the feature is equalized (Hohnhold et al. 2015)

#### Two Types of Learned Effect:

1. **User-learned effect:** Users changed their behavior in response to the Treatment and maintained that behavior
   - Example: Users who adapted their ad-clicking behavior after being exposed to higher ad-load (Hohnhold et al. 2015)

2. **System-learned effect:** The system "remembered" information from the Treatment period
   - Examples:
     - Treatment users updated their profiles → data persists post-experiment
     - Annoyed users opted out of emails during Treatment → they receive fewer emails even post-experiment
     - ML personalization: Treatment caused users to click more ads → model learned → system shows more ads even after revert

**When system-learned effect = 0:**  
In a perfect A/A post-period, the extrapolation from short-term to long-term is most reliable. Non-zero system effects arise from: permanent user state changes, opt-outs, unsubscribes, hitting impression limits, long-horizon personalization.

**Pros:**
- Isolates learned effect separately from exogenous factors
- More interpretable: explains *why* short-term ≠ long-term
- Given enough experiments, can build a model to extrapolate long-term from short-term (Gupta et al. 2019)

**Cons:**
- Still subject to dilution and survivorship bias (Dmitriev et al. 2016)
- Mitigation: Apply adjustment to the learned effect, or combine with cohort analysis

---

### Method 3: Time-Staggered Treatments

**Problem with methods above:** How do you know when "long enough" has passed? Watching a trend line doesn't work well — day-of-week effects and other temporal noise overwhelm the long-term signal.

**Solution:** Run **two identical Treatments with staggered start times**:
- $T_0$ starts at time $t = 0$
- $T_1$ starts at time $t = 1$

At any time $t > 1$, compare $T_1(t) - T_0(t)$. This is effectively an **A/A test** — same Treatment, only difference is **duration of exposure**.

**Convergence test:** Use a two-sample t-test. If $T_1(t) - T_0(t)$ is statistically insignificant (and practically small), the two Treatments have **converged** — i.e., the Treatment effect has stabilized.

> For this convergence test, it may be appropriate to set a **lower Type II error rate than the standard 20%**, even at the cost of a higher Type I error rate (>5%). You care more about not missing a real difference than about false positives here.

Once convergence is confirmed, apply **Method 2 (post-period analysis)** from that time point onward.

**Assumption:** $T_1(t) - T_0(t)$ is a **decreasing function of** $t$ — the two staggered variants grow more similar over time.

**Limitation:** The gap between staggered start times must be large enough that a difference can initially develop. If the learned effect takes time to manifest and both variants start near-simultaneously, there may be no observable difference at the start of $T_1$.

---

### Method 4: Holdback and Reverse Experiments

**When long-running experiments aren't feasible** due to time pressure or opportunity cost of withholding Treatment from users.

#### Holdback Experiment
- Launch Treatment to 90% of users
- Keep 10% in Control for weeks or months post-launch
- Control group = **holdback**
- **Limitation:** Small Control group → less statistical power. Confirm the reduced sensitivity doesn't undermine what you want to learn (Chapter 15)

#### Reverse Experiment
- Launch Treatment to 100% of users
- After some time, **ramp 10% of users back to Control**
- **Key benefit:** The entire network/marketplace has time to reach a new equilibrium under Treatment before you measure. If network effects or supply constraints affect results, this approach captures the equilibrium state rather than a transient one.
- **Limitation:** If the Treatment introduced a visible UX change, reverting those users may be confusing or degrading their experience

---

## 23.5 Summary of Long-Term Measurement Methods

| Method | Core Idea | Addresses | Key Limitation |
|---|---|---|---|
| Long-running experiment | Keep running, measure first vs. last week | Simplest baseline | Dilution, survivorship, leakage, exogenous confounds |
| Cohort analysis | Lock in stable cohort before experiment | Dilution, survivorship | External validity concerns if cohort isn't representative |
| Post-period analysis | Turn off experiment, measure learned effect in A/A phase | Exogenous factors, feature interactions | Still has dilution/survivorship; system-learned effects may be non-zero |
| Time-staggered treatments | Run same Treatment with staggered starts; check convergence | Determines *when* long-term is reached | Gap between staggered starts must be sufficient |
| Holdback / Reverse | Keep small Control (or revert small group) after launch | Launch pressure; network equilibrium | Less power (holdback); UX confusion (reverse) |

---

## Key Takeaways Across Both Chapters

1. **SUTVA is an assumption, not a guarantee.** Always ask: *is there a medium through which Treatment and Control can interact?*

2. **Bias direction depends on mechanism:** Shared-resource depletion → overestimate. Shared-resource improvement → underestimate.

3. **The solution must match the interference mechanism.** Social network interference requires graph-aware randomization; budget interference requires budget splitting; CPU interference requires architectural fixes.

4. **Short-term effects may not generalize.** User learning, network propagation, delayed measurement, and ecosystem change all create divergence between short-term and long-term effects.

5. **Invest in ecosystem and long-term metrics.** Even if you can't run a full long-term study for every experiment, establishing general principles (e.g., via post-period analysis on representative experiments) helps calibrate short-term metrics to predict long-term impact.
