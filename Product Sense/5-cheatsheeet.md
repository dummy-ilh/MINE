# Product Data Science Interview Cheatsheet
> Read daily. All frameworks, metrics, and drill questions in one place.

---

## SECTION 1 — QUESTION TYPE SKELETONS

### Metric Design
1. **Core promise** — one sentence, what does this product promise?
2. **Specific user** — name a real person, not "users"
3. **Success feeling** — what does success feel like for them?
4. **Business goal** — one sentence + the one word underneath it
5. **Hierarchy** — North Star → Primary → Guardrail → Diagnostic
6. **Test it** — "if this metric improved 20% would a real user's life be better?"

### Diagnosis (RICE)
- **R — Reality check** — tracking change? pipeline broken? definition changed?
- **I — Isolate** — geography → platform → segment → feature → time
- **C — Cause** — Internal / External / Organic / Data. Rank by probability × testability
- **E — Effect** — severity, blast radius, reversible?, response
> Always end with the fear sentence for that specific product

### Product Improvement
1. **Clarify goal** — acquisition / retention / monetisation? Pick one.
2. **Specific user** — age, context, moment, feeling
3. **Map journey** — step by step until pain appears naturally
4. **Biggest gap** — promise vs reality for this user
5. **Three solutions** — what, why it works, what's the risk
6. **Prioritise + measure** — impact × effort, north star + guardrail

### Tradeoff (DEBT)
- **D — Define** the REAL tradeoff underneath the surface one
- **E — Evaluate costs** — concrete, specific, reversible? for each side
- **B — Business context** — stage, competitive position, trust reserve
- **T — Take a position** — "Given X I'd do Y because Z. What would change my mind: W"

### Launch Decision (SCALE)
- **S — Signal strength** — significant? effect size meaningful? consistent across segments? sample representative?
- **C — Cost of wrong** — reversible? how many users? how severe? how fast does damage accumulate?
- **A — Alternatives** — scope reduction? segment limitation? timing adjustment?
- **L — Launch scope** — dogfood → 1% → geo → segment → full
- **E — Exit criteria** — define specific numbers BEFORE launch, not after

---

## SECTION 2 — A/B TESTING MASTER FRAMEWORK

### 7 Components of a Good Experiment
1. **Hypothesis** — intervention + population + metric + magnitude (MDE)
2. **Randomisation unit** — what minimises interference? user / session / geo / cluster
3. **Sample size** — MDE × baseline × 80% power × 95% significance → minimum N
4. **Duration** — max(sample size calc, 2 weekly cycles, novelty dissipation period)
5. **Primary + guardrail metrics** — pre-specified before running. one primary.
6. **Segmentation plan** — pre-specified: new/returning, platform, market, tenure
7. **Decision criteria** — what result leads to what decision, defined in advance

### 7 Ways Experiments Break
1. **Novelty effect** — users click new things out of curiosity. Run longer.
2. **Primacy effect** — users resist change initially. Run longer for UI changes.
3. **Selection bias** — wrong population in experiment. Check who has the opportunity.
4. **Interference/spillover** — treatment affects control. Fix: change randomisation unit.
5. **Sample ratio mismatch** — check assignment ratios FIRST before any analysis.
6. **Peeking** — stops when significance hits. Pre-commit to end date.
7. **Network effects contamination** — social products contaminate groups. Use cluster randomisation.

### When NOT to A/B Test
| Situation | Alternative |
|---|---|
| Sample too small | Qualitative research, case studies |
| Metric takes too long | Validated leading indicator proxy |
| Change too fundamental to isolate | Sequential testing, usability research |
| Network effects contaminate | Geographic or cluster randomisation |
| Experiment itself causes harm | Fake door test or conjoint analysis |
| Need to know WHY not WHAT | Pair with user interviews, session recordings |

### 4 Experiment Outcomes + Responses
- **Clear positive** — primary up, guardrails stable → launch with post-launch monitoring
- **Clear negative** — primary down or guardrail breach → don't launch, run post-mortem
- **Neutral** — no movement → underpowered? genuine null? investigate segments first
- **Mixed** — primary up BUT guardrail moving → do NOT launch. Guardrail breach = stop signal always

### The Senior Question — "Significant result, should you launch?"
Before saying yes, check all six:
1. Is effect size practically meaningful — not just significant?
2. Do results hold across key segments?
3. Are all guardrail metrics stable?
4. Is there a sample ratio mismatch?
5. Did novelty effects inflate the result?
6. Does it match your pre-specified hypothesis?

### Causal Inference Methods — When You Can't Randomise
| Method | When to use |
|---|---|
| Propensity score matching | Users self-selected into a behavior — match similar users who didn't |
| Difference-in-differences | Feature launched in some markets not others — compare before/after |
| Instrumental variables | Variable that affects treatment but not outcome directly |
| Synthetic control | One treated unit (country, city) — build synthetic comparison |
| Geo randomisation | Network effects make user randomisation impossible |
| Regression discontinuity | Threshold-based treatment — compare just above and below threshold |

---

## SECTION 3 — POWER PHRASES

### Universal Openers
> "Before I dive in — let me clarify the goal, identify the user, and then structure my thinking around [question type]."

> "Can I take 30 seconds to structure my thoughts?"

> "I want to make sure I'm solving the right problem before I suggest anything."

### Metric Design
> "Before I name a metric — I want to separate [engagement] from [real goal], because those can move in opposite directions."

> "The north star I'd propose is [X] — and the reason I'd use this over [obvious alternative] is that [X] can only improve if the real goal is being achieved."

### Diagnosis
> "The first thing I'd do before anything else is confirm this drop is real and not a measurement artifact."

> "I'd segment by [geo / platform / segment / feature / time] to find the smallest boundary around this problem before investigating cause."

### Tradeoff
> "The surface tradeoff here is [X vs Y]. But the real tradeoff is [deeper tension] — and that reframe changes how I evaluate this."

> "Given [context], I would choose [option] because [reasoning]. The cost I'm accepting is [specific cost]. The thing that would change my mind is [specific condition]."

### Experiment
> "Before I design the experiment I want to define my minimum detectable effect — the smallest lift that would make this feature worth building."

> "Statistical significance is necessary but not sufficient for a launch decision. I'd also check effect size, segment consistency, guardrail stability, and sample ratio mismatch."

### Senior Signal Phrases
> "I'd pair that with a guardrail metric — because any metric can be gamed, and I want to make sure we're not improving this number in ways that damage something we care about more."

> "This decision has implications beyond the product itself — specifically [regulatory / values / ecosystem] — and I'd want [legal / policy / leadership] involved before we proceed."

> "The data shows X. My interpretation is Y. But I'd want to triangulate with qualitative research before acting, because the metric doesn't fully capture Z."

---

## SECTION 4 — METRIC HIERARCHY BY PRODUCT CATEGORY

| Category | North Star | Primary | Guardrail | Diagnostic |
|---|---|---|---|---|
| **Social/Feed** (Instagram, TikTok) | DAU completing 3+ meaningful interactions | Session length, completion rate, D7 return, posts created | Wellbeing score, uninstall rate, regret score | Scroll depth, tap-through by content type, skip rate |
| **Search** (Google, Apple) | Successful Search Rate — intent satisfied, no reformulation | Long click rate, pogo-stick rate, zero-click success, reformulation rate | Latency (non-negotiable), revenue per query, trust score | CTR by position, reformulation by query type, abandonment by device |
| **Video Streaming** (YouTube, Netflix) | Sessions with 1+ completion AND return within 48h | Completion rate, session length, D7 retention, titles/month | Satisfaction score, post-session regret, cancellation rate | Completion by length, autoplay acceptance, browse-to-play |
| **Music** (Spotify, Apple Music) | Sessions with 1+ completion AND save/follow something new | Song completion rate, skip rate, save rate, playlist creation | Skip rate on recs, session abandonment, churn rate | Skip by context, completion by session type |
| **Navigation** (Maps) | Navigation sessions completing within 10min of ETA | Completion rate, abandonment rate, reroute rate, ETA accuracy | Safety incidents, crash rate, wrong turn rate | Reroute by region, abandonment by journey length |
| **Marketplace** (Airbnb, Uber) | Transactions where both sides rate 4+ stars | Conversion rate, supply acceptance, completion rate, repeat rate | Dispute rate, safety incidents, supply churn | Search-to-click by type, cancellation by segment |
| **Subscription** (Netflix, Spotify) | Paying subscribers active 3+ times/week | Active rate among subscribers, feature breadth, renewal rate | Involuntary churn, voluntary churn, NPS among payers | Churn by cohort, feature usage before churn |
| **AI Assistant** (Siri, Gemini) | Queries accepted without reformulation or abandonment | Completion rate, reformulation rate, feature breadth per user | Trust score, error rate on factual queries | Completion by query type, abandonment by device |
| **Professional Network** (LinkedIn) | Job seekers receiving 1+ recruiter response within 30 days | Application→response rate, time to first contact, connection rate | Ghost rate (cannot increase), spam rate, recruiter churn | Response rate by industry/level, application completion |
| **E-commerce** (Amazon) | Visitors who purchase AND repurchase within 90 days | Search→purchase conversion, cart abandonment, return rate, AOV | Return rate, customer service contact rate, review score | Conversion by source, abandonment by checkout step |

---

## SECTION 5 — METRIC DEEP DIVES (4 LEVELS)

### DAU — Daily Active Users
- **Definition** — Unique users performing ≥1 qualifying action in 24 hours
- **Hidden construction decisions** — What counts as qualifying? Session timeout threshold? Multiple devices = 1 or 2 DAU? Which timezone defines the 24h window?
- **When it lies** — Treats 2-second open = 45-minute session. Inflated by push notifications. Hides compositional shift — DAU flat while entire user base turns over. Day-of-week sensitive.
- **What optimising it sacrifices** — User wellbeing (dark patterns inflate DAU). Revenue per user (high DAU, low monetisation = vanity metric). Long-term retention (spike tactics suppress 90-day retention).

### Retention Rate — D1/D7/D30
- **Definition** — % of users who return on day 1, 7, or 30 after first use. Each cohort defined by first use date.
- **Hidden construction decisions** — When does day zero start — download, login, or first meaningful action? Exactly day 7 or within a window? How to handle timezone? Include resurrected users?
- **When it lies** — Lagging indicator — D30 tells you about users from a month ago. Doesn't capture frequency within the window. Sensitive to cohort quality — bad acquisition tanks retention without product changing.
- **What optimising it sacrifices** — Growth. Acquisition campaigns that prioritise volume over quality tank retention. The most common organisational conflict in tech.

### Session Length
- **Definition** — Total time from session start to session end or timeout
- **Hidden construction decisions** — What defines session start? What defines end — close, background, inactivity? What is the timeout threshold (30min? 60min?)? Does autoplay count? Does background playing count?
- **When it lies** — Doesn't distinguish active engagement from passive background. Can be inflated by autoplay. Long sessions can mean users can't find what they want. Hides settling behavior — watching something mediocre because nothing better appeared.
- **What optimising it sacrifices** — Satisfaction (features that increase length make users feel worst afterwards). Content diversity (creates content bubbles). User wellbeing (very long sessions may indicate loneliness or avoidance).

### Conversion Rate
- **Definition** — % of users completing a desired action — free to paid, visitor to signup, browse to purchase
- **Hidden construction decisions** — The denominator is the critical hidden decision. All visitors? All who saw paywall? All who started checkout? Time window — 24h vs 30 days vs ever? How to handle assisted conversions?
- **When it lies** — Point-in-time — doesn't capture users who convert months later. Aggregates very different journeys. Can be improved by reducing friction in ways that attract low-quality converters who then churn.
- **What optimising it sacrifices** — LTV (more conversions doesn't mean better users). User experience (aggressive conversion tactics damage trust long-term).

### Churn Rate
- **Definition** — % of users or subscribers who stop using the product in a given period
- **Hidden construction decisions** — How do you define churned for engagement products — 30 days? 60? For subscriptions: voluntary vs involuntary churn must be separated. Net churn vs gross churn?
- **When it lies** — Backward looking — by the time you see it, it's already happened. Averages across very different segments — power user churn is catastrophic, casual user churn is expected. Hides resurrection.
- **What optimising it sacrifices** — Voluntary vs involuntary churn have completely different causes and fixes. Conflating them leads to wrong interventions.

### NPS — Net Promoter Score
- **Definition** — % Promoters (9-10) minus % Detractors (0-6) on likelihood to recommend question
- **Hidden construction decisions** — Who receives the survey — all users or active users? When — after positive interaction inflates NPS. How often — fatigue affects response rates.
- **When it lies** — Stated preference not observed behavior. Single number hides distribution. Not comparable across industries or cultures. Doesn't tell you WHY it moved.
- **What optimising it sacrifices** — Engagement metrics. High NPS with declining engagement = users love but use less. Low NPS with high engagement = addicted but resentful.

### LTV — Lifetime Value
- **Definition** — Total revenue expected from a single customer over their entire relationship
- **Hidden construction decisions** — Simple: avg revenue/month × avg months retained. Complex: time value of money, variable revenue over time, reactivation, referral value. Prediction horizon matters enormously.
- **When it lies** — Prediction not fact. Averages across very different segments — top 10% may have 10x the LTV of median user. Can justify bad decisions with optimistic future assumptions.
- **What optimising it sacrifices** — CAC discipline. High predicted LTV makes expensive acquisition look justified until predictions prove wrong.

### Completion Rate
- **Definition** — % of users who finish a piece of content, flow, or task from start to finish
- **Hidden construction decisions** — What counts as completion — 90% watched? 95%? 100%? Does autoplay completion count? How to handle pause-and-return days later?
- **When it lies** — Doesn't distinguish engaged completion from passive. Short content completes more easily than long — always segment by length. Can be inflated by autoplay.
- **What optimising it sacrifices** — Discovery. Optimising completion pushes algorithm toward safe familiar content. Discovery requires recommending unfamiliar content users might abandon.

---

## SECTION 6 — WHAT COMPANIES ACTUALLY WANT

### The One Word Underneath Every Product
| Product | One Word | Real Want | Fear |
|---|---|---|---|
| Google Search | Default | Trust as the first place people go for every question | AI assistants and TikTok search replacing Google as the starting point |
| YouTube | Satisfaction | Daily habit competing with Netflix and TV simultaneously | Passive resentment — watching without enjoying, guilt building |
| Google Maps | Confidence | Trusted for every location decision — not just navigation | Reduced to navigation utility while discovery goes to Instagram |
| Gmail | Indispensability | Centre of digital identity — switching becomes unthinkable | Users treating Gmail as backup while managing important comms elsewhere |
| Gemini | Relevance | Default AI assistant replacing standalone ChatGPT | OpenAI habits so strong Google's AI products feel secondary |
| iPhone | Lock-in | Ecosystem so embedded switching feels impossible | Android closing the gap until switching cost no longer justifies premium |
| Siri | Trust | Intelligent layer across all Apple devices, anticipating needs | Users habituating to ChatGPT making Siri feel like a backup assistant |
| Apple Music | Integration | Chosen over Spotify because of perfect device integration | Spotify's discovery so superior loyal Apple users keep Spotify as primary |
| iCloud | Dependency | So much important data stored leaving Apple becomes unthinkable | Users choosing Google Photos for their most important data |
| Instagram | Habit | Daily emotional habit — first app opened when bored or lonely | Generational abandonment — young users quietly drifting to TikTok |
| LinkedIn | Activation | Users open daily — not just when job hunting | Being treated as a passive resume holder updated every few years |
| Spotify | Identity | Playlists, Wrapped, Discover Weekly feel like a piece of themselves | Apple Music integration making Spotify feel like it needs justification |
| Airbnb | Trust | Booking feels like the start of a good trip — not a transaction | Trust collapse from horror stories — the thing that ends the marketplace |
| Uber | Reliability | Unthinking default — opening a competitor never occurs | Driver supply thinning — wait times rising until the promise breaks |
| Netflix | Prestige | Must-watch content users talk about with friends | Perceived as most content but least must-watch — breadth without prestige |




| Category | North Star | Primary | Guardrail | Diagnostic |
|---|---|---|---|---|
| **Social/Feed** (Instagram, TikTok) | DAU completing 3+ meaningful interactions | Session length, completion rate, D7 return, posts created | Wellbeing score, uninstall rate, regret score | Scroll depth, tap-through by content type, skip rate |
| **Search** (Google Search, Apple Spotlight) | Successful Search Rate — intent satisfied, no reformulation | Long click rate, pogo-stick rate, zero-click success, reformulation rate | Latency (non-negotiable), revenue per query, trust score | CTR by position, reformulation by query type, abandonment by device |
| **Video Streaming** (YouTube, Apple TV+) | Sessions with 1+ completion AND return within 48h | Completion rate, session length, D7 retention, titles/month | Satisfaction score, post-session regret, cancellation rate | Completion by length, autoplay acceptance, browse-to-play |
| **Music** (Spotify, Apple Music, YouTube Music) | Sessions with 1+ completion AND save/follow something new | Song completion rate, skip rate, save rate, playlist creation | Skip rate on recs, session abandonment, churn rate | Skip by context, completion by session type |
| **Navigation/Maps** (Google Maps, Apple Maps) | Navigation sessions completing within 10min of ETA | Completion rate, abandonment rate, reroute rate, ETA accuracy | Safety incidents, crash rate, wrong turn rate | Reroute by region, abandonment by journey length |
| **Marketplace** (Airbnb, Uber) | Transactions where both sides rate 4+ stars | Conversion rate, supply acceptance, completion rate, repeat rate | Dispute rate, safety incidents, supply churn | Search-to-click by type, cancellation by segment |
| **Subscription** (YouTube Premium, Apple One) | Paying subscribers active 3+ times/week | Active rate among subscribers, feature breadth, renewal rate | Involuntary churn, voluntary churn, NPS among payers | Churn by cohort, feature usage before churn |
| **AI Assistant** (Google Gemini, Apple Siri) | Queries accepted without reformulation or abandonment | Completion rate, reformulation rate, feature breadth per user | Trust score, error rate on factual queries | Completion by query type, abandonment by device |
| **Professional Network** (LinkedIn) | Job seekers receiving 1+ recruiter response within 30 days | Application→response rate, time to first contact, connection rate | Ghost rate (cannot increase), spam rate, recruiter churn | Response rate by industry/level, application completion |
| **E-commerce** (Google Shopping, Apple App Store) | Visitors who purchase AND repurchase within 90 days | Search→purchase conversion, cart abandonment, return rate, AOV | Return rate, customer service contact rate, review score | Conversion by source, abandonment by checkout step |
| **Mobile OS / Platform** (Android, iOS) | Devices active daily with 0 critical errors in 30 days | Crash-free session rate, update adoption rate, app ecosystem health | Crash rate, security incident rate, battery drain complaints | Crash by OEM/device, update lag by region |
| **Browser** (Google Chrome, Apple Safari) | Sessions where user completes intended task without switching browser | Page load success rate, tab session depth, D30 retention | Crash rate, security vulnerability response time, privacy complaint rate | Load time by site category, abandonment by connection type |
| **Email** (Gmail, Apple Mail) | Threads where user reaches inbox zero within 24h | Open rate, reply rate, spam false-positive rate, D7 active rate | Unsubscribe rate, phishing click rate, data breach incidents | Open by category, reply latency by thread type |
| **Cloud Storage** (Google Drive, iCloud) | Users who store AND retrieve content within 30 days | Upload success rate, sync reliability, storage utilization, sharing rate | Data loss incidents (zero tolerance), sync error rate, overage complaint rate | Upload by file type, sync failure by device/OS |
| **Productivity Suite** (Google Workspace, Apple iWork) | Documents collaborated on by 2+ users within 7 days of creation | Co-edit session rate, sharing rate, cross-app usage, D30 retention | Data loss rate, export failure rate, compatibility complaint rate | Feature usage by doc type, collaboration by org size |
| **Smart Home / IoT** (Google Home, Apple HomeKit) | Devices with 1+ automation triggered daily | Device uptime, automation success rate, daily active device rate | Safety incident rate, false trigger rate, offline duration | Trigger by device type, failure by home network type |
| **Wearables / Health** (Google Pixel Watch / Fitbit, Apple Watch / Health) | Users logging health data 5+ days/week | Daily active wearable rate, goal completion rate, health metric trend accuracy | Medical misguidance incidents, battery life complaint rate, data accuracy complaints | Feature use by health goal, retention by fitness level |
| **Payments** (Google Pay, Apple Pay) | Transactions completed on first attempt | Transaction success rate, checkout time, repeat usage rate, merchant acceptance rate | Fraud rate (zero tolerance), declined transaction rate, dispute rate | Success rate by merchant type, abandonment by auth method |
| **App Store / Distribution** (Google Play, Apple App Store) | Developers with 1+ app reaching 1,000 DAU within 90 days of launch | App approval time, discovery-to-install rate, developer retention, review fairness score | Policy violation rate, malware detection miss rate, developer churn | Install by category, rating by app size |
| **Video Calling** (Google Meet, FaceTime) | Calls completed without quality drop or early termination | Call completion rate, audio/video quality score, join latency, D7 return rate | Call drop rate, security incident rate, CSAM detection miss rate | Quality by network type, drop rate by device |
| **Messaging** (Google Messages / RCS, iMessage) | Threads with 2+ exchanges within 24h of initiation | Message delivery rate, read rate, reply rate, D30 retention | Message failure rate, spam/scam report rate, privacy incident rate | Delivery by network, reply latency by thread type |
| **Podcasts** (Google Podcasts*, Apple Podcasts) | Sessions with 1+ episode completed AND subscription to a new show | Completion rate, subscription rate, D7 return, episode starts per session | Skip rate on recommendations, session abandonment, churn rate | Completion by episode length, skip by genre |
| **News** (Google News, Apple News+) | Sessions where user reads 2+ articles AND returns within 48h | Article completion rate, session depth, save rate, D7 retention | Publisher complaint rate, misinformation flag rate, subscription churn | Completion by topic, bounce rate by source |
| **Translation** (Google Translate, Apple Translate) | Queries where user accepts output without editing or re-querying | Query success rate, copy/use rate, D30 return | Mistranslation complaint rate, sensitive content error rate | Accuracy by language pair, re-query rate by domain |
| **Voice / Smart Speaker** (Google Assistant / Nest, Siri / HomePod) | Voice commands fulfilled without fallback to screen | Intent recognition rate, task completion rate, D7 engagement | Misfire rate, privacy incident rate (accidental activation), trust score | Completion by command type, failure by ambient noise level |
| **Gaming** (Google Play Games, Apple Arcade) | Sessions with 1+ level/match completed AND return within 72h | Session length, level completion rate, D7 retention, IAP conversion | Toxicity report rate, spending complaint rate, session abandonment | Completion by genre, churn by session frequency |
| **Education** (Google Classroom, Apple Education) | Students completing 1+ assigned task per week | Assignment completion rate, teacher active rate, D30 student retention | Academic integrity flag rate, parental complaint rate, data privacy incident rate | Completion by grade level, drop-off by subject |
| **Developer Tools** (Google Cloud / Firebase, Xcode / Apple Developer) | Developers deploying to production within 30 days of signup | Deployment success rate, SDK adoption rate, D30 retention, support ticket rate | Outage duration, API error rate, billing surprise rate | Deploy success by stack, ticket rate by SDK version |
| **Advertising Platform** (Google Ads, Apple Search Ads) | Campaigns where advertiser renews within 90 days | ROAS, CTR, conversion rate, cost-per-acquisition | Ad fraud rate, brand safety incident rate, advertiser churn | CTR by ad format, conversion by audience segment |
| **Photo Storage & Memories** (Google Photos, Apple Photos) | Users who upload AND view/share a memory within 30 days | Upload success rate, auto-backup reliability, memory engagement rate, sharing rate | Data loss incidents (zero tolerance), privacy misidentification rate (wrong person tagged), storage overage complaint rate | Engagement by memory type (People/Places/Dates), share rate by surface (link vs. social), re-view rate of auto-created albums |


---

## SECTION 7 — DIAGNOSIS SUB-QUESTION BANK

### R — Reality Check (Always First)
- Did our tracking or logging change recently?
- Did the definition of this metric change?
- Is the data pipeline or dashboard broken?
- Is this consistent across all measurement systems or just one?
- Are we comparing like-for-like time periods?
- Is this statistically significant or just noise at this sample size?

### I — Isolate by Geography
- Is this global or region specific?
- Which countries are affected and which are not?
- Is it concentrated in one large market that moves the global number?
- Is there a timezone pattern — did it start in one region and spread?

### I — Isolate by Platform
- Is this iOS, Android, desktop, or all platforms?
- Did an OS update ship recently that could have broken something?
- Is this app, mobile web, or desktop web?
- Is this a specific device type — phone, tablet, watch, TV?

### I — Isolate by User Segment
- Is this new users, returning users, or both?
- Is this power users or casual users?
- Is this paid users or free users?
- Is this a specific acquisition channel?

### I — Isolate by Time
- Is this a sudden cliff or a gradual slope? (cliff = internal change, slope = behavioral shift)
- What hour did it start? (specific hour suggests a deployment)
- Is there a day-of-week pattern?
- Is this one day or a multi-day trend?

### C — Cause Categories
- **Internal** — did we ship anything in the last 48-72 hours?
- **Internal** — did an A/B test get exposed to too large a segment?
- **Internal** — did we change notifications, pricing, or a default setting?
- **External** — did a competitor launch something?
- **External** — holiday, news event, or seasonal pattern?
- **External** — regulatory change or government action in a key market?
- **Organic** — is this a slow generational shift that crossed a threshold today?

### E — Effect
- Is this a temporary blip or structural shift?
- What else is this affecting or about to affect?
- Is the damage reversible or permanent?
- What's the immediate action vs long term fix?

---

## SECTION 8 — CAUSATION WARNING CARD

### Three Confounds That Appear Most in Interviews
1. **Selection bias** — users who self-select into a behavior are fundamentally different from those who don't. Never compare them directly.
2. **Survivorship bias** — you only see data from users who stayed. Churned users are invisible and they're the ones who would tell you what's wrong.
3. **Reverse causality** — LinkedIn Premium users post more — but they post more BECAUSE they're already invested, not because Premium caused posting.

### The Three Questions Before Any Causal Claim
1. Does A cause B?
2. Does B cause A? (reverse causality)
3. Does something else cause both? (confounding variable)

### The Answer When Asked About a Correlation
> "Before I act on that correlation I'd want to understand whether we're seeing causation or selection bias. The users who do X may simply be different from users who don't — and if we intervene to make more users do X, we might see no effect because it was the underlying characteristic driving both behaviors, not the behavior itself. I'd run a randomised experiment where I assign users to treatment and control, then measure the outcome causally."

---

## SECTION 9 — GOOGLE VS APPLE FILTER

| Dimension | Google | Apple |
|---|---|---|
| Starting point | Information and data | Human experience and emotion |
| Scale philosophy | Reach everyone adequately — billions | Reach fewer people brilliantly |
| Privacy stance | Important compliance requirement | Core product value — non-negotiable |
| Technology role | The story — PageRank, BERT, TPU | Invisible enabler — human is the hero |
| Decision sequence | Data first, intuition to confirm | Intuition first, data to confirm |
| Primary metrics | DAU, engagement, scale, SSR | Satisfaction, loyalty, ecosystem depth |
| Success definition | Most useful to most people | Most meaningful to the right people |
| Competitive advantage | Data and distribution | Taste and ecosystem lock-in |
| Time horizon | This quarter and next year | This decade |
| Anti-pattern | Ignoring global users — solutions that only work in English, US, wifi | Optimising for engagement — more time spent is not better at Apple |

### Signal Sentences

**Google signal:**
> "This needs to work as well on a 2G connection in rural Indonesia as on fiber in San Francisco."

**Apple signal:**
> "The technology should be invisible. What matters is whether this moment in someone's life is better."

> "I'd want to make sure this works within our privacy framework — ideally processing on device so user data never leaves their phone."

---

## SECTION 10 — SENIOR SIGNAL CHECKLIST

Run before finishing any answer. 7+ checks = senior-level answer.

- [ ] Did I name a specific human being — not just "users"?
- [ ] Did I name a guardrail metric AND explain why specifically that one?
- [ ] Did I connect my metric to actual human behavior — not just a number?
- [ ] Did I question the data before acting on it?
- [ ] Did I name the real tradeoff — not just the surface one?
- [ ] Did I take a position with a reason — not hedge endlessly?
- [ ] Did I say what would change my mind?
- [ ] Did I think about scale, global users, or ecosystem implications?
- [ ] Did I complete the loop from data finding to product decision?
- [ ] Did I name any values, regulatory, or trust implications if relevant?

### One Sentence Tests

| For | Ask yourself |
|---|---|
| Metrics | "If this metric improved 20% would a real user's life be meaningfully better?" |
| Features | "Am I adding a feature or closing a gap between what this product promises and what this user experiences?" |
| Experiments | "If I'm wrong about this — how bad is it and can I reverse it?" |
| Tradeoffs | "Am I optimising short term metrics at the cost of long term trust?" |
| Launch | "Do I understand this result well enough to stake my recommendation on it?" |
| Diagnosis | "Have I confirmed the data is real before theorising about why?" |

---

## SECTION 11 — FLYWHEELS, TWO-SIDED PLATFORMS, AND AHA MOMENTS

### The Flywheel Pattern
> More supply → More value → More demand → More supply

**When a flywheel metric drops:** "Where in the flywheel did it break? A break at one point cascades everywhere eventually."

**Common flywheels:**
- **YouTube** — more creators → more content → more viewers → more ad revenue → more creators
- **Airbnb** — more hosts → more listings → more guests → more bookings → more hosts
- **Google Search** — more queries → better data → better results → more trust → more queries
- **Uber** — more drivers → lower wait times → more riders → more rides → more drivers
- **Spotify** — more listeners → more data → better recs → more satisfaction → more listeners

### Two-Sided Platform Rule
> "Always measure both sides. A marketplace fails if either side is underserved. Measuring only riders and ignoring drivers is incomplete."

### Cold Start Problem
New marketplace: thin data → thin product → few users → no new data → flywheel won't start.
**Solution:** Seed it. Street View cars for Maps. Curated content before algorithmic content. Pay early suppliers.

### Aha Moments — Know These Cold
| Product | Aha Moment |
|---|---|
| Uber | Watching the car move toward you on the map in real time |
| Spotify | Discover Weekly gives you a song you love that you never heard |
| Instagram | Photo gets likes from people you actually care about |
| LinkedIn | A stranger views your profile and reaches out with an opportunity |
| Airbnb | Arriving and it looks exactly like the photos |
| YouTube | Algorithm recommends something perfect you didn't search for |
| Duolingo | Getting a word right that you got wrong last week |
| Gmail | Searching and finding an email from 5 years ago instantly |

---

## SECTION 12 — RICE DRILL QUESTIONS (Daily Practice)

Read the question, answer using RICE, then check the key insight.

### Google Search
**Q: Google Search DAU dropped 15% globally week over week.**
Key insight: The scariest hypothesis is behavioral substitution — ChatGPT, TikTok Search, Reddit. A bug fixes itself. Substitution doesn't.

**Q: Average searches per user per day declining but average searches per country increasing. Explain.**
Key insight: Simpson's Paradox. More countries joining (growing denominator) but new country users search less. Per-user average falls even as total volume grows.

**Q: CTR dropped 20% after new SERP layout launched.**
Key insight: First ask if the drop is bad. Zero-click success — users getting answers on SERP — is a win. CTR dropping because featured snippets answer queries is success not failure.

**Q: Search revenue up 10% but satisfaction scores down 8% simultaneously.**
Key insight: Leading vs lagging indicator trap. Revenue is lagging. Satisfaction is leading. You're consuming future health to show present performance. Act on satisfaction now.

### YouTube
**Q: YouTube watch time up 18% but creator channel subscriptions down 12%.**
Key insight: Quantity of consumption up, quality of connection down. Algorithm surfacing content users watch passively but don't connect with. Long-term creator ecosystem at risk.

**Q: YouTube mobile DAU flat but desktop DAU dropped 25%.**
Key insight: Platform-specific drops have platform-specific causes. Check Chrome/browser update. Mobile flat rules out a general product problem.

**Q: YouTube Premium conversion rate dropped 30% in one week.**
Key insight: Before external causes — check if anything changed in the Premium offer itself. Free trial length, pricing, paywall placement. A 30% drop in one week is almost always an internal change.

### Google Maps
**Q: Google Maps navigation sessions shorter but destination completion rate flat.**
Key insight: This might be GOOD — Maps getting users there faster. Shorter sessions with flat completion = efficiency not failure.

**Q: Google Maps place searches up 30% but conversion to navigation down 20%.**
Key insight: Users searching for places but not navigating. Three hypotheses: (1) place info answers their question without navigation, (2) irrelevant results require multiple searches, (3) something broke in the handoff flow.

### Apple
**Q: App Store conversion rate (browse to install) dropped 25% after redesign.**
Key insight: Primacy effect. Check if drop is concentrated in first week — if it recovers in weeks 2-3, users are adapting not rejecting.

**Q: In-app purchase revenue dropped 20% across ALL App Store categories simultaneously.**
Key insight: Cross-category simultaneous = payment infrastructure problem. Category-specific = content problem. Check Apple Pay and purchase confirmation flow first.

**Q: Apple Music daily listening hours up 15% but subscription renewal rate down 10%.**
Key insight: Passive habit ≠ valued habit. Users listening as background noise. Renewal reveals the truth that listening hours hide.

**Q: Siri invocations dropped 30% on HomePod but iPhone Siri flat.**
Key insight: Device-specific drops have device-specific causes. HomePod Siri dominated by a small number of use cases — breaking one explains a large drop.

### Instagram / Meta
**Q: Instagram Stories views dropped 20% but Feed engagement flat.**
Key insight: Surface-specific drop. Check if Stories bar placement changed. Check if major creators shifted content to Reels. Feed being flat rules out a general engagement problem.

**Q: Instagram new user D7 retention dropped 25% after new onboarding flow launched.**
Key insight: Direct causal signal — the new onboarding flow is almost certainly the cause. Find the step with highest abandonment. Revert while fixing.

**Q: WhatsApp message send rate dropped 15% in Brazil specifically.**
Key insight: Single market drops need market-specific investigation. Check local competitor, regulatory action, or infrastructure issue affecting delivery reliability.

### LinkedIn
**Q: LinkedIn Premium conversion up 25% but Premium churn up 20% three months later.**
Key insight: Conversion-churn mismatch. Aggressive conversion brought in wrong users. High conversion + high churn = acquired wrong users, not retained poorly.

**Q: LinkedIn job application rate dropped 20% week over week.**
Key insight: Sudden cliff = platform change. Gradual slope = macroeconomic. Distinguish the pattern before diagnosing cause.

### Spotify
**Q: Spotify podcast hours up 40% but music hours down 15% simultaneously.**
Key insight: Cannibalisation — not necessarily failure. Podcast content is cheaper than music licensing. Evaluate at business model level — gross margin may actually be improving.

**Q: Spotify free-to-Premium conversion dropped 20% after price increase.**
Key insight: Expected — but not necessarily a problem. Calculate if revenue increase from higher price offsets volume decrease. Segment by market — emerging markets more price-sensitive.

### Uber / Airbnb
**Q: Uber ride completion rate dropped 15% on Friday and Saturday nights specifically.**
Key insight: Time-specific drops reveal demand-supply imbalances at peak moments. Check driver-online rate on those nights. If driver supply is the issue — it's a supply problem not demand.

**Q: Airbnb host acceptance rate dropped 20% in summer months.**
Key insight: Check seasonal patterns first. Summer = peak travel = hosts more selective. Compare to same period last year before treating as a problem.

**Q: Airbnb search to booking conversion dropped 30% after new pricing display feature.**
Key insight: New total price display earlier causes self-selection. This might be GOOD for trust even though conversion drops. Check if booking quality improves among users who do convert.

---

## SECTION 13 — THE REAL TRADEOFFS UNDERNEATH SURFACE ONES

| Surface Tradeoff | Real Tradeoff |
|---|---|
| Revenue vs user experience | Short term extraction vs long term trust |
| Launch now vs experiment longer | Speed to market vs confidence in outcome |
| New users vs existing users | Growth vs depth |
| More ads vs user experience | Quarterly revenue vs product health decade |
| Engagement up, satisfaction down | Current metrics vs future retention |
| Conversion rate up, retention down | Volume of users vs quality of users |
| Feature completeness vs launch timing | Confidence vs momentum |

---

## SECTION 14 — THE PRODUCT FEAR SENTENCES

Use these to close your diagnosis with business judgment:

- **Google Search** — "The scariest scenario is behavioral substitution — users going to ChatGPT or TikTok search instead. A bug fixes itself. A behavior shift doesn't."
- **Instagram** — "The deeper fear is generational abandonment — young users quietly drifting to TikTok without ever formally leaving."
- **Siri** — "The specific fear is trust erosion — users stopped believing it works, and every failed response reduces the probability they try again tomorrow."
- **YouTube** — "The long-term fear is passive resentment — users watching without enjoying, accumulating guilt that eventually drives them away."
- **LinkedIn** — "The existential risk is passive drift — users treating LinkedIn as a resume holder they update every few years rather than a daily platform."
- **Airbnb** — "The thing that ends the marketplace is trust collapse — strangers stop being willing to sleep in each other's homes."
- **Spotify** — "The specific fear is that Apple Music's ecosystem integration makes Spotify's value proposition feel like it requires active monthly justification."
- **Uber** — "The reliability promise breaks when wait times spike — and a broken reliability promise is the fastest way to lose the default behavior."

---

## SECTION 15 — LAUNCH DECISION FILTER

### The Four Launch States
| Signal Strength | Cost of Being Wrong | Recommendation |
|---|---|---|
| Strong | Low | Full launch with monitoring |
| Strong | High | Staged rollout with tight exit criteria |
| Weak | Low | Launch to small segment to build signal |
| Weak | High | Do not launch — iterate and retest |
| Mixed | Any | Diagnose the inconsistency before deciding |

### Exit Criteria Template
Before any launch, define:
- If [primary metric] drops more than [X%] within [Y hours] → rollback immediately
- If [guardrail metric] increases more than [X%] → pause and investigate
- If [trust/safety signal] triggers at any level → full pause regardless of other metrics

### The Leadership Pressure Answer
> "I'd give leadership three things. First — what we know and don't know, and what risk we'd be taking by launching now. Second — a way to launch that limits that risk — a staged rollout with tight exit criteria. Third — a specific ask for what I need to feel confident. My job is not to make the decision for them. It's to make sure they're making it with full information."

---

## SECTION 16 — YOUR PERSONAL PATTERNS

### Natural Strengths (From 16 Lessons of Observation)
- **User empathy** — your multicultural user, the grandmother with the camera — instinctively specific
- **Causal reasoning** — spotted selection bias and reverse causality without being taught the terms
- **Decisive positioning** — said no to the Answer Engine clearly and with reasoning
- **Instinct before vocabulary** — your DAU investigation instincts were structurally correct before you knew the framework

### Consistent Gaps to Watch
- **Make costs concrete** — don't just name them. Specify the mechanism, the user, the magnitude.
- **Name the values/regulatory layer** — on big product decisions, always ask if there are implications beyond the product
- **Minimum detectable effect** — every experiment answer needs a specific number threshold
- **Product-specific metrics** — guardrails and diagnostics should fit the specific product, not be generic

### The Reframe That Changes Everything
> "I'm not adding features. I'm closing the gap between what this product promises and what this user actually experiences."

> "I'm not just diagnosing a metric. I'm finding the human behavior that changed — and understanding why."

---

*End of cheatsheet. Read daily. Answer 3-5 RICE questions daily. You're ready.*
