# Chapter 19 — System Design Case Studies

*(Same plain-language style as recent chapters. This chapter is three full mock interviews, written as dialogue, each with an L5-vs-L6 breakdown at the end.)*

---

## 19.1 How to use this chapter

Each mock below follows the same shape: an interviewer question, a strong candidate answer walked through step by step, and then a short breakdown of what separates a passing (L5) answer from a standout (L6) one. Read each mock once straight through, then come back and try answering the opening question yourself, out loud, before reading the model answer again.

---

## 19.2 Mock Interview 1: "Design an explore-exploit system for ad ranking"

**Interviewer**: "We show one ad per page view on our site. We have thousands of candidate ads, and new ones are added constantly. Design a system that decides which ad to show, balancing learning about new ads against showing ads we already know perform well."

**Candidate**: "Let me start by pinning down a few things about the problem. First — thousands of ads, constantly changing — that tells me a plain non-contextual bandit (treating every user the same) is going to leave a lot of value on the table, since different users almost certainly respond differently to different ads. So I'd want a **contextual bandit**, not a plain one.

Given the sheer number of ads — thousands, with new ones arriving all the time — I'd lean toward **Hybrid LinUCB or Linear Thompson Sampling**, rather than Disjoint LinUCB, specifically because hybrid-style shared weights let a brand-new ad borrow strength from patterns learned across the whole ad catalog, instead of needing to individually accumulate its own data from zero before it's any good. That directly solves the cold-start problem you'd otherwise have with thousands of individually-cold new ads.

Between LinUCB and Linear Thompson Sampling specifically, I'd lean Thompson Sampling for two reasons: first, the update step (recall it's just a simple posterior update) is cheap and parallelizes well, which matters at this scale — plausibly millions of page views a day. Second, published empirical results generally show Thompson-Sampling-style approaches performing at least as well as, often better than, UCB-style approaches in practice.

For the reward signal, I'd start simple — clicks — but flag immediately that pure click-optimization risks favoring clickbait-y ads over genuinely good ones, so I'd want to fold in a secondary signal like downstream conversion or advertiser-reported quality, either as a combined weighted reward or as a guardrail constraint, not just raw CTR.

For evaluation, before shipping any change to the ranking logic, I'd want an offline off-policy evaluation step — ideally Doubly Robust, since I likely won't have a fully uniform-random logging bucket, only whatever exploration the live bandit itself naturally produces — plus, ideally, carving out a small forced-random traffic slice specifically to support cleaner replay-method-style evaluation down the line.

Finally, two production concerns I'd flag proactively: ad performance decays over time (creative fatigue) — so I'd want a sliding-window or discounted variant, not a bandit that treats an ad's month-old data the same as this morning's. And I'd want monitoring for feedback loops — making sure a temporarily-unlucky new ad doesn't get starved of traffic so aggressively that it never gets a fair chance to recover."

**Interviewer**: "Good. How would you decide the weights in that combined clicks-and-conversion reward?"

**Candidate**: "Honestly, that's not a purely technical decision — I'd treat it as a product/business call, informed by data (e.g., what conversion rate historically follows from a given click rate for similar ads), but ultimately set in collaboration with whoever owns the business tradeoff between short-term engagement and long-term advertiser trust, not something I'd unilaterally pick as an engineering choice."

---

### Breakdown

- **L5 answer**: correctly identifies contextual bandits are needed, names LinUCB or Thompson Sampling as reasonable choices, and mentions clicks as the reward.
- **L6 answer (the one above)**: additionally (a) explicitly justifies hybrid-over-disjoint using the cold-start argument tied to the stated "thousands of ads, constantly added" detail in the prompt, (b) flags the clickbait/reward-design risk unprompted, (c) proactively brings in off-policy evaluation and the exploration-bucket pattern before being asked, (d) names both non-stationarity and feedback-loop guardrails unprompted, and (e) correctly locates the reward-weighting question as a joint technical/business decision rather than pretending it's purely an engineering call.

---

## 19.3 Mock Interview 2: "Design an explore-exploit system for a news feed's ranking"

**Interviewer**: "Users scroll a feed of news articles. We want to rank/select articles to maximize engagement, but the space of 'trending' content changes hour to hour. How would you approach this?"

**Candidate**: "The 'changes hour to hour' detail is the headline constraint here, so I want to design around non-stationarity from the start, not bolt it on as an afterthought.

Structurally, this is also a **combinatorial bandit**, not a single-arm-per-round problem — we're selecting and ordering a whole list of articles per feed load, not just one article. I'd flag the credit-assignment challenge that comes with that: if a user clicks one article out of ten shown, attributing that engagement cleanly back to 'was this specific article a good choice, independent of its position' is genuinely tricky, especially given position bias — users click on top-of-feed items more, regardless of quality. I'd want the reward model to account for position, not just raw click position.

For the core algorithm, I'd use a contextual bandit (context = user's reading history, time of day, topics currently trending) with **discounted or sliding-window** confidence/posterior updates — given how fast relevance decays here, I'd probably lean toward a fairly short window or a fairly aggressive discount factor compared to, say, the ad-ranking system in the previous example, since content freshness genuinely matters more here than in most ad-ranking contexts.

I'd also explicitly flag that this environment isn't just non-stationary because 'the world changes' — some of the drift is likely caused by the system's own behavior: heavily promoting certain articles can itself shift what's 'trending' (a feedback loop), so I'd want monitoring specifically for that self-reinforcing pattern, not just for organic topic drift.

For offline evaluation before shipping ranking changes, I'd lean on Doubly Robust again, but I'd note evaluation itself is trickier here than in the ad case, precisely because of the fast-changing non-stationary environment — logged data from even a few days ago may no longer represent current reality well, so I'd want to weight recent logged data more heavily in the evaluation itself, not just in the live serving policy."

---

### Breakdown

- **L5 answer**: identifies this as a contextual bandit problem with a non-stationarity concern, and mentions using recent data more.
- **L6 answer (the one above)**: additionally (a) explicitly names this as a *combinatorial* bandit problem and raises credit assignment and position bias unprompted, (b) makes a concrete, comparative judgment call (shorter window/more aggressive discount *than the ad-ranking case*) rather than a generic "handle non-stationarity" statement, (c) distinguishes organic drift from self-caused feedback-loop drift, and (d) extends the non-stationarity concern all the way through to the *evaluation* methodology, not just the live serving algorithm — a genuinely systems-level connection most candidates miss.

---

## 19.4 Mock Interview 3: "Design an explore-exploit system for a voice assistant's suggested-actions list"

**Interviewer**: "Our voice assistant shows a small list of 3–4 suggested quick actions when the home screen loads (e.g., 'play music', 'check weather', 'call mom'). Design the system that decides what to suggest."

**Candidate**: "Compared to the previous two examples, the scale here is very different — a small, largely fixed catalog of possible actions (probably dozens, not thousands), versus thousands of constantly-changing ads or articles. That changes my recommendation meaningfully.

With a small, fairly stable arm count, I don't think I need Hybrid LinUCB's shared-weight machinery — the cold-start problem that motivated hybrid sharing in the ad case isn't really present here, since there aren't many brand-new arms showing up regularly. I'd lean toward **Disjoint LinUCB or per-arm Linear Thompson Sampling**, one model per action, using context like time of day, day of week, and recent usage history — simpler to build, easier to debug, and I don't think we're leaving much value on the table by skipping shared weights here, unlike the ad-ranking case.

This is also a combinatorial bandit again (we're picking 3–4 actions, not 1), but at a much smaller scale than the news-feed case — with only a handful of slots and a few dozen possible actions, credit assignment is less of a headache; I'd probably start with a simpler heuristic (like treating each slot independently) rather than reaching for a fully general combinatorial-bandit formulation, and only add complexity if simpler approaches clearly underperform.

Stationarity-wise, I'd guess this environment is fairly stable — someone's daily routine doesn't shift hour to hour the way news trends do — so I wouldn't reach for aggressive sliding-window/discounting by default here; I'd start with plain (non-windowed) Thompson Sampling and only add drift-handling if we see evidence that it's actually needed, rather than assuming it up front the way I would for the news-feed case.

For reward, I'd think carefully before defaulting to 'did they tap the suggestion' — a suggestion that's *ignored* isn't necessarily bad if the user does that action anyway through a different path (e.g., they open the music app directly instead of tapping the suggestion) — so I'd want to think about whether I can get a more holistic signal than raw tap-through, though I'd probably ship a first version on tap-through and refine from there rather than blocking launch on a perfect reward signal."

---

### Breakdown

- **L5 answer**: identifies this as a smaller-scale contextual bandit problem and picks a reasonable algorithm.
- **L6 answer (the one above)**: additionally (a) explicitly contrasts this scenario against the *previous two mocks* to justify a genuinely different algorithm choice (disjoint over hybrid) grounded in the specific stated scale, (b) correctly scales down the combinatorial-bandit concern rather than over-engineering a small problem, (c) makes a reasoned, evidence-based call to *not* pre-emptively add non-stationarity handling, showing good engineering judgment about avoiding unnecessary complexity, and (d) raises a genuine, thoughtful reward-design subtlety (tap-through undercounting real value) while still giving a pragmatic "ship first version, refine later" recommendation rather than getting stuck theorizing.

---

## 19.5 The pattern across all three mocks

Notice the three-question habit from Chapter 17 (Section 17.5) — **how many arms, is the environment stationary or drifting, is there one metric or several** — doing real work in every single mock above, each time producing a genuinely *different*, scenario-specific recommendation rather than a generic "use a contextual bandit" answer repeated three times. This is the single most important meta-skill this whole syllabus has been building toward: **not memorizing which algorithm to name, but having a repeatable way to reason toward the right algorithm for whatever specific scenario shows up on interview day.**

---

## 19.6 Comprehension checks — plain words, minimal formulas

1. In the ad-ranking mock, why did the candidate recommend Hybrid LinUCB specifically, tying the recommendation to a detail stated in the prompt?
2. In the news-feed mock, what two distinct types of non-stationarity did the candidate distinguish, and why does that distinction matter?
3. In the voice-assistant mock, why did the candidate recommend a *different* algorithm than in the ad-ranking mock, despite both being contextual bandit problems?
4. What's one example, across the three mocks, of the candidate proactively raising a production concern before being asked?
5. Try applying the three-question framework (arm count, stationarity, single/multiple metrics) to a scenario of your own choosing (e.g., a food delivery app's restaurant recommendations) — what algorithm would you land on, and why?

---

*Next: Chapter 20 — Rapid-Fire Review & L5-vs-L6 Differentiators, the final chapter — consolidated comparison tables for every algorithm and estimator covered in this course, a bank of the most likely follow-up questions with model answers, and a full traps checklist pulled from all 19 prior chapters.*
