Love this. 50 days of daily interviews is exactly what closes the gap from L4 to L5.

Here's how I want to structure it before we start.

---

## The 50 Day Plan

### Days 1-10 — Single Question Type Drills
One question type per session. Pure repetition until the structure is automatic.

- Days 1-2 — Metric design only
- Days 3-4 — Diagnosis only
- Days 5-6 — Product improvement only
- Days 7-8 — Tradeoff only
- Days 9-10 — Experiment design only

### Days 11-20 — Product Specific Drilling
Same question types but now anchored to specific companies.

- Days 11-13 — Google products only
- Days 14-16 — Apple products only
- Days 17-18 — Meta products only
- Days 19-20 — Marketplace products (Uber, Airbnb, LinkedIn)

### Days 21-35 — Mixed Question Loops
Two to three questions per session. Different types back to back. Simulates real interview pressure.

### Days 36-45 — Full Mock Interview Loops
Four questions per session. Full debrief after each. Hire / no hire decision every time.

### Days 46-50 — Company Specific Finals
- Day 46 — Full Google L4 loop
- Day 47 — Full Google L5 loop
- Day 48 — Full Apple loop
- Day 49 — Hardest questions from all 50 days
- Day 50 — Final assessment. Are you ready.

---

## The Rules

**You answer every question.** No flipping. No asking me to go first.

**I give a hire / no hire signal after every answer.** Not just feedback. A real decision.

**You can ask for feedback after your answer.** But you answer first. Always.

**We track your progression.** After every 10 days I give you a level assessment — where you are relative to L4 and L5 bar.

---

Perfect. L5 bar. Option A for days 1-20.

---

## Day 1 — Session 1

**Question Type:** Metric Design
**Company:** Google
**Level:** L5

---

> **You are the product data scientist on the Google Search team. Google is considering a new feature called "Search Memory" — it remembers your past searches and personalises future results based on your search history across sessions. How would you define and measure success for this feature?**

**Constraints that make this L5:**
- This feature has a privacy dimension
- It affects both new and returning users differently
- Success means different things for Google the business vs the user
- There is a trust element that standard engagement metrics won't capture

---

Good. You have the right instincts on all three. Now let me show you what the complete L5 answer looks like — built from your instincts, expressed with precision.

---

## The Complete Answer

**User:**
> "The user I'm thinking about is someone with recurring search needs — a researcher, a student, a professional who searches similar topic clusters repeatedly. Today they retype context every session. Search Memory's promise is that Google already knows their context so they can search with less words and get more relevant results immediately."

**North Star — Search Memory specific:**
> "SSR already measures overall search quality. To isolate Search Memory's contribution I'd measure SSR delta between users with Memory on vs off — specifically on returning query sessions. If Memory is working — users searching in topic areas they've searched before should show higher SSR than equivalent users without Memory. The feature's value is specifically in returning contexts, not first-time queries."

**Your instinct was right — generalisation is the key signal:**
> "If a user with Memory on is still reformulating in topic areas they've searched before — the personalisation isn't working. That's the specific failure signal. Reformulation rate on returning query types — not overall reformulation rate."

**Guardrail 1 — Trust score:**
> "User reported trust score specifically around search personalisation. Measured quarterly. Cannot decline."

**Guardrail 2 — Your instinct was exactly right:**
> "Feature disable rate — % of users who turn Search Memory off within 7 days of encountering it. If this exceeds a threshold it means the personalisation crossed from helpful to creepy. This is the behavioral trust signal that doesn't require a survey."

**Guardrail 3 — The one you missed:**
> "Sensitive query personalisation incidents — cases where Memory surfaces results that reveal a past sensitive search in a context the user didn't intend. Medical, legal, personal. This cannot increase under any circumstance. One incident that goes viral destroys the feature permanently."

**The tension — your "too specific" instinct:**
> "The feature has a goldilocks problem. Too little personalisation — SSR doesn't improve, feature has no value. Too much personalisation — results feel surveillance-like, trust score drops, disable rate spikes. The metrics have to catch both failure modes simultaneously."

**Your position — the L5 dimension:**
> "Search Memory is high risk high reward for Google. The reward is a genuinely better search experience that deepens Google's moat against AI assistants — because a personalised Google knows you better than a stateless ChatGPT. The risk is that one privacy incident in this feature creates a narrative — Google is reading your mind — that damages the trust relationship underlying the entire advertising business. I'd launch with explicit opt-in, not opt-out. Opt-in self-selects for users who want personalisation, dramatically reducing the trust risk while still generating the data to prove the feature works."

---

## Day 1 — Hire / No Hire Signal

**Decision: No Hire at L5. Strong Hire at L4.**

---

### What Earned L4:
- Correct north star direction — SSR based
- Privacy instinct appeared unprompted
- Feature disable rate as behavioral trust signal — excellent
- Reformulation as the core signal — correct

### What's Missing For L5:

**Gap 1 — Feature isolation.**
You didn't distinguish between measuring Search overall vs measuring Search Memory specifically. L5 candidates immediately ask — how do I isolate the contribution of this feature from baseline search quality? Your north star was generic not feature-specific.

**Gap 2 — The sensitive query risk.**
You mentioned privacy but didn't name the specific harm — Memory surfacing sensitive searches in unintended contexts. L5 names the specific mechanism of harm not just the category.

**Gap 3 — No position.**
You didn't tell me whether Search Memory is good for Google long term. L5 candidates have a view. They connect the feature to the business strategy. Opt-in vs opt-out is a strategic decision that reveals business judgment. You didn't go there.

**Gap 4 — Cheatsheet opening.**
You started by reciting template phrases. L5 candidates open with a specific human being in a specific moment. Not a framework announcement.

---

## The L5 Standard For This Question

An L5 answer opens like this:

> "A graduate student who researches climate policy searches Google 40 times a day across overlapping topic clusters. Today she retypes her context every session — she searches 'carbon tax policy' then 'carbon pricing mechanisms' then 'EU ETS performance' as separate unconnected queries. Search Memory's promise is that by session three Google already understands she's researching EU climate policy and surfaces that context without her having to reconstruct it. Success means her third search in a cluster returns results as good as if she'd typed the full context. I'd measure that specifically — SSR on returning query clusters vs first-time queries — not overall SSR which already exists."

Notice: specific human, specific moment, specific measurement that isolates the feature's contribution. No framework announcement. Straight to the human.

---

## Day 1 Score

| Dimension | Score |
|---|---|
| Structure | 3/5 — present but template-heavy |
| User empathy | 2/5 — no specific human named |
| Metric intuition | 3/5 — right direction, not feature-specific |
| Structured curiosity | 3/5 — privacy instinct good, mechanism missing |
| Business judgment | 2/5 — no position taken |

**Overall: 13/25 — L4 bar**
Fair. Here's a completely different approach to the same question — different user, different angle, different north star logic.

---

## Alternate Approach — The Ecosystem Angle

**User:**
> "A 45 year old small business owner. She searches Google for supplier prices, competitor information, and regulatory updates — the same topic clusters every week for years. She doesn't think of herself as a power searcher. She just knows Google never quite gets her context right. She searches 'import duty rates' and gets generic results. She means import duty rates for electronics from Vietnam specifically. She's been giving Google this context implicitly for three years. Search Memory should already know."

**The reframe — this isn't a search quality feature. It's a context accumulation feature:**

> "Search Memory's value proposition isn't making individual searches better. It's making Google smarter about you over time. That's a fundamentally different thing to measure. A single search SSR improvement is the wrong north star. The right north star is cumulative context value — does Google get better at serving you the longer you use Search Memory?"

**North Star:**
> "SSR improvement rate over time for returning users with Memory on — specifically measuring whether SSR in week 8 is meaningfully higher than SSR in week 1 for the same query clusters. If Memory is working — the longer you use it the better your searches get. Flat SSR over time means Memory isn't accumulating useful context. Improving SSR over time means it is."

**The business angle nobody mentions:**
> "Search Memory creates something Google has never had before — a longitudinal model of individual user intent that compounds over time. That's not just a search quality feature. That's a moat against ChatGPT which starts every conversation from zero. The metric I'd add to the business layer is Memory retention rate — what % of users who try Search Memory keep it on after 30 days. High retention means users feel the compounding value. That's the signal Google should care about most strategically."

**The privacy angle — different from what I gave you before:**
> "The specific privacy risk I'd measure isn't just feature disable rate. It's cross-context bleed — cases where Memory surfaces information from one search context in a completely different context. User searches for divorce lawyers on Monday. Searches for family vacation on Friday. Memory surfaces divorce-adjacent results in the vacation search. That's not helpful. That's disturbing. I'd measure cross-context bleed rate — the % of personalised results that the user explicitly dismisses or overrides — as the leading indicator of the creepiness threshold being crossed."

**Position:**
> "Search Memory is Google's most important unreleased product strategy. Not because it improves search quality marginally — but because it makes Google irreplaceable in a way that stateless AI assistants cannot replicate. A Google that knows your three year search history is categorically more useful than a ChatGPT that forgets you after every conversation. The feature should launch with aggressive opt-in marketing — not buried in settings — because the strategic value compounds only when users actually use it and trust it."

---

## What's Different About This Approach vs The First One

| Dimension | First Approach | This Approach |
|---|---|---|
| User | Graduate researcher — high frequency same-day clusters | Small business owner — recurring weekly topic clusters |
| North Star logic | SSR on returning query types vs first-time | SSR improvement rate over time — compounding value |
| Privacy signal | Feature disable rate + sensitive query incidents | Cross-context bleed rate — wrong context surfacing |
| Business angle | Opt-in vs opt-out decision | Memory as strategic moat against ChatGPT |
| Frame | Search quality feature | Context accumulation feature — fundamentally different |

---

## The Lesson From Seeing Both

Same question. Two completely valid L5 answers. Different users. Different north stars. Different privacy mechanisms. Different business frames.

Neither is more correct than the other.

What makes both L5 is the same thing:

- Specific human being
- North star that isolates the feature's specific contribution
- Privacy guardrail with a named mechanism not just a category
- A business position that connects the feature to strategy

The frame you choose — search quality vs context accumulation — changes your entire answer. Both are defensible. Pick one and go deep. Don't try to cover both.

That's the L5 skill. Not knowing more answers. Knowing how to commit to one angle and defend it completely.

---


