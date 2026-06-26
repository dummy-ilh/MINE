## Lesson 1: What Is Product Sense, Really?

Most candidates think product sense means "knowing how to improve apps." That's surface-level.

**Real product sense is the ability to connect user behavior to business outcomes — and make decisions under ambiguity.**

Here's the mental model I use to evaluate candidates:

```
User has a need
    ↓
Product creates a behavior
    ↓
Behavior generates a signal (data)
    ↓
Signal informs a decision
    ↓
Decision changes the product
    ↓
(repeat)
```

A product data scientist lives in **all five layers simultaneously.** You're not just an analyst reading dashboards. You're the person who understands *why* a user does something, *what* to measure about it, and *what to do* when the number changes.

---

### The Three Failure Modes I See in Interviews

**Failure 1 — The Feature Monkey**
Candidate jumps straight to "we should add a dark mode / notification / AI feature." No grounding in user problems. No metric attached. Just vibes.

**Failure 2 — The Dashboard Reader**
Candidate can describe metrics fluently but can't explain *why* a metric matters to a user or a business. They say "DAU dropped" but can't tell me what human behavior that represents.

**Failure 3 — The Framework Robot**
Candidate memorizes HEART or AARRR and applies it mechanically. Interviewers smell this immediately. Frameworks are scaffolding — not the building.

---

### The One Thing Great Candidates Do Differently

They **start with the human.**

Before touching any metric, framework, or feature idea — they ask: *"Who is this person, what are they trying to do, and what does success feel like for them?"*

That's the foundation everything else is built on.

---

** — answer this:**

> A friend who knows nothing about tech asks you: *"What does Instagram actually want its users to do?"*

Tell me your answer in plain language. No jargon. Pretend you're explaining it over dinner.


"Instagram wants you to keep coming back every day. To do that, it needs to show you stuff you actually enjoy — videos, photos, memes, whatever keeps you scrolling. The more you enjoy it, the more you open it tomorrow. And every time you're scrolling, Instagram quietly slips in ads. Those ads work really well because you're already in a good mood, already engaged. So businesses pay Instagram a lot of money to reach you in that moment. That's the whole game — make you happy enough to return, and sell that attention to brands."
---


## Lesson 2: The Interviewer's Hidden Rubric

Here's something no prep course tells you. When I'm sitting across from you, I'm not just checking if your answer is "correct." There's no answer key.

I'm scoring you on **5 dimensions.** Silently. The whole time.

---

## The 5 Dimensions

### 1. Structure
*Can you think in an organized way under pressure?*

I'm watching: Do you jump around? Do you contradict yourself? Do you know where you're going when you start talking?

The best candidates **signal their structure out loud.**

> "I want to approach this in three parts — first the user, then the metric, then what I'd do about it."

That one sentence tells me you're in control. It also buys you thinking time.

---

### 2. User Empathy
*Do you actually understand that real humans use these products?*

This is what we just practiced. I'm listening for whether you can describe a user's life, motivation, and feeling — not just their "behavior" in the abstract.

**Weak:** "Users want to engage with content."

**Strong:** "A 24 year old opening Instagram on her lunch break wants a quick hit of something fun or relatable. She has 8 minutes. She's not looking for anything specific."

One of those is a data point. The other is a human being. I remember the human being.

---

### 3. Metric Intuition
*Do you know what to measure and why?*

This is where most data scientists think they'll shine — and where many still fail. Because knowing *how* to calculate a metric is not the same as knowing *which* metric matters and *what it's actually telling you about human behavior.*

I'll ask you: "How would you measure success for this feature?"

I'm not looking for a list of 10 metrics. I'm looking for **one or two metrics you can defend deeply.**

---

### 4. Structured Curiosity
*When something is wrong, do you know how to find out why?*

This is the diagnosis skill. A metric dropped. What do you do?

Weak candidates guess. Strong candidates **build a hypothesis tree** — systematically ruling things in and out before drawing conclusions.

I'll teach you exactly how to do this in Module 2.

---

### 5. Business Judgment
*Do you understand what actually matters to the company?*

This is the senior dimension. Junior candidates optimize features. Senior candidates ask "should we build this at all, and what does it cost us if we're wrong?"

I'm listening for whether you can zoom out — connect a product decision to revenue, competition, user trust, or long term strategy.

---

## How These Dimensions Are Weighted

Here's something important. The weighting shifts by level:

| Level | What matters most |
|---|---|
| L3 / Entry | Structure + Metric Intuition |
| L4 / Mid | User Empathy + Metric Intuition |
| L5 / Senior | Business Judgment + Structured Curiosity |

If you're interviewing for a senior role and you're only showing metric intuition — you're leaving the most important dimension on the table.

---

## The Meta-Skill Behind All 5

Every dimension is really testing the same underlying thing:

> **Can you make a good decision with incomplete information, and explain your reasoning clearly?**

That's the job. Every day. The interview is just a simulation of it.

---

## One Practical Trick

At the start of any product sense question, before you say anything substantive — say this:

> "Can I take 30 seconds to structure my thoughts?"

Then actually do it. Jot notes if you can. Every strong candidate I've seen does this. It signals confidence, not weakness. Weakness is rambling for 3 minutes and hoping something lands.

---
The order we'll build them in:

Structure — first, because it holds everything else together
User Empathy — second, because it's the foundation of every answer
Metric Intuition — third, the technical core
Structured Curiosity — fourth, the diagnosis skill
Business Judgment — last, because it needs all four others underneath it

---
## Lesson 3: The Anatomy of a Great Product

Before you can measure a product, improve a product, or diagnose a product — you need to understand what a product actually *is* at its core.

Most candidates treat products as collections of features. That's wrong. And it shows in their answers.

---

## A Product Is a Promise

Every great product makes a specific promise to a specific person.

- Google Search: **"Find what you're looking for in under 10 seconds."**
- Instagram: **"Never be bored or disconnected from people you care about."**
- Google Maps: **"Never be lost or late."**
- Spotify: **"Always have the right music for this moment."**
- Uber: **"A car will be there in minutes, anywhere."**

Notice what these promises are NOT. They're not feature lists. They're not mission statements. They're the **feeling the user gets when the product works perfectly.**

That promise is your north star for everything — metrics, features, tradeoffs, diagnostics.

---

## The Three Layers of Any Product

Every product has three layers. Most candidates only see one.

```
Layer 1 — Core Value
What fundamental problem does it solve?

Layer 2 — Experience
How does solving that problem feel?

Layer 3 — Business Model
How does the company capture value from solving it?
```

Let me show you this with Uber:

**Layer 1 — Core Value**
Connects people who need rides with people who drive. Solves the "I need to get somewhere without a car" problem.

**Layer 2 — Experience**
No cash, no calling, no uncertainty. You see the car moving toward you. You know the price upfront. You rate the driver. It feels modern, safe, in control.

**Layer 3 — Business Model**
Takes a cut of every ride. More drivers = lower wait times = more riders = more rides = more revenue. Classic marketplace flywheel.

---

## Why This Matters in Interviews

When you're asked "how would you improve Uber" — candidates who only see Layer 1 say things like:

> "Add more payment options" or "add a scheduling feature"

Candidates who see all three layers say:

> "The core promise is reliable transportation on demand. The experience breaks down when wait times spike or when the rider feels unsafe. The business breaks down when drivers churn. So I'd focus on the moment where all three layers are most at risk — which is the first 60 seconds after a rider requests a ride. That's where anxiety peaks, cancellations happen, and trust is built or broken."

Same product. Completely different depth of thinking.

---

## The Flywheel — The Most Important Business Pattern You Need to Know

Most scaled tech products run on a flywheel. Understanding it changes how you think about metrics and decisions.

Here's the basic pattern:

```
More users → More value → More users
```

But each product has its own specific version. Let me show you two:

**Google Search Flywheel:**
```
More searches
→ More data on what people want
→ Better search results
→ More trust in Google
→ More searches
```

**Airbnb Flywheel:**
```
More hosts
→ More listings, more locations
→ More guests find what they need
→ More bookings
→ More hosts want to join
→ More hosts
```

**Why this matters for you:**

When a flywheel product has a metric drop — you don't just look at that metric in isolation. You ask: *where in the flywheel did it break?* Because a break at one point cascades everywhere eventually.

---

## The Concept of the "Aha Moment"

Every great product has a moment where the user first *gets it.* Where the promise clicks.

- Uber: The first time you watch the car move toward you on the map in real time
- Spotify: The first time Discover Weekly gives you a song you love that you never heard before
- Instagram: The first time a photo you posted gets 50 likes from people you care about

**Why this matters:**

The Aha moment is almost always the most important thing to protect with your metrics. If users don't reach it — they churn. If they reach it quickly — they stay forever.

In interviews, when asked about onboarding, retention, or engagement — anchor to the Aha moment. Ask yourself: *what is the moment this product becomes indispensable, and how do we get every user there faster?*

---

## Your First Real Practice

Put all three layers together for this product:

> **YouTube**

Tell me:
- The core promise
- Where the experience most commonly breaks down
- What the flywheel looks like
- What the Aha moment is
Perfect. You just naturally did something most candidates take weeks to learn.

---

## What You Actually Did

You asked three questions instinctively:

**"Is it a repeating pattern"** — Is this real or noise?

**"Is it region specific"** — Is this everywhere or isolated?

**"Was anything introduced"** — Did we cause this?

That's the skeleton of a professional metric investigation. You didn't know that. You just thought logically.

---

## What This Tells Me About You

Your instincts are good. Genuinely. What you're missing is not intelligence — it's **vocabulary and structure** to express what you're already thinking, faster and more completely.

In an interview that answer would come out scattered. We need to make it come out **crisp and sequential** so the interviewer can follow your logic in real time.

That's exactly what Lesson 4 teaches.

---

## One Thing To Notice

Your three questions had a hidden order to them:

First you questioned **the data itself** — is this pattern real?

Then you questioned **the scope** — is this everywhere?

Then you questioned **the cause** — did something change?

That order is not random. It's actually the correct order of investigation. You go from "is this real" before you ever ask "why did this happen."

Most panicked candidates skip straight to why. They start theorizing causes before confirming the data is even telling them something true.

You didn't do that. Hold onto that instinct.

---

How would you answer. Dau dropping. Assume google search, insta, siri calls. 3 cases

Great question. Let me walk you through all three like I'm in the interview room. Watch the structure — same skeleton every time, different thinking inside it.

---

## The Universal Skeleton First

Every metric drop investigation follows this order:

```
1. Is this real?
2. Is this isolated or widespread?
3. Did we cause it?
4. Did the world cause it?
5. What user behavior does this represent?
```

Never skip steps. Never reorder them. Now let's apply it.

---

## Case 1 — Google Search DAU drops 15%

**Step 1 — Is this real?**

Is the data pipeline broken? Did tracking code change? Did a logging system fail? Google operates at such massive scale that a 15% drop could easily be a measurement artifact before it's a real phenomenon.

I'd check: is this drop consistent across all measurement systems we have, or only one dashboard?

**Step 2 — Is it isolated?**

- Is it one region or global?
- Is it one platform — mobile app, desktop, iOS, Android — or all of them?
- Is it one user segment — new users, returning users, a specific age group?

A global drop across all platforms is a five alarm fire. A drop only on Android in Southeast Asia is a very different problem.

**Step 3 — Did we cause it?**

- Did we ship anything in the last 48 hours? New ranking algorithm, UI change, autocomplete update?
- Did a feature experiment go wrong and get exposed to too many users?
- Did we change anything in notifications or the app that would reduce entry points?

**Step 4 — Did the world cause it?**

- Is there a holiday, election, or major news event that changes search behavior?
- Did a competitor launch something? A new AI search product that pulled users away?
- Is there an internet outage in a major market?

**Step 5 — What user behavior does this represent?**

This is the most important question and most candidates never get here.

DAU dropping means fewer people opened Google Search today. That means one of three things:

- They had fewer questions — unlikely, humans always have questions
- They went somewhere else to ask — ChatGPT, Perplexity, TikTok search
- Something broke and they couldn't get in

The scariest scenario for Google is option two. Because that's not a bug. That's a behavior shift. And behavior shifts don't fix themselves.

---

## Case 2 — Instagram DAU drops 15%

Same skeleton. Very different texture.

**Step 1 — Is this real?**

Same data integrity check. But Instagram also has a specific thing to check — did Apple or Google push an OS update that broke the app? Instagram is almost entirely mobile. An iOS update that crashes the app would tank DAU immediately.

**Step 2 — Is it isolated?**

- iOS vs Android split is critical here
- Is Reels DAU down or just Feed DAU? They measure separately
- Is it a specific country? Instagram is massive in India, Brazil, US — a drop in one of those moves the global number significantly

**Step 3 — Did we cause it?**

- Algorithm change to feed ranking?
- Did we change how Stories are surfaced?
- Any change to the home screen entry point?
- Did a new feature launch that's cannibalizing DAU — like if people are spending time in a new tab and not being counted correctly?

**Step 4 — Did the world cause it?**

- Is TikTok doing something? A viral TikTok moment that pulled attention away?
- Any PR crisis — a news story about Instagram harming mental health that caused a delete-the-app wave?
- Celebrity or cultural moment happening on a different platform?

**Step 5 — What user behavior does this represent?**

Instagram DAU dropping means fewer people opened the app today. For Instagram specifically I'd ask:

- Did they open it and leave immediately — so sessions are shorter but DAU is actually fine? Check session depth.
- Or did they genuinely not open it at all?

The difference matters enormously. Not opening at all suggests a habit broke. Opening and leaving suggests the content quality dropped. Two completely different fixes.

The deeper fear for Instagram is that DAU drop is a leading indicator of a generational shift. Younger users leaving for TikTok. That doesn't show up as a dramatic single day crash — it shows up as a slow 15% drift that feels like noise until it's too late.

---

## Case 3 — Siri Calls DAU drops 15%

This one is the most interesting. And the most different.

**Step 1 — Is this real?**

Siri "calls" — meaning invocations — are measured differently than app opens. Check whether the measurement of what counts as a Siri call changed. Did a logging update exclude certain invocation types — like Type to Siri or CarPlay?

**Step 2 — Is it isolated?**

- Is it Hey Siri drops or button press drops or both?
- Is it a specific device — iPhone, iPad, Apple Watch, HomePod, CarPlay?
- Is it a specific use case — music requests, navigation, messaging, general questions?

This matters enormously because Siri is not one product. It's ten products inside one name. A drop in HomePod Siri is completely unrelated to a drop in CarPlay Siri.

**Step 3 — Did we cause it?**

- Did an iOS update ship that changed the Hey Siri trigger sensitivity?
- Did we change Siri's response quality and users learned it's worse so they stopped trying?
- Did we change a default — like if Maps now opens instead of Siri for certain intents?

**Step 4 — Did the world cause it?**

- Did a new iPhone ship without something users relied on?
- Is there a cultural moment around AI assistants — maybe ChatGPT on iPhone is now so good that users switched?
- Seasonal — are people driving less, meaning CarPlay invocations are down?

**Step 5 — What user behavior does this represent?**

This is where Siri gets philosophically interesting.

Siri DAU dropping could mean:

- Users tried Siri, it failed them, they stopped trusting it
- Users found a better alternative for the same jobs
- The hardware moment that triggers Siri — driving, cooking, hands busy — happened less often

The first two are product failures. The third is just life.

But here's the Siri-specific insight that separates good answers from great ones:

> **Siri has a trust debt problem. Every failed response doesn't just lose that interaction — it reduces the probability the user tries again tomorrow.**

So a 15% DAU drop in Siri might represent months of accumulated failures finally crossing a threshold. It's not one bad day. It's the day users collectively gave up.

That changes your entire response. You're not looking for what happened this week. You're looking for a quality degradation trend that started months ago.

---

## What To Notice Across All Three

Same skeleton. But the **texture, the fear, and the implication** is completely different for each product.

Google's fear is substitution — someone else answering questions better.

Instagram's fear is generational abandonment — young users quietly leaving.

Siri's fear is trust erosion — users stopped believing it works.

**In an interview, naming the specific fear shows business judgment.** It tells the interviewer you understand what each company is actually protecting.

---

Ready for Lesson 4 now — The Goal → Metric → Signal chain. We'll turn this investigation instinct into a proactive system, not just a reactive one.


## The Master Question Bank

Reusable across any product, any metric drop, any interview. Organised by the 5 steps of the skeleton.

---

## Step 1 — Is This Real? (Data Integrity)

These apply to every single case. Always start here.

- Is this drop consistent across all our measurement systems or just one dashboard?
- Did our data pipeline or logging system change recently?
- Did our tracking code get updated or break?
- Is this a reporting lag or a genuine real-time drop?
- Are other metrics moving in a way that confirms this drop is real?
- Is the definition of this metric the same as it was last week?
- Did we change what counts as a "daily active user" recently?

---

## Step 2 — Is It Isolated? (Segmentation)

**By Geography**
- Is this global or region specific?
- Which regions are affected and which are not?
- Is this concentrated in one country that drives a large share of our total number?
- Is there a timezone pattern — did it start in one region and spread?

**By Platform**
- Is this iOS, Android, desktop, or all platforms?
- Is this app, mobile web, or desktop web?
- Is this a specific OS version — did an Apple or Google update just ship?
- Is this a specific device type — phone, tablet, watch, TV?

**By User Segment**
- Is this new users, returning users, or both?
- Is this power users or casual users?
- Is this a specific age group or demographic?
- Is this paid users or free users?
- Is this users in a specific acquisition channel?

**By Feature or Surface**
- Is this drop across the whole product or one specific feature?
- Is one entry point broken while others are fine?
- Is this affecting one use case but not others?

**By Time**
- Is this a one day drop or a multi day trend?
- Is there a day of week pattern?
- Is this a gradual decline or a sudden cliff?
- Did it start at a specific hour — which would suggest a deployment?

---

## Step 3 — Did We Cause It? (Internal Changes)

**Product and Engineering**
- Did we ship any code, feature, or experiment in the last 48-72 hours?
- Did an A/B test accidentally get exposed to too large a segment?
- Did we change any default settings or permissions?
- Did we change a core user flow — onboarding, login, home screen?
- Did we change an algorithm — ranking, recommendations, notifications?
- Did we reduce or change notifications that drive re-engagement?
- Did we change the app's entry points or navigation structure?

**Data and Measurement**
- Did we change how we define or count this metric?
- Did we migrate to a new analytics system?
- Did a third party SDK we rely on for tracking break?

**Infrastructure**
- Was there an outage or degraded performance in the last 48 hours?
- Did latency increase in a way that would make users give up?
- Did crash rates go up?

---

## Step 4 — Did The World Cause It? (External Changes)

**Competition**
- Did a competitor launch something significant?
- Did a competitor run a major marketing campaign?
- Is there a new product in the market solving the same need differently?

**Platform and Distribution**
- Did Apple or Google change App Store policies or rankings?
- Did an OS update break or change something about how our app works?
- Did a distribution partner change something?

**Macro and Cultural**
- Is there a public holiday in a major market?
- Is there a major news event pulling attention elsewhere?
- Is there a seasonal pattern we'd expect at this time of year?
- Is there a PR or trust crisis around our product or company?
- Did a viral social moment happen on a competitor platform?

**Regulatory**
- Did a government restrict or ban our product in a market?
- Did a new privacy law change what data we can collect or how we can notify users?

---

## Step 5 — What User Behavior Does This Represent?

These are the deepest questions. Most candidates never get here.

- What does a daily active user actually do in this product — what habit are they expressing?
- Did users try to use the product and fail, or did they not try at all?
- Is this a one time abandonment or a permanent habit break?
- Is this a leading indicator of something bigger — a slow generational or behavioral shift?
- What job were users hiring this product to do, and are they now hiring something else?
- Is this metric drop hiding a quality problem that's been accumulating for months?
- Are users still coming but doing less — or are they not coming at all?
- Is there a trust element — did something happen that made users less willing to engage?

---

## Bonus — Questions Specific To Two Sided Platforms

Use these whenever the product is a marketplace, social network, or creator platform.

- Is the drop on the supply side, demand side, or both?
- Did something change for creators/sellers that's now affecting viewers/buyers?
- Is the content or inventory quality dropping, which is causing demand to drop?
- Is a specific creator segment churning and taking their audience with them?
- Is the matching quality between supply and demand degrading?

---

## How To Use This In An Interview

Don't recite this list. That's the framework robot failure mode we talked about in Lesson 2.

Instead, use it as a **mental checklist you run silently** while you speak out loud about the most interesting and relevant questions for that specific product.

A good interview answer picks **6-8 of these questions**, sequences them logically, and explains *why* each one matters for *this specific product.*

The questions you choose and the order you choose them in — that's what shows your product judgment.

---

