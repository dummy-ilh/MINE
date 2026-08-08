# Chapter 10 — Contextual Bandits: Getting Started

*(Note: from this chapter on, going slower and lighter on notation — more plain language, fewer symbols at once, each equation introduced one small piece at a time.)*

---

## 10.1 The one-sentence idea

Every algorithm so far picks the same "best arm" for everybody, all the time. A **contextual bandit** picks a *different* best arm depending on **who you're serving right now**.

That's really it. Everything else in this chapter is just fleshing that one sentence out.

---

## 10.2 A simple story to anchor everything

Imagine you run a streaming app's homepage, and you have 3 things you could recommend: a comedy show, an action movie, and a documentary.

In the old (non-contextual) bandit world from Chapters 4–9, you'd assume there's *one* true best answer for *everyone* — say, "the comedy show has the highest average click rate across all users" — and you'd just try to find that one universal winner.

But that's obviously not how real users work. A teenager might love the comedy show. A retiree might love the documentary. If you only ever recommend "the one universal best option," you're leaving a lot of value on the table for everyone who isn't the "average" user.

**Contextual bandits fix this by letting the decision depend on information about the current user** — their age, their past-watch history, what device they're on, what time of day it is. This extra information is called the **context**.

---

## 10.3 What "context" means, concretely

The context is just a list of numbers describing the situation right now. For example, for one specific user visiting the homepage right now, the context might look like:

- Age: 34
- Number of comedies watched in the last month: 12
- Number of documentaries watched in the last month: 1
- Time of day: evening (let's say we encode this as 1 = evening, 0 = not evening)

That's it — four numbers describing "who is in front of us right now." We'll call this list of numbers $x$ (just a label — nothing fancy). Different users walking up to the homepage have different $x$'s.

---

## 10.4 What changes about the algorithm

In the old setup, each arm had **one single number** we cared about: its average reward, $\mu_i$. In the contextual setup, each arm's "goodness" now **depends on the context** — so instead of one number per arm, we need a **rule** per arm that takes the context in and spits out a predicted reward.

Think of it like this, plainly:

> "Given this user's context $x$, how good do I expect the comedy show to be *for this specific user*?"

We'll write this predicted goodness as $\mu_i(x)$ — read it simply as "arm $i$'s expected reward, *given* context $x$." The only thing that's new compared to Chapters 4–9 is this: **the reward we're trying to predict now depends on two things instead of one — which arm, AND who we're serving.**

---

## 10.5 Why this is genuinely harder — and why it's still learnable

Here's the catch, stated plainly: in the old bandit setting, there were only $K$ unknown numbers to learn (one mean per arm — maybe 3, maybe 10). In the contextual setting, there are effectively **infinitely many possible contexts** (every user is a little different), so there's no way to just "try every context enough times to learn its answer" the way we tried every arm enough times before.

The fix that makes this tractable: **we assume nearby/similar contexts have similar rewards**, and we use a simple, learnable *model* (like a straight-line/linear model — much more on this in Chapter 11) to predict $\mu_i(x)$ for **any** context $x$, even ones we've never seen exactly before, by generalizing from the contexts we *have* seen. This is exactly the same idea as ordinary machine learning (like regression) — we're just doing it *while* also deciding which arm to try, using bandit feedback (only seeing the reward for the arm we actually picked, same restriction as always).

---

## 10.6 A very simple worked example

Let's keep this as simple as possible. Say we only track **one** piece of context: how many comedies the user watched last month, call it $x$ (just a single number, like $x = 12$).

Suppose, in true reality (which the algorithm doesn't know), the comedy arm's expected reward follows a simple rule:

$$\mu_{\text{comedy}}(x) = 0.02 \times x$$

In plain words: **"for every extra comedy watched last month, this user's expected click probability on a new comedy recommendation goes up by 2 percentage points."** So:

- A user with $x = 5$ (watched 5 comedies last month): expected reward $= 0.02 \times 5 = 0.10$ (10% click chance)
- A user with $x = 20$: expected reward $= 0.02 \times 20 = 0.40$ (40% click chance)

Notice: **there is no longer one single number describing "how good is the comedy arm."** It genuinely depends on who's asking. That's the entire shift this chapter introduces, made as concrete as possible.

The documentary arm might have a completely different rule, e.g., $\mu_{\text{doc}}(x) = 0.30 - 0.01x$ (documentary appeal goes *down* as someone watches more comedies) — so for a heavy comedy-watcher ($x=20$), comedy wins (0.40 vs. $0.30 - 0.20 = 0.10$), but for someone who barely watches comedies ($x=1$), documentary wins ($0.30-0.01=0.29$ vs. comedy's $0.02$). **The best arm literally flips depending on who's in front of you** — this single sentence is the entire point of this chapter.

---

## 10.7 What "regret" means now

Regret still means the same basic thing as Chapter 2 — "how much worse did I do than the best possible choice" — just now the "best possible choice" is allowed to be a *different* arm for every different context, instead of one fixed best arm for everybody.

So the benchmark you're compared against, each round, is: **"whichever arm would have been best for *this specific user*, had you known their true $\mu_i(x)$ values in advance."** Using our example above: for the heavy comedy-watcher ($x=20$), the benchmark is comedy (0.40); for the light comedy-watcher ($x=1$), the benchmark is documentary (0.29). A good contextual bandit algorithm needs to correctly identify the *right* arm *per context*, not just one overall winner.

---

## 10.8 Why this connects back to ordinary supervised learning

Here's a genuinely useful way to think about it, in plain terms: a contextual bandit is a lot like a supervised learning problem (predict reward from context — just like predicting a label from features), **except you only ever get to see the "correct answer" (reward) for the one option you actually chose**, never for the options you didn't pick. This is exactly the bandit-feedback idea from Chapter 1, now layered on top of a prediction problem instead of a single unknown number. Keep this framing in your head going into Chapter 11 — LinUCB is really "ordinary linear regression, plus a bandit-style exploration bonus layered on top."

---

## 10.9 Production considerations (kept simple)

- This is **the single most common bandit setup used in real industry systems** — ad ranking, homepage recommendations, search result ranking, App Store/Assistant suggestions — because real users genuinely differ from each other, and ignoring that (as plain non-contextual bandits do) leaves real value on the table.
- Context isn't just about the user — it can include the **item** too (comedy show's genre tags, runtime, release year) and the **situation** (time of day, device). All of this together is often just concatenated into one long list of numbers, exactly like $x$ above, just longer.
- The "assume nearby contexts behave similarly" idea (Section 10.5) is doing a lot of work — if your assumed model shape is wrong (e.g., you assume a straight-line relationship but the real relationship is more complicated), your predictions can be systematically off. This tension — simple model (easier to learn quickly, less data needed) vs. complex model (more accurate, but needs much more data and is harder to attach clean bandit exploration guarantees to) — is a running theme through Chapters 11 and 12.

---

## 10.10 Interview traps (kept simple)

- **Saying a contextual bandit "just adds more features" without explaining that the best arm itself can now change per-user.** The key new idea isn't "more data" — it's that **the optimal decision is no longer the same for everyone**.
- **Forgetting that bandit feedback still applies.** You still only see the reward for the one arm you picked for that specific user — you don't get to peek at what would have happened if you'd shown that user a different option instead. This is why contextual bandits are still bandits, not just supervised learning.

---

## 10.11 Comprehension checks — try to answer these in plain words, no formulas needed

1. In one or two sentences, what's the core difference between a regular bandit and a contextual bandit?
2. In the worked example (Section 10.6), why does the best arm (comedy vs. documentary) depend on the user, instead of there being one universal answer?
3. Why can't we just "try every possible context enough times," the way we tried every arm enough times in Chapters 4–9?
4. How is a contextual bandit similar to ordinary supervised learning — and what's the one key way it's still different (still a "bandit")?

---

*Next: Chapter 11 — LinUCB, where we take the "optimism" idea from UCB1 (Chapter 4) and combine it with a simple straight-line prediction model, to build the single most commonly-referenced contextual bandit algorithm in interviews.*
