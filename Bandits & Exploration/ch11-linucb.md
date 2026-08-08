# Chapter 11 — LinUCB

*(Same slower, simpler style as Chapter 10 — plain language first, one small formula at a time.)*

---

## 11.1 The one-sentence idea

LinUCB is just **UCB1's "be optimistic about what you're unsure of" idea (Chapter 4), combined with a simple straight-line prediction model for each arm** (like the comedy-show example from Chapter 10).

That's the whole chapter, in one sentence. Let's build it up slowly.

---

## 11.2 Step 1: predicting reward with a straight line

Recall from Chapter 10 our simplified comedy example:

$$\mu_{\text{comedy}}(x) = 0.02 \times x$$

This was a **straight-line rule**: take the context number $x$, multiply it by some weight (0.02 here), and that's your prediction. In real life, you don't know that "0.02" in advance — **that's exactly the number you have to learn from data.** LinUCB's job is to learn that weight (and to do so per arm, since each arm can have its own weight), while also deciding which arm to try, using bandit feedback.

Let's give the unknown weight a name: call it $\theta$ (just a label, like calling an unknown "$x$" in algebra — nothing fancy). So the model for one arm is:

$$\text{predicted reward} = \theta \times x$$

If a user has $x = 20$ (watched 20 comedies last month) and our current best guess for $\theta$ is $0.018$, our prediction is $0.018 \times 20 = 0.36$.

**With more than one context feature** (say, both "comedies watched" and "time of day"), the idea is exactly the same, just with one weight per feature, added together — e.g., $\text{predicted reward} = \theta_1 \times x_1 + \theta_2 \times x_2$. Nothing conceptually new — just more terms in the same sum. We'll mostly stick to one feature in our worked examples to keep the numbers easy to follow, but know that in practice there are usually many features.

---

## 11.3 Step 2: learning $\theta$ from data (ridge regression, in plain words)

Every time you show the comedy arm to a user and observe whether they clicked, you get one more (context, reward) data point — e.g., "$x=20$, clicked" or "$x=5$, didn't click." **Ridge regression** is just the standard, well-known method for finding the "$\theta$" that best fits all your data points so far, with one small addition: it gently pulls $\theta$ toward zero when you don't have much data yet (this is the "ridge" part), so early guesses aren't wild before you've seen much evidence — similar in spirit to how the Beta(1,1) prior in Chapter 6 gently pulled early Thompson Sampling estimates toward the middle before data came in.

**You don't need to derive the ridge regression formula by hand for this course** — what matters for the interview is knowing (a) it's just "line of best fit, with a small stabilizing nudge," and (b) it also naturally tells you **how confident you should be** in your current $\theta$ estimate — which is the ingredient LinUCB actually needs next.

---

## 11.4 Step 3: turning "how confident" into an exploration bonus

This is the heart of the chapter, and it's a direct echo of Chapter 4. Recall UCB1's rule:

$$\text{UCB score} = \text{sample mean} + \text{bonus (bigger when you have less data)}$$

LinUCB does the exact same thing, just swapping "sample mean" for "the straight-line prediction," and "bonus based on number of pulls" for "bonus based on how unfamiliar this particular context is":

$$\text{LinUCB score for arm } i = (\text{predicted reward using current } \theta_i) + (\text{bonus based on how unfamiliar this context is})$$

**Where does "how unfamiliar this context is" come from?** In plain words: if you've seen lots of users with $x$ around 20, and a new user also has $x = 20$, you're on familiar ground — small bonus. If you've never seen any user with $x$ anywhere near, say, $x = 90$, your straight-line prediction there is much shakier — even if the line itself gives some specific number, you should trust it less — bigger bonus. Ridge regression's math (which we're not deriving by hand here) naturally produces exactly this "how unfamiliar is this input" number as a side effect — it's usually called the context's **uncertainty** for that arm.

**So the full LinUCB picture, in words**: for each arm, use your current straight-line model to make a prediction for this specific user, add a bonus that's bigger when this user's context is unfamiliar territory for that arm's model, and pick whichever arm has the highest total score. Exactly UCB1's philosophy — just applied per-context instead of per-arm-as-a-whole.

---

## 11.5 A very simple worked example

Let's keep this concrete and small. One arm (comedy), one context feature ($x$ = comedies watched last month). Suppose after a little bit of data, our current fitted line gives $\theta = 0.018$.

**User A**: $x = 20$, and we've seen *many* previous users with $x$ around 20, so the "unfamiliarity bonus" is small — say $0.02$.

$$\text{Score} = (0.018 \times 20) + 0.02 = 0.36 + 0.02 = 0.38$$

**User B**: $x = 90$ (an unusually heavy comedy-watcher — we've barely seen anyone like this before), so the unfamiliarity bonus is much larger — say $0.25$.

$$\text{Score} = (0.018 \times 90) + 0.25 = 1.62 + 0.25 = 1.87$$

(Yes, this can go above 1 — that's fine, it's just a score used for comparing arms, not literally a probability.)

**The takeaway from this example**: even though User B's raw prediction ($1.62$) already looks bigger than User A's ($0.36$) just from the line itself, notice the *bonus* is also much bigger for User B — because User B is unfamiliar territory. This is LinUCB happily leaning toward trying the comedy arm on User B, partly *because* we don't have much data on comedy-watchers that heavy yet — exactly the same "give under-explored situations a fair, principled chance" behavior we saw with plain UCB1 back in Chapter 4, just now playing out **per-context** instead of per-arm.

---

## 11.6 Disjoint LinUCB vs. Hybrid LinUCB — in plain words

Once you have **multiple arms** (comedy, action, documentary), there's a design choice: does each arm get its own **completely separate** straight-line model, or do arms **share some parts** of the model?

- **Disjoint LinUCB**: every arm has its own totally separate $\theta$, learned only from data where that specific arm was shown. Simple, easy to reason about — but if you have a *lot* of arms (e.g., thousands of possible ads), each one individually needs a fair amount of data before its own line is any good, which can be slow.

- **Hybrid LinUCB**: arms share some **common** weights (e.g., "time of day matters the same way across all arms") in addition to each arm having its own **arm-specific** weights. This lets data from *one* arm help improve predictions for *other* arms too, through the shared part — much more data-efficient when you have many arms, at the cost of a more complex model to set up.

**Simple rule of thumb for an interview answer**: use disjoint LinUCB when you have a small number of arms and plenty of data per arm; reach for hybrid LinUCB when you have many arms (especially new/rarely-shown ones) and want data to generalize across arms, not just within each arm separately.

---

## 11.7 Why this is the most interview-tested algorithm in this course

LinUCB shows up constantly in interviews for one simple reason: it's the natural, practical answer to "how would you build a real recommendation/ad-ranking system that adapts to each user, with principled exploration, without needing anything exotic?" It's simple enough to describe on a whiteboard in a few minutes (as we just did), grounded in the same "optimism" idea you already know from UCB1, and it's genuinely what many real production systems use or started from. Being able to walk through Sections 11.2–11.5 fluently, in plain language, is one of the highest-value things you can prepare in this entire course.

---

## 11.8 Production considerations (kept simple)

- **LinUCB assumes the true relationship is a straight line (or close to it).** Real relationships are often more complicated — this is a real limitation, and it's exactly why Chapter 12 introduces neural-network-based versions for when a straight line isn't good enough.
- **The "unfamiliarity bonus" (Section 11.4) naturally handles new users and new situations gracefully** — a brand new user with an unusual context automatically gets treated as "worth a bit of extra exploration," without needing any special-cased cold-start logic bolted on separately. This graceful handling of new/unusual situations is a big reason LinUCB is popular in practice.
- **Disjoint vs. Hybrid (Section 11.6) is a genuinely important real design decision**, not just theory — companies with huge ad/content catalogs (thousands+ of arms) lean toward hybrid-style sharing, almost out of necessity, since disjoint models would take too long to individually warm up.

---

## 11.9 Interview traps (kept simple)

- **Describing LinUCB as "just linear regression."** It's linear regression **plus** the UCB-style exploration bonus on top — leaving out the bonus term misses the entire "bandit" part of "contextual bandit," and would just be a supervised-learning answer to a bandit question.
- **Forgetting that the bonus depends on the *context*, not just on "how many times has this arm been pulled."** This is the key upgrade from Chapter 4's UCB1 (bonus depended only on pull-count) to LinUCB (bonus depends on how unfamiliar *this specific situation* is) — losing this distinction is a common and telling mistake.
- **Not being able to give a simple, plain reason to prefer disjoint vs. hybrid** when asked — "small number of arms with plenty of data each → disjoint; many arms, especially new ones → hybrid" is enough of an answer; you don't need to go deeper unless asked.

---

## 11.10 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: correctly explain that LinUCB combines a straight-line prediction model with a UCB-style bonus, and can describe disjoint vs. hybrid LinUCB at a high level.
- **L6 bar**:
  - Walks through a simple worked example like Section 11.5 unprompted, and specifically points out the moment where an unfamiliar context earns a large bonus — showing real mechanical understanding, not just definitions.
  - Gives a grounded, scenario-specific recommendation for disjoint vs. hybrid (e.g., "for a small set of homepage content categories I'd go disjoint; for a large, fast-rotating ad inventory I'd lean hybrid, since new ads need to borrow strength from the shared weights right away").
  - Connects LinUCB's "unfamiliar context gets a bigger bonus" behavior explicitly back to plain UCB1's "under-sampled arm gets a bigger bonus" behavior from Chapter 4, showing they see LinUCB as a natural, motivated extension rather than a brand-new, disconnected algorithm.

---

## 11.11 Comprehension checks — plain words, minimal formulas

1. In one sentence, what two ingredients does LinUCB combine?
2. In the worked example (Section 11.5), why did User B (an unusual, rarely-seen context) get a large bonus, even though their raw prediction was already high?
3. What's the difference between Disjoint LinUCB and Hybrid LinUCB, and when would you pick one over the other?
4. Why is it wrong to describe LinUCB as "just linear regression"?
5. How does LinUCB's exploration bonus differ from plain UCB1's exploration bonus (Chapter 4) — what does each one depend on?

---

*Next: Chapter 12 — Linear Thompson Sampling & Neural Bandits, where we do the same "add context" upgrade to Thompson Sampling instead of UCB, and briefly cover what happens when a straight line genuinely isn't good enough and you need a neural network instead.*
