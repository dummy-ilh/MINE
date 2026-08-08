# Chapter 1 — The Multi-Armed Bandit Problem

---

## 1.1 The story behind the name

Imagine you walk into a casino. There's a row of slot machines ("one-armed bandits" — "bandit" because they take your money, "one-armed" because of the pull-lever). Each machine has a different, *unknown* payout rate. Machine A might pay out on average $1 per pull. Machine B might average $0.50. Machine C might average $1.20. You don't know any of these numbers in advance — you only find out by pulling.

You have a fixed number of pulls (say, 1,000). Your goal: **walk out with as much money as possible.**

This is the entire multi-armed bandit (MAB) problem, in one paragraph. Everything in this syllabus is a variation on: *you have several options, each with an unknown reward, and you must decide, pull by pull, which one to try, balancing "try new things to learn more" against "use what you've already learned to cash in."*

That tension has a name: the **exploration-exploitation tradeoff**. We'll define it precisely in a moment, but hold onto the casino picture — every algorithm in this course is just a smarter way of deciding which slot machine to pull next.

---

## 1.2 Why should an ML engineer care about slot machines?

Because the slot machine is a stand-in for something you actually build:

| Casino framing | Real system |
|---|---|
| Slot machine ("arm") | An ad to show, a search result to rank, a recommendation, a UI variant, a price point |
| Payout of a pull | Click, purchase, watch-time, thumbs-up |
| Unknown payout rate | The true (unknown) click-through rate / conversion rate of that option |
| Pulling a lever | Serving that option to a real user and observing what happens |
| Limited pulls | Limited traffic / limited time before a decision is needed |

So "which slot machine should I pull?" becomes "which ad should I show this user right now, given that I don't yet know each ad's true click-through rate, and every impression I 'waste' on a bad ad is an impression I didn't spend on a good one?"

This is why bandits show up constantly in ad ranking, search ranking, recommendation systems, and app-store/assistant suggestion ranking at companies like Google and Apple — and why it's a favorite interview topic. It's a small, clean mathematical problem that maps directly onto money-making production systems.

---

## 1.3 Formal setup

Now let's write this precisely, because interviews will expect exact notation.

- There are **K arms** (options), indexed $i = 1, 2, \dots, K$.
- Each arm $i$ has a **true reward distribution** with mean $\mu_i$. This mean is *fixed but unknown* to you.
- Time proceeds in **rounds** $t = 1, 2, \dots, T$. $T$ is called the **horizon**.
- At each round $t$, you (the "agent" or "policy") choose one arm $A_t \in \{1, \dots, K\}$.
- You observe a **reward** $X_t$, drawn randomly from arm $A_t$'s distribution (mean $\mu_{A_t}$).
- You use everything you've observed so far — $(A_1, X_1), (A_2, X_2), \dots, (A_{t-1}, X_{t-1})$ — to decide $A_t$.
- Crucially, **you never observe the reward of an arm you didn't pull**. If you pull machine A, you learn nothing new about machine B's payout that round. This is called **bandit feedback** or **partial feedback**, and it's the single most important structural fact about this whole problem — contrast it with supervised learning, where you'd see the "correct answer" regardless of what you predicted.

Let $\mu^* = \max_i \mu_i$ be the mean reward of the **best arm**. Your goal, informally, is to make $A_t$ equal to the best arm as often as possible, as early as possible.

### Worked numerical example

Say $K = 3$ arms, with true means (unknown to the agent, known to us as the "casino owner" for illustration):

$$\mu_1 = 0.30, \quad \mu_2 = 0.50, \quad \mu_3 = 0.45$$

(Think of these as click-through rates: arm 1 converts 30% of the time, arm 2 converts 50% of the time, arm 3 converts 45% of the time — rewards are 0 or 1, i.e., Bernoulli.)

The best arm is arm 2, with $\mu^* = 0.50$.

Suppose in round 1 the agent (knowing nothing) picks arm 1. The reward $X_1$ is a coin flip that lands "1" with probability 0.30 — say it comes up $X_1 = 0$. The agent has now learned *one noisy data point* about arm 1's payout rate, and *nothing* about arms 2 or 3. That's the entire bandit feedback loop, one round at a time.

---

## 1.4 Bandits as a 1-step Markov Decision Process (MDP)

If you've studied reinforcement learning (or will, later), you'll recognize this immediately: a bandit is the simplest possible RL problem.

A full MDP has **states**, **actions**, **transitions between states**, and **rewards**. In a bandit:

- There is exactly **one state** (or no state at all) — nothing about the "world" changes between rounds. Pulling arm 1 doesn't change the payout rate of arm 2, and it doesn't move you into some different "situation."
- There are **actions** (the arms).
- There is **no transition** — every round starts fresh from the same (non-)state.
- There is a **reward**.

Because there's no state transition, there's no need for anything like a discount factor $\gamma$ or a value function over states — a bandit is "myopic" in the sense that every round is a fresh, independent decision (though *what you know* accumulates across rounds, even though the *world* doesn't change). This is why bandits are sometimes called **"one-step" or "stateless" reinforcement learning**.

**Contextual bandits** (Chapter 10 onward) add back a notion of "state" — the context (e.g., which user you're serving) — but still without transitions between states caused by your own actions. Full RL adds the transitions back in. So the hierarchy, from simplest to most complex, is:

$$\text{Bandits} \;\subset\; \text{Contextual Bandits} \;\subset\; \text{Full RL (MDPs)}$$

This containment relationship is a very common interview question — "how does a bandit relate to RL?" — and now you have the precise answer: it's an MDP with one state and no transitions.

---

## 1.5 The exploration-exploitation tradeoff, precisely

Informally: **exploitation** means picking the arm that currently looks best, based on the data you've gathered so far. **Exploration** means picking an arm you're less sure about, to gather more information, even though it might not currently look like the best option.

Let's make this concrete with our example. Suppose after 10 rounds, the agent has pulled arm 1 six times and observed an *average* reward of 0.33, and pulled arm 2 four times and observed an average reward of 0.25 (just bad luck — arm 2's true mean is actually 0.50, but with only 4 pulls, the observed average can easily be far from the truth).

- **Pure exploitation** says: arm 1 looks better (0.33 > 0.25) — keep pulling arm 1 forever.
- But arm 1's true mean is only 0.30, and arm 2's true mean is 0.50. Pure exploitation just permanently locked onto the *worse* arm because of an unlucky early sample on arm 2.

This is the core danger the entire field of bandit algorithms exists to solve: **with too little exploration, bad luck early on can permanently mislead a greedy policy.** With too much exploration, you waste rounds on arms you already have good evidence are bad. The entire syllabus from here is a series of increasingly clever mathematical answers to "how much should I explore, and which arm should I explore, given exactly what I currently know?"

---

## 1.6 The three bandit "families" — a map for the rest of the course

We'll go deep on each of these later, but you should have the shape of the whole landscape now:

1. **Stochastic bandits** (Chapters 2–9): each arm's reward comes from a fixed, unknown distribution (like our casino example). This is the "classical" setting, and where UCB and Thompson Sampling live.

2. **Adversarial bandits** (Chapter 8): no assumption of a fixed underlying distribution at all — rewards could be chosen by an adversary trying to make you look bad. Surprisingly, near-optimal algorithms (EXP3) still exist here.

3. **Contextual bandits** (Chapters 10–13): before choosing an arm, you observe some **context** (e.g., user features, query features), and the best arm can depend on that context. This is the setting that actually looks like most production ad/recommendation/ranking systems.

---

## 1.7 Production considerations (a first pass — we'll go much deeper in Phase 3)

Even at this early, "just the setup" stage, a few production realities are worth flagging, because interviewers love probing whether you understand that the textbook problem is a simplification:

- **Delayed reward.** In the casino, you see $X_t$ immediately after pulling. In a real ad system, "did the user convert" might not be known for hours or days. Algorithms that assume instant feedback need modification for this.
- **Non-stationary arms.** Real click-through rates drift — a slot machine in a textbook has a fixed $\mu_i$ forever, but a real ad's CTR decays as novelty wears off. (Chapter 13.)
- **Huge $K$.** A casino has 3–10 machines. A real ad-serving system might have millions of candidate ads. Plain "try every arm a bit" strategies don't scale — this is exactly why contextual bandits and function approximation (Chapter 11–12) matter in practice.
- **Batched, not sequential, updates.** The textbook model updates after every single pull. Real systems often serve millions of impressions between model-update cycles. This affects which algorithms are practical to deploy.

We're flagging these now so you build the right intuition from day one: **the clean math of Chapters 2–9 is a foundation, and Phase 3 (offline evaluation, production systems) is where you'll learn how the real deployed versions differ.**

---

## 1.8 Interview traps at the "problem setup" stage

Even before any algorithm is discussed, candidates lose points on this material in predictable ways:

- **Confusing "arm" with "state."** A common mistake is describing a bandit as having states, or saying "the bandit transitions between arms." There is no state and no transition in the plain (non-contextual) bandit — only actions and rewards. Say this explicitly if asked to define a bandit.
- **Forgetting bandit feedback is partial.** If asked "what makes this different from supervised learning," the correct core answer is: **you only observe the reward for the action you took, never for the actions you didn't take (no "ground truth" for the roads not taken).** This is sometimes called the **counterfactual problem**, and it's the entire reason Phase 3 (off-policy evaluation) is hard and important.
- **Jumping straight to an algorithm without stating the objective.** Interviewers often want to hear you state the goal (minimize regret, defined next chapter) before you start naming algorithms. Naming UCB or Thompson Sampling with no framing reads as memorization, not understanding.

---

## 1.9 L5-vs-L6 differentiating talking points

At the L5 level, correctly explaining the exploration-exploitation tradeoff and the casino/production mapping table above is sufficient. What separates L6-caliber answers:

- Spontaneously drawing the **containment hierarchy** (bandits ⊂ contextual bandits ⊂ full RL) without being asked, and explaining *why* each inclusion holds (what structural element is added at each step: context, then state transitions).
- Naming **specific production failure modes** unprompted (delayed reward, non-stationarity, scale) rather than needing to be prompted for them.
- Being able to say precisely *why* bandit feedback makes this fundamentally different from supervised learning — using the word "counterfactual" and connecting it forward to why off-policy evaluation (Phase 3) will be necessary, even before that material is taught. This shows the candcandidate sees the shape of the whole problem, not just the current chapter.

---

## 1.10 Comprehension checks

Before moving to Chapter 2, you should be able to answer all of these without notes:

1. In your own words, what is the exploration-exploitation tradeoff?
2. Why is a multi-armed bandit described as a "1-step" or "stateless" MDP? What's missing compared to a full MDP?
3. What does "bandit feedback" mean, and how is it different from the feedback you get in ordinary supervised learning?
4. In the worked numerical example (Section 1.3), why could an agent easily end up believing arm 1 is better than arm 2, even though arm 2's true mean reward is higher?
5. Name two things that make production bandit systems harder than the textbook formulation described in Section 1.3.

---

*Next: Chapter 2 — Regret, the metric that turns "explore vs. exploit" from a vague trade-off into something we can mathematically optimize.*
