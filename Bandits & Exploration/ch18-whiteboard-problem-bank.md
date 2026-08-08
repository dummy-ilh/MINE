# Chapter 18 — Whiteboard Problem Bank

*(Same slower, simpler style — plain language first, light on notation. This chapter is hands-on practice, not new theory.)*

---

## 18.1 What this chapter is for

Everything in Chapters 1–17 was building understanding. This chapter is pure **practice** — the kind of thing you'd actually be asked to do live, on a whiteboard or in a shared coding environment, in an interview. Three exercises:

1. Derive UCB1's confidence bound from Hoeffding's inequality, live, in simple steps.
2. Write ε-greedy, UCB1, and Thompson Sampling from scratch in plain Python — no libraries beyond basic randomness.
3. Hand-trace regret growth for a small 3-arm example, comparing how each algorithm behaves.

Take these slowly — the goal isn't to memorize the code, it's to be able to *reproduce* this kind of thing fluently under interview pressure.

---

## 18.2 Exercise 1: Deriving UCB1's bound from Hoeffding, step by step

Recall Hoeffding's inequality from Chapter 4 (Section 4.2), in plain words: *"the chance that the true mean is bigger than your sample mean plus a specific padding amount is small."* Written out:

$$P\Big(\mu > \hat\mu_n + \sqrt{\frac{\ln(1/\delta)}{2n}}\Big) \leq \delta$$

**The derivation, broken into small steps** (this is exactly how you'd talk through it live):

**Step 1 — decide how "small" you want the failure chance to be, and let it depend on time.** We want this confidence bound to hold well across the *whole* game, not just one round — so instead of a fixed $\delta$, we let it shrink as the round number $t$ grows (so our confidence gets stricter over time, matching intuition — more rounds played, more chances for something rare to happen, so we should demand more confidence). A standard, simple choice: $\delta = 1/t^4$ (a small number that shrinks fast as $t$ grows — the exact power, 4, is a technical detail from the full proof; you don't need to derive *why* 4 specifically, just know this is where "$t$" enters the formula).

**Step 2 — plug this choice of $\delta$ into the padding term.** The padding term was $\sqrt{\ln(1/\delta)/(2n)}$. Substituting $\delta = 1/t^4$:

$$\sqrt{\frac{\ln(t^4)}{2n}} = \sqrt{\frac{4\ln t}{2n}} = \sqrt{\frac{2\ln t}{n}}$$

(using the algebra rule $\ln(t^4) = 4\ln t$, then simplifying $4/2 = 2$).

**Step 3 — recognize this is exactly UCB1's bonus term.** Replace $n$ (generic sample size) with $N_i(t-1)$ (number of times arm $i$ has been pulled) — and you've arrived exactly at:

$$\text{bonus} = \sqrt{\frac{2\ln t}{N_i(t-1)}}$$

**That's the whole derivation.** If asked to "derive UCB1" live, this three-step structure (pick a shrinking failure probability tied to $t$ → plug into Hoeffding's padding formula → simplify) is exactly what's expected — you're not deriving Hoeffding's inequality itself from scratch (that's a deeper probability-theory result, out of scope to re-derive live), you're showing how UCB1's specific formula falls naturally out of Hoeffding's general formula plus one sensible choice ($\delta = 1/t^4$).

---

## 18.3 Exercise 2: ε-greedy from scratch

```python
import random

def epsilon_greedy(true_means, T, epsilon=0.1):
    K = len(true_means)
    counts = [0] * K          # N_i(t): how many times each arm pulled
    sums = [0.0] * K          # running sum of rewards per arm
    rewards_over_time = []

    for t in range(1, T + 1):
        if random.random() < epsilon:
            arm = random.randrange(K)               # explore: random arm
        else:
            # exploit: pick arm with highest sample mean so far
            avg = [sums[i] / counts[i] if counts[i] > 0 else 0.5 for i in range(K)]
            arm = avg.index(max(avg))

        # simulate pulling the arm: Bernoulli reward with the arm's true mean
        reward = 1 if random.random() < true_means[arm] else 0

        counts[arm] += 1
        sums[arm] += reward
        rewards_over_time.append(reward)

    return rewards_over_time
```

**Plain-English walkthrough of what this code does, line by line, in case the syntax is the tricky part**:
- `counts` and `sums` are just our familiar $N_i(t)$ and the running total of rewards per arm — everything needed to compute $\hat\mu_i(t)$.
- Each round, flip a weighted coin (`random.random() < epsilon`): if it lands in the "explore" bucket, pick a uniformly random arm.
- Otherwise, compute the current sample mean for every arm (`avg`), and pick whichever is highest — this is the greedy step.
- `0.5` is used as a placeholder guess for arms with zero pulls so far (never divide by zero) — a simple, common convention.
- Simulate the environment's response: a coin flip using that arm's *true* (secret, unknown-to-the-algorithm) mean.
- Update the bookkeeping, record the reward, move to the next round.

---

## 18.4 Exercise 3: UCB1 from scratch

```python
import math
import random

def ucb1(true_means, T):
    K = len(true_means)
    counts = [0] * K
    sums = [0.0] * K
    rewards_over_time = []

    # Step 1: pull every arm once, to initialize (avoids dividing by zero)
    for arm in range(K):
        reward = 1 if random.random() < true_means[arm] else 0
        counts[arm] += 1
        sums[arm] += reward
        rewards_over_time.append(reward)

    # Step 2: main loop, starting after initialization
    for t in range(K + 1, T + 1):
        ucb_scores = []
        for i in range(K):
            avg = sums[i] / counts[i]
            bonus = math.sqrt(2 * math.log(t) / counts[i])
            ucb_scores.append(avg + bonus)

        arm = ucb_scores.index(max(ucb_scores))
        reward = 1 if random.random() < true_means[arm] else 0
        counts[arm] += 1
        sums[arm] += reward
        rewards_over_time.append(reward)

    return rewards_over_time
```

**Plain-English walkthrough**: the initialization loop is just Chapter 4's "pull every arm once first" step, so `counts[i]` is never zero when we later divide by it. The main loop computes, for every arm, its sample mean plus the exploration bonus — exactly the formula from Section 18.2 — and picks whichever arm has the highest total score. Notice this is **deterministic** (no `random.random()` used to *choose* the arm — only used to simulate the environment's response) — exactly matching Chapter 4, Section 4.6's point that UCB1 makes no random decisions itself.

---

## 18.5 Exercise 4: Thompson Sampling from scratch

```python
import random

def thompson_sampling(true_means, T):
    K = len(true_means)
    alpha = [1] * K   # Beta(alpha, beta) parameters, starting at Beta(1,1) for every arm
    beta = [1] * K
    rewards_over_time = []

    for t in range(1, T + 1):
        # draw one random sample from each arm's current posterior
        samples = [random.betavariate(alpha[i], beta[i]) for i in range(K)]
        arm = samples.index(max(samples))

        reward = 1 if random.random() < true_means[arm] else 0
        rewards_over_time.append(reward)

        # update: increment alpha on success, beta on failure
        if reward == 1:
            alpha[arm] += 1
        else:
            beta[arm] += 1

    return rewards_over_time
```

**Plain-English walkthrough**: `alpha` and `beta` are exactly the two counters from Chapter 6, both starting at 1 (the flat, uninformed prior). `random.betavariate(alpha[i], beta[i])` does the actual random draw from each arm's current Beta posterior — this is Python's built-in Beta-distribution sampler, no need to implement Beta sampling by hand. Pick the arm with the highest drawn sample, observe the reward, and update exactly one counter (`alpha` on a success, `beta` on a failure) — notice how much shorter and simpler this is than the UCB1 code, with no logarithms or square roots at all, exactly echoing the practical-implementation-simplicity point from Chapter 6 (Section 6.8).

---

## 18.6 Exercise 5: hand-tracing regret growth for a 3-arm example

Let's use $\mu_1 = 0.30, \mu_2 = 0.50, \mu_3 = 0.45$ one more time (as throughout this whole course), and trace, in words, how cumulative regret would tend to look under each algorithm over, say, 1,000 rounds — this is the kind of comparison an interviewer might ask you to sketch or describe out loud.

**Pure greedy**: after the initial 3 pulls (one per arm, by convention), whichever arm happened to look best from that tiny, noisy sample gets locked onto forever. There's a real chance (maybe 1-in-3-ish, depending on luck) that this ends up being arm 1 or arm 3 instead of the true best (arm 2) — if so, cumulative regret grows in a straight line for the rest of the 1,000 rounds, at a rate of $0.05$ or $0.20$ per round (Chapter 2's $r_t = \mu^* - \mu_{A_t}$), ending somewhere in the hundreds by round 1,000. **Shape: a straight line, possibly a very steep one, depending on luck.**

**Constant-ε ε-greedy** ($\varepsilon=0.1$): correctly identifies arm 2 as best fairly early (never gets permanently stuck, unlike greedy), but keeps paying a small, constant regret rate forever from ongoing 10%-of-the-time exploration (recall the $\approx 0.0083$-per-round calculation from Chapter 3, Section 3.4) — cumulative regret still grows in a straight line, but a **much shallower** one than greedy's worst case. **Shape: a shallow straight line, from very early on.**

**UCB1**: cumulative regret grows quickly at first (lots of necessary early exploration across all 3 arms, especially between the close arms 2 and 3), then the growth rate visibly **slows down** as the confidence intervals narrow and UCB1 settles into consistently picking arm 2 — by round 1,000, the curve should look like it's flattening out, characteristic $O(\log T)$ shape (recall Chapter 2, Section 2.5's "1000× more traffic only doubles regret" intuition — the same flattening idea). **Shape: fast growth early, clearly bending flatter over time.**

**Thompson Sampling**: qualitatively similar overall shape to UCB1 (early exploration, later flattening, since both achieve $O(\log T)$-shaped regret) — but, per Chapter 7's empirical discussion (Section 7.5), the curve typically sits **below** UCB1's curve throughout — same overall bending shape, just consistently a bit lower.

**Sketch summary** (useful to be able to describe out loud even without literally drawing it): *"Greedy is a straight line that could be steep. Constant-ε ε-greedy is a shallower straight line. UCB1 curves up fast then bends flat. Thompson Sampling has that same bending shape as UCB1, just running a bit lower the whole way."*

---

## 18.7 Production considerations (kept simple)

- **These from-scratch implementations are genuinely close to what a real prototype/first-pass production implementation would look like** for a simple, non-contextual bandit — the core loop structure (maintain per-arm stats, score each arm, pick the max, update) barely changes as you move from a whiteboard sketch to real code, which is part of why bandits are popular in practice: the implementation complexity is genuinely low relative to the sophistication of the underlying idea.
- **In a real interview coding exercise, being able to write the UCB1 or Thompson Sampling loop cleanly, from memory, in under a few minutes** is a very achievable and very high-value skill to drill — these are short enough functions (15–20 lines) that fluency is realistic with a bit of practice.

---

## 18.8 Interview traps (kept simple)

- **Forgetting the initialization step in UCB1 code** (pulling every arm once first) — leads to a division-by-zero bug, and is exactly the kind of small, checkable detail interviewers notice.
- **Writing ε-greedy's exploration step to sample from *all* arms including ties incorrectly, or forgetting the placeholder value for never-pulled arms** — small implementation details that signal how carefully you've actually thought through the edge cases.
- **In the regret-shape sketch (Section 18.6), drawing (or describing) any curve that decreases** — remember from Chapter 2 that cumulative regret can never go down, only flatten.

---

## 18.9 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: can write correct, working code for at least ε-greedy and UCB1 given some time, and can roughly describe the regret-shape differences between algorithms.
- **L6 bar**:
  - Can walk through the Hoeffding-to-UCB1 derivation (Section 18.2) fluently and quickly, showing the formula isn't just memorized but understood as flowing from a specific, sensible choice ($\delta=1/t^4$).
  - Writes clean, correct code for all three algorithms with minimal hesitation, and proactively narrates the small implementation details (initialization, zero-division guards) rather than needing to be prompted to handle them.
  - Gives the full comparative regret-shape sketch (Section 18.6) fluently and unprompted, including *why* each shape looks the way it does (not just "it looks like this," but "it looks like this because...").

---

## 18.10 Comprehension checks — plain words, minimal formulas

1. Walk through, in your own words, the three-step derivation connecting Hoeffding's inequality to UCB1's bonus formula.
2. What's the one line of difference between the ε-greedy and UCB1 code, in terms of how the arm is chosen each round?
3. Why does the Thompson Sampling code not need a "pull every arm once first" initialization loop the way the UCB1 code does?
4. Sketch, in words, the four regret-growth shapes from Section 18.6, and explain why UCB1 and Thompson Sampling's curves bend flat while greedy's and ε-greedy's stay straight.
5. What specific bug would you introduce into the UCB1 code if you forgot the initialization step?

---

*Next: Chapter 19 — System Design Case Studies, where we run full dialogue-format mock interviews (with explicit L5-vs-L6 answer breakdowns) for designing explore-exploit systems for ad ranking, feed ranking, and App Store search — the same format used in your prior system design mock interviews.*
