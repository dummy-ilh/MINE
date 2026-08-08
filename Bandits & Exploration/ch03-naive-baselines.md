# Chapter 3 — Naive Baselines: Greedy, ε-Greedy, and Explore-Then-Commit

---

## 3.1 Why we study "bad" algorithms first

Every algorithm in Phase 1 (UCB, Thompson Sampling) exists to fix a specific, provable flaw in a simpler algorithm. If you don't understand *exactly* how and why the naive baselines fail — not just vaguely, but with a formal regret argument — then UCB and Thompson Sampling will feel like arbitrary formulas instead of *targeted fixes*. This chapter builds that foundation.

We'll use our running example throughout: $\mu_1 = 0.30, \mu_2 = 0.50, \mu_3 = 0.45$, so $\mu^* = 0.50$ (arm 2 is best).

---

## 3.2 Notation we'll need: sample means and pull counts

Before defining any algorithm, we need two pieces of bookkeeping that every bandit algorithm maintains:

- $N_i(t)$ = number of times arm $i$ has been pulled up through round $t$
- $\hat{\mu}_i(t)$ = the **sample mean** (empirical average reward) of arm $i$, based on the pulls so far:

$$\hat{\mu}_i(t) = \frac{1}{N_i(t)} \sum_{s : A_s = i, \, s \leq t} X_s$$

In plain English: "add up every reward you've ever seen from arm $i$, divide by how many times you've pulled it." This is just the running average — nothing fancy yet. Every algorithm in this course is fundamentally a rule for deciding which arm to pull next, as a function of these two quantities ($N_i(t)$ and $\hat{\mu}_i(t)$) for every arm.

**Important distinction to keep straight**: $\mu_i$ (no hat) is the *true, unknown* mean. $\hat{\mu}_i(t)$ (with a hat) is your *current estimate*, built from data, and it changes every time you pull arm $i$. Every algorithm only ever sees the hatted quantity — the whole game is making good decisions despite never knowing the true $\mu_i$.

---

## 3.3 The pure greedy algorithm

**Algorithm**: At each round $t$, pull whichever arm currently has the highest sample mean:

$$A_t = \arg\max_i \hat{\mu}_i(t-1)$$

(With some rule for the very first round(s), e.g., pull each arm once to initialize, since $\hat\mu_i$ is undefined with zero pulls.)

### Worked trace

Initialize by pulling each arm once (a common convention):

| Round | Pull | Observed $X_t$ | $\hat\mu_1$ | $\hat\mu_2$ | $\hat\mu_3$ |
|---|---|---|---|---|---|
| 1 | Arm 1 | 0 | 0.00 | — | — |
| 2 | Arm 2 | 0 | 0.00 | 0.00 | — |
| 3 | Arm 3 | 1 | 0.00 | 0.00 | 1.00 |

After this initialization, greedy looks at $\hat\mu_1 = 0.00, \hat\mu_2 = 0.00, \hat\mu_3 = 1.00$ and — because arm 3 currently has the highest sample mean — **locks onto arm 3 forever**, never pulling arms 1 or 2 again (until/unless a tie-break forces it, which we'll ignore for simplicity). Every future round becomes:

$$A_t = 3, \quad r_t = \mu^* - \mu_3 = 0.50 - 0.45 = 0.05$$

### The formal problem: linear regret

Because greedy never revisits arm 3's estimate is never corrected by pulling arms 1 or 2 again, and because there was nothing forcing it to try arm 2 (the true best arm) more than once, greedy has a **constant, nonzero probability of getting permanently stuck on a suboptimal arm** due to nothing more than an unlucky initial sample. Once stuck, it pays constant regret $r_t = \Delta > 0$ (where $\Delta = \mu^* - \mu_{\text{stuck arm}}$) **every single round, forever**:

$$R_T = \sum_{t=1}^{T} r_t \approx T \cdot \Delta = \Theta(T)$$

This is **linear regret** — exactly the bad shape we defined in Chapter 2, Section 2.5. The core flaw is structural, not a matter of bad luck in one run: greedy has **zero built-in mechanism to ever reconsider an arm once another arm looks better**, no matter how little data that decision was based on.

**This is the single most important lesson in the naive-baseline chapter**: *any algorithm that permanently stops trying an arm based on early, noisy evidence is vulnerable to linear regret.* Every algorithm from Chapter 4 onward is explicitly designed to never fully "close the door" on an arm — it just makes revisiting an already-mediocre-looking arm progressively less frequent, never zero, until there's enough evidence to be confident.

---

## 3.4 ε-greedy

**Algorithm**: at each round, flip a biased coin.
- With probability $1-\varepsilon$: **exploit** — pull $\arg\max_i \hat\mu_i(t-1)$ (the greedy choice)
- With probability $\varepsilon$: **explore** — pull an arm uniformly at random from all $K$ arms (including possibly the current best)

$\varepsilon \in (0, 1)$ is a fixed hyperparameter, e.g., $\varepsilon = 0.1$.

### Why this "fixes" the lock-on problem

Unlike pure greedy, ε-greedy will, with probability $\varepsilon / K$ on any given round, pull *any specific arm* — including ones that currently look bad. Over enough rounds, every arm keeps getting pulled infinitely often (since $\varepsilon$ never goes to zero), so $\hat\mu_i(t) \to \mu_i$ (the sample mean converges to the true mean) for every arm, for every $i$, as $t \to \infty$. This means ε-greedy **cannot get permanently stuck the way pure greedy can** — good.

### Worked example — why constant ε still gives linear regret

Here's the catch. Even after ε-greedy has correctly identified arm 2 as the best arm (say, after round 500, $\hat\mu_2$ is very accurately estimated and consistently the highest), it **still explores 10% of the time, forever**, because $\varepsilon = 0.1$ is a fixed constant that never shrinks.

On any exploration round (probability $\varepsilon = 0.1$), the algorithm picks uniformly among all 3 arms — so there's a $1/3$ chance it explores into arm 2 anyway (fine, no regret), but a $2/3$ chance it explores into arm 1 or arm 3 (regret incurred). So the *long-run rate* of regret per round, once the algorithm has converged on knowing arm 2 is best, is:

$$\mathbb{E}[r_t] \approx \varepsilon \cdot \left(\frac{1}{K}\sum_{i \neq i^*} (\mu^* - \mu_i)\right) = 0.1 \times \left(\frac{1}{3}\big[(0.50-0.30) + (0.50-0.45)\big]\right)$$

$$= 0.1 \times \left(\frac{0.20 + 0.05}{3}\right) = 0.1 \times 0.0833 = 0.00833$$

This is a small number per round — but it's **constant**, never shrinking, no matter how many rounds pass or how confident the algorithm has become. Summed over $T$ rounds:

$$R_T \approx 0.00833 \times T = \Theta(T)$$

**Still linear regret** — just with a much smaller constant than pure greedy's mistake rate. This is an important nuance: **constant-ε ε-greedy is a huge practical improvement over pure greedy (it doesn't get permanently stuck), but it is still asymptotically the wrong shape.** It keeps "spending" a fixed fraction of its traffic on exploration long after exploration has stopped being valuable — a wasted 10% forever, even at $T = 10{,}000{,}000$.

**Interview-critical takeaway**: this is exactly the gap that UCB and Thompson Sampling close. They both explore *adaptively* — a lot early on, less and less as confidence grows — rather than at a fixed rate forever.

---

## 3.5 ε-decay: shrinking ε over time

**Fix**: let $\varepsilon$ shrink as a function of $t$, most commonly:

$$\varepsilon_t = \min\left(1, \frac{c}{t}\right)$$

for some constant $c > 0$ — i.e., $\varepsilon_t \propto 1/t$.

### Why this recovers sublinear regret (intuition, not full proof)

With $\varepsilon_t \propto 1/t$, the *cumulative* amount of exploration up through round $T$ grows like $\sum_{t=1}^T \frac{c}{t} \approx c \ln T$ (this is the harmonic series, which grows logarithmically — hopefully a satisfying callback to why $\log T$ keeps appearing as the "good" answer throughout this course). Because total exploration only grows as $O(\log T)$ rather than $O(T)$ (constant-ε's linear growth), and because each unit of exploration only costs a bounded amount of regret, the resulting cumulative regret from exploration is also on the order of $O(\log T)$ — matching the Lai-Robbins lower bound's shape from Chapter 2!

**The catch, and why we still don't stop here**: getting $\varepsilon_t$-decay to actually achieve $O(\log T)$ regret in practice requires knowing (or correctly guessing) the constant $c$, and that constant needs to depend on the (unknown!) gaps $\Delta_i = \mu^* - \mu_i$ between arms — get $c$ wrong and you either explore too little (risk of permanent lock-on, like greedy) or too much (needlessly high constant factor). **UCB and Thompson Sampling (Chapters 4–7) achieve the same $O(\log T)$ shape without needing to hand-tune a decay schedule against unknown gaps** — this is their central practical advantage over ε-decay, and a very common interview question ("why not just use ε-decay?").

---

## 3.6 Explore-Then-Commit (ETC)

**Algorithm**: split the horizon into two phases.
1. **Explore phase** (rounds $1$ through $mK$): pull each of the $K$ arms exactly $m$ times, in round-robin, purely to gather data. ($m$ is a hyperparameter — the number of "exploration pulls per arm.")
2. **Commit phase** (rounds $mK+1$ through $T$): compute $\hat\mu_i$ for every arm using only the exploration-phase data, and pull $\arg\max_i \hat\mu_i$ for every remaining round, with **no further exploration**.

### Worked example

Say $m = 20$ (each arm gets pulled 20 times during exploration), so exploration lasts $20 \times 3 = 60$ rounds. Suppose after those 60 rounds, the sample means happen to be $\hat\mu_1 = 0.35, \hat\mu_2 = 0.55, \hat\mu_3 = 0.40$ (noisy, but in this case correctly ranking arm 2 as best). ETC now commits to arm 2 for all remaining $T - 60$ rounds, paying zero regret per round from that point on (since it locked onto the *actually* best arm this time).

### The tradeoff on $m$, and why ETC is regret-optimal only with the "right" $m$

- **$m$ too small**: the exploration phase is too short to reliably tell arms apart, so there's a meaningful chance the commit phase locks onto the *wrong* arm — and once committed, ETC (like greedy) **never explores again**, so a wrong commitment pays linear regret for the rest of the horizon, exactly like pure greedy.
- **$m$ too large**: exploration is reliable, but you've "wasted" many rounds round-robining through clearly-bad arms even after it became statistically obvious which arm was best — this wasted exploration is itself a source of regret (every pull of arm 1 or arm 3 during exploration costs $\Delta_i$ regret, even if you eventually commit correctly).

It's possible to derive an optimal $m^*$ that balances these two failure modes and gives ETC an $O(\log T)$-ish regret bound — but, critically, **that optimal $m^*$ depends on the unknown gaps $\Delta_i$ between arms**, exactly the same issue that plagued ε-decay above. In practice you don't know the gaps in advance (that's the whole problem!), so you can't actually compute the theoretically optimal $m$ — you can only guess, and a bad guess reverts ETC to one of the two failure modes above.

**This is the second and final piece of motivation for Phase 1**: both ε-decay and ETC *can* achieve good asymptotic regret, but only with hyperparameters that require knowledge you don't actually have. **UCB and Thompson Sampling are "the smart answer"** precisely because they use the *data itself*, adaptively, in real time, to decide how much to explore each arm — no pre-committed schedule, no hyperparameter that secretly depends on unknown gaps.

---

## 3.7 Side-by-side comparison table

| Algorithm | Ever stops exploring an arm entirely? | Regret shape | Core flaw |
|---|---|---|---|
| Pure greedy | Yes, permanently, after just 1 pull per arm | $\Theta(T)$ — linear | Can lock onto a suboptimal arm forever from one unlucky sample |
| ε-greedy (constant ε) | Never | $\Theta(T)$ — linear (smaller constant) | Keeps exploring at a fixed rate forever, even once confident |
| ε-decay ($\varepsilon_t \propto 1/t$) | Asymptotically, exploration rate → 0 | Can be $O(\log T)$ | Requires knowing/guessing gap-dependent constant $c$ |
| Explore-then-commit | Yes, permanently, after the fixed explore phase | Can be $O(\log T)$ with optimal $m$ | Optimal $m$ depends on unknown gaps; wrong $m$ reverts to linear-regret failure modes |

This table is worth memorizing in this exact shape — it's an extremely common whiteboard-comparison ask.

---

## 3.8 Production considerations

- **ε-greedy is still widely used in production** despite its linear-regret flaw, precisely *because* of its simplicity, predictability, and ease of reasoning about (a fixed, known fraction of traffic is "exploration traffic," which is easy to budget, monitor, and explain to stakeholders). The theoretical suboptimality is often an acceptable tradeoff against implementation and operational simplicity — a very real, very common industry decision, and worth mentioning if asked "would you ever use ε-greedy at Google?"
- **Explore-then-commit maps naturally onto a common industry pattern**: a fixed "test phase" (like a short, fixed-duration A/B test) followed by "rollout" (send 100% of traffic to the winner). This is literally how a great deal of real-world experimentation is run — which is exactly why this chapter is a strong bridge to your existing A/B testing curriculum: **a classic fixed-horizon A/B test *is* explore-then-commit**, and adaptive bandit algorithms are the answer to ETC's core weakness (committing early, with no going back, based on possibly-insufficient data).
- **ε-decay's practical difficulty** (needing to know the gaps in advance to set the decay constant) is precisely why real systems that want adaptive exploration reach for UCB or Thompson Sampling instead of hand-tuned decay schedules.

---

## 3.9 Interview traps

- **Saying "ε-greedy solves the exploration problem" without qualification.** It solves the *permanent lock-on* problem, but — as shown formally in Section 3.4 — with a constant ε it does **not** achieve sublinear regret. If asked "is ε-greedy a good algorithm," the correct nuanced answer references this exact distinction.
- **Confusing "explores forever" with "achieves good regret."** ε-greedy explores forever and still has bad (linear) regret. The fix isn't just "never stop exploring" — it's "explore at a *shrinking* rate, calibrated to actual uncertainty," which is what every subsequent algorithm does.
- **Describing ETC's optimal $m$ as something you'd compute directly in practice.** As shown in Section 3.6, the "optimal" $m$ requires the unknown gaps — a strong answer flags this circularity rather than presenting $m^*$ as straightforwardly computable.
- **Forgetting the initialization step for greedy/ε-greedy** (pulling each arm once before sample means are defined) — a small detail, but omitting it signals imprecision.

---

## 3.10 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describe all three baselines, correctly identify that pure greedy and constant-ε ε-greedy both produce linear regret, and give the basic reason why (permanent lock-on vs. fixed-rate wasted exploration).
- **L6 bar**:
  - Produces the actual small numerical regret-rate calculation like Section 3.4's $0.00833$-per-round example, unprompted, to make the "still linear, just a smaller constant" point concrete rather than asserted.
  - Explicitly names the **shared root cause** connecting ε-decay's and ETC's practical weakness — both require hyperparameters that secretly depend on the unknown gaps $\Delta_i$ — and uses this to motivate UCB/Thompson Sampling as the natural next step, without being prompted to make that connection.
  - Draws the ETC ↔ fixed-horizon-A/B-test parallel unprompted, linking this chapter to broader experimentation strategy (a strong signal of someone who's actually deployed these ideas, not just studied them).

---

## 3.11 Comprehension checks

1. Walk through, in your own words, exactly *why* pure greedy can produce linear regret from a single unlucky initial sample.
2. In the ε-greedy worked example (Section 3.4), why does the per-round regret rate stay constant even after the algorithm has correctly and confidently identified the best arm?
3. Why does ε-decay ($\varepsilon_t \propto 1/t$) recover an $O(\log T)$-shaped regret bound, at least in intuition? (Hint: think about the harmonic series.)
4. What is the practical difficulty with computing the "optimal" exploration-phase length $m$ for explore-then-commit?
5. Fill in the comparison table from Section 3.7 from memory — for each of the four algorithms, state whether it stops exploring, and its regret shape.

---

*Next: Chapter 4 — UCB1, the first algorithm in this course that achieves $O(\log T)$ regret without needing to know the arm gaps in advance — using a beautifully simple principle called "optimism in the face of uncertainty."*
