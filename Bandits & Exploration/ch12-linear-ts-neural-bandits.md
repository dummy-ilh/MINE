# Chapter 12 — Linear Thompson Sampling & Neural Bandits

*(Same slower, simpler style — plain language first, light on notation.)*

---

## 12.1 The one-sentence idea

Chapter 11 took UCB1 and gave it context (LinUCB). This chapter does the exact same upgrade, but to **Thompson Sampling** instead — call it **Linear Thompson Sampling**. Then, briefly, we look at what happens when a straight line genuinely isn't a good enough model, and you swap in a neural network instead ("Neural Bandits").

---

## 12.2 Quick recap: what made Thompson Sampling different (Chapter 6)

Remember the core Thompson Sampling idea: instead of computing an upper bound and picking the max (UCB's approach), you keep a **whole range of plausible values** for each arm (a posterior distribution), **randomly draw one guess** from that range for each arm, and pick whichever arm's random draw came out highest.

Linear Thompson Sampling does exactly this — the only change is that instead of "a range of plausible values for one number per arm" (like Beta(2,1) for a single click-rate), we now need "a range of plausible values for the *line* itself" (the weight $\theta$ from Chapter 11).

---

## 12.3 What "uncertainty about the line" looks like

Picture our comedy-arm example again: $\text{predicted reward} = \theta \times x$. Early on, with very little data, we're not just unsure about the *exact* value of $\theta$ — we might think it's anywhere from, say, $0.01$ to $0.03$, roughly equally plausible. As more data comes in, that plausible range narrows — maybe down to $0.017$–$0.019$ after a lot of observations.

This "plausible range for $\theta$" is exactly the Bayesian posterior idea from Chapter 6, just now it's a posterior over "which line is correct" instead of "which single click-rate is correct." The math for updating this range as new data comes in (again, not something you need to derive by hand) is a close cousin of the Normal-Normal update from Chapter 7 — it's often literally called **Bayesian linear regression**: ordinary line-fitting, but keeping track of a whole plausible range for the weights instead of collapsing to one single best-fit line immediately.

---

## 12.4 The algorithm, in plain steps

For each arm, at each round:

1. **Randomly draw one specific line** (one specific value of $\theta$) from that arm's current plausible range — just like Chapter 6 drew one random click-rate ($\theta_i$) from a Beta distribution, we're now drawing one random *line* from a range of plausible lines.
2. **Use that randomly-drawn line to make a prediction** for the current user's context $x$.
3. **Pick whichever arm's random-draw-based prediction is highest.**
4. **Observe the reward, and update that arm's plausible range of lines** to reflect the new data point (narrowing it slightly, and nudging the center toward what better fits the new observation).

Compare this, side by side, with LinUCB from Chapter 11: LinUCB computes "prediction + unfamiliarity bonus" and picks the max — a fixed, deterministic formula. Linear Thompson Sampling instead **randomly samples** a plausible line and uses that sampled line's prediction directly — no separate "bonus" term is ever explicitly computed; the randomness itself naturally does more exploring for arms/contexts we're less sure about, exactly the same relationship UCB1 and Thompson Sampling had back in Chapters 4–7, just carried over here.

---

## 12.5 A very simple worked example

Same comedy arm, same $x = 20$ (user watched 20 comedies last month). Say our current plausible range for $\theta$ is roughly $0.01$ to $0.03$ (we don't have a ton of data yet).

**Draw 1** (early in learning, wide range): we randomly draw $\theta = 0.028$ (a somewhat optimistic draw from our still-wide range). Predicted reward $= 0.028 \times 20 = 0.56$.

**Draw 2** (later, after more data has narrowed the range to roughly $0.017$–$0.019$): we randomly draw $\theta = 0.018$. Predicted reward $= 0.018 \times 20 = 0.36$.

Notice: early on, the random draws can swing quite a bit (0.56 in this case) because the plausible range is wide — this is exactly what generates useful, automatic exploration. Later, once the range has narrowed from more data, the random draws cluster tightly around the now-more-accurate estimate (like 0.36), and the algorithm settles into consistently good, low-variance decisions — precisely the same "wide range early → naturally more exploration; narrow range later → naturally more exploitation" behavior from Chapter 6's Beta-Bernoulli trace, just now happening to an entire line instead of a single number.

---

## 12.6 LinUCB vs. Linear Thompson Sampling — the same comparison as Chapter 7, one level up

Everything we said in Chapter 7 (Section 7.5) about UCB1 vs. Thompson Sampling carries over almost unchanged:

- Both aim for the same goal (good contextual decisions with principled exploration).
- LinUCB is **deterministic** — same history in, same decision out, always.
- Linear Thompson Sampling is **randomized** — same history in, different random draws can lead to different decisions.
- Empirically (as with the non-contextual case), Linear Thompson Sampling often performs very well in practice and is popular in real systems, for the same basic reasons as before: natural, automatic exploration that scales down smoothly as data accumulates, without needing a separately-computed bonus formula.

**Simple interview-ready summary sentence**: *"Linear Thompson Sampling is to LinUCB exactly what plain Thompson Sampling was to plain UCB1 — same underlying goal, same underlying uncertainty being tracked, just turned into a decision by random sampling instead of by taking an upper bound."*

---

## 12.7 When a straight line isn't good enough: Neural Bandits (kept high-level)

Everything in Chapters 11–12 so far assumes the true relationship between context and reward is roughly a **straight line** (or a simple weighted sum of features). Sometimes that's just not true — maybe the relationship is much more complicated (e.g., "this ad works great for users aged 25–35, but not for anyone younger or older" — that's a bump-shaped relationship, not a straight line, and no single straight-line model can capture it well).

**Neural Bandits** (the names **NeuralUCB** and **NeuralTS** are the two you'll most likely hear) swap out the straight-line model for a **neural network** — which can learn much more complicated, bumpy, non-straight-line relationships between context and reward. The core *idea* stays exactly the same as everything above:

- NeuralUCB: neural network's prediction + an uncertainty-based bonus (the LinUCB idea, just with a fancier underlying model)
- NeuralTS: randomly sample from a plausible range of "networks," rather than one exact network, and use the sampled network's prediction (the Linear Thompson Sampling idea, just with a fancier underlying model)

**What you need for an interview**: you do not need to know how to implement NeuralUCB/NeuralTS from scratch. You need to be able to say, plainly: *"When the true relationship between context and reward is likely too complex for a straight line, you can swap the underlying model for a neural network while keeping the same UCB-style or Thompson-Sampling-style decision rule on top — this trades away some simplicity and speed for the ability to capture more complicated patterns, and it's the natural next step when LinUCB/Linear TS underperform because your relationships genuinely aren't linear."*

---

## 12.8 Production considerations (kept simple)

- **Most production systems start with LinUCB or Linear Thompson Sampling, not neural bandits**, because simple linear models train fast, need less data to become reliable, and are much easier to debug and monitor — neural bandits are usually reached for only once there's clear evidence the true relationships are too complex for a line to capture well, and there's enough data/traffic to support training a neural model reliably.
- **Randomized exploration (Linear TS) is often preferred in practice for the same reasons as plain Thompson Sampling** (Chapter 7): naturally tight, well-calibrated exploration without needing to separately engineer a bonus formula, and it tends to perform very well empirically.
- **The step from "linear" to "neural" is really a step from "simple, fast, interpretable" to "flexible, but slower and needs more data"** — this simple-vs-flexible tradeoff is one of the most universal themes in all of applied ML, and framing neural bandits this way (rather than as some exotic separate topic) is a strong, grounded interview answer.

---

## 12.9 Interview traps (kept simple)

- **Thinking Linear Thompson Sampling and LinUCB solve different problems.** They solve the *same* problem (contextual bandit with a linear reward model) with two different decision-making styles (randomized sampling vs. deterministic upper bound) — mixing this up suggests the Chapter 6/7 UCB-vs-TS relationship wasn't fully internalized.
- **Describing neural bandits as a completely different algorithm family**, rather than "the same UCB/TS decision-making idea, with a more flexible underlying prediction model swapped in." The *decision rule* doesn't fundamentally change — only the model doing the predicting does.
- **Reaching for neural bandits as a default answer** without being able to justify *why* a linear model wouldn't be good enough for the scenario at hand — this reads as trend-following rather than grounded reasoning.

---

## 12.10 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: correctly explain that Linear Thompson Sampling is Thompson Sampling's idea applied to a linear model, and can describe, at a high level, what neural bandits are for.
- **L6 bar**:
  - Produces the one-sentence summary from Section 12.6 unprompted when comparing LinUCB and Linear TS.
  - Frames neural bandits using the simple-vs-flexible tradeoff (Section 12.8) rather than presenting them as a disconnected, advanced topic — showing they see the whole Chapters 4–12 arc as one continuous idea (uncertainty-aware exploration), repeatedly re-applied to increasingly flexible underlying models.
  - Can give a grounded example of a real scenario where a linear model would likely fail (like the "works only for ages 25–35" bump-shaped example) to justify when neural bandits would actually be worth their added cost.

---

## 12.11 Comprehension checks — plain words, minimal formulas

1. In one sentence, what's the relationship between Linear Thompson Sampling and LinUCB?
2. In the worked example (Section 12.5), why did the random draws for $\theta$ swing much more widely early on than later?
3. What does NeuralUCB/NeuralTS change compared to LinUCB/Linear TS, and what stays exactly the same?
4. Give one concrete example (it doesn't have to be the textbook one) of a context-reward relationship that a straight line would likely fail to capture well.
5. Why do most real systems start with linear models (LinUCB/Linear TS) rather than jumping straight to neural bandits?

---

*Next: Chapter 13 — Non-Stationary & Structured Bandits, where we cover what happens when the "best answer" drifts over time, when you need to pick a whole *set* of items instead of just one, and when you only get to compare two options against each other instead of getting a direct reward.*
