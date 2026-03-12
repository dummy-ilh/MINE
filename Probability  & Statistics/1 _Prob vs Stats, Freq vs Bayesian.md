# Probability vs. Statistics — And Frequentist vs. Bayesian



## Part 1: Probability vs. Statistics

### The One-Line Difference

| | You know... | You're finding... |
|---|---|---|
| **Probability** | The rules of the game | What will likely happen |
| **Statistics** | What happened | The rules of the game |

They are **mirror images** of each other.

---

### Probability: Working Forward

You have a fair coin. You know it's fair ($p = 0.5$). You flip it 100 times.

> *"What is the probability of getting 60 or more heads?"*

The process is known. You're computing what outcomes look like **coming out** of it.

$$P(\text{event}) = \; ?$$

---

### Statistics: Working Backward

You have a coin of unknown origin. You flip it 100 times and get 60 heads.

> *"Is this coin fair? What is the true probability of heads?"*

The outcome is known. You're inferring what process **produced** it.

$$\theta = \; ?$$

---

### A Simple Analogy

Imagine a bag with red and blue marbles.

- **Probability:** *"The bag has 30% red marbles. If I draw 10, how many reds do I expect?"*
- **Statistics:** *"I drew 10 marbles and got 7 red. How many red marbles are in the bag?"*

Same bag. Same marbles. Opposite direction of reasoning.

---

### Why This Matters

| | Probability | Statistics |
|---|---|---|
| What's known | The model & its parameters | The data |
| What's unknown | The outcome | The model / parameters |
| Reasoning style | Deductive ("given the rules, what happens?") | Inductive ("given what happened, what are the rules?") |
| Output | An exact probability | An estimate + some uncertainty |
| Real example | Insurance pricing | Clinical trial analysis |
| ML example | Computing $P(\text{class} \mid \text{input})$ at prediction time | Training a model (fitting parameters to data) |

---

### The Key Insight

> **Statistics is built on probability.** Every statistical method — hypothesis tests, confidence intervals, regression — uses probability under the hood. Probability is the engine. Statistics is what you build with it.

---

---

## Part 2: Frequentist vs. Bayesian

This is a debate about a deeper question:

> **What does the word "probability" actually mean?**

Two smart camps. Two honest answers. Neither is universally right.

---

### The Frequentist View: Probability = Long-Run Frequency

> *"Probability is what happens in the limit if you repeat the experiment forever."*

Flip a fair coin 10,000 times — heads comes up about 5,000 times. The probability of heads is 0.5 **because** that's the long-run frequency.

**What this means for parameters:**

The true parameter (say, the real probability of heads $p$) is a **fixed number** — it just happens to be unknown. It doesn't make sense to say "there's a 70% chance $p = 0.6$." The parameter isn't random. Either it equals 0.6 or it doesn't.

**What frequentists do:**

- Collect data
- Compute a **point estimate** (best single guess for $p$)
- Compute a **confidence interval** (a range that captures $p$ in 95% of repeated experiments)

---

### The Bayesian View: Probability = Degree of Belief

> *"Probability measures how confident you are, given what you currently know."*

Will it rain tomorrow? You can't repeat tomorrow. But you can still say "I think there's a 70% chance of rain" — that's a *belief*, updated by forecasts, clouds, season.

**What this means for parameters:**

Parameters are treated as **random variables** — not because they change, but because *you're uncertain* about them. You can assign probabilities to them.

**What Bayesians do:**

1. Start with a **prior** — your belief about $p$ before seeing data
2. Collect data
3. Update your belief using Bayes' theorem to get a **posterior** — your revised belief about $p$ after seeing data

$$\underbrace{P(\theta \mid \text{data})}_{\text{posterior}} \propto \underbrace{P(\text{data} \mid \theta)}_{\text{likelihood}} \times \underbrace{P(\theta)}_{\text{prior}}$$

In plain English: **new belief = what the data says × what you believed before**

---

### Side-by-Side: Coin Flip Example

You flip a coin 10 times and get **8 heads**.

**Frequentist:**
$$\hat{p} = \frac{8}{10} = 0.8$$
That's your estimate. Build a confidence interval around it. You do not say anything about the probability that $p$ takes any particular value — it's fixed, not random.

**Bayesian:**

Start with a prior that says "I think the coin is probably fair" — say, $p \sim \text{Beta}(2, 2)$ (a gentle prior centered at 0.5).

After observing 8 heads in 10 flips:
$$p \mid \text{data} \sim \text{Beta}(10, 4)$$

Now you have a full **distribution** over $p$. You can directly answer: "What is the probability that $p > 0.7$?" — just integrate the posterior.

---

### The Confidence Interval vs. Credible Interval Trap

This trips up almost everyone in interviews.

**Frequentist 95% Confidence Interval** — say, $[0.61, 0.99]$

What it means:
> "If I repeated this experiment many times and built an interval each time, 95% of those intervals would contain the true $p$."

What it does **NOT** mean:
> ~~"There is a 95% chance $p$ is in this interval."~~ ← This is wrong in the frequentist world. $p$ is fixed. It's either in the interval or it isn't.

**Bayesian 95% Credible Interval** — say, $[0.57, 0.95]$

What it means:
> "Given the data I observed, there is a 95% probability that $p$ lies in this interval."

This is the statement people *think* a confidence interval makes. Bayesian inference gives you that directly.

---

### The Full Comparison

| | Frequentist | Bayesian |
|---|---|---|
| **Probability is...** | Long-run frequency | Degree of belief |
| **Parameters are...** | Fixed unknowns | Random variables |
| **Prior beliefs** | Not allowed | Required (and explicit) |
| **What you get** | Point estimate + confidence interval | Full distribution over parameter |
| **Interval interpretation** | 95% of intervals contain true value | 95% probability parameter is in interval |
| **Subjectivity** | Objective — no assumptions beyond data | Prior introduces subjectivity (can be a feature) |
| **Strength** | No need to pick priors; widely accepted | Incorporates prior knowledge; handles small data well |
| **Weakness** | Confidence intervals are unintuitive; can't make probability claims about parameters | Choosing priors can be controversial |

---

### When Do They Agree?

With **large amounts of data**, the two approaches give nearly identical answers. The data overwhelms the prior, so your starting belief doesn't matter much. This is formalized as the **Bernstein–von Mises theorem**: with enough data, the posterior concentrates around the maximum likelihood estimate (MLE) regardless of the prior.

> **Practical takeaway:** With big data, use whichever is more convenient. With small data, Bayesian methods shine because the prior lets you incorporate domain knowledge.

---

### Where Each Dominates

| Domain | Why |
|---|---|
| **Frequentist**: clinical trials, regulatory stats, social science | Long tradition; results easier to audit and reproduce |
| **Bayesian**: ML, A/B testing, NLP, recommendation systems | Handles uncertainty naturally; updates efficiently as new data arrives |

**ML connection — this comes up in interviews:**

| ML technique | Hidden probabilistic interpretation |
|---|---|
| L2 regularization (weight decay) | Gaussian prior on weights |
| L1 regularization (LASSO) | Laplace prior on weights |
| Maximum Likelihood Estimation (MLE) | Frequentist parameter estimation |
| Maximum A Posteriori (MAP) | Bayesian estimation with a prior |
| Dropout | Approximate Bayesian inference |

Knowing these equivalences will make you stand out.

---

### The One-Paragraph Summary

**Frequentist:** The world has fixed truths. Probability is about what happens if you repeat experiments forever. Parameters are unknown constants — we estimate them, build intervals around them, and test hypotheses about them. We do not assign probabilities to parameters.

**Bayesian:** Probability is about your state of knowledge. You start with a belief (prior), update it with data, and end with a refined belief (posterior). Parameters can have probability distributions — because you're uncertain about them, not because they're random. This gives you richer, more interpretable outputs.

---

### Interview Cheat Sheet

**"What's the difference between frequentist and Bayesian statistics?"**

> Frequentists treat probability as long-run frequency and parameters as fixed unknowns. Bayesians treat probability as degree of belief and parameters as random variables with distributions. The practical difference shows up most clearly in how they interpret intervals: a frequentist confidence interval is about the procedure, not the parameter; a Bayesian credible interval is a direct probability statement about the parameter.

**"What is a confidence interval?"**

> If we repeated the experiment many times and built an interval each time, 95% of those intervals would contain the true parameter. It is *not* a statement that the parameter has a 95% chance of being in *this specific* interval.

**"What is a prior?"**

> Your belief about a parameter before seeing the data. It can be vague ("I have no idea") or informative ("based on past studies, I think $p$ is around 0.3"). The prior gets updated by the data to produce the posterior.

**"What is MAP vs MLE?"**

> MLE finds the parameter value that makes the observed data most likely — purely data-driven. MAP (Maximum A Posteriori) finds the parameter value that is most probable *given both the data and a prior* — it's MLE with a regularization term coming from your prior belief.

---

*Next: Random variables, expectation, and variance.*
