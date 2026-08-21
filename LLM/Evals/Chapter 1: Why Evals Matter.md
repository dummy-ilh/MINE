## LLM Evals — 10 Chapter Plan (Google/Apple Interview Prep)

| # | Chapter | Type |
|---|---------|------|
| 1 | Why Evals Matter + Taxonomy (intrinsic/extrinsic, offline/online) | Foundation |
| 2 | Classic NLP Metrics (BLEU, ROUGE, perplexity) — and why they break for LLMs | Foundation |
| 3 | Human Evaluation (rubrics, pairwise, inter-annotator agreement) | Core |
| 4 | LLM-as-a-Judge (bias, calibration, position/self-preference bias) | Core |
| 5 | Benchmark Suites (MMLU, HellaSwag, GSM8K, HumanEval, contamination) | Core |
| 6 | Task-Specific Evals (RAG faithfulness, summarization, code gen, agents) | Core |
| 7 | Safety & Red-Teaming Evals (toxicity, bias, jailbreaks) | Core |
| 8 | Statistical Rigor (confidence intervals, significance testing, power) | Core |
| 9 | **Debugging & Production Evals** (fixed) — monitoring, drift, A/B testing, CI/CD | Fixed |
| 10 | **Interview Q&A** (fixed) — rapid-fire + system design style questions | Fixed |

Now let's teach Chapter 1.

---

# Chapter 1: Why Evals Matter + Taxonomy

## Start with the pain, not the definition

Imagine you fine-tune a model to write customer support replies. It scores great on your training loss. You ship it. Within a week, customers complain the bot is "confidently wrong" — it gives fluent, grammatically perfect answers that are just... false.

Your loss curve told you nothing about this. **Loss measures how well the model predicts the next token on held-out text. It does not measure whether the model is useful, safe, or truthful.** That gap — between "low loss" and "good in the real world" — is the entire reason evals exist as a field.

So the first mental model: **evals are a proxy measurement problem.** You can't directly measure "is this model good?" — that's vague and unmeasurable. Instead you build a stack of proxies, each closer or further from the real thing you care about, and you have to know the tradeoffs of each proxy.

## The taxonomy — two independent axes

Interviewers love asking "how would you evaluate X" — and the strongest answers place the eval on a **2x2 grid**. Two axes, independent of each other:

**Axis 1: Intrinsic vs. Extrinsic**
- **Intrinsic** = does the model do the *sub-task* well, in isolation. Example: perplexity on a text corpus. Doesn't ask "does this help a real user."
- **Extrinsic** = does the model help with the *actual downstream task/goal*. Example: does a summarization model, when used in a support tool, reduce ticket resolution time?

Think of intrinsic as "is the engine efficient" and extrinsic as "does the car get you to work on time."

**Axis 2: Offline vs. Online**
- **Offline** = evaluated on a fixed, static dataset before deployment. No live users involved. Cheap, fast, repeatable.
- **Online** = evaluated on live traffic, real users, real consequences. Slower, riskier, but it's ground truth.

Put them together and you get 4 quadrants:

| | Intrinsic | Extrinsic |
|---|---|---|
| **Offline** | Perplexity on held-out set | Static benchmark like MMLU, HumanEval |
| **Online** | Live log-likelihood monitoring | A/B test on user satisfaction / task completion |

**Worked example — grounding this in numbers:**

Say you have two candidate summarization models, A and B.

- Offline intrinsic: Model A gets ROUGE-L = 0.42, Model B gets 0.38. A "wins" on this axis.
- Offline extrinsic: You run both on a benchmark of 200 real support tickets with human-graded "is this summary usable" (1-5 scale). A averages 3.1, B averages 3.9. B wins.
- Online: You A/B test on 10,000 live tickets. Ticket resolution time drops 8% with B, unchanged with A.

Now — which model do you ship? **B.** The ROUGE-L numbers, which looked favorable for A, were an intrinsic proxy that didn't correlate with the extrinsic, real-world outcome. This is *the* recurring interview trap: high performance on an intrinsic offline metric ≠ real-world quality. Good interview answer: "I'd never rely on a single-quadrant metric to make a shipping decision — I'd want offline extrinsic signal at minimum, and ideally online confirmation before full rollout."

## Why this ordering matters practically

You move through the quadrants in order of increasing cost and decreasing risk of being wrong:

1. **Offline intrinsic** — cheapest, fastest, run on every commit. Catches regressions early (e.g., perplexity spike = something broke).
2. **Offline extrinsic** — more expensive (often needs human labels or a benchmark), but tells you if the model is actually useful for the task.
3. **Online** — most expensive and highest-stakes (real users), but the only source of true ground truth about business impact.

A mature ML org gates deployment through all three: nothing reaches online evaluation unless it clears offline intrinsic and extrinsic thresholds first. This funnel *is* the answer to "how do you build an eval pipeline" in a system design interview.

## Quick check — try this yourself

A model gets state-of-the-art perplexity on the Pile dataset but users in production say its answers "feel off-topic." Which quadrant caught the good news, and which quadrant is now catching the bad news?

Perplexity on Pile = offline intrinsic (good). "Feels off-topic in production" = online extrinsic (bad). Classic mismatch — the model is a good next-token predictor in general but isn't extrinsically aligned with what users in *this specific application* need.

---

Next chapter (2) covers classic NLP metrics — BLEU, ROUGE, perplexity — the actual math behind them, and precisely why they fail once you move from translation/summarization into open-ended LLM generation. Want me to continue?
