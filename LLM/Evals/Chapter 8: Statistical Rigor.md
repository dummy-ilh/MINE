# Chapter 8: Statistical Rigor

## The problem this chapter solves

Every chapter so far gave you a *point estimate*: 71% BLEU, κ=0.73, 65% win rate, 7% ASR. Here's the question interviewers love to spring on you: **"Model B scored 2 points higher than Model A on your eval set — do you ship B?"** Most candidates say "yes, B is better." The correct instinct is: **is that 2-point gap real, or is it just noise from which 500 examples happened to be in your eval set?** This chapter is entirely about answering that question rigorously.

## Confidence Intervals — "how much would this number wobble if I re-sampled my eval set?"

**Intuition first.** Your eval set is a *sample* from a much larger space of possible prompts/tasks. If you'd happened to draw a slightly different 500 examples, your accuracy number would come out slightly different. A confidence interval quantifies that wobble.

**The formula (for a proportion, like accuracy):**

$$CI = \hat{p} \pm z \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

where $\hat{p}$ = observed accuracy, $n$ = number of eval examples, $z$ = 1.96 for a 95% CI.

**Worked numerical example.** Model A scores 82% accuracy on a 500-example eval set. What's the 95% CI?

Step 1 — standard error:
$$SE = \sqrt{\frac{0.82 \times 0.18}{500}} = \sqrt{\frac{0.1476}{500}} = \sqrt{0.0002952} ≈ 0.01718$$

Step 2 — margin:
$$1.96 \times 0.01718 ≈ 0.0337$$

Step 3 — interval:
$$82\% \pm 3.37\% \rightarrow [78.6\%, 85.4\%]$$

**How to read it:** you're 95% confident the model's *true* accuracy (if you had infinite eval examples) lies between 78.6% and 85.4%. Now the payoff — go back to the ship/no-ship question:

**Worked comparison.** Model A: 82% on 500 examples → CI [78.6%, 85.4%]. Model B: 84% on 500 examples → let's compute its CI too:
$$SE_B = \sqrt{\frac{0.84 \times 0.16}{500}} ≈ 0.01642, \quad \text{margin} = 1.96 \times 0.01642 ≈ 0.0322$$
$$84\% \pm 3.22\% \rightarrow [80.8\%, 87.2\%]$$

**The two intervals — [78.6, 85.4] and [80.8, 87.2] — overlap substantially.** This is your interview-ready red flag: *"A 2-point gap with overlapping confidence intervals this wide isn't strong evidence B is actually better — I'd want either a much larger eval set to narrow these intervals, or a proper paired significance test before recommending we ship B over A."*

**Practical lesson on sample size:** notice the margin shrank only from about ±3.4% to ±3.2% even though accuracy moved from 82% to 84% — the margin is governed by $n$, not by which model you're looking at. To tighten a ±3.4% margin down to, say, ±1%, you'd need roughly $(3.4/1)^2 ≈ 11.6\times$ more examples (since margin shrinks with $\sqrt{n}$) — i.e., roughly 5,800 examples instead of 500. This square-root relationship is worth having memorized: **halving your margin of error requires 4x the sample size**, not 2x.

## Significance Testing — is the gap between two models real?

Confidence intervals on each model *separately* are a decent gut-check (as above), but the statistically correct tool for "is A different from B" is a **paired significance test**, because in most eval setups both models are run on the *exact same* examples — that pairing carries information a separate-CI comparison throws away.

**Paired vs. unpaired — why it matters, intuitively:** if example #47 is just a genuinely hard example, *both* models are likely to struggle on it together. An unpaired test ignores this correlation and treats the two accuracy numbers as if they came from independent random samples, which overstates the apparent noise and makes it harder to detect a real difference. A paired test looks at the *per-example* differences, which cancels out this shared difficulty and gives you a more sensitive test.

**McNemar's test** — the standard paired test for binary correct/incorrect outcomes on matched examples.

**Worked numerical example.** Run Models A and B on the same 500 examples. Build a 2x2 table of who got each example right:

| | B correct | B incorrect |
|---|---|---|
| **A correct** | 380 | 20 |
| **A incorrect** | 40 | 60 |

The overall accuracies: A = (380+20)/500 = 80%, B = (380+40)/500 = 84%. Looks like B wins by 4 points. But McNemar's test only cares about the **discordant pairs** — examples where they disagreed (the 20 and 40 cells) — because the 380 (both right) and 60 (both wrong) cells give no information about which model is better.

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where $b$=20 (A right, B wrong), $c$=40 (A wrong, B right). (The "-1" is a continuity correction, standard practice for this test.)

$$\chi^2 = \frac{(|20-40|-1)^2}{20+40} = \frac{(19)^2}{60} = \frac{361}{60} ≈ 6.02$$

Compare against the chi-square critical value for 1 degree of freedom at p<0.05, which is 3.84. Since 6.02 > 3.84, **this result is statistically significant** — you can say with reasonable confidence that B genuinely outperforms A on this task, not just by luck of the eval set draw.

**Contrast with the earlier example:** notice this McNemar setup detected significance from a 4-point gap, while the separate-CI approach on my first worked example (2-point gap, overlapping CIs) was inconclusive. This isn't a contradiction — different gap sizes, different test sensitivity — but it does illustrate why the paired test is the more standard, more powerful tool: it's specifically designed to detect real differences using the discordant-pair information that a naive side-by-side CI comparison discards.

## Bootstrap resampling — the general-purpose tool when you don't have a clean formula

Not every eval metric (BLEU, an LLM-judge win rate, a composite rubric score) has a tidy closed-form significance test like McNemar's. The **bootstrap** is the general-purpose answer: instead of deriving a formula, simulate the sampling variability directly.

**Intuition, step by step:**
1. You have your eval set of $n$ examples with per-example scores for the metric you care about.
2. Resample $n$ examples *with replacement* from your eval set (some examples get picked multiple times, some not at all) — this simulates "what if I'd drawn a slightly different eval set from the same underlying distribution."
3. Recompute your metric (e.g., mean score, or win-rate difference between A and B) on this resampled set.
4. Repeat steps 2-3 many times (typically 1,000-10,000 times) — you now have a distribution of the metric across resamples.
5. The spread of that distribution *is* your confidence interval — e.g., take the 2.5th and 97.5th percentile of the bootstrap distribution for a 95% CI.

**Worked mini-example logic (illustrative, not hand-computable by hand at scale, which is exactly why this is a "run code" method rather than a formula):** suppose you bootstrap the win-rate-difference between Model A and B 1,000 times, and 960 of those 1,000 resampled differences are positive (B wins). That's strong evidence B is genuinely better — equivalent to a roughly 96% one-sided confidence that B > A. If instead only 550 of 1,000 resamples favored B, that's close to a coin flip — not trustworthy evidence of a real difference.

**When to reach for bootstrap vs. McNemar in an interview answer:** *"For a simple binary correct/incorrect paired comparison, McNemar's test is the standard, exact choice. For anything more complex — a continuous score, an LLM-judge win-rate, a composite metric — I'd use bootstrap resampling since it doesn't require a closed-form test and works for basically any metric."*

## Sample Size & Power — "how many eval examples do I even need?"

**Intuition.** Power is the probability your test correctly detects a real effect, if one truly exists. If your eval set is too small, even a genuinely better model might fail to show statistical significance — not because it isn't better, but because your test doesn't have enough data to tell signal from noise. This is called being **underpowered**.

**The four things in tension (know these relationships, not necessarily the full derivation):**
- **Effect size** — how big is the true difference you're trying to detect? Smaller true differences need more samples to detect reliably.
- **Significance level (α)** — usually 0.05; your tolerance for false positives.
- **Power (1-β)** — usually targeted at 0.80 or 0.90; your tolerance for false negatives (missing a real effect).
- **Sample size (n)** — what you're often solving for.

**Worked rule-of-thumb example.** Suppose from prior data you expect Model B to beat Model A by about 2 percentage points, and baseline accuracy is around 80%. Using a standard two-proportion power calculation targeting 80% power at α=0.05, detecting a 2-point gap around an 80% baseline typically requires on the order of several thousand examples per model (often 3,000-5,000+, depending on the exact numbers) —**far more than the 500-example eval set** used in the confidence-interval example earlier in this chapter. This is precisely *why* that earlier 2-point gap showed overlapping CIs — 500 examples was simply underpowered to detect a difference that small.

**Interview-ready synthesis:** *"Before I even run an eval comparison, I'd think about what effect size actually matters for the business decision, then back-calculate the sample size needed to detect it at reasonable power — rather than running whatever eval set I happen to have and hoping the result is significant."*

## Bringing Chapters 1-8 together

Every chapter before this one gave you a way to *produce* a number (perplexity, ROUGE, kappa, judge win-rate, pass@k, ASR, precision@k). This chapter is the layer that tells you **whether that number, or a difference between two numbers, is trustworthy** — via confidence intervals (how much does it wobble), significance tests (is a gap real), and power analysis (do I even have enough data to know). This is exactly the kind of "eval isn't just running a script, it's running a script *and* knowing whether to trust the output" framing that separates senior candidates from junior ones in interviews.

## Quick check

Your team reports "Model B beats Model A by 1.5 points on our 300-example eval set, p<0.05 using a t-test on the raw scores (not paired)." What's the first thing you'd push back on?

Two things worth flagging: (1) with n=300 and a small 1.5-point gap, that's a fairly small sample for a small effect — worth sanity-checking the CI width before trusting it; and (2) more importantly, if both models were run on the *same* 300 examples, an **unpaired** t-test is the wrong test — it throws away the paired structure of the data (per-example correlation) and can produce a misleading p-value. You'd want to redo this as a paired test (paired t-test, McNemar's if the metric is binary correct/incorrect, or a bootstrap on the paired differences) before trusting the significance claim.

---

Chapter 9 (fixed) is Debugging & Production Evals — monitoring, drift detection, A/B testing infrastructure, and CI/CD for eval pipelines. Want me to continue?
