# Chapter 16: Peeking & Sequential Testing

## 1. Intuition

This chapter formalizes a problem that's been lurking implicitly since Chapter 5: **all the sample-size and p-value math you've learned assumes you decide on a sample size in advance, collect exactly that much data, and look at the result exactly once.** In practice, almost nobody does this — dashboards update continuously, and it's incredibly tempting to check your experiment's p-value every day and stop as soon as it crosses 0.05. This chapter explains precisely why that's dangerous, and what the legitimate alternatives are.

The core intuition: **a p-value that crosses below 0.05 at some point during a running experiment, purely by chance, is far more likely to happen than a p-value that's below 0.05 at one single, pre-specified endpoint.** If you keep looking and stop the moment you see significance, you're implicitly running many tests (one per peek) and cherry-picking the one that happened to hit your threshold — inflating your true Type I error rate far above the nominal $\alpha=0.05$ you think you're operating at.

## 2. The Formal Problem

Under the standard (fixed-horizon) hypothesis testing framework, $\alpha=0.05$ means: "if $H_0$ is true, there's a 5% chance my test statistic crosses the critical value **at the one pre-specified sample size I committed to.**" This guarantee says nothing about what happens if you check the test statistic at many different sample sizes along the way and are free to stop at the first one that looks significant.

**The math**: even under a true null (no real effect), a running p-value computed continuously behaves like a random walk that will, with high probability, dip below 0.05 *at some point* during a long enough experiment, purely by chance — this is a known result related to the **Law of the Iterated Logarithm** in probability theory. Practically: if you peek at your experiment 10 times over its duration and stop at the first significant result you see, your *true* false-positive rate can inflate to well above 20-30%, even though each individual peek nominally used $\alpha=0.05$.

## 3. Legitimate Solutions

**Solution 1 — Fixed-horizon testing with strict discipline**: pre-commit to a sample size (via the power analysis in Chapter 5), don't look at the p-value until you hit it, and only make your decision at that single endpoint. This is the "textbook correct" approach but is often impractical, since teams want visibility into how an experiment is trending before it fully completes.

**Solution 2 — Group sequential design (pre-planned interim analyses with adjusted thresholds)**: pre-specify a small, fixed number of "looks" (e.g., 3 interim analyses plus one final analysis), and use a correction (like the **O'Brien-Fleming spending function** or Pocock boundaries) that makes early looks require a much stricter significance threshold than the final look, so that your *cumulative* Type I error rate across all planned looks still equals your target $\alpha$. This is the clinical-trials-standard approach, and it requires deciding the number and timing of looks in advance.

**Solution 3 — Always-valid sequential testing (mSPRT / anytime-valid p-values)**: modern experimentation platforms (this is the approach most associated with tech-company-scale continuous experimentation, as opposed to clinical trials with a handful of pre-planned looks) use methods based on the **mixture Sequential Probability Ratio Test (mSPRT)** or related martingale-based confidence sequences. These construct a p-value (or confidence interval) that remains statistically valid **no matter when or how many times you look** — you can genuinely check the dashboard every hour and stop whenever you want, with the guaranteed Type I error rate holding regardless of your stopping behavior. This is the practical solution that lets modern experimentation platforms show "peek-safe" p-values on live dashboards.

**The key tradeoff**: always-valid sequential methods (Solution 3) are generally more conservative for a given sample size than a fixed-horizon test would be (i.e., they require somewhat more data to achieve the same power), because they're paying a statistical "insurance premium" for the flexibility of unlimited peeking. This tradeoff is worth stating explicitly — it's not a free lunch, it's a deliberate, quantifiable exchange of some efficiency for validity under continuous monitoring.

## 4. Worked Example (Illustrating the Danger Numerically)

Suppose you run a true A/A test (both arms identical, so the true effect is exactly zero) and check the p-value once per day for 20 days, stopping and declaring "significant" the first time $p<0.05$.

Simulation-based results commonly cited in the experimentation literature: the *effective* false-positive rate under this "peek-and-stop" behavior, even though each individual day's test nominally uses $\alpha=0.05$, can easily reach **20-30%+** depending on the exact peeking cadence and correlation structure between consecutive days' test statistics (consecutive days share most of the same underlying data, so the peeks aren't independent, but the correlation isn't strong enough to prevent substantial inflation).

**Contrast**: using a proper group sequential design with, say, 4 pre-planned looks and O'Brien-Fleming-adjusted thresholds (early looks need something like $p<0.00001$, later looks relax toward the final $p<0.045$ or so, with exact values depending on the specific spending function), the *cumulative* false-positive rate across all 4 looks is controlled to stay at the target 5% — you get some visibility into the trend without paying the full uncontrolled peeking penalty.

**Practical framing for an interview answer**: "If someone asks me to just glance at results daily and ship the moment we cross significance, I'd explain that this inflates our real false-positive rate well beyond 5% — probably into the 20-30% range depending on how often we check — and I'd propose either committing to a fixed sample size upfront, or using our platform's always-valid sequential testing method if we have one, specifically so we CAN look whenever we want without breaking our error guarantees."

## 5. Production Considerations

- **Know whether your company's experimentation platform supports always-valid inference** — if it does (many large-scale platforms do, precisely because business stakeholders demand the ability to check in on experiments continuously), you should default to trusting the platform's built-in significance indicator rather than manually recomputing a naive p-value from raw counts, which would reintroduce the peeking problem the platform was built to solve.
- **"Peeking" isn't inherently forbidden — uncontrolled peek-and-stop behavior is the problem.** Looking at trends for monitoring/sanity-checking purposes (e.g., checking for SRM, checking guardrail metrics haven't catastrophically broken something) is fine and encouraged; the danger is specifically using an uncorrected p-value from an interim look as grounds to stop early and declare a win.
- **Business urgency pressure is the main practical driver of peeking problems** — teams under pressure to ship quickly are the most likely to peek and stop at the first favorable-looking result, so having pre-committed, documented stopping rules (and ideally always-valid inference infrastructure) is as much an organizational/cultural safeguard as a statistical one.

## 6. Interview Traps

- **Trap #1**: Not recognizing that continuous dashboard-checking with a naive stop-at-significance rule inflates the true Type I error rate — this is one of the most commonly tested "gotcha" scenarios in A/B testing interviews, often framed as "a PM wants to stop the experiment early because it just crossed significance, what do you say?"
- **Trap #2**: Proposing "just don't look until the end" as the only solution, without knowing that group sequential designs and always-valid sequential testing exist as more practically realistic alternatives that still permit legitimate interim monitoring.
- **Trap #3**: Believing always-valid sequential methods are a completely free improvement with no cost — failing to mention the conservativeness/efficiency tradeoff versus a fixed-horizon test signals surface-level knowledge of the topic.
- **Trap #4**: Confusing "peeking for monitoring purposes" (fine) with "peeking as grounds for an uncontrolled early stop" (the actual problem) — conflating these leads to overly rigid, unhelpful advice like "never look at results until the experiment ends," which isn't realistic or necessary.

## 7. L5-Differentiating Talking Points

- Being able to explain, even informally, WHY continuous peeking inflates the false-positive rate (the "p-value behaves like a random walk that's likely to dip below threshold at some point" intuition) rather than just asserting "peeking is bad" as a rule you were told, shows genuine statistical understanding.
- Naming both the older clinical-trials solution (group sequential design, O'Brien-Fleming) and the modern tech-industry solution (always-valid/mSPRT-based sequential testing) shows you know this problem has a real, evolving methodological history, not just one canned answer.
- Proactively stating the efficiency tradeoff of always-valid methods (more conservative, needs more data for equivalent power) demonstrates precise, non-oversold understanding of the solution's actual cost.
- Distinguishing legitimate monitoring-peeking from illegitimate stop-and-ship-peeking, and being able to articulate this nuance to a hypothetical impatient stakeholder, is exactly the kind of communication skill L5 interviews are listening for beyond pure technical correctness.

## 8. Comprehension Check

1. Explain, in your own words, why checking a p-value daily and stopping at the first significant result inflates the true Type I error rate above the nominal $\alpha$.
2. What is a group sequential design, and how does it allow legitimate interim analyses without inflating the overall Type I error rate?
3. What is the key statistical tradeoff of always-valid (mSPRT-based) sequential testing compared to a fixed-horizon test?
4. Is looking at your experiment dashboard daily always a statistical problem? Explain the distinction between monitoring and peek-and-stop behavior.
5. A PM says "the p-value just crossed 0.05 on day 3 of a planned 14-day experiment, let's ship now." What do you tell them, and what would you need to know about your experimentation platform to give a fully informed answer?

---
*Next: Chapter 17 — Multiple Testing*
