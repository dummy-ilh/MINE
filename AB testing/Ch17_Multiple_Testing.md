# Chapter 17: Multiple Testing

## 1. Intuition

Chapter 16 covered inflation of false-positive rate from looking at the *same* metric *multiple times over time*. This chapter covers the closely related but distinct problem: inflation of false-positive rate from looking at *many different metrics* (or many different segments, or many different variants) *at the same time*.

The core intuition: if you test 20 metrics simultaneously, each at $\alpha=0.05$, and none of them actually have a real effect, you'd still expect **about 1 of them to show "significant" results purely by chance** ($20 \times 0.05 = 1$ expected false positive). This is exactly why a common experimentation failure mode is: "we tested 15 metrics, one came back significant, let's call that our win" — that single significant metric is very plausibly noise, not signal, if it wasn't the pre-specified primary OEC (this directly connects back to why Chapter 9 insisted on committing to a single OEC before running the experiment).

## 2. The Formal Problem

If you run $m$ independent hypothesis tests, each at significance level $\alpha$, and the null hypothesis is true for all of them, the probability of **at least one false positive** across the whole family is:

$$P(\text{at least one false positive}) = 1-(1-\alpha)^m$$

For $\alpha=0.05$ and $m=20$: $1-(0.95)^{20} \approx 1-0.358 = 0.642$ — a **64% chance of at least one false positive** somewhere across those 20 tests, even though each individual test is nominally "safe" at 5%. This quantity, $P(\text{at least one false positive across the family})$, is called the **Family-Wise Error Rate (FWER)**.

## 3. Two Correction Frameworks

**Framework 1 — Controlling FWER (conservative, appropriate when even one false positive is costly)**:

- **Bonferroni correction**: simplest approach — divide your significance threshold by the number of tests: use $\alpha/m$ as your per-test threshold instead of $\alpha$. For $m=20$ tests at family-wise $\alpha=0.05$: each individual test needs $p<0.0025$ to be declared significant. This is simple but conservative — it can substantially reduce power, especially as $m$ grows large, since it treats each test as if it needed to independently guard against the whole family's error budget.
- **Holm-Bonferroni (step-down) correction**: a less conservative refinement — sort p-values from smallest to largest, and compare the $k$-th smallest p-value to $\alpha/(m-k+1)$ rather than a flat $\alpha/m$ for every test. This uniformly gives more power than plain Bonferroni while still controlling FWER, so it's generally preferred over plain Bonferroni whenever you specifically need FWER control.

**Framework 2 — Controlling False Discovery Rate (FDR) (less conservative, appropriate when you can tolerate SOME false positives among many discoveries, as long as the *proportion* of false positives among your "significant" results stays controlled)**:

- **Benjamini-Hochberg procedure**: sort p-values ascending as $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m}\alpha$. Declare all tests with $p \leq p_{(k)}$ as significant. This controls the **expected proportion of false positives among your declared discoveries** (FDR), rather than the probability of ANY false positive at all (FWER) — a fundamentally more lenient and often more appropriate standard when you're doing exploratory analysis across many metrics/segments and can tolerate a controlled fraction of false leads among many true findings.

**When to use which**: FWER control (Bonferroni/Holm) is appropriate for **confirmatory, high-stakes decisions** where even a single false positive is costly (e.g., "which single metric determines whether we ship this multi-million-dollar feature"). FDR control (Benjamini-Hochberg) is more appropriate for **exploratory analysis across many segments/metrics** where you expect and can tolerate some false leads, as long as most of your flagged findings are real (e.g., "which of these 50 user segments show a meaningfully different response to treatment, for follow-up investigation").

## 4. Worked Example

You ran an experiment and are looking at 10 secondary metrics beyond your pre-specified primary OEC (which itself showed no significant effect). The 10 secondary metric p-values, sorted ascending: 0.001, 0.008, 0.015, 0.021, 0.033, 0.041, 0.09, 0.15, 0.31, 0.52.

**Naive approach (no correction)**: at $\alpha=0.05$, you'd declare the first 6 metrics "significant" (all p ≤ 0.041). This is almost certainly a substantial overstatement given you're testing 10 metrics.

**Bonferroni correction**: threshold becomes $0.05/10 = 0.005$. Only the first metric (p=0.001) survives.

**Benjamini-Hochberg (FDR) correction**: for each $k$-th smallest p-value, compute $\frac{k}{10}\times0.05$:
- $k=1$: threshold $=0.005$; $p_{(1)}=0.001 \leq 0.005$ ✓
- $k=2$: threshold $=0.010$; $p_{(2)}=0.008 \leq 0.010$ ✓
- $k=3$: threshold $=0.015$; $p_{(3)}=0.015 \leq 0.015$ ✓
- $k=4$: threshold $=0.020$; $p_{(4)}=0.021 > 0.020$ ✗ (fails)
- $k=5$: threshold $=0.025$; $p_{(5)}=0.033 > 0.025$ ✗ (fails)

Find the **largest** $k$ where the p-value is still under its threshold — here that's $k=3$. So under Benjamini-Hochberg, the first **3** metrics are declared significant (p=0.001, 0.008, 0.015), even though $k=4$ and $k=5$ individually failed their own thresholds — the BH rule uses the largest passing $k$, not a simple "each one independently passes" rule.

**Comparison**: naive (uncorrected) → 6 "significant" findings; Bonferroni (FWER) → 1 finding; Benjamini-Hochberg (FDR) → 3 findings. This ordering (naive ≥ FDR ≥ FWER in number of declared discoveries) is generally true — FWER control is the strictest, FDR is a middle ground, and no correction is the most liberal (and least trustworthy).

## 5. Production Considerations

- **The best defense against multiple testing problems is Chapter 9's discipline**: pre-specify a single primary OEC before the experiment runs. If that's done properly, you mostly avoid the multiple-testing problem for your *ship decision* — it only becomes a live issue for secondary/exploratory metrics, which should be clearly labeled as exploratory/hypothesis-generating, not confirmatory, in your writeup.
- **Guardrail metrics (Chapter 10) are a related multiple-testing surface**: the more guardrails you add, the higher your chance some guardrail trips by pure chance, which is exactly why Chapter 10 flagged "guardrail proliferation" as a real risk — this chapter gives you the formal machinery (FWER/FDR correction) to actually manage that risk if you have many guardrails.
- **At Google-scale, thousands of experiments run concurrently across the company** — this raises an even higher-level multiple-testing question (how many "wins" across the whole company are actually noise), which is part of why standing infrastructure often includes company-wide meta-analysis of experiment win-rates to sanity-check whether the observed rate of "significant" launches is consistent with what you'd expect if the true underlying win rate were much lower.
- **Segment analysis (testing effect across many user segments) is a classic multiple-testing trap** — foreshadowing Chapter 18 (Simpson's Paradox & Segment Analysis): if you slice your data into 30 segments and look for "which segment did the treatment help," you're implicitly running 30 tests, and should apply FDR/FWER correction rather than reporting the single most impressive-looking segment as if it were pre-specified.

## 6. Interview Traps

- **Trap #1**: Not recognizing a multiple-testing scenario when it's described indirectly (e.g., "we looked at 15 different user segments and found this one segment showed a huge effect" is a multiple-testing red flag, even if the word "multiple testing" never appears in the prompt).
- **Trap #2**: Applying Bonferroni by default without knowing about Benjamini-Hochberg/FDR as a less conservative, often more appropriate alternative for exploratory analysis — many candidates only know one correction method.
- **Trap #3**: Confusing FWER (probability of ANY false positive) with FDR (expected proportion of false positives among declared discoveries) — these answer genuinely different questions and are appropriate in different contexts, not interchangeable "the same idea, two names."
- **Trap #4**: Applying correction to your single pre-specified primary OEC (unnecessary — if you truly only tested one pre-specified metric, there's no multiple-testing problem for that metric) while forgetting to apply it to the actual multiple-testing surface (secondary metrics, segments, guardrails).

## 7. L5-Differentiating Talking Points

- Correctly identifying a described segment-slicing or many-secondary-metrics scenario as a multiple-testing problem, unprompted, rather than needing the interviewer to say "multiple testing" explicitly, shows pattern recognition over rote knowledge.
- Being able to explain WHEN to prefer FWER control vs FDR control (confirmatory/high-stakes vs. exploratory/many-discoveries) rather than defaulting to one correction method universally, shows judgment about matching the statistical tool to the actual decision context.
- Connecting this chapter explicitly to Chapter 9 (OEC discipline as the primary defense) and forward to Chapter 18 (segment analysis) shows you see multiple testing as a thread running through several parts of the curriculum, not an isolated formula.
- Bringing up company-wide, meta-level multiple testing (thousands of concurrent experiments) shows systems-level thinking about experimentation infrastructure at Google's actual scale, beyond a single-experiment textbook framing.

## 8. Comprehension Check

1. If you run 15 independent tests at $\alpha=0.05$ each, and none have a true effect, what's the probability of at least one false positive? Compute it.
2. Explain the difference between what FWER controls and what FDR controls, and give a scenario where you'd prefer each.
3. Using the Benjamini-Hochberg procedure on p-values $[0.002, 0.009, 0.018, 0.024, 0.045]$ (5 tests, $\alpha=0.05$), which are declared significant? Show your work.
4. Why does pre-specifying a single primary OEC (Chapter 9) mostly protect you from the multiple-testing problem for your ship decision, even though you might still compute many secondary metrics?
5. A colleague slices experiment results by 25 different user segments and finds one segment with p=0.02, and wants to write it up as a key finding. What's your concern, and what would you recommend instead?

---
*Next: Chapter 18 — Simpson's Paradox & Segment Analysis*
