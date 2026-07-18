# Chapter 15: Multiple Testing Correction (Bonferroni, FDR) Across Many Metrics

## 1. Definition

Multiple testing correction adjusts significance thresholds when you're testing many hypotheses (metrics) simultaneously, to control the inflated risk of false positives that arises purely from running many tests at once.

- **Family-Wise Error Rate (FWER):** the probability of making *at least one* Type I error (false positive) across the entire set of tests. **Bonferroni correction** controls FWER by dividing α by the number of tests (m): α_adjusted = α/m.
- **False Discovery Rate (FDR):** the expected *proportion* of false positives among all tests declared significant (rather than the probability of any false positive at all). The **Benjamini-Hochberg (BH) procedure** controls FDR, and is generally less conservative than Bonferroni.

## 2. Layman Explanation

If you flip a fair coin once and get heads, that's unremarkable. But if you flip 20 fair coins simultaneously, getting at least one "unusual" run (like 5 heads in a row on some coin) becomes almost expected, purely by chance — not because any coin is rigged.

This is exactly the multiple testing problem: if you check 20 metrics in an experiment at the standard 5% significance threshold, you'd expect roughly 1 of them to look "significant" purely by chance even if the treatment does absolutely nothing. If you then report that one "significant" metric as a real finding without acknowledging you checked 19 others that didn't pan out, you're fooling yourself (and everyone else) about how much evidence you actually have.

**Bonferroni** is the strict, conservative fix: "if I'm testing 20 things, I'll only call something significant if it clears a much higher bar (α/20 instead of α) — because I want to be very sure I don't cry wolf on ANY of them."

**FDR (Benjamini-Hochberg)** is a more practical, less strict fix: "I accept that among everything I call significant, some small percentage might be false alarms — but I want to control that percentage (say, keep it under 5%), rather than guaranteeing zero false alarms across the board." This is more forgiving when you're testing many metrics and can tolerate a controlled rate of mistakes among your "wins," rather than needing near-certainty on every single one.

## 3. Formal Explanation

**Bonferroni correction:**

α_adjusted = α / m

where m is the number of hypotheses (metrics) tested. A metric's p-value must fall below this stricter threshold to be declared significant. This guarantees P(at least one false positive across all m tests) ≤ α — a strong, conservative guarantee.

**Downside:** Bonferroni becomes very conservative (low power) as m grows large — testing 50 metrics means each individual test needs p < 0.001 to be declared significant, which can cause you to miss real effects (increased Type II error rate) especially for weaker, but genuine, signals.

**Benjamini-Hochberg (BH) procedure for FDR control:**

1. Rank the m p-values from smallest to largest: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
2. Find the largest k such that p₍ₖ₎ ≤ (k/m) × α
3. Declare all hypotheses with rank ≤ k as significant

This adapts the threshold based on rank — later-ranked (larger) p-values need to clear a progressively stricter bar, but the very smallest p-values get a much more lenient threshold than Bonferroni would give them.

**Why BH has more power than Bonferroni:**
BH controls the *expected proportion* of false discoveries among your significant results, not the probability of any false positive at all — a fundamentally more lenient (but still principled) standard, which translates to higher power to detect true effects, especially when testing many metrics where some are truly non-null.

**When to use which:**
- Bonferroni (FWER control): appropriate when even a single false positive is very costly — e.g., safety-critical guardrail metrics where shipping on a false "no harm" signal could be damaging.
- BH/FDR: appropriate for exploratory analysis across many secondary metrics, where you're comfortable with a controlled rate of false discoveries in exchange for better power to catch real effects.

## 4. Levers — What Controls It, What Moves It

**Number of tests (m)**
- More metrics tested simultaneously → stricter Bonferroni threshold (α/m shrinks) → lower power per test. This is why teams should limit the number of *primary*, decision-driving metrics (see Chapter 8 — pre-registration) rather than testing dozens simultaneously and correcting after the fact.

**Choice of correction method**
- Bonferroni is simple but overly conservative for large m; BH/FDR retains more power at the cost of a looser, probabilistic (rather than absolute) guarantee against false positives.

**Independence vs. correlation among tests**
- Both Bonferroni and the standard BH procedure are valid under independence; when metrics are highly correlated (common in product analytics — e.g., many engagement metrics move together), Bonferroni remains valid but overly conservative, while a modified BH procedure (BH under positive dependence) can be used to retain more power appropriately.

**Confirmatory vs. exploratory framing**
- Metrics pre-registered as primary (Chapter 8) generally shouldn't need correction if there's only one — correction is specifically needed when multiple metrics could each independently trigger a "significant" conclusion. Clearly separating primary (confirmatory, no correction needed if truly singular) from secondary/exploratory (correction applied, or explicitly flagged as hypothesis-generating only) avoids over-correcting where it isn't needed.

## 5. Worked Example

Suppose you test 10 metrics in an experiment, with the following p-values (already sorted):

| Rank (k) | Metric | p-value |
|---|---|---|
| 1 | Metric A | 0.002 |
| 2 | Metric B | 0.008 |
| 3 | Metric C | 0.015 |
| 4 | Metric D | 0.021 |
| 5 | Metric E | 0.033 |
| 6 | Metric F | 0.041 |
| 7 | Metric G | 0.052 |
| 8 | Metric H | 0.09 |
| 9 | Metric I | 0.15 |
| 10 | Metric J | 0.44 |

**Bonferroni at α=0.05:** α_adjusted = 0.05/10 = 0.005. Only Metric A (p=0.002) clears this bar → only 1 metric declared significant.

**Benjamini-Hochberg at α=0.05:** Compute (k/m)×0.05 for each rank:
- k=1: 0.005 → 0.002 ≤ 0.005 ✓
- k=2: 0.010 → 0.008 ≤ 0.010 ✓
- k=3: 0.015 → 0.015 ≤ 0.015 ✓ (exactly at threshold)
- k=4: 0.020 → 0.021 > 0.020 ✗
- k=5: 0.025 → 0.033 > 0.025 ✗

Find the largest k where the condition holds: k=3 (Metric C) satisfies p₍₃₎ ≤ 0.015. Even though k=4 and k=5 fail individually, the rule is "find the LARGEST k where the condition holds" — checking further ranks confirms k=3 is the largest satisfying rank here (k=4 onward all fail). So BH declares Metrics A, B, and C significant (ranks 1-3).

**Comparison:** Bonferroni found 1 significant metric; BH found 3. This illustrates BH's higher power — it's willing to accept a small, controlled rate of false discoveries among 3 "wins" rather than Bonferroni's much stricter standard that only lets through the single strongest result.

## 6. Famous Q&A (Google / Apple style)

**Q: Your experiment tracks 25 metrics, and one shows p=0.03. Should you declare it significant at the standard α=0.05 threshold?**
A: Not without correction — if you're evaluating 25 metrics at α=0.05 each, you'd expect over 1 false positive by chance alone even with zero true effects (25 × 0.05 = 1.25 expected false positives), so a single p=0.03 result isn't strong evidence on its own. I'd apply a multiple testing correction — Bonferroni if this metric wasn't pre-registered as the single primary metric and you need a strict guarantee against any false positive, or Benjamini-Hochberg if you're comfortable with a controlled false discovery rate across several exploratory metrics. Under Bonferroni with m=25, the adjusted threshold would be 0.002 — p=0.03 wouldn't clear that bar.

**Q: Why might a company choose FDR control (Benjamini-Hochberg) over Bonferroni for a set of 50 secondary/exploratory metrics?**
A: Bonferroni becomes extremely conservative as the number of tests grows — with m=50, the adjusted threshold would be α/50 = 0.001, which could cause you to miss many real, meaningful effects (very low power) just to guarantee zero false positives across the board. FDR control is more appropriate for exploratory analysis where the goal is generating leads for further investigation rather than making a single, high-stakes ship decision — accepting that a controlled proportion (e.g., 5%) of your flagged "significant" metrics might be false positives is a reasonable tradeoff for much better power to catch real signals among many candidates.

**Q: A guardrail metric (e.g., crash rate) is one of 15 metrics tested in an experiment. Should it be corrected the same way as the other 14 exploratory metrics?**
A: I'd treat it differently — guardrail metrics like crash rate are typically high-stakes, "must not regress" checks where even one false negative (missing real harm) is much more costly than a false positive (unnecessarily blocking a launch). I'd apply a stricter, FWER-style control (or even evaluate it independently, outside the multiple-comparisons pool, at its own pre-committed threshold) rather than lumping it in with 14 exploratory engagement metrics under a shared FDR correction — mixing high-stakes safety checks with exploratory metrics under one correction scheme can dilute the sensitivity needed to catch genuine harm.

**Q: Your team ran an experiment, found no significant result on the primary metric, but a secondary metric was significant at p=0.04 out of 8 secondary metrics tested. A PM wants to ship based on this secondary win. What do you say?**
A: I'd first check whether this passes a reasonable multiple-testing correction — with 8 secondary metrics, Bonferroni would require p < 0.00625, which p=0.04 doesn't clear, and even under the more lenient BH procedure, a single p=0.04 among 8 tests is unlikely to survive correction unless several other secondary metrics also show strong signal. Beyond the statistical correction, I'd also point out that shipping on a secondary metric when the pre-registered primary metric showed no effect is exactly the kind of post-hoc goalpost-moving that pre-registration (Chapter 8) is meant to prevent — I'd recommend treating this secondary result as hypothesis-generating for a future, properly powered follow-up test rather than a basis for shipping now.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Testing many metrics at the standard α=0.05 without any correction, then reporting whichever one happened to be significant
- ❌ Applying Bonferroni to a huge number of exploratory metrics and concluding "nothing is significant" without considering FDR as a less conservative, still-principled alternative
- ❌ Lumping high-stakes guardrail metrics into the same correction pool as low-stakes exploratory metrics
- ❌ Forgetting that a single, truly pre-registered primary metric doesn't need multiple-testing correction — correction is about the number of *simultaneously tested* hypotheses, not every metric that exists in a dashboard
- ✅ Do: separate confirmatory (primary, pre-registered) from exploratory metrics before deciding whether/how to correct
- ✅ Do: use Bonferroni when a strict, zero-tolerance guarantee is needed; use BH/FDR when better power across many exploratory checks is more valuable

---
*Next: Chapter 16 — Sequential Testing & the Peeking Problem (always-valid p-values).*
