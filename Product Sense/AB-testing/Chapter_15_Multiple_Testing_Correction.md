# Chapter 15: Multiple Testing Correction (Bonferroni, FDR) Across Many Metrics
# Chapter 15 — Multiple Testing Correction (Bonferroni, FDR)
> **Interview tier:** Google / Meta / Apple DS / PM rounds  
> **Core risk this prevents:** false discovery inflation from testing many metrics simultaneously  
> **Prerequisite:** Chapter 8 (pre-registration) — understand why you separate primary from secondary metrics first

---

## 1. The One-Line Mental Model

> Testing 20 metrics at α = 0.05 each means you **expect 1 false positive by chance alone** — even if your treatment does absolutely nothing.

Multiple testing correction re-sets the bar so that the *family* of tests collectively maintains the error guarantee you actually care about.

---

## 2. The Coin Flip Intuition

```
  ONE COIN                          20 COINS SIMULTANEOUSLY
  ────────                          ──────────────────────────
  Flip once.                        Flip all 20 at once.
  Get heads.                        At least one gives 5+ heads
  Unusual? Maybe.                   in a row. Surprising?

  P(unusual result) = small         P(at least one unusual result
                                     somewhere) = much larger

  This is fine — one test,          THIS is the multiple testing
  one chance to err.                problem. You have 20 chances
                                    to get lucky.
```

The same logic applies to p-values. Each individual test has a 5% false positive rate. Run 20 tests and the *family-wide* chance of at least one false positive explodes — up to 64% if all tests are independent.

```
  P(≥1 false positive | m independent tests, all null) = 1 - (1 - α)^m

  m=1   → 5%
  m=5   → 23%
  m=10  → 40%
  m=20  → 64%
  m=50  → 92%
```

---

## 3. Two Error Rates — Know the Difference

```
┌─────────────────────────────────────────────────────────────────────┐
│               FWER vs. FDR — side by side                          │
│                                                                     │
│  FAMILY-WISE ERROR RATE (FWER)    FALSE DISCOVERY RATE (FDR)       │
│  ──────────────────────────────   ──────────────────────────────    │
│  "What is the probability of      "Of all the results I call        │
│   making AT LEAST ONE false        significant, what fraction        │
│   positive anywhere?"              are actually false positives?"    │
│                                                                     │
│  Answer: probability               Answer: expected proportion       │
│  Binary: did any false             Rate: tolerate some false         │
│   positive occur? (yes/no)          positives, control the rate      │
│                                                                     │
│  Appropriate when:                 Appropriate when:                 │
│  • Any false positive is           • You're doing exploratory        │
│    costly (safety, legal)            analysis across many metrics    │
│  • High-stakes guardrail           • False discoveries are           │
│    checks                            recoverable (future tests)      │
│  • Binary ship/no-ship             • Better power matters more       │
│    decision on each metric           than zero-false-positive        │
│                                      guarantee                       │
│  Controlled by: Bonferroni         Controlled by: Benjamini-         │
│                                     Hochberg (BH)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Bonferroni Correction

### The formula

```
  α_adjusted = α / m

  Where:
    α = your original significance threshold (usually 0.05)
    m = number of hypotheses (metrics) tested simultaneously

  To declare a metric significant: p-value < α/m
```

### Visual intuition

```
  UNCORRECTED (m=10, α=0.05)        BONFERRONI (m=10, α=0.05)
  ───────────────────────────        ───────────────────────────
  Each test has its own              All tests share a single
  5% false positive budget.          "family" budget of 5%.
                                     Each test gets 5%/10 = 0.5%

  ←────── 5% ──────→                 ←── 0.5% ──→
  ████████████████████               ████
  0                0.05              0   0.005    0.05

  Easy to clear.                     Much harder to clear.
  High false positive rate.          Conservative but safe.
```

### Power trade-off

```
┌──────────────────────────────────────────────────────────────────┐
│  THE BONFERRONI POWER PROBLEM                                    │
│                                                                  │
│  m = 50 metrics → α_adj = 0.05/50 = 0.001                       │
│                                                                  │
│  A real, meaningful effect with p = 0.008 would be MISSED       │
│  even though it's a genuine signal.                              │
│                                                                  │
│  This is why Bonferroni is too blunt for large exploratory       │
│  metric sets — you'll miss real effects at the cost of           │
│  maintaining a strict zero-false-positive guarantee.             │
│                                                                  │
│  Rule of thumb:                                                  │
│  Bonferroni = right for HIGH-STAKES metrics (few of them)        │
│  BH/FDR    = right for EXPLORATORY metrics (many of them)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Benjamini-Hochberg (BH) Procedure

### The algorithm — step by step

```
  SETUP: m tests, significance level α (e.g. 0.05)

  STEP 1: Rank all p-values smallest to largest
          p(1) ≤ p(2) ≤ p(3) ≤ ... ≤ p(m)

  STEP 2: For each rank k, compute the BH threshold:
          BH_threshold(k) = (k / m) × α

  STEP 3: Find the LARGEST k where p(k) ≤ BH_threshold(k)
          Call this k*

  STEP 4: Declare ALL hypotheses with rank 1 through k* significant
          (even if some intermediate ranks failed the check)
```

### Why BH is smarter than Bonferroni

```
  BONFERRONI                        BH
  ──────────                        ──
  Same threshold for every test     Adaptive threshold — smaller
  α/m regardless of rank            p-values get a lenient bar,
                                    larger ones get a stricter bar

  Treats all tests equally          Rewards strong evidence
  conservative for all              while still controlling
                                    the false discovery rate
```

---

## 6. Worked Example — Bonferroni vs. BH

10 metrics, α = 0.05. P-values sorted:

```
┌──────┬──────────┬─────────┬──────────────┬──────────────┬──────────┬──────────┐
│ Rank │  Metric  │ p-value │  Bonferroni  │ BH threshold │  BH pass?│  Result  │
│  (k) │          │         │ (α/m=0.005)  │ (k/m × 0.05) │          │          │
├──────┼──────────┼─────────┼──────────────┼──────────────┼──────────┼──────────┤
│   1  │    A     │  0.002  │  0.005  ✓    │    0.005     │    ✓     │   SIG    │
│   2  │    B     │  0.008  │  0.005  ✗    │    0.010     │    ✓     │   SIG    │
│   3  │    C     │  0.015  │  0.005  ✗    │    0.015     │    ✓     │   SIG    │
│   4  │    D     │  0.021  │  0.005  ✗    │    0.020     │    ✗     │   ---    │
│   5  │    E     │  0.033  │  0.005  ✗    │    0.025     │    ✗     │   ---    │
│   6  │    F     │  0.041  │  0.005  ✗    │    0.030     │    ✗     │   ---    │
│   7  │    G     │  0.052  │  0.005  ✗    │    0.035     │    ✗     │   ---    │
│   8  │    H     │  0.090  │  0.005  ✗    │    0.040     │    ✗     │   ---    │
│   9  │    I     │  0.150  │  0.005  ✗    │    0.045     │    ✗     │   ---    │
│  10  │    J     │  0.440  │  0.005  ✗    │    0.050     │    ✗     │   ---    │
└──────┴──────────┴─────────┴──────────────┴──────────────┴──────────┴──────────┘

  Bonferroni → 1 significant result (Metric A only)
  BH         → 3 significant results (Metrics A, B, C)
               Largest k where condition holds = k=3
```

The BH threshold line versus actual p-values visualised:

```
  p-value
  0.05 │                                                    J
       │                                           I
  0.04 │                                   H
       │                             G
  0.03 │                         F  ← BH threshold line (diagonal)
       │                    E   /
  0.02 │               D   /  ← D falls ABOVE line (not sig)
       │          C   /
  0.01 │     B   /  ← C, B, A all fall BELOW line (sig)
       │  A /
  0.00 │ /
       └─────────────────────────────────────────────────────
         1    2    3    4    5    6    7    8    9   10   rank

  Points BELOW the BH line = significant under BH
  Points ABOVE the BH line = not significant
```

---

## 7. When to Use Which — Decision Framework

```
                  How many metrics am I testing?
                         /              \
                      FEW              MANY
                    (1-5)            (6-50+)
                      │                  │
           How high is the stake?    What's the goal?
              /          \              /         \
           HIGH           LOW      CONFIRM      EXPLORE
        (guardrail,     (nice to   (ship on    (generate
         safety)         know)      this)       leads)
            │               │          │            │
        BONFERRONI       NONE       SINGLE       BH / FDR
        (or evaluate    needed      PRIMARY      (controlled
         solo at own               + no MTC      discovery
         threshold)                needed        rate)
```

### The critical insight most candidates miss

```
┌────────────────────────────────────────────────────────────────────┐
│  A single, truly pre-registered primary metric does NOT need       │
│  multiple testing correction.                                      │
│                                                                    │
│  MTC is about simultaneously tested hypotheses — not about        │
│  every metric that exists on a dashboard.                          │
│                                                                    │
│  Pre-registration (Ch. 8) + single primary = no correction needed  │
│  Multiple simultaneous primary candidates = correction needed      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. The Guardrail Metric Special Case

This is a favourite interview trap. Do you apply the same correction to guardrails as to exploratory metrics?

```
┌─────────────────────────────────────────────────────────────────────┐
│  WRONG approach:                                                    │
│  Lump crash rate + 14 engagement metrics under one BH correction   │
│                                                                     │
│  Why it fails:                                                      │
│  BH trades some false positives for power. That trade is fine      │
│  for engagement metrics where a false alarm costs little.          │
│  But for a guardrail like crash rate, a false NEGATIVE             │
│  (missing real harm) is catastrophic — you might ship something    │
│  that damages users.                                               │
│                                                                     │
│  RIGHT approach:                                                    │
│  • Evaluate guardrail metrics INDEPENDENTLY, outside the pool      │
│  • Use FWER-style control (Bonferroni) or their own               │
│    pre-committed thresholds                                        │
│  • Treat exploratory engagement metrics separately under BH        │
│                                                                     │
│  The asymmetry: false negative on a guardrail >> false positive   │
│  So you want high sensitivity (low threshold) for guardrails,     │
│  not the diluted sensitivity from sharing a pool with 14 others   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Correlation Among Metrics

A subtlety that separates strong candidates:

```
  ASSUMPTION                        REAL WORLD
  ──────────                        ──────────
  Bonferroni and BH assume          In product analytics, metrics
  tests are independent.            are often highly correlated.
                                    (CTR, sessions, engagement
                                     all move together)

  EFFECT ON BONFERRONI:             EFFECT ON BH:
  Remains valid — still             Standard BH still valid under
  controls FWER — but               positive dependence (Benjamini
  becomes even MORE                 & Yekutieli 2001), though
  conservative than                 there's a BY variant for
  necessary.                        arbitrary dependence.

  If metrics are correlated,        Positive correlation =
  you're doing fewer                BH standard procedure is
  "effective" independent           conservative enough — use it.
  tests than m suggests.
```

---

## 10. How it Connects to the Rest of the Framework

```
  CH. 8 PRE-REGISTRATION
  ┌─────────────────────────────────────────────────────────────────┐
  │  Commit to ONE primary metric before launch                     │
  │  → eliminates the need for MTC on the primary test              │
  │  → secondary/exploratory metrics still need MTC (Ch. 15)        │
  └─────────────────────────────┬───────────────────────────────────┘
                                 │ if multiple primaries or many
                                 │ secondary metrics, apply:
  CH. 15 MULTIPLE TESTING CORRECTION  ◄──────────────────────────────
  ┌───────────────────────────────────────────────────────────────────┐
  │  FEW high-stakes metrics → Bonferroni (FWER)                     │
  │  MANY exploratory metrics → BH/FDR                               │
  │  Guardrails → evaluated independently                            │
  └─────────────────────────────┬─────────────────────────────────────┘
                                 │
  CH. 16 SEQUENTIAL TESTING ◄───┘  (separate issue — early stopping)
  (peeking at results mid-experiment
   is a time-series version of MTC)
```

---

## 11. Red Flags — Spot the Error

```
┌─────────────────────────────────────────────────────────────────────┐
│  RED FLAGS IN THE WILD                                              │
│                                                                     │
│  ✗ Testing 25 metrics at α=0.05 each, reporting the one           │
│    that hit p=0.03 as a win — no correction applied               │
│    → Expected 1.25 false positives by chance. This proves nothing. │
│                                                                     │
│  ✗ Applying Bonferroni to 50 exploratory metrics, finding          │
│    nothing significant, concluding "no effect"                     │
│    → Bonferroni at m=50 is so strict (α=0.001) that real but      │
│      moderate effects are invisible. Use BH instead.              │
│                                                                     │
│  ✗ Guardrail metrics (crash rate, latency) lumped in the           │
│    same FDR correction pool as engagement metrics                  │
│    → Dilutes sensitivity for safety-critical checks.               │
│                                                                     │
│  ✗ "We only have one primary metric so we don't need correction"   │
│    — said while also testing 8 secondary metrics and ready         │
│    to ship on whichever one is significant                         │
│    → Correction IS needed for the secondary metrics.              │
│                                                                     │
│  ✓ One pre-registered primary + BH on secondary metrics           │
│  ✓ Guardrails evaluated at own threshold, outside the pool        │
│  ✓ Communicating explicitly: "Metric X is exploratory only"       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. Concept Summary — One Diagram

```
                    m hypotheses tested simultaneously
                              │
              ┌───────────────┴───────────────┐
              │                               │
        CONFIRMATORY                    EXPLORATORY
        (pre-registered                 (secondary /
         primary only)                   guardrail /
              │                          discovery)
              │                               │
      No MTC needed                          │
      (1 test = 1 chance          ┌──────────┴──────────┐
       to err)                    │                     │
                             HIGH-STAKES            MANY LOW-
                             FEW METRICS            STAKES METRICS
                             (guardrails)           (engagement,
                                  │                  retention...)
                                  │                     │
                            BONFERRONI              BH / FDR
                            α_adj = α/m          Rank p-values,
                            FWER control         adaptive threshold
                            "zero false          "controlled false
                             positive            discovery rate"
                             guarantee"
```

---

## 13. Flash Cards — Interview Prep

```
Q: What does FWER stand for and what does Bonferroni control?
A: Family-Wise Error Rate — probability of AT LEAST ONE false positive
   across all tests. Bonferroni: α_adj = α/m

Q: What does FDR stand for and what does BH control?
A: False Discovery Rate — EXPECTED PROPORTION of false positives
   among all results declared significant. BH controls this.

Q: Why is BH more powerful than Bonferroni?
A: BH controls a proportion (FDR), not a probability (FWER). A more
   lenient standard → higher power to detect true effects.

Q: Does a single pre-registered primary metric need MTC?
A: No. MTC is for simultaneously tested hypotheses. One primary = one
   test = no correction needed.

Q: With m=25 metrics, what is the Bonferroni threshold at α=0.05?
A: 0.05/25 = 0.002. A p=0.03 result does NOT clear this bar.

Q: Should guardrail metrics share a BH pool with exploratory metrics?
A: No. Evaluate guardrails independently — false negatives on safety
   metrics are catastrophic; you need full sensitivity, not the
   diluted sensitivity of a shared correction pool.

Q: What is the BH procedure in 3 steps?
A: (1) Rank p-values smallest to largest. (2) For each rank k, compute
   threshold = (k/m)×α. (3) Find the largest k where p(k) ≤ threshold.
   Declare all ranks 1 through k* significant.

Q: Primary metric not significant, but a secondary metric is p=0.04
   out of 8 secondary metrics. PM wants to ship. What do you say?
A: Bonferroni threshold = 0.05/8 = 0.00625. p=0.04 doesn't clear it.
   BH likely won't save it either with only one hit at rank ~4.
   Treat as hypothesis-generating. Pre-register this secondary metric
   as the primary in a new experiment.
```

---

## 14. Connections to Other Chapters

| Chapter | Topic | Connection |
|---|---|---|
| Ch. 8 | Pre-registration | Separating primary from secondary is what determines whether MTC applies |
| Ch. 9 | Power calculations | Bonferroni lowers power per test — must account for this when sizing |
| Ch. 16 | Sequential testing | Peeking mid-experiment is a temporal version of the same MTC problem |
| Ch. 19 | Long-term holdouts | Long-horizon secondary metrics still need MTC if tested alongside short-term ones |

---



## 15. Famous Q&A (Google / Apple style)

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
