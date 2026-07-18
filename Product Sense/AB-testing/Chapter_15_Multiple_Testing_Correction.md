# Chapter 15: Multiple Testing Correction (Bonferroni, FDR) Across Many Metrics

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

Two separate questions — let me take them one at a time.

---

## If 20 tests and 1 is significant after correction

**Short answer: it's a real signal, but a weak one. Treat it as a lead, not a conclusion.**

Here's why the "after correction" part actually matters a lot:

Before correction, 1 out of 20 being significant is basically meaningless — you'd expect that by pure chance. But *after* Bonferroni (threshold = 0.05/20 = 0.0025), that one result had to clear a much higher bar. It's not noise. Something probably moved.

The problem is **power**. To get 1 significant result out of 20 metrics after Bonferroni, you needed a fairly strong signal just to survive that strict threshold. The other 19 being null doesn't mean they have no effect — it means your test wasn't sensitive enough to detect smaller effects after the correction tightened the bar so hard.

So the inference is:

- The 1 significant metric probably has a real effect
- The 19 null results are **uninformative** — they could be truly null, or they could have real effects you didn't have power to detect
- If this metric was pre-registered as your primary → you can ship on it
- If it was one of 20 exploratory metrics → treat it as a hypothesis to pre-register and retest properly

The worst move is saying "19 things had no effect." Bonferroni at m=20 is so strict that moderate real effects simply don't show up.

---

## Why guardrail metrics stay out of the correction pool

Simple version: **the correction trades sensitivity for safety against false alarms. Guardrails need the opposite trade.**

Let's use a concrete example.

Say you're testing a new checkout flow. You track:

- 14 exploratory engagement metrics (time on page, scroll depth, etc.)
- 1 guardrail: **crash rate**

You pool all 15 under BH correction. BH's job is to say "among everything I flag as significant, keep false positives under 5%." To do that, it raises the bar for everything in the pool — including crash rate.

Now suppose the new checkout flow causes crashes to go up by 15%. The p-value on crash rate comes back at p = 0.04. Under the shared BH correction with 14 other metrics, that might not survive — BH looks at where crash rate ranks among all 15 p-values and might decide 0.04 isn't strong enough given the pool size.

**You just missed a real crash regression. You ship. Users crash.**

That's the problem. BH is designed to tolerate some false positives in exchange for catching more true positives. That's a perfectly fine trade for engagement metrics — if you falsely flag "scroll depth increased" you've wasted some follow-up effort, no big deal.

But for crash rate, the trade is backwards. You don't want to tolerate false *negatives* (missing real harm). You want maximum sensitivity — meaning a low, strict, independent threshold just for that metric.

```
  ENGAGEMENT METRICS          GUARDRAIL METRICS
  ─────────────────           ─────────────────
  False positive cost:        False positive cost:
  Low — wasted follow-up      Low — you block a launch
  work, that's it             unnecessarily. Annoying.

  False negative cost:        False negative cost:
  Low — missed a signal,      HIGH — you ship something
  run another test            that crashes/harms users

  So you want:                So you want:
  Fewer false positives       Fewer false NEGATIVES
  → raise the bar (BH)        → keep the bar LOW and
                                separate from the pool
```

In practice: evaluate crash rate at its own pre-committed threshold (say p < 0.05, independently), completely separate from whatever correction you apply to the 14 engagement metrics. If crash rate flags, you halt — regardless of what BH says about the engagement pool.



## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Testing many metrics at the standard α=0.05 without any correction, then reporting whichever one happened to be significant
- ❌ Applying Bonferroni to a huge number of exploratory metrics and concluding "nothing is significant" without considering FDR as a less conservative, still-principled alternative
- ❌ Lumping high-stakes guardrail metrics into the same correction pool as low-stakes exploratory metrics
- ❌ Forgetting that a single, truly pre-registered primary metric doesn't need multiple-testing correction — correction is about the number of *simultaneously tested* hypotheses, not every metric that exists in a dashboard
- ✅ Do: separate confirmatory (primary, pre-registered) from exploratory metrics before deciding whether/how to correct
- ✅ Do: use Bonferroni when a strict, zero-tolerance guarantee is needed; use BH/FDR when better power across many exploratory checks is more valuable

---
This is a classic and high-stakes topic for Product/DATA/DS interviews at Google, Meta, and Apple. 

When you run an A/B test, you are usually not looking at just one metric (e.g., "Revenue"). You are looking at **dozens, if not hundreds**, of secondary metrics (e.g., Click-Through Rate, Session Duration, Bounce Rate, specific funnel steps, and guardrail metrics like Crash Rate).

Here is exactly how to structure your answer, the specific questions you will be asked, and how to handle the counter-arguments that interviewers love to throw at you.

---

### Part 1: The "Cold Call" Opening Question
**Interviewer:** *"You ran an A/B test with 50 different success metrics. Your product manager says 3 metrics came back statistically significant at p < 0.05. Do you ship the feature?"*

**Your Instant Answer:** 
**"Absolutely not.** If we run 50 independent tests at α=0.05, the probability of seeing **at least one** false positive (Type 1 error) is \( 1 - (0.95)^{50} \approx 92\% \). We have almost certainly found noise. We need to correct for multiple comparisons before we make a shipping decision."

---

### Part 2: The Three Main Solutions (The "Framework")

You need to clearly differentiate when to use each method. Here is the interview cheat sheet:

| Method | What it does | Best used for | The Trade-off |
| :--- | :--- | :--- | :--- |
| **Bonferroni** | Divides your alpha (0.05) by the **number of metrics** (e.g., 0.05 / 50 = 0.001). | **Guardrail metrics** (Crashes, Revenue). Metrics where a false positive is **catastrophic** and you cannot afford to break anything. | Massively increases **False Negatives (Type II errors)**. If you have 50 metrics, you need a massive sample size to detect a tiny lift. |
| **FDR (False Discovery Rate)** - specifically **Benjamini-Hochberg** | Ranks p-values from smallest to largest. Finds the largest p-value that is still below \((rank / total) * 0.05\). | **Exploratory metrics** (Click-through on secondary UI elements, country-specific breakdowns). You are looking for "signals" to generate new hypotheses. | You will ship a few things that are actually false positives, but you explicitly accept that risk (usually 5% or 10% FDR). |
| **Hierarchical/Step-down** (e.g., Holm-Bonferroni) | Less conservative than Bonferroni. It sorts p-values and applies a sequentially less strict correction. | A "Goldilocks" compromise. Used when you have a primary metric (uncorrected) and secondary metrics (corrected). | More complex to explain to non-technical stakeholders. |

---

### Part 3: The "Meta/Google" Nuance (The Family-Wise Error Rate)

At Meta and Google, you rarely just blindly apply Bonferroni to all 50 metrics. The interviewer will push back:

**Interviewer:** *"But Bonferroni is too conservative. If I use it, I'll never ship anything because my sample size isn't big enough. What do we actually do in practice?"*

**Your Advanced Answer (The "Meta Way"):**
"We apply **metric hierarchy**:

1. **Primary Metric (The North Star):** We **do NOT** correct this. We run this at α=0.05. The entire experiment is powered for this one metric. If it fails, we don't ship, regardless of secondary metrics.
2. **Guardrail Metrics (Trust/Safety):** We correct these aggressively using **Bonferroni**, but we only have 2 or 3 of them (e.g., Revenue per User, Crash Rate). We must protect the user experience at all costs.
3. **Exploratory/Diagnostic Metrics (the other 45):** We apply **FDR (Benjamini-Hochberg)** at a 10% or 20% level. We treat these as directional. We use them to understand *why* the primary metric moved, but we do not use them as the sole reason to ship."

---

### Part 4: The "Apple" Nuance (The FWER vs. FDR Debate)

Apple cares deeply about user privacy and system stability. They will push you on the philosophical difference.

**Interviewer:** *"Explain to me in plain English the difference between Family-Wise Error Rate (FWER) and False Discovery Rate (FDR). When would you choose one over the other?"*

**Your Answer:**

- **FWER (Bonferroni):** "The probability of making **even one** false discovery across all tests." 
  - *Use when:* You are testing a new iOS update. If you are wrong about *any* battery-life metric, the press will destroy you. You need absolute certainty.
- **FDR (Benjamini-Hochberg):** "The expected **proportion** of false discoveries among *only the rejected hypotheses*." 
  - *Use when:* You are testing a new Search ranking algorithm. You are looking at 100 different search sub-queries. You expect 95 of them to be null (no change). You don't care if 3 out of the 10 that show improvement are actually false positives, because you just need to know which directions to iterate on next quarter.

---

### Part 5: The "Practical Execution" Question

**Interviewer:** *"Walk me through how you would apply the Benjamini-Hochberg (FDR) procedure manually on a whiteboard."*

**Your Script:**
"Let’s say I have 5 metrics with p-values:
`0.001, 0.008, 0.03, 0.045, 0.20`. I want to control FDR at 10% (q=0.10).

1. I sort them from **smallest to largest** (they already are).
2. I assign a rank (k) from 1 to 5.
3. I calculate the **critical value** for each: \((k / 5) * 0.10\).
   - Rank 1: \(0.2 * 0.10 = 0.02\) 
   - Rank 2: \(0.4 * 0.10 = 0.04\)
   - Rank 3: \(0.6 * 0.10 = 0.06\)
   - Rank 4: \(0.8 * 0.10 = 0.08\)
   - Rank 5: \(1.0 * 0.10 = 0.10\)
4. **The Rule:** Find the **largest** p-value that is *still less than or equal to* its critical value.
   - p=0.03 is less than 0.06 (Rank 3) → Significant.
   - p=0.045 is less than 0.08 (Rank 4) → Significant.
   - p=0.20 is NOT less than 0.10 (Rank 5) → Not significant.
5. **Conclusion:** The metrics with p=0.001, 0.008, 0.03, and 0.045 are all declared significant. The p=0.20 is not."

---

### Part 6: The "Stakeholder" Curveball

**Interviewer:** *"Your PM doesn't care about statistics. They see that 'Time on Site' went up with a p-value of 0.03, but Bonferroni says it's not significant. They want to ship it. What do you say?"*

**Your Answer:**
"I would reframe the risk in business terms. I would say: *'If we ship this based on Time on Site, there is a 90%+ chance that this lift is actually random noise from the other 49 metrics we looked at. If we ship this, and it actually hurts Retention (our primary metric) in the long run, we lose X million dollars. I recommend we run a follow-up, focused experiment—pre-registering only 'Time on Site' as the primary metric—to validate this finding before we commit engineering resources to a full rollout.'*"

---

### Part 7: The "Preregistration" Bonus Points (To sound Senior)

Add this at the end to get a "Hire" signal:
"To avoid this entire debate in the future, I always push the team to **pre-register our metrics** before the experiment starts. We specify:

- **1 OEC (Overall Evaluation Criterion):** No correction needed.
- **≤ 3 Guardrails:** Bonferroni correction.
- **The rest:** We bucket them into a 'Signals Dashboard' where we explicitly label them as 'FDR-controlled, exploratory' and do not allow them to trigger a 'Ship' decision without a validation experiment."

---

### Summary Table for your Interview Cheat Sheet:

| Scenario | What to apply |
| :--- | :--- |
| Testing a new Ads algorithm; breaking revenue is unacceptable. | **Bonferroni** on the 3 revenue guardrails. |
| Testing a new UI layout; you want to see which of 50 micro-interactions improved. | **FDR (Benjamini-Hochberg)** at 10%. |
| A regulatory/compliance test where the government fines you for any negative effect. | **Bonferroni** (or ideally, Sequential Testing with alpha-spending). |
| A PM cherry-picked 1 metric out of 100 that went up. | Reject the request; demand a **Holdout Validation** or a new, powered experiment. |
| You have a Primary Metric (Conversion) and 4 Secondary Metrics. | **Holm-Bonferroni** (step-down) for the secondaries, Primary stays at 0.05. |


Good question. This is actually a known failure mode of over-correcting — it has a name: **underpowered experimentation culture**.

---

## Why it happens

Bonferroni at large m is so strict that real, meaningful effects simply don't clear the bar. Teams run experiments, see "no significant results," and nothing ships. After a while people stop trusting the process — or worse, start gaming it.

```
  THE DEATH SPIRAL

  Strict correction → nothing significant
         ↓
  Teams run more metrics "just in case"
         ↓
  More metrics → even stricter correction
         ↓
  Even less gets through
         ↓
  Engineers bypass experiments entirely
  ("let's just ship it, A/B testing never finds anything")
```

This is arguably worse than the original problem. A few false positives are recoverable. A culture that stops experimenting is not.

---

## The real fixes

**1. Fewer metrics, not more correction**

The root cause is usually tracking too many metrics simultaneously. If you pre-register one primary metric per experiment (Ch. 8), you don't need correction at all on that test. The correction problem only appears when you've let metric sprawl happen.

```
  BAD                           GOOD
  ───                           ────
  20 metrics, all "primary"     1 primary metric
  → Bonferroni → nothing        → no correction needed
    survives                    → full power preserved
```

**2. Switch to BH for exploratory metrics**

If you genuinely need to test many secondary metrics, Bonferroni is the wrong tool. BH gives you far more power while still controlling the false discovery rate. Something that dies under Bonferroni at m=20 (threshold 0.0025) might survive BH easily if several other metrics also show strong signal.

**3. Check if you're underpowered before blaming correction**

Sometimes nothing gets through because the experiment wasn't run long enough, or the MDE was set too small for the traffic available. The correction is blamed but the real problem is sample size. Run the power calculation *before* the experiment, not after.

**4. Separate decision tiers explicitly**

```
  TIER 1 — SHIP DECISION        TIER 2 — LEARNING
  ──────────────────────        ──────────────────
  1 pre-registered primary      Secondary metrics under BH
  No correction needed          Results labeled "exploratory"
  Full power                    Feed future experiment designs
  Binary: ship or don't         No ship decision from these
```

Nothing in Tier 2 ships directly. But it also doesn't go to waste — a BH-significant secondary metric becomes next experiment's primary. You build a pipeline of hypotheses rather than demanding every experiment be a ship decision.

---

## The uncomfortable truth

If experiments consistently produce nothing, the problem is usually upstream — vague hypotheses with no real causal mechanism, MDEs set too small, or too many metrics diluting power. The correction is doing its job. The hypothesis generation isn't.

A well-run team should expect roughly:
- ~20-30% of experiments to ship (meaningful positive effects)
- ~40-50% to be true nulls (the change simply didn't matter)
- ~20-30% to be informative failures that sharpen the next hypothesis

If your ship rate is 0%, the correction isn't the problem.

No universal answer, but there's a strong consensus from teams that run experiments at scale.

---

## The ratio that actually works

```
  PER EXPERIMENT

  Primary metrics:     1  (the ship/no-ship decider)
  Guardrail metrics:   2–5
  Secondary/learning:  5–10

  Total:               ~8–15 metrics maximum
```

Beyond ~15, you're not learning more — you're just inflating your correction burden and making it harder to explain results to stakeholders.

---

## Why each tier has a natural limit

**Primary — exactly 1**

If you have 2 "primary" metrics you don't actually have a primary metric. You have a choice you're deferring until after results come in. That's the garden of forking paths problem (Ch. 8) in disguise. One metric, one ship decision.

**Guardrails — 2 to 5**

These cover the things you'd be embarrassed to have broken. For most products that's something like: crash rate, latency/load time, core revenue metric, support ticket volume. You rarely need more than 5 because genuinely catastrophic regressions tend to show up in a small number of places.

The ceiling is 5 because each guardrail needs to be evaluated at full sensitivity independently. The more you add, the more you're crying wolf on launches.

**Secondary/learning — 5 to 10**

These go under BH correction, so adding more is less costly than adding primaries or guardrails. But past ~10 you're tracking noise and the results become uninterpretable in a readout. A stakeholder can absorb "here are 7 secondary metrics and what they tell us." They cannot absorb 30.

---

## What the big platforms actually do

```
  Google / Microsoft ExP:
  Enforce declaration of primary metric at launch.
  Secondary metrics unlimited in dashboards but
  only primary drives the decision.

  Booking.com:
  Famous for running thousands of experiments simultaneously.
  Each individual experiment still has one OEC
  (Overall Evaluation Criterion) — their term for primary metric.

  Netflix:
  Uses a composite metric (single score) as primary
  to avoid the multi-primary problem entirely.
  Secondary metrics are learning only.
```

The pattern is consistent — one primary, everything else is learning.

---

## The composite metric trick

If you genuinely can't pick one metric because you care about several equally, the mature solution is to combine them into a single composite score before the experiment runs.

```
  INSTEAD OF:                   DO THIS:
  Primary 1: CTR                OEC = 0.4×(CTR) + 0.4×(revenue)
  Primary 2: Revenue                + 0.2×(session length)
  Primary 3: Session length
  → need correction             → one number, no correction,
  → ambiguous ship decision       weights decided before launch
```

The weights are a product/business decision made in advance. Controversial, but Netflix, Booking, and LinkedIn all use variants of this. The key is the weights are pre-committed — you don't tune them after seeing which weighting makes your experiment look better.

---

## When teams go wrong

```
  TOO FEW METRICS               TOO MANY METRICS
  ───────────────               ────────────────
  Miss important regressions    Correction kills everything
  Ship things that break        Nothing gets to production
  adjacent features             (the problem you just asked about)
  Guardrails are the            Stakeholders lose trust in
  classic thing to skip         the process
  "we'll notice in prod"        "A/B testing never finds
                                 anything"

  Sweet spot: enough guardrails to catch real harm,
  few enough secondaries that BH still has power
```

---

## Practical rule of thumb

Before adding any metric to an experiment, ask: **"what decision would change if this metric moved?"**

If the answer is "none — we'd just find it interesting," cut it. Every metric you add costs statistical power on everything else. The discipline of that question alone usually gets teams from 30 metrics down to 10 naturally.
