# Chapter 8: Hypothesis Formulation & Pre-Registration
# Chapter 8 — Hypothesis Formulation & Pre-Registration
> **Interview tier:** Google / Meta / Apple PM / DS rounds  
> **Core risk this prevents:** post-hoc rationalization, inflated false-positive rates, p-hacking

---

## 1. The One-Line Mental Model

> A hypothesis is a **bet placed before the race starts** — not a story told after watching the replay.

Pre-registration forces you to write down your bet *before* you look at any results. Without it, you're not testing a hypothesis — you're fishing.

---

## 2. What a Well-Formed Hypothesis Looks Like

### The four mandatory components

```
┌─────────────────────────────────────────────────────────────────┐
│                   ANATOMY OF A HYPOTHESIS                       │
│                                                                 │
│  ① TREATMENT      What exactly changes?                         │
│  ─────────────    (specific, not vague)                         │
│  "CTA button hex: #9E9E9E → #1565C0"                           │
│                                                                 │
│  ② MECHANISM      WHY should this change behaviour?             │
│  ─────────────    (the causal story — most candidates skip)     │
│  "Grey blends into background → low salience                    │
│   → users miss the CTA → suppressed CTR"                       │
│                                                                 │
│  ③ METRIC + DIR   One metric, one direction, committed now      │
│  ─────────────    (not a menu to pick from after results)       │
│  "Click-through rate — expect INCREASE"                         │
│                                                                 │
│  ④ MDE            Smallest effect worth shipping                │
│  ─────────────    (business decision → feeds sample size calc)  │
│  "≥ 3% relative lift on CTR"                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Weak vs. strong hypothesis

| Weak ❌ | Strong ✓ |
|---|---|
| "Let's see what the blue button does" | Specific treatment + mechanism + metric + MDE |
| "We think this might help engagement" | "We believe X will increase Y by Z because [mechanism]" |
| "Let's check if anything moves" | One pre-committed primary metric |
| Mechanism unstated | Causal chain written out explicitly |

---

## 3. Pre-Registration — What to Lock In

> **Rule:** anything not committed before data collection is *exploratory*, not confirmatory.  
> Exploratory findings generate new hypotheses — they never justify ship decisions alone.

```
┌────────────────────────────────────────────────────────────────┐
│               PRE-REGISTRATION CHECKLIST                       │
│                                                                │
│   Before a single row of data is collected, commit to:        │
│                                                                │
│   □  Primary metric          (one — the ship/no-ship decider) │
│   □  Guardrail metrics       (what would make you halt?)      │
│   □  Statistical test        (t-test? z-test? Mann-Whitney?)  │
│   □  Significance threshold  (α — typically 0.05)            │
│   □  One-sided or two-sided  (decide NOW, not after sign)     │
│   □  Sample size / duration  (from power calculation)         │
│   □  Planned subgroup cuts   (mobile vs desktop, new vs old)  │
│                                                                │
│   Anything added after launch = EXPLORATORY ONLY              │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. The Full Process — End to End

```
                         PRODUCT IDEA
                             │
                             ▼
              ┌──────────────────────────┐
              │  Specify causal mechanism │
              │  WHY would this help?     │
              └──────────────┬───────────┘
                             │
                    Mechanism clear?
                    /              \
                  NO               YES
                  │                 │
          Refine idea      ┌────────▼──────────┐
          (loop back)      │  Pick ONE primary  │
                           │  metric + direction│
                           └────────┬───────────┘
                                    │
                           ┌────────▼──────────┐
                           │  Set MDE, run      │
                           │  power calculation │
                           │  → sample size     │
                           └────────┬───────────┘
                                    │
                    ╔═══════════════▼═══════════════╗
                    ║      PRE-REGISTER              ║
                    ║  metric · α · sided-ness       ║
                    ║  test · duration · subgroups   ║
                    ╚═══════════════╤═══════════════╝
                                    │
                           ┌────────▼──────────┐
                           │   Run experiment   │
                           └────────┬───────────┘
                                    │
                     Primary metric significant?
                       /                    \
                     YES                    NO
                      │                      │
             ┌────────▼──────┐      ┌────────▼──────────────┐
             │ Check guardrail│      │    Null result         │
             │    metrics     │      │  Underpowered or no    │
             └────────┬───────┘      │  real effect           │
                      │              └────────┬───────────────┘
           Guardrails OK?                     │
              /       \                       ▼
            YES        NO            Exploratory analysis only
             │          │            Any signals → new test
     ┌───────▼──┐  ┌────▼────────┐   (properly pre-registered)
     │  SHIP ✓  │  │ INVESTIGATE │
     │          │  │ Do not ship │
     └──────────┘  └─────────────┘
```

---

## 5. The Core Statistical Problem — Garden of Forking Paths

This is the insight that separates candidates who understand experiments from those who just know the vocabulary.

### The math behind cherry-picking

```
  If you track N metrics at α = 0.05 and pick the "winner" after:

  P(at least 1 false positive | nothing real happening) = 1 - (0.95)^N

  N = 1  →  5%   ✓ fine
  N = 5  →  23%  ⚠
  N = 10 →  40%  ✗ bad
  N = 20 →  64%  ✗✗ broken
```

So 2 out of 15 metrics being "significant" at p < 0.05 is consistent with **pure noise**.

### Pre-registration vs. multiple testing correction — not the same thing

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  PRE-REGISTRATION                MULTIPLE TESTING CORRECTION    │
│  (Chapter 8)                     (Chapter 15)                  │
│                                                                 │
│  Prevents the INFORMAL problem:  Handles the FORMAL problem:    │
│  analyst looks at many metrics   you pre-specified multiple     │
│  after the fact and highlights   primary metrics and test them  │
│  whichever one moved — without   simultaneously — Bonferroni,  │
│  documenting that they did so    BH correction applied          │
│                                                                 │
│  Fix: commit to ONE primary      Fix: adjust α per test        │
│  metric before launch            (α/k for Bonferroni)          │
│                                                                 │
│  You need BOTH                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. One-Sided vs. Two-Sided — Why it Must be Decided First

```
  TWO-SIDED TEST                    ONE-SIDED TEST
  H₀: δ = 0                        H₀: δ ≤ 0  (or ≥ 0)
  Hₐ: δ ≠ 0                        Hₐ: δ > 0

  Critical region:                  Critical region:
  ──────────────                    ──────────────
  Split α across both tails         All α in one tail

  α/2 │           │ α/2             0  │             │ α
  ────┤           ├────             ───┤             ├───
   -z │     ...   │ +z               0 │     ...     │ +z

  Use when: you care about          Use when: you would NEVER
  effects in either direction       ship a result in the other
                                    direction — committed BEFORE
                                    data collection
```

**The p-hacking trap:** choosing one-sided after seeing the sign of the effect effectively halves your p-value requirement after the fact. You're claiming the stringency of a one-sided test while having implicitly done a two-sided look.

---

## 7. How to Spot a Broken Experiment — Red Flag Checklist

```
┌────────────────────────────────────────────────────────────────┐
│  RED FLAGS IN THE WILD                                         │
│                                                                │
│  🚩 "Let's see what happens to our key metrics"               │
│     → No pre-specified primary. Fishing license.              │
│                                                                │
│  🚩 "We'll declare success if ANY of these 5 metrics move"    │
│     → Inflated Type I error, no correction applied            │
│                                                                │
│  🚩 15 metrics tracked. 2 hit p < 0.05 at end.               │
│     → ~0.75 false positives expected by chance alone          │
│                                                                │
│  🚩 "Didn't find significance, but look at this segment..."   │
│     → Post-hoc segmentation = exploratory only, never ship    │
│                                                                │
│  🚩 Re-running with "minor tweaks" until something hits       │
│     → Iterative p-hacking. Each attempt inflates family error │
│                                                                │
│  🚩 Adding a new metric after results are in                  │
│     → Moving the goalposts. New metric → new experiment       │
│                                                                │
│  🚩 One-sided test chosen after seeing the direction          │
│     → Direct p-hack. Must be pre-committed                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Organizational Levers — What Mature Companies Do

```
  LOW MATURITY                        HIGH MATURITY
  (culture-dependent)                 (structurally enforced)
  ─────────────────────────────────────────────────────────
  "Trust the analyst to be             Platform requires you to
   rigorous"                           declare primary metric +
                                       sample size BEFORE
                                       experiment can launch

  Post-hoc metric selection            Primary metric locked in
  happens informally and               the experiment record;
  unconsciously                        changing it requires
                                       explicit audit trail

  Re-runs are free — keep             Re-runs consume α budget;
  iterating until significant          must disclose # attempts
  
  Examples of mature platforms:
  Google Overlapping Experiment         
  Framework · Microsoft ExP ·          
  Booking.com · Netflix         
```

---

## 9. Concept Map

```
                    ┌──────────────────────┐
                    │  HYPOTHESIS          │
                    │  Treatment           │
                    │  + Mechanism         │
                    │  + Metric/Direction  │
                    │  + MDE               │
                    └──────────┬───────────┘
                               │ feeds
                    ┌──────────▼───────────┐
                    │  POWER CALCULATION   │
                    │  MDE + baseline var  │──────► sample size
                    │  + desired power     │        & duration
                    └──────────┬───────────┘
                               │ inputs to
                    ┌──────────▼───────────┐
                    │  PRE-REGISTRATION    │◄── locks everything
                    │  Primary metric      │    before data
                    │  Guardrails          │    collection
                    │  Test & α & sided    │
                    │  Subgroup plan       │
                    └──────────┬───────────┘
                               │ then run
                    ┌──────────▼───────────┐
                    │  EXPERIMENT          │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
         ┌────────▼──────────┐   ┌──────────▼──────────┐
         │  CONFIRMATORY      │   │  EXPLORATORY         │
         │  Pre-registered    │   │  Not pre-registered  │
         │  primary metric    │   │  secondary metrics   │
         │  → ship decision   │   │  → new hypothesis    │
         └────────────────────┘   └──────────────────────┘
```

---

## 10. Common Confusions Cleared Up

### Pre-registration vs. HARKing

**HARKing** = Hypothesizing After Results are Known. You run a test, see what moved, then write up the hypothesis as if you'd predicted it. Pre-registration closes this loophole by creating a timestamped, immutable record.

### Iterative p-hacking vs. sequential peeking

```
  SEQUENTIAL PEEKING                  ITERATIVE P-HACKING
  ──────────────────                  ───────────────────
  One experiment, one hypothesis      Multiple experiments,
  Peek at results early               minor tweaks each time
  Stop when p < α                     Re-run until p < α
  
  Fix: sequential testing methods    Fix: pre-commit to N attempts
  (α-spending functions, always-      + adjust threshold across
  valid p-values, O'Brien-Fleming)    all attempts (FWER control)
  
  Chapter 16 topic                    Chapter 8/15 overlap
```

### Exploratory ≠ useless

Exploratory findings are *valuable* — they generate new, better hypotheses. The rule isn't "don't look at unregistered metrics." It's "don't let them drive ship decisions without a confirmatory follow-up."

---

## 11. The 90-Day Retention Trap (Classic Interview Scenario)

A PM adds "90-day retention" as a success criterion *after* a 2-week experiment concludes.

**Why this fails — two independent reasons:**

```
  PROBLEM 1: POWER                    PROBLEM 2: MOVING GOALPOSTS
  ─────────────────                   ──────────────────────────
  Experiment wasn't powered           Adding a success criterion
  for a 90-day metric.                after seeing results =
  Measurement horizon is 4.5x        post-hoc rationalization
  longer than the test duration.      even if well-intentioned.
  
  Variance estimates are              The pre-registration guarantee
  unreliable. Too few data            is broken the moment you
  points to detect real effects.      define "win" after the data
                                      is in.

  Correct fix: design a new,
  properly powered long-horizon
  holdout for retention
  (Chapter 19 territory)
```

---

## 12. Flash Cards — Interview Prep

```
Q: What are the 4 components of a well-formed hypothesis?
A: Treatment · Mechanism · Metric+Direction · MDE

Q: What's the false-positive risk of cherry-picking from 20 metrics?
A: ~64% (1 - 0.95^20) — far above the nominal 5% per test

Q: Why must one-sided vs. two-sided be decided before the experiment?
A: Choosing based on observed direction is p-hacking — halves the 
   effective α after the fact

Q: How is pre-registration different from multiple testing correction?
A: Pre-reg prevents informal cherry-picking by analysts.
   MTC (Bonferroni, BH) handles formally pre-specified multiple metrics.
   You need both.

Q: 2 of 15 metrics hit p < 0.05. Do you ship?
A: No. ~0.75 false positives expected by chance. Treat as 
   hypothesis-generating only. Pre-register the "winner" and retest.

Q: What is HARKing?
A: Hypothesizing After Results are Known — writing up the hypothesis 
   as if it were pre-specified after seeing what moved.

Q: What is the iterative p-hacking trap?
A: Re-running experiments with minor tweaks until significance is hit,
   without adjusting α for the number of attempts.

Q: When is exploratory analysis acceptable?
A: Always — but only for hypothesis generation. Never for ship/no-ship
   decisions without a confirmatory pre-registered follow-up.
```

---

## 13. Connections to Other Chapters

| Chapter | Topic | Connection |
|---|---|---|
| Ch. 9 | Sample size & power | MDE from hypothesis → feeds power calc |
| Ch. 15 | Multiple testing correction | Formal counterpart to pre-registration |
| Ch. 16 | Sequential / peeking | Different from iterative p-hacking |
| Ch. 19 | Long-term holdouts | Right tool for long-horizon metrics |

---


## 14. Famous Q&A (Google / Apple style)

**Q: A test launches with "let's see how it affects our key metrics" as the stated hypothesis, and 15 metrics are tracked. The test ends with 2 of the 15 showing p < 0.05. Should you ship?**
A: This is a strong signal the experiment wasn't properly pre-registered. With 15 metrics tested at α = 0.05 each, you'd expect roughly 0.75 false positives by chance alone even if nothing real is happening — finding 2 "significant" results isn't strong evidence of anything without knowing (a) which metric was pre-specified as primary, and (b) whether a multiple-comparisons correction was applied. Before shipping, I'd ask: was one of these 2 metrics the pre-declared primary metric? If not, I'd treat this as hypothesis-generating (something to test as a *primary* metric in a follow-up, properly powered experiment) rather than a ship decision.

**Q: Why does deciding "one-sided vs. two-sided test" need to happen before running the experiment rather than after seeing the direction of the effect?**
A: Because choosing the test direction based on the observed sign of the effect is a form of p-hacking — it effectively halves your p-value threshold's true stringency after the fact, since you're only ever "choosing" the side that makes your result look better. A one-sided test should only be used when you have a genuine, pre-specified reason to only care about one direction (e.g., you'd never ship a change that makes the product worse, so you only test for improvement) — and that decision has to be locked in before the data is observed, or the Type I error guarantee is broken.

**Q: An engineer says "we don't need to pre-register — we'll just look at the data with an open mind and see what's true." What's the flaw in this reasoning?**
A: The flaw is that "looking with an open mind" doesn't prevent unconscious cherry-picking — humans are very good at post-hoc rationalizing why the metric that moved favorably is "the one that really matters," even without deliberate dishonesty. Pre-registration isn't about distrust of the analyst's intentions; it's a structural safeguard against a well-documented cognitive bias (sometimes called "the garden of forking paths") where the true false-positive rate of an analysis balloons whenever the analysis plan is flexible and decided after seeing the data, even if each individual step seems reasonable in isolation.

**Q: A PM wants to add a "we'll also check retention at 90 days" as an afterthought once the 2-week experiment concludes. What would you say?**
A: I'd point out that checking 90-day retention wasn't part of the original design — the experiment likely wasn't run long enough or powered to reliably detect an effect on a metric with that much longer a measurement horizon, and adding it after the fact as a new "primary" criterion is a form of moving the goalposts. If 90-day retention is genuinely important, I'd suggest treating it as an input to designing a *new*, properly powered long-horizon holdout experiment (see Chapter 19 — long-term holdouts) rather than retrofitting it onto results from a test that wasn't built to answer that question.


1) This is a strong signal the experiment wasn't properly pre-registered. At α = 0.05 with 15 independent metrics, you'd expect ~0.75 false positives by chance alone under the null — finding 2 "significant" results is entirely consistent with noise.

The right questions: Was one of those 2 the pre-declared primary metric before launch? If yes, was a multiple-comparisons correction applied to the other 14? If neither is true, these findings are hypothesis-generating only. The correct move is to pre-register the metric that "won" as the primary in a fresh, properly powered experiment — not to ship.

2) Choosing based on the observed sign is a form of p-hacking. A one-sided test concentrates all of α in one tail — so if you pick the side that matches the direction you already observed, you've effectively halved your p-value requirement after the fact while claiming the stringency of a pre-committed test.

A one-sided test is only valid when you've genuinely pre-committed that a result in the other direction would never change your ship decision — e.g., "we would never ship a change that makes CTR worse." That judgment must be locked in before any data is observed.

3) Chnage of success to 90 days after 2 wekks run-Two separate problems, both fatal to using it as a decision criterion now.

Power: The experiment wasn't designed for a 90-day horizon. With a 2-week run you have far too few data points and unreliable variance estimates for a metric with 4.5× the measurement window. You'd have very low power to detect any real effect.

Goal-post moving: Declaring a new success criterion after seeing the primary results breaks the pre-registration guarantee even if the intent is legitimate. The fix: if 90-day retention genuinely matters, design a new long-horizon holdout (typically a persistent holdback) properly powered for that metric from the start.

4) Pre-registration prevents the informal problem: an analyst looking at many metrics after the fact and highlighting whichever moved — without documenting that they did so. Fix: commit to one primary metric before launch.

Multiple testing correction (Bonferroni, Benjamini-Hochberg) handles the formal problem: you pre-specified multiple primary metrics simultaneously and need to control the family-wise error rate. Fix: adjust α per test.

Yes, you need both. Pre-reg handles the undocumented analyst discretion; MTC handles the formally declared multiple endpoints. They're complementary, not substitutes.
