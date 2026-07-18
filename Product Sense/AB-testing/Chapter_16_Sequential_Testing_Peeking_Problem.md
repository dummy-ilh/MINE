# Chapter 16 (Rebuilt for Clarity): Multiple Testing Correction — FWER vs. FDR

*Quick note before starting: the source chapter's title says "Sequential Testing & the Peeking Problem," but the actual content is about a different (related) problem — testing many metrics/segments at once, not peeking at the same metric over time. This rebuild is titled to match the actual content: multiple testing correction. Sequential testing/peeking is its own topic (covered elsewhere in the curriculum) and isn't duplicated here.*

Same rebuild approach as Ch13/14: intuition and diagrams first, formula second, a diagnosis section for when correction is even needed and which framework to pick, step-by-step plug-in walkthroughs for both Bonferroni and Benjamini-Hochberg, and a dedicated why/why-not section.

---

## 1. Start Here: The Problem in One Picture

```
   You run ONE test at α = 0.05.
   If there's truly no effect, you have a 5% chance
   of a false alarm. Feels safe.

              ● ── 5% false-alarm risk


   You run TWENTY tests at α = 0.05 each — say, 20
   secondary metrics on the same experiment.
   If NONE of them have a true effect, what's the
   chance AT LEAST ONE looks "significant" anyway?

   ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
   5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5% 5%

   Twenty independent 5%-risk rolls of the dice.
   The chance that AT LEAST ONE comes up "false alarm"
   stacks up fast — not 5%, but ~64%.
```

That's the entire problem: **each individual test still looks "safe" at 5%, but the *family* of tests, taken together, is nowhere close to safe.** This is exactly the "we tested 15 metrics, one came back significant, let's call that our win" failure mode — that one winner is very plausibly just one of the dice rolls coming up false.

---

## 2. The Formula, Built From the Picture

If each test independently has a $(1-\alpha)$ chance of NOT being a false alarm, then the chance that ALL $m$ tests avoid a false alarm is $(1-\alpha)^m$ (multiply the "safe" probabilities together, since they're independent). So the chance of **at least one** false alarm is just 1 minus that:

$$P(\text{at least one false positive}) = 1-(1-\alpha)^m$$

```
STEP-BY-STEP for m=20, α=0.05:

  chance ONE test is "safe"          = 1 - 0.05 = 0.95
  chance ALL 20 tests are "safe"     = 0.95^20  ≈ 0.358
  chance AT LEAST ONE is a false alarm = 1 - 0.358 = 0.642
                                          ↑
                            64% — worse than a coin flip
```

This quantity — the probability of *any* false positive across the whole family — has a name: the **Family-Wise Error Rate (FWER)**.

---

## 3. Diagnosis: Do You Even Have a Multiple-Testing Problem?

This is the part most people skip, and it's the actual interview signal — recognizing the situation before reaching for a formula.

```
              How many things are you testing
              AT THE SAME TIME on this experiment?
              (metrics, segments, variants — anything
               where you're checking "is this one
               significant?" more than once)
                          │
              ┌────Just 1──┴──More than 1──────────┐
              ▼                                      ▼
     No multiple-testing problem.          You have a multiple-testing
     A single, pre-specified primary       problem. Continue below.
     OEC needs no correction — this
     is exactly why pre-registering
     ONE primary metric before the
     experiment runs is the best
     defense (Section 8).


        (continuing from "more than 1" branch)
                          │
                          ▼
              Is this a CONFIRMATORY, high-stakes
              decision (e.g., "does this single
              metric decide whether we ship a
              multi-million-dollar feature")?
                          │
              ┌────Yes───┴───No (exploratory)───────┐
              ▼                                       ▼
     Use FWER control                       Use FDR control
     (Bonferroni or Holm-Bonferroni)        (Benjamini-Hochberg)
     — even ONE false positive here          — you expect and can
     is costly, so be strict.                tolerate SOME false
                                              leads, as long as most
                                              flagged findings are real
                                              (e.g., screening 50 user
                                              segments for follow-up).
```

**The one-sentence version**: if a single wrong "win" would be expensive, control FWER (strict). If you're casting a wide net and following up on whatever looks promising anyway, control FDR (lenient but still principled).

---

## 4. How to Plug In — Bonferroni (FWER)

**The idea, before the formula**: if you have $m$ tests and want the *whole family's* false-positive risk to stay at $\alpha$, just give each individual test a much stricter bar — divide the risk budget evenly across all $m$ tests.

$$\text{per-test threshold} = \frac{\alpha}{m}$$

**Step-by-step, using this chapter's numbers** (10 secondary metrics, family-wise $\alpha=0.05$):

```
STEP 1 — Divide your risk budget:
    threshold = 0.05 / 10 = 0.005

STEP 2 — Compare EACH metric's own p-value to 0.005,
         independently (not sorted, not ranked — just
         a flat bar every test must individually clear):

    p-values (sorted for readability, not required):
    0.001   0.008   0.015   0.021   0.033   0.041   0.09   0.15   0.31   0.52
      ✓       ✗       ✗       ✗       ✗       ✗       ✗      ✗      ✗      ✗
   (0.001 < 0.005)   (all the rest are ABOVE 0.005)

STEP 3 — Result: only 1 metric (p=0.001) survives.
```

**Why it's "conservative"**: notice how brutal this was — 5 metrics would have looked significant at the naive, uncorrected $\alpha=0.05$ bar, but only 1 survives Bonferroni. That's the cost of guaranteeing the *whole family's* false-positive risk stays at 5% — you pay for it in lost power (real effects can get missed too, not just false ones filtered out).

**Holm-Bonferroni, the slightly less brutal version**: instead of the same flat $\alpha/m$ threshold for every test, sort p-values smallest to largest and give the $k$-th smallest a slightly looser threshold, $\alpha/(m-k+1)$:

```
Sorted p-values (10 tests): 0.001, 0.008, 0.015, 0.021, 0.033, ...

  k=1: threshold = 0.05/10 = 0.0050   →  compare to p=0.001  → ✓ survives
  k=2: threshold = 0.05/9  = 0.0056   →  compare to p=0.008  → ✗ fails, STOP
       (once one fails, everything after it also fails — you stop
        checking further down the sorted list)

Result: same as plain Bonferroni here (1 survivor), but Holm-Bonferroni
would have given MORE metrics a chance to survive in other datasets,
since only the 1st comparison uses the full α/m — every subsequent
one gets a slightly loosened bar.
```

---

## 5. How to Plug In — Benjamini-Hochberg (FDR)

**The idea, before the formula**: instead of protecting against ANY false positive (Bonferroni's strict goal), you're OK with some false positives, as long as they stay a controlled *fraction* of whatever you declare "significant." So the more tests you have, and the more of them look promising, the more lenient your bar becomes for the ones near the top.

**Step-by-step, same 10 p-values:**

```
Sorted p-values (k = rank, 1 = smallest):
  k    p-value    threshold = (k/m)×α = (k/10)×0.05
  1    0.001      0.005              → 0.001 ≤ 0.005 ✓
  2    0.008      0.010              → 0.008 ≤ 0.010 ✓
  3    0.015      0.015              → 0.015 ≤ 0.015 ✓
  4    0.021      0.020              → 0.021 > 0.020 ✗
  5    0.033      0.025              → 0.033 > 0.025 ✗
  6    0.041      0.030              → 0.041 > 0.030 ✗
  7    0.09       0.035              → fails
  8    0.15       0.040              → fails
  9    0.31       0.045              → fails
  10   0.52       0.050              → fails

FIND THE LARGEST k WHERE THE ROW STILL PASSES.
That's k=3 (rows 4 and 5 failed their OWN row's threshold,
but that doesn't matter — you look for the largest k that
passed, which is k=3, and declare EVERYTHING up to and
including that k significant).

Result: metrics 1, 2, and 3 (p = 0.001, 0.008, 0.015)
are declared significant.
```

**The one subtle trap here, worth memorizing**: rows 4 and 5 individually failed their own threshold, but that's irrelevant — BH doesn't require every row up to your cutoff to pass its own row-specific threshold, it just requires you to find the *largest* k that passes, then take everything at or below that k. This is the single most common mechanical mistake people make when computing BH by hand.

---

## 6. Side-by-Side: What Each Method Gives You on the Same Data

```
             Naive (no correction):  6 "significant" findings
             Benjamini-Hochberg:     3 "significant" findings
             Bonferroni:             1 "significant" finding
                     │                        │                │
                     ▼                        ▼                ▼
              Most liberal,           Middle ground,      Most strict,
              least trustworthy       tolerates SOME       guards against
                                      false positives      even ONE
                                      among discoveries     false positive
```

This ordering — naive ≥ FDR ≥ FWER, in number of things declared significant — holds generally. It's a useful sanity check: if your FDR-corrected count is somehow HIGHER than your naive count, you've made an arithmetic mistake somewhere.

---

## 7. FWER vs. FDR — What Each One Is Actually Promising You

```
   FWER (Bonferroni/Holm) promises:
   "The probability that even ONE of my declared
    findings is a false positive stays below 5%."
                    │
                    ▼
   Very strict. Good when a single wrong call
   is expensive — e.g., this one metric decides
   whether a multi-million-dollar feature ships.


   FDR (Benjamini-Hochberg) promises:
   "OF the findings I declare significant, the
    EXPECTED FRACTION that are false positives
    stays below 5%."
                    │
                    ▼
   More lenient. Good for exploratory screening —
   e.g., scanning 50 user segments for follow-up
   investigation, where finding "several real leads,
   with a controlled fraction of duds mixed in" is
   an acceptable, even expected, outcome.
```

Confusing these two is one of the most common interview slips on this topic — they answer genuinely different questions, not two names for the same idea.

---

## 8. Why NOT to Apply Correction (or: When It's Unnecessary)

```
REASON 1 — You only have ONE pre-specified primary metric
   If your experiment truly commits to a single OEC before
   running, and that's the only thing you're testing to make
   the ship decision, there's no multiple-testing problem for
   THAT decision — applying Bonferroni to a single test is a
   no-op (α/1 = α) and shows you don't understand what the
   correction is for.

REASON 2 — Over-correcting exploratory work kills useful signal
   If you're screening 50 segments purely to decide which ones
   deserve a follow-up look (not a final ship decision), applying
   strict FWER control (Bonferroni) can bury real, worth-investigating
   leads under an overly strict bar meant for high-stakes confirmatory
   decisions. FDR is usually the better fit here — using FWER anyway
   is a real cost, not just extra caution.

REASON 3 — Correcting tests that aren't actually simultaneous
   If you're looking at completely separate, previously-run,
   independent experiments (not one experiment's family of
   metrics), applying a joint correction across them may not
   be the right frame — think carefully about what "family" of
   tests you're actually correcting for before mechanically
   applying either method.
```

**The honest summary**: correction is for *families of simultaneous tests within one analysis decision*. Applying it where there's truly only one test wastes nothing but also does nothing; applying the wrong flavor (FWER where FDR was appropriate, or vice versa) has a real cost in either missed discoveries or an inflated false-positive rate.

---

## 9. Production Considerations

- **Pre-registration is the best defense, not a correction formula**: pre-specify a single primary OEC before the experiment runs. Done properly, this mostly sidesteps the multiple-testing problem for your *ship decision* — it only becomes a live issue for secondary/exploratory metrics, which should be clearly labeled exploratory/hypothesis-generating, not confirmatory, in your writeup.
- **Guardrail metrics are a related multiple-testing surface**: the more guardrails you add, the higher your chance some guardrail trips by pure chance — "guardrail proliferation" is a real risk, and this chapter's machinery (FWER/FDR) is exactly how you'd manage it if you have many guardrails.
- **At Google-scale, thousands of experiments run concurrently across the company** — this raises a higher-level multiple-testing question (how many company-wide "wins" are actually noise), often handled via meta-analysis of experiment win-rates to sanity-check whether the observed rate of "significant" launches is consistent with a much lower true win rate.
- **Segment analysis is a classic multiple-testing trap**: slicing data into 30 segments and looking for "which segment did the treatment help" is implicitly 30 tests — apply FDR/FWER correction rather than reporting the single most impressive-looking segment as if it were pre-specified.

---

## 10. Q&A

**Q: If you run 15 independent tests at α=0.05 each, and none have a true effect, what's the probability of at least one false positive?**
A: $1-(0.95)^{15} \approx 1-0.463=0.537$ — about a 54% chance of at least one false positive across the family, even though each individual test looks "safe" at 5%.

**Q: Explain the difference between what FWER controls and what FDR controls, and give a scenario for each.**
A: FWER controls the probability of even a single false positive anywhere in the family of tests — appropriate for confirmatory, high-stakes decisions like "which one metric determines whether we ship this feature." FDR controls the expected *proportion* of false positives among your declared discoveries — appropriate for exploratory work like screening many user segments for follow-up, where you can tolerate some false leads as long as most flagged findings are real.

**Q: A colleague slices experiment results by 25 different user segments, finds one segment with p=0.02, and wants to write it up as a key finding. What's your concern?**
A: This is an implicit 25-test multiple-testing scenario, even though it wasn't framed that way — with 25 tests at α=0.05, you'd expect roughly one false positive by chance alone even if nothing real is happening. I'd recommend applying an FDR correction (Benjamini-Hochberg) across all 25 segment p-values before treating any single one as a real finding, and framing this as exploratory/hypothesis-generating rather than confirmatory unless the segment was pre-specified before looking at the data.

**Q: Why does pre-specifying a single primary OEC protect you from the multiple-testing problem for your ship decision?**
A: Because the multiple-testing problem only arises when you're testing more than one thing and treating any of them as a basis for a decision. If your ship decision rests on exactly one pre-committed metric, there's nothing to correct for on that decision — Bonferroni applied to $m=1$ is just $\alpha/1=\alpha$, a no-op. The problem re-emerges the moment you look at secondary metrics and start treating an interesting one as if it were the plan all along.

**Q: When would applying Bonferroni actually be the wrong choice, even though you genuinely have multiple tests?**
A: When the context is exploratory rather than confirmatory — e.g., screening many segments or many metrics purely to decide what deserves a closer look, not to make a final high-stakes call. Bonferroni's strict FWER control can bury real, worth-investigating signals under a bar that's calibrated for "even one false positive is very costly," which isn't the actual risk profile of exploratory screening. Benjamini-Hochberg's FDR control is the better-matched tool there.

---

