# Chapter 11 — Novelty & Primacy Effects + Test Duration
> **Interview tier:** Google / Meta / Apple DS / PM rounds  
> **Core risk this prevents:** shipping on a transient signal, killing a genuinely good feature too early  
> **Key principle:** the effect you measure in week 1 may be a reaction to *change itself* — not the change being better

---

## 1. The One-Line Mental Model

> You are not measuring whether the feature is good.  
> You are measuring whether users have **finished reacting to the fact that something changed.**

Until that reaction settles, your metric is lying to you — in either direction.

---

## 2. The Two Effects — Definitions

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  NOVELTY EFFECT                   PRIMACY EFFECT                   │
│  ──────────────                   ─────────────                    │
│  Treatment looks BETTER           Treatment looks WORSE            │
│  than it truly is — early on      than it truly is — early on     │
│                                                                     │
│  Why: users are curious,          Why: existing users have         │
│  excited, exploring the           muscle memory / habits           │
│  new thing just because           baked into the OLD UI.           │
│  it is new.                       Change = friction, confusion,    │
│                                   slowing down — temporarily.      │
│                                                                     │
│  Time trend:                      Time trend:                      │
│  Starts HIGH → decays             Starts LOW → climbs              │
│  to true steady state             to true steady state             │
│                                                                     │
│  Risk if you stop early:          Risk if you stop early:          │
│  Ship something that will         KILL something that would        │
│  underperform its promise         have been genuinely good         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Both are threats to validity for the same reason: you are measuring a **transient reaction to change**, not the durable effect you care about for a permanent launch.

---

## 3. The Restaurant Analogies

These are the clearest intuition pumps for interviews.

```
  NOVELTY EFFECT — the new dish
  ──────────────────────────────
  Restaurant adds a new dish to the menu.
  Week 1: regulars order it out of curiosity.
  Sales spike. Looks like a hit.

  Month 2: novelty wears off.
  Regulars go back to their usual order.
  Sales settle at a much lower level.

  The dish wasn't better. It was just new.
  If you'd hired 3 extra chefs based on week 1 data → mistake.


  PRIMACY EFFECT — the new menu layout
  ──────────────────────────────────────
  Restaurant rearranges the entire menu.
  Week 1: regulars can't find their usual dish.
  Frustrated, they order the first familiar thing they see.
  Sales of the "better" featured item look WORSE.

  Month 2: regulars have learned the new layout.
  The featured item is now easier to find.
  Sales recover and then exceed the old baseline.

  The layout WAS better. It just looked broken for a while.
  If you'd reverted based on week 1 data → wasted a good change.
```

---

## 4. Time-Trend Signatures — What to Look For

The single most important diagnostic: **plot lift by day, not just the pooled average.**

```
  NOVELTY EFFECT SIGNATURE          PRIMACY EFFECT SIGNATURE
  ────────────────────────          ────────────────────────

  Lift                              Lift
   │                                 │              ┌────────
   │ *                               │             /
   │  *                              │            /
   │   **                            │           /
   │     ***                         │          /
   │        ****                     │─────────/
   │            ─────────            │ *  *  *
   │                                 │
   └──────────────────── Day         └──────────────────── Day

  Starts HIGH, decays to plateau    Starts LOW (or negative),
                                    climbs to plateau

  Correct read: use the plateau     Correct read: use the plateau
  NOT the early spike               NOT the early dip
```

If you use the pooled average across the whole experiment window, you blend the transient spike/dip with the true signal and get a number that accurately represents neither.

---

## 5. Worked Examples

### Example A — Onboarding tutorial (novelty effect)

Daily treatment effect on Day-1 retention:

```
  ┌─────────────────┬────────────────────────────────────────────┐
  │  Experiment day │  Treatment effect  │  What's happening     │
  ├─────────────────┼────────────────────┼───────────────────────┤
  │   Day 1–2       │     +4.5pp         │  Users excited,       │
  │   Day 3–4       │     +3.1pp         │  exploring, clicking  │
  │   Day 5–7       │     +2.0pp         │  Novelty fading       │
  │   Day 8–10      │     +1.6pp         │  Settling             │
  │   Day 11–14     │     +1.4pp         │  Nearly stable        │
  │   Day 15–21     │     +1.3pp         │  True steady state    │
  └─────────────────┴────────────────────┴───────────────────────┘

  Pooled 4-day average:  ~+3.8pp   ← what you'd report if you stopped early
  True steady-state:      ~+1.3pp   ← what actually persists

  Overestimate factor: ~3x
```

The effect IS real — it doesn't decay to zero. But reporting +3.8pp for a ship decision would set wildly unrealistic expectations. Every downstream model (revenue projections, headcount planning) built on that number would be wrong by 3x.

### Example B — Navigation redesign (novelty + primacy comparison)

```
  ┌──────┬────────────────────────────────────────────────────────┐
  │  Day │  Nav redesign lift  │  Interpretation                  │
  ├──────┼─────────────────────┼──────────────────────────────────┤
  │   1  │      +8.5%          │  Users actively exploring        │
  │   2  │      +6.0%          │  new layout — novelty peak       │
  │   3  │      +4.5%          │  Decay begins                    │
  │   4  │      +3.0%          │                                  │
  │   5  │      +2.2%          │  Rapid decay                     │
  │   6  │      +1.8%          │                                  │
  │   7  │      +1.7%          │  Plateau begins                  │
  │   8  │      +1.6%          │  ← Stable region                 │
  │   9  │      +1.7%          │  ← True steady-state ~1.6–1.7%  │
  │  10  │      +1.6%          │                                  │
  └──────┴─────────────────────┴──────────────────────────────────┘

  2-day pooled average:  ~7.25%    ← 4.5x overestimate
  True steady-state:      ~1.6%    ← what to report and ship on
```

Stopping after 2 days here doesn't just inflate your number — it makes the feature look like a breakthrough when it's actually a moderate improvement. Stakeholder expectations get miscalibrated for months.

---

## 6. Detection Methods

```
┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 1 — Day-by-day lift plot (always do this first)            │
│                                                                     │
│  Compute treatment effect separately for each day (or week).       │
│  Look for the curve shape: decaying → novelty, climbing → primacy  │
│  Only trust the pooled number AFTER you've confirmed a plateau.    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 2 — New vs. existing user segmentation (the key diagnostic)│
│                                                                     │
│  Novelty/primacy are fundamentally about EXISTING users reacting   │
│  to change. New users have no old habits — nothing to unlearn.     │
│                                                                     │
│  Existing users: volatile early effect that stabilises over time   │
│  New users:      flat, stable effect from day one                  │
│                                                                     │
│  If new users show a stable negative effect from day 1             │
│  → this is a REAL regression, not a primacy effect                 │
│  → primacy effect would only show in existing users                │
│                                                                     │
│  This is the only reliable way to distinguish primacy from         │
│  a genuine regression. Both look identical in pooled data.         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 3 — Days-since-first-exposure cohort analysis              │
│                                                                     │
│  Instead of plotting by calendar day, plot by days since each      │
│  user first saw the treatment. Users enter the experiment on        │
│  different days — calendar-day plots blur their adaptation curves  │
│  together. This normalises for staggered entry.                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 4 — Long-running holdout group                             │
│                                                                     │
│  A small permanent holdout (~1-5% of traffic) that never gets      │
│  treatment, maintained well past experiment conclusion.            │
│  Lets you compare treatment vs. control weeks or months later,     │
│  long after any novelty/primacy dynamics have faded.               │
│  Infrastructure investment — not a one-off trick.                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 5 — Fresh re-exposure experiment (novelty only)            │
│                                                                     │
│  Expose a fresh randomised sample to the same treatment AFTER      │
│  the initial novelty period has plausibly worn off in the          │
│  broader population. If the effect is smaller for the fresh        │
│  sample, the original lift was partly novelty-driven.              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Test Duration — Two Independent Constraints

This is the most commonly missed insight in interviews. Duration is not one number — it is the **maximum of two separate minimums**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  CONSTRAINT 1                     CONSTRAINT 2                     │
│  Statistical power                Effect stabilisation             │
│  ──────────────────               ─────────────────────            │
│  How long until you have          How long until the novelty/      │
│  enough users to detect           primacy curve has plateaued      │
│  your MDE at target α             and the effect is stable         │
│  and power?                       enough to trust?                 │
│                                                                     │
│  Comes from the power             Comes from day-by-day trend      │
│  calculation (Ch. 9)              analysis or historical data      │
│                                   from similar past launches       │
│                                                                     │
│  Run for:  MAX(constraint 1, constraint 2)                         │
│                                                                     │
│  Common mistake: running until constraint 1 is met,                │
│  declaring the experiment done, and never checking constraint 2    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What affects stabilisation time

```
  FASTER STABILISATION              SLOWER STABILISATION
  ────────────────────              ────────────────────
  Subtle change (small copy         Large visual redesign
  tweak, minor layout shift)        (full page overhaul)

  High exposure frequency           Low exposure frequency
  (daily-use feature like           (annual billing flow,
   a home feed)                      account settings)

  Younger product / mostly          Mature product with
  new users — no old habits          long-tenured power
  to unlearn                         users with deep habits

  Rule of thumb: 1-2 weeks          Rule of thumb: 4-8+ weeks
```

### The danger of a fixed default duration

Many teams default to "2-week experiments." This is fine as a minimum floor against day-of-week effects, but it is not a substitute for checking whether the effect has actually stabilised. A fundamental workflow change might not plateau for 6–8 weeks — a 2-week experiment gives you a number in the middle of the decay curve, which is the worst place to measure.

---

## 8. Distinguishing Novelty/Primacy from Other Confounds

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONFOUND 1 — Day-of-week seasonality                              │
│                                                                     │
│  A weekly seasonal pattern in the treatment effect can look        │
│  superficially like novelty decay.                                 │
│                                                                     │
│  Fix: compare same weekdays across weeks                           │
│  (Tuesday week 1 vs Tuesday week 2, not Day 1 vs Day 8)           │
│  before concluding it is a genuine novelty pattern.                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONFOUND 2 — Genuine regression vs. primacy effect                │
│                                                                     │
│  Both look identical in pooled data:                               │
│  early dip in the treatment arm.                                   │
│                                                                     │
│  KEY DIFFERENCE:                                                    │
│  Genuine regression → shows up in NEW users too                    │
│  Primacy effect     → concentrated in EXISTING users only          │
│                                                                     │
│  If new users show a stable negative from day 1 → real problem    │
│  If only long-tenured users dip and new users are fine → primacy  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONFOUND 3 — External event (news, competitor launch, etc.)       │
│                                                                     │
│  If something external happened around day 3 of your experiment,   │
│  it could shift both arms — or shift them asymmetrically.          │
│                                                                     │
│  Fix: compare the trend across both treatment AND control arms.    │
│  A genuine novelty/primacy pattern shows up in the DIFFERENCE      │
│  (treatment minus control). An external event usually shifts       │
│  both arms and leaves the difference more stable.                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. The Peeking Trap in Disguise

A stakeholder sees the early +8% result and wants to "extend just a bit longer to see if it holds." This sounds reasonable. It is not — unless it was pre-specified.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEGITIMATE extension                ILLEGITIMATE extension        │
│  ──────────────────────              ───────────────────────       │
│  Extension criterion was             Extension decided AFTER       │
│  written into the experiment         seeing early results          │
│  design doc BEFORE launch:           because the number looks      │
│                                      appealing and someone         │
│  "If day-by-day lift has             hopes it will hold            │
│   not plateaued by day 7,                                          │
│   extend to day 14."                 This is optional stopping     │
│                                      / p-hacking in disguise.     │
│  The trigger is pre-specified.       The early data influenced     │
│  Doesn't introduce bias.             a decision that should        │
│                                      have been fixed in advance.   │
└─────────────────────────────────────────────────────────────────────┘
```

The principle: the criteria for extending an experiment are part of the experiment design. They belong in the pre-registration doc (Ch. 8). Deciding them reactively reintroduces all the same problems pre-registration was designed to prevent.

---

## 10. The Full Decision Flow

```
  Run experiment
        │
        ▼
  Plot lift by day BEFORE looking at pooled average
        │
        ├── Monotone decay (starts high, falls) → NOVELTY EFFECT suspected
        │         │
        │         ▼
        │   Has it plateaued?
        │   YES → report plateau value, not pooled average
        │   NO  → extend experiment (if pre-specified rule allows)
        │         OR note result as provisional
        │
        ├── Starts low / negative, climbs → PRIMACY EFFECT suspected
        │         │
        │         ▼
        │   Segment by new vs. existing users
        │   Existing users dip, new users flat → confirmed primacy
        │   Both dip → could be genuine regression, investigate
        │
        ├── Flat from day 1 → no novelty/primacy
        │   Pooled average is trustworthy
        │   Proceed to standard ship decision
        │
        └── Rule out seasonality (compare same weekdays)
            Rule out external events (check both arms)
```

---

## 11. Red Flags — Spot the Error

```
┌─────────────────────────────────────────────────────────────────────┐
│  RED FLAGS IN THE WILD                                              │
│                                                                     │
│  ✗ Reporting a single pooled average without showing the           │
│    day-by-day trend. Pooled average is meaningless until           │
│    you know the curve has stabilised.                              │
│                                                                     │
│  ✗ Stopping the experiment after day 2 because results             │
│    "look amazing." Day 2 is the peak of almost every              │
│    novelty curve.                                                  │
│                                                                     │
│  ✗ Killing a feature after day 3 because of a dip,                │
│    without checking if it is concentrated in existing users        │
│    (primacy) vs. new users (genuine regression).                   │
│                                                                     │
│  ✗ Running a 2-week experiment on a fundamental workflow           │
│    redesign and trusting the result as stable. Power ≠            │
│    stabilisation. You can be powered and still mid-curve.          │
│                                                                     │
│  ✗ Concluding "novelty effect" from a day-by-day pattern           │
│    without ruling out day-of-week seasonality first.               │
│                                                                     │
│  ✗ Deciding to extend the experiment after seeing early            │
│    results without a pre-specified extension criterion.            │
│                                                                     │
│  ✓ Always plot lift by day before trusting pooled average          │
│  ✓ Segment new vs. existing users to confirm the diagnosis        │
│  ✓ Pre-specify extension criteria in the design doc               │
│  ✓ Treat power duration and stabilisation duration as separate    │
│    constraints, take the maximum                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. L5-Differentiating Talking Points

Things that separate a strong answer from a great one in the room:

**Naming both effects precisely** — most candidates say "novelty effect" for everything. Naming primacy as a distinct, opposite-signature phenomenon with a different causal story (habit/muscle-memory disruption vs. curiosity) shows genuine depth.

**The two-constraint framing** — explicitly saying "statistical power and effect stabilisation are two separate minimums, and I take the maximum of the two" is not common. It signals practical experience, not just textbook knowledge.

**Segmentation as the regression vs. primacy diagnostic** — saying you'd segment by user tenure to distinguish primacy from a genuine regression demonstrates you've thought about the confound, not just the pattern.

**Proactively flagging seasonality** — before concluding a day-by-day pattern is novelty/primacy, mentioning that you'd rule out day-of-week effects first shows rigour.

**Connecting extension decisions to pre-registration** — noting that the criterion for extending an experiment should be in the design doc before launch, and that deciding to extend after seeing early results is a form of peeking — shows you apply statistical discipline consistently across the experiment lifecycle, not just at the analysis stage.

**Long-running holdout as infrastructure** — framing holdouts as a platform investment rather than a one-off analysis trick shows systems-level thinking.

---

## 13. Flash Cards — Interview Prep

```
Q: What is a novelty effect and what does its time-trend look like?
A: Treatment looks better than it truly is early on because users
   are curious about the change. Time trend: starts high, decays
   to a smaller stable plateau.

Q: What is a primacy effect and what does its time-trend look like?
A: Treatment looks worse than it truly is early on because existing
   users have old habits/muscle memory disrupted. Time trend: starts
   low or negative, climbs to a stable plateau.

Q: What are the two independent duration constraints?
A: (1) Statistical power — enough n to detect MDE at target α.
   (2) Effect stabilisation — long enough for novelty/primacy curve
   to plateau. Run for MAX of the two. Power ≠ stabilisation.

Q: How do you distinguish a primacy effect from a genuine regression?
A: Segment by new vs. existing users. Primacy effect: concentrated
   in existing users (they have habits to unlearn). Genuine regression:
   shows up in new users too (they have nothing to unlearn).

Q: Your test shows +8% on day 1, +1.6% by day 8. What do you report?
A: Report the plateau (~1.6%) as the true steady-state effect — after
   confirming the curve has actually flattened, not still decaying.
   Do not report the pooled average. Do not report the day-1 number.

Q: A -5% dip in first 3 days recovers to neutral by day 10. Ship?
A: Don't kill it — looks like primacy. Confirm by checking: is the
   dip concentrated in existing users? If yes → primacy, let it
   run. If new users also dip → genuine regression, investigate.

Q: When is extending an experiment legitimate vs. p-hacking?
A: Legitimate only if the extension criterion was pre-specified in
   the design doc before launch. Deciding to extend after seeing
   early results is optional stopping — a form of peeking.

Q: Why is plotting by calendar day sometimes misleading?
A: Users enter the experiment on different calendar days. Plotting
   by days-since-first-exposure normalises for staggered entry and
   shows the true per-user adaptation curve.
```

---

## 14. Connections to Other Chapters

| Chapter | Topic | Connection |
|---|---|---|
| Ch. 8 | Pre-registration | Extension criteria must be pre-specified; detecting novelty post-hoc and extending without a rule reintroduces peeking risk |
| Ch. 9 | Power & sample size | Power calculation gives duration constraint 1 — stabilisation gives constraint 2; take the max |
| Ch. 16 | Sequential testing / peeking | Stopping early on a great day-2 result is exactly the peeking problem; novelty makes this worse because day-2 is the peak |
| Ch. 19 | Long-term holdouts | The infrastructure solution to novelty/primacy — holdouts let you observe the true long-run effect well past experiment conclusion |

---

*Last updated: 2026 · Source: Chapter 11 notes + Google/Meta interview patterns*

## 10. Famous Interview Q&A

**Q: Your test shows a huge +10% lift on day 1, but by day 10 it's down to +1%. What's happening, and what would you report as the "true" effect?**
A: A classic novelty effect — the initial spike likely reflects users engaging with the change simply because it's new and different, not because it's genuinely better, and that curiosity-driven behavior fades as users habituate. I'd report the plateaued, steady-state value (the ~1% seen by day 10) as the more trustworthy estimate of the long-run effect, but only after confirming the curve has actually flattened rather than still trending down — I'd want a few more days of stability before locking that number in as the ship-decision basis.

**Q: A redesign shows a -5% dip in the first three days but climbs back to neutral by day 10. Should you kill the feature based on the early negative result?**
A: Not necessarily — this looks like a primacy effect, where existing users are temporarily thrown off by the change and their performance dips before they adapt. Killing the feature based on the first 3 days risks discarding something that could be neutral or even positive in steady state. I'd want to let the test run long enough to see if the metric fully recovers and stabilizes, and specifically check whether the dip is concentrated in long-tenured users (who have old habits to unlearn) versus new users (who'd show no such dip, since they have nothing to "unlearn") — that segmentation would help confirm whether this is truly a primacy effect versus a genuine regression.

**Q: How would you decide how long to run a test to account for novelty effects, rather than just using the sample-size-based duration from a power calculation?**
A: I'd treat these as two separate, additive constraints on duration: the power calculation gives the *minimum* duration needed to reach statistical significance given your MDE and traffic, while the novelty/primacy consideration gives a *separate* minimum duration needed for the treatment effect itself to stabilize. The test should run for at least the longer of the two. In practice, I'd look at day-over-day lift trends during the test (or historical data from similar past launches on this same surface) to estimate how many days it typically takes for novelty effects to plateau, and use that as a floor on duration regardless of what the power calculation alone suggests.

**Q: Why might segmenting your test results by new vs. existing users be more informative than looking at the pooled average when diagnosing novelty effects?**
A: Novelty and primacy effects are fundamentally about *existing users reacting to change* — new users have no prior habits or expectations to be surprised by. If you see a decaying/increasing effect curve concentrated specifically in existing users while new users show a flat, stable effect from day one, that's strong confirmatory evidence you're looking at a genuine novelty/primacy pattern rather than a seasonal confound or an unrelated external factor that would affect both segments similarly.

**Q: A stakeholder wants to extend a promising-looking experiment "just a bit longer to see if the effect holds up" after seeing early results. How is this different from problematic peeking/optional stopping, and what would make it legitimate vs. illegitimate?**
A: It's legitimate only if the possibility of extending — and the specific criterion for doing so (e.g., "if day-by-day lift hasn't plateaued by day 7, extend to day 14") — was **pre-specified in the experiment design doc** before seeing results. If the decision to extend is being made reactively, purely because the current number looks appealing and someone hopes it'll hold up, that's indistinguishable from optional stopping / p-hacking: you're letting the data you've already seen influence a decision that should have been fixed in advance, which inflates the true false-positive rate of your eventual conclusion.

---

## 11. L5-Differentiating Talking Points

- Proactively proposing a day-by-day (or week-by-week) breakdown of the treatment effect as a standard part of your analysis plan, rather than only reporting a pooled number, shows this is baked into your default workflow, not an afterthought.
- Being able to correctly name and distinguish novelty vs. primacy effects and describe their **opposite** time-trend signatures demonstrates precise vocabulary and understanding, not just "effects can change over time."
- Proposing a long-running holdout group as an infrastructure investment (not just a one-off analysis trick) shows systems-level thinking about how a company should structure its experimentation platform to catch this class of issue routinely.
- Flagging the seasonality-confound risk unprompted, before concluding a pattern is genuinely novelty/primacy, shows rigor and prevents a common false-positive interpretation.
- Explicitly framing test duration as **two independent constraints** (statistical power vs. effect stabilization) and taking the max of the two, rather than treating duration as a single power-calculation output, is a strong signal of practical experience.
- Connecting a post-hoc "let's extend the test" instinct back to the pre-registration/peeking principle — insisting the extension criterion be pre-specified — shows you apply statistical rigor consistently, not just when it's convenient.

---

## 12. Comprehension Check (Self-Test)

1. Describe the opposite time-trend signatures of a novelty effect versus a primacy/learning-cost effect.
2. In the onboarding-tutorial worked example, why would reporting a single 4-day pooled effect size have been misleading, and by roughly what factor?
3. Name at least three detection methods for novelty/primacy effects beyond simply plotting the daily treatment effect.
4. Why is it important to rule out day-of-week seasonality before concluding a day-by-day pattern reflects a genuine novelty effect?
5. A stakeholder wants to extend a promising-looking experiment "just a bit longer to see if the effect holds up" after seeing early results. How would you distinguish this from problematic peeking/optional stopping, and what would make this a legitimate versus illegitimate decision?
6. Why is test duration governed by two separate, additive constraints rather than just the sample-size/power calculation? Name both constraints.
7. How does segmenting by new vs. existing users help distinguish a primacy effect from a genuine regression?
8. A redesign shows a -5% dip in the first three days but recovers to neutral by day 10. What would you recommend, and what additional analysis would strengthen your confidence in that recommendation?

---
This is the **silent killer** of A/B tests at scale. 

At Google (Search/YouTube), Meta (News Feed/Reels), and Apple (iOS features), you are almost never dealing with a static user base. Users adapt, get bored, or get excited. If you run a 2-week test and ship based on that, you are often just measuring the **"Shiny Object" effect**—not the long-term value. 

Interviewers love this topic because it separates "people who can run a t-test" from "people who understand human behavior over time." Here is your definitive playbook.

---

### Part 1: The "Cold Call" Opening Question
**Interviewer:** *"You launch a new, highly visible feature on the Homepage. Week 1 shows a massive +8% lift in clicks. Week 2 shows the lift dropping to +2%. Week 3 shows it flat at 0%. The PM wants to ship based on the Week 1 data. What do you tell them?"*

**Your Instant Answer:** 
**"Absolutely not. We are observing a classic Novelty Effect.** Users are clicking on Day 1 simply because it's *new* and visually catches their eye, not because it's inherently more useful. By Week 3, the 'newness' wears off, and they revert to their habitual behavior. We cannot ship until we observe the metric **stabilize** and reach a steady state. We need to run this test for at least 2-3 full business cycles (e.g., 4 weeks) to separate the novelty spike from the true long-term causal effect."

---

### Part 2: The Core Framework (The 4 Phases of User Response)

To get a "Hire" signal, draw this trajectory on the whiteboard. Explain that user behavior over time follows a non-linear curve:

| Phase | Name | What happens | Action |
| :--- | :--- | :--- | :--- |
| **1** | **Novelty Effect** | **Short-term positive.** Users click/interact because it's new, shiny, or surprising. (e.g., A new notification badge). | **Do NOT ship.** This is temporary. |
| **2** | **Primacy Effect** | **Short-term negative.** Users hate the change because it disrupts their muscle memory. They underperform while they "re-learn" the UI. (e.g., Moving the "Search" bar). | **Do NOT ship.** This is temporary. Wait for adaptation. |
| **3** | **Habituation (Steady State)** | **The true causal effect.** Users have gotten used to it. Their behavior is now driven by utility, not novelty or confusion. | **This is the ONLY phase we use to make a ship/no-ship decision.** |
| **4** | **Long-Term Maturation** | **Compound effects.** Does the feature change retention curves over 3-6 months? (e.g., Does it increase Lifetime Value?). | Requires a separate **Holdback / Retrospective Analysis**. |

---

### Part 3: The "Google" Nuance (The "Learning vs. Doing" Trap)

Google interviewers will push you hard on search quality.

**Interviewer:** *"We are testing a new Search algorithm that shows longer, more comprehensive snippets. In Week 1, Click-Through-Rate (CTR) drops by 5% because users are reading the snippet and not clicking. By Week 4, CTR recovers to flat. The PM says: 'See, it recovers, so it's safe to ship.' Do you agree?"*

**Your Advanced Answer:**
"No. The PM is conflating **recovery** with **improvement**. 

- In Week 1, we saw a **Primacy Effect** (users were reading, not clicking).
- In Week 4, we saw **Habituation** (they learned to read faster and resumed clicking). 

But the goal of a Search engine isn't just CTR; it's *user satisfaction*. If they are reading the snippet and *not* clicking, that might actually be a **win** (they found their answer without leaving the page!). 

**My pre-registered hypothesis** would have specifically broken this down:
1. **Short-term (Week 1):** We expected a CTR dip (Primacy).
2. **Long-term (Week 4+):** We expect **Search Exit Rate** (users leaving the search engine entirely) to drop, because they find answers faster. 
If I only measured CTR, I would cancel a great feature. We need to define **behavioral adaptation metrics** *before* the test starts."

---

### Part 4: The "Meta" Nuance (Network Effects & Interference)

Meta loves this because their products are social. They will throw a curveball about time.

**Interviewer:** *"You test a new feature that encourages users to tag more friends in comments. In Week 1, engagement is up 10%. In Week 2, it drops to 2%. Is this strictly a Novelty Effect, or could something else be happening?"*

**Your Answer:**
"It could be **Novelty**, but given this is a *social* feature, it could also be **Network Saturation / Interference**. 

- In Week 1, the treatment users tag their friends. Those friends (who are in the *Control* group) see the tags and log in more, contaminating the control group. 
- By Week 2, the novelty of being tagged wears off, and the network effect stabilizes. 
- The Week 1 lift was a combination of the feature *plus* the exogenous spillover effect on control users. The Week 2 flatline is the true effect.
- **My solution:** We cannot use a standard user-randomized test here. We would need to run a **Cluster-Randomized Test** (randomizing by geographic region or social graph clusters) and run it for at least **4 to 6 weeks** to let the entire social graph reach a new equilibrium before we measure."

---

### Part 5: The "Apple" Nuance (The "Ugly" Primacy Effect)

Apple focuses heavily on OS updates and hardware interactions. They will ask you about the *opposite* of novelty: the "First-Day Hate."

**Interviewer:** *"We push a new iOS keyboard layout. Day 1, typing speed drops 20% and error rates spike 30%. The engineering team wants to revert the change immediately. How do you handle this?"*

**Your Answer:**
"This is a textbook **Primacy Effect** due to muscle memory. We *expect* this disruption. 

**My strategy:**
1. **Don't panic.** I pre-register this expectation. I explicitly state: *'We hypothesize a negative effect in the first 48 hours (learning curve), followed by a recovery to baseline within 7 days.'*
2. **Segmented Analysis:** I look at the data by **User Tenure**. Power users (who type fast) will experience the worst Primacy effect, but they will adapt fastest. New users (who don't have muscle memory yet) might actually show an *immediate* improvement because the new layout is inherently better.
3. **The Decision:** I do NOT look at the overall average for the first 3 days. I only look at the **Day 7 to Day 14** window (the steady state). If typing speed is back to baseline or better by Day 7, we ship. If it's still worse by Day 14, we revert."

---

### Part 6: The "Test Duration" Math Problem

**Interviewer:** *"You calculate that you need 14 days to reach your required sample size. But you know there is a 7-day Novelty effect. How long do you actually run the test?"*

**Your Answer:**
"You don't run it for 14 days; you run it for **21 days (14 + 7)**. 
Crucially, you **discard** the first 7 days of data from your *final analysis*, or you model them out using a **Time-Trend Interaction** (e.g., `Treatment * Day`). 

However, discarding data reduces your power! You need to calculate your sample size based on the *variance* of the steady-state period (Days 8-21), not the full 21-day variance. Usually, this means you need to increase your sample size by ~20-30% to account for the fact that you are throwing away the first week."

---

### Part 7: The "Weekly Seasonality" Trap (The Counter-Argument)

**Interviewer:** *"But wait. What if the drop from Week 1 to Week 2 isn't a Novelty effect? What if it's just weekly seasonality? (e.g., People browse more on weekends than weekdays). How do you prove it's Novelty?"*

**Your Advanced Answer (The "Double-Diff" Method):**
"You are absolutely right. To distinguish **Novelty** from **Seasonality**, I look at the **Control group's trend** over the same period.

- If the Treatment's lift drops from +8% to +2%, but the Control group *also* drops by 6% during that same timeframe (due to seasonality), then **Novelty isn't the culprit**—the treatment lift is actually stable at +2% relative to the control.
- If the Control group is perfectly flat at 0% over the 3 weeks, but the Treatment drops from +8% to 0%, then we have confirmed pure **Novelty**.
- **The Rule:** I always plot the **Daily Treatment-Control Difference** over time. A *decaying* slope in that difference line is the definitive proof of a novelty/primacy effect, regardless of underlying seasonality."

---

### Part 8: The "Long-Term Holdback" (The Ultimate Meta/Google Solution)

To sound like a Senior/Staff DS, you need to explain how you solve this problem *after* the test ends.

**Interviewer:** *"We ship the feature based on a steady 4-week test. A year later, we realize retention dropped. How could we have caught this?"*

**Your Answer:**
"This is why we run **Long-Term Holdback (LTH) experiments** at scale. 

- For 99% of users, we roll out the feature.
- For a randomly selected 1% **Holdback** group, we intentionally *keep the feature off* for 3 to 6 months.
- We compare the **Long-Term Retention** (e.g., 90-Day Active User rate) between the Holdback group and the Treatment group.

Why does this solve Novelty/Primacy? Because by Month 6, the novelty has *completely* worn off for the treatment group. The Holdback group has also adapted to *not* having it. The difference between them at Month 6 is the true, durable, long-term causal effect. If that gap is negative, we revert the feature, even if our 4-week test said it was positive."

---

### Part 9: Summary Cheat Sheet for the Interview

| Interviewer's Trap | Your Response Framework |
| :--- | :--- |
| *"Week 1 looks great, let's ship."* | "Wait for the **Habituation phase**. Discard the first N days." |
| *"Users hate it on Day 1, let's revert."* | "That's the **Primacy Effect**. Let's wait for muscle memory to adapt." |
| *"We hit sample size in 10 days."* | "Run it for **10 Days + Ramp-up period**. Don't use the ramp-up days in the final p-value calculation." |
| *"How do I know if it's novelty or seasonality?"* | "Look at the **Control group's trend** and plot the **daily delta (T - C)**. Only a decaying delta proves novelty." |
| *"How do I stop long-term degradation?"* | "Run a **1% Long-Term Holdback** group for 3–6 months post-launch." |
| *"What's the minimum test duration?"* | **1 full business cycle** (e.g., 2 full weeks) **+** the estimated **adaptation period** (usually 3-7 days). Never less than 14 days." |


# Chapter 11 — Novelty & Primacy Effects + Test Duration
> **Interview tier:** Google / Meta / Apple DS / PM rounds  
> **Core risk this prevents:** shipping on a transient signal, killing a genuinely good feature too early  
> **Key principle:** the effect you measure in week 1 may be a reaction to *change itself* — not the change being better

---

## 1. The One-Line Mental Model

> You are not measuring whether the feature is good.  
> You are measuring whether users have **finished reacting to the fact that something changed.**

Until that reaction settles, your metric is lying to you — in either direction.

---

## 2. The Two Effects — Definitions

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  NOVELTY EFFECT                   PRIMACY EFFECT                   │
│  ──────────────                   ─────────────                    │
│  Treatment looks BETTER           Treatment looks WORSE            │
│  than it truly is — early on      than it truly is — early on     │
│                                                                     │
│  Why: users are curious,          Why: existing users have         │
│  excited, exploring the           muscle memory / habits           │
│  new thing just because           baked into the OLD UI.           │
│  it is new.                       Change = friction, confusion,    │
│                                   slowing down — temporarily.      │
│                                                                     │
│  Time trend:                      Time trend:                      │
│  Starts HIGH → decays             Starts LOW → climbs              │
│  to true steady state             to true steady state             │
│                                                                     │
│  Risk if you stop early:          Risk if you stop early:          │
│  Ship something that will         KILL something that would        │
│  underperform its promise         have been genuinely good         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Both are threats to validity for the same reason: you are measuring a **transient reaction to change**, not the durable effect you care about for a permanent launch.

---

## 3. The Restaurant Analogies

These are the clearest intuition pumps for interviews.

```
  NOVELTY EFFECT — the new dish
  ──────────────────────────────
  Restaurant adds a new dish to the menu.
  Week 1: regulars order it out of curiosity.
  Sales spike. Looks like a hit.

  Month 2: novelty wears off.
  Regulars go back to their usual order.
  Sales settle at a much lower level.

  The dish wasn't better. It was just new.
  If you'd hired 3 extra chefs based on week 1 data → mistake.


  PRIMACY EFFECT — the new menu layout
  ──────────────────────────────────────
  Restaurant rearranges the entire menu.
  Week 1: regulars can't find their usual dish.
  Frustrated, they order the first familiar thing they see.
  Sales of the "better" featured item look WORSE.

  Month 2: regulars have learned the new layout.
  The featured item is now easier to find.
  Sales recover and then exceed the old baseline.

  The layout WAS better. It just looked broken for a while.
  If you'd reverted based on week 1 data → wasted a good change.
```

---

## 4. Time-Trend Signatures — What to Look For

The single most important diagnostic: **plot lift by day, not just the pooled average.**

```
  NOVELTY EFFECT SIGNATURE          PRIMACY EFFECT SIGNATURE
  ────────────────────────          ────────────────────────

  Lift                              Lift
   │                                 │              ┌────────
   │ *                               │             /
   │  *                              │            /
   │   **                            │           /
   │     ***                         │          /
   │        ****                     │─────────/
   │            ─────────            │ *  *  *
   │                                 │
   └──────────────────── Day         └──────────────────── Day

  Starts HIGH, decays to plateau    Starts LOW (or negative),
                                    climbs to plateau

  Correct read: use the plateau     Correct read: use the plateau
  NOT the early spike               NOT the early dip
```

If you use the pooled average across the whole experiment window, you blend the transient spike/dip with the true signal and get a number that accurately represents neither.

---

## 5. Worked Examples

### Example A — Onboarding tutorial (novelty effect)

Daily treatment effect on Day-1 retention:

```
  ┌─────────────────┬────────────────────────────────────────────┐
  │  Experiment day │  Treatment effect  │  What's happening     │
  ├─────────────────┼────────────────────┼───────────────────────┤
  │   Day 1–2       │     +4.5pp         │  Users excited,       │
  │   Day 3–4       │     +3.1pp         │  exploring, clicking  │
  │   Day 5–7       │     +2.0pp         │  Novelty fading       │
  │   Day 8–10      │     +1.6pp         │  Settling             │
  │   Day 11–14     │     +1.4pp         │  Nearly stable        │
  │   Day 15–21     │     +1.3pp         │  True steady state    │
  └─────────────────┴────────────────────┴───────────────────────┘

  Pooled 4-day average:  ~+3.8pp   ← what you'd report if you stopped early
  True steady-state:      ~+1.3pp   ← what actually persists

  Overestimate factor: ~3x
```

The effect IS real — it doesn't decay to zero. But reporting +3.8pp for a ship decision would set wildly unrealistic expectations. Every downstream model (revenue projections, headcount planning) built on that number would be wrong by 3x.

### Example B — Navigation redesign (novelty + primacy comparison)

```
  ┌──────┬────────────────────────────────────────────────────────┐
  │  Day │  Nav redesign lift  │  Interpretation                  │
  ├──────┼─────────────────────┼──────────────────────────────────┤
  │   1  │      +8.5%          │  Users actively exploring        │
  │   2  │      +6.0%          │  new layout — novelty peak       │
  │   3  │      +4.5%          │  Decay begins                    │
  │   4  │      +3.0%          │                                  │
  │   5  │      +2.2%          │  Rapid decay                     │
  │   6  │      +1.8%          │                                  │
  │   7  │      +1.7%          │  Plateau begins                  │
  │   8  │      +1.6%          │  ← Stable region                 │
  │   9  │      +1.7%          │  ← True steady-state ~1.6–1.7%  │
  │  10  │      +1.6%          │                                  │
  └──────┴─────────────────────┴──────────────────────────────────┘

  2-day pooled average:  ~7.25%    ← 4.5x overestimate
  True steady-state:      ~1.6%    ← what to report and ship on
```

Stopping after 2 days here doesn't just inflate your number — it makes the feature look like a breakthrough when it's actually a moderate improvement. Stakeholder expectations get miscalibrated for months.

---

## 6. Detection Methods

```
┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 1 — Day-by-day lift plot (always do this first)            │
│                                                                     │
│  Compute treatment effect separately for each day (or week).       │
│  Look for the curve shape: decaying → novelty, climbing → primacy  │
│  Only trust the pooled number AFTER you've confirmed a plateau.    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 2 — New vs. existing user segmentation (the key diagnostic)│
│                                                                     │
│  Novelty/primacy are fundamentally about EXISTING users reacting   │
│  to change. New users have no old habits — nothing to unlearn.     │
│                                                                     │
│  Existing users: volatile early effect that stabilises over time   │
│  New users:      flat, stable effect from day one                  │
│                                                                     │
│  If new users show a stable negative effect from day 1             │
│  → this is a REAL regression, not a primacy effect                 │
│  → primacy effect would only show in existing users                │
│                                                                     │
│  This is the only reliable way to distinguish primacy from         │
│  a genuine regression. Both look identical in pooled data.         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 3 — Days-since-first-exposure cohort analysis              │
│                                                                     │
│  Instead of plotting by calendar day, plot by days since each      │
│  user first saw the treatment. Users enter the experiment on        │
│  different days — calendar-day plots blur their adaptation curves  │
│  together. This normalises for staggered entry.                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 4 — Long-running holdout group                             │
│                                                                     │
│  A small permanent holdout (~1-5% of traffic) that never gets      │
│  treatment, maintained well past experiment conclusion.            │
│  Lets you compare treatment vs. control weeks or months later,     │
│  long after any novelty/primacy dynamics have faded.               │
│  Infrastructure investment — not a one-off trick.                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  METHOD 5 — Fresh re-exposure experiment (novelty only)            │
│                                                                     │
│  Expose a fresh randomised sample to the same treatment AFTER      │
│  the initial novelty period has plausibly worn off in the          │
│  broader population. If the effect is smaller for the fresh        │
│  sample, the original lift was partly novelty-driven.              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Test Duration — Two Independent Constraints

This is the most commonly missed insight in interviews. Duration is not one number — it is the **maximum of two separate minimums**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  CONSTRAINT 1                     CONSTRAINT 2                     │
│  Statistical power                Effect stabilisation             │
│  ──────────────────               ─────────────────────            │
│  How long until you have          How long until the novelty/      │
│  enough users to detect           primacy curve has plateaued      │
│  your MDE at target α             and the effect is stable         │
│  and power?                       enough to trust?                 │
│                                                                     │
│  Comes from the power             Comes from day-by-day trend      │
│  calculation (Ch. 9)              analysis or historical data      │
│                                   from similar past launches       │
│                                                                     │
│  Run for:  MAX(constraint 1, constraint 2)                         │
│                                                                     │
│  Common mistake: running until constraint 1 is met,                │
│  declaring the experiment done, and never checking constraint 2    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What affects stabilisation time

```
  FASTER STABILISATION              SLOWER STABILISATION
  ────────────────────              ────────────────────
  Subtle change (small copy         Large visual redesign
  tweak, minor layout shift)        (full page overhaul)

  High exposure frequency           Low exposure frequency
  (daily-use feature like           (annual billing flow,
   a home feed)                      account settings)

  Younger product / mostly          Mature product with
  new users — no old habits          long-tenured power
  to unlearn                         users with deep habits

  Rule of thumb: 1-2 weeks          Rule of thumb: 4-8+ weeks
```

### The danger of a fixed default duration

Many teams default to "2-week experiments." This is fine as a minimum floor against day-of-week effects, but it is not a substitute for checking whether the effect has actually stabilised. A fundamental workflow change might not plateau for 6–8 weeks — a 2-week experiment gives you a number in the middle of the decay curve, which is the worst place to measure.

---

## 8. Distinguishing Novelty/Primacy from Other Confounds

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONFOUND 1 — Day-of-week seasonality                              │
│                                                                     │
│  A weekly seasonal pattern in the treatment effect can look        │
│  superficially like novelty decay.                                 │
│                                                                     │
│  Fix: compare same weekdays across weeks                           │
│  (Tuesday week 1 vs Tuesday week 2, not Day 1 vs Day 8)           │
│  before concluding it is a genuine novelty pattern.                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONFOUND 2 — Genuine regression vs. primacy effect                │
│                                                                     │
│  Both look identical in pooled data:                               │
│  early dip in the treatment arm.                                   │
│                                                                     │
│  KEY DIFFERENCE:                                                    │
│  Genuine regression → shows up in NEW users too                    │
│  Primacy effect     → concentrated in EXISTING users only          │
│                                                                     │
│  If new users show a stable negative from day 1 → real problem    │
│  If only long-tenured users dip and new users are fine → primacy  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONFOUND 3 — External event (news, competitor launch, etc.)       │
│                                                                     │
│  If something external happened around day 3 of your experiment,   │
│  it could shift both arms — or shift them asymmetrically.          │
│                                                                     │
│  Fix: compare the trend across both treatment AND control arms.    │
│  A genuine novelty/primacy pattern shows up in the DIFFERENCE      │
│  (treatment minus control). An external event usually shifts       │
│  both arms and leaves the difference more stable.                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. The Peeking Trap in Disguise

A stakeholder sees the early +8% result and wants to "extend just a bit longer to see if it holds." This sounds reasonable. It is not — unless it was pre-specified.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEGITIMATE extension                ILLEGITIMATE extension        │
│  ──────────────────────              ───────────────────────       │
│  Extension criterion was             Extension decided AFTER       │
│  written into the experiment         seeing early results          │
│  design doc BEFORE launch:           because the number looks      │
│                                      appealing and someone         │
│  "If day-by-day lift has             hopes it will hold            │
│   not plateaued by day 7,                                          │
│   extend to day 14."                 This is optional stopping     │
│                                      / p-hacking in disguise.     │
│  The trigger is pre-specified.       The early data influenced     │
│  Doesn't introduce bias.             a decision that should        │
│                                      have been fixed in advance.   │
└─────────────────────────────────────────────────────────────────────┘
```

The principle: the criteria for extending an experiment are part of the experiment design. They belong in the pre-registration doc (Ch. 8). Deciding them reactively reintroduces all the same problems pre-registration was designed to prevent.

---

## 10. The Full Decision Flow

```
  Run experiment
        │
        ▼
  Plot lift by day BEFORE looking at pooled average
        │
        ├── Monotone decay (starts high, falls) → NOVELTY EFFECT suspected
        │         │
        │         ▼
        │   Has it plateaued?
        │   YES → report plateau value, not pooled average
        │   NO  → extend experiment (if pre-specified rule allows)
        │         OR note result as provisional
        │
        ├── Starts low / negative, climbs → PRIMACY EFFECT suspected
        │         │
        │         ▼
        │   Segment by new vs. existing users
        │   Existing users dip, new users flat → confirmed primacy
        │   Both dip → could be genuine regression, investigate
        │
        ├── Flat from day 1 → no novelty/primacy
        │   Pooled average is trustworthy
        │   Proceed to standard ship decision
        │
        └── Rule out seasonality (compare same weekdays)
            Rule out external events (check both arms)
```

---

## 11. Red Flags — Spot the Error

```
┌─────────────────────────────────────────────────────────────────────┐
│  RED FLAGS IN THE WILD                                              │
│                                                                     │
│  ✗ Reporting a single pooled average without showing the           │
│    day-by-day trend. Pooled average is meaningless until           │
│    you know the curve has stabilised.                              │
│                                                                     │
│  ✗ Stopping the experiment after day 2 because results             │
│    "look amazing." Day 2 is the peak of almost every              │
│    novelty curve.                                                  │
│                                                                     │
│  ✗ Killing a feature after day 3 because of a dip,                │
│    without checking if it is concentrated in existing users        │
│    (primacy) vs. new users (genuine regression).                   │
│                                                                     │
│  ✗ Running a 2-week experiment on a fundamental workflow           │
│    redesign and trusting the result as stable. Power ≠            │
│    stabilisation. You can be powered and still mid-curve.          │
│                                                                     │
│  ✗ Concluding "novelty effect" from a day-by-day pattern           │
│    without ruling out day-of-week seasonality first.               │
│                                                                     │
│  ✗ Deciding to extend the experiment after seeing early            │
│    results without a pre-specified extension criterion.            │
│                                                                     │
│  ✓ Always plot lift by day before trusting pooled average          │
│  ✓ Segment new vs. existing users to confirm the diagnosis        │
│  ✓ Pre-specify extension criteria in the design doc               │
│  ✓ Treat power duration and stabilisation duration as separate    │
│    constraints, take the maximum                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. L5-Differentiating Talking Points

Things that separate a strong answer from a great one in the room:

**Naming both effects precisely** — most candidates say "novelty effect" for everything. Naming primacy as a distinct, opposite-signature phenomenon with a different causal story (habit/muscle-memory disruption vs. curiosity) shows genuine depth.

**The two-constraint framing** — explicitly saying "statistical power and effect stabilisation are two separate minimums, and I take the maximum of the two" is not common. It signals practical experience, not just textbook knowledge.

**Segmentation as the regression vs. primacy diagnostic** — saying you'd segment by user tenure to distinguish primacy from a genuine regression demonstrates you've thought about the confound, not just the pattern.

**Proactively flagging seasonality** — before concluding a day-by-day pattern is novelty/primacy, mentioning that you'd rule out day-of-week effects first shows rigour.

**Connecting extension decisions to pre-registration** — noting that the criterion for extending an experiment should be in the design doc before launch, and that deciding to extend after seeing early results is a form of peeking — shows you apply statistical discipline consistently across the experiment lifecycle, not just at the analysis stage.

**Long-running holdout as infrastructure** — framing holdouts as a platform investment rather than a one-off analysis trick shows systems-level thinking.

---

## 13. Flash Cards — Interview Prep

```
Q: What is a novelty effect and what does its time-trend look like?
A: Treatment looks better than it truly is early on because users
   are curious about the change. Time trend: starts high, decays
   to a smaller stable plateau.

Q: What is a primacy effect and what does its time-trend look like?
A: Treatment looks worse than it truly is early on because existing
   users have old habits/muscle memory disrupted. Time trend: starts
   low or negative, climbs to a stable plateau.

Q: What are the two independent duration constraints?
A: (1) Statistical power — enough n to detect MDE at target α.
   (2) Effect stabilisation — long enough for novelty/primacy curve
   to plateau. Run for MAX of the two. Power ≠ stabilisation.

Q: How do you distinguish a primacy effect from a genuine regression?
A: Segment by new vs. existing users. Primacy effect: concentrated
   in existing users (they have habits to unlearn). Genuine regression:
   shows up in new users too (they have nothing to unlearn).

Q: Your test shows +8% on day 1, +1.6% by day 8. What do you report?
A: Report the plateau (~1.6%) as the true steady-state effect — after
   confirming the curve has actually flattened, not still decaying.
   Do not report the pooled average. Do not report the day-1 number.

Q: A -5% dip in first 3 days recovers to neutral by day 10. Ship?
A: Don't kill it — looks like primacy. Confirm by checking: is the
   dip concentrated in existing users? If yes → primacy, let it
   run. If new users also dip → genuine regression, investigate.

Q: When is extending an experiment legitimate vs. p-hacking?
A: Legitimate only if the extension criterion was pre-specified in
   the design doc before launch. Deciding to extend after seeing
   early results is optional stopping — a form of peeking.

Q: Why is plotting by calendar day sometimes misleading?
A: Users enter the experiment on different calendar days. Plotting
   by days-since-first-exposure normalises for staggered entry and
   shows the true per-user adaptation curve.
```

---

## 14. Connections to Other Chapters

| Chapter | Topic | Connection |
|---|---|---|
| Ch. 8 | Pre-registration | Extension criteria must be pre-specified; detecting novelty post-hoc and extending without a rule reintroduces peeking risk |
| Ch. 9 | Power & sample size | Power calculation gives duration constraint 1 — stabilisation gives constraint 2; take the max |
| Ch. 16 | Sequential testing / peeking | Stopping early on a great day-2 result is exactly the peeking problem; novelty makes this worse because day-2 is the peak |
| Ch. 19 | Long-term holdouts | The infrastructure solution to novelty/primacy — holdouts let you observe the true long-run effect well past experiment conclusion |

---

*Last updated: 2026 · Source: Chapter 11 notes + Google/Meta interview patterns*
