# Chapter 18 (Rebuilt for Clarity): Heterogeneous Treatment Effects & Simpson's Paradox

Same rebuild approach as the previous chapters: intuition and diagrams first, formula second, a diagnosis section for telling real HTE apart from noise, a step-by-step plug-in walkthrough for the worked example, and a dedicated why/why-not section on when segmenting is safe versus dangerous.

---

## 1. Start Here: The Averaging Problem, in One Picture

```
   Overall result: "+0.5% lift, not significant. Meh feature."

   But what's actually happening underneath:

   NEW USERS           EXISTING USERS
   ┌─────────┐         ┌─────────┐
   │  +4.2%  │         │  -0.8%  │
   │  strong │         │  real   │
   │  WIN    │         │  HARM   │
   └─────────┘         └─────────┘
        │                    │
        └────── averaged ────┘
                  │
                  ▼
            "+0.5%, meh"

   The average didn't just hide the story — it ACTIVELY LIED
   about what's happening to every single person in the
   experiment. Nobody actually experienced "+0.5%." Every
   new user got a real win; every existing user got real harm.
```

**This is the entire chapter's motivating problem**: an average is a single number standing in for potentially many different, even opposite, realities. The average treatment effect (ATE) is only the whole story when the effect is genuinely the same for everyone — which is rarely a safe thing to assume without checking.

---

## 2. Simpson's Paradox — The Extreme, Counterintuitive Version

HTE (Section 1) is "the effects differ by group." Simpson's Paradox is the much stranger cousin: **the aggregate doesn't just average away the story — it reverses direction entirely, even though every single subgroup agrees with each other.**

```
   Drug trial. Look at each age group SEPARATELY:

   YOUNG PATIENTS                OLD PATIENTS
   Drug:    80% recover          Drug:    30% recover
   No drug: 70% recover          No drug: 20% recover
        ↑                             ↑
   Drug HELPS (+10pts)          Drug HELPS (+10pts)

   Drug helps in BOTH groups. Consistently. No disagreement
   anywhere. Now combine them into one number:

   COMBINED (naive pooling)
   Drug:    50% recover      ←  drug group happened to have
   No drug: 55% recover         mostly OLD patients (low baseline)
        ↑
   Drug appears to HURT overall (-5pts)!
```

**Why this happens**: it's not a math error and it's not a contradiction — it's because the *mix* of patients getting the drug versus not getting it differs between age groups. If most of the "drug" group happens to be old patients (who recover less often no matter what), and most of the "no drug" group happens to be young patients (who recover more often no matter what), the group *composition* difference — not the drug itself — dominates the combined number.

```
   THE MECHANICAL CAUSE, IN ONE PICTURE:

   Something (age) is correlated with BOTH:
      (a) which group you're compared against (drug vs. no drug)
      (b) the outcome itself (recovery rate)

   That "something" is a CONFOUNDER for the combined analysis,
   even though it's NOT a confounder within each separate
   age group (since age is held constant there).
```

---

## 3. Why Proper Randomization Mostly Protects You — And Where It Doesn't

```
              Did you segment using a PRE-TREATMENT
              characteristic (tenure, platform, geography —
              known before treatment was assigned)?
                            │
              ┌──────Yes────┴────No─────────────────────┐
              ▼                                            ▼
     SAFE. Under proper randomization,           DANGEROUS. This variable
     treatment and control should have            might itself have been
     similar mixes of this characteristic         CHANGED by treatment —
     by design — Simpson's-Paradox-style           e.g., "did the user open
     mix differences are unlikely to arise         the notification" could
     from randomization itself.                    have different rates in
                                                     treatment vs. control,
                                                     for reasons connected
                                                     to the outcome itself.
                                                             │
                                                             ▼
                                                    Comparing these groups
                                                    reintroduces exactly
                                                    the confounding/selection
                                                    bias randomization was
                                                    supposed to eliminate.
```

**The one-sentence version**: segmenting by something that existed *before* treatment touched anyone is safe. Segmenting by something treatment could have influenced is where Simpson's-Paradox-style traps sneak back in, even inside an otherwise clean randomized experiment.

**Concrete example of the dangerous version**: comparing "users who opened the redesigned notification" (treatment) to "users who opened the old notification" (control) sounds like a fair comparison, but the *set of people* who open a notification can itself be changed by the redesign — maybe the new design is opened disproportionately by a different kind of user than the old one was. Now you're not comparing treatment vs. control on equal footing anymore; you're comparing two self-selected, possibly very different groups.

---

## 4. Diagnosis: Is This Real HTE, or Just Noise?

This is the part that's easy to get wrong under pressure — a common trap is "subgroup A is significant, subgroup B isn't, so the effects must differ." That's not actually a valid inference on its own.

```
              Was this subgroup comparison
              PRE-SPECIFIED before the experiment ran?
                            │
              ┌──────Yes────┴────No──────────────────┐
              ▼                                        ▼
     Proceed to formal testing below.        Treat any finding as
                                              EXPLORATORY/hypothesis-
                                              generating only. Don't
                                              make a ship decision on
                                              it. Recommend a proper
                                              follow-up test, and if you
                                              looked at many subgroups,
                                              apply multiple-testing
                                              correction (Ch16 callback)
                                              before trusting any single
                                              one.

                            (continuing from Yes)
                            │
                            ▼
              Are you comparing subgroup effects by just
              eyeballing "one p-value is under 0.05, the
              other isn't"?
                            │
              ┌──────Yes────┴────No, I ran a formal test──┐
              ▼                                              ▼
     STOP — this is NOT a valid way to conclude          Good. A significant
     effects differ. Smaller subgroups have LESS         Treatment × Segment
     power, so "one crossed the line, one didn't"         interaction term
     is a common false pattern, not evidence of a         formally confirms
     genuine interaction. Run a formal interaction        the effects differ
     test instead (Section 5).                             — proceed to
                                                             Section 6 for
                                                             what to DO
                                                             about it.
```

---

## 5. How to Actually Test for HTE — Plug-In Walkthrough

**The wrong way** (common but invalid): "Segment A has p=0.001 (significant), Segment B has p=0.02 (also significant but different sign) → the effects clearly differ."

This FEELS right but isn't formally valid on its own — you need to directly test whether the *difference between the two subgroup effects* is itself statistically meaningful, not just eyeball two separate p-values.

**The right way — an interaction term**, conceptually:

```
    Set up a regression:

    Outcome = β₀ + β₁×(Treatment) + β₂×(Segment) + β₃×(Treatment × Segment) + error

    β₃ IS THE ANSWER YOU ACTUALLY WANT.

    β₁ alone would tell you the average treatment effect for
    the reference segment. β₃ tells you: "how much MORE (or
    less) is the treatment effect in one segment compared to
    the other?" If β₃ is statistically significant, THAT is
    your formal evidence the effects genuinely differ —
    not the fact that two separate p-values landed on
    different sides of 0.05.
```

**Why subgroup analysis needs MORE sample size than you'd expect**: detecting a main effect needs enough power to see a shift in the mean. Detecting an *interaction* (does the effect differ between groups) is a harder, subtler signal — you're not just asking "did anything move," you're asking "did it move by different amounts in different places," which typically requires a considerably larger sample than the main-effect test alone. This is a commonly underestimated planning requirement — if you know upfront you want to test for HTE, you often need to size the experiment for the interaction test, not just the main effect.

---

## 6. Full Worked Example, Step by Step

Overall result: +0.5% lift, not significant. Pre-specified subgroup breakdown by tenure:

| Segment | n | Conversion lift | p-value |
|---|---|---|---|
| New users (<30 days) | 40,000 | +4.2% | p=0.001 |
| Existing users (30+ days) | 160,000 | -0.8% | p=0.02 |

```
STEP 1 — Sanity-check that the weighted average roughly
         matches the reported overall number (this is just
         verifying the math, not the main point):

    weight(new)      = 40,000/200,000 = 0.2
    weight(existing) = 160,000/200,000 = 0.8

    weighted average = 0.2×(+4.2%) + 0.8×(-0.8%)
                      = 0.84% − 0.64%
                      = 0.20%
                      ≈ matches the reported ~+0.5% overall
                        (illustrative rounding)

STEP 2 — Notice what the overall number OBSCURED:
    Both subgroup findings are individually significant
    (p=0.001 and p=0.02) and point in OPPOSITE directions.
    The "meh, +0.5%, not significant" headline number is
    true as a pure average, but describes literally nobody's
    actual experience.

STEP 3 — Check: was this pre-specified?
    Yes (stated in the setup) → proceed to a formal
    interaction test, don't just eyeball the two p-values.

STEP 4 — Assuming the interaction test confirms the segments
         genuinely differ (β₃ significant), the ACTIONABLE
         conclusion isn't "ship" or "kill" based on the
         overall +0.5% — it's:

    "Ship to new users only. Exclude existing users."
                    │
                    ▼
    This requires a plausible MECHANISM, not just a
    statistical pattern: new users likely benefit from
    simplified onboarding (nothing to unlearn), while
    existing users experience friction from a familiar
    flow suddenly changing underneath them. The mechanism
    is what turns "the numbers differ" into a defensible,
    actionable business recommendation.
```

---

## 7. Why NOT to Trust a Subgroup Finding — The Failure Modes

```
FAILURE 1 — Segmenting by a post-treatment variable
   ("users who engaged with the new feature" vs. those who
   didn't) — this reintroduces exactly the confounding that
   randomization was supposed to remove, because the SET of
   engaged users can differ, in outcome-relevant ways,
   between treatment and control.

FAILURE 2 — Eyeballing two p-values instead of running an
            interaction test
   "One's significant, one isn't" is NOT the same as "the
   effects are formally different" — subgroups have less
   power, so this pattern shows up by chance often.

FAILURE 3 — Fishing through many post-hoc subgroups
   Slicing by 20 different segments and reporting whichever
   one looks most dramatic is a Chapter 16 multiple-testing
   problem wearing a different hat — apply correction, or
   at minimum, label it exploratory only.

FAILURE 4 — Reporting a real HTE finding with no mechanism
            and no action
   "The effect differs by segment" alone isn't a complete
   answer. The stronger answer connects the pattern to a
   plausible WHY and a concrete next step (target the launch,
   iterate for the harmed segment, etc.).

FAILURE 5 — Assuming Simpson's Paradox can't happen in a
            randomized experiment
   It's LESS likely from randomization itself (which should
   balance pre-treatment characteristics across arms by
   design), but it can still creep back in via post-treatment
   segmentation (Failure 1) — "we randomized properly" doesn't
   fully immunize you if your SEGMENTING variable was
   contaminated by treatment.
```

---

## 8. Q&A

**Q: Your experiment shows no significant overall effect, but you suspect the treatment might help new users while hurting existing users. How would you investigate this properly?**
A: I'd check whether this subgroup comparison was pre-specified before running the test — if so, I'd run a formal interaction test (Treatment × Segment term in a regression) rather than just comparing p-values within each subgroup separately, since subgroups naturally have less power and eyeballing "one is significant, one isn't" doesn't itself prove the effects differ. If this wasn't pre-specified, I'd still investigate it, but explicitly frame it as exploratory/hypothesis-generating, and recommend a follow-up experiment specifically designed and powered to test this interaction, rather than making a ship decision on an unplanned post-hoc subgroup finding.

**Q: Explain how Simpson's Paradox could occur even in a properly randomized experiment.**
A: In a properly randomized experiment, treatment and control should have similar subgroup composition by design — so classic Simpson's Paradox (driven by differing group mixes) is less likely to arise from the randomization itself. It's more likely to show up if you segment post-hoc using a variable that's affected by treatment (e.g., "engaged with the new feature vs. didn't") — this segmenting variable isn't a fixed pre-treatment characteristic, so its distribution can differ meaningfully between arms in ways correlated with the outcome, reintroducing exactly the kind of confounding that randomization was supposed to eliminate. This is why segmenting only by pre-treatment characteristics is the safe practice.

**Q: A subgroup analysis shows the treatment effect is +5% for iOS users and -3% for Android users, both individually significant. What would you check before recommending a platform-specific launch?**
A: First, I'd verify this was either pre-specified or, if discovered post-hoc, run a formal interaction test to confirm the difference between platforms is itself statistically significant (not just that one subgroup happens to cross 0.05 and the other doesn't — smaller subgroup sizes make this a common false pattern). I'd also look for a plausible mechanism — is there something about the iOS vs. Android implementation, UI rendering, or user base composition that would sensibly explain opposite-direction effects? If the interaction is confirmed and there's a sensible mechanism, a platform-specific launch (ship on iOS, hold or iterate on Android) would be a strong, defensible recommendation rather than defaulting to the muddled overall average.

**Q: Why is segmenting by a post-treatment variable (like "did the user open the notification") considered dangerous for causal claims, even within a randomized experiment?**
A: Because whether a user opens a notification is itself potentially influenced by treatment (e.g., a redesigned notification might have different open rates than the old one), so comparing "openers" between treatment and control isn't comparing like-for-like groups anymore — the set of people who open notifications in the treatment arm may be systematically different (in ways related to the outcome) from the set who open them in the control arm. This reintroduces selection bias into what was otherwise a clean, randomized comparison — segment only by variables fixed before treatment assignment to preserve the causal validity that randomization was designed to provide.

**Q: Why does testing for HTE typically require a larger sample size than testing for the main effect alone?**
A: Because an interaction effect (does the treatment effect differ between groups) is a subtler signal than a main effect (did the average move at all) — you're effectively trying to detect a difference-of-differences, which has more inherent noise relative to its size than a single average shift. If you know in advance that HTE detection matters for your experiment, you generally need to size the test for the interaction specifically, not just borrow the main-effect power calculation and assume subgroup analysis will "come along for free."

---

## 9. Comprehension Check

1. Using Section 2's drug-trial picture, explain in your own words why the combined number can reverse direction even though both age groups individually show the drug helping.
2. Using Section 3's flowchart, classify each of the following as safe or dangerous to segment by: (a) account creation date, (b) whether the user clicked the new button, (c) signup country, (d) whether the user's session length increased.
3. A colleague says "Segment A has p=0.03, Segment B has p=0.09, so the treatment clearly works better for Segment A." What's wrong with this reasoning, and what should they do instead?
4. Redo Section 6's Step 1 weighted-average check with new numbers: Segment 1 has n=25,000 and lift +6%, Segment 2 has n=75,000 and lift -1%. What's the weighted overall lift?
5. Why is "the effect differs by segment" an incomplete answer on its own, and what two things does a stronger answer add?
6. Explain why Simpson's Paradox is less likely to arise purely from proper randomization, but can still appear via a different route within the same experiment.

---
*This is a clarity-focused rebuild of Chapter 18, restructured around the averaging picture (Section 1) and the drug-trial Simpson's Paradox picture (Section 2) before any formula, a diagnosis flowchart separating real HTE from noise (Section 4), a plug-in walkthrough of the interaction-term logic (Section 5), and a dedicated failure-modes section (Section 7) covering post-treatment segmentation, p-value eyeballing, and unpowered interaction tests. All original formulas, worked numbers, and Q&A are preserved.*
