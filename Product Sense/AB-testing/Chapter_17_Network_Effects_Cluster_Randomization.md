# Chapter 17 (Rebuilt for Clarity): Sizing Cluster-Randomized & Switchback Tests (Design Effect)

Same rebuild approach as Ch13/14/16: intuition and diagrams first, formula second, a diagnosis section for what your real constraint is, a step-by-step plug-in walkthrough, and a dedicated why/why-not section — here framed as "why does a tiny ICC blow up your sample size, and what are your actual options when it does."

---

## 1. Start Here: The Illusion of a Big Sample

```
   "We have 50,000 riders across 10 cities. Huge sample!"

   Looks like this:                    Is actually closer to this:

   ● ● ● ● ● ● ● ● ● ● ● ● ● ●          ●          ●          ●
   ● ● ● ● ● ● ● ● ● ● ● ● ● ●    →     (city 1)   (city 2)   (city 3)
   ● ● ● ● ● ● ● ● ● ● ● ● ● ●          ...
   50,000 independent dots              10 "super-dots" — because
                                         riders in the SAME city all
                                         experienced the same local
                                         market conditions, same
                                         driver supply, same local
                                         pricing dynamics
```

If every rider in a city behaves similarly to every other rider in that same city (because they share the same local market), then knowing what one rider did tells you a lot about what the others in that city probably did too. You didn't really collect 50,000 *independent* pieces of information — you collected something closer to 10, dressed up as 50,000.

**This is the entire chapter.** Everything below is just making that picture precise enough to plug numbers into.

---

## 2. The Two Ingredients That Determine How Bad the Illusion Is

```
   INGREDIENT 1: Intra-Cluster Correlation (ICC, or ρ)
   "How similar are units WITHIN the same cluster to each other?"

        ρ = 0                                    ρ = 1
   ●     ●     ●     ●                    ● ● ● ● ● ● ● ●
   (riders in a city are              (riders in a city are
    totally unrelated to               PERFECTLY identical —
    each other — no redundancy)        100% redundant with
                                        each other)


   INGREDIENT 2: Cluster size (m)
   "How many units are packed into each cluster?"

        m = small (e.g., 10/city)        m = large (e.g., 5,000/city)
   The redundancy problem barely      The redundancy problem compounds
   has room to compound.              A LOT — many units all sharing
                                        the same "similarity tax."
```

**The key insight, before any formula**: ICC alone doesn't tell you how bad your problem is — it's ICC **combined with** cluster size. A tiny ICC can still be a disaster if your clusters are huge, because the redundancy "tax" gets paid by every single one of the hundreds or thousands of units inside that cluster.

---

## 3. Building the Design Effect Formula From the Picture

```
STEP 1 — If ρ=0 (no redundancy at all), your cluster of m units
         gives you exactly m independent pieces of information.
         No penalty. DEFF = 1.

STEP 2 — If ρ=1 (total redundancy), your cluster of m units gives
         you exactly 1 independent piece of information, no matter
         how big m is. Massive penalty. DEFF = m.

STEP 3 — Reality sits somewhere between these two extremes, and
         scales LINEARLY with both ρ and (m-1):

              DEFF = 1 + (m-1) × ρ

         Check it against the two extremes:
           ρ=0 → DEFF = 1 + (m-1)×0 = 1                    ✓ matches Step 1
           ρ=1 → DEFF = 1 + (m-1)×1 = m                     ✓ matches Step 2
```

That's the whole derivation — DEFF is just a straight-line interpolation between "no penalty" and "total redundancy," scaled by how big your clusters are.

**What DEFF actually does to your sample size:**

```
   n_effective = n_total / DEFF

   Your EFFECTIVE sample size is your raw count, DIVIDED by
   the penalty. A DEFF of 11 means your 50,000 riders are only
   as statistically useful as ~4,545 truly independent riders.

   Flipped around, for PLANNING purposes:

   n_total_needed = n_simple_random × DEFF

   Whatever a normal (non-clustered) power calculation told you
   to collect, MULTIPLY it by DEFF to get your real target.
```

---

## 4. Diagnosis: What's Your Actual Constraint?

This is the section that determines whether you even have a problem, and if so, which lever to pull.

```
                Do you know (or can you estimate from
                historical data) the ICC (ρ) for this
                outcome metric within your planned clusters?
                            │
              ┌──────No─────┴────Yes────────────────┐
              ▼                                        ▼
     Estimate it first — pull historical      Continue below.
     data, compute how similar units
     within a cluster are on this metric.
     You cannot size the test without this.

                            (continuing from Yes)
                            │
                            ▼
              Compute DEFF = 1 + (m-1)ρ using your
              PLANNED cluster size (m).
                            │
                            ▼
              Is DEFF close to 1 (say, under ~2)?
                            │
              ┌──────Yes────┴────No (DEFF is large)──┐
              ▼                                        ▼
     Cluster randomization costs you       You have a REAL problem.
     relatively little extra sample        Continue to Section 6 —
     size. Proceed with standard           your real constraint is
     power planning, scaled by DEFF.       probably the NUMBER OF
                                           CLUSTERS, not total units.
                                           Go check: how many
                                           independent clusters can
                                           you actually get?
```

**The one-sentence diagnostic**: don't ask "do I have enough total units" — ask "do I have enough independent *clusters*," because past a certain cluster size, adding more individuals inside existing clusters barely helps at all (see Section 6).

---

## 5. Plug-In Walkthrough — Full Worked Example

A standard (non-clustered) power calculation says you need $n=2{,}000$ riders total. You're planning to run this across 20 cities, ~500 riders/city ($m=500$), and historical data gives $\rho=0.02$ for this metric.

```
STEP 1 — Compute DEFF:
    DEFF = 1 + (m-1)×ρ
         = 1 + (500-1)×0.02
         = 1 + 499×0.02
         = 1 + 9.98
         = 10.98

STEP 2 — Translate into required total individuals:
    n_total = n_simple_random × DEFF
            = 2,000 × 10.98
            ≈ 21,960

STEP 3 — Sanity-check against what you actually have:
    20 cities × 500 riders/city = 10,000 riders available
    21,960 needed > 10,000 available
                        ↑
        Even though ρ=0.02 LOOKS tiny, it inflated your
        requirement past what 20 cities of 500 riders each
        can supply. You're short — and the fix is NOT
        "add more riders per city" (see Section 6).
```

**Why a "tiny" 0.02 did this much damage**: the formula multiplies ρ by $(m-1)$, not just by itself. With $m=500$, that's $\rho \times 499 \approx 10$ — a small correlation, multiplied by hundreds of co-clustered units, adds up to a large number. This is the single most commonly underestimated mechanic in this whole topic: **small ICC + large cluster size is not automatically safe.**

---

## 6. Diagnosis Continued: Why "Just Add More Riders" Doesn't Fix It

```
    If you're short on effective sample size, there are
    exactly two dials you can turn — and they are NOT
    equally effective:

    DIAL 1: Add more riders PER CITY (increase m)
    ────────────────────────────────────────────────
    Bad idea. DEFF grows roughly LINEARLY with m.
    Doubling riders-per-city roughly doubles DEFF too —
    you're adding redundant information, not independent
    information. Your effective sample size barely moves.

    DIAL 2: Add more CITIES (increase the number of clusters)
    ────────────────────────────────────────────────
    Good idea. Each new city is a genuinely new, independent
    "super-observation" — this is the dial that actually
    buys you statistical power.

    PICTURE:

    Adding riders within a city:        Adding a new city:

    ● ● ● ● ●  →  ● ● ● ● ● ● ● ●         ●     ●     ●     ●
    (city 1 gets bigger,                (a genuinely NEW,
     but it's still just                 independent data point
     ONE data point, diluted             — this is what actually
     across more correlated              moves your effective n)
     individuals)
```

**This is why the worked example's shortfall (21,960 needed vs. 10,000 available) can't be fixed by recruiting more riders in the same 20 cities** — you'd need either more cities, smaller/less-correlated clusters, a bigger acceptable MDE, or a switchback design (Section 7) to manufacture more independent "looks" over time instead of over space.

---

## 7. Switchback Tests — The Time-Based Version of the Same Problem

```
   Instead of splitting SPACE (cities) into treatment/control,
   a switchback test splits TIME — the same city alternates
   between treatment and control across different time blocks.

   City X:  [ TREATMENT ][ CONTROL ][ TREATMENT ][ CONTROL ]
             week 1        week 2     week 3        week 4

   The analogous redundancy problem: adjacent time blocks
   aren't fully independent either, if treatment's effect
   "carries over" into the next block (e.g., drivers who
   repositioned during a TREATMENT week don't instantly
   snap back once CONTROL starts).

   FIX: insert a buffer/burn-in period at each transition —
   throw away the data right at the switch, and only count
   data once the system has settled into the new condition:

   [ TREATMENT ][buffer][ CONTROL ][buffer][ TREATMENT ]
```

The number-of-clusters-vs-cluster-size tradeoff has a direct analogue here: **more, shorter switchback periods (with adequate buffers) generally give you more independent "looks" than fewer, longer periods** — but too-short periods risk not capturing the treatment's true steady-state effect, and burn-in time becomes a larger fraction of a short period, wasting more of your data collection relatively speaking.

---

## 8. Why the Naive Approach Fails — Consolidated

```
FAILURE 1 — Trusting raw unit count
   "We have 50,000 riders" ignores that riders within the
   same city are correlated — the real number that matters
   is the effective sample size AFTER dividing by DEFF.

FAILURE 2 — Dismissing small ICC as automatically negligible
   ρ=0.02 sounds tiny. But DEFF = 1+(m-1)ρ means it's
   multiplied by cluster size — at m=500, that "tiny" 0.02
   becomes a design effect of ~11x. Small ICC is only
   actually negligible if cluster size is ALSO small.

FAILURE 3 — Trying to fix a shortfall by adding riders
            within existing clusters
   This barely moves effective sample size, since DEFF
   scales with m — you're adding correlated, redundant
   information, not new independent information.

FAILURE 4 — Preferring fewer, larger clusters "for convenience"
   Operationally, whole cities feel simpler than 50 zip
   codes. Statistically, this is the worse choice — larger
   m directly inflates DEFF. Smaller, more numerous clusters
   are more efficient, AS LONG AS they still adequately
   contain whatever interference motivated cluster
   randomization in the first place (Chapter 7 callback —
   don't shrink clusters so much that interference leaks
   across the boundaries you just drew).

FAILURE 5 — Ignoring carryover in switchback designs
   Treating adjacent time blocks as fully independent when
   treatment's effect actually lingers into the next block
   — always insert a buffer/burn-in period at transitions.
```

---

## 9. Your Actual Options When the Numbers Don't Work

Continuing the worked example: 20 cities isn't enough for the required effective sample size. What do you actually do?

```
OPTION 1 — Accept a larger MDE
   Stop trying to detect a small effect. With few
   independent clusters, you fundamentally can't get
   fine-grained detection — accept you can only reliably
   catch bigger effects, and adjust the experiment's goal
   accordingly.

OPTION 2 — Use finer-grained clusters (if interference allows)
   Switch from "whole cities" to "neighborhoods within
   cities," IF the interference you're guarding against
   (Chapter 7) is genuinely contained at that smaller
   scale. This gives you more independent clusters without
   needing more raw geography.

OPTION 3 — Add a switchback dimension
   Run each of your 20 cities through multiple time blocks
   (with buffers), trading spatial clusters for temporal
   ones to manufacture more independent "looks" without
   needing new cities.

OPTION 4 — Be honest that the test is underpowered
   If none of the above are feasible, say so. Running an
   underpowered test and either over-interpreting a null
   result or getting lucky with a "significant" one that
   doesn't reflect real rigor is worse than not running it.
```

---

## 10. Q&A

**Q: Your marketplace test needs cluster randomization by city due to interference concerns. A colleague says "we have 50,000 riders across 10 cities, that's plenty of sample size." What's the flaw in this reasoning?**
A: Raw rider count is misleading under cluster randomization — what matters is the *effective* sample size after accounting for the design effect, which depends on both cluster size (5,000 riders/city here) and the intra-cluster correlation. If riders within the same city behave similarly (plausible, since they share local market conditions), the design effect could be substantial, and the effective sample size might be dramatically smaller than 50,000 — potentially closer to what 10 independent "super-observations" would provide, if ICC is high enough. I'd want to estimate ICC from historical data and compute DEFF before concluding the sample size is sufficient, rather than trusting the raw rider count.

**Q: Why does having more, smaller clusters tend to be more statistically efficient than fewer, larger clusters, for the same total number of units?**
A: Because the design effect formula (DEFF = 1 + (m-1)×ρ) shows that variance inflation grows with cluster size m — larger clusters mean more redundant, correlated observations packed into each cluster, while what actually drives independent information is closer to the *number* of clusters, not the total unit count. Spreading the same total units across more, smaller clusters reduces m, which directly reduces DEFF and increases your effective sample size for the same total data collected — as long as the smaller cluster size is still large enough to adequately contain whatever interference motivated cluster randomization in the first place.

**Q: An ICC of 0.02 sounds negligible. Why did it inflate the required sample size by roughly 11x in a cluster of 500?**
A: Because the design effect formula multiplies ICC by (cluster size − 1), not just ICC alone — even a small per-unit correlation compounds substantially when there are hundreds of units sharing that correlation within each cluster. This is a common interview trap: people intuitively dismiss a small ICC as unimportant without realizing its impact scales with cluster size, and large clusters (which are often exactly what you'd want operationally, e.g., big cities) are precisely where this compounding effect bites hardest.

**Q: You only have 5 cities available for a marketplace pricing test, but your DEFF-adjusted sample size calculation says you'd need the equivalent of 20+ independent clusters for adequate power. What are your options?**
A: A few honest options: (1) accept a much larger MDE — only commit to detecting bigger effects, since a small number of clusters simply can't provide enough independent information for fine-grained detection; (2) explore whether a finer-grained cluster definition (e.g., neighborhoods within each city rather than whole cities) can still adequately contain the interference while providing more independent clusters; (3) extend the test duration and use a switchback design within each of the 5 cities, trading spatial clusters for temporal ones to gain more independent "looks" over time; or (4) if none of these are feasible, be transparent with stakeholders that the test as designed is underpowered, rather than running it and either over-interpreting a null result or getting a lucky "significant" result that doesn't actually reflect adequate statistical rigor.

---

## 11. Comprehension Check

1. Using Section 4's diagnostic flowchart, walk through what you'd do if historical ICC data doesn't exist yet for your planned metric.
2. Compute DEFF for $m=200$, $\rho=0.05$. Then compute it again for $m=1{,}000$, same $\rho$. What does the comparison tell you about the "add more riders per cluster" strategy from Section 6?
3. Explain, without the formula, why ρ=0 gives DEFF=1 and ρ=1 gives DEFF=m — use the "super-dot" picture from Section 1.
4. A team wants to shrink their clusters from whole cities to individual zip codes to reduce DEFF. What must they check before doing this, per Section 8's Failure 4?
5. Why does a switchback design need a buffer/burn-in period, and what specifically goes wrong if you skip it?
6. Given the worked example's shortfall (need ~21,960, have 10,000), rank Section 9's four options from most to least statistically desirable, and explain your ranking.

---
*This is a clarity-focused rebuild of Chapter 17, restructured around a "super-dot" intuition picture for what clustering actually destroys, a derivation of DEFF from its two limiting cases (ρ=0, ρ=1), a diagnosis flowchart for identifying whether clusters-vs-units is your real constraint, and a dedicated section on why the naive fixes (more riders per city, fewer/bigger clusters) don't work. All original formulas, worked numbers, and Q&A are preserved.*
