# Chapter: P-Hacking & HARKing — the Human Side of False Positives

*(Sits alongside Chapters 16–17: where multiple testing and peeking cover mechanical sources of false positives, this chapter covers the ones that come from analyst choices — made honestly or otherwise.)*

## 1. Intuition

Chapters 16 and 17 covered false-positive inflation from *transparent* multiplicity — many metrics, many looks over time — where the fix is a formula (Bonferroni, BH, alpha-spending). P-hacking and HARKing are different: they're false-positive inflation from **hidden or undisclosed** multiplicity, often introduced through ordinary-seeming analysis choices rather than an obvious "we ran 20 tests."

**P-hacking**: massaging an analysis — consciously or not — until a non-significant result becomes significant, by trying multiple defensible-looking choices (which outliers to drop, which covariates to control for, when to stop collecting data, which subgroup to report) and reporting only the version that "worked."

**HARKing** (Hypothesizing After Results are Known): presenting a hypothesis discovered by *looking at the results* as though it had been the pre-specified prediction all along — rewriting the story backwards so an exploratory finding reads as confirmatory.

The core intuition to hold onto: **both practices can happen without anyone intending to deceive.** A researcher trying five reasonable outlier-removal rules and reporting the one that crossed p < 0.05 isn't necessarily lying — they may genuinely believe they picked the "right" rule. That's what makes this harder to guard against than deliberate fraud: the defense has to be structural (pre-registration, disclosure), not just "don't cheat."

## 2. Why this is a distinct problem from multiple testing

| | Multiple testing (Ch. 16–17) | P-hacking / HARKing |
|---|---|---|
| Multiplicity is | Visible — you know you ran *m* tests | Hidden — the reader never sees the other choices that were tried and discarded |
| Fix | Mathematical correction (Bonferroni, BH, alpha-spending) | Structural discipline (pre-registration, disclosure, replication) — no formula fixes it after the fact |
| Intent | Usually honest, just uncorrected | Can be entirely unintentional — "researcher degrees of freedom" exploited without anyone consciously cheating |
| Where it shows up | The analysis stage, on data you can see | The analysis stage *and* the writeup stage — HARKing specifically corrupts the narrative, not just the math |

This distinction matters because a reviewer can catch uncorrected multiple testing by looking at your analysis. They generally **cannot** catch p-hacking or HARKing by looking at your final writeup alone — the whole problem is that the discarded paths are invisible by the time you see the result. This is why the defenses below are procedural, not statistical.

## 3. Common p-hacking techniques

| Technique | What it looks like | Why it inflates false positives |
|---|---|---|
| **Optional stopping** | Peeking at results and stopping data collection the moment p < 0.05 | Same mechanism as Chapter 16's peeking problem — a hidden form of it when not disclosed |
| **Outlier exclusion flexibility** | Trying several reasonable outlier-removal rules, reporting whichever gives significance | Each rule is a different, undisclosed "test" — a garden of forking paths |
| **Covariate/control selection** | Trying several sets of control variables in a regression until one yields significance | Same mechanism — undisclosed multiplicity hidden inside "standard" model-building |
| **Subgroup slicing** | Testing the effect in many subgroups, reporting only the significant one, without disclosing how many were checked | This *is* multiple testing — but hidden, since the reader only sees the "winning" subgroup |
| **Metric flexibility** | Trying several plausible operationalizations of the outcome (e.g., revenue vs. revenue-per-user vs. log-revenue) and reporting the significant one | Each operationalization is functionally a different test |
| **Flexible sample composition** | Adjusting inclusion/exclusion criteria after seeing how they affect the result | Turns a fixed-population test into an undisclosed search over populations |

**The unifying mechanism** behind all of these is sometimes called the **"garden of forking paths"**: at almost every stage of a real analysis there are multiple defensible choices, and if you (consciously or not) let the data guide which choice you make, you've run a hidden multiple-comparisons problem even though your final writeup shows only one clean test.

## 4. Worked example — p-hacking

A team tests a new checkout flow. Primary metric (conversion rate) comes back at p = 0.09 — not significant.

- Someone notices 3 users with extremely long session durations and suggests excluding them as "clearly bots." Removing them brings p to 0.06 — still not significant, but closer.
- Someone else suggests the metric should really be "conversion rate excluding users who abandoned within the first 5 seconds" (a defensible-sounding refinement). This version comes in at p = 0.04.
- The team reports the second version as *the* result, with no mention of the first analysis or the outlier removal that was tried and didn't quite work.

**What went wrong**: none of these individual choices is obviously illegitimate — excluding bots and refining a metric definition are both things a careful analyst might do for good reasons. The problem is that **multiple candidate analyses were tried, and the one that crossed the threshold was the one reported**, with no disclosure that other versions existed and didn't clear it. This is p-hacking even though every individual step looks reasonable in isolation.

## 5. Worked example — HARKing

The same team's *primary*, pre-registered OEC (overall conversion rate) never reaches significance, even after settling on one metric definition honestly. While exploring the data, they notice mobile users on the new checkout flow converted meaningfully better than mobile users in control (p = 0.02, discovered by looking at a dozen segment breakdowns after the fact).

They then write the final report as: *"We hypothesized that mobile users would benefit most from the streamlined flow, and the data confirms this."*

**What went wrong**: the mobile-specific hypothesis didn't exist before the data was examined — it was reverse-engineered from a pattern found by looking at many segments (a multiple-testing problem in its own right, per Chapter 17) and then **narrated as if it had been predicted in advance**. This is the defining move of HARKing: it doesn't just fail to correct for the segment-search multiplicity, it actively disguises the fact that a search happened at all, making an exploratory finding look confirmatory to any reader.

## 6. Defenses — what actually works

Unlike multiple testing, there's no formula that retroactively fixes p-hacking or HARKing once they've happened — the defenses have to be structural, applied *before* you see the results:

- **Pre-registration.** Commit in writing — before looking at results — to your primary metric, your analysis plan (including outlier rules and covariates), and your stopping rule. This is the single strongest defense against both problems, because it removes the opportunity to choose after seeing the data.
- **Explicitly label exploratory vs. confirmatory findings.** If a segment effect is discovered by looking (as in the HARKing example), the writeup should say exactly that — "this was found through exploratory analysis, not predicted in advance, and should be treated as hypothesis-generating" — rather than dressing it up as confirmed.
- **Hold-out / replication requirement.** Treat any exploratory finding as provisional until it's tested again on fresh data with a pre-specified hypothesis this time — this is the only way to actually confirm a HARKed finding rather than just retell it more convincingly.
- **Disclose the full analysis, not just the winning version.** A report that says "we also tried X and Y, which didn't reach significance" is far more trustworthy than one that only shows the version that worked — this alone doesn't fix the statistics, but it lets a reader apply the correction (mentally or formally) that the writer didn't.
- **Separate the people/incentives where possible.** Some organizations reduce p-hacking risk structurally by having the analysis plan reviewed or locked by someone without a stake in the outcome before the data is unblinded.

## 7. How this connects to the rest of the curriculum

- **Chapter 9 (OEC design)**: a pre-specified OEC is itself a p-hacking/HARKing defense — it's much harder to go metric-shopping after the fact if you committed to one metric before the experiment ran.
- **Chapter 16 (peeking)**: optional stopping is p-hacking's version of the peeking problem — undisclosed, motivated-by-the-data stopping rather than a disclosed, corrected one.
- **Chapter 17 (multiple testing)**: subgroup p-hacking and multiple testing are mechanically the same problem — the difference is disclosure. A corrected, disclosed 15-segment search is legitimate exploratory analysis; the same search reported as if only one segment was ever examined is p-hacking.
- **Chapter 19 (ship decisions)**: the framework's insistence on returning to the pre-specified OEC first, and explicitly labeling secondary findings as exploratory, is a direct structural defense against both problems showing up in a real ship decision.

## 8. Do's and don'ts

| Do | Don't |
|---|---|
| Pre-register your primary metric, analysis plan, and stopping rule before seeing results | Decide your outlier rule, covariates, or metric definition after looking at how each choice affects the p-value |
| Disclose every analysis variant you tried, even the ones that didn't reach significance | Report only the version of the analysis that "worked" |
| Label a data-driven finding explicitly as exploratory / hypothesis-generating | Write up a post-hoc discovery as though it had been predicted in advance |
| Require replication on fresh data before treating an exploratory finding as confirmed | Treat a single exploratory p < 0.05 as sufficient evidence on its own |
| Apply the same multiple-testing correction (Chapter 17) to subgroup searches whether or not they "worked" | Only apply correction thinking when the uncorrected result would have been inconvenient |

## 9. Interview traps

- **Trap #1**: Treating p-hacking as always deliberate fraud — the stronger, more interview-differentiating answer is recognizing it as something that happens through ordinary, well-intentioned analyst choices ("researcher degrees of freedom"), which is exactly why it needs structural, not just ethical, defenses.
- **Trap #2**: Conflating p-hacking (a data-analysis practice) with HARKing (a reporting/narrative practice) — they compound in the same scenario but are distinct failure points, and a sophisticated answer distinguishes "how the result was produced" from "how the result was described afterward."
- **Trap #3**: Proposing only "just be honest" as the fix, without naming concrete structural defenses (pre-registration, disclosure, replication) — interviewers want mechanisms, not a promise.
- **Trap #4**: Not connecting this back to multiple testing / peeking as the same underlying statistical mechanism, just undisclosed rather than disclosed — missing this connection makes the answer feel like an isolated ethics point rather than a natural extension of the multiplicity chapters.

## 10. Comprehension check

1. Explain the difference between p-hacking and HARKing in one or two sentences each, and describe a scenario where both occur in the same analysis.
2. Why can't a statistical correction (like Bonferroni or BH) fix p-hacking after the fact, the way it can fix disclosed multiple testing?
3. A colleague says "I didn't p-hack, I just tried a few reasonable outlier rules and picked the most defensible one." What's the issue with this defense, even if made in good faith?
4. How does pre-registering an analysis plan address both p-hacking and HARKing simultaneously?
5. Design a lightweight process a small team could adopt (without full pre-registration infrastructure) to reduce HARKing risk in their experiment writeups.
