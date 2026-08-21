# Chapter 3: Human Evaluation

## Why we need humans at all

You just saw that BLEU/ROUGE/perplexity can't detect meaning, truthfulness, or helpfulness. So who *can* judge those things reliably? Humans. But "ask a human if it's good" is not a rigorous eval method by itself — it needs structure, or you get noisy, inconsistent, unusable data. This chapter is about *how* to structure human judgment so it becomes a real measurement, not just an opinion.

## Two ways to ask a human "is this good"

**1. Absolute rating (rubric / Likert scale)**
Show the annotator one output. Ask them to score it, e.g., 1–5, against a rubric (fluency, correctness, helpfulness).

**2. Pairwise comparison (A/B preference)**
Show the annotator two outputs (from Model A and Model B) for the same prompt. Ask: "which is better?" (or tie).

**Intuition for why pairwise usually wins:** Humans are bad at *absolute* judgments but good at *relative* judgments. Ask someone "rate this coffee 1–10" and you'll get inconsistent numbers across days/annotators. Ask "which of these two coffees is better" and people agree much more consistently. This is a well-documented finding in psychophysics (humans are better at relative comparison than absolute magnitude estimation) and it carries directly into LLM eval design — it's why leaderboards like Chatbot Arena use pairwise comparisons, not absolute scores.

**Tradeoff:** pairwise comparisons don't directly give you a "quality score" — they give you a ranking. You need a model (like Elo, which we'll touch on) to convert a pile of pairwise votes into a scalar leaderboard score. Absolute ratings are messier per-judgment but directly interpretable and cheaper to aggregate (just average them).

## Building a rubric — this is what actually gets asked in interviews

A bad rubric: "Rate this response 1-5 for quality." Too vague — every annotator interprets "quality" differently, so scores won't be comparable.

A good rubric decomposes quality into **specific, independently-gradable axes**. For a chatbot response, a typical decomposition:

1. **Fluency** — is it grammatical, natural-sounding?
2. **Relevance** — does it actually address the prompt?
3. **Factual correctness** — are the claims true?
4. **Helpfulness** — does it accomplish what the user actually needed?
5. **Safety** — does it avoid harmful/inappropriate content?

Each axis gets its own 1-5 scale with **anchored descriptions** at each point (e.g., 1 = "factually wrong in a way that would mislead the user," 5 = "fully accurate, no unsupported claims"). Anchoring matters enormously — without it, one annotator's "3" is another's "4."

## Inter-annotator agreement — how do you know your human labels are even trustworthy?

This is the part interviewers really probe, because it's where most people's understanding is shallow. If you give the same output to two annotators and they disagree constantly, your "ground truth" labels are just noise — and any model you evaluate against that noise is unreliable too.

**Why not just use raw agreement percentage?** Because agreement can happen by chance. If 90% of your labels are "safe" and 10% are "unsafe," two annotators could each just guess "safe" every time and get ~81% raw agreement (0.9 × 0.9 + 0.1 × 0.1) without actually paying attention. You need a metric that corrects for chance agreement.

### Cohen's Kappa (2 annotators)

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

where $P_o$ = observed agreement, $P_e$ = expected agreement by chance.

**Worked numerical example.** Two annotators each label 100 responses as "safe" or "unsafe."

Confusion matrix:
| | Annotator 2: Safe | Annotator 2: Unsafe |
|---|---|---|
| **Annotator 1: Safe** | 70 | 5 |
| **Annotator 1: Unsafe** | 5 | 20 |

**Step 1 — observed agreement $P_o$:** they agree on 70 + 20 = 90 out of 100 → $P_o$ = 0.90

**Step 2 — expected agreement by chance $P_e$:**
- Annotator 1 said "Safe" 75/100 times (0.75), Annotator 2 said "Safe" 75/100 times (0.75).
  → chance both say "Safe": 0.75 × 0.75 = 0.5625
- Annotator 1 said "Unsafe" 25/100 (0.25), Annotator 2 said "Unsafe" 25/100 (0.25).
  → chance both say "Unsafe": 0.25 × 0.25 = 0.0625
- $P_e$ = 0.5625 + 0.0625 = 0.625

**Step 3 — kappa:**
$$\kappa = \frac{0.90 - 0.625}{1 - 0.625} = \frac{0.275}{0.375} ≈ 0.733$$

**How to read it:** κ ranges from -1 to 1. Rough interpretation bands used widely in practice:
- < 0.20: slight agreement
- 0.21–0.40: fair
- 0.41–0.60: moderate
- 0.61–0.80: substantial ← our 0.733 lands here
- 0.81–1.0: near-perfect

A κ of 0.73 is generally considered *good enough to trust the labels* for most production eval pipelines, though top-tier academic benchmarks often want > 0.8.

### Krippendorff's Alpha — the generalization

Cohen's kappa only works for exactly 2 annotators on categorical labels. **Krippendorff's alpha** generalizes to: any number of annotators, missing data (not everyone labels everything), and different data types (categorical, ordinal, interval — like a 1-5 scale, which is technically ordinal, not purely categorical).

Intuition, without the full formula: instead of "observed vs. chance agreement," alpha compares **observed disagreement** to **expected disagreement**:

$$\alpha = 1 - \frac{D_o}{D_e}$$

where $D_o$ = observed disagreement (weighted by how far apart the labels are — useful for ordinal scales, where a 1-vs-2 disagreement is less bad than a 1-vs-5 disagreement), $D_e$ = expected disagreement by chance.

**When to reach for which in an interview answer:** "If I have exactly two annotators and simple categorical labels, Cohen's kappa is standard and easy to compute. If I have 3+ annotators, missing labels, or an ordinal rubric like a 1-5 scale where distance between scores matters, I'd use Krippendorff's alpha since it handles all of that in one unified framework."

## Practical pipeline notes (this is what "production" human eval actually looks like)

- **Sample size:** you don't label your whole eval set with 5 annotators each — too expensive. Common practice: label a subset (e.g., 200-500 examples) with 3 annotators to *measure* agreement, then scale the rest with fewer annotators (even 1) once you trust the rubric.
- **Annotator training/calibration:** run a pilot round, discuss disagreements as a team, refine the rubric anchors, then re-measure kappa. Agreement should go *up* after calibration — if it doesn't, the rubric itself is broken, not the annotators.
- **Gold/attention checks:** seed known-answer examples into the annotation batch to catch annotators who are rubber-stamping.

## Quick check

Two annotators rate 100 model outputs as "harmful" or "not harmful." Annotator 1 flags 10 as harmful, Annotator 2 flags 10 as harmful, and they agree on 96 total labels (both say harmful on 4, both say not-harmful on 92). Is κ going to look impressively high or surprisingly low, and why?

Compute it: $P_o$ = 0.96. $P_e$ = (0.10×0.10) + (0.90×0.90) = 0.01 + 0.81 = 0.82. κ = (0.96-0.82)/(1-0.82) = 0.14/0.18 ≈ **0.78** — looks decent, but notice how sensitive it was: if they'd only agreed on 3 of the harmful ones instead of 4, κ drops sharply. This is the classic behavior with **rare/imbalanced categories** (like "harmful" content, which is rare) — kappa is very sensitive to small changes when the positive class is small. Worth flagging in an interview: "for rare-event labels like safety flags, I'd want a larger sample before trusting a single kappa estimate."

---

Chapter 4 is LLM-as-a-Judge — using an LLM to grade another LLM's outputs, and the biases (position bias, verbosity bias, self-preference) you need to control for. Want me to continue?
## Krippendorff's Alpha — full worked example

### Setup: 3 annotators, nominal labels ("Safe" / "Unsafe"), 5 items, no missing data yet

| Unit | Annotator A | Annotator B | Annotator C |
|---|---|---|---|
| 1 | Safe | Safe | Unsafe |
| 2 | Unsafe | Unsafe | Unsafe |
| 3 | Safe | Safe | Safe |
| 4 | Unsafe | Safe | Unsafe |
| 5 | Safe | Unsafe | Unsafe |

**Core idea, before the math:** instead of one 2x2 table (which only works for 2 annotators), Krippendorff's alpha looks at *every pair of annotators who rated the same item*, tallies how often those pairs agreed vs. disagreed, and compares that to how often you'd expect them to disagree purely by chance given the overall mix of labels.

### Step 1 — count every within-unit pair

Each unit with 3 raters gives you 3×2 = 6 **ordered pairs** (A vs B, A vs C, B vs A, B vs C, C vs A, C vs B).

Unit 1 (S, S, U) → pairs: (S,S),(S,U),(S,S),(S,U),(U,S),(U,S) → 2 SS, 2 SU, 2 US
Unit 2 (U, U, U) → all 6 pairs are (U,U)
Unit 3 (S, S, S) → all 6 pairs are (S,S)
Unit 4 (U, S, U) → 2 US, 2 SU, 2 UU
Unit 5 (S, U, U) → 2 SU, 2 US, 2 UU

**Totals across all 30 pairs:** SS = 8, SU = 6, US = 6, UU = 10

### Step 2 — observed disagreement (Do)

Nominal categories get the simplest possible "distance": 0 if the pair matches, 1 if it doesn't.

Disagreeing pairs = SU + US = 6 + 6 = 12

$$D_o = \frac{12}{30} = 0.40$$

### Step 3 — expected disagreement by chance (De)

Pool *all* 15 individual ratings together (ignore who rated what): count how many are Safe vs Unsafe.

Safe = 7, Unsafe = 8, total = 15.

If you just grabbed two ratings at random from this pool, what's the chance they're different labels?

$$D_e = \frac{2 \times (7 \times 8)}{15 \times 14} = \frac{112}{210} \approx 0.533$$

### Step 4 — alpha

$$\alpha = 1 - \frac{D_o}{D_e} = 1 - \frac{0.40}{0.533} \approx 1 - 0.75 = 0.25$$

**Reading it:** 0.25 is weak agreement — much lower than the κ=0.73 we got in Chapter 3 with 2 annotators on a similar-sized problem. That's realistic: adding a 3rd annotator's independent judgment usually *does* surface more disagreement than a 2-annotator comparison hides.

---

## Now the nuance: adding a unit with missing data

Say a 6th item only got rated by A and C (B never saw it): A = Safe, C = Unsafe.

**This is the whole point of alpha vs. kappa** — you don't need to throw the item out, and you don't need every annotator to have rated every item. This unit just contributes fewer pairs: 2×1 = 2 ordered pairs → (S,U) and (U,S).

New totals: SS=8, SU=7, US=7, UU=10 → total pairs = 32
New pool: Safe=8, Unsafe=9, total=17

$$D_o = \frac{14}{32} = 0.4375 \qquad D_e = \frac{2(8)(9)}{17\times16} \approx 0.529$$

$$\alpha = 1 - \frac{0.4375}{0.529} \approx 0.17$$

Notice: the partially-rated unit slotted in seamlessly — no special handling, no dropped data, no need to pair-match it against a 3rd rater that doesn't exist. That's the exact capability Cohen's kappa doesn't have.

---

## The other nuance: ordinal/interval data changes what "disagreement" even means

Everything above used **nominal** distance: disagreement is binary, 0 or 1. That's fine for "Safe/Unsafe," but wrong for a 1–5 rubric score, where a 1-vs-2 disagreement is clearly less bad than a 1-vs-5 disagreement.

**Interval-level** data fixes this by squaring the numeric gap instead of using 0/1:

- Annotators rate 2 vs 3 → δ² = (2−3)² = 1
- Annotators rate 1 vs 5 → δ² = (1−5)² = 16

Under nominal scoring, *both* of those count as "1 disagreement, full stop." Under interval scoring, the second one is penalized 16× harder — because it genuinely represents a much bigger split in judgment.

**Ordinal-level** data (technically what a 1–5 Likert rubric is) sits between the two: it uses the *ranks* and how many rating categories separate two scores, without assuming the categories are evenly spaced numbers — useful when you can say "3 is worse than 2" but don't want to assume the psychological gap between 2-and-3 equals the gap between 4-and-5.

**Why this matters for your interview answer:** if you're evaluating a 1-5 human/LLM rubric score (Chapters 3-4), nominal Krippendorff's alpha would treat a 1-vs-2 near-miss exactly the same as a 1-vs-5 total blowup — that's the wrong measurement. You'd specify the **ordinal** (or interval) metric when computing alpha, precisely because the size of the disagreement, not just its presence, is informative.
