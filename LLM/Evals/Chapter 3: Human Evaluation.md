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
