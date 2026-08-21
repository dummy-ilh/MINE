# Chapter 4: LLM-as-a-Judge

## Why this exists

Chapter 3 showed human eval is the gold standard for judging meaning, correctness, and helpfulness. But human eval is slow and expensive — you can't run it on every model checkpoint, every day, at scale. So the field asked: **can we use a strong LLM to grade another LLM's outputs, as a fast, cheap proxy for human judgment?** Turns out — mostly yes, with important caveats. This is now one of the most heavily used eval techniques in industry, and interviewers expect you to know both how it works and exactly how it breaks.

## The basic setup

You give a "judge" LLM (usually a strong model like GPT-4 or Claude Opus) a prompt like:

> "Here is a user question, and two candidate responses (A and B). Decide which response is better, or declare a tie. Explain your reasoning, then output your verdict."

Or, for absolute scoring:

> "Rate this response on a 1-10 scale for helpfulness and accuracy. Explain your reasoning, then output a score."

This mirrors exactly the two modes from Chapter 3 (pairwise vs. absolute) — except the annotator is now a model instead of a human. Same tradeoffs apply: pairwise tends to be more reliable/consistent than absolute scoring, for the same psychophysics reason.

## Does it actually correlate with human judgment?

This is the first thing an interviewer wants to hear you validate, not assume. The standard practice: take a set of outputs that *also* have human ratings, run the LLM judge on the same set, and compute agreement (using the same Cohen's kappa / correlation tools from Chapter 3) between judge and human.

**Worked numerical example.** You have 50 response pairs, each with a human "A is better / B is better / tie" label and an LLM judge's label on the same pairs. Agreement:

| | LLM says A | LLM says B | LLM says Tie |
|---|---|---|---|
| **Human says A** | 18 | 2 | 1 |
| **Human says B** | 3 | 17 | 1 |
| **Human says Tie** | 2 | 1 | 5 |

Observed agreement $P_o$ = (18+17+5)/50 = 40/50 = 0.80

This is the kind of number papers report — well-prompted LLM judges (GPT-4-class) have been found to agree with human preference roughly 80-85% of the time on general chat quality, which is in the same ballpark as human-human agreement on the same task. That's the empirical justification for trusting LLM judges at all — **if human-vs-human agreement is also ~80%, then LLM-vs-human agreement of ~80% means the judge is about as good as another human rater**, not necessarily worse.

## The biases you must know cold

This is where most candidates only have surface knowledge. Know each one with a concrete mechanism.

### 1. Position bias
The judge tends to favor whichever response is shown **first** (or in some models, second), independent of actual quality.

**How to detect it — worked example.** You run the same pair (A, B) through the judge twice: once as (A, B) and once as (B, A) — i.e., swapped order.
- Order (A, B): judge picks A. 
- Order (B, A): judge picks A again (now shown second).

If the judge consistently follows *position* rather than *content*, you'd see it flip its preference to whichever slot is favored, regardless of which response is actually in that slot. **Fix:** always evaluate both orderings and only count it as a real preference if the judge agrees after swapping; otherwise, call it a tie.

### 2. Verbosity bias
Judges systematically prefer longer responses, even when the longer response isn't more correct or helpful — because more text *looks* more thorough.

**Concrete implication:** if Model A tends to write 3-sentence answers and Model B tends to write 8-sentence answers of similar accuracy, naive LLM-judge comparison will often favor B purely on length. **Fix:** explicitly instruct the judge to penalize unnecessary verbosity, or control for length in your analysis (e.g., report win rate conditioned on similar response lengths).

### 3. Self-preference / self-enhancement bias
A model used as a judge tends to rate outputs *from its own model family* more favorably than outputs from other models, even when quality is comparable — plausibly because it's more familiar with its own "style."

**Practical implication:** if you use GPT-4 as your judge to compare a GPT-4-derived model against a Claude-derived model, expect a systematic tilt toward the GPT-4-family response. **Fix:** use a judge model from a different family than any of the candidates being evaluated, or use an ensemble of judges from different families and average.

### 4. Anchoring / scale mis-calibration (for absolute scoring)
Judges tend to cluster scores in a narrow band (e.g., everything gets 7 or 8 out of 10), which compresses your ability to distinguish "good" from "great." This mirrors the exact reason pairwise beat absolute rating for humans in Chapter 3 — it applies just as much to LLM judges.

## Putting it together — a robust LLM-judge pipeline

An interview-ready description of "how would you set up LLM-as-judge in production":

1. **Pairwise, not absolute** — more reliable given the biases above.
2. **Swap order and average** — cancels position bias.
3. **Judge from a different model family** than the candidates — reduces self-preference bias.
4. **Explicit rubric in the prompt** — same anchoring principle as Chapter 3's human rubrics; don't just say "which is better," specify axes (correctness, helpfulness, safety).
5. **Chain-of-thought before verdict** — ask the judge to reason first, then output the label; this consistently improves judge accuracy versus asking for the verdict cold.
6. **Periodic human-agreement checks** — re-validate against a human-labeled sample regularly (models drift, prompts drift), using the Cohen's kappa approach from Ch3.
7. **Length-control your reporting** — either normalize for verbosity or report length-controlled win rates (this is literally what Chatbot Arena / AlpacaEval 2.0 added after verbosity bias was identified as a major confound).

## Quick check

You're comparing Model A vs. Model B with an LLM judge. Win rate for A is 65%. Then you swap the presentation order and rerun: now A's win rate is only 40%. What does this gap tell you, and which number should you trust?

**Big red flag for position bias.** Neither raw number should be trusted in isolation — average the two runs (swapped and unswapped) per example, and only count a genuine win when the judge prefers the same response regardless of order. A ~25-point swing between orderings is a sign the judge is responding to *position*, not *quality*, and this pipeline needs the swap-and-average fix before its output means anything.

---

Chapter 5 is Benchmark Suites — MMLU, HellaSwag, GSM8K, HumanEval — what each actually measures, their construction, and the contamination problem (models "memorizing" benchmark data during pretraining). Want me to continue?
