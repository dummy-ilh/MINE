# Module 8 — Evaluation (Master Notes, Maximum Depth)

## 0. Why evaluation is its own hard problem

Module 2 established that **perplexity/loss doesn't reliably predict downstream task performance**, and Module 3's emergent-abilities discussion showed that **the choice of metric itself can create or hide apparent capability jumps**. Evaluation is the discipline of trying to actually measure "is this model good," and it turns out to be genuinely difficult — every method below (benchmarks, human eval, LLM-as-judge) has real, well-documented failure modes, and knowing those failure modes specifically (not just the method names) is what separates a strong interview answer from a shallow one.

---

## 1. Benchmark Suites

### What they are
Fixed datasets of (input, correct-answer) pairs, covering broad knowledge/reasoning (MMLU), commonsense inference (HellaSwag), and many other specific skills — the model is run once over the whole set, and accuracy (or another automated metric) is reported as a single number.

### MMLU (Massive Multitask Language Understanding)
Roughly 14,000+ multiple-choice questions spanning 57 subjects (STEM, humanities, social science, professional fields like law and medicine), each question having exactly one correct answer among typically 4 options. The appeal is breadth — a single number that's meant to summarize competence across a very wide swath of human knowledge domains at once.

### HellaSwag
Commonsense-inference benchmark: given a sentence describing the start of an everyday situation, the model must pick the most plausible continuation among several options — importantly, the *wrong* options were specifically generated adversarially (via earlier, weaker language models) to be superficially fluent and "look right" on the surface, while actually being physically/logically implausible — designed specifically to be hard for models that pattern-match surface fluency without real situational understanding.

### The known flaws — this is the part interviewers actually want

**Contamination**: because these benchmarks are public datasets that have existed on the internet for years, and modern pretraining corpora are scraped from enormous swaths of the internet, there's a real, well-documented risk that **benchmark questions (or very close paraphrases) leak directly into a model's pretraining data** — meaning a high benchmark score can partly reflect memorization of the test set itself, not genuine reasoning capability. This is one of the most commonly raised criticisms of any single benchmark number, and detecting/quantifying contamination reliably is itself an unsolved, actively-researched problem (naive string-matching for exact leaked questions misses paraphrased leakage entirely).

**Saturation**: as models improve, scores on a fixed benchmark eventually cluster near the maximum possible score (near 100%), at which point the benchmark **stops being able to meaningfully differentiate** between a very good model and an even-better one — both score ~95%+ and the benchmark no longer tells you anything useful about which is actually stronger. This is exactly why the field continuously produces new, harder benchmarks (e.g., MMLU was itself later followed by "MMLU-Pro," a harder variant, specifically because base MMLU became saturated for frontier models) — benchmark saturation is a recurring, structural problem, not a one-time fluke.

**Format/prompt sensitivity**: multiple-choice benchmark scores can shift meaningfully based on seemingly-trivial formatting choices — how answer options are labeled (A/B/C/D vs 1/2/3/4), whether the model is asked to output just the letter vs the full answer text, few-shot example count and ordering — meaning reported benchmark numbers from different papers/labs aren't always strictly comparable even when nominally testing "the same benchmark," unless the exact evaluation harness/methodology is held constant.

---

## 2. Human Evaluation

### The core method
Have human raters directly assess model outputs — commonly via **pairwise comparison** (shown two model responses to the same prompt, pick the better one — exactly the same data-collection pattern as Module 5's reward-model preference data), sometimes via **absolute rating scales** (e.g., rate helpfulness 1-5), or via structured rubrics scoring specific dimensions (accuracy, helpfulness, harmlessness, etc. scored separately).

### Why pairwise comparison is generally preferred over absolute scoring (directly echoes Module 5's Bradley-Terry discussion)
The same reasoning from reward-model training applies here: absolute numeric ratings are noisy and inconsistent — a given rater's internal sense of "what does a 7/10 mean" drifts over time and differs from other raters' internal scales, whereas "is A better than B, looking at both side by side" is a comparison humans make far more consistently. Aggregating many pairwise comparisons (often via an Elo-style rating system, borrowed directly from chess ranking, mathematically closely related to the Bradley-Terry model from Module 5) produces a more reliable relative ranking across many models than trying to average noisy absolute scores.

### Known limitations (interview-relevant, often overlooked)
- **Expensive and slow** to scale — genuinely limits how often you can re-evaluate during rapid iteration, compared to an automated benchmark you can rerun in minutes.
- **Rater disagreement and inconsistency**: different human raters often disagree with each other on which response is "better," especially for open-ended, subjective, or nuanced prompts — this is typically quantified via **inter-annotator agreement** metrics (e.g., Cohen's kappa or similar statistics measuring how much raters agree beyond what chance alone would predict) — low agreement numbers on a given evaluation task are a real, reportable signal that the task itself may be too subjective/ambiguous to trust a single "ground truth" preference label for.
- **Length bias / superficial-quality bias**: human raters (and, as covered next, LLM judges even more so) have a well-documented tendency to prefer **longer, more verbose responses**, or responses with nicer surface formatting (bullet points, confident tone), somewhat independent of the actual correctness/quality of the content — a systematic bias that needs to be actively corrected for or at least acknowledged when interpreting preference-based evaluation results.

---

## 3. LLM-as-Judge

### The core idea
Use a strong LLM (often the most capable model available, e.g. GPT-4-class) to **evaluate other models' outputs** — either scoring a single response against a rubric, or performing the same pairwise-comparison task humans would do — as a cheap, fast, highly-scalable substitute for human evaluation.

### Why it's attractive
Dramatically cheaper and faster than human evaluation at scale — you can evaluate thousands of model outputs in the time/cost it would take to get even a handful of careful human judgments, making it practical to use as a rapid, frequent feedback signal during model development iteration (this directly connects back to Module 5's RLAIF, which is literally "LLM-as-judge" repurposed as a training signal rather than just a final evaluation report).

### The well-documented biases (a favorite, very specific interview topic)

**Position bias**: when shown two responses side by side (A then B, or B then A) and asked to pick the better one, LLM judges have been shown to have a measurable tendency to **favor whichever response appears first (or, in some setups, second)** in the presented order, independent of actual content quality — the standard mitigation is evaluating **both orderings** (A-then-B and B-then-A) and only counting a judgment as a genuine "win" if the same response wins in both orderings, discarding/treating-as-tie cases where the order flip changes the outcome.

**Verbosity bias**: LLM judges, like human raters, systematically tend to score longer, more detailed-looking responses higher, even when the additional length doesn't add genuine correctness or value — sometimes even penalizing a correct, appropriately concise answer relative to a padded, verbose one that says the same thing with more words.

**Self-preference bias**: an LLM judge has been shown to sometimes rate outputs **generated by the same model family/architecture as itself** more favorably — plausibly because it recognizes and is more "comfortable with" its own characteristic style/phrasing patterns, independent of actual quality — a subtle but real confound when using, say, "GPT-4 as judge" to evaluate outputs partly produced by GPT-4 itself or closely related models.

**Practical mitigation strategies to know**: randomizing/swapping presentation order (addresses position bias), explicitly instructing the judge model to ignore response length/style and focus only on substantive correctness (partial mitigation for verbosity bias, though imperfect), and using multiple different judge models from different families/providers and checking for agreement across them (helps surface and average out self-preference bias, since a bias favoring "responses that look like judge-model X's own style" would need to consistently favor the exact same responses across judges built by different labs — a genuine, content-based quality difference should hold up across multiple different judges, while a style-preference artifact often will not).

---

## 4. Hallucination

### Definition
When a model generates content that is **factually incorrect, fabricated, or unsupported by any real source/evidence**, while stated with the same fluent, confident tone as genuinely correct content — the core danger being that hallucinated content is often **not distinguishable from correct content by surface style alone**, which is exactly what makes it a practically serious problem rather than an obvious, easily-filtered error type.

### Why it happens — a mechanistic framing worth having ready
The CLM pretraining objective (Module 2) trains a model purely to predict **plausible next tokens** given context — it optimizes for fluency and statistical plausibility, with **no explicit built-in mechanism that distinguishes "this continuation is fluent and plausible" from "this continuation is actually, factually true."** A model can be highly confident (in the sense of assigning high probability to a token sequence) about a completion that is fluent, grammatically perfect, stylistically consistent with real facts elsewhere in its training data — and still be entirely fabricated, precisely because "sounds right" and "is right" are correlated but not identical signals in the training objective the model was actually optimized against.

### Measurement approaches
- **Fact-verification-based metrics**: decompose a generated response into individual factual claims, then check each claim against a trusted external knowledge source (a reference document, a knowledge base, or a search-retrieved source) — measuring the fraction of claims that are verifiably supported vs. unsupported/contradicted. This requires either a closed-domain reference (e.g., grading a summary against its specific source document — a more tractable, well-defined problem) or open-domain fact-checking against general world knowledge (much harder, since you need a reliable, comprehensive, up-to-date external ground truth to check against).
- **Consistency-based metrics (no external reference needed)**: sample multiple independent generations for the same prompt (e.g., at nonzero temperature, so outputs vary), and measure **how consistent the model's claims are with each other across those samples** — the reasoning being that a model that actually "knows" a fact will state it consistently across repeated samples, whereas fabricated/hallucinated content, having no real grounding in the model's actual knowledge, tends to vary/contradict itself across independent samples (this is the core idea behind methods like SelfCheckGPT) — a genuinely useful practical technique because it needs no external database at all, just multiple samples from the model itself.
- **Calibration-based framing**: a related, complementary lens on the same underlying problem — checking whether a model's expressed confidence (either explicitly stated, or implicitly reflected in its token-level probability) is well-**calibrated**, meaning that among all the claims a model states with (say) 90% apparent confidence, roughly 90% of those claims should actually turn out to be true. A well-calibrated model that says "I'm not sure, but possibly X" for genuinely uncertain claims is behaving more safely/usefully than a poorly-calibrated model that states both its correct and incorrect claims with identical, maximal confidence — this calibration gap (confidently-stated-but-wrong content) is often considered the most practically dangerous form of hallucination, since it's the hardest for a downstream reader to detect purely from the text's tone alone.

### Interview-ready synthesis
"Hallucination isn't a bug the model 'should' just avoid — it's a fairly direct consequence of the CLM training objective optimizing for plausible continuation rather than verified truth, with no built-in fact-checking mechanism. Measuring it well requires either an external ground-truth reference to check claims against, or reference-free approaches like sampling consistency, since raw fluency/confidence in the output text itself is not a reliable signal of factual correctness."

---

## 5. Side-by-side summary table (memorize this cold)

| | Benchmark Suites | Human Evaluation | LLM-as-Judge |
|---|---|---|---|
| Speed/cost | Fast, cheap, automated | Slow, expensive | Fast, cheap, scalable |
| Main failure mode | Contamination, saturation, format sensitivity | Rater disagreement, length/superficial-quality bias | Position bias, verbosity bias, self-preference bias |
| Ground truth needed? | Yes (fixed correct answers) | No (relative preference judgment) | No (relative preference judgment) |
| Common mitigation | New harder benchmarks, contamination checks | Inter-annotator agreement measurement, rubric structuring | Order randomization, multiple diverse judges, explicit length-invariance instructions |

---

## 6. Quick-fire Q&A (self-test)

**Q: What is benchmark contamination, and why is it hard to detect reliably?**
A: When benchmark questions (or close paraphrases) leak into a model's pretraining data, inflating scores through memorization rather than genuine capability. It's hard to detect because naive exact-string matching misses paraphrased leakage, and there's no fully reliable general method to prove a specific fact was or wasn't memorized from training data.

**Q: What is benchmark saturation, and what's the field's typical response to it?**
A: When most strong models cluster near the maximum score on a fixed benchmark, the benchmark loses its ability to differentiate between good and better models. The typical response is producing new, harder benchmark variants (e.g., MMLU-Pro following MMLU) once the original saturates.

**Q: Why is pairwise human comparison generally preferred over absolute numeric rating scales?**
A: Absolute scales are noisy and drift across raters and over time (inconsistent internal sense of what a given score means), while relative "is A better than B" judgments are much more consistent for humans to give — this mirrors exactly the reasoning behind using pairwise comparisons for reward-model training in Module 5.

**Q: Name the three well-documented LLM-as-judge biases and one mitigation for each.**
A: Position bias (favoring whichever response is shown first/second) — mitigated by evaluating both orderings and discarding order-sensitive results. Verbosity bias (favoring longer responses regardless of quality) — mitigated by explicit instructions to ignore length/style. Self-preference bias (favoring outputs stylistically similar to the judge's own model family) — mitigated by using multiple diverse judge models from different providers and checking cross-judge agreement.

**Q: Why does the CLM pretraining objective make hallucination a fairly direct, expected consequence rather than an incidental bug?**
A: CLM optimizes purely for predicting plausible, fluent next tokens given context, with no explicit mechanism distinguishing "fluent and plausible" from "factually true" — so a model can be highly confident about a fabricated continuation simply because it's statistically plausible-sounding, since plausibility and truth are correlated in training data but not identical as an optimization target.

**Q: How does a consistency-based hallucination-detection method (like SelfCheckGPT) work, and why doesn't it need an external reference?**
A: It samples multiple independent generations for the same prompt (varying via nonzero temperature) and measures how consistent the model's stated claims are with each other across samples — genuinely known facts tend to be repeated consistently, while fabricated content tends to vary/contradict itself across samples, so no external ground-truth database is required, just the model's own sampling variability.

**Q: What does it mean for a model to be "well-calibrated," and why is a calibration gap considered especially dangerous?**
A: A well-calibrated model's expressed confidence matches its actual accuracy — e.g., among all claims stated with 90% apparent confidence, about 90% should actually be true. A calibration gap (confidently stating false claims with the same tone as true ones) is especially dangerous because a reader has no textual/tonal cue to distinguish the wrong confident claims from the correct ones.

---
*End of Module 8 (maximum depth). Next: Module 9 — Interview-Style Synthesis (cross-module FAANG-style Q&A and system-design-flavored questions spanning everything covered so far).*
