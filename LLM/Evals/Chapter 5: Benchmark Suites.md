# Chapter 5: Benchmark Suites

## Why standardized benchmarks exist

Chapters 3-4 covered *bespoke* evaluation — you design a rubric, you evaluate your specific model on your specific task. Benchmarks flip this: a **fixed, public, standardized dataset** that every lab runs their model on, so results are comparable across papers, companies, and time. When a job posting says "improved MMLU by 4 points," they're leaning on this shared yardstick.

Know these four cold — what they test, how they're scored, and their specific weaknesses.

## MMLU (Massive Multitask Language Understanding)

**What it tests:** broad factual/academic knowledge across 57 subjects — law, medicine, history, math, professional exams — via multiple-choice questions.

**Format:** 4-choice multiple choice. Example style: "Which of the following is the primary function of the mitochondria? (A) Protein synthesis (B) ATP production (C) ..." Score = accuracy = fraction answered correctly.

**What it's good for:** breadth of world knowledge, a decent proxy for "how much did the model learn during pretraining."

**Weakness to name in an interview:** it's multiple choice, so a model can get partial credit from elimination strategies or superficial pattern matching, without deep understanding. It also doesn't test *generation* quality at all — a model could ace MMLU and still write terrible prose.

## HellaSwag

**What it tests:** commonsense reasoning about "what happens next" in everyday situations.

**Format:** given a sentence describing a scenario, pick the most plausible continuation from 4 options, where the wrong options are deliberately *adversarially generated* to be superficially plausible (this is called "Adversarial Filtering" — wrong answers are chosen specifically because an earlier, weaker model found them hard to distinguish from the right one).

**Worked intuition:** "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She..." — plausible continuation: "gets the dog wet, then walks away as it runs off." Implausible-but-tricky distractor: something that's grammatically fine but violates common sense about the scenario.

**Weakness:** because it's built via adversarial filtering against *older* models, newer/larger models can exploit the specific statistical artifacts of *how the wrong answers were generated*, rather than truly reasoning about the scenario — inflating scores in a way that doesn't reflect genuine commonsense improvement.

## GSM8K (Grade School Math 8K)

**What it tests:** multi-step arithmetic word-problem reasoning.

**Format:** open-ended (not multiple choice!) — the model must generate a full chain of reasoning and arrive at a final numeric answer, which is then string/number-matched against the ground truth.

**Worked example of a GSM8K-style problem:** "Sarah has 3 boxes of pencils. Each box has 12 pencils. She gives 8 pencils to a friend. How many pencils does she have left?" Correct reasoning: 3×12=36, 36-8=28. Scoring: did the model's final numeric answer equal 28? (Usually extracted via a regex/parser looking for the final number, often after a "The answer is" marker.)

**Why this benchmark matters so much in LLM eval history:** GSM8K is the benchmark most associated with demonstrating **chain-of-thought prompting** works — models scored far higher when prompted to "think step by step" before answering versus answering directly. It's a standard benchmark to cite when discussing reasoning capability and CoT.

**Weakness:** exact-match on the final number is brittle — a model with correct reasoning but a formatting slip (e.g., outputs "28 pencils" vs. expected "28") can get marked wrong if the parser is strict; conversely, a model can occasionally get the right final number via flawed reasoning (lucky arithmetic) and get marked correct despite bad process.

## HumanEval

**What it tests:** functional code generation.

**Format:** the model is given a function signature + docstring (e.g., "def has_close_elements(numbers, threshold): '''Check if any two numbers are closer than threshold'''") and must generate the function body. Scoring is **not** string match — it's **execution-based**: the generated code is run against a held-out set of unit tests, and it's marked correct only if it passes all of them.

**The pass@k metric — this is a specific formula interviewers like to probe:**

$$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

where $n$ = total samples generated per problem, $c$ = number of those samples that pass all tests, $k$ = how many attempts you're "allowed."

**Worked numerical example.** For a given coding problem, you sample $n=10$ completions from the model, and $c=3$ of them pass all unit tests. What's pass@1?

$$\text{pass@1} = 1 - \frac{\binom{10-3}{1}}{\binom{10}{1}} = 1 - \frac{\binom{7}{1}}{\binom{10}{1}} = 1 - \frac{7}{10} = 0.30$$

**Intuition check:** this matches the naive expectation — if 3/10 samples pass, the probability a single random sample passes is 3/10 = 0.30. The formula is really computing "probability that at least one of $k$ randomly-drawn-without-replacement samples is among the $c$ successes" — it exists in this combinatorial form (rather than just resampling k times, which would be a noisier/biased estimator) so it can be computed *exactly* from a single batch of $n$ samples, unbiased, without needing to actually redraw.

**Why pass@k specifically, not just accuracy:** code generation is naturally used with multiple sampling in practice (e.g., Copilot-style tools generate several suggestions), so "does at least one of k attempts work" is a more realistic production metric than "did the single greedy-decoded output work."

## The contamination problem — the single most important caveat in this whole chapter

**The core issue:** these benchmarks are public and have existed on the internet for years. Modern LLMs are pretrained on enormous scrapes of the internet. If GSM8K's actual questions-and-answers appear (even indirectly, via forum discussions, GitHub repos, or blog posts referencing them) in the pretraining corpus, the model isn't "reasoning" to solve them — it may be **recalling memorized answers**.

**Why this is such a big deal in interviews right now:** contamination inflates benchmark numbers in ways that don't reflect real capability, and it's a known, ongoing problem — labs have been caught (and have self-reported) contamination issues, and it's part of why benchmark numbers across model releases aren't always apples-to-apples.

**How labs try to detect/mitigate it (know at least 2-3 of these):**
- **N-gram overlap checks** — search the pretraining corpus for exact or near-exact matches to benchmark question text; flag/report the overlap percentage.
- **Canary strings** — some benchmark creators embed a unique, unlikely string (a "canary") into the benchmark file itself; if a model can be shown to have seen that exact string, it's proof the raw benchmark file leaked into training data.
- **Held-out / fresh benchmarks** — construct a *new* test set with the same format and difficulty as an established benchmark, but using recent/novel content that couldn't have been in any pretraining cutoff, then compare performance on old vs. new to check for a gap (a big drop on the fresh set is a contamination signal).
- **Perturbation testing** — slightly rephrase benchmark questions (change surface wording, keep the underlying problem identical) and see if the score holds; if performance drops sharply on a rephrased-but-equivalent version, that suggests the original score relied partly on memorized surface form rather than genuine reasoning.

**Interview-ready one-liner:** *"Public benchmark scores should be read skeptically — I'd want to know whether contamination checks (n-gram overlap, canary strings, or comparison against a fresh held-out variant) were done, because a high MMLU or GSM8K score can partly reflect memorization rather than generalization."*

## Quick check

A new 70B model claims to beat GPT-4 on GSM8K by 5 points. What's the single most important follow-up question to ask before trusting that number?

Was contamination checked? Specifically: was there an n-gram overlap analysis against the pretraining corpus, and/or was the score validated on a fresh, held-out set of equivalent-difficulty math problems that couldn't have leaked into training? Without that, a 5-point GSM8K gap could be partially or entirely a memorization artifact rather than a genuine reasoning improvement.

---

Chapter 6 is Task-Specific Evals — RAG (faithfulness/relevance), summarization, code generation, and agent evaluation, where general benchmarks like the ones above don't apply. Want me to continue?
