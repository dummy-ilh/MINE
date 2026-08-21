# Chapter 2: Classic NLP Metrics — BLEU, ROUGE, Perplexity

## Why start here if they're "old" and "flawed"?

Because every interviewer expects you to know exactly *why* they're flawed, with numbers — not just "they're outdated." Vague criticism sounds like you memorized a blog post. Precise criticism, with a worked example, sounds like you've actually used them.

## Perplexity — "how surprised is the model?"

**Intuition first.** Perplexity measures how well a model predicts a sequence of text. If the model assigns high probability to the actual next word every time, it's "not surprised" — low perplexity. If it keeps assigning low probability to what actually comes next, it's "very surprised" — high perplexity.

**The formula:**

$$PPL(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_1, \dots, w_{i-1})\right)$$

That's a mouthful — let's build it with numbers instead.

**Worked numerical example.** Say the true sentence is "the cat sat down" (4 words). Your model, at each position, assigns this probability to the *actual* next word:

| word | model's P(actual word) |
|---|---|
| the | 0.20 |
| cat | 0.10 |
| sat | 0.05 |
| down | 0.25 |

Step 1: take log of each probability (natural log):
- log(0.20) = -1.609
- log(0.10) = -2.303
- log(0.05) = -2.996
- log(0.25) = -1.386

Step 2: average them: (-1.609 - 2.303 - 2.996 - 1.386) / 4 = -8.294 / 4 = -2.074

Step 3: negate and exponentiate: PPL = exp(2.074) ≈ **7.96**

**How to read that number:** perplexity ≈ 8 means "on average, the model was as confused as if it had to pick uniformly among ~8 equally likely words at each step." Lower is better. A perfect model (P=1 every time) gets perplexity = 1. A model that's totally clueless across a 50,000-word vocabulary approaches perplexity = 50,000.

**Why it fails for LLM evals:** Perplexity only tells you the model is a good *language model* — fluent, grammatical, statistically plausible. It says nothing about whether the content is *true*, *helpful*, or *on-topic*. A model can hallucinate fluently and still get great perplexity, because hallucinated text can still be high-probability, grammatical text. This is exactly the offline-intrinsic trap from Chapter 1.

## BLEU — "how much n-gram overlap with a reference?"

**Intuition first.** Originally built for machine translation. You have a reference (human) translation and a candidate (model) translation. BLEU asks: what fraction of the model's word chunks (n-grams) actually appear in the reference?

**Worked numerical example.**

Reference: "the cat is on the mat"
Candidate: "the cat is on mat"

**Step 1 — unigram precision:** of the candidate's 5 words, how many appear in the reference (with clipping — can't double-count)?
Candidate words: the, cat, is, on, mat — all 5 appear in reference. Precision-1 = 5/5 = 1.0

**Step 2 — bigram precision:** candidate bigrams: "the cat", "cat is", "is on", "on mat" (4 bigrams).
Reference bigrams: "the cat", "cat is", "is on", "on the", "the mat" (5 bigrams).
Matches: "the cat" ✓, "cat is" ✓, "is on" ✓, "on mat" ✗ (reference has "on the", not "on mat").
Precision-2 = 3/4 = 0.75

**Step 3 — combine (geometric mean of precision-1 through precision-4, typically):**
For simplicity with just two orders: geometric mean of (1.0, 0.75) = √(1.0 × 0.75) ≈ 0.866

**Step 4 — brevity penalty (BP):** candidate has 5 words, reference has 6. Candidate is shorter, so BLEU penalizes it (otherwise a model could just output one safe word and get artificially high precision):

$$BP = \exp\left(1 - \frac{\text{ref\_len}}{\text{cand\_len}}\right) = \exp(1 - 6/5) = \exp(-0.2) ≈ 0.819$$

**Final BLEU ≈ 0.866 × 0.819 ≈ 0.71**

**Why it fails for LLM evals:** BLEU only rewards *surface-level word overlap*. "The cat is on the mat" vs. "A feline rests atop the rug" — semantically identical, but BLEU would score this near zero because almost no words overlap. For open-ended generation (chatbot replies, creative writing, code explanations) where there's no single "correct" reference, BLEU is close to meaningless — there often isn't one right answer to overlap against.

## ROUGE — "BLEU's cousin, but recall-oriented, used for summarization"

**Intuition first.** BLEU asks "how much of what the model said is correct" (precision-flavored). ROUGE asks "how much of what *should have been said* did the model capture" (recall-flavored). That's why BLEU dominates translation and ROUGE dominates summarization — in summarization, missing key information is the bigger sin than including extra words.

**Worked numerical example — ROUGE-1 (unigram recall):**

Reference summary: "the market fell sharply today"  (6 words)
Candidate summary: "the market fell today"  (4 words)

Overlapping unigrams: the, market, fell, today → 4 words match.

ROUGE-1 recall = matches / reference length = 4/6 ≈ **0.667**
ROUGE-1 precision = matches / candidate length = 4/4 = **1.0**
ROUGE-1 F1 = 2 × (P × R)/(P + R) = 2 × (1.0 × 0.667)/(1.667) ≈ **0.80**

**Why it fails for LLM evals:** Same core problem as BLEU — surface overlap, not meaning. A summary can paraphrase perfectly and score low ROUGE, or copy irrelevant phrases verbatim from the reference and score artificially high. It also can't detect **faithfulness** — a summary can score high ROUGE while still containing a hallucinated fact, as long as the wording happens to overlap.

## The common thread — why Chapter 4 (LLM-as-judge) exists

All three metrics — perplexity, BLEU, ROUGE — share one blind spot: **none of them understand meaning.** They're all counting/statistical measures over tokens. Once LLMs started doing open-ended, multi-correct-answer tasks (chat, reasoning, summarization with no single reference), the field needed something that evaluates *semantic* and *factual* quality. That gap is exactly why LLM-as-a-judge (Chapter 4) and embedding-based semantic metrics emerged.

**Interview-ready one-liner to have ready:** *"BLEU/ROUGE measure lexical overlap, perplexity measures fluency — none of them measure truthfulness, helpfulness, or semantic correctness, which is why modern LLM evals lean on human eval and LLM-as-judge instead."*

## Quick check

Two candidate answers to "What's the capital of France?":
- A: "The capital of France is Paris."
- B: "Paris is France's capital city."

If the reference is exactly "The capital of France is Paris," which one gets a much higher BLEU score, even though both are equally correct?

**A** — near-perfect n-gram overlap with the reference. B says the same true thing with different wording and would score much lower. This is the exact failure mode you should be able to explain live in an interview.

---

Chapter 3 is Human Evaluation — rubrics, pairwise comparison, and the stats behind inter-annotator agreement (Cohen's kappa, Krippendorff's alpha). Want me to continue?
