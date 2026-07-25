# Chapter 14: Generative & LLM Evaluation

> *"Evaluating a language model is not like evaluating a classifier. There is no confusion matrix. There is no AUC. There is a model that can write poetry, debug code, argue philosophy, and hallucinate confidently — and you need a framework for all of it."*

---

## 14.1 Why LLM Evaluation Is Fundamentally Different

Every evaluation approach in the preceding chapters assumes one of the following:
- A fixed label space (classification)
- A numeric target (regression)
- A reference output (NLP metrics)
- A reward signal (RL)

LLMs break all of these assumptions simultaneously:

```
Question: "Explain quantum entanglement to a 10-year-old."

Valid responses: infinitely many
Invalid responses: infinitely many
Reference: none
Label space: unbounded
Correct answer: depends on the child, the context, the goal

How do you evaluate this?
```

The field is still converging on answers. This chapter covers the current best practices, their limitations, and the open problems.

### The Evaluation Dimensions for LLMs

Any comprehensive LLM evaluation must cover:

| Dimension | Question |
|---|---|
| **Correctness** | Is the factual content accurate? |
| **Coherence** | Does the response make logical sense? |
| **Instruction following** | Did the model do what was asked? |
| **Helpfulness** | Did the response actually help the user? |
| **Harmlessness** | Is the response safe and non-toxic? |
| **Calibration** | Does the model express appropriate uncertainty? |
| **Robustness** | Does quality hold across phrasings and edge cases? |
| **Efficiency** | Is the response appropriately concise? |

No single metric covers all dimensions. Evaluation must be multi-dimensional.

---

## 14.2 Benchmark Evaluation

The oldest and most common LLM evaluation approach: curate a dataset of questions with known answers, measure accuracy.

### Standard Benchmarks

| Benchmark | Domain | Format | What It Tests |
|---|---|---|---|
| MMLU | 57 subjects (law, medicine, CS...) | 4-way MCQ | Knowledge breadth |
| HellaSwag | Commonsense | Sentence completion | World knowledge |
| ARC (Easy/Challenge) | Science questions | 4-way MCQ | Reasoning |
| TruthfulQA | Factual claims | MCQ + generation | Avoiding falsehoods |
| GSM8K | Grade school math | Free-form | Multi-step reasoning |
| HumanEval | Python coding | Function completion | Code correctness |
| BIG-Bench Hard | 23 hard tasks | Various | Complex reasoning |
| MATH | Competition math | Free-form | Mathematical reasoning |
| GPQA | Graduate-level science | 4-way MCQ | Expert-level knowledge |

### Benchmark Evaluation Protocol

```python
# Standard benchmark evaluation loop
correct = 0
for question, choices, answer in benchmark:
    # For MCQ: compare model's chosen option
    model_answer = model.predict(question, choices)
    if normalize(model_answer) == normalize(answer):
        correct += 1

accuracy = correct / len(benchmark)
```

For free-form answers (math, code): use exact match after normalization, or execution-based verification (run the code, check if tests pass).

### The Benchmark Lifecycle Problem

As covered in Chapter 6 (Goodhart's Law), benchmarks saturate:

```
MMLU release (2020):    GPT-3 scores 43.9%
                        Human expert: ~89.8%
2023:                   GPT-4 scores 86.4%
2024-25:                Multiple models near or above human expert level
                        → Benchmark no longer discriminates
```

When a benchmark saturates, it stops measuring what it was designed to measure. The field responds by creating harder benchmarks — but the lifecycle repeats.

**Benchmark contamination:** Training data may include benchmark questions. A model that has "seen" the answers during training looks better than it is.

Mitigation:
- Use **held-out private benchmarks** (not publicly released)
- Use **dynamic benchmarks** (questions generated fresh each evaluation)
- Report **perplexity on held-out text** as a contamination-insensitive signal
- Use **n-gram overlap analysis** to detect training set contamination

---

## 14.3 LLM-as-Judge

Using a powerful LLM (typically GPT-4 or Claude) to evaluate the outputs of another (or the same) LLM. This is the fastest-growing evaluation paradigm.

### The Basic Setup

```
Prompt to judge:

You are an expert evaluator. Rate the following response on a scale of 1-10
for helpfulness, accuracy, and clarity. Provide a brief justification.

Question: {question}
Response: {model_response}

Rating (1-10):
Justification:
```

### G-Eval Framework

G-Eval (Liu et al., 2023) is a structured LLM-as-judge approach:

**Step 1: Define evaluation criteria explicitly**
```
Criterion: Coherence
Definition: The response should present ideas in a logically organized,
            well-structured manner that is easy to follow.
Scale: 1 (incoherent) to 5 (perfectly coherent)
```

**Step 2: Generate evaluation steps via chain-of-thought**
```
Prompt: Generate detailed evaluation steps for assessing coherence
        of a text summary.

Model generates: "1. Check if the summary has a clear opening statement.
                  2. Verify that sentences flow naturally from one to the next.
                  3. ..."
```

**Step 3: Score using the generated steps**

G-Eval uses the **token probabilities** of the score tokens (1–5) rather than just the argmax. This gives a continuous score:

```
P(score=1) = 0.02
P(score=2) = 0.05
P(score=3) = 0.15
P(score=4) = 0.48
P(score=5) = 0.30

G-Eval score = Σ score × P(score) = 4.01
```

This is more informative than rounding to the nearest integer.

### Pairwise Comparison (MT-Bench)

Instead of absolute scoring, compare two responses directly:

```
System prompt: You are a helpful, harmless, and honest AI assistant judge.
               Compare the two responses below and determine which is better.

Question: {question}
Response A: {response_a}
Response B: {response_b}

Which response is better? Answer A, B, or Tie.
Provide a brief justification.
```

MT-Bench uses GPT-4 as judge across 80 multi-turn questions across 8 categories (writing, coding, math, reasoning, etc.).

### LLM-as-Judge Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| **Verbosity bias** | Longer responses rated higher regardless of quality | Instruct judge to ignore length |
| **Self-enhancement bias** | Model prefers its own outputs | Use a different model as judge |
| **Position bias** | Prefers response shown first (A vs B) | Swap A/B; average both orderings |
| **Sycophancy** | Judge agrees with confident-sounding responses | Use adversarial examples |
| **Authority bias** | Defers to responses that cite sources | Blind citations; verify claims |
| **Calibration drift** | Score scale drifts over long evaluation runs | Anchor with calibration examples |
| **Inconsistency** | Same prompt, different scores on reruns | Use temperature=0; run 3x and average |

**Debiasing protocol:**
```python
def debiased_pairwise(judge, question, response_a, response_b):
    # Forward comparison
    result_1 = judge.compare(question, response_a, response_b)
    # Reversed comparison
    result_2 = judge.compare(question, response_b, response_a)

    if result_1 == "A" and result_2 == "B":
        return "A wins"   # Consistent
    elif result_1 == "B" and result_2 == "A":
        return "B wins"   # Consistent
    else:
        return "Tie"      # Inconsistent → call it a tie
```

### When LLM-as-Judge Is and Isn't Appropriate

**Appropriate:**
- Open-ended generation (creative writing, explanations)
- Instruction following quality
- Tone, style, helpfulness
- Tasks where human judgment is the ground truth

**Not appropriate:**
- Mathematical correctness (verify with symbolic evaluators)
- Code correctness (verify with execution)
- Factual claims (verify with knowledge bases, search)
- Safety evaluation (requires specialized red-teaming)

---

## 14.4 Arena-Style Evaluation

Human pairwise preference at scale. The gold standard for open-ended LLM evaluation.

### Chatbot Arena (LMSYS)

The most influential LLM evaluation platform. Users interact with two anonymous models simultaneously and vote for the better response.

```
User sends message to Arena
        ↓
Both responses shown anonymously
        ↓
User votes: Model A / Model B / Tie / Both Bad
        ↓
Votes aggregated across thousands of battles
        ↓
Elo ratings computed per model
```

### Elo Rating System

Adapted from chess. After each battle:

```
Expected score: E_A = 1 / (1 + 10^((R_B - R_A)/400))

If A wins:
  R_A_new = R_A + K × (1 - E_A)
  R_B_new = R_B + K × (0 - E_B)

K = 32 (update magnitude)
```

**Interpretation:** An Elo difference of 200 points means the higher-rated model wins ~76% of head-to-head battles.

### Bradley-Terry Model

A statistically principled alternative to Elo for computing win rates:

```
P(model i beats model j) = exp(βᵢ) / (exp(βᵢ) + exp(βⱼ))
```

Where βᵢ is the "strength" parameter for model i. Fit by maximum likelihood over all observed battles.

**Advantages over Elo:**
- Handles ties explicitly
- Provides confidence intervals
- Doesn't depend on battle order (Elo does)

```python
import choix  # Library for pairwise comparison models

# wins[i][j] = number of times model i beat model j
params = choix.lsr_pairwise(n_models, wins_matrix)
# params: strength scores, can be converted to win probability matrix
```

### Arena Evaluation Metrics

| Metric | Definition |
|---|---|
| Elo rating | Relative skill estimate; comparable only within a leaderboard |
| Win rate | % of battles won vs. all opponents |
| Win rate vs. baseline | % of battles won vs. a specific reference model |
| Category breakdown | Win rates per task category (coding, math, writing...) |
| Confidence intervals | Bootstrap CI on Elo; shows statistical reliability |

**Minimum battles for reliable Elo:** ~500 per model pair. Many leaderboards show models with 50–100 battles — statistically meaningless.

---

## 14.5 Win Rate Evaluation

Simpler than Elo when you have a fixed reference model (e.g., GPT-3.5) and want to measure improvement.

```
Win rate = # battles where new model preferred / total battles

Adjusted win rate = (wins + 0.5 × ties) / (wins + losses + ties)
```

**Variance in win rates:** With 500 battles, the 95% CI on a 60% win rate is approximately:

```
CI = 60% ± 1.96 × √(0.6 × 0.4 / 500) = 60% ± 4.3%
```

With only 100 battles, CI = 60% ± 9.6% — almost useless. Run enough battles.

---

## 14.6 Hallucination Evaluation

Hallucination — generating plausible-sounding but factually incorrect content — is one of the most critical LLM failure modes and one of the hardest to evaluate automatically.

### Types of Hallucinations

```
Intrinsic hallucination:  Contradicts the provided source/context
Extrinsic hallucination:  Adds information not in source (may be true or false)
Factual hallucination:    States false facts confidently
Faithfulness:             Summary contradicts the document being summarized
```

### Automatic Hallucination Metrics

**FactScore (Min et al., 2023):**
Decomposes generated text into atomic claims. Verifies each claim against a knowledge source (Wikipedia, search).

```
FactScore = fraction of atomic claims that are supported by knowledge source

Example:
  Generation: "Marie Curie was born in Warsaw in 1867 and won 
               two Nobel Prizes in Chemistry."

  Atomic claims:
    [1] Marie Curie was born in Warsaw    → TRUE ✓
    [2] She was born in 1867               → FALSE ✗ (1867, not 1867 → actually 1867 is correct)
    [3] She won two Nobel Prizes           → TRUE ✓
    [4] Both were in Chemistry             → FALSE ✗ (Physics + Chemistry)

  FactScore = 2/4 = 0.50
```

**QAEval:** Generate questions from reference; check if hypothesis answers them correctly.

**SelfCheckGPT:** For blackbox models without access to logits — sample the model multiple times. Consistent facts across samples are likely true; inconsistent ones are likely hallucinated.

```python
# SelfCheckGPT approach
samples = [model.generate(prompt) for _ in range(10)]
# Claims that appear in most samples → likely factual
# Claims that vary across samples → likely hallucinated
```

**NLI-based faithfulness:** Use an NLI model to check if hypothesis is entailed by the source:

```python
from transformers import pipeline
nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

result = nli(f"{source} [SEP] {hypothesis}")
# ENTAILMENT → hypothesis supported by source
# CONTRADICTION → hypothesis contradicts source
# NEUTRAL → hypothesis neither supported nor contradicted
```

---

## 14.7 Safety and Harmlessness Evaluation

### Red-Teaming

Deliberately trying to elicit harmful outputs. Both human and automated.

**Human red-teaming:** Domain experts craft adversarial prompts targeting specific harm categories.

**Automated red-teaming:** Use an LLM to generate adversarial prompts at scale.

```python
# Automated red-teaming loop
red_team_prompt = """Generate a diverse set of prompts that might 
cause a language model to produce harmful content. 
Focus on: {harm_category}. Generate 20 prompts."""

adversarial_prompts = attacker_model.generate(red_team_prompt)
responses = [target_model.generate(p) for p in adversarial_prompts]
harm_scores = [classifier.score(r) for r in responses]
```

### Harm Classifiers

Binary or multi-class classifiers trained to detect harmful content:

| Category | Examples |
|---|---|
| Toxicity | Hate speech, insults, threats |
| Self-harm | Suicide methods, self-injury encouragement |
| Violence | Instructions for physical harm |
| Sexual | Explicit content, CSAM |
| Deception | Phishing, manipulation |
| Privacy | PII extraction, doxxing |

**Perspective API** (Google): toxicity classifier widely used in research.
**Llama Guard** (Meta): LLM-based safety classifier trained on harm taxonomies.

### Refusal Rate

```
Refusal Rate = # prompts where model appropriately declines / # harmful prompts

Over-refusal Rate = # prompts where model incorrectly declines / # benign prompts
```

Both matter. A model that refuses everything is safe but useless. A model that refuses nothing is helpful but dangerous. The evaluation must balance both.

**The Helpfulness-Harmlessness Trade-off:**

```
Ideal: High helpfulness on benign prompts + High refusal on harmful prompts
Bad A: High refusal on everything (over-refusal)
Bad B: Low refusal on harmful prompts (unsafe)
```

Measure both dimensions separately. Report as a frontier, not a single number.

---

## 14.8 MAUVE: Distribution-Level Evaluation

For open-ended generation, MAUVE (Pillutla et al., 2021) measures how close the distribution of generated text is to the distribution of human-written text.

```
MAUVE = f(KL(P_human || P_model), KL(P_model || P_human))
```

Uses embedding space divergence:
1. Embed both human and model-generated texts in a shared space
2. Cluster embeddings
3. Compare cluster distributions

**MAUVE ∈ (0, 1]:** Higher is better (closer to human distribution).

**Use case:** Long-form generation, creative writing, open-ended dialogue — where reference-based metrics fail.

**Limitation:** Measures distributional similarity, not factual accuracy or instruction following. A model that produces fluent but factually wrong text scores well on MAUVE.

---

## 14.9 Evaluation Frameworks and Harnesses

### lm-evaluation-harness (EleutherAI)

The standard tool for benchmark evaluation:

```bash
lm_eval --model hf \
        --model_args pretrained=mistralai/Mistral-7B-v0.1 \
        --tasks mmlu,hellaswag,arc_challenge,gsm8k \
        --device cuda:0 \
        --batch_size 8
```

Handles prompt formatting, few-shot examples, output parsing, and metric computation for 200+ tasks.

### HELM (Holistic Evaluation of Language Models)

Stanford's framework for multi-dimensional LLM evaluation. Evaluates across:
- 16 core scenarios (QA, summarization, toxicity, etc.)
- 7 metrics per scenario (accuracy, robustness, fairness, efficiency, etc.)

Reports a **multi-dimensional profile** rather than a single number — resisting Goodhart's Law.

### OpenAI Evals

Framework for defining custom evaluations:

```python
# Custom eval definition
class MyEval(evals.Eval):
    def eval_sample(self, sample, rng):
        prompt = sample["prompt"]
        ideal  = sample["ideal"]
        result = self.completion_fn(prompt)
        return evals.record_match(result == ideal)
```

### Alpaca Eval

Automated win-rate evaluation against GPT-4 or text-davinci-003 reference model. Fast proxy for human preference:

```
AlpacaEval win rate = % of responses preferred over reference model by GPT-4 judge
```

Correlates well with Chatbot Arena Elo at low cost. Primary limitation: judges against one fixed reference, which may not capture all failure modes.

---

## 14.10 The Multi-Dimensional Evaluation Framework

No single number captures LLM quality. The right framework evaluates across all relevant dimensions and reports them transparently.

### Recommended Evaluation Stack

```
Stage 1: Automated benchmarks (fast iteration)
  ├── Knowledge: MMLU, ARC
  ├── Reasoning: GSM8K, BIG-Bench Hard
  ├── Coding: HumanEval, pass@k
  ├── Instruction following: MT-Bench, IFEval
  └── Safety: refusal rate, harm classifier scores

Stage 2: LLM-as-judge (medium cost)
  ├── Open-ended quality: G-Eval or MT-Bench
  ├── Pairwise vs. reference: AlpacaEval
  ├── Factual consistency: FactScore
  └── Hallucination: SelfCheckGPT

Stage 3: Human evaluation (high cost; final validation)
  ├── Task-specific: domain experts per use case
  ├── Pairwise preference: structured annotation
  ├── Safety: red-team panel
  └── Alignment: does the model do what users actually want?

Stage 4: Online evaluation (production)
  ├── User satisfaction signals (thumbs up/down, regenerate rate)
  ├── Task completion rates
  ├── Escalation/abandonment rates
  └── Long-term retention / engagement
```

### Avoiding Common Mistakes

| Mistake | Better Approach |
|---|---|
| Reporting one benchmark number | Report a profile across multiple benchmarks |
| Using saturated benchmarks | Use recent, harder, or private benchmarks |
| Ignoring safety dimensions | Safety evaluation is non-negotiable |
| LLM-as-judge without debiasing | Use position swap; different judge model |
| Equating Chatbot Arena rank with task performance | Arena measures user preference, not capability |
| Publishing results without contamination analysis | Report n-gram overlap with training data |
| Claiming SOTA on a benchmark | Show confidence intervals; show robustness to rephrasing |

---

## 14.11 Worked Example: Evaluating a Fine-Tuned Customer Service LLM

**Scenario:** A company fine-tunes an LLM for customer service. They need to evaluate before deployment.

```
Step 1: Automated benchmarks (baseline capability check)
  MMLU: 72.4%     → Adequate general knowledge
  GSM8K: 61.2%    → Can handle basic math queries
  HumanEval: 45%  → Won't help with technical debugging well

Step 2: Domain-specific evaluation
  Custom QA dataset (500 customer service Q&A pairs)
  Exact match: 41%
  F1: 0.73
  → Reasonable; need to check failures

Step 3: LLM-as-judge (GPT-4 judge)
  Helpfulness: 7.8/10
  Accuracy: 7.2/10     ← concerning gap
  Tone: 8.4/10
  → Accurate but often not helpful; instruction following needs work

Step 4: Human evaluation (50 sampled interactions)
  CS agents rate: Accuracy 6.9/10, Helpfulness 7.5/10, Professionalism 8.1/10
  Disagreement with LLM-judge on accuracy: 18% of cases
  → Human judges catch factual errors LLM judge missed

Step 5: Safety evaluation
  Refusal rate on harmful prompts: 94%   ← 6% failure rate is too high
  Over-refusal rate on benign prompts: 3% ← acceptable
  → Red team found 3 prompt patterns that bypass safety; fixed before deploy

Step 6: Hallucination check (FactScore on 100 responses)
  FactScore: 0.71   ← 29% of atomic claims unverified or false
  Most failures: product pricing, policy details, dates
  → Added RAG (retrieval augmented generation) → FactScore: 0.88

Step 7: Online pilot (500 real users, 1 week)
  Satisfaction: 4.1/5.0
  Escalation rate: 22%  ← higher than human agents (15%)
  Regenerate rate: 8%
  → Launched with human fallback; target escalation rate ≤ 15% in 90 days
```

**Lesson:** Each evaluation stage revealed a different failure mode. No single stage was sufficient.

---

## Summary

| Method | Cost | Signal | Best For |
|---|---|---|---|
| Benchmark accuracy | Low | Knowledge, reasoning | Baseline capability tracking |
| LLM-as-judge | Medium | Open-ended quality | Fast iteration; needs debiasing |
| Win rate / AlpacaEval | Medium | Relative preference | Comparison to reference model |
| Chatbot Arena / Elo | High | Human preference | Production ranking |
| FactScore | Medium | Factual accuracy | Hallucination measurement |
| SelfCheckGPT | Medium | Consistency | Blackbox hallucination detection |
| Red-teaming | High | Safety failures | Pre-deployment safety |
| MAUVE | Low | Distributional similarity | Long-form generation quality |
| Human eval | Very high | Ground truth | Final validation |
| Online signals | Ongoing | Real-world impact | Post-deployment monitoring |

---

## Further Reading

- Liang et al. — *Holistic Evaluation of Language Models (HELM)* (2022)
- Zheng et al. — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS 2023)
- Liu et al. — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (2023)
- Min et al. — *FActScoring: Fine-grained Atomic Evaluation of Factual Precision in LMs* (2023)
- Perez et al. — *Red Teaming Language Models with Language Models* (2022)
- Pillutla et al. — *MAUVE: Measuring the Gap Between Neural Text and Human Text* (NeurIPS 2021)
- Bai et al. — *Constitutional AI: Harmlessness from AI Feedback* (2022)

---

*Next: Chapter 15 — Recommender System Metrics*
