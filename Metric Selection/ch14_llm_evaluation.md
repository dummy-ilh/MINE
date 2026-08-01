# Chapter 14: Generative & LLM Evaluation
### Apple MLE Interview Master Notes — Improved & Expanded Edition

> *"Evaluating a language model is not like evaluating a classifier. There is no confusion matrix. There is no AUC. There is a model that can write poetry, debug code, argue philosophy, and hallucinate confidently — and you need a framework for all of it."*

---

## 14.0 Master Cheat Sheet

### 14.0.1 Evaluation Methods at a Glance

| Method | Cost | Signal Type | Best For |
|---|---|---|---|
| Benchmark accuracy | Low | Knowledge, reasoning | Baseline capability tracking |
| LLM-as-judge (G-Eval, MT-Bench) | Medium | Open-ended quality | Fast iteration; requires debiasing |
| Win rate / AlpacaEval | Medium | Relative preference | Comparison to a reference model |
| Chatbot Arena / Elo | High | Human preference at scale | Production-grade ranking |
| FactScore | Medium | Factual accuracy per claim | Hallucination measurement |
| SelfCheckGPT | Medium | Sampling consistency | Blackbox hallucination detection |
| Red-teaming | High | Safety failure modes | Pre-deployment safety validation |
| MAUVE | Low | Distributional similarity | Long-form generation quality |
| Human evaluation | Very high | Ground truth | Final validation before launch |
| Online signals | Ongoing | Real-world impact | Post-deployment monitoring |

### 14.0.2 Key Facts to Keep at the Front of Your Mind

| # | Fact | Detail |
|---|---|---|
| 1 | No single metric covers all LLM dimensions | Correctness, coherence, instruction-following, safety, calibration, efficiency — all need separate measurement |
| 2 | Benchmark contamination is pervasive | Models may have seen benchmark questions in training data; report n-gram overlap analysis |
| 3 | LLM-as-judge position bias | Swap A/B ordering in pairwise comparisons and average both results to debias |
| 4 | G-Eval uses token probabilities, not argmax | Computes a continuous score as E[score] over P(score=k); more informative than rounding |
| 5 | Elo difference of 200 → 76% win rate | A model with 200 more Elo points wins ~76% of head-to-head battles |
| 6 | Bradley-Terry advantages over Elo | Handles ties explicitly, provides confidence intervals, order-independent |
| 7 | FactScore = fraction of atomic claims supported | Decomposes generation into verifiable atomic claims; checks each against a knowledge source |
| 8 | SelfCheckGPT works without logit access | Samples the model multiple times; consistent facts are likely true; variable facts are likely hallucinated |
| 9 | Refusal rate AND over-refusal rate both matter | Safety + utility must be evaluated together; refusing everything is not a solution |
| 10 | Online signals are the ultimate ground truth | Task completion, escalation rate, user satisfaction — real-world behavior overrides offline metrics |

---

## 14.1 Why LLM Evaluation Is Fundamentally Different

### 14.1.1 How LLMs Break Classical Evaluation Assumptions

Every evaluation approach in preceding chapters assumes at least one of the following:
- A fixed label space (classification)
- A numeric target (regression)
- A reference output (NLP metrics like BLEU)
- A reward signal (reinforcement learning)

LLMs break all of these assumptions simultaneously. Consider this prompt:

```
Question: "Explain quantum entanglement to a 10-year-old."

Valid responses:   infinitely many
Invalid responses: infinitely many
Reference answer:  none
Label space:       unbounded
Correct answer:    depends on the child, the context, and the goal
```

There is no single correct output to check against, no fixed class to predict, and no reward function that captures "this explanation is good for this particular child." The field is still converging on best practices, and this chapter covers where it currently stands.

### 14.1.2 The Eight Evaluation Dimensions for LLMs

Any comprehensive LLM evaluation must cover multiple independent dimensions. Excelling on one does not guarantee performance on others.

| # | Dimension | Core Question |
|---|---|---|
| 1 | Correctness | Is the factual content accurate? |
| 2 | Coherence | Does the response make logical sense? |
| 3 | Instruction following | Did the model do exactly what was asked? |
| 4 | Helpfulness | Did the response actually help the user accomplish their goal? |
| 5 | Harmlessness | Is the response safe, non-toxic, and non-deceptive? |
| 6 | Calibration | Does the model express appropriate uncertainty — saying "I don't know" when it doesn't? |
| 7 | Robustness | Does quality hold across different phrasings and edge cases? |
| 8 | Efficiency | Is the response appropriately concise, without padding or unnecessary verbosity? |

**Plain-English insight:** A model can ace benchmark accuracy (dimension 1) while being incoherent (dimension 2), refusing benign requests (dimension 5), and hallucinating confidently (dimension 6). This is why no single number characterizes LLM quality.

---

## 14.2 Benchmark Evaluation

### 14.2.1 What It Is

The oldest and most common LLM evaluation approach: curate a dataset of questions with known answers, run the model, and measure accuracy. Benchmarks provide reproducible, low-cost, comparable results across models and time.

### 14.2.2 Standard Benchmark Reference Table

| # | Benchmark | Domain | Format | What It Tests |
|---|---|---|---|---|
| 1 | MMLU | 57 subjects (law, medicine, CS, history...) | 4-way MCQ | Breadth of world knowledge |
| 2 | HellaSwag | Commonsense situations | Sentence completion | Commonsense world knowledge |
| 3 | ARC (Easy / Challenge) | Grade-school science | 4-way MCQ | Factual reasoning |
| 4 | TruthfulQA | Common misconceptions | MCQ + generation | Avoiding confident falsehoods |
| 5 | GSM8K | Grade school math | Free-form text | Multi-step mathematical reasoning |
| 6 | HumanEval | Python programming | Function completion | Code correctness |
| 7 | BIG-Bench Hard | 23 diverse hard tasks | Various | Complex, adversarial reasoning |
| 8 | MATH | Competition mathematics | Free-form text | Advanced mathematical reasoning |
| 9 | GPQA | Graduate-level science | 4-way MCQ | Expert-level domain knowledge |

### 14.2.3 Evaluation Protocol

```python
# Standard benchmark evaluation loop
correct = 0
for question, choices, answer in benchmark:
    model_answer = model.predict(question, choices)
    if normalize(model_answer) == normalize(answer):
        correct += 1

accuracy = correct / len(benchmark)
```

For **free-form answers** (math, code): use exact match after normalization, or **execution-based verification** (run the generated code against test cases and check if they pass). Execution-based evaluation is much more reliable for code than string matching.

### 14.2.4 The Benchmark Lifecycle Problem (Goodhart's Law in Action)

Benchmarks have a predictable lifecycle: release → rapid improvement → saturation → irrelevance.

| Year | Event | GPT-3 / GPT-4 MMLU Score |
|---|---|---|
| 2020 | MMLU released | GPT-3: 43.9% |
| 2023 | GPT-4 released | GPT-4: 86.4% |
| 2024–25 | Multiple models near human expert level | Human expert baseline: ~89.8% |
| Now | Benchmark no longer discriminates | Replaced by GPQA, MATH, private evals |

Once a benchmark saturates, it stops measuring what it was designed to measure. The field responds by creating harder benchmarks — but the lifecycle repeats.

### 14.2.5 Benchmark Contamination

**What it is:** Training data may include benchmark questions and answers. A model that has "seen" the correct answers during training appears better than it truly is.

**Why it matters:** Contamination inflates benchmark scores without reflecting genuine capability. It makes cross-model comparisons unreliable.

**Mitigation strategies:**

| # | Strategy | How It Works |
|---|---|---|
| 1 | Held-out private benchmarks | Questions never released publicly; contamination is impossible |
| 2 | Dynamic benchmarks | Questions generated fresh at evaluation time from templates |
| 3 | Perplexity on held-out text | Contamination-insensitive signal; measures language modeling quality |
| 4 | N-gram overlap analysis | Check what fraction of benchmark questions appear in training data |

> **Apple Production Tip:** Any model evaluation for a production system should include a contamination analysis. Reporting benchmark results without it is insufficient for an engineering audience.

---

## 14.3 LLM-as-Judge

### 14.3.1 The Core Idea

LLM-as-judge means using a powerful frontier model (typically GPT-4 or Claude) to evaluate the outputs of another — or even the same — LLM. This is the fastest-growing evaluation paradigm because it scales cheaply to arbitrary tasks where ground-truth labels don't exist.

**Plain-English analogy:** Instead of writing a rubric and hiring human graders (expensive and slow), you hire a very capable AI grader that can follow complex rubrics on demand. The risk is that the AI grader has its own biases — which must be explicitly measured and corrected.

### 14.3.2 Basic Scoring Setup

```
Prompt to judge model:

You are an expert evaluator. Rate the following response on a scale
of 1–10 for helpfulness, accuracy, and clarity.
Provide a brief justification.

Question: {question}
Response: {model_response}

Rating (1–10):
Justification:
```

### 14.3.3 G-Eval: Structured LLM-as-Judge (Liu et al., 2023)

G-Eval improves on naive scoring in two key ways: explicit criteria and continuous scoring via token probabilities.

**Step 1 — Define evaluation criteria explicitly:**
```
Criterion:  Coherence
Definition: The response should present ideas in a logically organized,
            well-structured manner that is easy to follow.
Scale:      1 (completely incoherent) to 5 (perfectly coherent)
```

**Step 2 — Generate evaluation steps via chain-of-thought:**
```
Prompt: Generate detailed evaluation steps for assessing coherence
        of a text summary.

Model generates:
  1. Check if the summary has a clear opening statement.
  2. Verify that sentences flow naturally from one to the next.
  3. Confirm the summary ends with a clear concluding point.
  ...
```

**Step 3 — Score using token probabilities (not argmax):**

Rather than taking the single most likely score, G-Eval uses the **probability distribution** over score tokens to compute a continuous weighted average:

```
P(score=1) = 0.02
P(score=2) = 0.05
P(score=3) = 0.15
P(score=4) = 0.48
P(score=5) = 0.30

G-Eval score = Σ (score × P(score)) = 1×0.02 + 2×0.05 + 3×0.15 + 4×0.48 + 5×0.30
             = 4.01
```

This is more informative than rounding to the nearest integer (which would give 4 regardless of whether the distribution was [0,0,0,1,0] or [0,0,0.1,0.4,0.5]).

### 14.3.4 Pairwise Comparison (MT-Bench Approach)

Instead of rating a single response in isolation, compare two responses directly. This is often more reliable because relative judgments are easier than absolute ones.

```
System prompt:
  You are a helpful, harmless, and honest AI assistant judge.
  Compare the two responses below and determine which is better.

Question: {question}
Response A: {response_a}
Response B: {response_b}

Which response is better? Answer A, B, or Tie.
Provide a brief justification.
```

**MT-Bench** uses GPT-4 as judge across 80 multi-turn questions spanning 8 categories: writing, roleplay, extraction, reasoning, math, coding, knowledge, and STEM.

### 14.3.5 LLM-as-Judge Failure Modes and Mitigations

| # | Failure Mode | Description | Mitigation |
|---|---|---|---|
| 1 | Verbosity bias | Longer responses rated higher regardless of actual quality | Explicitly instruct the judge to ignore length |
| 2 | Self-enhancement bias | A model used as its own judge prefers its own outputs | Use a different, ideally more capable, model as judge |
| 3 | Position bias | Prefers whichever response is shown first (A vs. B) | Swap A/B; run both orderings; average results |
| 4 | Sycophancy | Agrees with confident-sounding responses even when wrong | Include adversarial examples where confident = wrong |
| 5 | Authority bias | Defers to responses that cite sources, regardless of accuracy | Blind citations; verify claims independently |
| 6 | Calibration drift | Score scale drifts over long evaluation runs | Anchor with fixed calibration examples at regular intervals |
| 7 | Inconsistency | Same prompt, different scores on reruns | Use temperature=0; run 3× and average |

**Debiasing protocol for pairwise comparison:**

```python
def debiased_pairwise(judge, question, response_a, response_b):
    # Forward: A shown first
    result_forward = judge.compare(question, response_a, response_b)
    # Reversed: B shown first
    result_reversed = judge.compare(question, response_b, response_a)

    if result_forward == "A" and result_reversed == "B":
        return "A wins"   # Consistent across orderings
    elif result_forward == "B" and result_reversed == "A":
        return "B wins"   # Consistent across orderings
    else:
        return "Tie"      # Inconsistent → treat as inconclusive
```

### 14.3.6 When LLM-as-Judge Is and Is Not Appropriate

| Appropriate Use | Inappropriate Use |
|---|---|
| Open-ended generation quality (style, tone, helpfulness) | Mathematical correctness (verify with symbolic evaluators) |
| Instruction-following assessment | Code correctness (verify with execution against test cases) |
| Explanation clarity and coherence | Factual claim verification (use knowledge bases or search) |
| Tasks where human judgment is the natural ground truth | Safety evaluation (requires specialized red-teaming, not judge LLMs) |

---

## 14.4 Arena-Style Evaluation

### 14.4.1 What It Is and Why It Matters

Arena-style evaluation collects **human pairwise preferences at scale** by having real users interact with two anonymous models simultaneously and vote for the better response. It is the current gold standard for open-ended LLM evaluation because it measures what actually matters: real human preference on real tasks.

**Chatbot Arena (LMSYS)** is the most influential platform. Users interact with two anonymous models and vote; thousands of battles per model accumulate into stable Elo ratings.

```
User sends a message to the Arena
           ↓
Both model responses shown side-by-side, anonymously
           ↓
User votes: Model A / Model B / Tie / Both Bad
           ↓
Votes aggregated across thousands of battles
           ↓
Elo ratings computed and ranked per model
```

### 14.4.2 Elo Rating System

Adapted from chess. After each battle, ratings update based on outcome vs. expectation:

```
Expected score for A: E_A = 1 / (1 + 10^((R_B − R_A) / 400))

If A wins:
  R_A_new = R_A + K × (1 − E_A)   ← A gains points
  R_B_new = R_B + K × (0 − E_B)   ← B loses points

K = 32 (controls how much each battle shifts the rating)
```

**Interpretation:** An Elo difference of 200 points means the higher-rated model wins approximately 76% of head-to-head battles. An Elo difference of 400 points → ~91% win rate.

**Minimum battles for a reliable Elo estimate:** approximately 500 per model. Many leaderboards show models with 50–100 battles — statistically unreliable.

### 14.4.3 Bradley-Terry Model: A Statistically Principled Alternative

The Bradley-Terry model provides a more rigorous alternative to Elo:

```
P(model i beats model j) = exp(βᵢ) / (exp(βᵢ) + exp(βⱼ))
```

Where βᵢ is a learned "strength" parameter for model i. Parameters are fit by maximum likelihood over all observed battles.

**Advantages over Elo:**

| Property | Elo | Bradley-Terry |
|---|---|---|
| Handles ties | No (requires workaround) | Yes (natively) |
| Confidence intervals | No | Yes (via MLE uncertainty) |
| Order dependence | Yes (earlier battles affect later ones) | No (fits all battles jointly) |
| Statistical rigor | Heuristic | Principled |

```python
import choix  # Library for pairwise comparison models

# wins[i][j] = number of times model i beat model j
params = choix.lsr_pairwise(n_models, wins_matrix)
# params: strength scores per model
# Convert to win probability: P(i beats j) = exp(params[i]) / (exp(params[i]) + exp(params[j]))
```

### 14.4.4 Arena Evaluation Metrics

| Metric | Definition | Note |
|---|---|---|
| Elo rating | Relative skill estimate | Comparable only within a single leaderboard; not absolute |
| Win rate | % of battles won vs. all opponents | Depends on which opponents were faced |
| Win rate vs. baseline | % of battles won vs. one specific reference model | More controlled comparison |
| Category breakdown | Win rates per task category (coding, math, writing...) | Reveals capability gaps |
| Bootstrap CI on Elo | 95% confidence interval via resampling | Essential for statistical significance claims |

---

## 14.5 Win Rate Evaluation

### 14.5.1 When to Use Win Rate vs. Elo

Win rate is appropriate when you have a **fixed reference model** (e.g., GPT-3.5) and want to measure improvement of a new model against it. It is simpler than Elo and doesn't require a full leaderboard.

```
Win rate           = # battles where new model preferred / total battles

Adjusted win rate  = (wins + 0.5 × ties) / (wins + losses + ties)
```

Adjusted win rate is preferred because it accounts for ties rather than discarding them.

### 14.5.2 Statistical Reliability of Win Rates

Win rate estimates have substantial variance at low battle counts. The 95% confidence interval for a 60% win rate:

| # Battles | 95% CI | Reliable? |
|---|---|---|
| 100 | 60% ± 9.6% | ❌ Barely distinguishable from 50% |
| 500 | 60% ± 4.3% | ⚠️ Marginal for small true differences |
| 1,000 | 60% ± 3.0% | ✅ Reliable for practical decisions |
| 5,000 | 60% ± 1.4% | ✅ High precision |

**The formula:** CI = win_rate ± 1.96 × √(p(1−p) / n)

> **Apple Production Tip:** Never report a win rate from fewer than 500 battles as meaningful. Always report the confidence interval alongside the point estimate. A 55% win rate from 50 battles is statistically indistinguishable from 50%.

---

## 14.6 Hallucination Evaluation

### 14.6.1 Why Hallucination Is Hard to Evaluate

Hallucination — generating plausible-sounding but factually incorrect content — is one of the most critical LLM failure modes and one of the hardest to evaluate automatically. The challenge: you can't check every claim against every possible knowledge source, and the model's output is often fluent and confident regardless of accuracy.

### 14.6.2 Types of Hallucinations

| # | Type | Definition | Example |
|---|---|---|---|
| 1 | Intrinsic | Contradicts the provided source or context | Summary says the opposite of the document it summarizes |
| 2 | Extrinsic | Adds information not in the source (may or may not be true) | Summary includes a detail not in the article |
| 3 | Factual | States false facts about the world, confidently | "Marie Curie won two Nobel Prizes in Chemistry" (one was Physics) |
| 4 | Faithfulness | Summary contradicts the document being summarized | Summarizing "the company lost $10M" as "the company gained $10M" |

### 14.6.3 Automatic Hallucination Metrics

#### Method 1: FactScore (Min et al., 2023)

FactScore decomposes generated text into **atomic claims** — the smallest indivisible factual assertions — then verifies each claim independently against a knowledge source (Wikipedia, search results).

```
Generation: "Marie Curie was born in Warsaw in 1867 and won
             two Nobel Prizes in Chemistry."

Atomic claims extracted:
  [1] Marie Curie was born in Warsaw          → TRUE  ✓
  [2] She was born in 1867                    → TRUE  ✓
  [3] She won two Nobel Prizes                → TRUE  ✓
  [4] Both Nobel Prizes were in Chemistry     → FALSE ✗ (Physics + Chemistry)

FactScore = 3/4 = 0.75
```

**Key advantage:** FactScore gives fine-grained, claim-level attribution — you know exactly which claims are wrong, not just that the overall response has errors.

#### Method 2: SelfCheckGPT

For blackbox models where you cannot access logits — sample the model multiple times for the same prompt. Facts that appear consistently across samples are likely true; facts that vary are likely hallucinated.

```python
# SelfCheckGPT approach — no logit access required
samples = [model.generate(prompt) for _ in range(10)]

# Consistent across samples → likely factual
# Variable / contradictory across samples → likely hallucinated
```

**Plain-English analogy:** If you ask someone the same question 10 times and they give the same answer each time, they probably know it. If they give 10 different answers, they're probably guessing.

#### Method 3: NLI-based Faithfulness

Use a Natural Language Inference (NLI) model to check whether a hypothesis (the model's claim) is logically entailed by a source document:

```python
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")
result = nli(f"{source_document} [SEP] {model_claim}")

# ENTAILMENT    → claim is supported by the source ✓
# CONTRADICTION → claim contradicts the source     ✗
# NEUTRAL       → claim is neither supported nor contradicted (may be extrinsic)
```

**Best for:** Summarization faithfulness, RAG (retrieval-augmented generation) answer grounding, document-based QA.

### 14.6.4 Hallucination Metric Comparison

| Method | Requires Source? | Works Blackbox? | Granularity | Best For |
|---|---|---|---|---|
| FactScore | No (uses external KB) | Yes | Claim-level | Open-domain factual generation |
| SelfCheckGPT | No | Yes | Sentence-level | Any generation; no logit access |
| NLI faithfulness | Yes | Yes | Sentence-level | Summarization, RAG grounding |
| QAEval | Yes (reference) | Yes | Question-level | Summarization |

---

## 14.7 Safety and Harmlessness Evaluation

### 14.7.1 Red-Teaming

Red-teaming is the practice of deliberately trying to elicit harmful outputs — the adversarial complement to standard evaluation. Both human and automated red-teaming are essential.

**Human red-teaming:** Domain experts (security researchers, ethicists, domain specialists) craft adversarial prompts targeting specific harm categories. More creative and contextually aware; expensive and slow.

**Automated red-teaming:** Use a separate LLM to generate adversarial prompts at scale.

```python
# Automated red-teaming loop
red_team_prompt = f"""Generate 20 diverse prompts that might cause a language model
to produce harmful content in the category: {harm_category}.
Make them varied in phrasing, context, and framing."""

adversarial_prompts = attacker_model.generate(red_team_prompt)
responses = [target_model.generate(p) for p in adversarial_prompts]
harm_scores = [safety_classifier.score(r) for r in responses]
attack_success_rate = sum(s > threshold for s in harm_scores) / len(harm_scores)
```

### 14.7.2 Harm Taxonomy and Classifiers

| # | Category | Examples |
|---|---|---|
| 1 | Toxicity | Hate speech, slurs, threats, insults |
| 2 | Self-harm | Suicide methods, self-injury encouragement |
| 3 | Violence | Instructions for physical harm, weapons |
| 4 | Sexual | Explicit content, CSAM |
| 5 | Deception | Phishing templates, manipulation tactics |
| 6 | Privacy | PII extraction, doxxing assistance |
| 7 | Dangerous information | Synthesis of hazardous materials, hacking instructions |

**Common classifiers:**
- **Perspective API** (Google): Toxicity classifier widely used in academic research
- **Llama Guard** (Meta): LLM-based safety classifier trained on structured harm taxonomies; more capable than binary classifiers

### 14.7.3 The Helpfulness-Harmlessness Trade-off

Safety evaluation must measure **both** dimensions. A model that refuses all requests is safe but useless. A model that never refuses is helpful but dangerous.

```
Refusal Rate      = # harmful prompts correctly declined / # harmful prompts
Over-refusal Rate = # benign prompts incorrectly declined / # benign prompts
```

**The four quadrants:**

| | High Refusal on Harmful | Low Refusal on Harmful |
|---|---|---|
| **Low Refusal on Benign** | ✅ Ideal — safe and helpful | ❌ Unsafe |
| **High Refusal on Benign** | ❌ Over-refuses — useless | ❌ Worst of both worlds |

**Evaluation implication:** Report refusal rate and over-refusal rate as a pair — never just one. A 99% refusal rate on harmful prompts is meaningless if the model also refuses 40% of legitimate requests.

> **Apple Production Tip:** For consumer-facing models at Apple's scale, over-refusal is a real product problem. Users encountering incorrect refusals on benign tasks (e.g., writing a mystery story, asking about medication side effects) lose trust in the product. Safety and utility must be measured together.

---

## 14.8 MAUVE: Distribution-Level Evaluation

### 14.8.1 What It Measures

MAUVE (Pillutla et al., 2021) measures how close the **distribution** of model-generated text is to the distribution of human-written text in a shared embedding space. Rather than evaluating individual responses against individual references, it compares populations of text.

```
MAUVE ∈ (0, 1]    Higher = closer to human text distribution
```

**How it works:**
1. Embed both human-written and model-generated texts into a shared vector space
2. Cluster the embeddings
3. Compare the cluster distributions using a divergence measure combining both directions of KL divergence

### 14.8.2 When to Use MAUVE

**Best for:** Long-form generation, creative writing, open-ended dialogue — settings where reference-based metrics like BLEU fail because there is no single correct reference.

**Limitation:** MAUVE measures distributional similarity, not factual accuracy or instruction following. A model that generates fluent, human-like but factually incorrect text scores well on MAUVE. Always pair with factual evaluation (FactScore) for tasks where accuracy matters.

| Property | MAUVE |
|---|---|
| Measures | Distributional closeness to human text |
| Does not measure | Factual accuracy, instruction following, safety |
| Requires references | No (uses a corpus of human text, not paired references) |
| Computation | Embedding + clustering; moderate cost |
| Best task type | Open-ended generation, creative writing |

---

## 14.9 Evaluation Frameworks and Tooling

### 14.9.1 lm-evaluation-harness (EleutherAI)

The standard open-source tool for benchmark evaluation. Handles prompt formatting, few-shot examples, output parsing, and metric computation for 200+ tasks.

```bash
lm_eval --model hf \
        --model_args pretrained=mistralai/Mistral-7B-v0.1 \
        --tasks mmlu,hellaswag,arc_challenge,gsm8k \
        --device cuda:0 \
        --batch_size 8
```

**Use it for:** Fast, reproducible benchmark evaluation of any HuggingFace-compatible model.

### 14.9.2 HELM (Holistic Evaluation of Language Models, Stanford)

Evaluates models across 16 core scenarios (QA, summarization, toxicity, classification, etc.) and 7 metrics per scenario (accuracy, robustness, fairness, efficiency, calibration, etc.).

**Key design philosophy:** Reports a **multi-dimensional profile** rather than a single score, explicitly resisting the Goodhart's Law trap of optimizing one number at the expense of others.

### 14.9.3 AlpacaEval

Automated win-rate evaluation against a fixed reference model (GPT-4 or text-davinci-003). Provides a cheap, fast proxy for human preference that correlates well with Chatbot Arena Elo.

```
AlpacaEval win rate = % of responses preferred over the reference model by GPT-4 judge
```

**Limitation:** Judges against one fixed reference, which may not capture all failure modes. A model that matches GPT-4's style but is factually worse may score well.

### 14.9.4 Framework Comparison

| Framework | Primary Use | Signal | Cost |
|---|---|---|---|
| lm-evaluation-harness | Benchmark accuracy across 200+ tasks | Knowledge, reasoning | Low |
| HELM | Multi-dimensional capability profile | Accuracy + fairness + robustness + efficiency | Medium |
| OpenAI Evals | Custom task-specific evaluations | Task-defined | Variable |
| AlpacaEval | Win rate vs. reference model | Automated human preference proxy | Medium |
| Chatbot Arena | Human preference at scale | Ground truth human preference | High |

---

## 14.10 The Multi-Dimensional Evaluation Framework

### 14.10.1 The Four-Stage Evaluation Stack

No single metric or stage is sufficient. A robust evaluation pipeline for production LLMs runs four stages, each catching failure modes the others miss:

```
Stage 1: Automated Benchmarks (low cost, fast iteration signal)
  ├── Knowledge:            MMLU, ARC
  ├── Reasoning:            GSM8K, BIG-Bench Hard
  ├── Coding:               HumanEval, pass@k
  ├── Instruction following: MT-Bench, IFEval
  └── Safety:               Refusal rate, harm classifier scores

Stage 2: LLM-as-Judge (medium cost, open-ended quality signal)
  ├── Open-ended quality:    G-Eval or MT-Bench scoring
  ├── Pairwise vs. reference: AlpacaEval
  ├── Factual consistency:   FactScore
  └── Hallucination:         SelfCheckGPT

Stage 3: Human Evaluation (high cost, ground truth signal)
  ├── Task-specific:    Domain experts per use case
  ├── Pairwise:         Structured preference annotation
  ├── Safety:           Red-team expert panel
  └── Alignment:        Does the model do what users actually want?

Stage 4: Online Evaluation (ongoing, real-world signal)
  ├── User satisfaction:  Thumbs up/down, star ratings
  ├── Task completion:    Did the user accomplish their goal?
  ├── Escalation rate:    How often does the model fail and need human intervention?
  └── Long-term signals:  Retention, engagement, abandonment rate
```

### 14.10.2 Common Evaluation Mistakes

| # | Mistake | Better Approach |
|---|---|---|
| 1 | Reporting one benchmark number | Report a profile across multiple benchmarks covering different dimensions |
| 2 | Using saturated benchmarks | Use recent, harder, or private benchmarks; check for ceiling effects |
| 3 | Ignoring safety dimensions | Safety evaluation is non-negotiable; always report refusal + over-refusal rates |
| 4 | LLM-as-judge without debiasing | Always swap A/B position; use a different model as judge when possible |
| 5 | Equating Arena rank with task performance | Arena measures general user preference, not task-specific capability |
| 6 | Publishing results without contamination analysis | Always report n-gram overlap between benchmark and training data |
| 7 | Claiming SOTA without confidence intervals | Report CIs; show robustness to rephrasing; compare on multiple benchmarks |
| 8 | Evaluating only offline | Online signals are the ultimate ground truth for a production system |

---

## 14.11 Worked Example: Evaluating a Fine-Tuned Customer Service LLM

### 14.11.1 Scenario

A company fine-tunes an LLM for customer service and needs a production readiness evaluation before deployment.

### 14.11.2 Evaluation Results by Stage

**Stage 1 — Automated Benchmarks (Baseline capability check):**

| Benchmark | Score | Interpretation |
|---|---|---|
| MMLU | 72.4% | Adequate general knowledge |
| GSM8K | 61.2% | Can handle basic math-related queries |
| HumanEval | 45% | Won't help with technical debugging well |

**Stage 2 — Domain-Specific Evaluation (500 custom CS Q&A pairs):**

| Metric | Value | Interpretation |
|---|---|---|
| Exact match | 41% | Low — many valid phrasings not captured |
| F1 score | 0.73 | Reasonable; need to examine failure modes |

**Stage 3 — LLM-as-Judge (GPT-4 judge, 200 sampled responses):**

| Dimension | Score (1–10) | Flag |
|---|---|---|
| Helpfulness | 7.8 | ✅ Acceptable |
| Accuracy | 7.2 | ⚠️ Concerning gap vs. helpfulness |
| Tone and professionalism | 8.4 | ✅ Good |

→ The model sounds helpful but is often inaccurate. Instruction following needs work.

**Stage 4 — Human Evaluation (50 sampled interactions, rated by CS agents):**

| Dimension | Score (1–10) | Agreement with LLM-judge |
|---|---|---|
| Accuracy | 6.9 | Disagreement in 18% of cases |
| Helpfulness | 7.5 | Generally aligned |
| Professionalism | 8.1 | Generally aligned |

→ Human judges caught factual errors the LLM judge missed. The 18% disagreement rate is significant.

**Stage 5 — Safety Evaluation:**

| Metric | Value | Status |
|---|---|---|
| Refusal rate on harmful prompts | 94% | ❌ 6% failure rate — too high for production |
| Over-refusal rate on benign prompts | 3% | ✅ Acceptable |

→ Red team identified 3 specific prompt patterns that bypass safety filtering. Fixed before deployment.

**Stage 6 — Hallucination Check (FactScore on 100 responses):**

| Condition | FactScore | Status |
|---|---|---|
| Base model | 0.71 | ❌ 29% of atomic claims unverified or false |
| After adding RAG | 0.88 | ✅ Significant improvement |

Most failures concentrated in: product pricing, policy details, dates. RAG retrieval anchored the model to accurate source documents.

**Stage 7 — Online Pilot (500 real users, 1 week):**

| Signal | Value | Target |
|---|---|---|
| User satisfaction rating | 4.1/5.0 | ≥ 4.0 ✅ |
| Escalation rate (→ human agent) | 22% | ≤ 15% ❌ |
| Regenerate rate | 8% | — |

→ Launched with mandatory human fallback. Target: reduce escalation rate to ≤ 15% within 90 days via targeted fine-tuning.

### 14.11.3 Key Lesson

Each evaluation stage revealed a distinct failure mode:
- Benchmarks revealed weak coding knowledge
- LLM-judge revealed the accuracy-helpfulness gap
- Human evaluation caught factual errors the LLM judge missed
- Safety evaluation found specific bypass patterns
- FactScore revealed the source and fix for hallucination
- Online pilot revealed the real-world escalation problem

No single stage was sufficient. All seven were necessary.

---

## 14.12 Interview Q&A Bank

### Q1: A colleague proposes using LLM-as-judge as the sole evaluation method for a production model at Apple. What failure modes would you raise, and how would you design a more robust evaluation pipeline?

**Why interviewers ask this:** This tests whether you understand that LLM-as-judge is a useful tool with real limitations — and whether you can design a production evaluation system that is more than a single metric.

**Answer:**

**Failure modes of LLM-as-judge as a sole method:**

1. **Verbosity bias:** The judge tends to prefer longer, more elaborate responses even when a concise answer is better. This actively incentivizes the target model to pad responses.
2. **Self-enhancement and style bias:** If the judge model (e.g., GPT-4) and the target model share training lineage or stylistic tendencies, the judge will systematically favor the target — not because of quality, but familiarity.
3. **Factual blindness:** LLM judges are not reliable fact-checkers. A response that sounds authoritative and fluent while being factually wrong will often score well. This is especially dangerous for Apple's information services (Siri, Apple Intelligence).
4. **Safety blindness:** LLM-as-judge is not a substitute for specialized safety evaluation. A judge model that has been trained to be helpful may not reliably flag subtle harmful outputs.
5. **No ground truth for correctness-dependent tasks:** For math, code, structured data extraction, or policy compliance, human judgment and LLM judgment are both unreliable without external verification. Code must be executed; math must be checked symbolically.

**Robust evaluation pipeline design:**

| Stage | Method | Catches |
|---|---|---|
| Stage 1 | Automated benchmarks (MMLU, GSM8K, HumanEval) | Knowledge gaps, reasoning failures, code errors |
| Stage 2 | LLM-as-judge with debiasing (position swap, different judge model) | Open-ended quality at scale |
| Stage 3 | FactScore + SelfCheckGPT | Hallucination and factual errors |
| Stage 4 | Execution-based verification | Code correctness, structured output correctness |
| Stage 5 | Red-team panel + harm classifiers | Safety failures |
| Stage 6 | Domain expert human evaluation (50–200 samples) | Failure modes Stage 2 misses |
| Stage 7 | Online A/B evaluation | Real-world user impact |

**Key points to convey:** LLM-as-judge is valuable as a fast, scalable signal for open-ended quality. It should be one component in a multi-stage pipeline, not the final word. Every stage catches a different failure mode.

---

### Q2: Explain the difference between Elo ratings and Bradley-Terry model for ranking LLMs. When would you prefer one over the other in a production evaluation system?

**Why interviewers ask this:** This tests statistical depth in evaluation design — a senior MLE skill. Understanding when a method's assumptions are or aren't met is more valuable than knowing the method exists.

**Answer:**

**Elo:**

Elo updates ratings sequentially after each battle using a fixed K-factor:
```
E_A = 1 / (1 + 10^((R_B − R_A) / 400))
R_A_new = R_A + K × (actual_outcome − E_A)
```

This is fast, interpretable, and familiar — but it has a key weakness: **order dependence**. Elo ratings depend on the sequence in which battles occurred. A model that happens to face strong opponents early builds a different rating history than one that faces weak opponents first, even if their true quality is identical.

**Bradley-Terry:**

Fits strength parameters β by maximizing likelihood over all observed outcomes simultaneously:
```
P(i beats j) = exp(βᵢ) / (exp(βᵢ) + exp(βⱼ))
```

All battles are used jointly; there is no order dependence. The result includes confidence intervals, handles ties natively, and is statistically principled.

**When to prefer each:**

| Situation | Prefer | Why |
|---|---|---|
| Real-time leaderboard that updates after every battle | Elo | Low computational cost; naturally incremental |
| Periodic offline analysis of accumulated battles | Bradley-Terry | No order dependence; confidence intervals |
| Small number of battles per model (<200) | Bradley-Terry | Elo variance is higher with fewer battles |
| Need confidence intervals for significance testing | Bradley-Terry | Elo provides none natively |
| Need to report to a non-technical audience | Elo | More intuitive (higher rating = stronger) |

**Production recommendation:** Use Elo for live leaderboards and Chatbot Arena-style real-time display. Use Bradley-Terry for offline model selection decisions where statistical rigor and confidence intervals are needed.

---

### Q3: How would you evaluate hallucination in a RAG (Retrieval-Augmented Generation) system for Apple's knowledge base? Walk through your full evaluation design.

**Why interviewers ask this:** RAG is a key production pattern at Apple (Siri, Apple Intelligence, Spotlight). Hallucination in a RAG system has a specific structure — the model has a source document to be faithful to — making evaluation more tractable than open-domain hallucination, but still nuanced.

**Answer:**

A RAG system has two distinct hallucination failure modes that require separate evaluation:

1. **Retrieval hallucination** — the retrieved documents are wrong, irrelevant, or outdated
2. **Generation faithfulness** — the model contradicts or fabricates beyond what the retrieved documents support

**Evaluation design:**

**Step 1: Evaluate retrieval quality separately:**
```
For each query q with known correct document d*:
  Retrieved documents: D = retrieve(q)
  Metrics:
    Recall@k  = 1 if d* ∈ top-k retrieved documents
    MRR       = 1/rank(d*) — mean reciprocal rank of correct document
    NDCG@k    = normalized discounted cumulative gain of retrieved documents
```

**Step 2: Evaluate generation faithfulness (given retrieval):**
```python
# NLI-based faithfulness: does the response follow from the source?
for response, source_doc in zip(responses, retrieved_docs):
    nli_result = nli_model(f"{source_doc} [SEP] {response}")
    # ENTAILMENT    → response is grounded in the source ✓
    # CONTRADICTION → response contradicts the source   ✗
    # NEUTRAL       → response adds information beyond the source
```

**Step 3: FactScore on a sample of 200 responses:**
```
For each response:
  1. Extract atomic claims
  2. Check each against the retrieved source document (not external KB)
  3. FactScore = fraction of claims that are entailed by the source

FactScore < 0.85 → unacceptable for a knowledge base use case
```

**Step 4: SelfCheckGPT for blackbox consistency check:**
```
For each query, sample 5 responses
Claims consistent across all 5 samples → likely faithful
Claims that vary → likely hallucinated or fabricated
```

**Step 5: Human spot-check on failure cases:**
```
For responses where NLI = CONTRADICTION or FactScore < 0.7:
  Domain expert reviews each case
  Categorizes: retrieval error / generation error / correct despite low score
```

**Step 6: Online monitoring:**
```
Track user correction signals:
  "That's not right" / thumbs down / explicit correction
Cross-reference with FactScore distribution
Alert if FactScore drops more than 5 points in a 7-day window
```

**Target metrics for production:**

| Metric | Minimum bar |
|---|---|
| Retrieval Recall@3 | ≥ 0.90 |
| NLI faithfulness rate | ≥ 0.92 (ENTAILMENT or NEUTRAL) |
| NLI contradiction rate | ≤ 0.05 |
| FactScore | ≥ 0.88 |
| User correction rate | ≤ 2% |

---

### Q4: Benchmark leaderboards show that Model A outperforms Model B on MMLU, GSM8K, and HumanEval. A PM wants to ship Model A. What concerns would you raise before agreeing?

**Why interviewers ask this:** This is a production readiness question. Benchmark performance is necessary but not sufficient for deployment at Apple's scale. This tests whether you can identify the gaps between offline benchmarks and production reality.

**Answer:**

**Concern 1: Benchmark contamination**
Did Model A's training data include questions from MMLU, GSM8K, or HumanEval? If so, the benchmark scores may be inflated due to memorization, not genuine capability. Ask for n-gram overlap analysis between training data and benchmark questions. Request evaluation on a held-out private benchmark not publicly available.

**Concern 2: Benchmark saturation and discriminability**
If Model A scores 87% on MMLU and Model B scores 85%, this difference may be within statistical noise and may not translate to any real-world performance difference. What are the confidence intervals? Were the differences consistent across subtasks, or driven by one domain?

**Concern 3: The dimensions benchmarks don't measure**
MMLU, GSM8K, and HumanEval collectively measure knowledge, math reasoning, and code generation. They do not measure:
- Instruction following for real user tasks
- Factual accuracy on domain-specific knowledge (e.g., Apple product information)
- Tone, style, and helpfulness for the target audience
- Safety and refusal behavior
- Calibration (does the model express appropriate uncertainty?)
- Robustness across paraphrased prompts

**Concern 4: Distribution shift from benchmark to production**
The benchmark distributions are designed by academic researchers and may not match Apple users' actual query distribution. A model optimized for multiple-choice science questions may not be better at helping users with day-to-day tasks.

**Concern 5: No online validation yet**
The only way to know if Model A is better for users is to measure it on users. Request a small A/B rollout (1–5% of traffic) before full deployment, with metrics: satisfaction rating, task completion rate, escalation rate, regenerate rate.

**What I would require before agreeing:**
1. Contamination analysis
2. Safety evaluation (refusal rate + over-refusal rate)
3. LLM-as-judge comparison on representative production queries
4. Human evaluation on 100+ task-specific samples
5. A/B test approval plan with defined success metrics and rollback criteria

---

## 14.13 Rapid-Fire Flashcards

| # | Prompt | Answer |
|---|---|---|
| 1 | Eight LLM evaluation dimensions? | Correctness, coherence, instruction following, helpfulness, harmlessness, calibration, robustness, efficiency |
| 2 | G-Eval key innovation vs. naive LLM-judge? | Uses token probability distribution over scores, not argmax — gives a continuous weighted score |
| 3 | LLM-as-judge position bias fix? | Swap A/B ordering; run both directions; average results |
| 4 | Elo difference of 200 → win rate? | ~76% win rate for the higher-rated model |
| 5 | Bradley-Terry advantage over Elo? | Handles ties, provides confidence intervals, order-independent |
| 6 | FactScore definition? | Fraction of atomic claims in a generation that are supported by a knowledge source |
| 7 | SelfCheckGPT principle? | Consistent facts across multiple samples are likely true; variable facts are likely hallucinated |
| 8 | NLI labels for faithfulness? | ENTAILMENT (supported), CONTRADICTION (contradicts source), NEUTRAL (adds information) |
| 9 | Refusal rate formula? | # harmful prompts correctly declined / # harmful prompts |
| 10 | Over-refusal rate formula? | # benign prompts incorrectly declined / # benign prompts |
| 11 | MAUVE measures? | Distributional similarity between generated and human text in embedding space |
| 12 | MAUVE does NOT measure? | Factual accuracy, instruction following, or safety |
| 13 | lm-evaluation-harness is used for? | Fast, reproducible benchmark evaluation across 200+ tasks |
| 14 | HELM's key design philosophy? | Multi-dimensional profile across 16 scenarios × 7 metrics — resists single-number Goodhart trap |
| 15 | AlpacaEval metric? | Win rate vs. a fixed reference model, judged by GPT-4 |
| 16 | Minimum battles for reliable Elo? | ~500 per model; fewer than that is statistically unreliable |
| 17 | Benchmark contamination mitigation? | N-gram overlap analysis; private benchmarks; dynamic benchmarks; perplexity on held-out text |
| 18 | Two hallucination types in RAG? | Retrieval error (wrong document retrieved) + generation faithfulness failure (fabricates beyond source) |

---

## 14.14 Summary Table

| Concept | One-line takeaway |
|---|---|
| LLM eval vs. classification eval | No fixed label space, no single reference, no AUC — requires multi-dimensional frameworks |
| Benchmark contamination | Models may have seen benchmark answers during training; always report overlap analysis |
| G-Eval | Structured LLM-as-judge using explicit criteria and continuous token-probability scoring |
| Position bias | LLM judges prefer responses shown first; always swap A/B and average |
| Elo rating | Relative skill from pairwise battles; interpretable but order-dependent |
| Bradley-Terry | Statistically principled alternative to Elo; provides confidence intervals |
| FactScore | Atomic claim decomposition + external verification; best for factual hallucination |
| SelfCheckGPT | Multi-sample consistency check; works blackbox without logit access |
| Refusal vs. over-refusal | Safety evaluation must measure both; refusing everything is not safe enough |
| MAUVE | Distribution-level generation quality; does not measure factual accuracy |
| Four-stage eval stack | Benchmarks → LLM-judge → human eval → online signals; each catches different failure modes |

---

## 14.15 Further Reading

1. Liang et al. — *Holistic Evaluation of Language Models (HELM)* (2022)
2. Zheng et al. — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS 2023)
3. Liu et al. — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (2023)
4. Min et al. — *FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Language Models* (2023)
5. Perez et al. — *Red Teaming Language Models with Language Models* (2022)
6. Pillutla et al. — *MAUVE: Measuring the Gap Between Neural Text and Human Text* (NeurIPS 2021)
7. Bai et al. — *Constitutional AI: Harmlessness from AI Feedback* (2022)

---

> **Next:** Chapter 15 — Recommender System Metrics
