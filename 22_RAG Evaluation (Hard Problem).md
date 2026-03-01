Excellent.
Day 22 is where most RAG systems quietly fail.

> Evaluation of RAG is fundamentally harder than training it.

Today we go deep into:

* Why RAG evaluation is hard
* Faithfulness
* Groundedness
* Answer relevance
* RAGAS
* G-Eval
* Production evaluation strategy
* FAANG-level Q&A

---

# 1️⃣ Why RAG Evaluation Is Hard

Unlike classification:

* There is no single correct answer.
* LLM outputs are free-form.
* Answers can be partially correct.
* Hallucinations may be subtle.

And worse:

> Retrieval and generation errors interact.

So evaluation must consider:

1. Retrieval quality
2. Generation quality
3. Alignment between them

---

# 2️⃣ Core Dimensions of RAG Evaluation

We separate into 3 primary axes:

| Dimension        | What it checks                          |
| ---------------- | --------------------------------------- |
| Faithfulness     | Is answer supported by context?         |
| Groundedness     | Are claims traceable to retrieved docs? |
| Answer Relevance | Does answer actually address question?  |

Each captures a different failure mode.

---

# 3️⃣ Faithfulness

## Definition

Faithfulness measures:

> Whether the answer is logically supported by retrieved context.

An answer can be relevant but unfaithful.

---

### Example

Context:

```
Refunds allowed within 30 days.
```

Question:

> Can I return after 45 days?

Answer:

> Yes, but late fees apply.

This is:

* Fluent
* Relevant
* Not faithful

---

## How to Measure Faithfulness

### 1️⃣ LLM-as-Judge

Prompt:

```text id="zk82pa"
Given:
Context: ...
Answer: ...

Is every claim in the answer supported by the context?
Respond YES or NO.
```

This works surprisingly well.

---

### 2️⃣ Entailment Models

Use NLI (Natural Language Inference):

Check if:

Context ⇒ Answer

If contradiction or neutral → unfaithful.

---

### 3️⃣ Token-Level Attribution

Match answer spans to context spans.

Measure:

% answer tokens backed by context.

---

# 4️⃣ Groundedness

Often confused with faithfulness.

Subtle difference:

* Faithfulness = logically supported.
* Groundedness = explicitly traceable to specific document chunks.

---

### Example

If model paraphrases correctly but cannot cite source:

Faithful
But not grounded.

Groundedness requires:

* Citation accuracy
* Span alignment
* Traceability

---

## Measuring Groundedness

1. Force citation output.
2. Verify citation exists in retrieved docs.
3. Check span overlap.

Advanced systems compute:

> Attribution accuracy.

---

# 5️⃣ Answer Relevance

This measures:

> Does the answer address the user's question?

An answer can be faithful but irrelevant.

Example:

Context:

```
Refunds allowed within 30 days.
```

Question:

> What are store opening hours?

Answer:

> Refunds are allowed within 30 days.

Faithful but irrelevant.

---

## Measuring Relevance

Use LLM-as-judge:

```text id="1a4pvt"
Does the answer fully address the user question?
Score from 1–5.
```

Or binary YES/NO.

---

# 6️⃣ Why Simple Exact Match Fails

Traditional NLP metrics:

* BLEU
* ROUGE
* Exact match

Fail because:

* Multiple correct phrasings exist.
* LLMs paraphrase.
* Answers are generative.

So we rely on semantic evaluation.

---

# 7️⃣ RAGAS (RAG Assessment Framework)

RAGAS is a framework specifically designed for RAG evaluation.

It computes:

* Faithfulness
* Answer Relevance
* Context Precision
* Context Recall

Without needing ground-truth answers.

---

## How RAGAS Works (High-Level)

Given:

* Question
* Retrieved context
* Generated answer

It uses LLM scoring prompts internally to compute metrics.

---

### Key Metrics

### 🔹 Faithfulness

Is answer supported by context?

### 🔹 Answer Relevance

Does answer address question?

### 🔹 Context Recall

Did retrieval capture necessary information?

### 🔹 Context Precision

Did retrieval include mostly relevant chunks?

---

## Why RAGAS Is Powerful

* No human labels required
* Works for dynamic corpora
* Evaluates retrieval + generation

But:

It depends on LLM judge quality.

---

# 8️⃣ G-Eval

G-Eval is a structured evaluation method using LLMs.

Instead of asking:

> Is this good?

It uses:

* Chain-of-thought scoring
* Structured rubric
* Step-by-step reasoning

Example:

```text id="thg6la"
Evaluate the answer on:
1. Correctness
2. Completeness
3. Faithfulness

Explain reasoning.
Give score 1–5.
```

This improves judge reliability.

---

## Why G-Eval Matters

LLM-as-judge can be noisy.

Structured evaluation:

* Reduces randomness
* Increases consistency
* Improves correlation with human judgments

---

# 9️⃣ Full RAG Evaluation Strategy (Production)

A mature system evaluates at 3 layers.

---

## Layer 1: Retrieval Evaluation

Metrics:

* Recall@K
* MRR
* Context precision
* Context recall

Offline labeled dataset required.

---

## Layer 2: Generation Evaluation

Metrics:

* Faithfulness
* Groundedness
* Relevance
* Toxicity

Use LLM judge + NLI model.

---

## Layer 3: End-to-End Evaluation

Human eval:

* 1–5 helpfulness score
* Binary correctness
* Hallucination detection

Used for periodic audits.

---

# 🔟 Hard Problem: Evaluating Hallucination

Hallucinations can be:

* Fabricated facts
* Subtle distortions
* Unsupported reasoning

Best current approach:

1. Extract claims from answer.
2. Verify each claim against context.
3. Score support ratio.

This is still an open research problem.

---

# 1️⃣1️⃣ Tradeoffs of LLM-as-Judge

Pros:

* Scalable
* Cheap
* No manual labels

Cons:

* Biased toward verbose answers
* May share same hallucination bias
* Not perfectly calibrated

Senior candidates mention:

> Use different model family as judge to reduce bias.

---

# 1️⃣2️⃣ FAANG-Style Interview Questions

---

## Q1. How do you evaluate a RAG system?

Strong structured answer:

1. Evaluate retrieval separately using Recall@K.
2. Evaluate generation using faithfulness + relevance.
3. Use LLM-as-judge with structured rubric.
4. Periodically validate with human evaluation.
5. Monitor live production metrics.

---

## Q2. Difference between faithfulness and relevance?

Faithfulness:
Is answer supported by context?

Relevance:
Does answer address the question?

They measure different failure modes.

---

## Q3. How do you measure hallucination?

* Claim extraction
* NLI-based entailment
* LLM-as-judge
* Citation validation

There is no perfect automatic metric yet.

---

## Q4. What are limitations of RAGAS?

* Relies on LLM scoring
* Sensitive to judge prompt
* May inflate scores
* Doesn’t fully capture subtle reasoning errors

---

# 1️⃣3️⃣ Advanced Insight (Senior Level)

True RAG evaluation requires:

> Decomposing retrieval and generation.

If you evaluate only end-to-end:

You don’t know whether failure came from:

* Missed retrieval
* Poor ranking
* Context overload
* Hallucination

Decomposition is mandatory for improvement.

---

# 1️⃣4️⃣ Emerging Direction

New research areas:

* Attribution-aware training
* Retrieval confidence calibration
* Learned evaluation models
* Self-consistency scoring

Evaluation is still an open frontier.

---

# Final Takeaways

* Faithfulness ≠ relevance
* Groundedness requires traceability
* RAGAS automates multi-metric scoring
* G-Eval improves LLM-as-judge reliability
* Decompose retrieval vs generation in evaluation

---

