# Hallucinations in LLMs: Detection, Metrics, Mitigation

## 1. What is a hallucination?

A hallucination is fluent, confident output that is **not supported by** either (a) the source material given to the model, or (b) ground truth in the world. Two useful axes:

**Intrinsic vs. extrinsic:**
- **Intrinsic** — contradicts the given source/context directly. E.g., source says "the meeting is Tuesday," model says "the meeting is Wednesday."
- **Extrinsic** — adds information that isn't in the source and can't be verified from it (may or may not be true, but it's unsupported).

**Faithfulness vs. factuality** (the distinction most eval libraries actually use):
- **Faithfulness** — is the output consistent with the *given context* (e.g., in a RAG/summarization setting)? This is checkable without external knowledge.
- **Factuality** — is the output true in the *real world*? Requires an external knowledge source or ground truth to check.

A summarization hallucination is usually a faithfulness problem. An open-domain QA hallucination ("who won the 1932 election in country X") is a factuality problem.

## 2. Why hallucinations happen (causes)

- **Next-token training objective** rewards *fluency*, not truth — the model is optimized to produce plausible continuations, and a plausible-sounding wrong fact scores nearly as well during training as a correct one.
- **Missing knowledge** — the fact simply wasn't in training data, or was rare/underrepresented, so the model's best guess is a statistically-plausible confabulation.
- **Knowledge conflict** — training data itself contained contradictory claims about the same fact.
- **Exposure bias** — during generation the model conditions on its *own* previous outputs, so one early wrong token can compound into a fully fabricated chain (this is why hallucinations often "snowball" — a false premise gets confidently elaborated).
- **Sampling randomness** — higher temperature / top-p sampling increases hallucination rate by construction, since it deliberately selects lower-probability tokens sometimes.
- **RAG-specific**: retrieval brought back irrelevant or contradictory documents, or the model ignored/misweighted the retrieved context in favor of its parametric memory.

## 3. Detection methods

### A. Self-consistency / sampling-based (no external knowledge needed)
**Idea:** if the model actually "knows" a fact, resampling the same question at moderate temperature should give consistent answers. If it's confabulating, samples will disagree.

**SelfCheckGPT** (Manakul et al.) is the canonical implementation:
1. Generate one main response.
2. Sample N (e.g., 10-20) additional stochastic responses to the same prompt.
3. For each sentence in the main response, measure how well it's supported by/consistent with the N samples (via BERTScore similarity, NLI entailment, n-gram overlap, or asking an LLM directly).
4. Sentences with low cross-sample support are flagged as likely hallucinated.

Fully black-box — works even via API access with no access to logits.

### B. Uncertainty-based (needs token probabilities/logits)
- **Token-level probability** — low probability assigned to the actual generated token is a hallucination signal, but a badly-calibrated model can be confidently wrong, so this is a weak signal alone.
- **Semantic entropy** (Kuhn et al., 2023) — sample multiple full responses, cluster them by *meaning* (via bidirectional NLI entailment: do response A and B entail each other?), then compute entropy over the meaning-clusters rather than over raw token sequences. High semantic entropy = model is unsure at the level of meaning, not just wording — a stronger signal than raw perplexity, because two differently-worded but same-meaning answers should NOT count as "disagreement."

### C. Retrieval / evidence-grounded verification
1. Decompose the output into atomic factual claims (one verifiable statement each).
2. For each claim, retrieve supporting/contradicting evidence (search engine, knowledge base, or the RAG source documents).
3. Use an NLI (entailment) model or LLM-judge to classify each claim as *supported*, *contradicted*, or *not enough info*.

This is the basis of **FActScore** and most production fact-checking pipelines — it's the most reliable approach but requires a trusted evidence source and is slower/costlier.

### D. LLM-as-judge
Ask a separate (often stronger) LLM: "Given this source context and this claim, is the claim fully supported, partially supported, or unsupported?" Cheap and flexible, but inherits the judge model's own blind spots and biases — validate against human-labeled samples before trusting it at scale.

## 4. Metrics

| Metric | What it measures | Needs external KB? |
|---|---|---|
| **FActScore** | % of atomic facts in a generation that are individually verified as supported by a trusted reference | Yes |
| **SelfCheckGPT score** | Per-sentence inconsistency across resampled generations (0=fully consistent, 1=fully hallucinated) | No |
| **Semantic entropy** | Uncertainty measured over meaning-clusters of sampled answers | No (self-contained) |
| **Faithfulness (RAGAS)** | Fraction of claims in the answer that are entailed by the retrieved context | Uses the RAG context itself |
| **Answer relevancy (RAGAS)** | Whether the answer actually addresses the question (a relevant but faithfulness-perfect answer can still dodge the question) | No |
| **HHEM (Vectara Hallucination Evaluation Model)** | A small fine-tuned classifier giving a 0-1 "consistent with source" score for summarization/RAG outputs | Uses source doc |
| **TruthfulQA accuracy** | % of answers matching human-verified true answers on a benchmark of questions designed to elicit common misconceptions | Benchmark-based |
| **BLEU / ROUGE** | N-gram overlap with a reference answer — **not** a good hallucination metric; a paraphrased-but-faithful answer scores low, and a wrong answer that happens to share words scores high | Reference-based, weak signal |

**Important nuance:** faithfulness and factuality metrics answer different questions. A RAG system can be 100% faithful (perfectly reflects its retrieved documents) while those documents themselves are wrong — that's a retrieval problem, not a generation hallucination, but it looks identical to the end user.

## 5. Worked example (FActScore-style, by hand)

Suppose a model, asked "Tell me about Marie Curie," outputs:

> "Marie Curie was a physicist and chemist born in Warsaw in 1867. She won two Nobel Prizes, in Physics and Chemistry. She was the first person ever to win a Nobel Prize in three different fields."

Decompose into atomic claims:
1. Marie Curie was a physicist and chemist. → **Supported**
2. Born in Warsaw in 1867. → **Supported**
3. Won two Nobel Prizes, in Physics and Chemistry. → **Supported**
4. First person to win a Nobel Prize in three different fields. → **Contradicted** (she won in two fields, not three — this is a hallucinated elaboration, likely pattern-matched from her genuinely being "one of the first to win in two different fields")

FActScore = supported claims / total claims = 3/4 = **0.75**

This single example shows the classic hallucination pattern: three grounded facts followed by one confident, plausible-sounding, false embellishment — exactly the "snowballing" failure mode described in Section 2.

## 6. Resolution / mitigation strategies

**At the data/training level:**
- **RLHF/DPO with truthfulness-focused preference data** — explicitly reward answers that decline or hedge over confidently wrong ones.
- **Factuality-aware fine-tuning** — penalize the model during training for claims that don't check out against a reference (used in some FActScore-guided fine-tuning pipelines).

**At the architecture/decoding level:**
- **RAG (retrieval-augmented generation)** — ground answers in retrieved, verifiable source text rather than parametric memory. The single most effective lever for factuality in production systems.
- **Contrastive decoding / DoLa (Decoding by Contrasting Layers)** — contrast the output distribution of later vs. earlier transformer layers; factual knowledge tends to sharpen in later layers, so amplifying that contrast reduces hallucination without retraining.
- **Lower temperature / constrained decoding** for factual queries — reduces the randomness that directly causes some hallucinations.

**At the prompting level:**
- **Explicit permission to abstain** — "If you don't know, say so" measurably reduces confident fabrication (models are heavily trained to always produce *an* answer).
- **Chain-of-thought + self-verification** — ask the model to list claims, then separately verify each one, then revise.
- **Citation-required prompting** — require every claim to be attributed to a specific retrieved passage; unattributable claims get dropped.

**At the system level (post-hoc):**
- **Verify-then-correct pipelines** — generate → decompose into claims → verify each against evidence → regenerate/edit unsupported claims. This is what most production hallucination-mitigation systems actually do (it's the FActScore pipeline used defensively rather than just for eval).
- **Guardrails / output filtering** — block or flag responses whose faithfulness/HHEM score falls below a threshold before they reach the user.
- **Knowledge editing (ROME, MEMIT)** — directly edit specific factual associations stored in FFN weights, for surgically correcting a known-wrong fact without full retraining (research-stage, not yet common in production).

## 7. Libraries & tools

| Library | What it does |
|---|---|
| **RAGAS** | Open-source RAG evaluation: faithfulness, answer relevancy, context precision/recall — the most widely used RAG-specific eval suite |
| **DeepEval** | General LLM eval framework with a hallucination metric, faithfulness metric, G-Eval (LLM-judge scoring), integrates with pytest |
| **TruLens** | Instruments LLM apps to track groundedness, relevance, and other feedback functions across a trace, good for RAG pipeline debugging |
| **Vectara HHEM** | Small open-weight classifier model specifically for scoring hallucination/consistency in summarization and RAG outputs; also maintains a public hallucination leaderboard for major LLMs |
| **SelfCheckGPT** (GitHub: potsawee/selfcheckgpt) | Reference implementation of the sampling-based black-box detection method described above |
| **Guardrails AI** | Lets you define output validators (including a "provenance"/groundedness check) that re-ask or filter the model if a response fails a check |
| **OpenAI Evals** | General eval framework; supports custom factuality graders, often used with an LLM-as-judge grader for factual QA |
| **Galileo / Patronus AI** | Commercial platforms with dedicated hallucination-detection APIs (Patronus's "Lynx" model is trained specifically for this) |
| **LangChain / LlamaIndex evaluators** | Built-in "faithfulness" and "correctness" evaluator chains, typically LLM-as-judge under the hood, convenient if already using those frameworks |

**Practical starting point** if you just need something working quickly: RAGAS for RAG-specific faithfulness scoring, or SelfCheckGPT if you have no reference documents at all and need a black-box check.

## Quick interview cheat-sheet

| Ask | One-line answer |
|---|---|
| Faithfulness vs factuality? | Faithfulness = consistent with given context; factuality = true in the real world |
| Cheapest hallucination check with no external KB? | SelfCheckGPT — sample N times, check cross-sample consistency |
| Best factuality metric for research papers? | FActScore — decompose into atomic facts, verify each against a trusted source |
| Why is semantic entropy better than raw token entropy? | Clusters by meaning (via NLI) first, so paraphrases don't falsely count as disagreement |
| Single most effective production fix? | Grounding via RAG + citation-required prompting |
| Why do hallucinations "snowball"? | Autoregressive generation conditions on its own prior (possibly wrong) output, so one bad token compounds |
