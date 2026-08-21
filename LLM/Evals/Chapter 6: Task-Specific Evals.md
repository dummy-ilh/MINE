# Chapter 6: Task-Specific Evals

## Why general benchmarks aren't enough here

MMLU tests knowledge. GSM8K tests math reasoning. HumanEval tests standalone function correctness. But none of them capture what happens when an LLM is embedded in a **system** — retrieving documents, summarizing long inputs, acting across multiple steps. These systems fail in ways specific to their architecture, so you need eval methods built for the architecture, not generic language-quality metrics.

## RAG Evaluation (Retrieval-Augmented Generation)

**The key insight: a RAG system has two separate components that can each fail independently — retrieval and generation.** A great generator fed bad documents still produces a bad answer. A great retriever feeding a generator that ignores the context still produces a bad answer. So you must evaluate them *separately*, not just judge the final answer as a black box.

### Retrieval-side metrics

**Precision@k / Recall@k** — of the top-k retrieved documents, how many are actually relevant (precision), and of all the relevant documents that exist, how many did we retrieve in the top-k (recall)?

**Worked numerical example.** A query has 4 truly relevant documents in the corpus. Your retriever returns the top 5 documents, of which 2 are relevant.
- Precision@5 = 2/5 = 0.40 (of what you retrieved, 40% was actually useful)
- Recall@5 = 2/4 = 0.50 (of what existed to find, you found half)

**MRR (Mean Reciprocal Rank)** — cares about *where* the first relevant document lands. If the first relevant doc is at rank 3, reciprocal rank = 1/3. Averaged across many queries. Good for "did we surface something useful near the top," which matters because generators tend to weight earlier context more heavily.

### Generation-side metrics — the RAG-specific ones interviewers really want

**Faithfulness (a.k.a. groundedness):** does the generated answer only contain claims that are actually supported by the retrieved documents? This is the RAG-specific version of hallucination detection — the model shouldn't state anything the retrieved context doesn't support, even if the claim happens to be true in the real world.

**Worked example of how faithfulness is actually scored (the "claim decomposition" method):** Take the generated answer, break it into atomic claims via an LLM. For each claim, ask a judge LLM: "is this claim entailed by the retrieved context?" Yes/no per claim. Faithfulness score = (# claims supported by context) / (total claims).

Example — retrieved context: "The Eiffel Tower was completed in 1889 and is 330 meters tall." Generated answer: "The Eiffel Tower was completed in 1889, is 330 meters tall, and receives 7 million visitors per year."
- Claim 1 (completed 1889): supported ✓
- Claim 2 (330m tall): supported ✓
- Claim 3 (7 million visitors/year): NOT in context ✗ — even if this happens to be true, it's a faithfulness violation because it wasn't grounded in the retrieved documents.

Faithfulness = 2/3 ≈ 0.67

**Answer relevance:** does the answer actually address the user's question (independent of whether it's grounded)? Typically scored by generating several hypothetical questions the answer *would* be a good response to, then measuring semantic similarity between those and the actual user question — if the answer is off-topic, the generated hypothetical questions won't resemble the real one.

**Context relevance:** were the retrieved documents actually relevant to the query in the first place? (This overlaps with precision@k above, but often re-scored by an LLM judge rather than a fixed relevance label.)

**Why decompose into these three (faithfulness, answer relevance, context relevance) instead of one score:** each failure mode needs a different fix. Low context relevance → fix the retriever/embedding model. Low faithfulness with high context relevance → the generator is ignoring good context and hallucinating — fix the prompt or the generator. Low answer relevance with high faithfulness → the model is grounded but not actually answering the question asked. This is exactly the diagnostic value tested by frameworks like **RAGAS**, which formalizes these three metrics.

## Summarization Evaluation

Beyond ROUGE (Chapter 2, and its known weaknesses), summarization needs metrics that catch its specific failure modes:

- **Faithfulness/factual consistency** — same core idea as RAG faithfulness: does the summary state anything not supported by the source document? This is the single most common real-world summarization failure and the reason ROUGE alone is considered insufficient in production settings.
- **Coverage/informativeness** — does the summary capture the key points, or does it fixate on one section and miss others?
- **Coherence** — does the summary read as a well-formed, logically flowing text, not just a bag of extracted facts?
- **Conciseness** — is it appropriately compressed, not padded with filler?

**Practical scoring approach in production:** typically an LLM judge scores each axis independently (1-5) using a rubric like Chapter 3's, sometimes combined with an automatic factual-consistency classifier (a smaller model fine-tuned specifically to detect entailment/contradiction between source and summary sentences).

## Code Generation Evaluation

We covered execution-based scoring (pass@k) in Chapter 5 for standalone functions. In more realistic "code assistant" settings, evaluation extends further:

- **Functional correctness** — pass@k against unit tests, as before.
- **Code quality metrics** — beyond "does it run," does it follow style conventions, avoid obvious inefficiencies, avoid security anti-patterns (e.g., SQL injection risk, hardcoded secrets)? Often checked via static analysis tools layered on top of the LLM output, not the LLM judge itself.
- **Edit-distance / patch-based eval (e.g., SWE-bench style):** for tasks like "fix this GitHub issue," the model's output is a code diff, evaluated by actually applying the patch and running the repository's real test suite — much closer to real engineering evaluation than an isolated function.

## Agent Evaluation — the newest and hardest category

Agents chain multiple LLM calls with tool use, memory, and multi-step planning — so evaluation must assess the whole *trajectory*, not just a final text output.

**Task success rate:** did the agent actually accomplish the end goal (e.g., "book this flight," "fix this bug and get tests passing")? Usually binary or partial-credit, checked programmatically when possible (e.g., did the booking API return a confirmed reservation?).

**Trajectory/process evaluation:** beyond just success/failure, did the agent take a *reasonable path* — right tool calls, no redundant/wasted steps, correct order of operations? Two agents can both "succeed" but one takes 3 efficient tool calls and the other flails through 15 including several errors — that's a real quality difference task success rate alone won't show you.

**Worked example of the distinction:** Task = "find the weather in Paris and email it to John." Agent A: calls weather API once, calls email API once, done — 2 steps, task succeeds. Agent B: calls weather API, gets confused, calls it again, tries to email but uses the wrong address, retries, eventually succeeds — 6 steps, one wasted retry, one near-failure. Both get "task success = 1," but any real production eval needs to also report **steps-to-completion** and **error/retry rate** to distinguish A from B.

**Tool-use correctness:** for each tool call the agent makes, was the tool chosen appropriately, and were the arguments correctly formatted/valid? Often evaluated per-call, similar to a precision metric — (# correct tool calls) / (total tool calls).

## The unifying theme across this whole chapter

Every task-specific eval above follows the same design pattern you should recognize instantly in an interview: **decompose the system into its functional stages, and measure each stage separately**, because black-box "is the final output good" scoring can't tell you *what to fix*. Retrieval vs. generation in RAG. Faithfulness vs. coverage vs. coherence in summarization. Functional correctness vs. code quality in codegen. Task success vs. trajectory efficiency in agents. This decomposition-for-diagnosis principle is worth stating explicitly if asked "how would you design an eval suite for [some new system]" — it's the transferable framework.

## Quick check

A RAG-based customer support bot has high context relevance (retriever is finding the right documents) but low faithfulness (answers include invented details). Where's the bug — in the retriever or the generator — and what would you actually change?

**The generator.** Good documents are being retrieved (context relevance is fine), but the generation step is failing to stay grounded in them — it's adding unsupported claims. Fix options: tighten the prompt to explicitly instruct "only use information from the provided context," reduce generation temperature, or fine-tune/select a generator model with better instruction-following for grounded QA. Swapping out the retriever wouldn't help — that's not where the problem lives.

---

Chapter 7 is Safety & Red-Teaming Evals — toxicity, bias, and jailbreak/adversarial testing. Want me to continue?
