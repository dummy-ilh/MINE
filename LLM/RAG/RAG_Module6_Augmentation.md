# RAG Module 6 — Augmentation & Generation

---

## 6.1 Prompt construction: context ordering

Once retrieval + reranking (Modules 4-5) produce a final set of chunks, they must be assembled into a prompt for the generator. The order and framing of that assembly materially affects answer quality — this isn't a trivial formatting step.

### "Lost in the middle"
Empirically, LLMs exhibit a **U-shaped attention/recall curve** over long contexts: information placed at the **very beginning** or **very end** of the context is used more reliably than information placed in the **middle**. A relevant chunk buried in the middle of a long concatenated context can be effectively ignored even though it's technically "in context."

**Practical implication for RAG**: don't just concatenate retrieved chunks in arbitrary or purely relevance-descending order. Two common mitigations:
- **Place the most relevant chunk(s) at the beginning and/or end** of the context, with lower-confidence chunks in the middle — sometimes called "sandwiching" the highest-value content
- **Reduce the number of chunks / total context length** where possible (ties directly to reranking's top-n cutoff, Module 5.6) — the fewer tokens between the model and the truly relevant content, the less opportunity for the "lost in the middle" effect to bite

**Interview framing**: this is a good example of a case where *retrieval was correct* (the right chunk was fetched) but *generation still failed* — a distinct failure mode from retrieval failure (Module 8's taxonomy), and a common trap for candidates who only think to debug the retrieval stage.

### Context window budgeting
Total prompt = system instructions + retrieved chunks + conversation history (if any) + user query + reserved space for the output. Budgeting requires explicit tradeoffs:
- More retrieved chunks (higher k/n) → broader evidence coverage, but more competition for attention (worse "lost in the middle" risk) and higher cost/latency
- Reserve output tokens *before* filling context with retrieved chunks — a common bug is filling the context so close to the model's max that the generated answer gets truncated mid-sentence

---

## 6.2 Citation / attribution strategies

Grounding generated answers to specific retrieved spans serves two purposes: **user trust** (letting users verify claims) and **faithfulness enforcement** (constraining the model to answer from evidence, one lever against hallucination).

- **Chunk-ID citation**: instruct the model to tag each claim with the source chunk ID it came from (`[doc_3]`), then map IDs back to actual source documents/URLs for display
- **Span-level grounding**: more granular — model is prompted/trained to output the *exact substring* of the source chunk that supports a claim, not just which chunk. Higher verifiability, harder to get an LLM to do reliably without specific prompting/fine-tuning for it.
- **Post-hoc attribution**: instead of asking the generator to self-report citations (which it can hallucinate, citing a chunk that doesn't actually support the claim), run a *separate* verification pass — for each generated sentence, check via NLI (natural language inference) or an LLM judge whether any retrieved chunk actually entails that sentence. More reliable than self-reported citation, at the cost of an extra inference pass.

**Interview trap**: self-reported citations from the generating model are **not proof of faithfulness** — a model can cite `[doc_2]` for a claim that `doc_2` doesn't actually support (a specific, common hallucination pattern: real citation marker, fabricated content). This distinction — citation presence vs citation *correctness* — is exactly what faithfulness evaluation (Module 7) is designed to catch, and it's a strong answer if asked "how do you know your RAG system isn't hallucinating even though it cites sources."

---

## 6.3 Handling contradictory or redundant retrieved chunks

Real corpora are messy — retrieval can surface:
- **Redundant chunks**: near-duplicate content from multiple sources saying the same thing (wastes context budget, Module 6.1's budgeting problem)
- **Contradictory chunks**: e.g. an outdated policy doc and its updated replacement, both retrieved, saying different things

**Mitigations**:
- **Deduplication before generation**: cluster or near-duplicate-detect retrieved chunks (e.g. via embedding similarity threshold) and collapse near-identical ones, freeing context budget for genuinely distinct information
- **Recency/authority metadata as a tiebreaker**: if metadata includes timestamps or source authority tiers, either filter out stale versions before generation, or explicitly instruct the generator to prefer the more recent/authoritative source when chunks conflict
- **Explicit instruction to surface conflict rather than silently pick one**: for high-stakes domains, sometimes the correct behavior isn't to silently resolve the contradiction but to tell the user "sources disagree — policy A says X, policy B says Y" rather than presenting one confidently as if it were undisputed

---

## 6.4 Context compression / summarization before generation

When retrieved context (even after reranking) is still too large for the budget, or is noisy relative to the actual question, compress before feeding to the generator:

- **Extractive compression**: select only the most relevant *sentences* within each retrieved chunk (rather than the whole chunk) — often done with a lightweight relevance-scoring pass per sentence
- **Abstractive compression**: use a smaller/cheaper LLM to summarize each retrieved chunk down to just the information relevant to the query, before passing to the (usually more expensive) main generator
- **Tradeoff**: compression reduces token cost and can reduce "lost in the middle" risk (shorter context = less to get lost in), but each compression step is itself a place where information can be dropped or subtly distorted — compression is a lossy operation stacked in front of an already-lossy retrieval pipeline, and errors compound.

**When it's worth it**: high query volume where token cost dominates economics, or very long/noisy source documents (e.g. full legal contracts) where the useful signal is a small fraction of the retrieved chunk's raw text.

---

## 6.5 Agentic RAG — LLM decides whether/what/when to retrieve

Standard RAG performs **one fixed retrieval step per query**, regardless of whether the query actually needs external knowledge or whether one retrieval pass suffices. Agentic RAG makes retrieval a **decision** the model makes rather than a **fixed pipeline stage**.

### Tool-calling retrieval
Retrieval is exposed to the LLM as a callable tool/function (same framing as your Agents-track tool-use material) — the model decides *whether* to call it at all (skip retrieval entirely for questions answerable from parametric knowledge or conversation history), and can call it *multiple times* with self-generated queries if the first pass is insufficient. This is architecturally the same idea as the iterative/ReAct-style multi-hop retrieval from Module 4B, generalized beyond just multi-hop questions to *all* retrieval decisions.

### Self-RAG
The model is trained (via special reflection tokens) to explicitly critique its own retrieval and generation process at each step — e.g. emitting a token that signals "the retrieved passage is NOT relevant, retrieve again" or "this generated sentence is NOT well-supported by the retrieved passage, revise." Bakes faithfulness self-checking directly into the generation loop rather than treating it as a separate post-hoc evaluation step (contrast with Module 6.2's post-hoc attribution verification, and Module 7's evaluation-as-a-separate-process framing).

### Corrective RAG (CRAG)
Adds an explicit **retrieval quality check** immediately after retrieval and before generation: a lightweight evaluator (often a smaller model) scores whether retrieved documents are actually relevant/correct/ambiguous. Based on that score:
- **Correct** → proceed to generation as normal
- **Incorrect** → discard retrieved docs, fall back to a different strategy (e.g. web search, query reformulation + retry)
- **Ambiguous** → combine both retrieved docs and supplementary retrieval (e.g. web search) before generation

**Interview framing for the whole 6.5 section**: agentic RAG exists because *fixed single-shot retrieval assumes every query has the same shape and difficulty*, which is false — trivial factual queries don't need it, complex/multi-hop queries need iteration, and some queries the initial retrieval simply fails on and need a fallback strategy. The cost is the same one seen throughout this syllabus: more LLM calls, more latency, more $ — agentic RAG should be reached for when query difficulty is genuinely heterogeneous, not applied uniformly to every request as a default.

---

## Interview Q&A drill

**Q: Your RAG system retrieved the correct passage (verified by manual inspection), but the generated answer is still wrong or incomplete. What's your first hypothesis?**
A: Since retrieval succeeded, this points to a generation-stage failure, most commonly "lost in the middle" — the correct chunk may have been placed in the middle of a long concatenated context and effectively under-attended to by the model relative to chunks near the start/end. First diagnostic step: check where the correct chunk was positioned in the final prompt. If it was buried mid-context, the fix is reordering (surface high-confidence chunks to the start/end) or reducing total context length (tighter top-n from reranking) rather than anything in the retrieval pipeline itself.

**Q: How do you actually know your RAG system's citations are trustworthy, not just present?**
A: Citation presence and citation correctness are different things — a model can emit a syntactically valid citation marker pointing to a real retrieved chunk while the claim it's attached to isn't actually supported by that chunk's content (a specific hallucination pattern, not a formatting bug). The reliable way to verify is a post-hoc check: for each generated claim, run an NLI model or LLM-judge pass asking "does the cited source actually entail this claim" — independent of what the generator itself claims to be citing. This is essentially the faithfulness/groundedness evaluation from Module 7 applied at the individual-citation level rather than the whole-answer level.

**Q: When would you use Corrective RAG (CRAG) over standard single-shot RAG?**
A: When retrieval failures are a known, non-trivial source of bad answers in your system — e.g. a corpus with coverage gaps, or a query distribution that includes questions genuinely outside the corpus's scope — and you want the system to actively detect and route around bad retrievals rather than blindly generating from whatever was retrieved, even if irrelevant. The cost is an added evaluation step (and possibly a fallback retrieval path like web search) on every query, so it's worth it when the failure mode it targets — confidently generating from irrelevant retrieved context — is actually showing up in your error analysis, not as a default architectural choice for every RAG system.

**Q: What's the tradeoff of compressing retrieved chunks before generation, and when is it worth it?**
A: Compression (extractive or abstractive) reduces token cost and can reduce "lost in the middle" risk by shrinking total context, but it's a lossy operation — information can be dropped or subtly altered during compression, and this error stacks on top of whatever imprecision already exists from retrieval and chunking. It's worth it at high query volume where token cost meaningfully affects unit economics, or when source documents are long and low-density (e.g. full legal contracts where the relevant content is a small fraction of the chunk) such that the compression's information loss is smaller than the noise-reduction benefit it provides.

---

**Next up: Module 7 — Evaluation (faithfulness, relevance, groundedness, RAGAS/TruLens, LLM-as-judge, golden eval sets).** Say the word when ready.
