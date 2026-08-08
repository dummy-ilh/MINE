# RAG Module 8 — Diagnosis & Debugging

---

## 8.1 The failure taxonomy

Every bad RAG answer traces back to a failure in one (or more) of four distinct places. Naming this taxonomy explicitly, unprompted, is one of the highest-signal things you can do in a system-design or debugging interview.

1. **Retrieval miss** — the relevant chunk was never fetched at all (didn't make it into top-k). Root causes live in Modules 1-4: embedding domain mismatch, bad chunking, wrong index/metric, insufficient k, dense-only retrieval missing an exact-term query.
2. **Retrieval-but-ignored** — the relevant chunk *was* fetched (present in the context sent to the generator) but the generator didn't use it. Root cause lives in Module 6: "lost in the middle" positioning, context overloaded with noise/redundant chunks crowding it out, poor prompt instructions.
3. **Hallucination-despite-context** — the generator fabricates or distorts claims not supported by the retrieved context, even though good context was present and positioned well. A pure generation-stage failure — model overriding provided context with parametric "knowledge," or extrapolating beyond what's actually stated.
4. **Stale-index** — the retrieved content itself is outdated or has been superseded, but the index hasn't been updated (Module 3.6) — the system is faithfully grounding its answer in context that is simply wrong because it's old. This is neither a retrieval bug nor a generation bug in the usual sense — everything "worked correctly" against a stale snapshot of truth.

**Why this ordering matters as a debugging sequence**: each failure type requires ruling out the ones before it. You cannot productively debug generation (#2/#3) until you've confirmed retrieval (#1) succeeded, and you cannot conclude "hallucination" (#3) until you've confirmed the correct chunk was both retrieved *and* well-positioned (ruling out #1 and #2). Jumping straight to "let's fix the prompt" without checking retrieval first is the single most common debugging mistake candidates describe.

---

## 8.2 The debugging workflow: isolate retrieval before blaming generation

Given a bad answer, work strictly in this order:

**Step 1 — Inspect retrieved chunks directly.** Before looking at the generated answer at all, log and manually inspect exactly which chunks were retrieved (and in what order/rank) for the failing query. This single step resolves failure type #1 immediately: is the relevant chunk present anywhere in the retrieved set, yes or no?

**Step 2 — If the chunk is absent (failure type #1):** the problem is upstream, in embedding/chunking/indexing/retrieval strategy (Modules 1-4), not generation. Check:
- Does the query use rare terms/acronyms/exact IDs that dense retrieval would miss (Module 4.1)? → test with BM25 alone, see if sparse retrieval finds it when dense doesn't
- Is the relevant information split across multiple chunks awkwardly, such that no single chunk fully captures it (chunking issue, Module 2)?
- Is k too small (Module 5.6's retrieval-vs-reranking cutoff tradeoff)?

**Step 3 — If the chunk is present but the answer is still wrong (failure types #2 or #3):** the problem is downstream, in augmentation/generation (Module 6). Check:
- Where was the correct chunk positioned in the final assembled prompt — buried in the middle ("lost in the middle," #2)?
- Was the context cluttered with redundant/irrelevant chunks competing for attention (Module 6.3)?
- Does the generated claim actually trace back to *any* content in the context (run a faithfulness check, Module 7.3) — if not, this is #3, pure hallucination despite good context, and the fix is prompt-level (stronger grounding instructions) or model-level (a model less prone to overriding context with parametric knowledge).

**Step 4 — If the chunk is present, well-positioned, faithfully used, and *still* wrong:** check whether the source content itself is outdated (failure type #4, stale index) — verify against the current ground truth outside the system. If so, this is an ingestion-pipeline/index-freshness problem (Module 3.6), not a retrieval or generation bug at all.

**Interview framing**: this workflow is valuable to state explicitly because it demonstrates you don't randomly poke at prompts when something goes wrong — you have a systematic elimination process that narrows the failure to a specific pipeline stage before proposing a fix.

---

## 8.3 Common root causes, mapped to symptoms

| Symptom | Likely root cause | Module |
|---|---|---|
| Fails specifically on queries with acronyms/IDs/exact terms | Dense-only retrieval, no sparse/hybrid component | 4.1-4.3 |
| Fails specifically on paraphrased queries, works on queries that echo source wording | Embedding domain mismatch, or corpus never covered paraphrase variety in training | 1.7 |
| Answer references content but gets facts subtly wrong | Chunk boundary split a fact/table awkwardly, or context ordering buried the precise numeric detail | 2, 6.1 |
| Correct chunk retrieved (verified in logs) but ignored in the answer | "Lost in the middle" positioning, or context overloaded with irrelevant chunks | 6.1, 6.3 |
| Answer confidently states something no retrieved chunk supports | Hallucination despite context — weak grounding instructions, or a model that leans on parametric knowledge | 6.2, 6.5 (Self-RAG/CRAG as mitigations) |
| Answer was correct last month, wrong now, nothing else changed | Stale index — source document updated but not re-ingested/re-embedded | 3.6 |
| Retrieval quality degraded gradually over weeks with no code changes | IVF cluster centroids drifting from data distribution (Module 3.2/3.6), or query distribution shift not represented in original eval set | 3, 7.7 |
| Works fine on eval set, users report bad answers in production | Synthetic eval set doesn't represent real query distribution (Module 7.6's weakness) | 7.6, 7.7 |

---

## 8.4 Monitoring in production

Debugging a single bad case is reactive; production monitoring is proactive — catching degradation before it accumulates into a pattern of user complaints.

- **Retrieval latency** (p50/p95/p99, broken down by stage — embedding, ANN search, reranking, generation): a latency regression in one stage specifically points to that stage's infra (e.g. an index that's grown past its efficient operating range and needs resharding, Module 3.7)
- **Cache hit rate**: many production RAG systems cache embeddings for repeated/similar queries — a dropping cache hit rate can signal a shift in query diversity/distribution worth investigating on its own
- **Drift in query distribution**: track the statistical shape of incoming queries over time (e.g. topic clustering of queries, or embedding-space centroid drift) — a meaningful shift signals that your original golden eval set (Module 7.6) may no longer represent current real usage, and retrieval/chunking choices tuned against the old distribution may be quietly underperforming on the new one, even with zero code changes (directly connects to the "gradual degradation, no code changes" row in 8.3's table)
- **Faithfulness/relevance sampling in production**: periodically run the Module 7 LLM-judge metrics on a sample of live production traffic (not just the static offline eval set) to catch regressions the offline set wouldn't surface — this is the online evaluation practice from Module 7.7 applied specifically as an ongoing monitoring signal rather than a one-time A/B test

---

## Interview Q&A drill

**Q: A user reports a wrong answer. Walk me through exactly how you'd debug it, in order.**
A: First, pull the logged retrieved chunks for that exact query before looking at the generated answer at all — this immediately tells me whether the relevant information was fetched. If it's absent, the problem is upstream in retrieval (check whether it's a dense-embedding blind spot on rare terms, a chunking issue splitting the needed fact across chunks, or k being too small) — I wouldn't touch the prompt or generation config at this point, since the answer was never going to be correct regardless of generation quality. If the relevant chunk is present, I'd next check where it was positioned in the assembled context (lost-in-the-middle risk) and whether it was crowded out by redundant/irrelevant chunks. If it was well-positioned and still ignored or contradicted, I'd run a faithfulness check on the specific claim to confirm it's a genuine hallucination rather than a subtle misreading, and only then look at prompt-level grounding instructions or model choice. Finally, if everything upstream checks out and the answer is still wrong, I'd verify the source content itself isn't simply stale relative to current ground truth.

**Q: Retrieval quality has degraded gradually over the past month with no code or model changes. What are your top hypotheses?**
A: Two leading hypotheses, both consistent with "no code changes but gradual drift": first, if using an IVF-based index, the cluster centroids were trained on a data snapshot and haven't been retrained — as new documents are added over time, the actual data distribution drifts from those original centroids, degrading recall gradually without any explicit failure. Second, the incoming query distribution itself may have shifted (new topics, new phrasing patterns, a new user segment) in a way the original golden eval set doesn't represent — meaning the system's actual real-world performance was degrading even though it would still score fine against the now-outdated offline eval set. I'd check both: inspect index staleness/retraining schedule, and compare recent production query samples against the original eval set's topic/embedding distribution to look for drift.

**Q: How do you distinguish "hallucination despite good context" from "retrieval fetched the wrong context and the model just repeated it faithfully"?**
A: Both would look identical from the final answer alone — wrong information stated confidently. The distinguishing step is checking faithfulness specifically: does the wrong claim in the answer actually trace back to something stated in the retrieved context? If yes, this is a retrieval/content problem — the model was faithful to bad or stale evidence (points to stale-index or a chunking/retrieval issue, not a generation flaw). If no — the retrieved context doesn't contain or support the claim at all — that's pure hallucination despite good context, a generation-stage failure. This is exactly why faithfulness is measured as its own metric independent of overall answer correctness (Module 7.3): it isolates precisely this distinction.

**Q: What production signal would tell you your offline eval set has become stale, before users start complaining?**
A: Monitoring drift in the live query distribution — e.g. clustering incoming query embeddings over time and watching for the centroid or topic mix shifting away from what the original golden eval set represents. A meaningful shift means the system might be silently underperforming on a growing segment of real traffic even while continuing to score well on the (now unrepresentative) static offline eval set, since offline eval only measures performance against the queries it happens to contain. Combined with periodic LLM-judge faithfulness/relevance sampling directly on production traffic — not just the offline set — this surfaces regressions before they show up as user complaints.

---

**Next up: Module 9 — System design & interview synthesis (end-to-end walkthrough, scaling, security, advanced architectures, practice question bank).** Say the word when ready.


# Common RAG Issues: Diagnosis & Fix Guide

Here's a comprehensive, interview-ready breakdown of RAG issues with diagnosis and fixes - using the Andrew Ng style (simple first, then technical):

---

## 📊 QUICK SUMMARY

RAG systems fail in predictable ways. The 5 main issues are: **Missing Content**, **Wrong Chunk**, **Missing Context**, **Wrong Output Format**, and **Hallucination**. Each has specific symptoms, diagnostic tests, and proven fixes. Think of it like a doctor's diagnostic manual - identify the symptoms, run the tests, apply the cure.

---

## 🔍 THE 5 MAIN RAG FAILURE MODES

### 1. MISSING CONTENT (The "It's Not There" Problem)

**The Andrew Ng Take:** 
*"Imagine asking a librarian for a book, but the book isn't in the library. The librarian can't help you because the information simply doesn't exist in the system."*

**What It Is:** 
The relevant information isn't in your knowledge base at all.

**Symptoms:**
- LLM says "I don't have information about that" or gives generic responses
- Responses are vague with no specific details
- Confidence scores from retrieval are very low

**Diagnosis Tests:**
```
Test 1: Direct Question Test
- Ask the question directly to your retriever
- If no chunks come back with high scores → Missing Content

Test 2: Keyword Search Test  
- Search with exact keywords from the question
- Zero or irrelevant results → Missing Content

Test 3: Semantic Search Test
- Use embedding similarity search
- No results above threshold → Missing Content
```

**🔧 Fixes:**

| Fix | How To | Example |
|-----|--------|---------|
| **Add the Data** | Identify gaps and add documents | User asks about "2024 tax laws" but you only have 2023 data → Add 2024 docs |
| **Expand Sources** | Include more diverse sources | Add FAQs, documentation, blog posts, support tickets |
| **Data Augmentation** | Create synthetic Q&A pairs | Generate 50 Q&A pairs from each document |
| **Web Fallback** | Add a "search web" capability | When confidence < 0.7, trigger web search |

**Code Example - Data Augmentation:**
```python
# Simple data augmentation for RAG
def augment_document(doc_text):
    # Generate different versions of same content
    variations = [
        doc_text,  # Original
        doc_text.lower(),  # Lowercase
        doc_text.replace("LLM", "Large Language Model"),  # Expand acronyms
        "Question: " + doc_text,  # Q format
    ]
    return variations
```

**Interview "Gotcha":** 
*"But isn't this just adding more data?"* - No, it's about **strategic** data addition. You need to analyze query patterns and fill specific gaps, not just dump more data.

---

### 2. WRONG CHUNK (The "Wrong Book" Problem)

**The Andrew Ng Take:**
*"The book IS in the library, but the librarian brings you the wrong volume. You asked for Volume 2, but got Volume 1 - related but useless."*

**What It Is:**
The right information exists, but retrieval fetches the wrong chunk/chunks.

**Symptoms:**
- Retrieved chunks are tangentially related but not directly relevant
- The LLM's answer is "close but wrong"
- Important details are missing from the response

**Diagnosis Tests:**
```
Test 1: Visual Inspection
- Print the retrieved chunks alongside the query
- If they don't directly answer the question → Wrong Chunk

Test 2: Overlap Analysis
- Check if key terms from query appear in retrieved chunks
- Low term overlap → Wrong Chunk

Test 3: Semantic Distance Test
- Compute similarity between query and retrieved chunks
- If similarity < 0.7 for top results → Wrong Chunk
```

**🔧 Fixes:**

| Fix | How To | Example |
|-----|--------|---------|
| **Adjust Chunk Size** | Find optimal size (256-512 tokens usually best) | 1000-token chunks have too much noise; 100-token chunks miss context |
| **Overlap Chunks** | Use sliding window overlap | Chunks: 0-500, 250-750, 500-1000 (50% overlap) |
| **Hybrid Search** | Combine keyword + semantic search | BM25 + Dense Retrieval with weighted scores |
| **Re-ranking** | Use a cross-encoder to re-rank top results | Retrieve 20 chunks, re-rank to get top 5 |
| **Query Expansion** | Add synonyms and related terms | "iPhone" → "iPhone, Apple phone, iOS device" |

**Chunk Size Impact Visualization:**
```
Too Small (50 tokens):    "The quick brown fox..." 
                           → Missing context
                           
Optimal (256 tokens):     "The quick brown fox jumps over the lazy dog. 
                           The dog didn't react. This happened in a..."
                           → Good balance

Too Large (1000 tokens):  "The quick brown fox jumps... [800 words] ... 
                           and that's how we built the system"
                           → Too much noise
```

**Code Example - Hybrid Retrieval:**
```python
def hybrid_search(query, docs, k=5):
    # BM25 (keyword) scores
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = sorted(range(len(docs)), key=lambda i: bm25_scores[i])[-k:]
    
    # Dense (semantic) scores  
    query_embedding = embed(query)
    dense_scores = [cosine_similarity(query_embedding, d.embedding) for d in docs]
    dense_results = sorted(range(len(docs)), key=lambda i: dense_scores[i])[-k:]
    
    # Combine with weights
    combined = {}
    for i in set(bm25_results + dense_results):
        combined[i] = 0.7 * dense_scores[i] + 0.3 * bm25_scores[i]
    
    return sorted(combined.items(), key=lambda x: x[1])[-k:]
```

**Interview "Gotcha":**
*"Why not just use huge chunks?"* - Because huge chunks increase retrieval time AND the LLM loses focus. More context = more distractions = lower quality answers.

---

### 3. MISSING CONTEXT (The "Out of Context" Problem)

**The Andrew Ng Take:**
*"The librarian gives you the right book AND the right page... but it's in a language you don't understand. The information is there, but you're missing the context to use it."*

**What It Is:**
The right chunk is retrieved, but it's missing crucial contextual information like:
- What section it's from
- Who wrote it
- When it was written
- Related concepts that were in nearby chunks

**Symptoms:**
- Answers are correct but lack nuance
- Important caveats or conditions are missing
- Responses seem "disconnected" from the larger topic

**Diagnosis Tests:**
```
Test 1: Metadata Check
- Does your chunk include section headers or document info?
- If not → Missing Context

Test 2: Reference Test
- Does the chunk reference things outside itself?
- If yes → Those references need to be included

Test 3: Parent Document Test
- Does the answer make MORE sense if you read the full page?
- If yes → Missing Context
```

**🔧 Fixes:**

| Fix | How To | Example |
|-----|--------|---------|
| **Add Metadata** | Tag chunks with document info | Document: "Tax_Guide_2024.pdf", Section: "Chapter 3", Page: 45 |
| **Parent Document Retrieval** | Retrieve the full document or larger section | Retrieve small chunk, then expand to include surrounding text |
| **Contextual Embedding** | Include surrounding text in the embedding | Embed "Section: Pricing | Chunk: Price is $100" |
| **Summary First** | Retrieve summaries, then expand | Search summaries → get full section for relevant summary |
| **Hybrid Chunking** | Multiple chunk sizes for same doc | Store: small (for precision) + large (for context) |

**Parent Document Retrieval Visual:**
```
Scenario: User asks "What's the maximum refund?"

Query → Chunk Retrieved:
"Maximum refund is $5,000..."
✅ The answer is here!

But the ORIGINAL document says:
"Maximum refund is $5,000... 
⚠️ ONLY for individuals earning under $50,000/year."

Without context → Missing the condition → Wrong answer!

Fix: Store parent document too:
Chunk 1: "Maximum refund is $5,000..." → Link to Parent
Parent: Full section with all conditions → Fetch Parent → Get complete answer
```

**Code Example - Parent Document Retrieval:**
```python
def parent_document_retrieval(query, small_chunks, parent_chunks, k=5):
    # Step 1: Retrieve small chunks
    retrieved = retrieve(query, small_chunks, k=k*3)
    
    # Step 2: Get their parents
    parent_ids = set([chunk.parent_id for chunk in retrieved])
    
    # Step 3: Retrieve the full parents
    parents = [parent_chunks[id] for id in parent_ids]
    
    # Step 4: Re-rank or combine
    final_context = combine_context(parents)
    
    return final_context
```

**Interview "Gotcha":**
*"But doesn't this just mean we should always use big chunks?"* - No, because big chunks have too much noise for retrieval. The trick is: **retrieve small, expand to large.**

---

### 4. WRONG OUTPUT FORMAT (The "Wrong Language" Problem)

**The Andrew Ng Take:**
*"The librarian found the right information, tells it to you... but in French when you asked in English. The answer is correct, but unusable."*

**What It Is:**
The LLM has the right information but outputs it in the wrong format, structure, or level of detail.

**Symptoms:**
- Answer is correct but too long/too short
- Information is there but not organized as requested
- Output is a wall of text when a list was requested

**Diagnosis Tests:**
```
Test 1: Constraint Check
- Does the output meet all prompt requirements?
- If not → Wrong Output Format

Test 2: Output Parsing
- Can you parse the output into expected structure?
- If parsing fails → Wrong Output Format

Test 3: Human Evaluation
- Is the answer useable as-is?
- If you'd need to reformat → Wrong Output Format
```

**🔧 Fixes:**

| Fix | How To | Example |
|-----|--------|---------|
| **Better Prompting** | Be explicit about format | "Output as a bulleted list with 3 items" |
| **Structured Outputs** | Use JSON/XML templates | "Reply in JSON: {question, answer, confidence}" |
| **Few-Shot Examples** | Show examples of good outputs | "Here's how a good answer looks: [EXAMPLE]" |
| **Chain of Thought** | Ask for reasoning step-by-step | "First think, then format the answer..." |
| **Output Validators** | Re-prompt if format is wrong | Check output → If bad → "Reformat as: [TEMPLATE]" |

**Template Example:**
```python
SYSTEM_PROMPT = """
You are a helpful assistant. ALWAYS structure your answer as:

1. SUMMARY: One sentence summary
2. DETAILS: 2-3 bullet points with details
3. CITATION: Reference the source document

Example:
SUMMARY: The maximum refund is $5,000 for eligible individuals.
DETAILS: 
- Must earn under $50,000/year
- Must file before April 15
CITATION: Tax Guide 2024, Chapter 3, Page 12
"""
```

**Interview "Gotcha":**
*"Why can't we just tell the LLM to format properly?"* - Because LLMs need **specific, repeated, and reinforced** instructions. One instruction isn't enough. You need prompts, examples, and validation.

---

### 5. HALLUCINATION (The "Made It Up" Problem)

**The Andrew Ng Take:**
*"The librarian can't find the information, but instead of saying 'I don't know', they confidently make something up."*

**What It Is:**
The LLM generates information that seems plausible but isn't in the retrieved context or is factually wrong.

**Symptoms:**
- Answers are detailed but unverifiable in the source
- Response contradicts retrieved chunks
- Includes numbers/dates/facts that don't exist in context
- Overly confident tone despite no source evidence

**Diagnosis Tests:**
```
Test 1: Source Check
- Can you find the exact answer in the retrieved chunks?
- If no → Hallucination (or missing content)

Test 2: Consistency Check
- Does the answer contradict known facts?
- If yes → Hallucination

Test 3: Specificity Test
- Are there specifics that aren't in the source?
- If yes → Hallucination

Test 4: Re-answer Test
- Re-run with same query and different temperature
- Different answers = high hallucination risk
```

**🔧 Fixes:**

| Fix | How To | Example |
|-----|--------|---------|
| **Better Retrieval** | More relevant context → less hallucination | Improve chunking, search, re-ranking |
| **Stronger Prompting** | "Only use information from the provided context" | "If the answer isn't in the context, say 'I don't know'" |
| **Temperature Control** | Lower temperature (0.0-0.3) | temp=0.1 for factual consistency |
| **Faithfulness Check** | Verify answer against source | RAGAS faithfulness metric |
| **Citation Required** | Force LLM to cite sources | "Cite the exact text you're using" |
| **Self-Consistency** | Ask 3 times, take majority | 3 answers with temp=0.3 → majority vote |

**Prompting to Reduce Hallucination:**
```python
ANTI_HALLUCINATION_PROMPT = """
You are a truth-seeking assistant. Follow these rules STRICTLY:

1. ONLY use information from the provided context below
2. If the context doesn't contain the answer, say "I don't have enough information"
3. For every fact you state, cite the source line from context
4. If information seems incomplete, say what's missing
5. NEVER make up information, numbers, or dates

Context:
{context}

Question: {question}

Your answer (with citations):
"""
```

**Hallucination Detection Code:**
```python
def detect_hallucination(context, answer):
    # Simple overlap check
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    
    # If answer has new factual words not in context
    # Example: "profit" appears in answer but not context
    new_facts = answer_words - context_words
    
    # Flag as potential hallucination
    if len(new_facts) > 0.2 * len(answer_words):  # 20% new info
        return True
    
    return False
```

**Interview "Gotcha":**
*"But hallucinations still happen even with good retrieval!"* - True. Hallucination is an LLM problem, not just a RAG problem. Mitigation strategies include: better prompting, self-consistency, and output validation.

---

## 🏥 DIAGNOSTIC FLOWCHART

```
Is the answer wrong?
    │
    ├─→ Is the information missing from all sources?
    │       └─→ YES: MISSING CONTENT → Add data, expand sources
    │       └─→ NO: Continue
    │
    ├─→ Is the retrieved chunk relevant but incomplete?
    │       └─→ YES: MISSING CONTEXT → Add metadata, use parent retrieval
    │       └─→ NO: Continue
    │
    ├─→ Is the retrieved chunk not directly answering the query?
    │       └─→ YES: WRONG CHUNK → Adjust chunk size, hybrid search, re-rank
    │       └─→ NO: Continue
    │
    ├─→ Is the answer formatted incorrectly?
    │       └─→ YES: WRONG OUTPUT → Better prompting, structured outputs
    │       └─→ NO: Continue
    │
    └─→ Is the answer plausible but not in the context?
            └─→ YES: HALLUCINATION → Stricter prompting, citation required
```

---

## 📈 MONITORING & METRICS

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Hit Rate** | % of queries with relevant retrieval | > 80% |
| **MRR (Mean Reciprocal Rank)** | Position of first relevant result | > 0.7 |
| **Faithfulness** | % of answer supported by context | > 90% |
| **Answer Relevance** | How well answer matches query | > 85% |
| **Context Relevancy** | How relevant retrieved chunks are | > 75% |

**Implementation:**
```python
# Using RAGAS metrics
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

metrics = [faithfulness, answer_relevancy, context_relevancy]
results = evaluate(dataset, metrics=metrics)
print(results)
```

---

## 🎯 QUICK REFERENCE CARD

| Issue | Quick Fix | Advanced Fix |
|-------|-----------|--------------|
| Missing Content | Add more docs | Query-aware data collection |
| Wrong Chunk | Adjust chunk size | Hybrid search + re-ranking |
| Missing Context | Add metadata | Parent doc retrieval |
| Wrong Format | Better prompts | Structured output + validation |
| Hallucination | Stricter prompts | Self-consistency + faithfulness check |

---

## 🎙️ INTERVIEW Q&A (With Andrew Ng Style)

**Q: "How would you debug a RAG system that's giving wrong answers?"**

*A: "Think of it like a doctor diagnosing a patient. I'd check the symptoms systematically:*

*First, I'd look at the retrieval step. Are we retrieving the right chunks? If not, it's a retrieval problem - adjust chunk size, try hybrid search.*

*Second, I'd check if the right information is even in our database. If not, it's a data problem - add more sources.*

*Third, I'd examine what the LLM is doing with the retrieved context. Is it ignoring parts? Then it's a prompting problem.*

*Finally, I'd check the output - is it formatted correctly? If not, add examples to the prompt.*

*I never assume it's just one problem. It's usually a combination."*

---

**Q: "What's the most common RAG mistake you see?"**

*A: "Using the wrong chunk size. People either use chunks that are too small and lose context, or too large and drown in noise.*

*It's like trying to find a specific sentence in a book. If you grab the whole book, it's too much. If you grab just one word, it's not enough. You need the right paragraph - maybe 250-300 words - with some overlap between paragraphs.*

*The best practice is to experiment. Start with 256 tokens with 50% overlap, then adjust based on your use case."*

---

**Q: "How do you handle multi-document questions in RAG?"**

*A: "This is where it gets interesting. For questions that need information from multiple sources, you need to be careful:*

*One approach is to retrieve from multiple documents, then combine the answers - like asking the LLM to synthesize.*

*Another approach is to use a multi-step retrieval process: first find relevant documents, then within those documents find the right chunks.*

*The key trick: after retrieving, re-rank the chunks using a cross-encoder. This gives you the most relevant chunks regardless of which document they're from.*

*But be careful! The more documents you combine, the more likely you are to have contradictions or hallucinations."*

---

## 💡 PRO TIPS

1. **Always log everything:** Store queries, retrieved chunks, and answers for debugging
2. **Test with simple questions first:** "What is this document about?" vs. complex queries
3. **A/B test your changes:** Change one thing at a time and measure the impact
4. **Human evaluation is still king:** Automated metrics are good, but humans catch nuance
5. **RAG is an iterative process:** You'll never get it perfect on the first try

---

This should give you everything you need for interviews on RAG issues, diagnosis, and fixes!
