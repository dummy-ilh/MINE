# RAG Interview Prep — Day 11
## Query Transformation: HyDE, Multi-Query, Decomposition

---

## 🚀 Quick Summary

So far, every retrieval technique (Days 7–10) has assumed the query itself, as typed by the user, is what gets embedded or matched against the corpus. Query transformation challenges that assumption: sometimes the raw query is short, ambiguous, phrased very differently from how the answer is written, or actually contains multiple distinct sub-questions bundled together — and in all of these cases, transforming the query *before* retrieval (rather than sending it as-is) measurably improves what gets retrieved. Today covers the four standard techniques: **multi-query generation**, **HyDE (Hypothetical Document Embeddings)**, **query decomposition** for multi-hop questions, and **step-back prompting** — plus the honest cost side of the ledger, since every one of these techniques trades extra LLM calls (latency, cost) for better retrieval.

**Think of it like asking a reference librarian a vague question.** If you walk in and ask "that thing about the battery issue," a good librarian doesn't just search the catalog for your exact words — they might rephrase your question a few different ways in their head (multi-query), imagine what a helpful answer might actually say and search for text that resembles *that* (HyDE), or ask you a clarifying follow-up and break your question into parts if it's really three questions at once (decomposition). Query transformation is teaching the retrieval system to do this "librarian's mental rephrasing" step automatically, using an LLM, before ever touching the index.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Query transformation** | Modifying or expanding the query before retrieval, rather than using the raw user input as-is |
| **Multi-query generation** | Using an LLM to generate several varied phrasings of the same query, retrieving for each, and combining results |
| **HyDE (Hypothetical Document Embeddings)** | Generating a hypothetical answer to the query via an LLM, then embedding *that* hypothetical answer (instead of the query) for retrieval |
| **Query decomposition** | Breaking a complex, multi-part question into simpler sub-questions, retrieving separately for each |
| **Step-back prompting** | Generating a more general/abstract version of the query first, to retrieve broader supporting context |
| **Query-document asymmetry** | The structural mismatch between how questions are phrased and how answers/documents are written (from Day 8) |
| **Pseudo-relevance feedback** | A classical IR technique: assume the top initial results are relevant, extract terms from them, and expand the query with those terms |

---

# PHASE 1 — Intuition & Visual Map

## Why the raw query often isn't the best thing to search with

```
   USER QUERY:  "why does my battery die so fast"

   ACTUAL RELEVANT DOCUMENT TEXT:  "Battery degradation over time is
   normal; capacity typically decreases to around 80% after 500 full
   charge cycles. To maximize longevity, avoid extreme temperatures..."

   Notice: the query is short, casual, first-person, symptom-framed.
   The document is technical, third-person, cause-and-explanation-framed.
   Even a good embedding model has to bridge a real STYLE gap, not just
   a topic gap — this is Day 8's query-document asymmetry problem,
   showing up as a retrieval quality issue.
```

This gap — between how people *ask* things and how information is actually *written* — is the unifying problem every technique today addresses, just via different mechanisms.

## The four techniques, at a glance

```
                      THE QUERY HAS A PROBLEM — WHICH ONE?
                                    │
     ┌───────────────┬──────────────┼──────────────┬───────────────┐
     ▼               ▼              ▼               ▼
  "It's phrased    "It's phrased   "It's actually   "It's too
   one narrow       very           several          specific,
   way — I might     differently    questions        missing
   miss variants"    from how       bundled          broader
                      answers are    together"        context"
                      written"
     │               │              │               │
     ▼               ▼              ▼               ▼
  MULTI-QUERY       HyDE          DECOMPOSITION   STEP-BACK
```

---

# PHASE 2 — Deep Dive: Mechanics, Math, and Worked Examples

## 1. Multi-Query Generation

**Mechanism:** prompt an LLM to generate several (typically 3-5) alternative phrasings of the user's original query, run retrieval separately for *each* phrasing, then combine all the resulting candidate sets (via simple union, or more robustly via RRF from Day 9).

**Worked example:**
```
Original query: "how do I fix AirPods that won't connect"

LLM-generated variants:
  1. "AirPods Bluetooth pairing troubleshooting steps"
  2. "why won't my AirPods connect to my phone"
  3. "resetting AirPods connection issues"
  4. "AirPods not showing up in Bluetooth devices list"

→ Retrieve top-k for EACH of the 4 variants (plus optionally the
  original query itself = 5 retrieval calls total)
→ Combine all 5 candidate sets via RRF (Day 9)
```

**Why it helps:** any single phrasing might miss a relevant document that happens to use different vocabulary than that specific phrasing — casting a wider net across several phrasings increases the chance that *at least one* variant's vocabulary overlaps well with how the relevant document is actually written. This directly attacks the vocabulary-mismatch problem without needing any special architecture — it's a "brute force more attempts" solution, in a good sense.

**The cost side (state this explicitly, always):** multi-query generation requires (a) one LLM call to generate the variants, and (b) N separate retrieval calls (one per variant) instead of one — directly multiplying retrieval latency/cost by roughly the number of variants, before even combining results. This is the fundamental trade-off with every query transformation technique: better recall, at the cost of extra LLM calls and retrieval calls.

---

## 2. HyDE (Hypothetical Document Embeddings)

**The core insight:** instead of embedding the *query* and searching for documents close to it, generate a **hypothetical answer** to the query using an LLM, then embed *that hypothetical answer* and search for real documents close to it.

**Why this works — the actual mechanism:** this directly attacks the query-document asymmetry problem from the diagram above. A query ("why does my battery die so fast") and a real answer document are structurally different kinds of text — but a *hypothetical* answer generated by an LLM ("Battery degradation occurs naturally over charge cycles due to...") is written in the same *style and register* as real answer documents, even though its specific facts might be wrong or made up. Embedding this hypothetical answer, rather than the raw query, means you're now doing document-to-document style similarity matching instead of query-to-document style matching — which an embedding model (trained mostly on natural text-to-text similarity, not question-to-answer similarity specifically) is often structurally better at.

```
   STANDARD RETRIEVAL:                    HyDE:

   query ──▶ embed ──▶ search              query ──▶ LLM generates a
                                                       HYPOTHETICAL answer
   (query-to-document                                       │
    style mismatch)                                          ▼
                                                       hypothetical answer
                                                       ──▶ embed ──▶ search

                                            (document-to-document style
                                             match — closes the gap)
```

**Worked conceptual example:**
```
Query: "why does my battery die so fast"

HyDE-generated hypothetical answer (from an LLM, NOT retrieved —
just generated, and it doesn't need to be factually correct):
  "Battery capacity naturally decreases over time due to chemical
   degradation within the cells. Factors such as frequent full
   discharge cycles, exposure to extreme temperatures, and age all
   contribute to reduced battery life and faster depletion."

→ This hypothetical text is embedded (not the original query)
→ The resulting vector is used to search the real corpus
→ Real documents with similar STYLE and CONTENT (actual battery
  degradation explanations) rank highly, because the hypothetical
  answer's embedding sits in a similar region of vector space to
  real answer documents, even though the hypothetical text itself
  is discarded after generating its embedding
```

**Critical nuance (a favorite interview trap):** the hypothetical document is **never shown to the user and never fact-checked** — it's purely a retrieval aid, generated and then discarded once its embedding has been used for the search. Its factual accuracy doesn't matter for retrieval purposes; only its stylistic/structural resemblance to real answer documents matters. This is a common point of confusion — HyDE is not "generate an answer and show it," it's "generate a fake answer purely to get a better *search vector*."

**Cost trade-off:** one extra LLM call (to generate the hypothetical document) before retrieval even starts — cheaper than multi-query in terms of retrieval calls (still just one retrieval call, using the hypothetical document's embedding), but adds LLM generation latency up front.

---

## 3. Query Decomposition (Multi-Hop)

**The problem:** some queries genuinely require synthesizing information from *multiple* distinct pieces of evidence — no single chunk in the corpus fully answers the question, because the question itself bundles multiple sub-questions.

**Worked example:**
```
Original query: "Is the AirPods Pro battery life better than the
                  AirPods Max, and which one is cheaper?"

Decomposed into sub-questions:
  1. "What is the battery life of AirPods Pro?"
  2. "What is the battery life of AirPods Max?"
  3. "What is the price of AirPods Pro?"
  4. "What is the price of AirPods Max?"

→ Retrieve separately for EACH sub-question (likely hitting
  different chunks/documents for each)
→ Pass all retrieved evidence, plus the original composite question,
  to the generator, which synthesizes the final comparative answer
```

**Why this matters in practice:** a single embedding of the original composite query would try to find one chunk that's simultaneously about AirPods Pro battery life, AirPods Max battery life, AND both products' prices — such a chunk almost certainly doesn't exist in the corpus, since real documents are usually about one product's specs, not comparative summaries across products. Decomposition sidesteps this by recognizing the query as multiple simpler, individually-answerable questions, each of which likely *does* have a well-matching chunk somewhere in the corpus.

**How decomposition is typically triggered:** an LLM call classifies/detects whether a query is "simple" (single-hop, answerable from one focused retrieval) or "complex" (needs decomposition) — sometimes as an explicit classification step, sometimes implicitly by always attempting decomposition and letting a trivial single-sub-question case degrade gracefully back to standard single-query retrieval.

---

## 4. Step-Back Prompting

**Mechanism:** before retrieving for the specific query, first prompt an LLM to generate a more general, "stepped-back" version of the question, retrieve for *that* broader query too, and use the resulting broader context alongside (not instead of) the specific query's own retrieval.

**Worked example:**
```
Original specific query: "What was Apple's exact operating margin
                            in the AirPods product line for Q3 2024?"

Step-back query: "How does Apple report financial performance and
                    margins for its hardware product lines?"

→ Retrieval for the step-back query surfaces broader contextual
  documents (e.g., general explanation of how Apple's segment
  reporting works, what "operating margin" means in their filings)
→ Retrieval for the original specific query surfaces the actual
  Q3 2024 AirPods-specific numbers (if available in the corpus)
→ Both sets of retrieved context are combined for generation
```

**Why it helps:** very specific queries can retrieve narrowly-matching but context-poor chunks — the generator might find the exact number but lack the surrounding conceptual framing needed to correctly interpret or present it. Step-back prompting deliberately retrieves *both* the precise answer-level content and the broader conceptual scaffolding around it, which tends to improve answer quality on queries that require some background understanding to interpret correctly, not just a literal fact lookup.

---

## Classical Technique — Query Expansion / Pseudo-Relevance Feedback (worth knowing for breadth)

**Mechanism (pre-LLM classical IR technique, still relevant to know):** run an initial retrieval pass with the raw query, assume the top-N results are relevant (without verifying), extract frequently-occurring terms from those top results, and add those terms to the original query for a second retrieval pass.

**Why it's worth mentioning:** this is conceptually the ancestor of LLM-based query expansion techniques — the core idea (use an initial signal to enrich the query before a final retrieval) predates LLMs entirely, and knowing this shows breadth/historical grounding rather than treating query transformation as an entirely new LLM-era invention. The key risk, shared with its modern descendants, is **query drift**: if the initial top results happen to be off-topic, expanding the query with terms from them can actively make the second-pass retrieval worse, not better — the same assumption-based fragility as trusting an unverified retrieval pass to guide the next one.

---

## Cost/Latency Reality Check — The Honest Trade-off Table

| Technique | Extra LLM calls | Extra retrieval calls | Typical latency added | Best for |
|---|---|---|---|---|
| **Multi-query** | 1 (generate variants) | N (one per variant) | Moderate-high (scales with variant count) | Vocabulary-mismatch-prone corpora, no single obviously "right" phrasing |
| **HyDE** | 1 (generate hypothetical doc) | 1 (using the hypothetical doc's embedding) | Moderate (one extra generation call, but still just one retrieval pass) | Strong query-document style asymmetry (short casual queries vs. formal documents) |
| **Decomposition** | 1-2 (detect + decompose) | N (one per sub-question) | High (scales with number of sub-questions, plus synthesis complexity) | Genuinely multi-hop / comparative / multi-part questions |
| **Step-back** | 1 (generate step-back query) | 2 (specific + step-back) | Moderate (roughly doubles retrieval, plus one generation call) | Queries needing conceptual framing alongside a specific fact |

**Worked example — cumulative latency cost of multi-query:**
```
Base retrieval latency (single query): ~50ms (Day 8's bi-encoder+ANN estimate)
LLM call to generate 4 query variants: ~300-500ms (a real, non-trivial cost)
4 additional retrieval calls (parallelizable): ~50ms each, but if run
  in parallel rather than sequentially, adds maybe ~50-70ms total
  (not 200ms) assuming sufficient parallel capacity

Total added latency ≈ 350-570ms on top of the base ~50ms retrieval
```
This is a genuinely significant latency addition — often the single biggest reason query transformation techniques are used selectively (via query routing, similar to Day 9's routing discussion) rather than applied unconditionally to every query. A query that's already simple, unambiguous, and single-hop gains little from these techniques while paying the full latency cost.

> **Why This Matters callout:** If asked "would you apply query transformation to every query," the strong answer is no — describe a routing/triggering strategy (e.g., only decompose queries detected as multi-part, only apply HyDE for query types known to suffer from vocabulary mismatch) rather than universally paying the latency cost of an LLM call (or several) before every single retrieval, echoing the same query-aware-routing principle from Day 9's hybrid search weighting.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What core problem does HyDE address, and why does generating a fake (possibly factually wrong) answer actually help retrieval?

<details>
<summary>Show answer</summary>

HyDE addresses the query-document asymmetry problem — a short, casually-phrased query and a formally-written answer document are structurally different kinds of text, and an embedding model can struggle to match them directly despite being topically related. Generating a hypothetical answer (even a factually imperfect one) via an LLM produces text written in the same style/register as real answer documents; embedding that hypothetical answer and searching with it becomes a document-to-document style match rather than a query-to-document style match, which embedding models are typically better at. The hypothetical answer's factual correctness doesn't matter, since it's discarded after being embedded — only its stylistic resemblance to real documents matters for retrieval.
</details>

---

**Q2 (Easy — conceptual).** Why would you decompose a query like "compare the battery life and price of AirPods Pro vs. AirPods Max" instead of retrieving directly for it?

<details>
<summary>Show answer</summary>

This is a composite query bundling multiple distinct sub-questions (two products, two attributes each). A single embedding of the whole query would need to match a chunk that's simultaneously about both products' battery life and both products' prices — such a chunk likely doesn't exist, since real documents typically cover one product's specs at a time. Decomposing into separate sub-questions (each product's battery life, each product's price) lets each sub-question retrieve against chunks that actually do exist and match well individually, with the generator synthesizing the final comparison afterward.
</details>

---

**Q3 (Medium — conceptual).** Compare multi-query generation and HyDE in terms of how many retrieval calls each requires, and explain the trade-off.

<details>
<summary>Show answer</summary>

Multi-query generation requires N retrieval calls (one per generated variant, typically 3-5), directly multiplying retrieval cost/latency by the number of variants, in exchange for casting a wider vocabulary net across multiple phrasings. HyDE requires only 1 retrieval call, using the embedding of a single generated hypothetical document — cheaper on the retrieval side, but relies on that one hypothetical document successfully bridging the style gap; if the LLM-generated hypothetical answer happens to miss the relevant angle of the question, HyDE doesn't get the "multiple attempts" benefit multi-query provides. Multi-query trades more retrieval cost for broader coverage; HyDE trades a single generation call for closing the query-document style gap without multiplying retrieval calls.
</details>

---

**Q4 (Medium — conceptual, gotcha).** A team implements HyDE and is confused because the system sometimes returns retrieval results based on facts that don't actually appear anywhere in the hypothetical document that was shown in their debug logs — they suspect a bug. What's actually going on?

<details>
<summary>Show answer</summary>

Likely not a bug — this may be a misunderstanding of how HyDE works. The hypothetical document is only used to produce an *embedding* for searching; the actual retrieved results come from real documents in the corpus that are similar (in the embedding space) to that hypothetical document, not from the hypothetical document's specific claimed facts. It's expected and normal for the retrieved real documents to contain different (and hopefully more accurate) specific facts than whatever the LLM happened to hypothesize — the hypothetical document's job was only to produce a useful search vector, not to be factually authoritative or even to closely match the retrieved documents' exact content.
</details>

---

**Q5 (Medium — cost/latency reasoning).** Your RAG system's p95 latency budget for retrieval is 200ms. A colleague suggests applying multi-query generation (4 variants) to every single query to improve recall. What would you push back on, and what would you suggest instead?

<details>
<summary>Show answer</summary>

I'd push back on applying it universally: the LLM call to generate variants alone can cost several hundred milliseconds, likely blowing through a 200ms retrieval budget before even running the actual retrieval calls — and many queries (simple, unambiguous, single-hop) gain little from multi-query generation while still paying its full latency cost. I'd suggest a routing strategy instead: apply multi-query generation selectively, triggered by signals suggesting the query is ambiguous, underspecified, or has historically suffered from vocabulary-mismatch retrieval failures (similar to Day 9's query-routing logic for hybrid search weighting), rather than paying the cost unconditionally for every query regardless of whether it needs the extra help.
</details>

---

**Q6 (Hard — synthesis across techniques).** A user asks: "Why did my AirPods battery life get worse after the last update, and is this covered under warranty?" Walk through how you'd combine multiple query transformation techniques (from today) to handle this well, and justify each choice.

<details>
<summary>Show answer</summary>

This query has two properties worth addressing: it's genuinely multi-part (a causal/technical question about battery degradation after an update, AND a separate warranty-coverage question), and the first part is phrased casually in a way that likely differs stylistically from how the answer is documented. I'd apply **decomposition** first, splitting it into "why would battery life decrease after a software update" and "is battery degradation covered under warranty" — since these are genuinely different topics likely answered by different chunks/documents. For the first sub-question specifically, I'd consider layering **HyDE** on top, since a casual "why did my battery get worse" phrasing plausibly differs in style from a formal technical explanation of software-related battery behavior. For the warranty sub-question, decomposition alone is probably sufficient, since warranty language tends to be more directly matchable without a strong style gap. I would NOT add multi-query generation on top of all of this, since stacking three query transformation techniques would compound latency significantly for marginal additional benefit — decomposition plus selectively-applied HyDE already addresses both underlying problems (multi-part bundling and style mismatch) without over-engineering the pipeline.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Applying every query transformation technique to every query regardless of need — ignoring the real, often-significant latency cost of extra LLM calls.
- ❌ Misunderstanding HyDE as "generate and show an answer" rather than "generate a throwaway document purely to produce a better search vector."
- ❌ Being confused when HyDE's retrieved results don't match the hypothetical document's specific (possibly incorrect) facts — this is expected behavior, not a bug.
- ❌ Using a single embedding for a genuinely multi-part/comparative query instead of decomposing it — expecting one chunk to match content that spans multiple products/topics/facts simultaneously.
- ❌ Forgetting that pseudo-relevance feedback (and its LLM-era descendants) can suffer "query drift" if the initial assumed-relevant results are actually off-topic.
- ❌ Not having a routing/triggering strategy — paying full query-transformation latency cost unconditionally instead of reserving it for query types known to need it.

---

# 📌 Cheat Sheet (Day 11)

**The unifying problem:** raw user queries are often short, ambiguous, stylistically mismatched with real answer documents, or secretly multi-part — all four techniques address a different flavor of this.

**Multi-query:** generate N phrasings, retrieve for each, fuse (e.g., via RRF) — casts a wider vocabulary net, costs N retrieval calls.

**HyDE:** generate a hypothetical (possibly wrong) answer, embed *that* instead of the query, retrieve with it — closes the query-document style asymmetry gap, costs 1 extra LLM call but only 1 retrieval call. Hypothetical doc is discarded after embedding, never shown to the user.

**Decomposition:** break a multi-part/comparative query into sub-questions, retrieve separately, synthesize — necessary when no single chunk could plausibly answer the whole composite question.

**Step-back prompting:** retrieve for both the specific query AND a more general "stepped-back" version — adds conceptual framing alongside precise facts.

**The universal trade-off:** every technique here adds LLM calls and/or retrieval calls, meaningfully increasing latency — use query-aware routing to apply these selectively, not unconditionally.

---

*End of Day 11 — Retrieval week's teaching content complete. Next up — Day 12: Review Day (cold Q&A across Days 7-11, integrated with Days 1-6).*
