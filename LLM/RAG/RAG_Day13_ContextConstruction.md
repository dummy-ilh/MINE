# RAG Interview Prep — Day 13
## Context Construction & Lost-in-the-Middle

---

## 🚀 Quick Summary

Retrieval and reranking (Days 7–10) get you a good, well-ordered set of chunks — but *how you assemble those chunks into the actual prompt* sent to the generator is its own critical design problem, with its own well-documented failure mode: **lost-in-the-middle**. Empirical research on long-context LLMs consistently shows models attend most reliably to information near the *start* and *end* of a context, with a measurable dip for information buried in the *middle* — meaning where you place your best evidence in the prompt matters, not just whether it's included at all. Today covers the mechanics of that effect, practical context-ordering strategies, prompt template design for RAG, and how to actually budget a context window across competing needs (instructions, retrieved evidence, conversation history, generation headroom).

**Think of it like packing a suitcase a TSA agent will only glance at the top and bottom of.** If your most important item is buried in the middle of the suitcase, there's a real chance it gets overlooked even though it's technically "in there." Context construction is packing the suitcase deliberately — putting the things that matter most where they're most likely to actually be seen, not just included.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Lost-in-the-middle** | The empirical tendency of LLMs to attend less reliably to information placed in the middle of a long context vs. the start or end |
| **Primacy effect** | Stronger attention/recall for information near the *start* of context |
| **Recency effect** | Stronger attention/recall for information near the *end* of context |
| **Context ordering** | The deliberate arrangement of retrieved chunks within the prompt, not just which chunks are included |
| **Sandwiching** | Placing the most important content at both the start and end of context, leaving less critical content in the middle |
| **Context window budget** | The allocation of a finite token window across system instructions, retrieved context, conversation history, and generation headroom |
| **Citation markers** | Structured tags/identifiers attached to each chunk in the prompt so the generator can attribute claims back to specific sources |

---

# PHASE 1 — Intuition & The Empirical Effect

## What "lost in the middle" actually means, concretely

The core empirical finding (from long-context LLM evaluation research) is a **U-shaped performance curve**: given a long context containing one clearly relevant piece of information, models answer correctly most reliably when that information sits near the **beginning** or **end** of the context, and performance measurably *degrades* when the same information is placed in the **middle** — even though every version of the test has the exact same relevant fact, technically present in the context either way.

```
   MODEL ACCURACY ON "FIND THE FACT" TASK, BY POSITION OF THE FACT
   IN A LONG CONTEXT (illustrative U-shape, not exact real numbers):

   Accuracy
   100% ┤●                                              ●
    90% ┤ ●                                            ●
    80% ┤  ●                                          ●
    70% ┤    ●                                      ●
    60% ┤       ●                                ●
    50% ┤           ●                        ●
    40% ┤                ●●            ●●
        └──────────────────────────────────────────────────
        Start          ...        Middle        ...      End
                          Position of the relevant fact
```

**Why this matters for RAG specifically:** this isn't a theoretical curiosity — it directly determines how you should order retrieved chunks before sending them to the generator. If your reranker (Day 10) has correctly identified the single most relevant chunk, but you then place that chunk 4th out of 8 in the prompt, you may be handing the model a context where its *best* evidence sits in exactly the position it's least reliable at using — undermining the whole point of a careful retrieval and reranking pipeline.

## Why this happens (mechanistic intuition, not full derivation)

While the precise underlying cause is still an active research area, the practical intuition worth stating in an interview: attention mechanisms in transformers don't treat all positions perfectly uniformly in practice, and training data / positional encoding schemes tend to create stronger learned associations for information near sequence boundaries. The **practical takeaway matters more than the precise mechanism** for an interview — you should know the effect exists, roughly its shape, and what to do about it, more than you need a from-scratch theoretical explanation.

---

# PHASE 2 — Deep Dive: Practical Context Construction Strategies

## 1. Context Ordering — Sandwiching

**The strategy:** given a ranked list of retrieved/reranked chunks (best to worst), don't simply place them in that order start-to-end. Instead, deliberately position the **most relevant chunks at both the start and the end** of the context, pushing less critical (but still potentially useful) chunks toward the middle.

**Worked example — ordering 6 reranked chunks (ranked 1=best to 6=worst) using sandwiching:**
```
Rerank order (by relevance): [1, 2, 3, 4, 5, 6]

Naive ordering (just use rerank order top-to-bottom):
  Position 1 (start): chunk 1  ← good
  Position 2:          chunk 2
  Position 3:          chunk 3  ← "middle," lost-in-the-middle risk
  Position 4:          chunk 4  ← "middle," lost-in-the-middle risk
  Position 5:          chunk 5
  Position 6 (end):    chunk 6  ← this is actually your WORST chunk,
                                    sitting in a high-attention position —
                                    wasteful and potentially misleading

Sandwiched ordering (best chunks at both ends, worst in the middle):
  Position 1 (start): chunk 1   ← best chunk, high-attention position ✓
  Position 2:          chunk 3
  Position 3:          chunk 5  ← weaker chunks pushed to the
  Position 4:          chunk 6     middle, where attention is weakest
  Position 5:          chunk 4     anyway — lower cost if they're
  Position 6 (end):    chunk 2   ← 2nd-best chunk, also a high-attention
                                     position ✓
```
**Why this matters in practice:** the naive "just list them in rerank order" approach accidentally wastes a high-attention position (the very end of the context) on your *worst* retrieved chunk, while your reranker's hard-won top result sits in a comparatively higher-risk middle-ish position by the time you account for whatever system instructions or other content precede it. Sandwiching deliberately protects your two most important pieces of evidence by placing them exactly where the model is most likely to actually use them.

## 2. How Many Chunks to Include (k) — More Isn't Always Better

**The tension:** including more chunks (higher k) increases the chance that *some* relevant evidence is present (recall-side benefit), but also (a) pushes more content into the vulnerable "middle" zone, and (b) dilutes the context with more marginal/irrelevant material, both of which can hurt generation quality even while technically improving retrieval-side recall.

**Practical implication:** the optimal k for *generation quality* is not necessarily the same as the optimal k for *retrieval Recall@k* (Day 4/Module 7's distinction) — a k that maximizes Recall@k might actually be higher than the k that maximizes downstream faithfulness/answer relevance, because past a certain point, additional chunks add more lost-in-the-middle risk and noise than they add useful coverage. This is exactly why chunk count/context size should be tuned against **downstream generation metrics** (faithfulness, answer relevance — Module 7/evaluation week concepts), not retrieval metrics alone — echoing the same "don't optimize one stage in isolation" theme from earlier weeks.

> **Why This Matters callout:** If asked "would increasing k always improve your RAG system," the strong answer is no, and cites lost-in-the-middle directly: past some point, more retrieved chunks add more risk of diluting/burying the truly relevant evidence in a low-attention middle zone than they add useful additional coverage — the right k should be tuned empirically against generation-stage metrics, not assumed to monotonically improve with more context.

## 3. Context Window Budget Allocation

**The problem:** a context window is a finite, shared resource across several competing needs — you can't spend it all on retrieved evidence alone.

**Typical components competing for budget:**

| Component | Purpose | Typical share of budget (illustrative) |
|---|---|---|
| **System instructions** | Persona, output format rules, citation format requirements, refusal/uncertainty instructions | Small, fixed (~5-10%) |
| **Retrieved context** | The actual evidence chunks from retrieval/reranking | Largest share (~50-70%) |
| **Conversation history** | Prior turns in a multi-turn interaction | Variable — can compete heavily with retrieved context in long conversations |
| **Generation headroom** | Reserved space for the model's actual output | Fixed, must be reserved *before* filling the rest, not as an afterthought |

**Worked numerical example — budgeting an 8,000-token context window:**
```
Total context window:        8,000 tokens
Reserve for generation:      1,500 tokens (model's actual answer)
System instructions:           400 tokens (fixed, persona + format rules)
Conversation history budget: 1,500 tokens (recent turns, possibly summarized
                                            if the conversation is long)
──────────────────────────────────────────
Remaining for retrieved context: 8,000 - 1,500 - 400 - 1,500 = 4,600 tokens

If average chunk size is 400 tokens (Day 3):
  max chunks that fit ≈ 4,600 / 400 = 11.5 → 11 chunks
```
**Why this matters in practice:** this is the concrete mechanism by which conversation history "competes" with retrieved evidence for space — in a long-running multi-turn conversation, if history isn't summarized or truncated (a design decision worth calling out explicitly), it can silently crowd out how many retrieved chunks fit in the remaining budget, degrading retrieval-side generation quality for reasons that have nothing to do with retrieval itself. Reserving generation headroom *first*, before allocating the rest, is also worth stating explicitly — a common mistake is filling the context greedily with input content and only discovering there's insufficient room left for a complete answer.

## 4. Prompt Template Design for RAG

**A well-structured RAG prompt typically includes, in order:**
1. **System instructions** — persona, output format, explicit citation requirements, and critically, **explicit instructions to only use the provided context** and to say "I don't know" / decline when the context doesn't contain the answer (this is the runtime mechanism behind Module 7's faithfulness metric and the "no good answer exists" eval slice).
2. **Retrieved context, clearly delimited per chunk** — using structural markers (e.g., XML-style tags, numbered source labels) so the model can distinguish where one chunk ends and another begins, and so it has a clean way to cite back to a specific source.
3. **The user's actual query** — typically placed at (or very near) the **end** of the prompt, both because that's a natural place for it and because recency-effect attention means the model's "final impression" before generating is the actual question being asked.

**Worked example — a well-structured context block:**
```
<source id="1" title="AirPods Pro Support Guide">
Battery life for AirPods Pro is up to 6 hours of listening time on
a single charge, with an additional 24 hours available via the
charging case.
</source>

<source id="2" title="AirPods Pro Return Policy">
Products may be returned within 14 days of the original purchase
date for a full refund, provided they are in original condition.
</source>

Instructions: Answer the user's question using ONLY the information
in the sources above. Cite the source id(s) you used in your answer.
If the sources do not contain enough information to answer, say so
explicitly rather than guessing.

User question: What is the battery life of AirPods Pro?
```
**Why this structure matters:** explicit `<source id>` tags give the model a clean, unambiguous way to produce citations (directly supporting faithfulness verification, Module 7 §7.3), the "ONLY use the information in the sources" instruction is the runtime behavioral counterpart to the faithfulness metric measured after the fact, and placing the actual question at the very end leverages the recency effect for the part of the prompt that most directly determines what the model should focus on producing.

## 5. Handling Very Large Retrieved Context — Map-Reduce Style Processing

**The problem:** sometimes the amount of genuinely relevant retrieved content exceeds what reasonably fits (or should be trusted) in a single context window, even after careful budget allocation.

**Map-reduce style approach (borrowed from distributed computing terminology):**
1. **Map step:** split the retrieved chunks into smaller groups, and run a separate generation pass over each group independently (e.g., "summarize what's relevant to the query from this subset of chunks").
2. **Reduce step:** combine the outputs of all the map-step passes into a final synthesis pass, producing the actual answer.

**Trade-off to state explicitly:** this technique trades additional LLM calls (cost, latency — the same fundamental trade-off as Day 11's query transformation techniques) for the ability to effectively process more total retrieved content than a single context window could hold at once, without needing to arbitrarily drop chunks. It's most relevant for use cases needing broad synthesis across a genuinely large volume of relevant material (e.g., "summarize everything relevant across 50 documents"), rather than typical single-fact-lookup RAG queries where a well-curated top-k already fits comfortably.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What is the "lost in the middle" effect, and why does it matter for how you order retrieved chunks in a RAG prompt?

<details>
<summary>Show answer</summary>

It's the empirically observed tendency of LLMs to attend less reliably to information placed in the middle of a long context, compared to information near the start or end — a U-shaped accuracy curve by position. For RAG, this means simply including the right chunk in the context isn't enough — *where* it's placed in the prompt affects how reliably the model actually uses it, so the most relevant chunks should be positioned near the start and/or end of the context rather than buried in the middle.
</details>

---

**Q2 (Easy — conceptual).** What is "sandwiching" in the context of RAG prompt construction?

<details>
<summary>Show answer</summary>

Deliberately placing the most relevant retrieved chunks at both the beginning and the end of the context (the two high-attention zones per lost-in-the-middle), while pushing less critical chunks toward the middle — rather than simply listing chunks in rerank order from top to bottom, which can accidentally waste a high-attention position (the end) on a weaker chunk.
</details>

---

**Q3 (Medium — calculation).** A context window is 6,000 tokens. You reserve 1,200 for generation, 300 for system instructions, and 800 for conversation history. If chunks average 350 tokens, how many chunks fit in the remaining budget?

<details>
<summary>Show answer</summary>

```
remaining = 6000 - 1200 - 300 - 800 = 3700 tokens
chunks that fit ≈ 3700 / 350 ≈ 10.6 → 10 chunks
```
</details>

---

**Q4 (Medium — conceptual).** Why might increasing k (the number of retrieved chunks included in context) hurt generation quality even if it improves Recall@k?

<details>
<summary>Show answer</summary>

Higher k means more content is pushed into the "middle" of a longer context, increasing lost-in-the-middle risk for genuinely relevant chunks, and also dilutes the context with more marginal/irrelevant material, which can lower context relevance (Module 7) even while retrieval-side Recall@k technically improves (since Recall@k only checks whether relevant docs appear anywhere in top-k, not how well the generator actually uses them). This is why the optimal k for generation quality should be tuned against downstream generation metrics, not just retrieval recall in isolation.
</details>

---

**Q5 (Medium — conceptual).** Why should generation headroom be reserved first when budgeting a context window, rather than filled in last after allocating everything else?

<details>
<summary>Show answer</summary>

If input content (system instructions, retrieved context, conversation history) is allocated first without reserving space for the output, there's a risk of discovering too late that insufficient room remains for the model to produce a complete answer — leading to truncated or cut-off generations. Reserving a fixed generation budget upfront, before allocating the rest of the window to input content, guarantees the model always has room to actually finish its response.
</details>

---

**Q6 (Hard — system design synthesis).** Design the context construction strategy for a RAG system handling long multi-turn conversations where conversation history is competing for space with retrieved context, given a fixed 10,000-token window. Walk through your reasoning.

<details>
<summary>Show answer</summary>

I'd reserve a fixed generation budget first (e.g., 1,500 tokens), then a fixed system instruction budget (e.g., 400 tokens). For conversation history, rather than including every prior turn verbatim (which would grow unboundedly and increasingly crowd out retrieved context as the conversation lengthens), I'd implement a summarization or truncation strategy — e.g., keep the last 2-3 turns verbatim (recency matters for immediate conversational coherence) and summarize older turns into a compact running summary, capping conversation history at a fixed budget (e.g., 1,500-2,000 tokens) regardless of actual conversation length. This leaves the remaining, largest share of the window (roughly 6,000+ tokens) reliably available for retrieved context, sandwiched appropriately (best chunks at start/end) rather than having that budget silently shrink as a conversation grows longer — directly addressing the "conversation history competing with retrieved context" tension without letting one component starve the other unpredictably.
</details>

---

**Q7 (Hard — conceptual, ties across weeks).** Explain how lost-in-the-middle interacts with the reranker's job (Day 10) — specifically, why does a reranker's output ordering matter even more once you understand this effect?

<details>
<summary>Show answer</summary>

A reranker's entire value proposition (Day 10) is producing an accurate relevance ordering among a candidate set — but that value is only fully realized if the *downstream context construction* respects that ordering when placing chunks in the prompt. If the reranker correctly identifies the best chunk but it then gets placed in a low-attention middle position (e.g., because chunks are ordered by document ID or retrieval-source order rather than rerank score), the reranker's accurate judgment is effectively wasted — the model may still fail to use that top-ranked evidence reliably. This is why context construction (today's topic) should be considered a direct continuation of the reranking stage, not an independent afterthought — sandwiching the reranker's top results at the start and end of context is what actually cashes in the reranker's accuracy improvement into better generation behavior.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Assuming that if a relevant chunk is anywhere in the context, the model will reliably use it — position matters, not just presence.
- ❌ Ordering chunks by retrieval-source or document order instead of by relevance/rerank score when constructing the prompt, wasting the reranker's careful ordering.
- ❌ Assuming more retrieved chunks (higher k) is always better — past a point, more chunks add lost-in-the-middle risk and noise faster than they add useful coverage.
- ❌ Filling the context window greedily with input content before reserving generation headroom, risking truncated outputs.
- ❌ Letting conversation history grow unboundedly in multi-turn systems, silently crowding out retrieved context over the course of a long conversation.
- ❌ Not giving the generator explicit citation-format and "say I don't know" instructions — relying on faithfulness to emerge implicitly rather than being explicitly instructed for at runtime.

---

# 📌 Cheat Sheet (Day 13)

**Lost-in-the-middle:** U-shaped attention/accuracy curve by position — start and end are high-attention, middle is where relevant info is most likely to be underused, even though it's technically present.

**Sandwiching:** best chunks at both start AND end of context; weaker chunks pushed to the middle, where their lower quality costs less.

**k tuning:** more chunks ≠ always better — optimal k should be tuned against downstream generation metrics (faithfulness, answer relevance), not just Recall@k.

**Budget allocation:** reserve generation headroom first, then fixed system instructions, then bounded/summarized conversation history, then retrieved context gets what's left — don't let history grow unboundedly and silently starve retrieved context.

**Prompt structure:** clearly delimited sources with citation IDs + explicit "use only this context, say I don't know if insufficient" instructions + query placed near the end (recency effect).

**Very large context needs:** map-reduce style multi-pass processing trades extra LLM calls for handling more total content than one window can hold.

**Golden interview line:** *"A reranker's ordering is only as valuable as what you do with it downstream — sandwiching the top-ranked chunks at the start and end of context is what actually converts a reranker's accuracy into better generation, given the empirical lost-in-the-middle effect."*

---

*End of Day 13. Next up — Day 14: Context Window Management & Compression.*
