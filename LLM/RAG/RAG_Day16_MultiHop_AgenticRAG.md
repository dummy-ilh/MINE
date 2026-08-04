# RAG Interview Prep — Day 16
## Multi-Hop & Agentic RAG

---

## 🚀 Quick Summary

Day 11's query decomposition was **static** — the system decides all sub-questions upfront, retrieves for each, and combines results in one pass. Today's topic is **dynamic, iterative** multi-hop retrieval: the system retrieves, *reads and reasons about what it found*, and only then decides what to retrieve next — a loop, not a one-shot plan. This is the foundation of **agentic RAG**, where retrieval becomes a tool the model can choose to invoke, repeatedly, deciding for itself when it has enough information to stop. This is a genuinely more powerful (and more failure-prone) pattern than anything covered so far, and it's a common "show me you understand the frontier, not just the basics" interview topic.

**Think of it like the difference between a shopping list and a real investigation.** Static decomposition (Day 11) is like writing a shopping list before you go to the store — you decide upfront exactly what you need. Agentic multi-hop retrieval is like a detective following a case — the detective doesn't know in advance what the third clue will be, because it depends entirely on what the first two clues revealed. Each step's plan depends on the *actual result* of the previous step, not a plan made before any evidence was gathered.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Static decomposition** | Pre-planning all sub-questions upfront, before seeing any retrieval results (Day 11) |
| **Dynamic/iterative multi-hop retrieval** | Retrieving, reasoning about the result, and deciding the *next* retrieval step based on what was actually found |
| **Agentic RAG** | Framing retrieval as a tool an LLM agent can choose to invoke repeatedly, deciding when and what to retrieve and when it has enough to answer |
| **ReAct (Reason + Act)** | A prompting pattern interleaving explicit reasoning steps ("Thought") with tool-use actions ("Action") and their results ("Observation") |
| **IRCoT** | Interleaving Retrieval with Chain-of-Thought — retrieval steps woven directly into a multi-step reasoning process |
| **Self-RAG / reflection** | The model critiques its own retrieved content or draft answer and decides whether to retrieve again before finalizing |
| **Stopping criterion** | The rule that determines when an iterative/agentic retrieval loop should stop and produce a final answer |

---

# PHASE 1 — Intuition & Visual Map

## Static decomposition vs. dynamic multi-hop, side by side

```
   STATIC DECOMPOSITION (Day 11)              DYNAMIC MULTI-HOP (today)

   Question ──▶ LLM plans ALL sub-Qs          Question ──▶ Retrieve (hop 1)
                 upfront                                        │
                      │                                          ▼
          ┌───────────┼───────────┐                    LLM reads result,
          ▼           ▼           ▼                     REASONS about what
       Sub-Q1      Sub-Q2      Sub-Q3                    it learned, and
          │           │           │                       DECIDES next step
          ▼           ▼           ▼                            │
       Retrieve    Retrieve    Retrieve                         ▼
          │           │           │                    Retrieve (hop 2) —
          └───────────┴───────────┘                     based on hop 1's
                      │                                   ACTUAL content,
                      ▼                                   not a pre-made plan
              Combine & Generate                                │
                                                                  ▼
                                                          LLM reasons again...
                                                          repeat until enough
                                                          info gathered
                                                                  │
                                                                  ▼
                                                             Generate answer
```

**The core distinction to articulate clearly:** static decomposition's sub-questions are fixed the moment the original question is parsed — hop 2 doesn't know or care what hop 1 actually returned. Dynamic multi-hop's second retrieval step is *literally determined by* what the first retrieval returned — if hop 1 comes back empty or surprising, the plan adapts, which static decomposition structurally cannot do.

## When you actually need dynamic multi-hop instead of static decomposition

- ✅ **Genuinely sequential dependency questions** — "Who is the CEO of the company that acquired the company that made the original iPod click wheel?" — you can't know what to search for in hop 2 until hop 1 tells you *which company* acquired the click-wheel maker. Static decomposition can't pre-plan this because the second sub-question's content literally doesn't exist until the first is answered.
- ✅ Situations where the system needs to **verify or follow up on** what it found — "the retrieved chunk mentions a policy change but doesn't give the effective date — go find that specifically."
- ❌ Not needed for questions where sub-questions are genuinely independent and knowable upfront (Day 11's AirPods Pro vs. AirPods Max battery/price example) — static decomposition handles that case perfectly well, more cheaply.

---

# PHASE 2 — Deep Dive: Mechanisms and Worked Examples

## 1. The ReAct Pattern (Reason + Act)

**Mechanism:** the model is prompted to interleave three repeating elements: **Thought** (explicit reasoning about the current state and what to do next), **Action** (invoking a tool — here, retrieval — with specific parameters), and **Observation** (the actual result returned by that action) — looping through Thought → Action → Observation as many times as needed before producing a final answer.

**Worked example — a genuinely sequential multi-hop question:**
```
Question: "What is the battery life of the AirPods model that was
           released the same year as the first iPhone SE?"

Thought 1: I need to find out what year the first iPhone SE was
           released before I can identify which AirPods model that
           corresponds to.
Action 1:  retrieve("iPhone SE first generation release year")
Observation 1: "The first-generation iPhone SE was released in
                March 2016."

Thought 2: Now I need to find which AirPods model was released in
           2016.
Action 2:  retrieve("AirPods model released 2016")
Observation 2: "The original AirPods (1st generation) were released
                in December 2016."

Thought 3: Now I have the specific model — I need its battery life.
Action 3:  retrieve("original AirPods 1st generation battery life")
Observation 3: "The original AirPods offer approximately 5 hours of
                listening time on a single charge."

Thought 4: I now have enough information to answer the question.
Final answer: "The AirPods model released the same year as the
              first iPhone SE (2016) is the original AirPods
              (1st generation), which offers approximately 5 hours
              of battery life on a single charge."
```
**Why this example matters:** notice that **hop 2's query** ("AirPods model released 2016") could not have been written *before* hop 1 executed — the year "2016" only became known as a result of Observation 1. This is the concrete proof of why static decomposition (Day 11) is structurally incapable of handling this specific question — there's no way to pre-plan a sub-question that depends on a fact you don't have yet.

---

## 2. Self-RAG / Reflection

**Mechanism:** rather than always retrieving in a fixed pattern, the model is trained or prompted to **critique its own retrieved content** at each step — assessing whether what was retrieved is actually sufficient and relevant, and explicitly deciding whether to retrieve again, retrieve differently, or proceed to generate an answer.

**Key capability this adds beyond basic ReAct:** ReAct's loop structure is often driven by a relatively fixed "keep going until you have an answer" logic. Self-RAG-style reflection adds an explicit **quality gate** at each step — "was that retrieval actually good, or should I try a different query" — which is a more deliberate, quality-aware version of the same iterative idea, closer in spirit to Day 15's runtime groundedness guardrail concept, but applied *during* the retrieval process rather than only at the final output.

**Worked conceptual example:**
```
Action 1: retrieve("AirPods warranty coverage battery")
Observation 1: [returns a chunk about general product warranty terms,
                but nothing specifically about battery degradation
                coverage]

Reflection: This retrieved content doesn't actually address battery-
            specific warranty coverage — it's too general. I should
            retrieve again with a more targeted query rather than
            proceeding with insufficient information.

Action 2: retrieve("AirPods battery degradation warranty policy")
Observation 2: [returns a more specifically relevant chunk]

Reflection: This is now sufficiently relevant and specific. Proceed
            to generate the answer.
```
**Why this matters in practice:** without this reflection step, a naive iterative pipeline might proceed to generate an answer from Observation 1's overly-general content, producing an answer that's technically "grounded" in *something* retrieved but not actually well-matched to the specific question — reflection catches this before it propagates into a low-quality final answer.

---

## 3. IRCoT (Interleaving Retrieval with Chain-of-Thought)

**Mechanism:** similar in spirit to ReAct, but frames the interleaving specifically around **chain-of-thought reasoning steps** — each reasoning step in a multi-step chain-of-thought can trigger a fresh retrieval specifically to support *that* reasoning step, rather than treating retrieval as a separate tool-call structure layered on top of reasoning. The core idea — reasoning and retrieval should inform each other step-by-step, not happen in two disconnected phases — is the same underlying principle as ReAct, just with slightly different framing/origin (IRCoT comes from academic multi-hop QA research specifically).

**Why it's worth knowing by name:** if an interviewer asks about multi-hop RAG techniques and you only know "ReAct," naming IRCoT (and knowing it's conceptually a close cousin, specifically framed around chain-of-thought rather than general tool-use) demonstrates broader familiarity with the actual research landscape, not just one popular pattern.

---

## Stopping Criteria — When Does the Loop End?

This is a critical, often-overlooked practical design question: an iterative/agentic retrieval loop needs an explicit rule for when to stop, or it risks running indefinitely (cost/latency) or stopping too early (incomplete answers).

| Stopping mechanism | How it works | Risk if poorly tuned |
|---|---|---|
| **Max hop limit** | Hard cap on the number of retrieval iterations (e.g., stop after 4 hops regardless of state) | Too low: genuinely complex questions get cut off incomplete. Too high: wasted cost/latency on unproductive loops |
| **LLM self-assessment** | The model itself judges, after each hop, whether it now has sufficient information to answer confidently | Depends on the model's judgment being well-calibrated — an overconfident model might stop too early (same risk profile as Day 15's refusal-calibration problem, just inverted: stopping too early here is like under-refusing, generating from insufficient info) |
| **Marginal information gain** | Stop if the most recent retrieval hop didn't surface meaningfully new information compared to what's already been gathered | Requires a way to measure "new information," adding its own complexity (e.g., embedding-similarity comparison between the new observation and everything already gathered) |
| **Groundedness/confidence threshold** | Similar to Day 15's runtime guardrail — stop once a groundedness check on a draft answer clears a confidence threshold | Same threshold-calibration trade-off discussed on Day 15 and Day 14 (semantic caching) — needs empirical validation, not a guessed constant |

**Worked latency/cost example — why hop count matters so much:**
```
Each hop: 1 retrieval call (~50ms) + 1 LLM reasoning call (~500ms
          for a Thought+Action generation) ≈ 550ms per hop

1 hop (simple query, no multi-hop needed): ~550ms
3 hops (moderate multi-hop question):       ~1,650ms
6 hops (complex, poorly-bounded loop):      ~3,300ms

Compare to a single-shot RAG response (Day 8 estimate): ~50-100ms
retrieval + ~800ms generation ≈ 850-900ms total
```
**Why this matters in practice:** even a "moderate" 3-hop agentic retrieval process can take roughly **2x longer** than a standard single-shot RAG pipeline, and a poorly-bounded loop that runs 6+ hops can be **3-4x** the latency — this is the direct, concrete cost of agentic RAG's added capability, and it's exactly why stopping criteria and hop limits are not an optional afterthought but a core design requirement, not a nice-to-have.

---

## Failure Modes Specific to Multi-Hop/Agentic RAG

**1. Error propagation across hops:** if hop 1's retrieval returns wrong or misleading information, hop 2's reasoning is built directly on top of that faulty foundation — the error compounds rather than being independently correctable, unlike static decomposition (Day 11) where each sub-question's retrieval is independent and one bad result doesn't necessarily corrupt the others.

**Worked example of error propagation:**
```
Hop 1 retrieves an OUTDATED document stating "AirPods Max was
released in 2019" (actually incorrect — it was 2020).

Hop 2's query, built on this wrong fact: "What other Apple products
launched alongside the 2019 AirPods Max?" — this query is now
built on a false premise, and any retrieval based on it is
searching for content that doesn't correspond to reality, likely
returning irrelevant or further-confusing results.

Final answer: confidently wrong, built on a compounding chain of
errors rather than one isolated mistake.
```

**2. Infinite or unproductive loops:** without a well-designed stopping criterion, an agent can get "stuck" repeatedly retrieving similar, unhelpful content without making genuine progress toward an answer — burning cost and latency with no corresponding quality benefit.

**3. Overconfident early stopping:** the inverse problem — the model judges (incorrectly) that it has sufficient information after hop 1, when the question genuinely required hop 2 or 3, producing an incomplete or shallow answer that "looks" confident despite missing necessary follow-up information.

> **Why This Matters callout:** If asked "what are the risks of agentic/multi-hop RAG compared to simpler approaches," error propagation is the single most important answer to lead with — it's the mechanism-specific risk that doesn't really exist in the same form for single-shot or static-decomposition RAG, where a bad retrieval in one sub-question doesn't corrupt the reasoning behind a different, independent sub-question.

---

## Static Decomposition vs. Dynamic Multi-Hop — Decision Table

| | Static decomposition (Day 11) | Dynamic multi-hop / agentic (today) |
|---|---|---|
| **Sub-questions known upfront?** | Yes, planned before any retrieval | No, determined step-by-step based on actual results |
| **Handles genuinely sequential dependencies?** | No — cannot pre-plan a sub-question requiring a fact not yet known | Yes — this is exactly its core strength |
| **Latency/cost** | Bounded, predictable (N sub-questions = N retrievals, known upfront) | Variable, can be significantly higher, requires stopping-criteria management |
| **Error propagation risk** | Lower — sub-questions are independent | Higher — later hops build directly on earlier hops' results |
| **Implementation complexity** | Lower | Higher — requires loop management, stopping criteria, reflection logic |
| **When to use** | Multi-part/comparative questions with independently-answerable components | Genuinely sequential, "answer depends on a fact you don't have yet" questions |

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What's the fundamental difference between static query decomposition (Day 11) and dynamic multi-hop retrieval?

<details>
<summary>Show answer</summary>

Static decomposition plans all sub-questions upfront, before any retrieval happens — hop 2's sub-question doesn't depend on hop 1's actual result. Dynamic multi-hop retrieval determines each step based on what the previous step actually returned — the second retrieval query is literally constructed using information learned from the first retrieval's result, which static decomposition cannot do since it commits to all sub-questions before seeing any results.
</details>

---

**Q2 (Easy — conceptual).** Give an example of a question that requires dynamic multi-hop retrieval rather than static decomposition, and explain why.

<details>
<summary>Show answer</summary>

"Who is the CEO of the company that acquired the company that made the original iPod's click wheel?" This requires first finding which company made the click wheel, then finding which company acquired *that* company, then finding *that* acquirer's CEO — each step's query depends on a fact that doesn't exist until the previous step is resolved, so there's no way to write all the sub-questions upfront before any retrieval happens.
</details>

---

**Q3 (Medium — conceptual).** Describe the ReAct pattern's three repeating elements and what each one does.

<details>
<summary>Show answer</summary>

Thought: explicit reasoning about the current state and what should happen next. Action: invoking a tool (here, retrieval) with specific parameters, based on that reasoning. Observation: the actual result returned by the action. These three repeat in a loop — each new Thought is informed by the most recent Observation — until the model determines it has sufficient information to produce a final answer.
</details>

---

**Q4 (Medium — conceptual).** What does Self-RAG-style reflection add on top of a basic ReAct loop?

<details>
<summary>Show answer</summary>

Basic ReAct loops typically follow a fairly fixed "keep retrieving until you have an answer" pattern. Self-RAG-style reflection adds an explicit quality gate at each step — the model critiques whether the just-retrieved content is actually sufficient and relevant, and can decide to retrieve again with a different/more targeted query if not, rather than proceeding with a low-quality retrieval result just because the loop's basic structure says to move forward. It's a more deliberate, quality-aware version of iterative retrieval, similar in spirit to a runtime groundedness check but applied during the retrieval process itself.
</details>

---

**Q5 (Medium — calculation).** Each hop in an agentic RAG pipeline costs ~600ms (retrieval + reasoning). Compare total latency for a 2-hop question vs. a poorly-bounded loop that runs 7 hops, and explain the practical implication.

<details>
<summary>Show answer</summary>

```
2 hops: 2 × 600ms = 1,200ms
7 hops: 7 × 600ms = 4,200ms
```
The 7-hop case is 3.5x slower than the 2-hop case. This is the concrete cost of not having a well-designed stopping criterion — an unproductive or poorly-bounded loop doesn't just risk a bad answer, it directly and substantially inflates latency, which is why hop limits and stopping criteria are core design requirements for agentic RAG, not optional refinements.
</details>

---

**Q6 (Hard — conceptual, failure mode).** Explain error propagation in multi-hop retrieval with a concrete mechanism, and explain why this risk is largely absent in static decomposition.

<details>
<summary>Show answer</summary>

In dynamic multi-hop retrieval, each subsequent hop's query is constructed using information learned from earlier hops — if an early hop retrieves incorrect or outdated information, later hops' queries get built on that faulty premise, and their retrieval results (even if individually accurate relative to the flawed query) compound the original error rather than correcting it, since nothing in the pipeline independently re-validates the earlier hop's fact. In static decomposition, sub-questions are planned independently upfront and don't depend on each other's retrieval results — a bad retrieval on one sub-question doesn't feed into or corrupt a different sub-question's retrieval, since they're not sequentially dependent.
</details>

---

**Q7 (Hard — system design synthesis).** Design an agentic RAG system for answering complex, potentially multi-hop customer questions, including your stopping criterion choice and how you'd mitigate error propagation risk. Justify your choices.

<details>
<summary>Show answer</summary>

I'd start with a lightweight classification step (similar to Day 11's decomposition-triggering logic) to detect whether a query likely needs multi-hop reasoning at all — routing genuinely simple queries to standard single-shot RAG to avoid paying agentic overhead unnecessarily. For queries routed to the agentic path, I'd implement a ReAct-style loop with a **combined stopping criterion**: a hard max-hop limit (e.g., 4 hops, as a safety net against unproductive loops) combined with LLM self-assessment after each hop (so the loop can stop earlier than the max when genuinely sufficient information is gathered) — relying on either mechanism alone has known failure modes (self-assessment alone risks overconfident early stopping or unbounded loops if never confident; a fixed hop count alone wastes cost on questions answerable in fewer hops). To mitigate error propagation specifically, I'd add a lightweight Self-RAG-style reflection check after each hop — specifically evaluating whether the retrieved observation is consistent with the query's premise and sufficiently specific, rather than blindly building the next hop's query on an unvalidated result — catching potentially faulty foundational facts before they propagate into subsequent hops' queries, rather than only catching problems at the very end.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Reaching for dynamic multi-hop/agentic RAG for questions that static decomposition (Day 11) already handles perfectly well and more cheaply — genuinely independent sub-questions don't need an iterative loop.
- ❌ Building an agentic retrieval loop with no explicit stopping criterion, risking runaway cost/latency or premature incomplete answers.
- ❌ Assuming later hops in a multi-hop chain are independently verifiable — error propagation means an early wrong fact can silently corrupt everything built on top of it.
- ❌ Relying on LLM self-assessment alone as a stopping criterion without a hard max-hop safety net.
- ❌ Not accounting for the real, often multiplicative latency cost of multi-hop pipelines when discussing agentic RAG as if it were a strictly-better upgrade with no trade-offs.
- ❌ Confusing ReAct and IRCoT as fundamentally different techniques rather than close cousins sharing the same core "interleave reasoning and retrieval" principle.

---

# 📌 Cheat Sheet (Day 16)

**Static (Day 11) vs. dynamic (today):** static plans sub-questions upfront, cannot handle sequential dependencies; dynamic multi-hop determines each step from the actual previous result, handling genuinely sequential "need fact A before I can even formulate the query for fact B" questions.

**ReAct:** Thought (reason) → Action (retrieve) → Observation (result), looped until sufficient information is gathered.

**Self-RAG/reflection:** adds an explicit quality gate — critique retrieved content, retry with a better query if insufficient, rather than blindly proceeding.

**IRCoT:** conceptually a close cousin of ReAct, framed specifically around interleaving retrieval with chain-of-thought reasoning steps.

**Stopping criteria:** max-hop limit (safety net) + LLM self-assessment (efficiency) + optionally groundedness-threshold gating — combine, don't rely on just one.

**Biggest unique risk:** error propagation — a wrong early-hop fact silently corrupts everything built on top of it, a failure mode largely absent in static decomposition's independent sub-questions.

**Golden interview line:** *"Static decomposition plans upfront; agentic multi-hop retrieval plans as it goes — which is exactly what you need for genuinely sequential questions, but it introduces error propagation and unbounded-cost risks that a fixed-plan approach simply doesn't have, so it should be reserved for queries that actually need it, not applied by default."*

---

*End of Day 16. Next up — Day 17: Failure Modes Catalog (hallucination, over-reliance on parametric knowledge, "I don't know" calibration).*
