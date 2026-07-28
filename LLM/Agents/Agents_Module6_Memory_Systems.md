# Agents Module 6 — Memory Systems (Master Notes, Expanded)

## 0. The problem this module solves — recap and extend Module 1's core insight

Module 1 established the critical architectural fact: **the LLM itself is stateless between calls** — any apparent "memory" an agent has is an engineering construct, not a property of the model. This module is about **how** that construct is actually built: what different kinds of information an agent needs to persist, over what timescales, and how each kind gets stored, retrieved, and eventually discarded. The taxonomy below (working / episodic / semantic / procedural) is borrowed directly from cognitive psychology's model of human memory — a deliberate and useful analogy, since each human memory type maps onto a genuinely distinct engineering problem for agents.

---

## 1. Working Memory (the context window itself)

### What it is
The current prompt/context window — everything the model can directly attend to in this single forward pass: the task, recent conversation turns, recent tool observations, and anything else explicitly included in this call's input. This is memory in the weakest sense (it's not "stored" anywhere persistent — it's just *the input*), but it's the only memory type the model can use *without* any retrieval step, since it's already sitting directly in context.

### The core constraint — finite context window
The context window has a hard size limit (measured in tokens — directly connects to LLM Basics Module 1's tokenization and Module 6's KV-cache memory-cost discussion, since a longer context means a larger KV cache to hold in memory during generation). As an agent's task runs longer (more Thought-Action-Observation cycles, Module 4), the accumulated transcript can **exceed the context window**, forcing some form of truncation, summarization, or offloading to a different memory type (below) — working memory alone cannot be the *only* memory mechanism for any sufficiently long-running agent task.

### Practical management strategies
- **Sliding window / truncation**: simply drop the oldest turns once the window fills up — cheap, but risks losing genuinely important early context (e.g., the original task statement, if it scrolls out of the window entirely, which is why many implementations pin the original task/system instructions and only truncate the middle history).
- **Running summarization**: periodically compress older parts of the transcript into a shorter summary (itself an LLM call), keeping the gist while freeing up context budget — trades some information loss (summarization is lossy) for the ability to retain a much longer effective history than raw truncation alone would allow.

---

## 2. Episodic Memory

### What it is
Memory of **specific past events/experiences** — "what happened," tied to a particular time/instance. For an agent, this means storing records of past task attempts, past conversations, past tool-call results — retrievable later, typically by relevance to the current situation, not by keeping everything in the active working-memory context at all times.

### How it's typically implemented
The dominant pattern is: **store past episodes as text (or structured records) in an external database, compute a vector embedding for each stored episode (or a searchable index over it), and retrieve the most relevant past episodes for the current situation via similarity search** (embedding the current query/situation, then finding stored episodes with the closest embeddings — this is the same embedding-similarity mechanism underlying retrieval-augmented generation) — retrieved episodes are then **injected into the working-memory context** for the current call, since (per Section 1) that's the only way the frozen model can actually condition its behavior on them.

### Concrete example
An agent that has previously helped a user debug a specific recurring error message: on a new conversation where a similar error appears, episodic memory retrieval surfaces "on [past date], user hit a similar error, and the fix that worked was X" — this specific past episode gets pulled into context, letting the current call benefit from that concrete prior experience without the entire past conversation having stayed in the working-memory window the whole time.

### Key design tension: what counts as "relevant enough" to retrieve
Retrieval is itself a **ranking/similarity problem, not a perfect-recall problem** — a poorly-tuned episodic memory system can retrieve irrelevant past episodes (diluting the useful context with noise) or fail to retrieve a genuinely relevant one (if the embedding similarity between the current situation and the useful past episode happens to be low despite real conceptual relevance) — this is a genuine, unsolved-in-general information retrieval challenge, not a solved mechanical lookup.

---

## 3. Semantic Memory

### What it is
Memory of **general facts and structured knowledge**, decoupled from any specific episode/event it came from — "what is true," not "what happened and when." The distinction from episodic memory (a favorite precise-definition interview question): episodic memory is "on Tuesday, the user told me their favorite color is blue"; semantic memory is just "the user's favorite color is blue" — the fact itself, extracted and stored independent of the specific conversational instance it was learned in.

### How it's typically implemented
Often a **structured knowledge base** (key-value facts, a knowledge graph, or a more loosely structured vector-searchable fact store) — populated either by explicit extraction (an LLM call that reads a conversation/episode and extracts durable facts worth remembering long-term, distinct from the transient conversational flow around them) or by direct structured input (a user explicitly stating a preference that gets parsed and stored as a fact). Retrieval works similarly to episodic memory (embedding/similarity search, or direct key lookup for structured facts), with results injected into working-memory context when relevant to the current task.

### Why separating semantic from episodic matters practically
If every fact remained buried inside full episodic records (entire past conversation transcripts), retrieving "what is the user's favorite color" would require finding and re-parsing the specific past conversation where it was mentioned — inefficient and fragile. Extracting and storing the **distilled fact itself** in semantic memory makes retrieval direct and cheap, and — importantly — makes it possible to **update a fact cleanly** when it changes (e.g., the user's preference changes) without needing to somehow invalidate or reconcile old episodic transcripts, which should remain an accurate historical record even after the fact itself is updated.

---

## 4. Procedural Memory

### What it is
Memory of **learned skills, routines, or "how to do things"** — not a specific fact or event, but a reusable method/procedure. In humans, this is the memory type behind things like "how to ride a bike" — knowledge that manifests as capability rather than as a recallable fact or episode.

### How it's typically implemented for agents
This is the least standardized of the four memory types, but common patterns include: **storing successful tool-use sequences or code snippets as reusable, named procedures** (once an agent solves a particular kind of subproblem successfully, save the sequence of actions/code that worked as a callable "skill" for future similar situations, rather than re-deriving the same solution from scratch every time), or, in some systems, actually **fine-tuning the underlying model** (LLM Basics Module 4) on successful trajectories to bake a capability more directly into the weights rather than keeping it as an explicit, retrievable, external record. The Voyager paper (an agent operating in Minecraft) is a commonly cited concrete example: the agent builds up an explicit, growing **skill library** of verified, reusable code functions (e.g., "craft a wooden pickaxe") that it can directly call again later, rather than re-planning that entire subtask's reasoning from scratch every time it recurs.

### Why this is architecturally distinct from episodic/semantic memory
Episodic and semantic memory are both **retrieved as text and injected into context** for the frozen model to read and reason over. Procedural memory, when implemented as a callable skill/tool (rather than as weight fine-tuning), is closer to **Module 2's tool-use mechanism** than to a memory-retrieval mechanism — the "memory" isn't reasoned-over text, it's a directly-executable capability the agent can invoke, sidestepping the need to re-derive the procedure via reasoning at all. This is a genuinely useful distinction to draw explicitly if asked to compare all four memory types: procedural memory is the one type that can bypass the "retrieve text into context, then reason over it" pattern entirely.

---

## 5. Memory write, read, and consolidation strategies

### Write (deciding what's worth remembering at all)
Not everything that happens should become a permanent memory — a naive "store everything" approach leads to enormous, noisy memory stores that hurt retrieval quality (Section 2's relevance-ranking problem gets worse as the haystack grows) and cost (embedding and storing everything isn't free). Common strategies: an explicit **importance-scoring step** (often another LLM call, asked to rate how significant/memorable a given piece of information is) that gates what actually gets written to persistent memory, versus what's allowed to simply fall out of working memory and be forgotten.

### Read (retrieval at inference time)
Covered above per memory type — generally embedding-similarity search (episodic, semantic) or direct capability invocation (procedural), with retrieved results injected into the working-memory context for the current call.

### Consolidation (a less commonly implemented but conceptually important idea)
Analogous to human sleep-based memory consolidation: periodically **process and compress accumulated episodic memories into more durable semantic facts or updated procedural skills**, rather than leaving episodic memory as an ever-growing, unprocessed log. E.g., after many episodes of a user mentioning food preferences in passing, a consolidation pass might extract and write a single durable semantic fact ("user is vegetarian") rather than relying on retrieval to re-surface scattered individual episodic mentions every time. This directly reduces the burden on the retrieval step (Section 2's relevance-ranking challenge) by proactively distilling the most durably useful information out of the noisier raw episodic stream.

### Forgetting mechanisms
Memory stores that only ever grow eventually face the same relevance-dilution problem noted in Section 2 — some systems implement explicit forgetting/decay (down-weighting or removing memories that haven't been retrieved/reinforced in a long time, or that have been explicitly superseded by updated information, e.g., a changed semantic fact) — a direct practical tradeoff between "never lose potentially useful information" and "keep the retrievable memory store small and high-signal enough to actually be useful."

---

## 6. Side-by-side summary table (memorize this cold)

| | Working | Episodic | Semantic | Procedural |
|---|---|---|---|---|
| What it stores | Current context window contents | Specific past events/interactions | Distilled general facts | Reusable skills/routines |
| Timescale | This call only | Persistent, tied to specific instances | Persistent, instance-independent | Persistent, capability-level |
| Typical storage mechanism | N/A — it IS the input | Vector DB + similarity search over event records | Structured KB or vector store of extracted facts | Named skill library (callable code/tools) or weight fine-tuning |
| How it reaches the model | Already present, no retrieval needed | Retrieved and injected into context | Retrieved and injected into context | Directly invoked as a capability (bypasses context-reasoning) |
| Human-memory analogy | Short-term/working memory | "What happened and when" | "What is true" | "How to do it" |

---

## 7. Quick-fire Q&A (self-test)

**Q: Why can't working memory alone serve as an agent's only memory mechanism for long-running tasks?**
A: The context window has a hard token limit; as a task accumulates more Thought-Action-Observation cycles, the transcript can exceed that limit, forcing truncation, summarization, or offloading older information to a persistent memory type outside the context window.

**Q: Give the precise distinction between episodic and semantic memory, with a concrete example of each for the same underlying information.**
A: Episodic memory ties information to a specific past instance/event ("on Tuesday, the user told me their favorite color is blue"); semantic memory stores the distilled fact itself, decoupled from when/how it was learned ("the user's favorite color is blue").

**Q: Why is separating semantic memory from episodic memory practically useful, beyond just conceptual tidiness?**
A: It makes fact retrieval direct and cheap (no need to re-locate and re-parse a full past conversation to find one fact), and makes updating a fact clean when it changes, without needing to invalidate or reconcile old episodic records, which should remain an accurate historical record even after the underlying fact is updated.

**Q: How is procedural memory architecturally different from episodic and semantic memory in how it reaches the model?**
A: Episodic and semantic memory are retrieved as text and injected into the context window for the model to reason over. Procedural memory, when implemented as a callable skill (rather than via weight fine-tuning), is directly invoked as an executable capability — closer to Module 2's tool-use mechanism than to a retrieve-and-reason memory pattern, bypassing the need to re-derive the procedure through reasoning at all.

**Q: What problem does memory consolidation address, and give a concrete example of what it produces.**
A: It addresses the growing noise/dilution problem in an ever-accumulating episodic memory store by periodically distilling accumulated episodes into more durable, directly-retrievable semantic facts. Example: many scattered episodic mentions of food preferences get consolidated into a single semantic fact like "user is vegetarian," rather than relying on retrieval to re-surface the individual scattered mentions each time.

**Q: What's the core tradeoff behind implementing an explicit forgetting/decay mechanism in an agent's memory store?**
A: Balancing "never lose potentially useful information" against keeping the retrievable memory store small and high-signal enough that retrieval quality doesn't degrade as the store grows — an ever-growing, never-pruned memory store makes relevant-memory retrieval harder and noisier over time.

---
*End of Agents Module 6 (expanded). Next: Module 7 — Multi-Agent Systems & Orchestration (architectures, communication protocols, failure modes).*
