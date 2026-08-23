# LLM Caching — Master Notes

*Theory and practice, from the attention mechanism up to production serving systems.*

---

## 0. Primer

**What is "LLM caching"?** It's an umbrella term for three distinct techniques that all avoid redundant computation, but at different layers of the stack:

| Layer | Name | What it avoids recomputing | Where it lives |
|---|---|---|---|
| Model internals | **KV cache** | Attention Key/Value vectors for tokens already processed | Inside the model, every inference call |
| Request layer | **Prefix / prompt caching** | KV cache for a *shared prefix* across multiple requests (e.g. a system prompt) | Serving engine (vLLM, TGI) or API layer (Anthropic/OpenAI prompt caching) |
| Application layer | **Semantic / response caching** | The entire inference call, for queries that mean roughly the same thing as a prior one | Application code, sits in front of the model entirely |

**Why it matters:** LLMs generate text one token at a time, and naively, each new token re-derives information about every prior token from scratch. That's enormous waste — and at production scale (millions of requests, long conversations, shared prompts) it's the difference between a serving system that's affordable and one that isn't. This is why every real inference engine (vLLM, TensorRT-LLM, llama.cpp, SGLang) treats caching as core infrastructure, not an optional optimization.

**Real-world touchpoint:** Anthropic's API exposes prompt caching directly — if you send the same long system prompt or document repeatedly, you pay less and wait less on cache hits, because the underlying KV state for that prefix is reused server-side rather than recomputed. That feature is a direct, user-facing application of Chapter 3's "prefix caching" concept.

**Roadmap:**
1. **Intuition** — why redundant computation happens and what a KV cache actually stores
2. **Math** — the attention formula, a worked numeric example, and the memory-footprint formula that explains why caching creates its own problem
3. **Practice** — verified code, production systems (PagedAttention, prefix caching, RadixAttention, semantic caching), and diagnostics

---

## 1. Intuition & the Core Problem

### 1.1 The redundant computation problem

LLMs generate text **one token at a time**, and every new token needs to "look back" at every token that came before it — that's what self-attention does.

Naively, at every generation step you'd recompute attention over the *entire* sequence so far. Generating token 500 would recompute the Key/Value projections for tokens 1–499 *again*, even though those tokens haven't changed and their K/V vectors are identical to last step's.

**Analogy:** you're annotating a growing document, and every time you add one sentence, you're forced to re-read and re-annotate the *entire document from page 1* before writing the next sentence. Insane — you already annotated pages 1–50 last time; keep those notes and only annotate the new sentence. That's exactly what a KV cache does.

### 1.2 What specifically gets cached

Each token produces three vectors: **Query (Q)**, **Key (K)**, **Value (V)**.
- **Q** is used only by the current token, to ask "what am I looking for?" — needed transiently, never cached.
- **K** and **V** represent a token and are read by *every future token* that attends back to it — once computed, they never change, so they're cached.

### 1.3 Example: where this shows up in practice

- **Chatbot with a long conversation history:** every new user turn requires attending back over the whole conversation. Without caching, turn 50 would redo the attention work for turns 1–49 from scratch — with caching, only the new turn's tokens get processed.
- **Coding assistant reading a large file:** the file's tokens get cached once; every subsequent question about that file reuses the cached K/V instead of re-reading the file "from scratch" internally.

### 1.4 When it helps a lot vs. barely matters

**Helps a lot:** long autoregressive generation, long/growing sequences, high-concurrency serving, when latency (time-to-next-token) matters.

**Barely matters:** a single forward pass with no generation loop (e.g. classification), very short outputs (1–2 tokens, setup overhead can rival the savings), and training (teacher-forcing processes the whole sequence in parallel — no sequential loop to cache across).

### 1.5 The contrast case: what breaks without it

- Compute cost for generating a sequence of length *n* goes from roughly **O(n) with caching to O(n²) without it**.
- Latency degrades as a conversation grows — token 1000 takes dramatically longer than token 10.
- This is why every production inference engine treats the KV cache as first-class infrastructure.

### 1.6 Q&A — quick check

1. **Q: Why can K and V be cached but not Q?**
   A: Q is only needed for the current token's own attention step and is never reused. K and V are read by every future token, so they're worth storing.

2. **Q: What happens to latency as a conversation grows, without caching?**
   A: It gets worse and worse — each new token redoes more and more redundant work, roughly squaring the total cost.

3. **Q: Does caching help during training?**
   A: No — training processes the whole sequence in parallel, so there's no step-by-step generation loop to amortize work across.

---

## 2. The Math

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| $n$ | sequence length processed so far |
| $d_{model}$ | model hidden dimension |
| $h$ | number of attention heads |
| $d_k$ | dimension per head ($d_{model}/h$) |
| $L$ | number of transformer layers |
| $b$ | batch size |

### 2.2 The attention formula

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $QK^T$ — similarity score between the current query and every cached key
- $\sqrt{d_k}$ — numerical-stability scaling factor
- softmax — turns scores into a probability distribution over tokens to attend to
- multiplying by $V$ — produces a weighted blend of cached value vectors — the actual retrieved content

**The key point:** $K$ and $V$ for tokens $1..n-1$ *are* the cache. Only $Q$ for the new token is computed fresh each step.

### 2.3 Worked example — one attention step by hand

Cache already holds (with $d_k=2$): $K_1=[1,0]$, $K_2=[0,1]$, $V_1=[1,2]$, $V_2=[3,4]$.

Generating token 3, compute only its query: $Q_3=[1,1]$.

1. Dot products: $Q_3\cdot K_1 = 1$, $Q_3\cdot K_2 = 1$
2. Scale by $\sqrt{2}\approx1.414$: $[0.707,\ 0.707]$
3. Softmax (equal → equal weights): $[0.5,\ 0.5]$
4. Weighted sum: $0.5[1,2] + 0.5[3,4] = [2,\ 3]$

$K_1,K_2,V_1,V_2$ were read straight from the cache — never recomputed. Only $Q_3$ and the four arithmetic steps were fresh work.

### 2.4 Cache size — the formula and a worked example

$$
\text{Cache size (bytes)} = 2 \times L \times n \times d_{model} \times b \times \text{bytes}_{dtype}
$$

Think of this as answering one question: **"how many numbers am I storing, and how many bytes does each number cost?"** Every factor is just a count of *something*, multiplied together:

| Term | What it counts | Why it's there |
|---|---|---|
| **2** | K *and* V | You're not storing one set of vectors per token, you're storing two — a Key vector and a Value vector. This is fixed, not tunable. |
| **$L$** | number of transformer layers | Every layer has its *own* attention mechanism with its *own* K/V vectors. A 32-layer model caches 32 independent copies of K/V per token — not one shared copy. |
| **$n$** | sequence length so far | Every token processed adds one more K vector and one more V vector to the cache, at every layer. This is the term that keeps *growing* as generation continues — the reason the cache isn't a fixed cost. |
| **$d_{model}$** | hidden dimension | Each K or V vector for a single token, at a single layer, has this many numbers in it. Bigger models → bigger vectors → more bytes per token. |
| **$b$** | batch size | Every concurrent sequence needs its *own* cache — there's nothing to share between two unrelated conversations. |
| **$\text{bytes}_{dtype}$** | bytes per number | fp32 = 4 bytes, fp16/bf16 = 2 bytes, int8 = 1 byte. This is the one factor you can shrink for free (with some quality tradeoff) — it directly scales the total. |

Multiply them all together and you get total bytes. **Nothing here is exotic — it's literally `count of numbers × size of each number`,** same as computing the size of any array.

### Worked example, walked through slowly

7B-class model: $L=32$ layers, $d_{model}=4096$, fp16 (2 bytes/number), context $n=4096$ tokens, $b=1$ sequence.

**Step 1 — how many K/V vectors total?**
$L \times n \times b = 32 \times 4096 \times 1 = 131{,}072$ vectors of K, and the same number of V vectors. (One K vector and one V vector for every token, at every layer.)

**Step 2 — how many numbers per vector?**
$d_{model} = 4096$ numbers per vector.

**Step 3 — total count of numbers, K and V combined:**
$2 \times 131{,}072 \times 4096 = 1{,}073{,}741{,}824$ numbers.

**Step 4 — convert to bytes:**
$1{,}073{,}741{,}824 \times 2 \text{ bytes} = 2{,}147{,}483{,}648 \text{ bytes} \approx \mathbf{2\ GiB}$

**The per-token shortcut:** dividing that 2 GiB by 4096 tokens gives **512 KB of cache per token** for this model — meaning every single token you generate or feed in permanently costs half a megabyte of memory, for as long as it stays in context. That's the number worth memorizing, because it lets you estimate cache cost for *any* context length instantly: 1000 tokens ≈ 500 MB, 8000 tokens ≈ 4 GB, and so on.

**Why this matters at scale:** one user at 4k context costs 2 GiB. Sixteen concurrent users at that same context cost **32 GiB** — likely *more* than the ~14 GB the model's own weights take up in fp16. That's the "memory wall": the model itself is a fixed, one-time cost, but the cache is a *per-user, per-token* cost that keeps growing — and it can dwarf the model.

### The GQA saving, explained

Normally, every attention head has its own K and V — if a model has $h=32$ heads, that's 32 separate K vectors and 32 separate V vectors per token, per layer, all folded into that $d_{model}$ term.

**Grouped-query attention (GQA)** changes this: multiple query heads share *one* K/V head instead of each having their own. If 4 query heads share 1 K/V head, you've cut the number of *distinct* K/V vectors being stored by 4x — even though you still have 4x as many query heads doing the actual attending.

Concretely: the 2 GiB cache above assumed one K/V pair per head. With 4-way GQA, the K/V portion shrinks to **roughly 512 MiB** for the same model, same context, same batch size — $n$ and $L$ are untouched, only the *effective* $d_{model}$ term used for K/V shrinks. This is why GQA is standard in most modern production models (Llama 2 70B, Llama 3, Mistral) — it's a direct, free-ish way to fight the memory wall.

---

### Q&A — quick check

1. **Q: If you double the context length, what happens to cache size?**
   A: It exactly doubles. $n$ is the term that counts "how many tokens have I cached," and every additional token adds a fixed, identical chunk of K/V data — there's no dampening effect, so the relationship is perfectly linear.

2. **Q: Why does batch size multiply cache cost instead of sharing it?**
   A: The cache stores each *sequence's own* tokens' K/V — two unrelated conversations have completely different tokens, so there's nothing in common to reuse. Batch size is a straight multiplier because you're literally maintaining $b$ independent caches side by side.

3. **Q: How does switching from fp16 to int8 affect max context length for a fixed memory budget?**
   A: It roughly doubles the context you can support. You're not changing how many numbers you store (that's still $2 \times L \times n \times d_{model} \times b$) — you're just halving the cost per number, which halves total bytes and therefore lets $n$ grow twice as far before you hit the same memory ceiling.

---

## 3. Practical, Diagnostics & Q&A

### 3.1 Minimal KV cache — verified in code

Implemented and ran both a recompute-every-step version and a cached version; outputs matched to numerical precision, and the cache gave a measured speedup at scale.

```python
def attend(q, K, V):
    scores = (q @ K.T) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V

def generate_no_cache(tokens):
    outputs = []
    for t in range(1, len(tokens) + 1):
        prefix = tokens[:t]
        K = prefix @ W_k          # recomputed every step
        V = prefix @ W_v          # recomputed every step
        q = tokens[t-1:t] @ W_q
        outputs.append(attend(q, K, V))
    return torch.cat(outputs, dim=0)

def generate_with_cache(tokens):
    K_cache, V_cache = None, None
    outputs = []
    for t in range(len(tokens)):
        new_tok = tokens[t:t+1]
        k_new, v_new = new_tok @ W_k, new_tok @ W_v   # only new token
        K_cache = k_new if K_cache is None else torch.cat([K_cache, k_new])
        V_cache = v_new if V_cache is None else torch.cat([V_cache, v_new])
        q = new_tok @ W_q
        outputs.append(attend(q, K_cache, V_cache))
    return torch.cat(outputs, dim=0)
```

**Results:**
```
Max difference between cached and uncached outputs: 4.77e-07   ✓ correctness confirmed
seq_len=800
Without cache: 0.0567s
With cache:    0.0327s
Speedup:       1.73x
```

### 3.2 Production systems

| Problem | System / Technique | Fix |
|---|---|---|
| Memory fragmentation from pre-allocating max-length blocks | **PagedAttention** (vLLM) | Splits cache into fixed-size, non-contiguous blocks, allocated on demand — like OS memory paging |
| Redundant cache across requests sharing a prefix | **Prefix / prompt caching** (vLLM, Anthropic/OpenAI APIs) | Hashes prefixes, shares cached K/V across requests with an identical prefix |
| Partial (non-exact-prefix) overlaps across many sequences | **RadixAttention** (SGLang) | Generalizes prefix caching into a radix tree, allows intelligent eviction |
| Avoiding inference entirely for repeat-ish queries | **Semantic / response caching** (application layer) | Embedding-similarity lookup on whole query/response pairs — not KV caching, a different layer entirely |

**Example — prefix caching in action:** a customer-support bot sends the same 2,000-token system prompt with every request. Without prefix caching, that prompt's K/V is recomputed for every single user message. With it, the first request computes and stores that prefix once; every subsequent request (even from different users) reuses it and only pays for the new user message — this is essentially what Anthropic's and OpenAI's "prompt caching" API features do.

**Example — semantic caching in action:** an FAQ-answering bot gets "what's your refund policy?" and "how do refunds work?" — different wording, same intent. A semantic cache can recognize the similarity via embeddings and return the previously-generated answer without calling the model at all.

### 3.3 Diagnostics

| Symptom | Cause | Fix |
|---|---|---|
| Latency still degrades on long conversations even with caching on | Cache itself has grown large enough to be memory/bandwidth-bound | Quantized cache (int8), GQA/MQA, truncate/summarize old context |
| OOM under concurrent load | Cache scales linearly with context length *and* batch size | PagedAttention-style allocation, admission control, lower max concurrent context |
| Shared system prompt gives no cross-request speedup | Prefix caching not enabled | Enable prefix/prompt caching at the serving layer |
| Garbled output after editing earlier messages in a conversation | Stale cache — cached K/V no longer matches the (edited) prompt | Cache keys must be exact token-prefix hashes; invalidate on any upstream edit |
| Batching makes latency *worse* | Padding to the longest sequence in a static batch wastes compute | Continuous batching (iteration-level scheduling) |

### 3.4 Q&A — quick check

1. **Q: What's the single biggest win for many requests sharing one system prompt?**
   A: Prefix caching — compute and store that prefix's K/V once, reuse across every request.

2. **Q: Why does an app on a memory-constrained device slow down and crash during a long conversation, even though the model loaded fine?**
   A: The KV cache grows with conversation length and can outgrow available memory even though the fixed-size model weights never change.

3. **Q: What's the difference between prefix caching and semantic caching?**
   A: Prefix caching reuses exact-match model internals (K/V); semantic caching skips inference entirely for queries that mean roughly the same thing as a prior one. They operate at different layers and can be used together.

## Section 4 — Four techniques you'll hear about constantly, explained simply

These all attack the "memory wall" problem from Section 2 — the cache getting too big — but from four different angles. Here's each one in plain language with a concrete example.

---

### 4.1 Quantized cache (int8)

**The idea, in one sentence:** store each number in the cache using fewer bits, so the same cache takes less memory.

Normally each number in the K/V cache is stored as fp16 (16 bits = 2 bytes). Quantizing to int8 stores it in 8 bits = 1 byte instead — literally half the space, for the same numbers.

**Analogy:** imagine you're keeping notes and normally you write each number out to 4 decimal places ("3.1416"). Quantizing is like rounding to 1 decimal place ("3.1") instead — you lose a little precision, but the note takes way less space on the page. Do that for millions of numbers and the savings add up fast.

**Concrete example:** from Section 2, a 7B model at 4k context needs 2 GiB of cache in fp16. Switch to int8, and it drops to **1 GiB** — same content, same context length, half the memory. That "extra" GiB you saved can now go toward serving more users or supporting longer context.

**The catch:** you lose some numerical precision, which *can* slightly hurt output quality — it's a genuine tradeoff, not a free lunch, which is why it's not always turned on by default.

---

### 4.2 GQA / MQA (Grouped-Query / Multi-Query Attention)

**The idea, in one sentence:** instead of every attention "head" keeping its own separate K and V, make several heads *share* one K/V.

**Analogy:** imagine a newsroom with 32 reporters (that's your attention heads), each covering the same story from a different angle (that's what makes them useful — different perspectives). Normally, each reporter keeps their *own* private notebook of facts (K/V). GQA is like saying: "you 4 reporters share one shared notebook of facts, and each of you just brings your own *questions* to it." The reporters still ask different questions and still get different insights out — but you're only maintaining 1 notebook instead of 4.

- **MQA (Multi-Query Attention)** is the extreme version: *all* heads share just **one** K/V.
- **GQA (Grouped-Query Attention)** is the middle ground: heads are split into small groups, each group shares one K/V.

**Concrete example:** from Section 2's GQA example — 4 query heads sharing 1 K/V head shrinks the cache 4x, turning a 2 GiB cache into ~512 MiB, with zero change to context length or number of layers. This is why most modern production models (Llama 3, Mistral) use GQA by default — it's baked into the model architecture itself, not something you turn on later.

---

### 4.3 PagedAttention (vLLM)

**The idea, in one sentence:** stop reserving one giant continuous memory block per conversation "just in case" it gets long — allocate memory in small chunks, only as needed.

**Analogy:** imagine renting a parking garage. The naive approach is: every car that enters gets reserved an entire floor to itself, "in case" it needs to grow into a bus later. Most cars never grow — so you're wasting almost the whole floor per car, and you run out of floors fast even though there's tons of unused space scattered around.

PagedAttention is like switching to individual parking *spots* instead of whole floors: each car gets exactly as many spots as it currently needs, one at a time, and a central directory (the "block table") just keeps track of which spots belong to which car — even if those spots aren't next to each other.

**Concrete example:** without PagedAttention, if you set aside memory assuming every conversation might reach 4096 tokens, but the average conversation is actually only 500 tokens, you're wasting ~87% of that reserved memory per conversation — multiplied across thousands of concurrent users, that's the difference between serving 50 people and serving 400 people on the same GPU. This is the single biggest reason vLLM became the standard serving engine — it's mostly a memory-utilization fix, not a speed fix.

---

### 4.4 RadixAttention (SGLang)

**The idea, in one sentence:** PagedAttention shares memory *within* one conversation efficiently — RadixAttention shares cache *across different requests* that overlap, even partially.

**Analogy:** think of a shared family tree instead of individual family trees. If 1,000 users all start their conversation with the same company system prompt, then branch off into totally different questions, a plain prefix cache only helps if requests share the *exact same starting sequence*. RadixAttention builds a tree structure (like a family tree, or a filing cabinet with shared folders that branch off into sub-folders) where any two requests that share *any* common prefix — even a partial one, like the first 200 tokens of a 2000-token shared document — automatically reuse that shared portion, no matter how the rest of their requests differ.

**Concrete example:** imagine a customer support system where 500 users each start by pasting the *same* 3-page product manual, then ask 500 completely different follow-up questions. A basic prefix cache handles this fine (since the whole manual is an identical prefix). But now imagine 500 users paste *slightly different but overlapping* excerpts of that manual — RadixAttention can still detect and reuse the *overlapping chunks*, where a simple "does this match the wire exactly" prefix cache would miss the shared portion entirely and recompute everything from scratch. It also intelligently decides what to keep in memory vs. evict when things get tight (like an LRU cache), across *all* active requests at once, not just one at a time.

---

### Quick recap — which one solves which problem?

| Technique | Fixes | Cost/tradeoff |
|---|---|---|
| Quantized cache (int8) | Cache too big in memory | Slight precision/quality loss |
| GQA/MQA | Cache too big *by design* | Built into model architecture, can't retrofit onto an already-trained model easily |
| PagedAttention | Wasted memory from over-reserving | Requires a serving engine that supports it (vLLM) |
| RadixAttention | Cache not shared across similar-but-not-identical requests | More complex to implement/maintain than simple prefix matching |


## Section 5 — Advanced Q&A, simplified

This builds directly on Sections 1–4. A few of these questions restate things you've already seen in a stricter, more "interview-ready" form — I'll point out where that overlap is, then add the genuinely new pieces (prefill vs. decode, MLA, eviction policies, attention sinks).

---

### 5.1 Prefill vs. Decode — the two phases of generation

Every LLM request actually has two very different phases, and they bottleneck on completely different hardware resources:

**Prefill (processing your prompt, before the first output token):** the whole prompt is known upfront, so the model processes *all* prompt tokens at once, in parallel, as one big matrix multiplication. This phase is **compute-bound** — you're limited by how fast the GPU can crunch numbers. This is also where the *initial* KV cache gets built.

**Decode (generating each output token, one at a time):** each new token depends on the one before it, so this can't be parallelized the same way — one token in, one token out, over and over. Each step has to pull the *entire* KV cache (weights + all cached K/V) out of memory. This phase is **memory-bandwidth-bound** — you're limited by how fast you can *read* memory, not how fast you can compute.

**Simple analogy:** prefill is like reading an entire book cover-to-cover in one sitting (compute-heavy, but done once). Decode is like writing one new sentence, then having to flip back through every previous page to check consistency, then writing the next sentence, then flipping through again — over and over. The "flipping through pages" is the memory-bandwidth cost, and it's why decode is slow token-by-token even on a fast GPU.

**Why this matters practically:** Time-to-first-token (TTFT) is a prefill problem — optimize compute. Time-per-output-token (TPOT) is a decode problem — optimize memory bandwidth and cache size (which is exactly what Sections 2–4 are about).

---

### 5.2 The memory formula, reconciled

You'll see this formula written two equivalent ways:

$$\text{Memory} = 2 \times B \times S \times L \times n_{kv} \times d_k \times \text{bytes}$$

versus Section 2's version using $d_{model}$. They're the same thing — $n_{kv} \times d_k$ **is** $d_{model}$ split into "number of KV heads" × "size per head." Section 2 used the collapsed form; this version shows *why* GQA/MQA helps: shrinking $n_{kv}$ (fewer KV heads) shrinks total memory directly, without touching anything else.

**Worked example (Llama-3-70B):** $L=80$, $n_{kv}=8$, $d_k=128$, fp16, $S=4096$, $B=1$:

$$2 \times 1 \times 4096 \times 80 \times 8 \times 128 \times 2 \approx 1.34 \text{ GB per user}$$

Notice $n_{kv}=8$ here, not 64 (Llama 3 70B actually has 64 query heads) — that gap *is* GQA in action, already baked into the number.

---

### 5.3 Multi-Head Latent Attention (MLA) — the newest cache-shrinking trick

GQA and MQA shrink the cache by having heads *share* K/V. MLA (used in DeepSeek-V2/V3) takes a different approach: instead of storing full-size K/V vectors at all, it compresses them down into a much smaller "latent" vector first, then reconstructs the full K/V on the fly when needed.

**Simple analogy:** GQA is like several reporters sharing one notebook (Section 4.2). MLA is like *compressing* the notebook itself — instead of writing full sentences, you write a shorthand code that expands back into the full sentence when read. You store the compact shorthand (cheap), and only "decompress" it at the moment you actually need the full detail. The compressed version takes a fraction of the space of a full K/V vector.

**Why it matters:** it gets a bigger memory reduction than GQA typically does, which is part of why DeepSeek models could support long context cheaply.

---

### 5.4 Compression and eviction — three more tools, simplified

**Quantization variants (INT8, INT4, FP8):** same idea as Section 4.1 — fewer bits per stored number — just taken further. INT4 is half the size of INT8, which is half of FP16. The tradeoff scales too: more compression, more risk of quality loss. Techniques like KIVI and QuaRot are specific recipes for doing this compression *without* losing much accuracy (e.g. by only compressing the parts of the cache that tolerate it well).

**Eviction policies (H2O, Scissorhands):** instead of shrinking every number, just **throw away whole tokens** from the cache that don't seem to matter anymore.

*Analogy:* think of it like a whiteboard that's filling up — instead of writing smaller (quantization), you erase old notes that clearly aren't being referenced anymore, and keep the ones people keep pointing back to. These methods track which tokens actually get high attention weight ("heavy hitters") and keep those, discarding the ones that rarely get attended to.

**Attention sinks / StreamingLLM:** a surprising empirical finding — the *very first few tokens* of a sequence get disproportionately high attention weight no matter what they are, almost like a "dumping ground" the model always glances at. StreamingLLM exploits this: keep those first few tokens permanently, plus a sliding window of the most recent tokens, and evict everything in between. This lets a model handle an effectively *infinite* stream of text in *fixed* memory, because the cache size stops growing — it's a fixed-size window plus a fixed handful of "anchor" tokens.

*Analogy:* imagine taking notes during an all-day meeting — you can't keep everything, so you keep your notes from the opening framing (which everything else gets interpreted through) plus your notes from the last 10 minutes (immediately relevant), and let the middle fade.

---

### 5.5 Prefix caching, precisely

This solidifies Section 4.4 (RadixAttention) with the general mechanism: because K/V for a given prefix is **deterministic** (same tokens always produce the same K/V, given the same model), a system can match incoming requests against previously-seen prefixes — either exact string matching, or a radix tree for partial-prefix matching (RadixAttention) — and skip straight to reusing the cached blocks. A cache hit turns an expensive prefill (compute-bound matrix multiplication) into a cheap memory lookup.

---

### 5.6 Semantic caching, precisely

This solidifies the "application layer" caching from Section 0's primer and Section 4's recap table. It sits *in front of* the model entirely, not inside it:

1. Convert the incoming prompt into an embedding vector.
2. Compare it (cosine similarity) against embeddings of previously-answered prompts.
3. If similarity clears a threshold (commonly ~0.95), return the old cached *response* directly — the model is never called at all.

**Key distinction to remember:** prefix caching (5.5) still runs the model, just skips redundant *internal* computation. Semantic caching (5.6) skips the model *entirely* for a "close enough" match — which is why it needs a similarity threshold and tolerance for approximate matches, whereas prefix/KV caching must be exact (a single different token upstream invalidates everything downstream).

---

Want this folded into the master notes file as Section 5, alongside a refresh of Section 4?
