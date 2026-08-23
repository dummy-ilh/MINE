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

**Example — 7B-class model** ($L=32$, $d_{model}=4096$, fp16, $n=4096$, $b=1$):

$2 \times 32 \times 4096 \times 4096 \times 1 \times 2 = 2{,}147{,}483{,}648 \text{ bytes} \approx \mathbf{2\ GiB}$

That's **512 KB of cache per token** — a useful number to keep in your head. Serve 16 concurrent users at this context length and the cache alone needs **32 GiB** — often more than the model weights themselves. This is the **"memory wall"** of LLM serving: caching fixes the *compute* problem from Chapter 1 but creates a new *memory* problem, which Chapter 3's production techniques exist to solve.

**Second example — the GQA saving:** if a model uses grouped-query attention with 4 query heads sharing 1 K/V head instead of 4 separate K/V heads, the K/V portion of the cache shrinks by 4x — a 2 GiB cache (from above) drops to roughly 512 MiB, with no change to $n$ or $L$.

### 2.5 Q&A — quick check

1. **Q: If you double the context length, what happens to cache size?**
   A: It exactly doubles — $n$ is a plain linear multiplier in the formula.

2. **Q: Why does batch size multiply cache cost instead of sharing it?**
   A: Each sequence in a batch has its own independent tokens and K/V values — nothing to share across different sequences.

3. **Q: How does switching from fp16 to int8 affect max context length for a fixed memory budget?**
   A: Roughly doubles it — half the bytes per value means half the cache size for the same content.

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
