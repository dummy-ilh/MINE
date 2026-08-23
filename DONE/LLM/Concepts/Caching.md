## Phase 1 — Intuition & the Core Problem

### The redundant computation problem

Here's the thing about how LLMs generate text: they produce **one token at a time**, and each new token needs to "look back" at every token that came before it (that's what self-attention does — every token computes a weighted sum over all prior tokens' representations).

The naive way to do this: at every single generation step, recompute attention over the *entire* sequence so far, from token 1 all the way to the current one. If you're generating token 500, you recompute the Key and Value projections for tokens 1 through 499 *again*, even though those tokens haven't changed and their K/V vectors are mathematically identical to what they were in the previous step.

**Analogy:** imagine you're proofreading a growing document, and every time you add one new sentence, you re-read and re-annotate the *entire document from page 1* before you're allowed to write the next sentence. That's insane — you already annotated pages 1 through 50 last time, nothing on those pages changed, so just keep your old notes and only annotate the new sentence. That's exactly what a KV cache does: it keeps the "notes" (Key and Value vectors) from prior tokens so you never redo work you've already done.

### What specifically gets cached

In self-attention, each token produces three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.
- Q is used *only* by the current token, to ask "what am I looking for?"
- K and V represent *every* token in the sequence and are used by *all future tokens* that attend back to this one.

Since K and V for a given token never change once that token is processed (they only depend on that token's own embedding and the layer weights, not on what comes after), they're the perfect candidates to store and reuse. Q, by contrast, is only needed transiently for the current step, so it's never cached.Gray boxes above = "already have K,V cached, just reuse them." The coral box = the only new computation this step. That's the entire efficiency win of KV caching in one picture.

### When it helps a lot vs. barely matters

**KV caching matters most when:**
- Generating long outputs autoregressively (chatbots, code generation, long-form writing) — the savings compound with every token
- Sequences get long — the quadratic-ish waste of full recomputation grows fast with context length
- You're serving at scale with many concurrent users — wasted recomputation directly burns GPU cycles you're paying for
- Latency (time-to-next-token) matters, not just total throughput

**KV caching barely matters (or isn't the bottleneck) when:**
- You're doing a single forward pass with no generation (e.g. just scoring/classifying a fixed input) — there's nothing to cache across since there's no autoregressive loop
- Outputs are very short (1-2 tokens) — the setup/bookkeeping overhead can rival the savings
- You're memory-constrained rather than compute-constrained — ironically, the cache itself becomes the *new* bottleneck (more on this in Phase 2, this is the famous "memory wall" of LLM serving)
- Training (not inference) — during training you process the whole sequence in parallel with teacher forcing, so there's no sequential generation loop to cache across in the same way

### The contrast case: what breaks without it

If you strip caching out of a production LLM serving system:
- **Compute cost explodes.** Generating a sequence of length *n* token-by-token without caching costs roughly O(n²) in attention compute (each of the n steps redoes O(step length) work) instead of the O(n) it costs with caching.
- **Latency degrades as the conversation gets longer** — token 1000 takes dramatically longer to generate than token 10, because you're redoing 1000x the work instead of 1x.
- This is precisely why every production inference engine (vLLM, TensorRT-LLM, Hugging Face's `generate()`, llama.cpp, etc.) treats the KV cache as a first-class citizen, not an optimization bolted on later.

---

## Chapter 1 Q&A — Google / Apple / Meta style

*(Answers below — try the questions yourself first.)*

**Google-style (systems/scale framing):**
1. You're serving an LLM to millions of users with long conversation histories. Without KV caching, how does generation latency scale with conversation length, and why?
2. Why can't you cache the Query vector the same way you cache Key and Value?

**Apple-style (on-device/efficiency framing):**
3. You're deploying an LLM on a phone with tight RAM. Why might KV caching, despite saving compute, actually create a *new* constraint you have to design around?
4. For a very short on-device task (e.g. classifying a one-line notification as urgent/not-urgent), would you expect KV caching to give a meaningful speedup? Why or why not?

**Meta-style (research/architecture framing):**
5. Why is KV caching a technique used at *inference* time but not something that changes how the model is *trained*?
6. If K and V for a token never change once computed, why doesn't the same logic apply to intermediate activations elsewhere in the model (e.g. the FFN outputs)?

---

<details>
<summary>Answers (click to expand — or just scroll)</summary>

1. Without caching, latency grows roughly quadratically — each new token requires recomputing attention over the entire growing history, so a 1000-token conversation does ~1000x more redundant work per step than a 10-token one, and each subsequent token gets slower to generate than the last.
2. Q is only needed transiently to answer "what is *this* token looking for right now" — it's consumed immediately in the current step's attention computation and never referenced again by future tokens. K and V, by contrast, are read by *every future token* that attends back to this position, which is exactly why they're worth storing.
3. The KV cache grows linearly with sequence length and must be held entirely in memory for the whole generation — on a memory-constrained device, a long conversation can make the cache itself the binding constraint, even though it's saving you compute.
4. Not much — the win from caching comes from *amortizing* the setup cost across many generation steps. For a single short output, the overhead of maintaining the cache can roughly offset the compute saved.
5. Training processes the entire target sequence in parallel via teacher-forcing (all ground-truth tokens are available upfront), so there's no sequential, one-token-at-a-time loop to amortize work across — caching is specifically a fix for the *sequential generation* pattern that only exists at inference.
6. FFN and other intermediate activations for a token also don't change after that token is processed — and in principle they *could* be cached too, but K/V are singled out because they're the only intermediate values that get *re-read by other tokens* (via attention); FFN outputs for token *i* are never consulted when processing token *i+1*, so there's nothing to reuse there.

</details>

---

## Phase 2 — The Math

### Notation

| Symbol | Meaning |
|---|---|
| $n$ | sequence length (number of tokens processed so far) |
| $d_{model}$ | model's hidden dimension |
| $h$ | number of attention heads |
| $d_k$ | dimension per head ($d_k = d_{model}/h$) |
| $L$ | number of transformer layers |
| $b$ | batch size (concurrent sequences) |
| $Q, K, V$ | Query, Key, Value matrices |

### The attention formula

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Term by term:
- **$QK^T$** — dot product between the current query and every cached key. This is a similarity score: "how relevant is token $j$ to what I'm asking right now?"
- **$\sqrt{d_k}$** — a scaling factor. Without it, dot products in high dimensions get large, which pushes softmax into a near-one-hot regime and kills gradient flow. It's purely a numerical-stability fix, not a conceptual one.
- **$\text{softmax}(\cdot)$** — converts raw scores into a probability distribution (all weights sum to 1). This is literally the "how much attention to pay to each token" step.
- **$V$** — multiplying the weights by $V$ produces a weighted blend of the *cached value vectors*. This is the payload actually being retrieved — Q and K only decide *how much* of each V to use.

This is exactly where Phase 1's picture becomes precise: $K$ and $V$ for tokens 1..n-1 are **the cache**. Only $Q$ for the new token gets computed fresh each step.

### Worked example — one attention step, by hand

Say we've already generated 2 tokens, and their $K$/$V$ vectors are sitting in the cache with $d_k=2$:

$$K_1=[1,0],\quad K_2=[0,1] \qquad V_1=[1,2],\quad V_2=[3,4]$$

Now we generate token 3. We compute **only** its query: $Q_3=[1,1]$.

**Step 1 — dot products with cached keys (no recomputation of $K_1,K_2$):**
$$Q_3\cdot K_1 = (1)(1)+(1)(0) = 1 \qquad Q_3\cdot K_2 = (1)(0)+(1)(1) = 1$$

**Step 2 — scale by $\sqrt{d_k}=\sqrt{2}\approx1.414$:**
$$\text{scaled} = [1/1.414,\ 1/1.414] \approx [0.707,\ 0.707]$$

**Step 3 — softmax** (equal scores → equal weights):
$$\text{softmax}([0.707, 0.707]) = [0.5,\ 0.5]$$

**Step 4 — weighted sum of cached values:**
$$0.5\times[1,2] + 0.5\times[3,4] = [2,\ 3]$$

That's the full output for token 3's attention — and notice: $K_1, K_2, V_1, V_2$ were **read straight from the cache**, never recomputed. Only $Q_3$, the dot products, and the softmax were fresh work. That's the entire computational saving, made concrete.

### The other half of the math: how big does the cache get?

$$
\text{Cache size (bytes)} = 2 \times L \times n \times d_{model} \times b \times \text{bytes}_{dtype}
$$

The leading **2** is for storing both K *and* V. Everything else is just "how many numbers are we storing, and how many bytes per number."

**Worked example** — a 7B-parameter-class model (Llama-2-7B-like config): $L=32$, $d_{model}=4096$, fp16 ($\text{bytes}_{dtype}=2$), context $n=4096$, batch $b=1$:

$$
2 \times 32 \times 4096 \times 4096 \times 1 \times 2
$$

Step by step:
- $2 \times 32 = 64$
- $64 \times 4096\ (n) = 262{,}144$
- $262{,}144 \times 4096\ (d_{model}) = 1{,}073{,}741{,}824$
- $\times 1\ (batch) = 1{,}073{,}741{,}824$
- $\times 2\ (bytes) = 2{,}147{,}483{,}648 \text{ bytes} \approx \mathbf{2\ GiB}$

**One sequence, at full 4k context, needs 2 GiB just for its KV cache** — on top of the model weights themselves (~14 GB in fp16 for a 7B model). Divide that by 4096 tokens and you get **512 KB of cache per token** — a useful rule-of-thumb number to keep in your head.

Now scale it: serve **16 concurrent users** at that same context length, and the cache alone needs $16 \times 2\text{ GiB} = 32\text{ GiB}$ — often *more* memory than the model weights. This is the famous **"memory wall"** of LLM serving, and it's the entire reason Phase 3's production techniques (paged attention, prefix caching, eviction policies) exist: caching solves the *compute* redundancy problem from Phase 1, but in doing so it creates a brand-new *memory* problem that has to be engineered around.

---

## Chapter 2 Q&A — Google / Apple / Meta style

**Google-style (scale/serving framing):**
1. A model has $d_{model}=8192$, $L=80$, fp16 weights. If you double the context length from 2k to 4k tokens for a single request, what happens to the KV cache size, and why exactly that factor?
2. Why does batch size multiply the cache cost linearly rather than being "shared" across requests?

**Apple-style (memory-constrained framing):**
3. If you switch a model's KV cache from fp16 to int8 (1 byte instead of 2), what's the effect on maximum supportable context length for a fixed memory budget?

**Meta-style (architecture/research framing):**
4. Why does grouped-query attention (GQA) — where multiple query heads share one K/V head — reduce KV cache size, and where in the formula above does that saving show up?
5. In the worked attention example, why would the softmax weights *not* be [0.5, 0.5] if $Q_3$ were instead $[2, 0]$?

<details>
<summary>Answers</summary>

1. Cache size scales linearly in $n$, so doubling context doubles the cache — from the formula, $n$ is a direct multiplicative factor with nothing that dampens it, so 2k→4k means exactly 2x the memory.
2. Each request in a batch has its own independent sequence of tokens with its own K/V values — there's no redundancy to share across *different* sequences, so cache memory is $b$ separate copies, not one shared cache.
3. Halving bytes-per-value halves total cache size, which (all else equal) lets you support roughly double the context length within the same memory budget — this is exactly why quantized KV caches are a common production lever.
4. GQA reduces the *effective* number of K/V heads being stored (several query heads reuse one shared K/V head), which shrinks the $d_{model}$-sized K/V footprint per token — in the formula, this effectively reduces the per-token K/V dimension being cached, without touching $n$ or $L$.
5. With $Q_3=[2,0]$, the dot product with $K_1=[1,0]$ becomes $2$ while with $K_2=[0,1]$ it stays $0$ — an unequal, larger gap between scores, which after scaling and softmax produces a skewed distribution favoring $K_1$ (softmax amplifies larger gaps, it doesn't preserve them linearly), so the weights would no longer be equal.

</details>

---

## Phase 3 — Practical, Diagnostics & Q&A

### Minimal KV cache, verified

I implemented and ran both versions (recompute-every-step vs. cached) end to end — outputs matched to numerical precision, and the cache gave a real speedup at scale:

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
Even this toy, single-head, unoptimized implementation shows a real speedup at 800 tokens — and the gap widens as sequence length grows, exactly matching the O(n²) vs O(n) difference from Phase 2.

### Production systems built on this idea

The naive cache above (a growing Python list/tensor) doesn't survive contact with real serving workloads. Three problems show up immediately at scale, and each has a named fix:

**1. Memory fragmentation → PagedAttention (vLLM)**
Naively, each sequence pre-allocates a contiguous memory block sized for the *maximum* possible sequence length — wasteful, since most sequences are shorter. PagedAttention borrows the OS idea of paging: the KV cache is split into fixed-size blocks, allocated on demand, non-contiguous in memory but tracked via a block table. This eliminates fragmentation and lets memory be shared/reused far more efficiently.

**2. Redundant cache across requests → Prefix caching**
If 1,000 requests all start with the same system prompt, naive serving recomputes and stores identical K/V for that prefix 1,000 times. Prefix caching (vLLM's `enable_prefix_caching`, and the technique behind Anthropic's/OpenAI's API-level "prompt caching") hashes prefixes and shares the cache across requests that share them — pure win when many requests share a long common prefix (system prompts, few-shot examples, long documents queried repeatedly).

**3. Which sequences to keep in memory → RadixAttention (SGLang)**
Generalizes prefix caching into a radix tree over *all* active and recently-completed sequences, so partial overlaps (not just exact-prefix matches) can still share cache, and eviction can be done intelligently (LRU-style) when memory is tight.

**A different layer entirely — Semantic/response caching**
This is *not* KV caching. It sits at the application layer: cache the entire model *response* for a query, and on a new query, check if something semantically similar was already answered (via embedding similarity) and return the cached response directly, skipping inference altogether. Complementary to KV caching, not a replacement — KV caching speeds up computation *within* a generation; semantic caching avoids generation *entirely* for repeat-ish queries.

### Diagnostics — concept-specific failure modes

| Symptom | Why it happens (specific to caching) | Fix |
|---|---|---|
| Latency degrades sharply as conversations get longer, even with caching on | Cache is correctly avoiding recomputation, but the cache itself has grown large enough to become memory/bandwidth-bound — you're now paying for *reading* a huge cache, not recomputing it | Use quantized KV cache (int8), or attention variants that shrink cache size (GQA/MQA), or truncate/summarize old context |
| OOM errors under concurrent load that didn't happen with fewer users | Cache memory scales linearly with both context length *and* batch size — many concurrent long conversations exceed the memory budget the model weights alone would need | PagedAttention-style dynamic allocation, request queuing/admission control, or reducing max concurrent context |
| Two requests with an identical system prompt show no latency benefit from each other | No prefix caching enabled — each request independently computes and discards the shared prefix's K/V | Enable prefix/prompt caching at the serving layer |
| Response looks correct once, but garbled after multi-turn edits to earlier messages | Stale cache — an earlier cache entry was reused, but the underlying prompt/context changed upstream of it, so cached K/V no longer corresponds to the current prompt | Ensure cache keys are derived from exact token-prefix hashes, invalidate on any upstream edit, never cache-key on approximate/semantic similarity for KV caches (that's fine for semantic caching, not for KV) |
| Batching many requests together makes per-request latency *worse*, not better | Ragged/uneven cache lengths across a batch force padding to the longest sequence, wasting compute and memory on padding tokens | Continuous batching (iteration-level scheduling) instead of static batching — vLLM/TGI both do this |

---

## Chapter 3 Q&A — Google / Apple / Meta style

**Google-style (infra/scale framing):**
1. You're running a multi-tenant LLM API where many customers share the same long system prompt. What single serving-layer optimization gives you the biggest win here, and why?
2. Why does naive static batching hurt latency when sequences in a batch have very different lengths?

**Apple-style (on-device/resource-constrained framing):**
3. On a memory-constrained device running a long multi-turn conversation, you notice the app slows down and eventually crashes with an out-of-memory error, even though the model itself loaded fine. What's the most likely cause, and what's one mitigation?

**Meta-style (systems/research framing):**
4. Explain the difference between prefix caching and semantic caching. Could a system use both simultaneously? What would each be responsible for?
5. Why is a hash of the exact token prefix used as the cache key for KV/prefix caching, rather than a semantic embedding similarity check?

<details>
<summary>Answers</summary>

1. Prefix caching — since all requests share a long identical prefix, computing and storing that prefix's K/V once and reusing it across every request eliminates the single largest source of redundant compute in this workload.
2. Padding shorter sequences up to the longest one in the batch wastes compute and memory on padding tokens that carry no information, and the whole batch is gated by the slowest (longest) sequence, so short requests wait unnecessarily.
3. The KV cache grows linearly with conversation length and is held entirely in memory; over a long conversation it can eventually exceed the device's available RAM even though the (fixed-size) model weights were never the problem — mitigating it means quantizing the cache, capping/summarizing context, or evicting older turns.
4. Prefix caching operates on exact token-level matches within the model's internal K/V state, avoiding recomputation; semantic caching operates at the application layer on entire query/response pairs, avoiding inference altogether for similar-meaning queries. Yes, they can coexist: semantic caching would catch/shortcut repeat-ish questions before inference even starts, and prefix caching would speed up whatever inference calls do go through.
5. KV cache correctness is exact — even a tiny prompt change alters every downstream token's true attention output, so approximate/semantic matching would silently serve *wrong* cached values; exact hashing guarantees the cached K/V is byte-identical to what true computation would have produced. Semantic similarity is fine for whole-response caching (Q4) precisely because there correctness tolerance is looser — a "close enough" answer is often acceptable at that layer, but never inside the model's own attention computation.

</details>

---

That closes out the curriculum — intuition, math, and production practice. Want me to package all three chapters into a single reference doc (markdown or Word) so you have something to revisit later?
