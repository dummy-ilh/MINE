# Module 9 — Interview-Style Synthesis (Master Notes, Maximum Depth)

> **Legend for this enhanced version:**
> - 📌 **Added Explanation** — expanded intuition, background, or clarification
> - 🧮 **Numerical Example** — a worked, step-by-step calculation
> - ❓ **Interview Q&A** — new Apple/Google-style ML interview questions with model answers
>
> All of your original text is preserved exactly as written, in its original order. Additions are inserted as clearly-tagged blocks directly below the relevant original section.

---

## 0. How to use this module

This isn't new material — it's **cross-module synthesis**: questions that force you to connect ideas across Modules 1-8, the way a real interviewer actually probes (they rarely ask "define BPE" in isolation; they ask "walk me through what happens when a user sends a prompt," and expect you to touch tokenization → inference → serving in one answer). Each section below groups questions by the *kind* of interview question, not by module, since that's how they'll actually show up.

> 📌 **Added Explanation — why "synthesis" questions are harder than they look**
> In simple terms: a synthesis question isn't testing whether you *remember* facts from each module — it's testing whether those facts live in your head as **one connected mental model** instead of nine separate flashcard decks. The giveaway that a candidate has only memorized isolated facts is that they answer a "walk me through X" question with a single fact from a single module and then stop, instead of naturally continuing the chain ("...and *because* of that, the next thing that happens is..."). The model answers below are written to demonstrate that connective tissue explicitly — notice how almost every sentence ends by explaining *why* it leads to the next step, not just *that* it does.

---

## 1. "Walk me through end-to-end" questions

### Q: A user sends a prompt to a deployed chat LLM. Walk me through everything that happens, technically, from input to output.

**Strong answer structure** (this is the shape to reproduce, not memorize verbatim):
1. **Tokenization** (Module 1): the raw text prompt is tokenized using the model's fixed vocabulary (e.g. byte-level BPE) into a sequence of integer token IDs.
2. **Prefill / initial forward pass**: all prompt tokens are processed through the model in a single parallel forward pass — this is where the KV cache (Module 6) gets populated for every prompt token's Key/Value vectors, across every layer.
3. **Autoregressive decoding begins** (Module 6): the model computes a probability distribution over the vocabulary for the first new token; a decoding strategy (temperature + top-p, typically, for a chat product) samples the next token.
4. **Loop**: that new token's K/V is appended to the cache (not recomputed from scratch — this is exactly why KV caching matters for latency), and the process repeats — generate one token, sample, append to cache, repeat — until an end-of-sequence token or length limit is hit.
5. **(Optionally) Speculative decoding** (Module 6) may be used under the hood to reduce the number of expensive full forward passes needed for a given number of output tokens, without changing the output distribution.
6. **Serving-level considerations** (Module 7): the underlying weights are likely served in a reduced-precision format (fp16/bf16, possibly further quantized to int8/int4) to fit memory and reduce latency/cost; if it's a MoE model, only a small subset of experts are active per token.
7. **Detokenization**: the generated token IDs are converted back into human-readable text using the same tokenizer's vocabulary.

**What this question is actually testing**: whether you understand that tokenization, decoding, and serving efficiency are not independent trivia facts, but a single connected pipeline — losing points here usually means answering only one piece (e.g. only decoding strategies) without touching how it connects to caching/serving.

> 📌 **Added Explanation — "prefill" vs "decode" phase, in simple terms**
> This distinction (step 2 vs steps 3-4) is one of the most commonly under-explained parts of this answer, so it's worth having crisp: **prefill** processes the *entire prompt at once* — because you already have all the prompt tokens sitting there, the model can run them through every layer in one big parallel matrix multiplication (this is compute-heavy but very GPU-efficient, since GPUs love large parallel operations). **Decode** is fundamentally different: you only know one new token at a time (you need the model's own previous output before it can produce the next one), so each decode step is a comparatively tiny, sequential, one-token-at-a-time forward pass — which is *memory-bandwidth-bound* rather than compute-bound, since you're mostly just reading the whole model's weights from memory to process one token. This prefill/decode asymmetry is *why* the KV cache matters so much: without it, every decode step would have to redo the expensive attention computation over the *entire* growing sequence from scratch, turning an already-slow sequential process into a quadratically-slower one.
>
> 🧮 **Numerical Example — why caching saves real work**
> Suppose a prompt is 500 tokens long, and the model generates 100 new tokens.
> - **Without caching**: at each of the 100 decode steps, the model must recompute attention over the *entire* sequence-so-far. Step 1 attends over 501 tokens, step 2 over 502, ..., step 100 over 600 tokens. Total attention-relevant work is proportional to the sum $501+502+\dots+600 \approx 100 \times 550 = 55{,}000$ token-attention operations.
> - **With caching**: at each decode step, the model only computes the new token's Key/Value once and attends against the *already-stored* cache — so each step does work proportional to just the current sequence length being *read*, not recomputed. The K/V *computation* work across the 100 new tokens is only $100$ token-worth of new K/V projections (each attends over the growing cache, but that read is far cheaper than recomputing K/V for every prior token again).
> The qualitative takeaway the notes are pointing at: caching turns an O(N²)-flavored recomputation pattern into an O(N) pattern for the actual K/V generation cost, which is the single biggest practical reason production LLM serving is feasible at all for longer conversations.

### Q: Walk me through how you'd take a base pretrained model to a production-ready chat assistant.

**Strong answer structure**:
1. **Pretraining** (Module 2): CLM objective on a massive, deduplicated, deliberately-mixed corpus, at a roughly Chinchilla-optimal token-to-parameter ratio (Module 3) for the chosen compute budget.
2. **SFT / instruction tuning** (Module 4/5): fine-tune on diverse instruction-formatted (prompt, ideal-response) pairs — this is the stage that teaches the model to actually follow instructions rather than just autocomplete plausible web text.
3. **Reward model training** (Module 5): collect pairwise human preference comparisons on SFT-model outputs, train a scalar reward model via the Bradley-Terry loss.
4. **RLHF (PPO) or DPO** (Module 5): align the policy toward human preferences using the reward model + KL-constrained PPO, or skip the explicit reward model entirely with DPO's direct preference loss.
5. **Evaluation** (Module 8) at every stage — benchmark suites for broad capability tracking, human eval/LLM-as-judge for the specific alignment-sensitive behaviors, hallucination measurement, watching for benchmark contamination in reported numbers.
6. **Serving optimization** (Module 7) for production: quantization, possibly MoE architecture choice made far earlier (at pretraining time, not a late add-on), KV-cache-aware serving infrastructure.

> 📌 **Added Explanation — why the *order* of these stages matters, not just the list**
> A subtlety worth naming explicitly in an interview: each stage progressively *narrows* what the model is optimized for, starting from the broadest possible objective. Pretraining optimizes for "predict plausible text in general" (the broadest possible signal, using the most data). SFT narrows this to "follow instructions in the specific format/style we want." RLHF/DPO narrows it further to "of the responses that follow instructions correctly, prefer the ones humans actually like better" (tone, helpfulness, safety). This is a funnel, not a set of independent switches — and it's *why* each stage's evaluation (step 5) needs to check something different: you evaluate pretraining with broad benchmarks/perplexity, SFT with instruction-following accuracy, and RLHF/DPO specifically with human/LLM-judge preference comparisons, because each stage introduces failure modes the previous stage's metrics wouldn't catch.

---

## 2. "Compare and decide" questions (these test judgment, not just recall)

### Q: You have a fixed budget — would you rather train a bigger model on less data, or a smaller model on more data?

Answer using Module 3's Chinchilla result directly: for a **fixed compute budget**, the compute-optimal choice is **not** "as big as possible" — it's roughly balancing N and D at the ~20-tokens-per-parameter ratio. State the GPT-3-vs-Chinchilla numbers as concrete evidence (GPT-3: ~1.7 tok/param, badly undertrained; Chinchilla: ~20 tok/param, smaller but outperformed GPT-3). Then add the Module 3 nuance: if **inference cost over the model's lifetime** also matters (not just training-compute-optimality), you might deliberately choose an even smaller model trained on even more tokens than Chinchilla-optimal (the Llama approach) — a genuinely balanced answer names both considerations rather than treating Chinchilla's ratio as the final word.

> 📌 **Added Explanation — the Chinchilla compute formula, derived**
> The core relationship referenced throughout this module is:
> $$C \approx 6ND$$
> **Symbols:**
> - $C$ = total training compute, typically measured in FLOPs (floating-point operations)
> - $N$ = number of model parameters
> - $D$ = number of training tokens seen
> - The constant $6$ comes from a standard approximation: a forward pass costs roughly $2ND$ FLOPs (2 FLOPs — one multiply, one add — per parameter per token, for the matrix multiplications that dominate compute), and training requires a backward pass that costs roughly twice the forward pass ($4ND$), so forward + backward together is $2ND + 4ND = 6ND$.
>
> **Why it matters / intuition:** this formula says training compute is essentially a simple product of "how big is the model" times "how much data did it see," which means for a *fixed* compute budget $C$, there's a direct tradeoff — you can afford a bigger $N$ only by using a smaller $D$, and vice versa. The Chinchilla paper's contribution was empirically finding the *ratio* of $N$ to $D$ that gets the best possible loss for a given fixed $C$ — and that ratio turned out to be roughly 20 tokens per parameter, not the much larger model / much smaller data ratios earlier models like GPT-3 had used.
>
> 🧮 **Numerical Example — GPT-3 vs. Chinchilla, in concrete numbers**
> - GPT-3: ~175 billion parameters, trained on ~300 billion tokens. Tokens-per-parameter ratio: $300\text{B} / 175\text{B} \approx 1.7$ tokens/parameter — far below the ~20 ratio, meaning GPT-3 was significantly **undertrained** relative to its size; a lot of its compute budget went into parameters that never saw enough data to be fully "worth" their size.
> - Chinchilla: ~70 billion parameters, trained on ~1.4 trillion tokens. Ratio: $1{,}400\text{B} / 70\text{B} = 20$ tokens/parameter.
> - Despite having **less than half** the parameters of GPT-3, Chinchilla was trained for roughly the *same total compute budget* $C \approx 6ND$ (smaller $N$, proportionally larger $D$, same product) and **outperformed** GPT-3 on downstream benchmarks — direct empirical evidence that "bigger model, less-scaled data" was leaving performance on the table for a fixed compute budget.
>
> 📌 **Added Explanation — the Llama-style "overtrained" counter-nuance**
> If compute-optimal training minimizes *training* cost for a given final loss, why would anyone deliberately train a *smaller-than-Chinchilla-optimal* model on even *more* tokens than the 20:1 ratio suggests? Because Chinchilla-optimality only accounts for the **one-time training cost** — it says nothing about **inference cost**, which is paid over and over, every single time the deployed model serves a request, for the entire lifetime of the product. A smaller model is cheaper and faster to run in production, so if you expect to serve billions of queries, it can be worth spending *extra* training compute (beyond the "optimal" point for training-cost-per-loss) to squeeze more capability into a smaller, cheaper-to-serve model — trading a one-time training-compute inefficiency for a much larger cumulative inference-cost saving.

### Q: Fine-tune, prompt, or RAG — how do you decide?

Directly reuse Module 4's decision framework: prompting for fast iteration/no labeled data/frequently-changing behavior; RAG for knowledge that's large or changes frequently (avoids retraining to update facts); fine-tuning for persistent behavior/style/format changes. Close with the honest production nuance: these are usually **combined**, not mutually exclusive (a fine-tuned base model, augmented with RAG, further steered by per-request prompting).

> 📌 **Added Explanation — "in simple terms" decision heuristic**
> A quick mental shortcut: ask **"is the thing I want to change a fact, or a behavior?"** Facts (today's date, a company's latest product catalog, a document's contents) belong in RAG, because facts change and re-training every time a fact changes is wasteful and slow. Behaviors (always respond in a certain tone, always format output as JSON, always refuse certain requests) belong in fine-tuning, because you want that behavior baked in reliably across *every* interaction, not dependent on retrieving the "right" context each time. Prompting sits on top of both as the cheapest, fastest lever — good for one-off adjustments you're not sure you want to commit to permanently yet.

### Q: RLHF (PPO) or DPO — which would you pick for a real alignment pipeline, and why?

Reuse Module 5's balanced closer: DPO for simplicity/stability/cost (no reward model, no RL infrastructure); RLHF/PPO if you specifically want a standalone reusable reward model (for reranking, red-teaming, evaluation) or believe online RL exploration reaches preference signal a fixed offline DPO dataset can't. **Don't just pick one — name the actual tradeoff**, since interviewers are testing whether you understand *why* the tradeoff exists, not just which one is currently trendier.

> 📌 **Added Explanation — the Bradley-Terry → DPO derivation, fully worked**
> This is explicitly flagged later in the notes as "the single most likely whiteboard-derivation question," so here it is spelled out step by step.
>
> **Step 1 — Bradley-Terry preference model:**
> $$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$
> **Symbols:** $x$ = the prompt, $y_w$ = the "winning" (preferred) response, $y_l$ = the "losing" response, $r(x,y)$ = the scalar reward model's score for response $y$ given prompt $x$, $\sigma(\cdot)$ = the logistic sigmoid function $\sigma(z) = 1/(1+e^{-z})$. **Intuition:** the probability that humans prefer the winning response over the losing one is a sigmoid of the *difference* in their reward scores — bigger reward gap → more confidently preferred, exactly like the Bradley-Terry model discussed for LLM evaluation in Module 8.
>
> **Step 2 — the RLHF objective being optimized (KL-constrained reward maximization):**
> $$\max_{\pi} \; \mathbb{E}_{y\sim\pi(y|x)}\big[r(x,y)\big] - \beta \, D_{KL}\big(\pi(y|x) \,\|\, \pi_{SFT}(y|x)\big)$$
> **Symbols:** $\pi$ = the policy (the model) being optimized, $\pi_{SFT}$ = the fixed reference policy (the SFT checkpoint, before alignment), $\beta$ = a coefficient controlling how strongly the policy is penalized for drifting away from $\pi_{SFT}$, $D_{KL}$ = KL divergence, measuring how different two probability distributions are. **Intuition:** you want to maximize reward, but *not* by drifting arbitrarily far from the well-behaved SFT model — the KL term is a leash preventing reward hacking (Module 5).
>
> **Step 3 — the closed-form optimal policy for that objective (a known result):**
> $$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{SFT}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)$$
> **Symbols:** $Z(x) = \sum_y \pi_{SFT}(y|x)\exp(r(x,y)/\beta)$ is a normalizing constant (a "partition function," summing over all possible responses $y$ so the distribution sums to 1) — importantly, $Z(x)$ depends only on the prompt $x$, not on any specific response $y$. **Intuition:** the KL-constrained reward-maximization problem above has a known closed-form solution: the optimal policy simply *reweights* the reference policy's probabilities by an exponential function of the reward, scaled by $1/\beta$ — responses with higher reward get exponentially boosted probability relative to what $\pi_{SFT}$ already assigned them.
>
> **Step 4 — invert Step 3 to express reward in terms of the policy:**
> Rearranging Step 3 algebraically:
> $$r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{SFT}(y|x)} + \beta \log Z(x)$$
> **Why this step is the clever trick of the whole derivation:** it lets you substitute this expression for $r(x,y)$ *directly back into* the Bradley-Terry loss from Step 1 — turning a loss that needed an explicit reward model $r$ into a loss expressed purely in terms of policies ($\pi^*$ and $\pi_{SFT}$), which are things you can directly optimize via gradient descent.
>
> **Step 5 — substitute into Bradley-Terry, watch $Z(x)$ cancel:**
> Because Step 1's expression only ever needs the *difference* $r(x,y_w) - r(x,y_l)$, and both $y_w$ and $y_l$ share the *same* prompt $x$ (so the same $Z(x)$ term), the $\beta\log Z(x)$ terms are identical for both and **cancel out** when you subtract:
> $$r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^*(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{SFT}(y_l|x)}$$
> Substituting this into the Bradley-Terry sigmoid gives the final **DPO loss** (as a negative-log-likelihood to minimize):
> $$\mathcal{L}_{DPO} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{SFT}(y_l|x)}\right)$$
> **Why this is elegant / the whole point of DPO:** you never needed to train a separate reward model $r$ at all, and you never needed $Z(x)$ (which would require summing over every possible response — intractable) — it cancelled out algebraically. You directly optimize the policy $\pi_\theta$ using nothing but pairs of (preferred, dispreferred) responses and the frozen reference policy $\pi_{SFT}$, which is exactly why DPO is simpler and more stable to implement than full RLHF/PPO (no reward model training stage, no RL loop, no reward-hacking-prone online sampling).
>
> 🧮 **Numerical Example**
> Suppose $\beta = 0.1$, and for a given prompt: $\pi_\theta(y_w|x)=0.4$, $\pi_{SFT}(y_w|x)=0.2$ (the policy has increased probability on the preferred response relative to the reference), and $\pi_\theta(y_l|x)=0.1$, $\pi_{SFT}(y_l|x)=0.15$ (decreased probability on the dispreferred response).
> $$\beta\log\frac{0.4}{0.2} - \beta\log\frac{0.1}{0.15} = 0.1\log(2) - 0.1\log(0.667) = 0.1(0.693) - 0.1(-0.405) = 0.0693+0.0405=0.110$$
> $$\mathcal{L}_{DPO} = -\log\sigma(0.110) = -\log(0.5275) \approx 0.640$$
> Compare this to a case where the policy had moved in the *wrong* direction (decreased $y_w$ probability, increased $y_l$ probability) — the term inside $\sigma(\cdot)$ would be negative, $\sigma$ would output something below 0.5, and $-\log(\cdot)$ would be a *larger* loss — exactly the gradient signal needed to push the policy back toward preferring $y_w$ over $y_l$.

### Q: Full fine-tuning or LoRA?

Full fine-tuning if you have abundant compute/data and need maximum possible task adaptation with no constraint on trainable capacity (e.g., a large domain shift). LoRA (Module 4) when compute/memory is constrained, when you need to serve many different task-specialized variants cheaply (small adapters swapped over one shared frozen base), or when the needed weight update plausibly has low intrinsic rank — which empirically covers a very large fraction of real fine-tuning use cases, which is exactly why LoRA became the default rather than the exception in practice.

> 📌 **Added Explanation — the LoRA parameter-count formula, derived**
> LoRA's core idea: instead of updating the full weight matrix $W \in \mathbb{R}^{d\times d}$ directly (which requires $d^2$ trainable parameters), you freeze $W$ entirely and learn a low-rank *update* on top of it:
> $$\Delta W = BA, \qquad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times d}$$
> **Symbols:** $d$ = the hidden dimension of the weight matrix being adapted, $r$ = the chosen "rank" of the update (a small number, e.g. 8), $B$ and $A$ = the two small trainable matrices whose product approximates the needed weight change.
>
> **Why it's used / intuition:** the hypothesis behind LoRA (empirically well-supported) is that the *change* a model's weights need to undergo during fine-tuning for a specific task has low "intrinsic rank" — meaning it doesn't need the full $d\times d$ space of possible updates, just a much smaller $r$-dimensional subspace. Rather than training a full $d\times d$ matrix, you train two much smaller matrices whose product lives in that smaller subspace — dramatically cutting the number of trainable parameters while still capturing most of the useful adaptation.
>
> 🧮 **Numerical Example — the exact d=4096, r=8 case referenced**
> Full fine-tuning of one $d\times d$ weight matrix with $d=4096$: $4096 \times 4096 = 16{,}777{,}216$ trainable parameters (~16.8 million) for *that one matrix alone*.
> LoRA with rank $r=8$: $B$ has $4096 \times 8 = 32{,}768$ parameters, $A$ has $8 \times 4096 = 32{,}768$ parameters, total $= 65{,}536$ trainable parameters.
> Reduction factor: $16{,}777{,}216 / 65{,}536 = 256$ — exactly the "256x reduction" the original notes flag as the number to have ready to reproduce live. In simple terms: you get to adapt the model's behavior while training **less than half a percent** of the parameters a full update would need, which is why LoRA checkpoints are tiny (megabytes instead of gigabytes) and why serving many task-specific LoRA adapters on top of one shared frozen base model is so cheap.

---

## 3. "Diagnose the failure" questions (a very common Google/Apple MLE interview pattern)

### Q: Your fine-tuned model's benchmark scores look great, but it performs poorly for real users. What's going on, and how would you investigate?

Pull directly from Module 8: first suspect is **benchmark contamination** (check whether benchmark-style questions leaked into pretraining/fine-tuning data) or **benchmark saturation combined with format sensitivity** (the benchmark may no longer differentiate meaningfully, or the eval harness's exact formatting doesn't match how real users actually phrase requests). Second-order suspect: the benchmark measures narrow capability (e.g. multiple-choice knowledge, MMLU-style) that doesn't correlate well with the actual open-ended, conversational task real users need (this is the Module 2/3 "loss/benchmark doesn't equal downstream usefulness" theme resurfacing). Investigation plan: run targeted human evaluation or LLM-as-judge (Module 8) directly on a sample of real production-style queries, specifically checking for the known biases (length, position) that might be inflating the perception of quality if you're only looking at automated proxy metrics.

### Q: Users report the model "sounds confident but says wrong things." Diagnose and propose a fix.

This is a hallucination + calibration question (Module 8), directly connected to the CLM objective's mechanism (Module 2): CLM optimizes for fluent, plausible continuation, with no built-in fact-verification signal, so confident tone and factual correctness are correlated but not identical in what the model was trained to produce. Propose: (a) consistency-based hallucination measurement (sample multiple generations, check agreement, per SelfCheckGPT-style methods) to *quantify* the problem before fixing it, (b) calibration-aware fine-tuning/alignment (reward honest expressions of uncertainty during RLHF/DPO preference data collection — i.e., bake "say 'I'm not sure' when appropriate" into the preference-comparison training signal itself, not just correctness), (c) RAG-style grounding (Module 4) for domains where hallucination risk is highest and external verifiable sources exist.

### Q: Your RLHF-trained model's outputs have gotten weird/degenerate compared to the SFT checkpoint — repetitive praise, strange phrasing, gaming-the-metric behavior. What happened, and how do you fix it?

Textbook **reward hacking** (Module 5) — the policy has drifted too far from `π_SFT` and found a blind spot in the reward model's approximation of true human preference (Goodhart's Law). Fix: increase the KL-penalty coefficient β to constrain drift more tightly, inspect/retrain the reward model on more diverse preference data specifically covering the failure mode observed, or consider whether the reward model was undertrained/overfit on too narrow a preference distribution in the first place.

> 📌 **Added Explanation — connecting this back to the DPO derivation above**
> Notice that $\beta$ here is the *exact same* symbol from the KL-constrained objective in Step 2 of the Bradley-Terry→DPO derivation above — it's not a coincidence, it's literally the same knob. Whether you're running PPO with an explicit reward model or DPO, $\beta$ controls the same underlying tradeoff: how far the policy is allowed to drift from $\pi_{SFT}$ in pursuit of higher reward. Raising $\beta$ pulls the policy back closer to $\pi_{SFT}$ (safer, more conservative, less prone to exploiting reward-model blind spots), while lowering $\beta$ allows more aggressive optimization toward reward (riskier, more prone to the exact degenerate behavior described in this question). This is a good example of the kind of connection ("this diagnosis question and that whiteboard-derivation question share a literal variable") that signals synthesis-level understanding.

### Q: A long-context deployment is running out of GPU memory as conversations get longer, even though the model itself fits fine in memory. Why, and what are your options?

This is testing whether you separate model-weight memory from **KV cache memory** (Module 6) — KV cache grows linearly with sequence length × layers × hidden dim × batch size, and can become the dominant memory consumer for long conversations/large batches, entirely independent of the (fixed) model weight footprint. Options: Multi-Query Attention/Grouped-Query Attention (Module 6) to shrink cache size architecturally, quantizing the KV cache itself (not just weights), reducing max batch size or max context length as a serving-config lever, or RoPE-scaling/ALiBi-style techniques (Module 6) if the real goal is supporting longer contexts without proportionally exploding memory.

> 📌 **Added Explanation — the KV cache memory formula, made explicit**
> The linear relationship referenced can be written out concretely:
> $$\text{KV cache memory} \approx 2 \times L \times S \times H \times B \times P$$
> **Symbols:** $L$ = number of transformer layers, $S$ = sequence length (number of tokens cached), $H$ = hidden dimension per layer (sometimes further split as num-heads × head-dim), $B$ = batch size (number of concurrent sequences being served), $P$ = bytes per parameter for the precision used (e.g. 2 bytes for fp16), and the leading $2\times$ accounts for storing *both* the Key and the Value vectors (not just one).
>
> 🧮 **Numerical Example — the ~2GB-for-one-sequence figure**
> Take a mid-sized model: $L=32$ layers, $H=4096$ hidden dim, sequence length $S=4096$ tokens, batch size $B=1$, fp16 precision so $P=2$ bytes:
> $$2 \times 32 \times 4096 \times 4096 \times 1 \times 2 \text{ bytes} = 2 \times 32 \times 4096 \times 4096 \times 2$$
> $$= 2{,}147{,}483{,}648 \text{ bytes} \approx 2.15\text{ GB}$$
> This matches the "~2GB for one sequence" figure the notes flag — and critically, this number scales **linearly** with both sequence length and batch size: double the context length or double the number of concurrent users, and this ~2GB becomes ~4GB, entirely separate from whatever memory the frozen model weights themselves consume. This is exactly why a model that "fits fine" in memory on its own can still cause out-of-memory errors purely from cache growth as conversations lengthen or concurrency increases.

---

## 4. "Explain like I'm the interviewer, and I'll interrupt with follow-ups" — derivation-style questions

These are the ones where being able to actually write the formula, not just gesture at it, separates strong from weak answers. Have these fully loaded:

- **Chinchilla compute formula**: `C ≈ 6ND`, and the ~20 tokens/parameter compute-optimal ratio, plus the concrete GPT-3 vs Chinchilla numbers (Module 3).
- **LoRA parameter count**: `ΔW = BA`, with the explicit d=4096, r=8 → 256x reduction example (Module 4) ready to redo live on a whiteboard.
- **Bradley-Terry + DPO derivation**: from `P(y_w≻y_l) = σ(r(y_w)-r(y_l))`, through the closed-form optimal policy `π* = (1/Z)π_SFT·exp(r/β)`, to the final DPO loss with `Z(x)` cancellation (Module 5) — this is the single most likely "derive this on the whiteboard" question in the whole syllabus, given how clean and self-contained the derivation is.
- **KV cache complexity**: O(N²) naive vs O(N) cached, plus the concrete ~2GB-for-one-sequence memory example (Module 6).
- **Quantization formula**: `scale = (max-min)/(2^n-1)`, plus why NF4 uses non-uniform quantile-based spacing instead (Module 7).

> 📌 **Added Explanation — the quantization scale formula, derived**
> Uniform (linear) quantization maps a continuous range of real values into a fixed number of discrete integer levels:
> $$\text{scale} = \frac{\max - \min}{2^n - 1}$$
> **Symbols:** $\max, \min$ = the maximum and minimum values in the tensor being quantized (e.g. a weight matrix's values), $n$ = the number of bits used per quantized value (e.g. $n=8$ for int8), $2^n - 1$ = the number of distinct integer "buckets" available minus one (e.g. 255 for 8-bit, giving integer levels 0-255).
>
> **Why it's used / intuition:** you're converting a continuous range of floating-point numbers into a small number of evenly-spaced discrete "buckets" — the scale factor tells you the real-number width represented by one integer step. To quantize a value $x$: $q = \text{round}((x-\min)/\text{scale})$, and to recover an approximation later: $\hat{x} = q\times\text{scale} + \min$.
>
> 🧮 **Numerical Example**
> Suppose a weight tensor's values range from $\min=-2.0$ to $\max=2.0$, quantized to int8 ($n=8$, so $2^8-1=255$ levels):
> $$\text{scale} = \frac{2.0-(-2.0)}{255} = \frac{4.0}{255} \approx 0.0157$$
> A weight value of $x=1.0$ quantizes to $q = \text{round}((1.0-(-2.0))/0.0157) = \text{round}(191.1) = 191$. Dequantizing back: $\hat{x} = 191\times0.0157 + (-2.0) \approx 1.0=1.0007$ — a tiny rounding error (~0.0007), which is the unavoidable cost of representing a continuous range with only 256 discrete levels.
>
> **Why NF4 uses non-uniform, quantile-based spacing instead:** uniform quantization spends its limited discrete levels *evenly* across the min-max range — but neural network weights are typically **not** uniformly distributed; they tend to cluster densely near zero (roughly Gaussian/bell-curve shaped) with a few outlier values far from zero. If you space your quantization levels evenly across the full range, you "waste" many of your precious 16 levels (for 4-bit) on the sparse, rarely-occurring extreme values, while the densely-populated near-zero region — where most of the actual information lives — gets coarsely represented by relatively few levels. NF4 (NormalFloat4) instead places its quantization levels at the **quantiles of a standard normal distribution**, meaning levels are packed more tightly where weight values are actually most likely to fall (near zero) and more sparsely where they're rare (the tails) — extracting more effective precision from the same 4 bits by matching the quantization grid to the *actual shape* of the weight distribution rather than assuming it's uniform.

---

## 5. Rapid-fire cross-module connections (say these out loud, unprompted, when relevant — they signal real understanding, not memorized definitions)

- Perplexity (Module 2) not predicting downstream task performance is the **same underlying theme** as emergent abilities (Module 3) being partly a metric artifact, and as benchmark scores (Module 8) not predicting real-user satisfaction — in all three cases, **the proxy metric and the thing you actually care about are correlated but not identical**, and that gap is where a lot of real-world model-evaluation mistakes happen.
- The Bradley-Terry model (Module 5, reward modeling) and Elo-based human-eval aggregation (Module 8) are **mathematically the same underlying pairwise-comparison framework**, just applied at different stages of the pipeline (training signal vs. final evaluation).
- KL divergence shows up in **three unrelated-sounding places** that are worth explicitly connecting if asked: PPO's policy-drift penalty (Module 5), DPO's implicit β-weighted preference loss (Module 5, same underlying constraint reformulated), and distillation's teacher-student matching loss (Module 7) — same mathematical tool, three different purposes (constrain drift, encode implicit reward, transfer knowledge).
- MoE (Module 7) breaking the "all parameters are active per token" assumption directly means the **Chinchilla scaling-law formula `C≈6ND` (Module 3) needs adjustment for MoE models** — the effective compute-relevant parameter count is the *active* parameter count per token, not total parameters — a genuinely open research nuance worth naming if scaling laws come up in the context of a MoE architecture.

> 📌 **Added Explanation — KL divergence, defined once so all three uses are grounded**
> Since KL divergence is name-dropped three times above, it's worth having the base definition solid:
> $$D_{KL}(P\|Q) = \sum_{x} P(x)\log\frac{P(x)}{Q(x)}$$
> **Symbols:** $P$ = the distribution you're measuring the "extra cost/difference" of, $Q$ = the reference distribution you're comparing against, $x$ ranges over all possible outcomes. **Intuition:** it measures how much "surprise" or "extra information" you'd incur if you assumed outcomes came from distribution $Q$ when they actually come from distribution $P$ — it's zero when $P$ and $Q$ are identical, and grows as they diverge. It is *not* symmetric ($D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ in general), which is itself a common interview gotcha question.
> - In **PPO**, $P=\pi$ (the current policy being trained) and $Q=\pi_{SFT}$ (the frozen reference) — penalizing $D_{KL}(\pi\|\pi_{SFT})$ keeps the policy from drifting too far from sensible SFT behavior while chasing reward.
> - In **DPO**, this same KL constraint is never computed explicitly at training time — it's *implicitly* encoded because the whole loss was derived (see Section 2 above) directly from the closed-form solution to that KL-constrained objective — so DPO achieves the same effective constraint "for free," algebraically, without ever running a separate KL computation during training.
> - In **distillation**, $P$ = the teacher model's output distribution and $Q$ = the student model's output distribution — minimizing $D_{KL}(P\|Q)$ trains the student to mimic the teacher's full probability distribution over outputs, not just its single top prediction, transferring more nuanced "dark knowledge" than a simple hard-label loss would.

---

## 6. Final self-check — can you do all of these cold?

> 📌 **Added Explanation — worked answers for the self-check, since these are the questions of the module**

- [ ] **Derive the DPO loss from Bradley-Terry + the KL-constrained RL optimal policy, on a whiteboard, unprompted.**
  ✅ *Answer:* See the full 5-step derivation worked out in Section 2 above (Bradley-Terry sigmoid → KL-constrained objective → closed-form optimal policy → invert for reward → substitute and cancel $Z(x)$ → final DPO loss). The key moves to hit, in order: (1) state Bradley-Terry as a sigmoid of reward difference, (2) state the KL-constrained RL objective being solved, (3) quote the known closed-form solution $\pi^* \propto \pi_{SFT}\exp(r/\beta)$, (4) invert it to write $r$ in terms of $\pi^*/\pi_{SFT}$, (5) substitute into Bradley-Terry and note $Z(x)$ cancels because it's shared between the winning and losing response for the same prompt.

- [ ] **Compute LoRA's parameter reduction for a given d and r, from scratch.**
  ✅ *Answer:* Full matrix has $d^2$ parameters; LoRA has $2dr$ parameters (from $B\in\mathbb{R}^{d\times r}$ and $A\in\mathbb{R}^{r\times d}$). Reduction factor $= d^2/(2dr) = d/(2r)$. For $d=4096, r=8$: $4096/16 = 256$ — matching the worked example above.

- [ ] **Explain why GPT-3 was undertrained relative to Chinchilla, with the actual tokens/parameter numbers.**
  ✅ *Answer:* GPT-3 used ~1.7 tokens/parameter (175B params, 300B tokens) vs. Chinchilla's ~20 tokens/parameter (70B params, 1.4T tokens) — Chinchilla's ratio was empirically found to be closer to compute-optimal for a given fixed training-compute budget $C\approx6ND$, meaning GPT-3's compute was spent on "too many parameters, too little data per parameter" relative to the optimal split.

- [ ] **Explain KV caching's O(N²)→O(N) complexity shift, and estimate cache memory for a given model size/context length.**
  ✅ *Answer:* Without caching, generating $N$ tokens autoregressively while recomputing full attention over the growing sequence each time costs work proportional to $1+2+\dots+N \approx O(N^2)$. With caching, each new token's K/V is computed once and stored, so total K/V-computation work across generation is $O(N)$. Memory estimate uses $2\times L\times S\times H\times B\times P$ (see worked example above): e.g. ~2.15GB for a 32-layer, 4096-hidden-dim model at 4096 sequence length, batch size 1, fp16.

- [ ] **Name and explain all three LLM-as-judge biases, with a mitigation for each.**
  ✅ *Answer:* Position bias (favors whichever response is shown first/second; mitigated by evaluating both orderings and discarding order-flip-sensitive results) — verbosity bias (favors longer responses regardless of quality; mitigated by explicit instructions to ignore length/style) — self-preference bias (favors outputs stylistically similar to the judge's own model family; mitigated by using multiple diverse judges from different providers and checking cross-judge agreement). [Full detail in Module 8.]

- [ ] **Explain why CLM pretraining makes hallucination a structural consequence, not an incidental bug.**
  ✅ *Answer:* CLM optimizes purely for predicting plausible, fluent next tokens given context, with no explicit built-in mechanism distinguishing "fluent and plausible" from "factually true" — so a model can be highly confident about a fabricated continuation simply because it's statistically plausible-sounding; plausibility and truth are correlated in training data but are not the same thing the model was actually optimized to produce.

- [ ] **Walk through the full pretrain → SFT → RM → RLHF/DPO pipeline without skipping a stage.**
  ✅ *Answer:* (1) Pretraining: CLM objective on a huge, deduplicated corpus at a roughly Chinchilla-optimal N:D ratio for the compute budget. (2) SFT: fine-tune on (prompt, ideal-response) pairs so the model learns to follow instructions rather than just autocomplete web text. (3) Reward model: collect pairwise human preferences on SFT outputs, train a scalar reward model via the Bradley-Terry loss. (4) RLHF (PPO, using the reward model plus a KL penalty against the SFT policy) or DPO (skipping the explicit reward model, optimizing preferences directly via the derived closed-form loss) — either way producing the final aligned policy.

- [ ] **Give a balanced (not one-sided) answer to "fine-tune vs prompt vs RAG" and "RLHF vs DPO."**
  ✅ *Answer:* Fine-tune vs. prompt vs. RAG: prompting for fast iteration and frequently-changing instructions with no labeled data; RAG for large or frequently-changing *knowledge* (avoids retraining just to update facts); fine-tuning for persistent behavior/style/format changes that should hold across every interaction — and in real production systems these are typically combined rather than chosen exclusively. RLHF vs. DPO: DPO for simplicity, training stability, and lower infrastructure cost (no separate reward model, no RL loop); RLHF/PPO when you specifically want a standalone reusable reward model (useful for reranking, red-teaming, or evaluation elsewhere) or when online RL exploration is believed to reach preference signal that a fixed, offline DPO preference dataset can't capture.

If any of these feel shaky, that's a pointer back to the specific module above — everything on this list is covered in full depth somewhere in Modules 1-8.

---

## ❓ 7. Added Interview Q&A (Apple / Google-style ML Engineer questions)

**Q1: Your team is deciding between serving a dense 70B model vs. an MoE model with 70B total parameters but only ~13B active per token. Walk through the tradeoffs.**

*Model answer:* The core tradeoff is compute/memory-per-token vs. total capacity. The MoE model gets to have a much larger total parameter count (and thus more "capacity" to store diverse knowledge/skills across its experts) while only paying the *inference compute cost* of the ~13B active parameters per token — meaning the effective FLOPs per forward pass, and therefore latency, looks much closer to a 13B dense model than a 70B dense model, even though total memory footprint (all experts must still be loaded, typically) is closer to the full 70B. This matters directly for the Chinchilla-style scaling-law discussion: the compute-optimal token count should be computed against the *active* parameter count, not the total, since that's what's actually driving FLOPs during both training and inference. The dense 70B model, by contrast, pays the full 70B-parameter compute cost on every single token, so it will be meaningfully slower/more expensive per request despite having the same *total* parameter count as the MoE model. The practical downside of MoE is more engineering complexity (routing logic, load balancing across experts, and typically still needing enough memory/VRAM to hold all experts even though only a few fire per token) — so the decision comes down to whether the deployment is latency/throughput-sensitive enough at scale to justify that added complexity for the FLOPs savings.

**Q2: A colleague suggests using perplexity on a held-out validation set as the single metric to decide which of two checkpoints to ship. What's your response?**

*Model answer:* I'd push back, citing the throughline from Module 2/3/8: perplexity measures how well the model predicts the next token on that specific held-out distribution, which correlates with — but doesn't guarantee — downstream task usefulness, instruction-following quality, or alignment with what real users actually want. Two checkpoints could have nearly identical perplexity while differing meaningfully in, say, instruction-following accuracy or hallucination rate, especially post-SFT/RLHF where the training objective has already diverged from pure next-token prediction. I'd recommend perplexity as one fast, cheap early signal (useful for catching gross regressions quickly) but require it be paired with task-specific benchmarks and some pairwise human/LLM-judge evaluation on realistic prompts before a shipping decision — exactly the multi-layered evaluation approach from Module 8, rather than trusting one proxy number.

**Q3: You need to fit a 70B-parameter model to run inference on a GPU with limited VRAM. Name at least three distinct techniques you could combine, and explain what each actually saves.**

*Model answer:* First, weight quantization (Module 7) — e.g. going from fp16 (2 bytes/parameter) down to int4/NF4 (0.5 bytes/parameter) cuts the model's static weight-memory footprint roughly 4x, at some cost to numerical precision, mitigated for NF4 by using quantile-based non-uniform bucket spacing rather than naive uniform quantization. Second, KV cache reduction — techniques like Grouped-Query Attention reduce the number of distinct Key/Value projections that need to be cached per token, directly shrinking the linear-in-sequence-length memory cost that's separate from the static weight footprint; alternatively, quantizing the KV cache itself (not just the weights) gets similar savings on that specific memory pool. Third, if the model is MoE, you could consider techniques to only keep the most-frequently-routed experts resident in fast memory (offloading rarely-used experts to slower storage), trading some latency on cold-expert calls for a smaller resident memory footprint. I'd note these are complementary, not exclusive — a real production deployment often combines all three (quantized weights + efficient attention variant + expert-offloading if MoE) rather than picking just one.

**Q4: Explain, in terms an engineering leader without ML background would understand, why "the benchmark went up but users are complaining" is not a contradiction.**

*Model answer:* I'd frame it with an analogy: imagine you measure a customer service team purely by "average call length" and it goes down — that looks like an improvement on paper, but if it went down because agents started hanging up on frustrated customers faster, the metric improved while the actual experience got worse. Benchmarks are the same kind of proxy: they're a fixed, narrow test (often multiple-choice trivia-style questions) that's convenient to measure automatically, but real user satisfaction depends on things the benchmark was never designed to capture — tone, handling of ambiguous or open-ended requests, honesty about uncertainty, avoiding overly verbose or falsely-confident answers. A model can genuinely get "smarter" on the narrow thing the benchmark tests while simultaneously getting worse on things users actually experience day to day, especially if those user-facing qualities were never part of what improved the benchmark score in the first place. That's why any real evaluation process needs to combine the cheap automated benchmark with actual human or LLM-judge evaluation on real usage patterns, rather than trusting the single number.

**Q5: If you could only pick one evaluation method — benchmark suites, human evaluation, or LLM-as-judge — for an ongoing production model, which would you pick and why? (Trick question — how do you handle it?)**

*Model answer:* I'd name it as a trick question directly: no single one of these should be used alone in a serious production setting, because each has a distinct, well-documented blind spot the others are specifically good at catching (benchmark contamination/saturation vs. rater disagreement/length-bias vs. position/verbosity/self-preference bias) — relying on just one means you're fully exposed to exactly that method's failure mode with nothing to cross-check against. If genuinely forced to pick one for cost reasons, I'd lean toward LLM-as-judge for an *ongoing* production monitoring signal specifically because of its speed/cost/scale advantage, which matters most for continuous monitoring rather than a one-time gate — but I'd immediately caveat that this only works if it's periodically validated against a smaller human-evaluation sample (checking agreement rate) so you'd catch if the judge itself started drifting or exhibiting one of its known biases undetected. So the honest answer is: pick one for day-to-day cost reasons if forced, but treat that as a calculated risk requiring periodic cross-validation, not a permanent substitute for the other two.

---
*End of Module 9 (maximum depth, enhanced). This completes the LLM Basics syllabus (Modules 1-9) — tokenization through interview synthesis, all at full depth with formulas, numerical examples, and standalone real-world usage notes.*
