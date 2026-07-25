# Module 7 — Efficiency & Serving (Master Notes, Maximum Depth)

## 0. The framing — where does cost actually come from

Two separate cost axes matter for LLMs: **training cost** (one-time, dominated by FLOPs = floating point operations, per Module 3's `C ≈ 6ND`) and **inference/serving cost** (recurring, dominated by memory bandwidth and memory footprint far more than raw FLOPs, per Module 6's KV-cache discussion). This module covers the techniques that attack both axes: quantization and mixed precision reduce the *size* of numbers (bytes per value), MoE reduces *how many parameters are active* per forward pass, and distillation reduces the *number of parameters* entirely by training a smaller model to imitate a larger one.

---

## 1. Quantization

### The core idea, in plain words
Neural network weights are normally stored as 32-bit or 16-bit floating point numbers. Quantization stores them using **fewer bits** (commonly 8-bit or 4-bit integers), which directly shrinks memory footprint and can speed up computation (integer arithmetic is cheaper than floating point on many hardware paths), at the cost of some precision/accuracy loss.

### The basic math — uniform (linear) quantization
To convert a floating-point value `x` (within some range `[min, max]`) into an n-bit integer representation:
```
scale = (max - min) / (2^n - 1)
x_quantized = round( (x - min) / scale )
```
And to reconstruct (dequantize) an approximate floating-point value back:
```
x_reconstructed = x_quantized × scale + min
```

### Numerical worked example — INT8 quantization
Suppose a layer's weight values range from `min=-2.0` to `max=2.0`, and we're quantizing to 8-bit (n=8, so `2^8 - 1 = 255` possible integer levels).
```
scale = (2.0 - (-2.0)) / 255 = 4.0 / 255 ≈ 0.01569
```
Take an actual weight value `x = 0.73`:
```
x_quantized = round( (0.73 - (-2.0)) / 0.01569 ) = round( 2.73 / 0.01569 ) = round(173.99) = 174
```
Reconstruct back to float:
```
x_reconstructed = 174 × 0.01569 + (-2.0) = 2.730 - 2.0 = 0.730
```
In this case reconstruction was nearly exact — but consider a value very close to a quantization boundary, `x = 0.735`:
```
x_quantized = round( 2.735 / 0.01569 ) = round(174.32) = 174 (same bucket as 0.73)
x_reconstructed = 0.730 (same as before)
```
The **quantization error** here is `0.735 - 0.730 = 0.005` — small individually, but this rounding error is present on *every single weight* in the model, and it compounds through many layers of matrix multiplication during the forward pass — this cumulative compounding is exactly why aggressive quantization (very low bit-widths) can noticeably degrade model output quality if not done carefully.

### Why 4-bit quantization needs a smarter scheme — NF4 (NormalFloat4)
Uniform/linear quantization (as above) spaces the quantization levels **evenly** across the value range. But pretrained neural network weights are empirically **not uniformly distributed** — they overwhelmingly follow an approximately **Gaussian (normal) distribution**, clustered densely near zero with a long thin tail at extreme values. Using evenly-spaced quantization levels wastes precision on the sparse tail region and under-represents the dense near-zero region where most of the actual weight values live.

**NF4's fix**: instead of evenly-spaced levels, choose quantization levels that are **evenly spaced in cumulative probability** under a standard normal distribution (i.e., placed at the quantiles of a Gaussian) — meaning more quantization levels are concentrated near zero (where weight density is high) and fewer levels out in the tails (where weight density is low) — this matches the actual empirical distribution of weights far better than uniform spacing, which is precisely why QLoRA (Module 4) specifically uses NF4 rather than plain uniform 4-bit quantization, and reports much smaller accuracy degradation as a result.

### GPTQ and AWQ (post-training quantization methods worth naming)
- **GPTQ**: quantizes weights **layer by layer**, and for each layer, uses **second-order information (an approximation of the Hessian, the matrix of second derivatives of the loss w.r.t. weights)** to decide the *order* in which to quantize individual weights and to adjust the remaining not-yet-quantized weights in that layer to compensate for the error just introduced — quantizing greedily while correcting for accumulated error, rather than quantizing every weight independently/simultaneously with no error correction.
- **AWQ (Activation-aware Weight Quantization)**: observes that not all weights are equally important — weights that multiply against **consistently large-magnitude activations** have an outsized effect on the layer's output, so AWQ identifies and **preserves higher precision for that small salient subset of weights** (roughly 1% of weights, identified by looking at activation statistics, not weight magnitude), while aggressively quantizing the rest — the key insight being "look at activations, not just weights, to decide what precision each weight actually needs."

### Numerical intuition for the memory savings
A 70B parameter model in fp16 (2 bytes/param): `70 × 10^9 × 2 bytes = 140 GB`. The same model in 4-bit (0.5 bytes/param): `70 × 10^9 × 0.5 bytes = 35 GB` — a **4x memory reduction**, turning a model that needs multiple high-end GPUs just to hold in memory into one that can fit on a single consumer-grade GPU (this exact reduction is what QLoRA leverages, as covered in Module 4).

---

## 2. Mixed Precision Training

### The core idea
During *training* (not just inference), use **lower-precision formats (fp16 or bf16, 16-bit) for the bulk of computation** (forward pass, most of backward pass) while **keeping a master copy of weights in fp32 (32-bit)** for the actual optimizer weight-update step — combining most of the speed/memory benefit of low precision with most of the numerical stability of full precision.

### Why you can't just train fully in fp16 — the concrete numerical problem
fp16 has a much smaller representable range than fp32 — specifically, very small gradient values (common in deep networks, especially early in training or in later layers during backprop) can **underflow to exactly zero** in fp16, since fp16's smallest representable positive normal number is around `6.1 × 10^-5`, and many real gradient values during training fall well below that. If a gradient underflows to zero, that weight simply **stops learning** — a genuine training failure, not just a minor precision inconvenience.

### The fix — loss scaling
Before computing gradients, **multiply the loss by a large scaling factor** (e.g., 1024 or higher), so all the gradients computed during backprop (which are proportional to the loss, by the chain rule) get scaled up proportionally too, pushing them up out of fp16's underflow range. After backprop, **before the optimizer update step**, divide the gradients back down by that same scaling factor to restore their true magnitude — the loss scaling only affects the backward pass numerics, never the actual mathematical result.

**Numerical example**: suppose the true gradient for some weight is `0.00003` — this is dangerously close to fp16's underflow floor and risks rounding to zero. Scale the loss by 1024 before backprop: the computed (scaled) gradient becomes `0.00003 × 1024 ≈ 0.0307` — comfortably representable in fp16 with good precision. After backprop completes, divide by 1024 again: `0.0307 / 1024 ≈ 0.00003` — recovering the correct true gradient value, now safely computed without ever passing through the dangerous near-zero range in fp16 during the actual backward-pass arithmetic.

### bf16 (bfloat16) vs fp16 — the key structural difference (a favorite interview distinction)
Both are 16-bit formats, but they allocate their bits differently:
- **fp16**: 1 sign bit, 5 exponent bits, 10 mantissa bits — smaller exponent range (more prone to overflow/underflow) but more mantissa precision.
- **bf16**: 1 sign bit, **8 exponent bits** (same as fp32!), 7 mantissa bits — same dynamic range as fp32 (much less prone to overflow/underflow), but less mantissa precision (fewer significant digits) than fp16.

**Practical consequence**: bf16 rarely needs the loss-scaling trick described above (its exponent range matches fp32, so underflow/overflow during training is far less of a concern), which is why bf16 has become the dominant training format for large modern LLMs (simpler training recipe, fewer numerical-stability hyperparameters to tune) — the tradeoff is bf16 has coarser precision *within* a given exponent range (fewer mantissa bits than fp16), which in practice matters less for large-model training than avoiding underflow/overflow does.

### Gradient checkpointing (activation checkpointing) — a separate, complementary memory-saving technique
During the forward pass, a naive implementation stores every intermediate activation (the output of every layer) in memory, because the backward pass needs them to compute gradients (chain rule). For a very deep model, this activation memory can dominate total memory usage. **Gradient checkpointing** trades compute for memory: instead of storing *all* intermediate activations, only store activations at a subset of "checkpoint" layers, and **recompute the discarded activations on-the-fly during the backward pass** (by re-running the forward computation for just that segment, starting from the nearest stored checkpoint) when they're needed for gradient computation.

**Numerical intuition**: for a model with L layers, naive activation storage is `O(L)`. With checkpointing at, say, `√L` evenly-spaced points, activation memory drops to roughly `O(√L)`, at the cost of roughly **one extra forward pass's worth of recomputation** during backward (since discarded segments are recomputed once each) — a concrete, commonly cited rule of thumb is gradient checkpointing adds ~30-40% more compute time in exchange for very substantial (multiples-x) activation memory reduction, which is often exactly the right trade when memory (not compute time) is the binding constraint for fitting a large model's training on available hardware.

---

## 3. Mixture-of-Experts (MoE)

### The core idea — decoupling total parameters from active (compute) parameters
Every technique so far reduces the *precision* or *memory* of parameters, but keeps all parameters active on every forward pass. MoE instead changes the **architecture**: replace a single large feed-forward network (FFN) layer with **many parallel "expert" FFN sub-networks**, plus a small **router/gating network** that, for each input token, selects only a small subset (commonly just 1 or 2) of those experts to actually process that specific token — the rest of the experts do zero work for that token.

### The routing mechanism, explained
The router is typically a small learned linear layer that takes the token's hidden representation and outputs a probability/score over all `E` experts (via softmax), then selects the **top-k** highest-scoring experts (k is usually 1 or 2, far smaller than E, which might be 8, 16, or even hundreds of experts) to actually process that token — the token's FFN output becomes a weighted combination (using the router's scores) of just those k selected experts' outputs, with all non-selected experts contributing nothing and requiring no computation for this token.

### Numerical worked example of the parameter/compute decoupling
Say a model has `E=8` experts, each expert being a standalone FFN of size equivalent to `F` parameters, and the router uses **top-2** routing.

**Total parameters** (what must be *stored*, in memory/on disk): `8 × F` (all 8 experts' weights exist and must be stored, regardless of routing).

**Active parameters per token** (what's actually *computed* during a forward pass for any single token): only `2 × F` (just the 2 selected experts do any work) — **plus** the (comparatively tiny) router itself.

```
Total params = 8F,  Active params per token = 2F  →  only 25% of total parameters are "active" for any given token.
```

**Why this matters practically**: the model has the **representational capacity of an 8F-parameter model** (since all 8F parameters exist and different tokens can route to different expert combinations, letting different experts specialize) while only paying the **inference compute cost of a 2F-parameter model** for any individual token — this is the central appeal of MoE: **you get to scale total parameter count (and thus capacity/knowledge) largely independently of per-token inference compute cost**, which is a fundamentally different scaling lever than the dense-model scaling laws covered in Module 3 (Chinchilla's `C ≈ 6ND` formula implicitly assumes every parameter is active on every token — MoE breaks that assumption).

### The known training challenge — load balancing
If left unconstrained, the router can degenerate into a **collapsed state** where it always routes most tokens to just one or two "favorite" experts (a self-reinforcing pattern: an expert that gets more training data early on becomes better, which makes the router favor it even more, starving the other experts of training signal entirely) — wasting the majority of the model's total parameter capacity, since unused experts never improve. The standard fix is adding an **auxiliary load-balancing loss** term during training that explicitly penalizes uneven routing distribution across experts, pushing the router toward using all experts roughly equally over the training corpus, so the full parameter capacity actually gets utilized.

### Where MoE is used standalone in practice
**Mixtral 8x7B** (Mistral) is the clearest, most commonly cited real-world example — "8x7B" literally names the architecture: 8 experts, each roughly 7B-parameter-sized, with top-2 routing, giving roughly 47B *total* parameters but only around 13B *active* parameters per token (2 experts × ~7B, plus shared components) — meaning it runs at roughly the inference cost of a ~13B dense model while having the knowledge capacity closer to a much larger model. Google's **Switch Transformer** and GPT-4 (widely reported/rumored, though not officially confirmed by OpenAI) are other commonly referenced MoE-based models.

---

## 4. Distillation

### The core idea, in plain words
Train a smaller "student" model to **mimic the outputs of a larger, already-trained "teacher" model**, rather than (or in addition to) training the student purely on raw ground-truth labels — the intuition being that the teacher's full output probability distribution carries richer information than a single hard ground-truth label alone.

### Why the teacher's full distribution helps — "dark knowledge"
A ground-truth label is a single "hard" answer (e.g., the correct next token is exactly "cat," full stop). But a well-trained teacher model's output distribution might assign, say, P(cat)=0.7, P(dog)=0.2, P(mouse)=0.05, and small residual probability spread across other tokens — this reveals **relative similarity information** ("cat" and "dog" are both plausible animal-continuation words, much more so than an unrelated word) that a single hard label simply cannot express. This extra signal embedded in the full probability distribution is often called **"dark knowledge"** (a term from Hinton et al.'s original distillation paper) — training the student to match this full soft distribution (not just the single correct label) transfers more nuanced information about the teacher's learned decision boundaries than hard-label training alone ever could.

### The distillation loss — softened targets
To make the teacher's distribution even more informative (spread more visibly across near-miss options rather than nearly all probability mass sitting on the single top token), distillation typically applies a **temperature** (same mechanism as Module 6's decoding temperature) to *both* the teacher's and student's logits when computing the distillation loss — using T > 1 to soften/flatten both distributions before comparing them:
```
L_distill = KL( softmax(z_teacher / T) || softmax(z_student / T) )
```
(comparing the softened teacher distribution against the softened student distribution via KL divergence — same KL divergence concept introduced in Module 5's PPO discussion, here used to measure how well the student's output distribution matches the teacher's, rather than to measure drift from a reference policy). This is very often combined (as a weighted sum) with a standard hard-label cross-entropy loss against the true ground-truth labels as well, rather than using the distillation loss in complete isolation.

### Where distillation is used standalone in practice
**DistilBERT** is the textbook example — trained to mimic BERT's output distributions (and hidden-state representations, in a more complete implementation) while using roughly 40% fewer parameters, reported at the time of release to retain around 97% of BERT's language-understanding performance on the GLUE benchmark while running significantly faster and reducing model size substantially. Distillation is also a common step in producing smaller "mini" or "flash"-tier variants of production LLM families, where a smaller model is trained partly on outputs generated by the larger flagship model.

---

## 5. Side-by-side summary table (memorize this cold)

| | Quantization | Mixed Precision Training | MoE | Distillation |
|---|---|---|---|---|
| What it reduces | Bytes per stored/computed value | Bytes per value during training compute | Active params per token (not total params) | Total parameter count entirely |
| Applies to training or inference? | Mostly inference (also QLoRA training) | Training | Both (architecture choice) | Training (produces a smaller model) |
| Key risk/tradeoff | Compounding rounding error, especially at low bit-widths | Gradient underflow (fp16) unless handled (loss scaling / bf16) | Router collapse without load-balancing loss | Student capacity ceiling — can't exceed what teacher's knowledge + soft labels can transfer |
| Concrete technique to name | NF4, GPTQ, AWQ | bf16, loss scaling, gradient checkpointing | Top-k routing, auxiliary load-balancing loss | Soft-label KL loss with temperature ("dark knowledge") |

---

## 6. Quick-fire Q&A (self-test)

**Q: Write the linear quantization formula and explain why quantization error compounds across a deep network.**
A: `x_quantized = round((x - min)/scale)`, `x_reconstructed = x_quantized × scale + min`. Every weight in every layer incurs a small rounding error individually; since layers feed into each other through matrix multiplications, these small errors accumulate/propagate through the network's depth, so aggressive (low-bit) quantization can meaningfully degrade output quality even though each individual rounding error is small.

**Q: Why does NF4 use non-uniform quantization levels, and what determines their spacing?**
A: Pretrained weights are empirically approximately Gaussian-distributed (dense near zero, sparse in the tails), so NF4 spaces quantization levels evenly in cumulative probability under a normal distribution — concentrating more precision where weight density is high (near zero) and less where it's low (the tails), unlike uniform quantization which wastes precision on the sparse tail region.

**Q: What specific numerical failure does loss scaling prevent during fp16 mixed-precision training?**
A: Gradient underflow — very small gradient values common in deep networks can round to exactly zero in fp16's limited range, halting learning for those weights; scaling the loss up before backprop (then dividing the resulting gradients back down before the optimizer step) keeps gradients within fp16's representable range without changing their true mathematical value.

**Q: What's the structural difference between fp16 and bf16, and why does it matter for training stability?**
A: bf16 uses 8 exponent bits (same range as fp32) and 7 mantissa bits, while fp16 uses 5 exponent bits and 10 mantissa bits — bf16's wider exponent range makes it far less prone to underflow/overflow than fp16, which is why bf16 training rarely needs the loss-scaling workaround that fp16 training requires.

**Q: In an MoE model with 8 experts and top-2 routing, what fraction of total parameters are active per token, and why is this the central appeal of MoE?**
A: 2/8 = 25% of total parameters are active per token. This decouples total model capacity (all 8 experts' worth of parameters, enabling specialization) from per-token inference compute cost (only 2 experts' worth), letting you scale total parameter count largely independently of inference cost — a different scaling lever than dense-model Chinchilla scaling, which assumes all parameters are active on every token.

**Q: What failure mode does the MoE auxiliary load-balancing loss prevent, and why is it self-reinforcing without that loss?**
A: It prevents router collapse, where the router increasingly favors a small subset of "favorite" experts — a self-reinforcing loop, since experts that receive more training data early on become better, making the router favor them even more, starving other experts of training signal and wasting the model's total parameter capacity.

**Q: What is "dark knowledge" in distillation, and why can't a hard ground-truth label convey it?**
A: The relative-similarity information embedded in a teacher model's full soft output distribution (e.g., assigning meaningful probability to near-miss but plausible tokens, not just the single correct answer) — a hard label only specifies the single correct class with no information about which incorrect classes were "almost right," so training only on hard labels discards this richer signal that the teacher's full distribution carries.

**Q: Why does distillation apply a temperature to both teacher and student logits before computing the KL loss?**
A: Raising temperature (T>1) softens/flattens both distributions, spreading visible probability mass across near-miss tokens rather than concentrating it almost entirely on the single top token — this makes the teacher's relative-similarity ("dark knowledge") information more visible and easier for the student to learn from, compared to using the raw, sharply-peaked, unmodified distributions.

---
*End of Module 7 (maximum depth). Next: Module 8 — Evaluation (benchmark suites and their flaws, human eval, LLM-as-judge, hallucination measurement).*
