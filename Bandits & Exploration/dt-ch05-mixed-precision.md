# Chapter 5 — Mixed Precision Training

*(Plain language first, with the bit-level detail worked through concretely since it's genuinely useful for understanding what follows.)*

---

## 5.1 What a "32-bit float" actually is, in plain terms

Before comparing fp32/fp16/bf16, let's get one key idea straight: every floating-point number format splits its bits into two jobs — some bits for the **exponent** (how big or small the number can be — its *range*) and some bits for the **mantissa** (how many meaningful digits it can represent — its *precision*). This exponent-vs-mantissa split is the entire story of this chapter — everything below is just different ways of dividing up a fixed number of bits between these two jobs.

---

## 5.2 The three formats, compared

| Format | Total bits | Exponent bits (range) | Mantissa bits (precision) |
|---|---|---|---|
| fp32 (standard "single precision") | 32 | 8 | 23 |
| fp16 (standard "half precision") | 16 | 5 | 10 |
| bf16 ("bfloat16") | 16 | 8 | 7 |

**The single most important thing to notice in this table**: fp16 and bf16 are both 16 bits total (half the memory of fp32), but they split those 16 bits very differently. **fp16 sacrifices range** (only 5 exponent bits, versus fp32's 8) **to keep more precision** (10 mantissa bits). **bf16 does the opposite** — it keeps fp32's *exact same* 8 exponent bits (same range as fp32!), sacrificing precision instead (only 7 mantissa bits). This single design difference explains almost everything else in this chapter.

---

## 5.3 Why fewer exponent bits (fp16) causes real training problems

**What "range" actually controls, concretely**: with only 5 exponent bits, fp16 can only represent numbers roughly between $6 \times 10^{-8}$ and $65{,}504$ — a much narrower range than fp32's roughly $10^{-38}$ to $10^{38}$. Any number smaller than about $6\times10^{-8}$ in fp16 simply becomes **zero** (this is called **underflow**) — the information is completely lost, not just rounded imprecisely.

**Why this matters for training, specifically**: gradients, especially deep in a network during backpropagation, are often **very small numbers** — it's completely normal for gradient values to be far smaller than 1, sometimes many orders of magnitude smaller. In fp16, many of these small gradient values can underflow straight to zero, meaning **those weights simply stop receiving any update at all** — not a small, noisy update, but literally zero, silently, without any error being thrown. This can cause training to stall or diverge in ways that are genuinely confusing to debug if you don't know to look for it specifically.

---

## 5.4 The fix: Loss Scaling

**The idea, in plain words**: if small gradients are underflowing to zero because they're too close to fp16's smallest representable range, **just make them bigger before they get there** — multiply the loss by some large constant (say, 1024) *before* backpropagating. Since gradients scale proportionally with the loss, every gradient computed during that backward pass also gets multiplied by roughly that same 1024 — pushing previously-underflowing small values back up into fp16's safely-representable range.

**Then, right before the optimizer actually uses the gradients to update the weights, divide back down by that same constant (1024)** — undoing the artificial scaling, so the actual weight update is mathematically the same as if you'd never scaled anything, you've just avoided losing information to underflow along the way.

### A simple worked example

Say a particular gradient value, computed in true (unscaled) terms, would be $0.00000004$ (that's $4\times10^{-8}$) — dangerously close to fp16's roughly $6\times10^{-8}$ underflow floor, and might round to zero.

- **Scale the loss by 1024 before backprop**: the computed gradient in the backward pass becomes roughly $0.00000004 \times 1024 \approx 0.000041$ (that's $4.1\times10^{-5}$) — comfortably within fp16's representable range, no underflow.
- **Divide by 1024 after backprop, before the optimizer step**: $0.000041 / 1024 \approx 0.00000004$ — back to the original, correct value, safely recovered without ever having been lost to underflow along the way.

**Why the scaling constant itself needs to be chosen carefully**: too small a scaling factor doesn't push small gradients far enough out of the underflow danger zone; too large a scaling factor risks pushing some gradients *up* past fp16's maximum representable value instead (**overflow**, which produces "infinity" or "NaN" values — a different failure mode, equally bad). **Dynamic loss scaling** (used by essentially every real mixed-precision training framework) automatically adjusts this constant during training: it periodically tries increasing the scale, and if it ever detects an overflow (a NaN/Inf gradient), it immediately backs the scale down and skips that particular update — an elegant, self-correcting solution that avoids needing to hand-tune the scaling constant for every model.

---

## 5.5 Why bf16 mostly sidesteps this whole problem

Recall from Section 5.2: bf16 has the **exact same 8 exponent bits as fp32** — meaning bf16 has **exactly the same representable range as fp32**, just with less precision within that range. Since the underflow problem in Section 5.3 was specifically caused by fp16's *narrow range* (only 5 exponent bits), and bf16 doesn't share that narrow range at all, **small gradients essentially never underflow in bf16 the way they can in fp16** — loss scaling becomes largely unnecessary.

**The tradeoff bf16 makes instead**: with only 7 mantissa bits (versus fp16's 10), bf16 numbers are less *precise* within whatever range they're representing — two very close-together numbers might get rounded to the same bf16 value, where fp16 (or fp32) could have told them apart. In practice, this precision loss turns out to matter much less for deep learning training than the underflow problem it avoids — which is exactly why **many newer large-model training setups prefer bf16 outright**, treating it as a more robust default that doesn't require loss scaling's extra machinery, rather than fp16's precision-over-range tradeoff.

**Simple interview-ready summary sentence**: *"fp16 keeps more precision but sacrifices range, which causes small-gradient underflow that loss scaling has to work around; bf16 keeps fp32's full range and sacrifices precision instead, which mostly avoids the underflow problem in the first place, at the cost of somewhat less precise individual number representations."*

---

## 5.6 Automatic Mixed Precision (AMP) — what it actually does under the hood

**The core idea**: not every single operation in a neural network benefits equally from running in reduced precision. AMP frameworks (like PyTorch's `torch.cuda.amp`) maintain an internal list of which operations are considered **safe to run in fp16/bf16** (typically large matrix multiplications, where the speed and memory benefit is large and the precision loss tends to be tolerable) versus which operations should **stay in fp32** (typically operations known to be numerically sensitive, like certain reductions/normalizations) — automatically casting tensors to the appropriate precision for each operation, and automatically handling loss scaling (Section 5.4) behind the scenes if fp16 is being used.

**Why this matters practically**: you generally don't need to manually decide, operation by operation, what precision to use — AMP's job is specifically to apply the "use reduced precision where it's safe, full precision where it's numerically risky" policy automatically, which is exactly why mixed-precision training is called *mixed* — it's not "everything in fp16/bf16," it's a deliberate, automatically-managed mix of precisions across different parts of the computation.

---

## 5.7 Production considerations

- **Mixed precision typically provides both a memory savings and a real speed increase** — memory, because storing activations and (in fp16's case) often gradients in half the bytes directly cuts memory usage; speed, because modern GPU tensor cores are specifically built to perform fp16/bf16 matrix multiplications significantly faster than fp32 ones — this dual benefit (not just memory, but real throughput) is a large part of why mixed precision is close to universal in practice, not a niche optimization.
- **bf16 has become the increasingly preferred default for large language model training specifically**, precisely because of the underflow-avoidance argument in Section 5.5 — worth knowing this as a concrete, current industry trend, not just an abstract format comparison.
- **The optimizer states themselves are often still kept in fp32** even during otherwise-mixed-precision training (an fp32 "master copy" of the weights, updated using fp16/bf16-computed gradients) — this is a real, common practical detail: precision-sensitive accumulation over many training steps benefits from fp32's precision even when the forward/backward computation itself runs in reduced precision.

---

## 5.8 Interview traps

- **Confusing fp16 and bf16's tradeoffs** — this is the single most common, most checkable mistake in this whole chapter. The correct, precise statement: fp16 sacrifices *range* for *precision*; bf16 sacrifices *precision* for *range* (keeping fp32's range). Getting this backward is an easy, serious error to avoid.
- **Describing loss scaling as simply "making gradients bigger" without explaining why** (underflow specifically, not general numerical noise) — a strong answer explicitly connects loss scaling to fp16's narrow exponent range and the underflow failure mode from Section 5.3.
- **Not knowing that optimizer states are often kept in fp32 even during mixed-precision training** (Section 5.7) — a candidate who assumes "mixed precision" means "everything is fp16/bf16, full stop" is missing a real, common implementation detail.

---

## 5.9 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describes fp32 vs. fp16 vs. bf16 at a high level, and knows loss scaling exists to address underflow.
- **L6 bar**:
  - Correctly and precisely states the exponent-bits-vs-mantissa-bits tradeoff (Section 5.2) and can explain *why* this specific bit-allocation difference is the root cause of every downstream behavior difference between fp16 and bf16.
  - Walks through a concrete loss-scaling worked example (like Section 5.4) showing an actual gradient value crossing the underflow threshold and being recovered, rather than describing loss scaling only in the abstract.
  - Proactively mentions the fp32 "master weights" detail (Section 5.7) and the current industry trend toward bf16 for large-model training, showing awareness of both a fine implementation detail and current practice, not just textbook theory.

---

## 5.10 Comprehension checks

1. In plain words, what's the difference between what the exponent bits and the mantissa bits each control in a floating-point number?
2. Why does fp16 specifically (not bf16) risk gradient underflow during training?
3. Walk through the loss-scaling worked example from Section 5.4 in your own words — why does scaling the loss up, then dividing the gradient back down, recover the correct value without losing precision to underflow?
4. Why does bf16 mostly avoid the underflow problem that fp16 faces, and what does bf16 give up in exchange?
5. What does Automatic Mixed Precision (AMP) actually decide, operation by operation, and why isn't "mixed precision" the same as "everything runs in fp16/bf16"?

---

*Next: Chapter 6 — Tensor Parallelism, opening Phase 2 with the mechanics of splitting a single layer's matrix multiplication across devices.*
