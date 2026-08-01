# Chapter 5 — Multi-Head Attention (Master Notes, Apple MLE Prep)

> Goal of this doc: explain from memory why one head can't do the job, prove the surprising fact that multi-head costs *zero extra parameters* over a single full-width head (until W_O), defend concatenation over averaging, and connect head-pruning research to why this design is robust to redundancy.

---

## 0. One-sentence version

> "One attention head has to compress every type of relationship — syntax, co-reference, causality, semantics — into a single competing view of the sentence, so BERT instead runs 12 smaller attention operations in parallel, each free to specialize in a different relationship type, then concatenates and mixes their outputs back into one unified representation."

---

## 1. Why one head genuinely can't do it — not just "more is better"

### 1.1 The competition problem, stated mechanically

A single attention head produces exactly **one** `[seq_len × seq_len]` weight matrix per token — one number per pair, representing "how much should I blend in from this other token." If "it" in the trophy/suitcase sentence needs to simultaneously signal *strong* co-reference weight toward "trophy" **and** the token "was" needs *strong* causal-relationship weight toward "big," these are two entirely different softmax distributions being demanded of the *same* underlying Q/K/V projection. A single head's $W_Q, W_K, W_V$ have to find one compromise projection that serves every relationship type at once — gradient descent will push it toward whatever mixture minimizes loss on average, which generally means **no single relationship type gets a strong, clean signal**; they all get diluted.

**What if we just made one head's dimension bigger — say, 768-d instead of 64-d — hoping it has enough room to encode everything?** This doesn't fix the actual problem: the bottleneck isn't dimensionality, it's that there's still only **one softmax distribution per token**. Even with more dimensions to store richer Value content, you're still forced to produce a single weighting over all other tokens — you can't have that one weighting simultaneously "60% toward trophy for co-reference reasons" and "70% toward big for causal reasons" in one distribution that sums to 1. The fix has to be *architectural* (more independent weightings), not just *dimensional* (more room per weighting).

### 1.2 What multiple heads actually buys you

Running $h$ independent attention operations gives you $h$ **independent softmax distributions per token**, each with its own learned $W_Q, W_K, W_V$ — so head 3 can freely learn "attend heavily to the co-referent noun" while head 7 simultaneously learns "attend heavily to the syntactic subject," with neither competing against the other for the same distribution's probability mass. **Nobody assigns these roles** — Section 5.5's finding that specific heads consistently learn syntax, co-reference, positional patterns, etc. emerges purely because specialization reduces loss during pre-training, and gradient descent finds it.

---

## 2. The parameter-count surprise — multi-head isn't "12x the projection cost"

This is worth stating explicitly because it's a genuinely counter-intuitive and interview-relevant fact the original chapter's numbers already contain but don't call out directly.

**Single big head (hypothetical, 768-d Q/K/V)**: $W_Q$ alone would be `[768 × 768]` = **589,824 parameters**.

**Multi-head (12 heads, 64-d each)**: each head's $W_{Qi}$ is `[768 × 64]` = 49,152 parameters. Twelve of them: $12 \times 49,152 = $ **589,824 parameters** — **exactly the same total**.

**Why this is exactly true, not approximately**: concatenating 12 separate `[768 × 64]` matrices along their output dimension produces one `[768 × 768]` matrix — splitting one big projection into 12 heads and running them "in parallel" is mathematically just **partitioning the output columns of one big matrix multiply into 12 groups and doing the softmax separately within each group**. You're not adding projection parameters by going multi-head; you're changing *how the attention computation is structured* (independent softmaxes per group of columns) while keeping the *total* Q/K/V projection parameter count identical.

**Where the real extra cost comes from**: $W_O$, the `[768 × 768]` final mixing matrix — **589,824 additional parameters** that a naive single-head design wouldn't need (a single head's output is already 768-d and could in principle skip a mixing step, though in practice most single-head designs would still want some output projection too). The headline insight for an interview: **multi-head attention's real innovation isn't spending more parameters on projections — it's restructuring one projection into independent per-group softmaxes, at the modest added cost of one mixing matrix at the end.**

---

## 3. Concatenate, don't average — why this matters

The original chapter states heads are concatenated but doesn't defend *why not average instead* — worth making explicit, since it's a natural "why not the simpler option" question.

**What if we averaged the 12 heads' outputs instead of concatenating?** Averaging would force every head's output to be blended with **equal, fixed weight** immediately, in the same 64-d-equivalent space, before the network has any chance to learn which heads matter more for which purpose. Concatenation instead **preserves every head's output as a fully separate, uncollapsed block** — head 3's syntactic signal sits in its own 64 dimensions, completely undiluted by head 7's co-reference signal — and only *then* lets $W_O$ (a fully learned matrix) decide, per output dimension, how much of which head to draw from. Averaging is a fixed, hand-picked combination rule (equal weight, immediately collapsed); concatenation + learned $W_O$ is a **fully general, learnable combination rule** that could in principle even reproduce averaging as a special case (if that's what training found optimal) — but isn't restricted to it.

**Why $W_O$ specifically matters, mechanically**: without it, the concatenated vector is just 12 independent chunks sitting side by side with **zero interaction between them** — dimension 65 (start of head 2's block) has no learned relationship to dimension 1 (start of head 1's block) unless something explicitly mixes them. $W_O$ is exactly that mixing step: it's a full `[768×768]` matrix, so every output dimension can be a learned combination of *every* input dimension across *all* heads — this is what lets later layers receive a genuinely unified representation rather than 12 unrelated sub-vectors bolted together.

---

## 4. What if we picked a different number of heads?

The original chapter names 12 heads at 64-d each but doesn't explore the tradeoff space — worth covering since "why 12, why not more/fewer" is a natural follow-up question.

**What if we used only 4 heads (192-d each)?** Fewer, wider subspaces — each head has more room to encode richer content per relationship type, but fewer independent "slots" for distinct relationship types to specialize into. You'd likely see more competition/blending within each head again, partially reintroducing the single-head problem, just to a lesser degree.

**What if we used 96 heads (8-d each)?** Many, very narrow subspaces — plenty of independent softmax "slots" for specialization, but each head's Q/K/V vectors are so low-dimensional they may not have enough representational room to compute a meaningful relevance signal at all (an 8-dimensional key vector can only distinguish so many distinct "types" of relevant content). There's also compute/memory overhead in managing many small parallel operations. This is a real empirical tradeoff, not a free lunch in either direction — 12 heads × 64-d was an empirically-tuned choice for BERT-base's scale (larger models like BERT-large use 16 heads × 64-d, keeping the *per-head* width constant while adding more heads as the model widens — suggesting 64-d per head specifically was found to be a robust "enough room per head" sweet spot, and scaling is mostly done by adding more heads/layers, not shrinking per-head width).

---

## 5. Why redundant heads exist, and why pruning them barely hurts

**The original chapter's finding (30-40% of heads prunable with minimal accuracy loss) is worth explaining mechanically, not just citing**: nothing in training explicitly prevents two different heads from independently converging to similar, overlapping specializations — if a particular relationship pattern (say, "attend to the immediately preceding token") is broadly useful, gradient descent has no reason to prevent *multiple* heads from partially learning it, especially with 144 total heads (12 layers × 12 heads) providing ample redundant capacity. This is conceptually similar to over-parameterized networks generally having redundant capacity relative to what's strictly needed to fit the training objective — multi-head attention isn't special in having this property, it's just easy to *measure* here because you can literally zero out one head's contribution and check what breaks.

**What if we designed BERT with far fewer heads from the start, assuming this redundancy means most heads are "wasted"?** This doesn't straightforwardly work — the redundancy is discovered only *after* training with the full head count; there's no guarantee that training a smaller-head-count model from scratch would discover the *same* set of useful specializations as a subset of a larger model's heads (over-parameterization can help the model actually *find* good solutions during training, even if fewer parameters would have sufficed to represent the final solution). This is why head-pruning is typically done as a **post-training compression step**, not as a training-time architecture choice.

---

## 6. Full pipeline, boosted with the "why" at each step

```
Input X [seq_len × 768]
    ↓
Split into 12 independent projections (mathematically: partition one [768×768]
projection's output columns into 12 groups of 64)              ← same total Q/K/V params as one big head
    ↓
Each head runs full attention (Chapter 4) independently         ← 12 separate softmax distributions, no competition between them
    ↓
Concatenate all 12 outputs → [seq_len × 768]                    ← preserves each head's signal undiluted
    ↓
Multiply by learned W_O [768 × 768]                              ← the ONLY new parameters vs. a single big head; mixes across heads
    ↓
Output: [seq_len × 768] — same shape as input, richer content
```

---

## 7. Design-choice summary table, boosted

| Design choice | Why | What breaks without it |
|---|---|---|
| Multiple heads (not one wide head) | Each head gets its own independent softmax distribution — no forced competition between relationship types in one weighting | One head must compromise across all relationship types simultaneously in a single distribution, diluting every signal |
| 12 heads × 64-d (not fewer/wider or more/narrower) | Empirically tuned balance: enough independent "slots" for specialization, enough dimensionality per slot for a meaningful relevance signal | Too few heads → residual competition; too many/narrow → each head too weak to encode useful relevance signal |
| Concatenate heads (not average) | Preserves every head's output undiluted, in its own dimensions, before any combination is learned | Averaging forces a fixed, immediate, equal-weight blend before the network can learn which heads matter for what |
| $W_O$ final mixing matrix | Lets every output dimension draw from a learned combination of *all* heads, not just sit as disconnected blocks | Concatenated output stays as 12 unrelated chunks with zero cross-head interaction |
| Multi-head costs no extra Q/K/V params vs. one big head | Splitting one [768×768] projection into 12 [768×64] groups is a re-partitioning, not new parameters | N/A — this is a property, not a design tradeoff; worth knowing to correctly reason about where BERT's parameters actually go |

---

## 8. Diagnostics — misconceptions to pre-empt

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "Multi-head attention costs roughly 12x the parameters of single-head attention" | The total Q/K/V projection parameter count is identical whether you use one 768-d head or twelve 64-d heads — it's a repartitioning of the same projection, not 12 separate full-size ones | The only genuinely new parameters vs. a single-head design are in $W_O$ (589,824 params) |
| "Averaging the heads' outputs would work about as well as concatenating" | Averaging forces an immediate, fixed, equal-weight blend before any learning can decide which heads matter for which purpose; concatenation + learned $W_O$ is a strictly more general combination rule | Concatenation preserves each head's undiluted signal; $W_O$ learns the mixing, rather than having it hand-fixed to "equal average" |
| "More heads is always better, since more specialization slots can't hurt" | Narrower per-head dimensionality eventually limits how much relevance signal a single head's Q/K/V can meaningfully encode; there's a real tradeoff, not a monotonic improvement | 12×64 (or 16×64 for BERT-large) reflects an empirically-tuned balance, not "maximize head count" |
| "Redundant/prunable heads mean the architecture is badly designed" | Redundancy is a common byproduct of over-parameterized training generally, and appears to aid the *training process itself* in finding good solutions, even if fewer heads would suffice to represent the final result | Post-training pruning being possible is a property of over-parameterized networks broadly, not evidence of a flawed initial design |
| "Each head attends to a completely different token every time, like a hard switch" | Attention weights are soft, continuous probability distributions (softmax output) — every head still computes some (possibly small) weight toward every token, not a hard, discrete choice | Specialization means *which* tokens get *high* weight tends to differ meaningfully across heads, not that any head ignores most tokens entirely |

---

## 9. Q&A practice set (self-test — answers below the line)

**Q1 (easy).** In one sentence, why can't a single attention head simultaneously give strong weight to both a co-reference relationship and a syntactic relationship for the same token?

**Q2 (easy).** What is the shape of $W_O$, and what does it do that concatenation alone does not?

**Q3 (medium — calculation).** If BERT-large uses 16 heads with $d_{model}=1024$, what is $d_k$ per head, assuming the same per-head-width convention as BERT-base?

**Q4 (medium).** Why does splitting one 768-d projection into 12 heads of 64-d each not increase the total number of Q/K/V parameters, compared to a hypothetical single 768-d head?

**Q5 (medium).** Why does concatenation preserve more information into the next step than averaging would, specifically?

**Q6 (hard).** Explain the mechanical reason multi-head attention ends up with redundant/prunable heads, and why this doesn't necessarily indicate the architecture is over-sized in a wasteful way.

**Q7 (hard).** If you doubled the number of heads to 24 while keeping $d_{model}=768$ fixed, what specifically would you expect to degrade, and why?

**Q8 (hard — spot the bug).** An engineer implements multi-head attention but forgets $W_O$ entirely, just returning the raw concatenated head outputs as the block's output. Describe what capability is specifically lost, using the co-reference/syntax example from Section 1.

---
---

### Answers

**A1.** A single head produces exactly one softmax distribution per token (one set of attention weights that must sum to 1 across all other tokens), so it cannot simultaneously place strong probability mass on a co-reference target and a separate, different syntactic target within that same one distribution — the two demands compete for the same fixed budget of attention weight.

**A2.** $W_O$ has shape `[768 × 768]`. Concatenation alone just places all 12 heads' outputs side by side as independent, non-interacting 64-d blocks; $W_O$ is a fully learned linear mixing step that lets every output dimension be a combination of information from *all* heads, giving the network a way to actually integrate the heads' separate signals into one coherent representation rather than leaving them as disconnected chunks.

**A3.** $d_k = d_{model} / h = 1024 / 16 = 64$ — the same per-head width as BERT-base. This matches the note in Section 4 that scaling to larger BERT variants is typically done by adding more heads/layers while keeping per-head width around 64, rather than shrinking it.

**A4.** Concatenating 12 separate `[768×64]` projection matrices along their output dimension is mathematically equivalent to one `[768×768]` matrix — going multi-head is a re-partitioning of the *columns* of that one big projection into 12 independently-softmaxed groups, not the creation of 12 separate full-sized projections. The parameter count for Q (or K, or V) is $768 \times 64 \times 12 = 768 \times 768$ either way — identical.

**A5.** Averaging collapses all 12 heads into one fixed, equal-weighted blend immediately, before any subsequent computation can decide how much to trust or emphasize any particular head's signal — information from underrepresented but important heads could get washed out by the averaging itself. Concatenation keeps every head's 64-d output as a fully distinct, undiluted block; nothing is lost or blended at this step, and the *learned* $W_O$ matrix — rather than a fixed averaging rule — decides how to combine them, which is a strictly more expressive combination rule (it could in principle learn to approximate averaging, but isn't limited to it).

**A6.** With 144 total heads across BERT's 12 layers, there's no architectural mechanism preventing multiple heads from independently discovering similar, overlapping useful patterns (e.g., several heads all partially learning "attend to the previous token") — gradient descent has no built-in pressure toward maximal diversity across heads, just toward minimizing loss, and with ample redundant capacity available, some duplication is a natural byproduct. This doesn't necessarily mean the architecture is wastefully over-sized in a bad way: over-parameterization (having more capacity than strictly needed for the final solution) is widely observed to help the *training process itself* find good solutions more reliably, even when a smaller network could in principle represent a similar final function — the redundancy may be functionally useful during training even though it's prunable after training.

**A7.** Doubling heads to 24 while keeping $d_{model}=768$ fixed means $d_k = 768/24 = 32$ per head — each head's Q/K/V vectors get half as wide as BERT-base's 64-d heads. You'd expect each individual head to have less representational room to encode a meaningful, discriminative relevance signal (a 32-dimensional key vector has less capacity to distinguish many distinct "types" of relevant content than a 64-dimensional one), which could make individual heads' attention patterns noisier or less specialized, even though you now have more of them. This is exactly the "many, very narrow subspaces" tradeoff described in Section 4 — more independent softmax slots, but each with less room to compute something meaningful.

**A8.** Without $W_O$, the block's output is just the 12 heads' outputs sitting side by side as independent 64-d chunks with zero interaction between them — head 3's co-reference signal (say, occupying dimensions 129-192) and head 7's syntactic signal (dimensions 385-448) never get combined into anything unified; they remain two separate, uncombined pieces of information in the same output vector. Any downstream computation (the next Transformer layer, or a task head) that needs to jointly reason about *both* signals together — e.g., "resolve what 'it' refers to AND use that to correctly attach the causal clause" — has no learned mechanism to relate those two chunks to each other; it would have to work with them as separately-addressable but uncombined slices, which defeats much of the purpose of having specialized heads whose insights are meant to jointly inform a unified representation for later layers.

---

## 10. Quick recap card (last-minute review)

- **The problem one head can't solve**: a single head produces one softmax distribution per token — it can't simultaneously give strong, undiluted weight to multiple different relationship types (co-reference, syntax, causality) in that one distribution.
- **12 heads × 64-d**: each gets its own independent softmax, free to specialize, with no explicit assignment — specialization emerges purely from gradient descent finding it useful.
- **Surprising parameter fact**: splitting one big 768-d projection into 12×64-d heads costs *zero* extra Q/K/V parameters — it's the same total projection, re-partitioned into independently-softmaxed groups. The only genuinely new cost is $W_O$ (`[768×768]`, ~590K params).
- **Concatenate, don't average**: concatenation preserves every head's signal undiluted; $W_O$ then learns the mixing rule, rather than having it hand-fixed to equal-weighted averaging.
- **Redundant/prunable heads are expected**: over-parameterized capacity helps training find good solutions even if not all of it is strictly needed afterward — this is why 30-40% of heads can often be pruned post-training with minimal accuracy loss.
- **12×64 is a tuned tradeoff, not a law**: fewer/wider heads → residual competition; more/narrower heads → each head too weak to encode a useful relevance signal.

*(Chapters 6 and 7 pick up here: the residual connection and LayerNorm wrapping this attention output, and the Feed-Forward Network that completes one Transformer block.)*
