# Chapter 4 — Self-Attention (Master Notes, Apple MLE Prep)

> Goal of this doc: derive *why* attention needs three separate projections (not just one), derive the √d_k scaling from actual variance math (not just "it works"), catch the symmetry artifact hiding in the chapter's own worked example, and be able to reproduce the four-step formula on a whiteboard with a tiny example.

---

## 0. One-sentence version

> "Self-attention lets every token compute a weighted average of every other token's information, where the weights are learned relevance scores — computed by projecting each token into a query (what I'm looking for), key (what I offer), and value (what I actually hand over) — so 'cat' stops being just its own static embedding and becomes a blend shaped by whatever else is in the sentence."

---

## 1. Why three separate vectors, not one

### 1.1 The naive first attempt — and why it silently breaks

The simplest possible thing you could try: skip the $W_Q$, $W_K$, $W_V$ projections entirely and just compute $X \cdot X^T$ directly — every token's raw embedding dotted with every other token's raw embedding, used both as "query" and "key."

**This is closer to what the chapter's own worked example actually did** — it set $W_Q = W_K = W_V = $ identity, which means $Q = K = V = X$. Look closely at the worked numbers: `Score[the→cat] = -0.166` and `Score[cat→the] = -0.166` — **identical**. That's not a coincidence, it's a mathematical necessity: if $Q = K$, then $Q K^T$ is a **symmetric matrix** ($\text{score}(i,j) = x_i \cdot x_j = x_j \cdot x_i = \text{score}(j,i)$, since dot product is commutative). The chapter used the identity matrix purely to make the arithmetic easy to follow by hand — but it's worth explicitly flagging that this was a simplification with a real consequence, not just "smaller numbers."

**Why symmetry is actually a problem for language**: consider "sat" and "cat" in "the cat sat." You'd generally want "sat" to attend strongly to "cat" (its subject — verbs need their subjects to disambiguate meaning), but there's no reason "cat" needs to attend equally strongly back to "sat" specifically — "cat" might care more about an adjective describing it ("the **fluffy** cat sat"). Relevance in language is inherently **directional/asymmetric**: A finding B relevant doesn't imply B finds A equally relevant. A symmetric attention matrix structurally cannot represent that asymmetry — it's forced to treat "how much does sat care about cat" and "how much does cat care about sat" as the exact same number.

### 1.2 What separate, learned $W_Q$/$W_K$/$W_V$ actually buys you

With **different, independently learned** projection matrices, $Q \ne K$ in general, so $QK^T$ is no longer forced to be symmetric — the model is now *free* to learn that "sat" strongly queries toward things-that-look-like-subjects while "cat" doesn't symmetrically query back toward things-that-look-like-verbs. This asymmetry is exactly what lets BERT learn the subject-verb attention pattern shown in Section 5 below (`sat → cat: 0.52`), which would be structurally impossible if $Q$ and $K$ were tied to be the same projection.

**The library/search metaphor, made precise**: Query = your search terms (what *this* token is trying to find). Key = the index tags every *other* token is broadcasting (what they claim to offer). Value = the actual content handed over once a match is found. These are three different *purposes* for the same underlying vector, and giving each purpose its own learned linear transformation is what lets the network optimize each role separately rather than forcing one representation to awkwardly serve three different jobs.

**What if we used separate $W_Q, W_K$ but tied $W_V = W_K$ (value = key)?** You'd save some parameters, and some efficient-attention variants do experiment with tying projections. The cost: the vector used to *decide relevance* (key) would be forced to be identical to the vector actually *handed over as content* (value) — but "what makes me a relevant match for your query" and "what information I actually want to contribute if matched" aren't necessarily the same thing to encode efficiently in one shared vector. Keeping all three independent gives maximum representational freedom at the cost of 3x the parameters versus one shared projection.

---

## 2. Computing Q, K, V — dimension bookkeeping

```
W_Q, W_K, W_V:  each [768 × 64]     ← 64 = 768 / 12 heads (Chapter 5 preview)
Q = X · W_Q :   [seq_len × 768] · [768 × 64] = [seq_len × 64]
K = X · W_K :   same shape
V = X · W_V :   same shape
```

**Why 64 and not 768 per head**: this is the single-head slice; BERT actually splits the 768-d model dimension across 12 parallel heads (each getting a 64-d slice) rather than running one attention operation in the full 768-d space. That's next chapter's topic — for now, treat this chapter's math as what happens *inside one head*.

---

## 3. The four-step mechanism — simplified formula, term by term

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Step | Formula piece | Plain-language meaning |
|---|---|---|
| 1 | $QK^T$ | Every token's query dotted with every other token's key → a raw "how relevant is this pairing" score for every pair |
| 2 | $/\sqrt{d_k}$ | Rescale those raw scores down so softmax doesn't saturate (derivation below) |
| 3 | $\text{softmax}(\cdot)$, row-wise | Turn each token's row of scores into a probability distribution over "how much attention to pay to each other token," summing to 1 |
| 4 | $\times V$ | Use those probabilities as mixing weights to blend every token's value vector into one output vector per token |

**The one-sentence mental model**: for every token, look at everyone (step 1), decide how much each of them matters right now (steps 2-3), then take a weighted average of what they're offering (step 4).

---

## 4. Why divide by √d_k — the actual derivation, not just "it works"

The chapter's explanation ("keeps softmax in a regime where gradients are meaningful") is correct but skips *why* the raw scores get large in the first place, and *why specifically* $\sqrt{d_k}$ is the right correction factor rather than some other number.

**The variance argument, simplified**: suppose each component of $Q$ and $K$ is roughly independent, with mean 0 and variance 1 (a reasonable approximation after the network is well-trained/initialized). The dot product $Q \cdot K = \sum_{i=1}^{d_k} q_i k_i$ sums $d_k$ independent terms, each with variance 1 (since $q_i, k_i$ are mean-0, variance-1, their product has expected variance ≈1). **Variance of a sum of independent terms adds up**: variance of the dot product ≈ $d_k \times 1 = d_k$. So the **standard deviation** of the raw dot product grows as $\sqrt{d_k}$ — meaning as you increase the dimension $d_k$ of your query/key vectors, the raw attention scores naturally get *larger in magnitude* just from summing more terms, with no change in how "confident" the model actually is about the match. Dividing by $\sqrt{d_k}$ exactly cancels this dimension-dependent growth, bringing the score's variance back down to approximately 1 regardless of $d_k$.

**Why large raw scores specifically break softmax (the mechanism, concretely)**: softmax is $e^{z_i} / \sum_j e^{z_j}$. When the $z$ values are large in magnitude and spread apart, the largest one's exponential dwarfs the others (exponentials amplify differences), pushing the output toward one-hot — nearly all probability mass on a single token, ≈0 on the rest. The chapter's own illustrative numbers show this: unscaled scores like `[38.4, -4.98, 0.72]` produce a softmax of essentially `[1.00, 0.00, 0.00]`. **Why a one-hot softmax is bad for training, specifically**: the gradient of softmax with respect to its inputs is proportional to $p_i(1-p_i)$ for each output probability $p_i$ — when $p_i$ is very close to 0 or 1, this quantity is near zero. Near-zero gradients mean backprop through that softmax carries almost no learning signal — this is structurally the same *symptom* as the vanishing-gradient problem from Chapter 1's RNN discussion (a near-zero multiplier killing gradient flow), just arising from a different mechanism (softmax saturation vs. repeated recurrent multiplication).

---

## 5. Reading the attention matrix — with the caveat this chapter's example undersells

**The example's identity-matrix weights make every row's "attend most to self" pattern (0.495, 0.498, 0.395 on the diagonal) somewhat misleading as an intuition** — it looks like tokens mostly attend to themselves, which is an artifact of using untrained identity projections, not a general property of attention. **In a genuinely trained BERT**, self-attention weight is often much lower — the whole *point* of attention is to pull in information from elsewhere, and a well-trained head learns to route attention toward the tokens that actually disambiguate meaning, exactly like the `sat → cat: 0.52` subject-verb example the chapter gives for a real trained model.

**What if a head learns to attend almost entirely to itself (like the un-trained example above) even after training?** This does happen for *some* heads in practice — not every one of BERT's 144 total heads (12 layers × 12 heads) discovers an interesting long-range pattern; some heads specialize in fairly local or near-diagonal attention, which is a legitimate, sometimes-useful learned behavior (e.g., a head that mostly reinforces a token's own identity while a different head handles long-range dependency-parsing-like relationships). This is exactly why BERT uses *12 heads per layer* rather than one — different heads are free to specialize differently (Chapter 5).

---

## 6. Complexity — O(n²), and why it's a *memory* problem too, not just compute

The chapter correctly flags $O(n^2)$ compute (score matrix is `[seq_len × seq_len]`, one dot product per pair). **Worth adding explicitly**: this is also an $O(n^2)$ **memory** problem — the full attention weight matrix has to be materialized (stored) to compute the softmax and the subsequent weighted sum, not just computed and discarded. At sequence length 4096, that's ~16.8M floats *per head, per layer* — multiply by 12 heads and 12 layers and the memory cost for attention matrices alone becomes the dominant bottleneck, often before compute does.

**What if we just used a bigger GPU and accepted O(n²)?** This works up to a point (and is exactly what happens for BERT's 512-token budget), but it doesn't scale — doubling sequence length quadruples both compute and memory, so pushing toward document-length or book-length contexts with full O(n²) attention becomes prohibitively expensive well before you run out of other ideas.

**The real fixes, briefly** (useful interview context beyond what the chapter names):
- **Sparse attention** (Longformer, BigBird): each token only attends to a fixed-size local window plus a few global tokens, dropping compute/memory to roughly $O(n)$ or $O(n\sqrt{n})$ — the tradeoff is losing guaranteed direct access between arbitrary far-apart token pairs (though information can still propagate indirectly across layers).
- **Low-rank/linear approximations** (Linformer, Performer): approximate the $O(n^2)$ attention matrix with a lower-rank factorization, trading a small amount of accuracy for large efficiency gains.
- **FlashAttention** (a systems-level, not algorithmic, fix): computes the exact same $O(n^2)$ attention math but restructures the computation to avoid ever materializing the full attention matrix in slow memory, exploiting GPU memory hierarchy — same complexity class, much better real-world speed/memory footprint. Worth naming specifically since it's a common current interview topic distinct from the sparse-attention approaches above.

---

## 7. Design-choice summary table, boosted

| Design choice | Why | What breaks without it |
|---|---|---|
| Separate $W_Q$, $W_K$, $W_V$ (not shared/identity) | Lets relevance be directional/asymmetric — A can find B relevant without B equally finding A relevant | $Q=K$ forces a symmetric attention matrix — provably cannot represent asymmetric relationships like subject→verb attention |
| Scale by $\sqrt{d_k}$ | Cancels the $d_k$-dependent growth in raw dot-product variance, keeping softmax out of its saturated (near-zero-gradient) regime | Large-$d_k$ raw scores push softmax toward one-hot, gradients vanish, learning stalls |
| Softmax row-wise (not some other normalization) | Produces a proper probability distribution per token, so the weighted value sum is a genuine (convex) blend, not an unbounded combination | Without normalization to sum-to-1, output magnitudes would be uncontrolled and harder to compose across stacked layers |
| Full $O(n^2)$ dense attention at 512 tokens | At BERT's modest sequence budget, dense attention is affordable and gives every token guaranteed direct access to every other token | Beyond a few thousand tokens, both compute and memory become prohibitive — motivates sparse/linear/FlashAttention alternatives |

---

## 8. Diagnostics — misconceptions to pre-empt

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "Attention naturally makes tokens mostly attend to themselves, like the worked example shows" | That pattern is an artifact of using identity projection matrices for pedagogical simplicity, not a general property of trained attention | Trained heads often route most of their weight *away* from self, toward whichever tokens actually disambiguate meaning |
| "Q and K are basically the same thing, just named differently" | If $W_Q = W_K$, the resulting attention matrix is provably symmetric — trained BERT uses independent projections specifically so relevance can be directional | The chapter's own numbers (`the→cat` = `cat→the`) are the tell that Q=K was used, not a general attention property |
| "√d_k scaling is just an empirical trick someone found helps" | It's derivable from first principles: dot-product variance grows linearly with dimension, so its standard deviation grows as √d_k, and dividing by √d_k is exactly what's needed to normalize it back to constant scale | It's a principled variance-normalization step, not a heuristic guess |
| "O(n²) is only a compute problem — throw a faster GPU at it" | The full attention matrix must be stored to compute softmax and the weighted sum, so memory scales O(n²) too, and memory bandwidth is often the actual bottleneck before raw FLOPs are | Both compute *and* memory scale quadratically — this is why sparse/linear attention and systems fixes like FlashAttention both matter |
| "Self-attention already produces the model's final contextual understanding of a token" | This chapter shows just one attention operation inside one layer, one head; the output still passes through a feed-forward layer, and the whole thing repeats across 12 stacked layers with 12 parallel heads each | Real contextual richness comes from the full 12-layer × 12-head stack, not a single attention computation |

---

## 9. Q&A practice set (self-test — answers below the line)

**Q1 (easy).** In one sentence, what does each of Q, K, and V represent conceptually?

**Q2 (easy).** Why must the attention weight matrix's rows each sum to exactly 1?

**Q3 (medium).** The chapter's worked example shows `Score[the→cat] == Score[cat→the]`. Why does this happen, and would you expect it in a fully trained BERT model?

**Q4 (medium — calculation).** If $d_k = 16$ instead of 4, and each component of Q and K is approximately mean-0/variance-1, roughly what standard deviation would you expect the raw (unscaled) dot-product scores to have? What's the correct scaling divisor?

**Q5 (medium).** Why does a near-one-hot softmax output cause a training problem, specifically — what breaks mechanically?

**Q6 (hard).** Explain why tying $W_V = W_K$ (using the same projection for keys and values) would be a real, if subtle, representational limitation — not just a parameter-count optimization.

**Q7 (hard).** Why is self-attention's memory cost, not just its compute cost, described as O(n²) — what specifically has to be stored?

**Q8 (hard — spot the bug).** A colleague implements attention but scales by dividing by $d_k$ instead of $\sqrt{d_k}$. For BERT-base ($d_k=64$), what would you expect to observe in training, and why?

---
---

### Answers

**A1.** Query is what a token is searching for in other tokens; Key is what a token advertises about itself to be found by others' queries; Value is the actual content a token contributes if it's selected as relevant.

**A2.** Because the attention weights are used as mixing coefficients for a weighted average of Value vectors in step 4 — for the output to be a proper blend (a convex combination) with a controlled, comparable magnitude across tokens and layers, the weights assigned to any one token's row must form a valid probability distribution, which by definition sums to 1. Softmax guarantees this automatically.

**A3.** It happens because the worked example set $W_Q = W_K = W_V = $ identity, so $Q = K = X$, and the dot product is commutative ($x_i \cdot x_j = x_j \cdot x_i$), which forces the score matrix to be symmetric. In a fully trained BERT, $W_Q$ and $W_K$ are independently learned and generally different matrices, so $Q \ne K$ and there's no mathematical reason for the resulting attention matrix to be symmetric — and empirically it isn't; directional patterns like `sat → cat: 0.52` without an equally strong reverse weight are exactly what asymmetric Q/K projections make possible.

**A4.** Since variance of a sum of $d_k$ independent, variance-1 terms is approximately $d_k$, the standard deviation is $\sqrt{d_k} = \sqrt{16} = 4$. The correct scaling divisor is $\sqrt{d_k} = 4$, which brings the score's standard deviation back down to approximately 1 regardless of the chosen $d_k$.

**A5.** Softmax's gradient with respect to its inputs is proportional to $p_i(1-p_i)$ for each output probability. When the softmax output is near one-hot (one $p_i \approx 1$, the rest $\approx 0$), this quantity is near zero for every output, meaning the gradient signal flowing backward through that softmax during backpropagation is extremely small — the attention weights (and everything upstream of them) receive almost no learning signal from that step, stalling training on that pathway.

**A6.** Key and Value serve genuinely different purposes: Key encodes "what would make me a relevant match for someone's query" (a matching/retrieval signal), while Value encodes "what information I actually want to contribute once matched" (a content signal). Forcing these into the same learned projection means the network can't optimize a vector purely for matchability and a separate vector purely for content richness — it has to find one compromise representation serving both roles, which is a real (if often small in practice) loss of representational flexibility compared to letting them specialize independently.

**A7.** The attention weight matrix (shape `[seq_len × seq_len]`, one entry per token pair) has to be fully computed and held in memory simultaneously in order to run the row-wise softmax (which needs the whole row's values before it can normalize) and then perform the weighted sum with V. Since this matrix's size itself grows as (sequence length)², the memory required to store it during the forward (and backward) pass grows quadratically too — independent of, and often more limiting than, the raw compute (FLOPs) cost.

**A8.** Dividing by $d_k=64$ instead of $\sqrt{d_k}=8$ over-corrects — instead of normalizing the score's standard deviation back to ~1, it shrinks scores by a factor of 8 more than necessary (dividing by 64 vs. the correct 8), crushing them toward zero. Since softmax of near-identical, near-zero inputs produces an almost-uniform distribution (roughly equal attention weight to every token, regardless of actual relevance), you'd likely observe attention that fails to discriminate at all — the model would struggle to learn to focus on genuinely relevant tokens, since the pre-softmax signal has been squashed too flat to distinguish "somewhat relevant" from "highly relevant" pairs. Training would likely be slow to converge or plateau at a worse loss, since the model is effectively starved of a usable relevance signal from attention.

---

## 10. Quick recap card (last-minute review)

- **Core idea**: every token computes a weighted average of every other token's Value vector, where weights = learned relevance (Query·Key match).
- **Why 3 separate projections**: tying Q=K forces a *symmetric* attention matrix (provable from dot-product commutativity) — real language relationships are directional (subject→verb ≠ verb→subject in strength), so independent $W_Q, W_K, W_V$ are what make asymmetric attention patterns possible.
- **The √d_k scaling is derived, not tuned**: dot-product variance grows linearly with dimension $d_k$ → standard deviation grows as $\sqrt{d_k}$ → dividing by $\sqrt{d_k}$ normalizes it back to constant scale, keeping softmax out of its near-zero-gradient saturated regime.
- **The formula, one line**: $\text{softmax}(QK^T/\sqrt{d_k})\,V$ — scores, scale, weights, blend.
- **O(n²) is compute *and* memory**: the full `[seq_len × seq_len]` matrix must be materialized to run softmax — this is why long-context models need sparse/linear attention or systems tricks like FlashAttention.
- **One head is not the whole story**: a single attention operation is only one slice of one layer; BERT's real richness comes from 12 heads × 12 layers, each free to specialize (next chapter).

*(Chapter 5 — Multi-Head Attention — picks up here: why running 12 of these attention operations in parallel, each in its own 64-d subspace, captures far more than one 768-d head could alone.)*
