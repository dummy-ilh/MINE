# Chapter 14: Sequence-Aware Models (GRU4Rec, SASRec, Transformer-Based Recs)

## 1. Intuition

Every architecture through Chapter 13 treats a user's history as an unordered **set** of past interactions (aggregated into a static embedding or feature). But real user behavior has **order and recency** — what a user watched five minutes ago predicts what they want next far more strongly than something from six months ago, and the *sequence* of actions (binge-watching a series, then switching genres) carries information a bag-of-interactions representation destroys.

Sequence-aware models reframe recommendation as **next-item prediction**: given the ordered sequence of items a user has interacted with, predict the next item. This is structurally identical to language modeling (predict the next word given previous words) — which is exactly why the same architectures (RNNs, then Transformers) that revolutionized NLP were directly ported into recsys.

## 2. GRU4Rec — RNNs for Session-Based Recommendation

GRU4Rec (Hidasi et al., 2016) applies a **GRU** (Gated Recurrent Unit, a simplified alternative to LSTM) to model a user's session as a sequence.

**Mechanism**: at each step $t$ in the session, the GRU takes the current item (embedded) and the previous hidden state, and produces an updated hidden state summarizing everything seen so far:

$$h_t = \text{GRU}(x_t, h_{t-1})$$

where $x_t$ = embedding of the item interacted with at step $t$, $h_t$ = hidden state after $t$ steps. The hidden state $h_t$ is then projected to a score over the entire item catalog (via a final linear layer + softmax, or more efficiently, sampled softmax given catalog size) to predict the **next** item in the session:

$$\hat{y}_{t+1} = \text{softmax}(W h_t + b)$$

**Why GRU over vanilla RNN**: the same vanishing-gradient motivation as in general sequence modeling — gating mechanisms let the model retain relevant long-range signal (e.g., "user started this session by searching for a specific brand" staying relevant many steps later) while forgetting irrelevant transient noise, which a vanilla RNN struggles to do over longer sequences due to gradient decay through many BPTT steps.

**Session-based framing specifically**: GRU4Rec was originally designed for **session-based** recommendation (e.g., anonymous e-commerce sessions with no persistent user ID) — the model doesn't require a stable user identity across sessions at all, just the sequence of actions within the current session. This is a distinct use case from the persistent-user two-tower/deep models in Chapters 12-13, and it's a common interview distinction: sequence models can operate in a user-cold-start-immune way, since they only need the current session's behavior, not historical user identity.

## 3. SASRec — Self-Attentive Sequential Recommendation

SASRec (Kang & McAuley, 2018) replaces the RNN with a **self-attention** mechanism (the same core mechanism underlying Transformers), directly addressing two RNN limitations: (1) RNNs process sequentially, meaning training can't fully parallelize across sequence positions, and (2) RNNs still struggle to capture very long-range dependencies despite gating, since information must be compressed through a single fixed-size hidden state passed step by step.

**Mechanism**: given a sequence of item embeddings $[x_1,\ldots,x_n]$ (plus positional embeddings, since self-attention has no inherent notion of order without them), self-attention computes, for each position, a weighted combination of *all* other positions' representations, where the weights are learned based on relevance:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

with $Q, K, V$ (query, key, value) all linear projections of the same input sequence (self-attention). Crucially, **causal masking** is applied — when predicting the next item after position $t$, the model is only allowed to attend to positions $1,\ldots,t$, not future positions, exactly mirroring the causal/autoregressive masking used in language model Transformers, since at inference/prediction time future items obviously don't exist yet.

**Direct advantage over GRU4Rec**: any position can directly attend to any earlier position in a single step (no need to pass information sequentially through many hidden-state updates), making it much easier for the model to pick up on a relevant action from many steps back — e.g., "user looked at hiking boots 20 actions ago" can directly influence "recommend hiking socks now" without that signal having to survive 20 sequential GRU state updates.

## 4. Worked Numerical Example — Self-Attention Over a Short Sequence

Sequence of 3 items (already embedded, $d=2$, ignoring positional embeddings and $Q/K/V$ projection matrices for simplicity — treating the raw embeddings directly as $Q=K=V$):

$x_1 = [1.0, 0.0]$ (action movie), $x_2=[0.9,0.1]$ (another action movie), $x_3=[0.1,0.9]$ (a drama)

Predicting the representation at position 3 (i.e., "what should influence the next-item prediction after this sequence"), with **causal masking** meaning position 3 can attend to positions 1, 2, and 3 (all previous-or-current, since it's the last position, nothing is masked out here).

**Compute attention scores** $QK^T$ for query = $x_3$ against keys $x_1,x_2,x_3$ (using raw dot products, $d=2$ so $\sqrt{d}=1.414$):
$$x_3\cdot x_1 = 0.1(1.0)+0.9(0.0)=0.10$$
$$x_3\cdot x_2 = 0.1(0.9)+0.9(0.1)=0.09+0.09=0.18$$
$$x_3\cdot x_3 = 0.1(0.1)+0.9(0.9)=0.01+0.81=0.82$$

**Scale by $\sqrt{d}$:** $0.10/1.414=0.0707$, $0.18/1.414=0.1273$, $0.82/1.414=0.5799$

**Softmax:**
$e^{0.0707}=1.0733$, $e^{0.1273}=1.1358$, $e^{0.5799}=1.7860$; sum = 3.9951

Weights: $1.0733/3.9951=0.2686$, $1.1358/3.9951=0.2843$, $1.7860/3.9951=0.4471$

**Weighted output (attention over values, $V=[x_1,x_2,x_3]$):**
$$\text{output} = 0.2686\,[1.0,0.0]+0.2843\,[0.9,0.1]+0.4471\,[0.1,0.9]$$
$$=[0.2686,0]+[0.2559,0.0284]+[0.0447,0.4024]$$
$$=[0.2686+0.2559+0.0447,\;0+0.0284+0.4024]=[0.5692,0.4308]$$

Interpretation: the output representation at position 3 is dominated by $x_3$ itself (weight 0.447, since it's most similar to itself by definition), but meaningfully incorporates $x_1$ and $x_2$ (weights 0.269 and 0.284) — the drama at position 3 still "remembers" the two action movies earlier in the session with moderate weight, producing a blended representation. This blended vector — not just the raw $x_3$ embedding — is what feeds into the next-item prediction layer, meaning the model's prediction reflects the *whole causally-visible sequence*, appropriately weighted by relevance, not just the most recent item.

## 5. Transformer-Based Recs — Scaling the Idea

Full Transformer-based recommenders (extending SASRec's single-layer self-attention idea) stack multiple self-attention layers with multi-head attention (parallel attention computations with different learned $Q/K/V$ projections, capturing different types of relationships simultaneously — e.g., one head might specialize in genre-similarity patterns, another in recency patterns) and feed-forward layers between them, following the same general Transformer block structure as the "Attention Is All You Need" architecture. Modern industrial sequential recommenders (e.g., variants used at YouTube, Alibaba's BST — Behavior Sequence Transformer) follow this pattern, often combining the sequence-transformer output with other non-sequential features (user demographics, context) before a final prediction layer — i.e., the sequence tower becomes one input into a broader architecture (potentially fused with two-tower or Wide&Deep-style components), rather than existing as a completely standalone model in production.

## 6. Production Considerations

- Sequential models add real **latency cost** at serving time relative to static embedding lookups — encoding a user's full recent history through even a single self-attention layer is more expensive than a two-tower model's simple embedding lookup (Ch. 12), so sequential encoders are often placed at a stage of the funnel where the added cost is affordable (e.g., as part of a richer final-stage ranking feature, or as the user tower's history encoder within an otherwise-efficient two-tower retrieval system).
- Session length variability is a real engineering concern — very short sessions (1-2 items) give the model little to attend over, while very long histories (months of activity) require truncation or hierarchical approaches (e.g., summarizing older history into a coarser representation while keeping recent history at full resolution) to keep sequence length computationally tractable, since self-attention cost grows quadratically with sequence length.
- These models are naturally suited to capturing **short-term intent shifts** (e.g., a user suddenly browsing for a gift for someone else) that static, long-term-aggregate user embeddings (as in classical two-tower user towers) tend to smooth over or miss entirely — this is a genuine, complementary strength, not a strict replacement for long-term preference modeling.

## 7. Interview Traps

- Treating GRU4Rec and SASRec as interchangeable "sequence models" without being able to state the concrete architectural difference (recurrent hidden-state passing vs. self-attention) and its practical consequence (parallelizability, long-range dependency handling).
- Forgetting causal masking — proposing a self-attention-based next-item predictor that can "see the future" (attend to positions after the prediction point) would be a data leakage bug in training, and interviewers may specifically probe whether you'd remember to mask.
- Assuming sequence models fully replace static user/item embeddings in production — in practice they're typically a component/tower *within* a larger system (e.g., as the user tower's history encoder in two-tower retrieval, Ch. 12), not a wholesale replacement for the entire pipeline.
- Not mentioning the quadratic-in-sequence-length cost of self-attention as a practical constraint requiring truncation/hierarchical handling of long user histories.

## 8. L5-Differentiating Talking Points

- Frame sequence-aware models as solving a **different, complementary** problem from static two-tower/Wide&Deep models — capturing short-term intent shifts and order-sensitivity, not replacing long-term preference modeling — and explicitly note that production systems usually fuse both (e.g., BST-style architectures) rather than choosing one exclusively.
- Explain SASRec's advantage over GRU4Rec specifically via parallelization (training) and direct long-range attention (any-to-any position access in one step) rather than a vague "attention is better" claim.
- Mention causal masking unprompted when describing self-attention for next-item prediction — a small but telling detail that shows you understand this is autoregressive sequence modeling, not just "applying attention to a list of items."
- Note the quadratic cost of self-attention and the resulting need for history truncation/hierarchical summarization as a real production engineering concern, not just a modeling detail — showing awareness of the mismatch between academic architecture design and serving-latency constraints.

## 9. Comprehension Check

1. How does the session-based framing of GRU4Rec make it naturally robust to user cold-start, in a way that persistent-user embedding models are not?
2. What specific architectural change does SASRec make relative to GRU4Rec, and what two concrete benefits does that provide?
3. Why is causal masking necessary when using self-attention for next-item prediction?
4. In the worked attention example, why did position 3's output representation still meaningfully incorporate positions 1 and 2 despite being most similar to itself?
5. Why do production systems typically use sequence-aware models as one component within a larger architecture rather than as a standalone end-to-end recommender?
