# Chapter 15: Graph Neural Networks for Recsys (PinSage-Style)

## 1. Intuition

Every model so far treats the user-item interaction matrix as a table of independent rows/pairs. But that matrix is really a **bipartite graph**: users and items are nodes, and an interaction is an edge. Two items are implicitly connected if they share many co-interacting users, even if no direct feature says so — that's exactly the kind of structural, multi-hop signal classical MF/two-tower models never explicitly exploit beyond what falls out of the embedding training itself.

Graph Neural Networks (GNNs) make this graph structure the explicit computational substrate: an item's embedding is built by **aggregating information from its neighbors in the graph** (the users who interacted with it, and transitively, the other items those users interacted with), not just from its own standalone features. This lets embeddings capture multi-hop collaborative signal directly and explicitly, rather than hoping it emerges implicitly from a dot-product loss.

## 2. The Graph Setup

Bipartite graph $G=(U, I, E)$: users $U$, items $I$, edges $E$ = observed interactions. In PinSage's actual application (Pinterest's production recommender for "pins"), the graph is item-item (pins and boards they belong to), but the same core mechanism generalizes to user-item bipartite graphs.

**Core GNN idea — neighborhood aggregation (message passing)**: an item's embedding at layer $l$ is computed from its own embedding at layer $l-1$ plus an aggregation of its neighbors' embeddings at layer $l-1$:

$$h_i^{(l)} = \text{UPDATE}\Big(h_i^{(l-1)},\; \text{AGGREGATE}\big(\{h_j^{(l-1)} : j \in N(i)\}\big)\Big)$$

where $N(i)$ = neighbors of item $i$ in the graph. Stacking $L$ layers means each item's final embedding incorporates information from up to $L$ hops away — a 2-layer GNN lets an item's embedding reflect not just its direct co-interactors, but its **neighbors' neighbors** too (e.g., "items that people who liked items-similar-to-this also liked"), a genuinely multi-hop collaborative signal that a single-layer embedding lookup (MF, two-tower) cannot directly represent.

## 3. PinSage's Specific Aggregation — Importance Pooling

Rather than uniformly averaging all neighbors (which treats a barely-related neighbor the same as a strongly-related one), PinSage uses **random-walk-based importance weighting**: simulate short random walks starting from item $i$, and weight each neighbor's contribution by how frequently it's visited during these walks (an efficient proxy for "how structurally important/close is this neighbor," related to personalized PageRank).

$$h_{N(i)} = \sum_{j \in N(i)} \alpha_{ij}\, h_j^{(l-1)}, \quad \alpha_{ij} \propto \text{visit count of } j \text{ in random walks from } i$$

This means a neighbor visited frequently in random walks from $i$ (structurally "close," in a graph-connectivity sense, not just directly-adjacent) contributes more to $i$'s updated embedding than a neighbor that's only weakly/rarely connected — a graph-native analogue to attention weighting (Ch. 14), but derived from graph topology rather than learned feature similarity.

## 4. Why Sample Neighbors Rather Than Use All of Them

At Pinterest/production scale, some nodes (highly popular items) can have **millions** of neighbors — aggregating over all of them at every layer, for every node, at every training step, is computationally infeasible. PinSage's solution: **fixed-size neighbor sampling** — for each node, sample a fixed number of neighbors (e.g., top-K by random-walk importance weight) rather than using the full neighbor set. This keeps the computational cost per node constant regardless of how many actual neighbors a popular node has, at the cost of not using 100% of the available graph signal per node per step — a deliberate, necessary trade-off for tractability, directly analogous to Chapter 9's negative sampling trade-off (use a manageable subset rather than the full, intractable set).

## 5. Worked Numerical Example — 2-Hop Aggregation

Simplified graph: Item A is connected to Users 1, 2. User 1 is also connected to Item B. User 2 is also connected to Item C. (So A is 1-hop from Users 1,2, and 2-hop from Items B, C.)

Layer-0 embeddings (raw features, $d=2$): $h_A^{(0)}=[1.0,0.0]$, $h_1^{(0)}=[0.8,0.2]$, $h_2^{(0)}=[0.6,0.4]$, $h_B^{(0)}=[0.9,0.1]$, $h_C^{(0)}=[0.3,0.7]$

**Layer 1 — update User 1 and User 2's embeddings** by aggregating their neighbors (assume simple mean aggregation for this illustration, and User 1's only item-neighbor is A and B; User 2's only item-neighbor is A and C):

$$h_1^{(1)} = \text{mean}(h_A^{(0)}, h_B^{(0)}) = \text{mean}([1.0,0.0],[0.9,0.1]) = [0.95, 0.05]$$
$$h_2^{(1)} = \text{mean}(h_A^{(0)}, h_C^{(0)}) = \text{mean}([1.0,0.0],[0.3,0.7]) = [0.65, 0.35]$$

**Layer 2 — update Item A's embedding** by aggregating User 1 and User 2's **layer-1** embeddings (which now already encode B and C's information transitively):

$$h_A^{(2)} = \text{mean}(h_1^{(1)}, h_2^{(1)}) = \text{mean}([0.95,0.05],[0.65,0.35]) = [0.80, 0.20]$$

Compare to $h_A^{(0)}=[1.0,0.0]$: the layer-2 embedding has shifted meaningfully toward incorporating the direction of $h_B$ and $h_C$ (both of which had a non-trivial second coordinate, especially $h_C=[0.3,0.7]$), even though A has **no direct edge** to B or C at all — this is the concrete mechanism by which 2-hop message passing lets item A's final embedding reflect "items liked by people who also liked A," a genuinely multi-hop collaborative signal, purely through graph structure, without B or C ever being a direct feature input to A.

## 6. Comparison to Two-Tower (Chapter 12)

GNN-based embeddings and two-tower embeddings aren't mutually exclusive — PinSage-style GNN embeddings are commonly used as a **feature input** into a broader retrieval system (e.g., feeding GNN-derived item embeddings into an otherwise-standard two-tower item tower, or using them directly as the precomputed item embeddings for ANN retrieval, Ch. 17), rather than as a wholesale replacement for the two-tower serving pattern. The key practical distinction: a standard two-tower item tower computes an item's embedding purely from that item's own features; a GNN-derived item embedding is explicitly a function of the graph neighborhood, capturing collaborative structure the two-tower's independent per-item computation cannot see on its own.

## 7. Production Considerations

- GNN embeddings are typically computed **offline in a batch pipeline** (not real-time per-request) — the graph structure and multi-hop aggregation are recomputed periodically (e.g., daily), and the resulting item embeddings are then stored and served exactly like any other precomputed embedding (feeding into ANN indices, Ch. 17, just as two-tower item embeddings do).
- Neighbor sampling strategy (how many neighbors, how they're weighted/selected) is a real hyperparameter with accuracy/cost trade-offs — too few sampled neighbors loses signal; too many increases training and (if done at serving time for any real-time component) inference cost.
- GNNs are particularly valuable in domains with **rich, explicit relational structure beyond simple user-item interaction** (Pinterest's boards/pins, social network follow-graphs, knowledge graphs of product relationships) — the value proposition is strongest when there's real graph structure to exploit beyond what a flat interaction matrix already captures.

## 8. Interview Traps

- Describing GNN aggregation without mentioning that popular nodes can have huge neighbor counts, and therefore without mentioning neighbor sampling as a necessary scalability mechanism — this is the single most commonly-tested production-reality detail for this topic.
- Treating GNN-based recs as a replacement for two-tower retrieval rather than as a complementary embedding-generation technique that typically feeds into the same downstream ANN-based serving infrastructure.
- Not being able to explain concretely what a 2-hop (vs 1-hop) embedding captures that a direct-feature-only embedding cannot — vague "graph neural networks capture structure" answers without the "friends-of-friends" concrete mechanism (as in Section 5) read as memorized rather than understood.
- Confusing uniform-average aggregation with PinSage's importance-weighted (random-walk-based) aggregation — the specific random-walk importance weighting is a distinguishing, testable detail of PinSage versus a generic GNN.

## 9. L5-Differentiating Talking Points

- Explain PinSage's random-walk importance weighting as a graph-native analogue to attention (Ch. 14) — weighting neighbors by structural importance rather than uniform averaging — showing you connect ideas across architectures rather than treating each as an isolated technique.
- Bring up neighbor sampling as a scalability necessity unprompted, and draw the explicit parallel to negative sampling in BPR/two-tower training (Ch. 9, 12) — both are instances of the same general pattern: use a tractable, representative subset instead of the full (intractably large) set.
- State clearly that GNN-derived embeddings are typically a **feature-generation technique feeding into existing serving infrastructure** (ANN retrieval, Ch. 17) rather than a separate end-to-end serving system — showing you understand how this fits into the broader system architecture from Module 5, not just the algorithm in isolation.
- Note that the domains where GNNs provide the most lift are those with genuine rich relational structure beyond flat interactions (social graphs, board/pin structures, knowledge graphs) — showing judgment about *when* this added complexity is actually worth it, not reflexively proposing GNNs for any recsys problem.

## 10. Comprehension Check

1. What does a 2-layer GNN aggregation capture that a standard single-embedding-lookup model (MF, two-tower) does not?
2. Why does PinSage use random-walk-based importance weighting instead of simple uniform-average neighbor aggregation?
3. Why is neighbor sampling necessary at production scale, and what specific problem does it solve?
4. How do GNN-derived embeddings typically fit into a broader production serving architecture, relative to two-tower embeddings?
5. In what kind of recommendation domain is a GNN-based approach likely to provide the most additional value over standard collaborative filtering, and why?
