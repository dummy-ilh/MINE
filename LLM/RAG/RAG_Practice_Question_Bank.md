# RAG Practice Question Bank

Standalone question bank, separate from the embedded Q&A drills in each module file. Organized by category and roughly increasing difficulty. Use this for mock-interview drilling — cover the answer, try to answer cold, then check yourself against the module notes.

---

## Category A: Core Concepts (warm-up / screening-round level)

1. What problem does RAG solve that fine-tuning and long-context alone don't?
2. Explain the difference between a bi-encoder and a cross-encoder.
3. What is the purpose of chunking, and what's the core tradeoff in choosing chunk size?
4. What's the difference between dense and sparse retrieval, and what does each miss?
5. What is BM25, and what do the `k1` and `b` parameters control?
6. What is an ANN algorithm, and why is it needed instead of exact kNN?
7. Name three vector database options and one differentiator for each.
8. What is reranking, and why is it a separate stage from initial retrieval?
9. What are the three metrics in the "RAG triad" and what does each measure?
10. What is hallucination in the context of RAG specifically, and how is it different from generic LLM hallucination?

---

## Category B: Deep-Dive / Mechanism-Level (mid-level / onsite round)

11. Derive or explain the InfoNCE contrastive loss used to train embedding models.
12. Explain hard negative mining and why in-batch negatives alone are insufficient.
13. Walk through how HNSW search works, layer by layer.
14. Explain IVF-PQ and why product quantization trades accuracy for memory.
15. What is ColBERT's MaxSim operator, and why is late interaction a genuine middle ground between bi-encoders and cross-encoders?
16. Explain Reciprocal Rank Fusion and why it uses rank instead of raw scores.
17. What is HyDE, and why does embedding a hallucinated document improve retrieval instead of hurting it?
18. Explain "lost in the middle" and its implications for how you order retrieved context.
19. Walk through pointwise, pairwise, and listwise LLM-as-reranker approaches and their cost/quality tradeoffs.
20. Explain nDCG and when you'd prefer it over Recall@k or MRR.
21. What are the known biases in LLM-as-judge evaluation, and how do you mitigate each?
22. Explain the pre-filter vs post-filter vs integrated-filter tradeoff for metadata-constrained vector search.
23. Walk through Self-RAG and Corrective RAG and how they differ from standard single-shot RAG.
24. Explain why embeddings from two different model versions can't be mixed in the same index, and what a safe migration looks like.
25. What's the difference between context relevance and Recall@k as metrics — don't they measure the same thing?

---

## Category C: Multi-Hop Specific

26. Why does single-hop retrieval fail on compositional/bridge-entity questions?
27. Compare IRCoT, Self-Ask, and decomposition-based multi-hop approaches.
28. What is error propagation in multi-hop retrieval, and what mitigates it?
29. How do you decide when to stop iterating in an iterative multi-hop retrieval loop?
30. When would you use graph-based multi-hop (GraphRAG-style traversal) instead of iterative dense retrieval?
31. What evaluation datasets are standard for multi-hop RAG, and what does each specifically stress?
32. Why can a multi-hop system get the final answer right for the wrong reason, and how do you catch that?

---

## Category D: Diagnosis / Debugging (scenario-based)

33. A user reports a wrong answer. Walk through your debugging process in order.
34. Retrieval quality has degraded gradually over a month with no code changes. What are your hypotheses?
35. The correct chunk was retrieved (verified in logs) but the model still got the answer wrong. What do you check next?
36. A generated answer cites a real document, but the citation doesn't actually support the claim. How would you catch this systematically, not just in this one case?
37. Your offline eval metrics look great but production user satisfaction is dropping. What's your first hypothesis and how do you investigate?
38. You suspect your embedding model has a domain-adaptation gap. What's the fastest way to confirm or rule this out?
39. Answer quality was fine for months and suddenly dropped after a routine content update. What's your leading hypothesis?

---

## Category E: System Design (open-ended, expect follow-ups)

40. Design a RAG system for customer support over product documentation, sub-second latency requirement.
41. Design a RAG system for enterprise search with per-user/per-team access control.
42. Design a RAG system that must handle both trivial factoid queries and complex multi-hop queries efficiently — how do you avoid paying multi-hop cost on every request?
43. How would you scale a RAG system from 100K to 100M documents? What's the first thing that breaks?
44. Design a RAG system where source documents update multiple times per day and answers must reflect edits within minutes.
45. How would you design cost monitoring for a RAG system in production — what are the line items and which typically dominates?
46. Design a multi-tenant RAG system — shared index or per-tenant index, and why?
47. Walk through migrating a production RAG system to a new embedding model with zero downtime.

---

## Category F: Judgment / Tradeoff (no single right answer — argue your reasoning)

48. When is hybrid retrieval not worth the added complexity?
49. When would you choose pgvector over a managed vector DB like Pinecone, even at meaningful scale?
50. When is GraphRAG worth its overhead, and when is it a poor fit?
51. Your reranker is your single biggest latency cost after generation. What are your options, and what does each cost you?
52. When would you use Corrective RAG versus accepting some retrieval failures as a base rate?
53. When is context compression worth its own risk of information loss?
54. Weighted-sum fusion vs RRF for combining dense and sparse retrieval — when would you actually invest in the former?
55. Your corpus has adversarially many near-duplicate documents. How does this affect retrieval, reranking, and generation, and what would you change at each stage?

---

## How to use this bank

- **Timed drill**: pick 5 questions cold, answer out loud in under 90 seconds each, then check against the relevant module.
- **Depth check**: for Category B/C, try to go one level deeper than the question asks — e.g. for Q11, don't just state the formula, explain *why* the temperature parameter matters.
- **Scenario chaining**: for Category D, practice narrating your elimination process (Module 8's workflow) out loud rather than jumping to a guessed root cause.
- **System design pacing**: for Category E, practice the scoping-questions-first move (Module 9.1, Step 1) before describing any architecture — interviewers weight this heavily.
