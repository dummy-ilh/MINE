# Agents Module 3 — Chain-of-Thought & Reasoning Prompting (Master Notes, Expanded)

## 0. Why this module sits between "tool use" and "ReAct"

Tool use (Module 2) gives an agent the ability to *act*. But deciding **what to do, and in what order**, especially for multi-step reasoning tasks, is a separate problem — this module covers the prompting techniques that improve the model's raw reasoning quality, which ReAct (Module 4) will then interleave with actions. Understanding CoT deeply first is what makes ReAct's design make sense rather than feeling like an arbitrary combination.

---

## 1. Standard Chain-of-Thought (CoT)

### The core idea, in plain words
Instead of asking the model to jump straight to a final answer, prompt it (via instruction, or via few-shot examples showing worked-out reasoning) to **generate intermediate reasoning steps in text, before** the final answer — "Let's think step by step" is the canonical zero-shot trigger phrase from the original CoT paper (Wei et al., 2022).

### Why this actually helps — the computational mechanism, not just "it seems to work"
This directly connects to the CLM objective from LLM Basics Module 2: the model generates output **autoregressively, one token at a time, and every previously generated token becomes part of the context for generating the next one.** If the model is forced to jump directly to a final numeric/short answer, all the "thinking" has to happen implicitly, compressed into the hidden-state computation of a single forward pass with no ability to condition later computation on earlier intermediate conclusions. If instead the model first generates explicit reasoning tokens, **each subsequent token gets to condition on the actual text of the reasoning so far** — effectively giving the model extra sequential computation steps (extra forward passes, one per generated token) to work through a problem, rather than compressing all reasoning into the fixed-depth computation of the transformer's layers in one pass.

**Interview-ready framing**: "CoT works because it trades output length for effective computation depth — a transformer has a fixed number of layers, so a single-pass 'jump straight to the answer' forces all reasoning into that fixed depth. Generating intermediate steps lets the model spend additional sequential forward passes on the problem, with each step conditioning on genuinely-computed prior steps, not just hidden internal activations."

### Numerical example of the compounding-error framing (directly reuses Agents Module 1's math)
Suppose a multi-step arithmetic problem requires 4 correct sequential reasoning steps, and per-step correctness without CoT (implicit, compressed reasoning) is 60%, while with explicit CoT it rises to 85% (a plausible, often-observed gap for genuinely multi-step problems):
```
Without CoT: 0.6^4 ≈ 13.0% end-to-end success
With CoT:    0.85^4 ≈ 52.2% end-to-end success
```
This is the same compounding-error math from Module 1's foundations section, now applied to *reasoning steps* rather than *agent actions* — worth explicitly naming this parallel if asked, since it shows the same underlying principle (errors compound multiplicatively across sequential steps; anything that raises per-step accuracy has an outsized effect on end-to-end success) applies at both the reasoning level and the action level.

### Zero-shot vs. few-shot CoT
- **Zero-shot CoT**: just append "Let's think step by step" (or similar) to the prompt — no worked examples needed, relies purely on the model's instruction-following/alignment training to produce good step-by-step reasoning.
- **Few-shot CoT**: include a small number of example problems in the prompt, each showing the *full reasoning chain* (not just the final answer) before the model's own problem — this is In-Context Learning (LLM Basics Module 4) applied specifically to reasoning-chain format, not just input→output mapping; the model pattern-matches the *style and granularity* of reasoning shown, not just the task itself.

---

## 2. Self-Consistency

### The core idea
Standard CoT generates **one** reasoning chain (typically greedy or low-temperature) and commits to its final answer. Self-consistency instead: **sample multiple independent CoT reasoning chains** (at nonzero temperature, so they genuinely differ — same mechanism as LLM Basics Module 6's temperature sampling), let each chain reach its own final answer independently, then **take a majority vote** across all the sampled final answers as the reported result.

### Why this helps — the statistical reasoning
Different sampled reasoning chains can take genuinely different (but each individually plausible) paths through a problem — some paths will make an error, some won't. **If the model is more likely to arrive at the correct answer via at least one "typical," well-reasoned path than any single specific wrong path is likely to be independently reproduced across multiple samples**, then majority voting concentrates probability mass on the correct answer, since errors are less likely to be *consistent* across independently-sampled chains than the correct reasoning is.

### Numerical worked example
Suppose for a given problem, the true correct answer has a 55% chance of being reached by any single sampled CoT chain, while three different *specific* wrong answers each individually have only a ~15% chance of being reached (55% + 15% + 15% + 15% = 100%, three wrong answer "buckets").

Sample **5 independent chains** and take majority vote. The correct answer, with 55% per-chain probability, is overwhelmingly likely to get a plurality of the 5 votes (this can be worked out via binomial probability, but the qualitative point is what matters for an interview): **because no single wrong answer path shares that same 55% probability, the correct answer's votes don't have to compete against an equally-likely wrong-answer cluster** — the wrong answers split their votes across 3 different specific mistakes, further diluting any single wrong answer's chance of winning the majority. This "wrong answers tend to disagree with each other, correct answers tend to agree" asymmetry is the actual mechanism behind why self-consistency reliably improves over single-chain CoT in practice.

### The direct cost tradeoff to name explicitly
Self-consistency multiplies inference cost by the number of sampled chains (5 samples = ~5x the generation cost of single-chain CoT) — a clean, concrete quality-vs-cost dial, directly analogous to LLM Basics Module 6's beam-width tradeoff (more parallel exploration = better quality, proportionally higher cost) and Module 6's speculative-decoding-style compute/latency tradeoffs generally.

---

## 3. Least-to-Most Prompting and Decomposition Strategies

### The core idea
Rather than asking the model to solve a complex problem directly (even with CoT), **explicitly decompose the problem into an ordered sequence of simpler subproblems first**, then solve them **in order, feeding each subproblem's solution into the context for solving the next one** — a two-phase process: (1) decomposition (ask the model to break the problem into an ordered list of simpler subproblems), (2) sequential solving (solve subproblem 1, then subproblem 2 using subproblem 1's solution as additional context, and so on).

### Why this is meaningfully different from plain CoT
Plain CoT generates one continuous reasoning stream for the *whole* problem at once — if the model's initial framing/approach to the overall problem is subtly wrong, everything downstream in that single chain inherits the error, with no natural checkpoint to reconsider the decomposition itself. Least-to-most explicitly **separates "what are the right subproblems" from "solve each subproblem"** into distinct steps, which (a) makes each individual solving step easier (smaller, more constrained problems are generally more reliably solved than one large compound problem, echoing exactly the same "smaller steps, higher per-step accuracy" reasoning from Section 1's numerical example), and (b) creates a natural point to verify/adjust the decomposition itself if something looks wrong, before committing to solving all the subproblems.

### Concrete example
Complex question: "If a store had 120 apples, sold 30% on Monday, and then sold half of what remained on Tuesday, how many are left?"

**Decomposition step** (model generates the subproblem list): 
1. Compute how many apples were sold on Monday.
2. Compute how many apples remained after Monday.
3. Compute how many were sold on Tuesday (half of the Monday-remainder).
4. Compute the final remaining count.

**Sequential solving**: solve subproblem 1 (30% of 120 = 36) → feed "36 sold Monday" into context for subproblem 2 (120-36=84 remain) → feed that into subproblem 3 (half of 84 = 42 sold Tuesday) → feed that into subproblem 4 (84-42=42 remain). Each step is a small, easy arithmetic operation with the *previous, verified* result as clean input — rather than the model trying to hold and correctly sequence all four operations implicitly within one CoT stream.

---

## 4. Side-by-side summary table (memorize this cold)

| | Standard CoT | Self-Consistency | Least-to-Most |
|---|---|---|---|
| Number of reasoning chains generated | 1 | Multiple (sampled independently) | 1, but explicitly staged/decomposed |
| Mechanism for improvement | Extra sequential computation via explicit intermediate tokens | Majority vote across independent samples, exploiting error-disagreement asymmetry | Separates "what are the subproblems" from "solve each one," reducing per-step complexity |
| Extra inference cost vs. plain single-shot answer | Modest (longer output, same 1 generation) | High (Nx generations for N samples) | Modest-to-moderate (decomposition step + sequential solving steps) |
| Best suited for | General multi-step reasoning | Problems where errors are inconsistent across samples but correct answers are consistent | Compound problems with a clear natural subproblem structure |

---

## 5. Quick-fire Q&A (self-test)

**Q: Explain, mechanistically (not just "it works empirically"), why generating intermediate reasoning tokens improves accuracy on multi-step problems.**
A: A transformer has a fixed computational depth per forward pass; forcing a direct final answer compresses all reasoning into that fixed depth in one pass. Generating explicit intermediate tokens autoregressively lets each subsequent token condition on genuinely-computed prior reasoning text, effectively trading output length for additional sequential computation steps rather than compressing everything into one pass's hidden-state computation.

**Q: What's the core statistical mechanism that makes self-consistency's majority voting work?**
A: Correct reasoning chains tend to independently arrive at the same (correct) answer with relatively high, consistent per-chain probability, while different wrong reasoning paths tend to arrive at different, specific wrong answers, splitting the wrong-answer vote share across multiple distinct incorrect options — so the correct answer's votes aren't diluted by competing against an equally-concentrated wrong-answer cluster.

**Q: What's the direct cost tradeoff of self-consistency, and what's an analogous tradeoff from LLM Basics you'd cite for comparison?**
A: It multiplies inference cost by the number of sampled chains (N samples ≈ Nx generation cost) — directly analogous to beam search's width-vs-cost tradeoff (LLM Basics Module 6), where more parallel exploration improves quality at proportionally higher compute cost.

**Q: What specifically does least-to-most prompting add beyond plain CoT, and why does that separation matter?**
A: It explicitly separates decomposing the problem into an ordered list of subproblems from solving each subproblem sequentially, rather than reasoning through the whole compound problem in one continuous chain. This matters because it creates a natural checkpoint to verify/adjust the decomposition before committing to solving it, and because smaller, more constrained subproblems are generally more reliably solved than one large compound problem — the same "smaller steps → higher per-step accuracy" principle behind CoT's own benefit.

**Q: If you had a fixed compute/latency budget and had to choose one technique for a complex, genuinely-compound multi-step problem, how would you decide between self-consistency and least-to-most?**
A: If the failure mode is *inconsistent* errors across different reasoning attempts on the *same* framing of the problem, self-consistency's majority voting directly targets that. If the failure mode is more about the model choosing a *wrong overall approach/decomposition* to the compound problem in the first place, least-to-most's explicit decomposition step targets that more directly — a genuinely balanced answer names both the mechanism-match and that they're not mutually exclusive (you could decompose first, then apply self-consistency to each subproblem).

---
*End of Agents Module 3 (expanded). Next: Module 4 — ReAct (Reasoning + Acting): the Thought-Action-Observation loop, and why interleaving beats CoT-then-act.*
