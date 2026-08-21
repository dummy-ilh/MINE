# Chapter 9: Debugging & Production Evals

## Why this chapter is different from everything before it

Chapters 1-8 assumed a fairly clean setting: you have a model, you have an eval set, you compute a number. Production is messier: the model is live, traffic is changing under you, the eval pipeline itself can silently break, and by the time a metric drops you need to know *why*, fast. This chapter is about the engineering and monitoring discipline around evals — the part that turns "we ran an eval once" into "we'd know within hours if something broke."

## The eval funnel, revisited as an engineering pipeline

Recall the Chapter 1 funnel: offline intrinsic → offline extrinsic → online. In production, this becomes a literal **CI/CD gate structure**:

1. **Pre-commit / PR-level checks** — fast, cheap offline-intrinsic evals (perplexity sanity checks, a small regression test suite of known tricky prompts) run automatically on every model/prompt change, like unit tests in software engineering.
2. **Pre-deployment gate** — a larger offline-extrinsic eval suite (the benchmarks from Ch5, task-specific evals from Ch6, safety/red-team suite from Ch7) run before a candidate model is allowed to reach any real traffic. Must clear a threshold, often with the statistical rigor from Ch8 (is the new model *significantly* better/not-worse, not just numerically different) before promotion.
3. **Canary / shadow deployment** — the new model runs on a small slice of live traffic (or "shadow mode," where it processes real traffic but its output isn't shown to users, only logged and compared against the current production model).
4. **Full A/B test** — the new model serves a real portion of traffic, with online metrics (task success, user engagement, complaint rate) tracked against the control.
5. **Post-launch continuous monitoring** — even after full rollout, evals keep running against live traffic samples, because the world (user behavior, adversarial attempts, data distribution) keeps shifting.

**The core engineering principle to state in an interview:** *"Eval isn't a one-time gate before launch — it's a continuous pipeline that runs before, during, and after deployment, with increasing cost and increasing realism at each stage, and automatic rollback triggers if a stage fails."*

## A/B Testing for LLM systems specifically

This connects directly to Chapter 8's statistical toolkit, applied online instead of offline.

**What's different about online A/B tests vs. offline eval comparisons:**
- You're measuring **real behavioral outcomes** (did the user complete their task, did they re-ask the question, did they escalate to a human agent, did they churn) rather than a judge's or classifier's proxy score.
- **Traffic allocation and randomization** matter enormously — users must be randomly and consistently bucketed (same user always sees the same variant) to avoid contamination between arms.
- **Guardrail metrics** — beyond the primary metric you're trying to improve (e.g., task success rate), you monitor guardrails that must *not* regress: latency, cost per query, safety-flag rate, complaint rate. A model that improves the primary metric but blows up latency or safety flags shouldn't ship, even with a "winning" primary result.

**Worked example of a guardrail failure catching what a primary metric alone would miss.** New summarization model: task success (user doesn't re-ask) improves from 78% to 83% — a clear win by Chapter 8's significance-testing standards. But the safety-flag rate guardrail also ticks up from 0.3% to 0.9% — a 3x increase in a rare-but-severe failure mode. **Correct call: don't ship**, or ship only after investigating the safety regression, even though the primary metric result looks great. This is exactly the kind of "would you ship this" trap interviewers set up.

**Sequential testing / peeking problem:** a very common practical bug — teams check A/B test results every day and stop as soon as they see p<0.05, without correcting for the fact that repeatedly "peeking" at accumulating data inflates the false-positive rate far above the nominal 5%. Fix: either commit to a fixed sample size/duration decided in advance (using the power analysis from Ch8), or use a sequential testing methodology (e.g., alpha-spending functions) explicitly designed to allow valid early stopping.

## Drift Detection — the production-specific problem offline eval can't catch

**The core issue:** your offline eval set was built at some point in time. Real user traffic changes — new topics trend, user phrasing shifts, adversarial users adapt, upstream data sources (for RAG) get updated. A model that was well-calibrated against last quarter's traffic distribution can silently degrade against this quarter's traffic, even with zero code changes — because the *input distribution* moved, not the model.

**Two distinct kinds of drift to distinguish (interviewers like this precision):**
- **Input/data drift** — the distribution of incoming queries has changed (e.g., users start asking about a new product feature that didn't exist when the eval set was built).
- **Output/performance drift** — for the same *type* of query, the model's actual quality is degrading (e.g., a RAG system's faithfulness score is quietly dropping because the underlying document index went stale).

**How you actually detect drift in production, concretely:**
- **Statistical distribution monitoring** — track embedding-space statistics of incoming queries over time (e.g., mean/covariance shift, or a distributional distance like population stability index or KL divergence between this week's query embeddings and a reference baseline); a sharp shift flags input drift worth investigating.
- **Shadow re-scoring** — periodically re-run a sample of live production traffic through your full offline eval pipeline (LLM-judge scoring, faithfulness checks, etc.) even though it already went out to real users — this catches output drift that pure online business metrics might miss or be slow to reflect (e.g., a slow faithfulness decay might not show up in "task success" for a while if users don't immediately notice hallucinated details).
- **Automatic alerting on eval metric thresholds** — the same way you'd alert on latency/error-rate in classic software monitoring, set alert thresholds on rolling eval scores (e.g., "alert if faithfulness on the last 1,000 shadow-scored queries drops below 0.85").

## Debugging a regression — the actual workflow when a metric drops

This is the part interviewers want to see you can do live, not just describe abstractly. When an eval metric drops, the debugging process mirrors the decomposition principle from Chapter 6:

1. **Confirm it's real, not noise** — apply Chapter 8's tools: is the drop outside the confidence interval / statistically significant, or could this just be normal eval-set sampling variance?
2. **Isolate which stage regressed** — for a RAG system, check retrieval metrics (precision@k, MRR) separately from generation metrics (faithfulness, relevance) — did retrieval degrade (e.g., stale index, embedding model swapped silently) or did generation degrade (e.g., a prompt template change, a model version bump)?
3. **Slice the regression** — is the drop uniform across all query types, or concentrated in one segment (e.g., only affects a specific language, only affects long documents, only affects a specific customer)? Slicing often reveals the root cause immediately (e.g., "faithfulness only dropped for queries against the newly-added document category" points straight at an indexing bug for that category).
4. **Check for upstream silent changes** — a shockingly common real root cause: someone updated an embedding model, a chunking strategy, a prompt template, or a dependency version without it being flagged as a "model change" — eval regressions are often caused by infrastructure changes, not model changes. Always check deployment/config diffs, not just model weights.
5. **Reproduce offline, fix, re-validate through the full funnel** — once isolated, reproduce the failure in a fast offline eval, fix it, and re-run the full gate sequence (offline → canary → A/B) rather than hot-fixing straight to full production.

**Worked example tying it together.** Faithfulness score on shadow-scored production traffic drops from 0.91 to 0.79 over one week.
- Step 1: check CI — 0.79 vs 0.91 on a sample of 2,000 queries is well outside noise, confirmed real.
- Step 2: retrieval metrics (precision@5) are flat — unchanged. So it's a generation-side issue, not retrieval.
- Step 3: slicing reveals the drop is concentrated entirely in queries handled after Tuesday 2pm.
- Step 4: checking deployment logs — a prompt template change shipped Tuesday 1:45pm, adjusting the system prompt's wording around context usage.
- Step 5: root cause confirmed — revert or fix the prompt template, re-validate faithfulness returns to ~0.91 offline before re-deploying.

This is a realistic "walk me through how you'd debug this" interview answer, end to end.

## Quick check

Your online A/B test shows the new model wins on task success rate with p<0.001 after just 2 days of data, and your team wants to ship immediately. What two concerns from this chapter should you raise before agreeing?

**(1) The peeking/sequential-testing problem** — was this the pre-committed sample size/duration, or did the team check daily and stop as soon as they saw significance? Stopping early on repeated peeks inflates false-positive rate well beyond the nominal threshold, so a "quick" significant result deserves scrutiny about how it was obtained. **(2) Guardrail metrics** — has anyone checked latency, cost, and safety-flag rate haven't regressed? A win on the primary metric alone isn't sufficient to ship; you'd want to confirm no guardrail metric moved in the wrong direction before approving the rollout.

---

Chapter 10 (fixed) is Interview Q&A — rapid-fire and system-design-style questions pulling together everything from Chapters 1-9. Want me to continue?
