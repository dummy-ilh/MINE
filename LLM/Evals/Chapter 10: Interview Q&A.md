Taking that as "10" since Chapter 9 is already done — moving to the final chapter.

# Chapter 10: Interview Q&A

Rapid-fire questions pulling together Chapters 1-9. Each includes a tight, interview-ready answer.

---

**Q1: Why can't you just use loss/perplexity to decide whether to ship a model?**
Loss/perplexity is an offline-intrinsic proxy — it measures fluency and next-token prediction quality, not truthfulness, helpfulness, or task success. A model can have great perplexity and still hallucinate confidently or be extrinsically useless for the actual task. Always validate with offline-extrinsic and eventually online metrics before shipping.

**Q2: BLEU vs. ROUGE — when do you use each, and why do both struggle with modern LLM eval?**
BLEU is precision-oriented (used for translation: how much of what was generated is correct). ROUGE is recall-oriented (used for summarization: how much of what should've been said was captured). Both rely on surface n-gram overlap against a single reference, so they penalize valid paraphrases and can't detect hallucinated-but-fluent content — which is why LLM-as-judge and embedding-based semantic metrics have largely replaced them for open-ended generation.

**Q3: Walk me through pass@k.**
$\text{pass@}k = 1 - \binom{n-c}{k}/\binom{n}{k}$, where n = samples generated, c = samples passing all unit tests. It's the probability at least one of k randomly-drawn samples (without replacement, computed exactly from one batch) succeeds. Used for code gen because real usage often samples multiple completions.

**Q4: Why is pairwise comparison generally preferred over absolute (1-5) scoring, for both human and LLM judges?**
Humans (and LLM judges mimicking similar behavior) are more consistent at relative judgments than absolute magnitude estimation — absolute scores cluster/anchor (everyone says "4") and drift across raters/sessions, while "which is better" comparisons are more reproducible. Tradeoff: pairwise gives rankings, not directly interpretable scalar scores, so you need something like Elo aggregation to get a leaderboard number.

**Q5: Name three biases in LLM-as-judge and how you'd mitigate each.**
Position bias (favors whichever response is shown first/second) → swap order and average. Verbosity bias (favors longer answers) → explicit length-penalty instructions or length-controlled reporting. Self-preference bias (favors same-family outputs) → use a judge from a different model family than any candidate, or ensemble judges across families.

**Q6: How do you know if your human-annotated labels are trustworthy?**
Measure inter-annotator agreement corrected for chance — Cohen's kappa for 2 annotators/categorical labels, Krippendorff's alpha for 3+ annotators, missing data, or ordinal scales. Raw agreement percentage is misleading with imbalanced classes since chance agreement can already be high.

**Q7: Compute Cohen's kappa given: observed agreement 88%, annotator 1 flags positive 20% of the time, annotator 2 flags positive 25% of the time.**
$P_e = (0.20\times0.25) + (0.80\times0.75) = 0.05+0.60=0.65$. $\kappa=(0.88-0.65)/(1-0.65)=0.23/0.35≈0.657$ — substantial agreement.

**Q8: What's the contamination problem in benchmarks, and how do you detect it?**
Public benchmarks (MMLU, GSM8K) may leak into pretraining corpora, so high scores can reflect memorization rather than genuine capability. Detect via n-gram overlap checks against training data, canary strings embedded in benchmark files, comparing performance on the original vs. a fresh held-out variant, or perturbation testing (rephrase questions, see if score holds).

**Q9: Decompose RAG evaluation into its components and explain why you can't just judge the final answer.**
Retrieval-side: precision@k, recall@k, MRR. Generation-side: faithfulness (is every claim grounded in retrieved context), answer relevance (does it address the question), context relevance (were the right docs retrieved). Decomposing lets you diagnose *where* to fix — bad retrieval needs a retriever/embedding fix, bad faithfulness with good retrieval needs a prompt/generator fix.

**Q10: A RAG faithfulness score is high but context relevance is low. What's broken?**
The retriever — it's not finding relevant documents, but whatever it does retrieve, the generator is faithfully sticking to it (not hallucinating beyond the context). Fix the retrieval/embedding pipeline, not the generator.

**Q11: How would you evaluate an agent, beyond simple task success rate?**
Trajectory/process quality (efficient path, no wasted or erroneous tool calls), tool-use correctness (right tool, valid arguments per call), and steps-to-completion/error-retry rate — two agents can both succeed but one may take 3 clean steps vs. another taking 15 with errors and retries, which task success alone won't surface.

**Q12: What's Expected Maximum Toxicity, and why not just report average toxicity?**
Sample k completions per prompt, take the max toxicity score per prompt, then average across prompts. Average toxicity hides tail risk — a model that's 99% fine but occasionally severely toxic is still a real production risk, since a single bad output shown to one user is the actual failure event, not the average.

**Q13: What's Attack Success Rate, and why should it be broken down by severity?**
ASR = successful jailbreaks / total attack attempts. A single blended number hides risk — 7% ASR could mean mildly off-color jokes or genuinely dangerous content. Severity-tiered reporting (low/medium/high-risk categories) is needed for a meaningful risk picture.

**Q14: Model B beats Model A by 2 points on a 500-example eval. Do you ship B?**
Not on that alone. Compute confidence intervals for each (margin ∝ 1/√n) — with n=500 a 2-point gap often has overlapping CIs, meaning it could be noise. Better: run a paired significance test (McNemar's for binary correct/incorrect, bootstrap for other metrics) since both models were evaluated on the same examples, which is more powerful than comparing separate CIs.

**Q15: Why paired tests over unpaired for model comparison?**
Both models are run on the same examples, so per-example difficulty is correlated — an unpaired test ignores that correlation, overstates apparent noise, and is less sensitive to real differences. Paired tests (McNemar's, paired t-test, bootstrap on paired differences) use that shared structure and detect real gaps more reliably.

**Q16: What's the "peeking problem" in A/B testing and how do you avoid it?**
Checking results daily and stopping as soon as p<0.05 inflates the false-positive rate well above the nominal 5%, because you're effectively running many tests and taking the best one. Fix: commit to a sample size/duration in advance via power analysis, or use a sequential testing method (e.g., alpha-spending) explicitly designed for valid early stopping.

**Q17: Online A/B test shows a clear win on the primary metric. What else do you check before shipping?**
Guardrail metrics — latency, cost per query, safety-flag rate, complaint rate. A model that improves the primary metric but regresses a guardrail (e.g., 3x more safety flags) shouldn't ship without further investigation, even with a statistically significant primary win.

**Q18: Faithfulness drops from 0.91 to 0.79 on production shadow-scoring. Walk me through debugging it.**
(1) Confirm it's statistically real, not sampling noise. (2) Isolate the stage — check retrieval metrics separately from generation metrics to see which regressed. (3) Slice by segment/time to localize the cause. (4) Check for silent upstream changes — prompt template, embedding model, chunking strategy, dependency bumps — these are a very common root cause, not model weight changes. (5) Reproduce offline, fix, and re-validate through the full offline→canary→A/B gate sequence before re-deploying.

**Q19: What's the difference between input drift and output/performance drift?**
Input drift = the distribution of incoming queries changes (new topics, new phrasing). Output/performance drift = for the same query type, model quality degrades over time (e.g., stale RAG index). Detect input drift via embedding-distribution monitoring (e.g., population stability index); detect output drift via periodic shadow re-scoring of live traffic through the full offline eval pipeline.

**Q20: Design an eval pipeline for a new production LLM feature, end to end.**
Funnel structure: (1) fast offline-intrinsic checks on every commit (regression prompt suite), (2) offline-extrinsic gate before deployment (task benchmarks, human/LLM-judge rubric scoring, safety/red-team suite — with statistical significance testing against the current production model, not just raw numeric comparison), (3) canary/shadow deployment comparing against production, (4) full A/B test with guardrail metrics and a pre-committed sample size to avoid peeking, (5) continuous post-launch monitoring — drift detection plus periodic shadow re-scoring — with automatic alerting if key metrics cross thresholds.

---
# Full Worked Case Study: Evaluating a RAG-Powered Support Agent

## The system

"TechCorp Support Agent" — a user asks a question, the agent:
1. Retrieves relevant docs from a knowledge base (RAG)
2. Reasons about what to do
3. Optionally calls tools (`check_order_status`, `issue_refund`, `update_shipping_address`, `escalate_to_human`)
4. Returns a final answer to the user

This is deliberately the hard case — it's *both* a RAG system (Chapter 6) and an agent (Chapter 6), so you need both eval frameworks stacked, not one or the other.

---

## One single trace, evaluated end to end

**User query:** *"My order #4471 hasn't arrived and it's been 2 weeks. I want a refund."*

**Agent's actual trajectory (logged):**

```
Step 1: retrieve(query="order 4471 hasn't arrived, refund policy")
  → returns 3 docs: [refund_policy.md, shipping_delays_faq.md, order_lookup_guide.md]

Step 2: call_tool(check_order_status, order_id="4471")
  → returns: {status: "in transit", shipped: "2024-08-01", carrier_delay: true}

Step 3: reasoning: "Order is delayed but still in transit, not lost. Per refund_policy.md,
  refunds for delay (not loss) require 21+ days. Only 14 days have passed."

Step 4: call_tool(issue_refund, order_id="4471")   ← PROBLEM: shouldn't have called this yet
  → returns: {refund_issued: true, amount: $89.99}

Step 5: final_answer: "I've issued a full refund of $89.99 for your delayed order."
```

### Step A — Retrieval-side scoring (from Chapter 6)

Ground truth: for this query type, the "gold" relevant docs are `refund_policy.md` and `shipping_delays_faq.md` (2 relevant docs exist in the corpus for this query).

Retrieved top-3: `refund_policy.md` ✓, `shipping_delays_faq.md` ✓, `order_lookup_guide.md` ✗ (not relevant — it's about how to look up order numbers, not delays/refunds)

- **Precision@3** = 2/3 ≈ 0.67
- **Recall@3** = 2/2 = 1.0 (found everything relevant that existed)
- **MRR** = 1/1 = 1.0 (first relevant doc, `refund_policy.md`, was rank 1)

Retrieval is doing its job here — high recall, decent precision, best doc ranked first.

### Step B — Generation/faithfulness scoring (from Chapter 6)

Decompose the final answer into atomic claims and check each against retrieved context:

| Claim | Supported by retrieved docs? |
|---|---|
| "Full refund of $89.99 issued" | Technically true (tool did issue it) but **contradicts** `refund_policy.md`, which the agent itself correctly cited in its own reasoning (14 days < 21-day threshold) |

**Faithfulness verdict: FAIL.** This is the interesting nuance — faithfulness isn't just "is this sentence grounded in *some* document," it's "does the final action/answer contradict what the retrieved context actually says." The agent's own Step 3 reasoning correctly concluded a refund wasn't yet warranted, but Step 4 did it anyway. That's a **reasoning-to-action inconsistency**, a failure mode specific to agents that plain RAG eval (which only judges text output) wouldn't even have a slot to record.

### Step C — Agent-side scoring (from Chapter 6)

- **Task success:** ambiguous/fail — the user got a refund, which superficially "resolves" their complaint, but it's a **policy violation** (refund issued 7 days before the eligibility threshold). If your success criterion is naive ("did the user get an outcome"), this scores as a win. If it's correctly defined ("did the agent take the *correct, policy-compliant* action"), this is a fail. **This is exactly why Chapter 6 said task success alone is insufficient** — you need the process/trajectory check too.
- **Tool-use correctness:** `check_order_status` call — correct, right tool, right argument. `issue_refund` call — **incorrect**: called prematurely, contradicting the agent's own retrieved policy and its own stated reasoning.
- **Trajectory efficiency:** 4 steps for what should've been a 3-step trace (retrieve → check status → explain the 21-day policy, no refund tool call) — 1 wasted/harmful step, not just an inefficient one.

**This single trace shows the diagnostic value of decomposition:** retrieval = fine, faithfulness = fail, tool-use = fail, naive task-success = misleadingly "pass." If you only tracked one blended "did the customer get helped" metric, this failure would be invisible in your dashboards until it caused a real financial/policy problem at scale.

---

## Now scale it: aggregate results across a 200-example eval set

Run this same instrumented pipeline across 200 test conversations, each independently scored on every axis above.

| Metric | Score | Read |
|---|---|---|
| Precision@3 (retrieval) | 0.71 | Retriever generally finds relevant docs, some noise |
| Recall@3 (retrieval) | 0.88 | Rarely misses the truly relevant doc entirely |
| MRR (retrieval) | 0.79 | Relevant doc usually ranked near top |
| Faithfulness (generation) | 0.86 | 14% of final answers/actions contradict retrieved context |
| Tool-use correctness | 0.91 | 9% of tool calls are wrong tool, bad args, or premature/unauthorized action |
| Naive task success (did user get *an* outcome) | 0.94 | Looks great — and is the misleading number |
| Policy-compliant task success (correct action per policy) | 0.79 | The real number that matters |
| Avg steps-to-completion | 3.4 | Baseline for efficiency tracking over time |

**The gap that matters most:** naive task success (0.94) vs. policy-compliant task success (0.79) — a **15-point gap**. That gap is a direct, quantified measure of exactly the failure mode from the single trace above: the agent "resolves" things in ways that look successful on the surface but violate business logic. If this were your only production dashboard, showing 94% success, leadership would think the system is performing great — while it's actually issuing unauthorized refunds 1 in 5 times among ambiguous cases.

**Where do you point the fix, using the decomposition principle from Chapter 6?**
- Retrieval scores (0.71–0.88) are solid → retriever isn't the bottleneck.
- Faithfulness (0.86) is decent but not the main driver of the 15-point success gap.
- Tool-use correctness (0.91) is the closest correlate — premature/unauthorized tool calls are the dominant root cause. **Fix target: the policy that decides *when* the agent is allowed to invoke `issue_refund`**, likely via a stricter tool-calling guardrail (e.g., require an explicit "eligibility confirmed: yes/no" reasoning step, checked programmatically, before the refund tool is even exposed to the model) — not a retrieval or prompt-wording fix.
Based on what's actually reported by candidates (Glassdoor, IGotAnOffer, Exponent, DataInterview) for ML/DS roles at these companies — here's what's real vs. what I built for you earlier.

## Google

Google interviewers push candidates to apply statistical reasoning to evaluation and trade-offs — confidence intervals, calibration, thresholding, and interpreting noisy offline results rather than pure theory recall. Two concrete reported scenarios:

- A model deployed for ad-policy enforcement suddenly flags twice as many ads as violations overnight, while offline eval on the labeled set is unchanged — candidates must design monitoring/alerting/retraining to distinguish input drift, label delay, and model regression, specifying at least three concrete signals with thresholds. This is almost exactly Chapter 9's drift-detection framework (input drift vs. output/performance drift), applied live.
- A search ranking model's offline NDCG@10 improves from 0.612 to 0.616 on 50,000 queries — candidates must decide if that's a real improvement given per-query scores are heavy-tailed. This is Chapter 8 territory — significance testing on a skewed metric, not a clean proportion, so bootstrap resampling is the right instinct.

Also reported: comparing the trade-offs between offline batch evaluation and online A/B testing for a recommendation system, and designing a system to fine-tune and serve an LLM for customer support, optimized for throughput and memory on TPUs (evals show up as a sub-component of these system design rounds, not usually standalone).

## Meta

Meta's reported style leans product-safety-flavored rather than pure-stats. One detailed reported prompt: candidates are asked to design deployment-time mitigation for an LLM-powered writing assistant in Instagram DMs that must not leak phone numbers or emails from conversation history, specifying the plan across prompting, retrieval, filtering, and logging, and stating what offline and online metrics would prove it's actually working. That's essentially Chapter 7 (safety/red-teaming evals) plus Chapter 9 (online monitoring) stitched together as one product scenario.

Meta interviewers are also reported to move between theory — bias/variance, regularization, calibration — and applied troubleshooting like data leakage, class imbalance, and offline-vs-online gaps, and system design rounds have included designing an evaluation framework for ad ranking as a standalone prompt.

## Apple

Less "interview trivia" available publicly, but Apple's own job postings for this exact function (Siri AI Quality Engineering) spell out almost verbatim what they screen for: using LLMs to automate large-scale data generation and evaluation job execution, building LLM judges, detecting anomalies, and streamlining ML evaluation workflows, plus continuously evaluating and improving model performance through A/B testing and human feedback loops. If you're prepping for Apple specifically, lean hard into Chapter 4 (LLM-as-judge) and Chapter 9 — that's literally the job description.

## General LLM-role pattern (cross-company)

One recruiting-side source made a point worth internalizing for interview strategy: the candidate who wants to fine-tune everything is signaling their instinct runs toward expensive, exciting work rather than the work that ships — interviewers are told to weight evals heavily, because that's where senior candidates pull away from everyone else.

**The honest caveat:** none of these sources published a clean "Q1, Q2, Q3" LLM-evals-specific bank the way DSA sites do for LeetCode — eval questions mostly show up embedded inside system-design or "debug this production issue" prompts, not as standalone quiz questions. So Chapter 10's rapid-fire format I gave you earlier is a reasonable *simulation* of interview style, but these five scenarios above are the actual reported ones — worth having a rehearsed answer for each specifically, since they're the real thing that's been asked.
---

## The interview-ready synthesis

*"For a RAG-powered agent, I wouldn't evaluate it as one blended 'did it work' number. I'd score retrieval and generation faithfulness the way I would for any RAG system, then separately score tool-use correctness and trajectory efficiency the way I would for any agent — and critically, I'd define task success as policy-compliant success, not just 'did the user get an outcome,' because those two diverge exactly in the cases that matter most, like an agent taking an action that resolves the symptom while violating the underlying policy. The decomposition is what lets you localize the fix — in this case, tool-use correctness pointed straight at over-eager tool invocation, not retrieval or faithfulness."*
