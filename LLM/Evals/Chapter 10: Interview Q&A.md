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

That closes the 10-chapter LLM Evals curriculum — foundations (1-2), human and LLM judgment (3-4), benchmarks and task-specific evals (5-6), safety (7), statistics (8), production engineering (9), and this consolidated Q&A (10).Curriculum complete — all 10 chapters delivered. Want me to package this into a self-contained markdown reference doc, or move on to something else?
