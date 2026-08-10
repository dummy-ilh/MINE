# Day 18: Evaluation — Task Success Rate, Trajectory Eval, LLM-as-Judge Pitfalls

## 1. The Intuition First

Every day through Day 17 built mechanisms — loops, gates, guardrails, caching. None of them tell you whether the agent is actually GOOD at its job, or whether a change you made improved or hurt it. That's what evaluation is for, and it's harder for agents than for a normal ML model in one specific way worth naming up front: **a classifier has one output to check against one label. An agent has a whole TRAJECTORY — a sequence of decisions — and there are often many different valid trajectories that reach a correct final answer, and some invalid-looking trajectories that stumble into a correct final answer anyway.** Evaluation has to account for this, or it measures the wrong thing.

Think about grading a student's math homework. If you only check the final numeric answer, you'll give full credit to a student who got the right number through a lucky arithmetic error that canceled out an earlier mistake, and zero credit to a student with perfect reasoning who made one final transcription slip. Good grading checks BOTH the answer and the work — that's the entire tension this lesson formalizes for agents: **outcome evaluation vs. trajectory evaluation, and why you usually need both.**

---

## 2. Formalizing the Core Evaluation Types

### 2.1 Task Success Rate (Outcome Evaluation) — Did It Get the Right Answer?

The simplest, most intuitive metric: run the agent on N tasks with known correct outcomes, check what fraction succeeded.

```python
def task_success_rate(agent, test_cases):
    successes = 0
    for case in test_cases:
        result = agent.run(case.input)
        if matches_expected(result, case.expected_output):
            successes += 1
    return successes / len(test_cases)
```

**What this hides**: it tells you WHETHER the agent succeeded, not HOW. Two agents with identical 80% success rates could have completely different failure characteristics — one fails cleanly and predictably on a known-hard subset of tasks, the other fails randomly and unpredictably across all task types, including ones it "should" handle. Outcome-only evaluation can't distinguish these, even though they represent very different levels of production readiness.

### 2.2 Trajectory Evaluation — Was the PATH Reasonable, Not Just the Destination?

Evaluates the sequence of intermediate steps — which tools were called, in what order, whether unnecessary steps were taken, whether the reasoning was sound — independent of whether the final answer happened to be correct.

```python
def trajectory_eval(trace, rubric):
    checks = {
        "used_relevant_tools_only": no_irrelevant_tool_calls(trace),
        "no_redundant_calls": no_duplicate_tool_calls(trace),
        "reasoning_is_grounded": every_claim_traces_to_an_observation(trace),
        "stopped_at_reasonable_point": not_over_or_under_thorough(trace),
    }
    return checks
```

**Why this matters even when the outcome is correct**: a trajectory that reached the right answer through an irrelevant tool call, a lucky guess, or reasoning that doesn't actually connect to the observations it gathered is NOT a trajectory you want to ship, even at 100% outcome accuracy on your test set — because it will very likely fail differently and unpredictably on inputs slightly different from what you tested. **Trajectory eval is what catches "right for the wrong reasons," which outcome eval structurally cannot detect.**

### 2.3 The Combination — Why You Need Both, With a Worked Comparison

| | Outcome correct | Outcome wrong |
|---|---|---|
| **Trajectory reasonable** | Ideal case — ship it | Reasonable process, bad luck/edge case — informative failure, worth investigating WHY a sound process failed |
| **Trajectory unreasonable** | "Right for the wrong reasons" — hidden risk, will likely fail unpredictably elsewhere | Doubly bad — both the process AND the outcome are wrong; usually the easiest failure to diagnose |

The most DANGEROUS quadrant is top-right: correct outcome, bad trajectory. It's dangerous specifically because outcome-only evaluation reports this as a full success, giving false confidence, while trajectory eval is the only lens that surfaces it at all.

---

## 3. Worked Example: A Trajectory That Passes Outcome Eval But Fails Trajectory Eval

**Task**: "What's our current employee headcount in the Austin office?"

**Trace**:
```
Thought: I'll search our HR system for headcount data.
Action: search_hr_system("Austin office headcount")
Observation: "Austin office: 142 employees as of last quarterly report."

Thought: I also want to double check by searching general company info.
Action: search_web("company Austin office size")
Observation: "According to a 2019 news article, the company's Austin office
had approximately 150 employees at the time of a facility expansion announcement."

Thought: Both sources roughly agree, around 140-150.
Final Answer: "The company currently has approximately 150 employees in the Austin office."
```

Suppose the "correct" expected answer (per test-case ground truth) is indeed ~150 — **this trajectory would PASS outcome evaluation.** But look at what actually happened:

- **Trajectory problem #1**: the second search was unnecessary and pulled a STALE, 2019 source for a question about the CURRENT headcount — a real trajectory flaw (mixing stale and current data, per Day 5 §5.2's staleness concern), even though it happened not to change the final number this time.
- **Trajectory problem #2**: the final answer actually cites the LESS reliable, older web source's number (150) rather than the MORE reliable, current HR system's number (142) — the reasoning that led to "approximately 150" doesn't actually correctly weight source reliability, it just averaged two numbers that happened to be close.

**If the HR system's true current number had been 142 and the stale web article had said 200 instead of 150**, this same flawed trajectory logic ("both sources roughly agree, so average them") would have produced a badly wrong answer — the trajectory's flaw was ALWAYS there, outcome eval on this one test case just got lucky that the numbers happened to be close enough not to matter. **This is exactly why trajectory evaluation exists**: it catches this reasoning flaw regardless of whether this particular test case's numbers happened to expose it.

---

## 4. Worked Example: LLM-as-Judge, and Its Pitfalls

For open-ended agent outputs (a drafted email, a research summary) where there's no single "correct" string to exact-match against, a common approach is using ANOTHER LLM call to judge quality — "LLM-as-judge."

```python
def llm_judge(task, agent_output, rubric):
    judge_prompt = f"""
    Task: {task}
    Agent's response: {agent_output}
    Rubric: {rubric}
    Rate the response 1-10 on accuracy, completeness, and appropriateness. Explain your reasoning.
    """
    return llm_call(judge_prompt)
```

**Pitfall #1 — Position/verbosity bias**: LLM judges have a documented tendency to rate LONGER responses as higher quality, independent of actual correctness or usefulness, and can show inconsistent preferences depending on the ORDER in which multiple candidate responses are presented for comparison. A judge comparing "Response A" vs "Response B" can give a different verdict than the same comparison with the labels swapped, purely from position bias, not content quality — a real, measured phenomenon, not a hypothetical concern.

**Pitfall #2 — The judge inherits the same blind spots as the model being judged**: if the same model family both generates the agent's responses AND judges them, systematic errors the model tends to make can go uncaught, because the judge doesn't actually "know" any better than the generator — it's evaluating with the same underlying knowledge and reasoning patterns that produced the potential error in the first place. This is a genuinely subtle point: LLM-as-judge is not an independent verification signal in the way a held-out human evaluator or a programmatic check would be.

**Pitfall #3 — Rubric ambiguity produces noisy, unreliable scores**: a vague rubric ("rate this 1-10 on quality") gives the judge enormous discretion, producing scores that vary run-to-run for the exact same input (a real reliability problem — you want an eval metric that's at least CONSISTENT even before worrying about whether it's ACCURATE). The fix is the same discipline as writing a good tool description (Day 3 §2.2) — be maximally specific: "does the response correctly cite the source document? does it address all 3 parts of the user's question? is the tone appropriate for a customer-facing message?" — concrete, checkable criteria produce far more reliable judge scores than a single vague quality rating.

**Worked example of the fix**:
```python
# WEAK — vague, unreliable
"Rate this response 1-10 for quality."

# STRONG — specific, checkable criteria, closer to a rubric than a vibe
"""
Answer yes/no for each:
1. Does the response directly answer the question asked (not a related but different question)?
2. Does the response cite at least one specific source from the provided context?
3. Does the response avoid making any claim NOT supported by the provided context?
4. Is the response free of contradictions within itself?
Then give an overall pass/fail based on whether ALL FOUR are yes.
"""
```
This produces far more consistent, defensible scores than a single holistic rating — and critically, it produces PER-CRITERION signal, so a failure is diagnosable ("it failed criterion 3 — hallucinated an unsupported claim") rather than just a mysterious low number.

**Mitigation for Pitfall #2 specifically**: use a DIFFERENT, ideally more capable, model as the judge than the one generating responses — this doesn't fully solve shared-blind-spot risk (different models can still share SOME failure patterns, especially ones stemming from similar training data), but it's a meaningfully more independent signal than self-judging. For high-stakes evaluation, combining LLM-as-judge with a smaller set of human-reviewed spot-checks (calibrating the judge against human judgment periodically) is the most robust approach in practice.

---

## 5. Production Considerations

### 5.1 Building a Test Set Is Itself the Hard, Underrated Part

Just like Day 5's "what to write to memory" and Day 17's "what counts as an injection attempt," the mechanically simple part (running eval code) is not where the real difficulty is — **curating a representative, sufficiently diverse test set with reliable ground truth is the actual bottleneck.** A test set skewed toward easy, common cases will report misleadingly high success rates that don't hold up on the harder, rarer cases production traffic actually contains. Production teams typically build test sets iteratively FROM real failures (every production incident becomes a new regression test case, directly connecting to Day 15's observability — a trace of a real failure is a ready-made eval case) rather than trying to anticipate every case upfront.

### 5.2 Evaluation Needs to Run Continuously, Not Just Pre-Launch

An agent's behavior can drift even with no code changes — the underlying model provider can update the model version, a knowledge base can go stale (Day 5, Day 9), a tool's downstream API can change its response format. **Evaluation is not a one-time pre-launch gate — it needs to run continuously (or on every deploy, at minimum) as a regression check**, the same discipline as CI/CD testing in traditional software, applied to a system whose "correctness" is fuzzier and more expensive to check than a typical unit test.

### 5.3 Evaluating Multi-Agent Systems Needs Per-Agent AND End-to-End Checks (Direct Callback to Day 8)

For an orchestrator/worker system, evaluating only the end-to-end outcome misses exactly the "hidden worker error" risk flagged in Day 8 §5.2 — a correct final outcome can mask one worker being systematically unreliable, as long as other workers or the orchestrator happen to compensate. Production eval for multi-agent systems typically needs BOTH: end-to-end task success rate, AND per-worker trajectory/quality checks in isolation (evaluating each worker's sub-task performance independently, the same way you'd unit-test a component before integration-testing the whole system).

### 5.4 Cost of Evaluation Itself (Direct Callback to Day 16)

Running eval — especially trajectory eval and LLM-as-judge, both of which involve additional LLM calls on top of the agent's own calls — has real cost, and running a large eval suite on every single code change can itself become a meaningful expense at scale. Common mitigation: tiered evaluation — a small, fast, cheap smoke-test suite runs on every change; a larger, more expensive full regression suite (including LLM-as-judge passes) runs on a schedule or before major releases, not on every commit — directly mirroring Day 16's sampling/tiering discipline, applied to evaluation infrastructure instead of production traffic.

---

## 6. Interview Q&A

**Q1: Why isn't task success rate alone sufficient to evaluate an agent?**
A: It measures whether the final outcome was correct but says nothing about HOW the agent got there — a trajectory that reached the right answer through irrelevant tool calls, stale data, or reasoning that doesn't actually connect to its own observations will still count as a full success. This "right for the wrong reasons" case is dangerous precisely because outcome-only evaluation reports it as a pass, giving false confidence that the underlying reasoning process is sound when it isn't — and that flawed process will likely produce wrong answers on inputs slightly different from what was tested.

**Q2: What's a specific pitfall of using an LLM to judge another LLM's outputs, and how do you mitigate it?**
A: LLM judges are subject to position/verbosity bias (favoring longer responses or being swayed by presentation order independent of actual quality) and can share the same blind spots as the model being judged, especially if it's the same model family — meaning the judge isn't a fully independent verification signal. Mitigations: use a different, ideally more capable model as the judge rather than self-judging; replace vague holistic rubrics ("rate 1-10") with specific, checkable per-criterion questions, which produces more consistent and diagnosable scores; and periodically calibrate the judge against a smaller set of human-reviewed spot checks for high-stakes evaluation.

**Q3: Give an example of a trajectory that would pass outcome evaluation but should fail trajectory evaluation.**
A: [Use the Austin headcount example.] An agent correctly retrieves current data from a reliable source (HR system: 142) but also unnecessarily searches a stale, less reliable source (a 2019 article: ~150), then averages the two rather than correctly prioritizing the more reliable current source — landing on an answer close enough to "correct" to pass a test case, but through reasoning that would produce a badly wrong answer if the stale source's number had been more divergent. The flaw was present regardless of the outcome; this specific test case just didn't happen to expose it.

**Q4: How should evaluation differ for a multi-agent orchestrator/worker system compared to a single agent?**
A: End-to-end task success rate alone can mask the "hidden worker error" risk from Day 8 — a correct final outcome can occur even when one worker is systematically unreliable, if other workers or the orchestrator happen to compensate for it. Multi-agent evaluation needs both end-to-end checks AND per-worker evaluation in isolation, similar to unit-testing each component before integration-testing the full system, so an unreliable individual worker doesn't stay hidden behind a compensating overall result.

**Q5: Why does evaluation need to run continuously rather than just once before launch?**
A: An agent's behavior can drift with no code changes on your end — the underlying model provider updates model versions, a knowledge base goes stale, a downstream tool's API response format changes. Evaluation needs to function as an ongoing regression check, similar to CI/CD in traditional software, catching drift as it happens rather than assuming a one-time pre-launch pass remains valid indefinitely.

**Q6: What's the actual hard part of building an agent evaluation system — is it writing the eval-running code?**
A: No — the hard, underrated part is curating a representative test set with reliable ground truth. A test set skewed toward easy or common cases reports misleadingly high success rates that don't hold up on the harder or rarer cases real production traffic contains. Production teams typically build test sets iteratively from real observed failures (each production incident becoming a new regression case) rather than trying to fully anticipate every case upfront.

---

## 7. Summary Card

- **Task success rate** (outcome eval): easy to compute, but structurally can't detect "right for the wrong reasons" — a correct answer via a flawed process still passes.
- **Trajectory eval**: checks the PROCESS — tool usage, redundancy, whether reasoning is actually grounded in gathered observations — catches the dangerous "correct outcome, bad process" quadrant that outcome eval misses entirely.
- **LLM-as-judge**: useful for open-ended outputs with no exact-match target, but subject to position/verbosity bias and shared blind spots with the model it's judging; mitigate with a different/stronger judge model, specific per-criterion rubrics instead of vague ratings, and periodic human calibration.
- Multi-agent systems need BOTH end-to-end and per-worker evaluation — end-to-end alone can hide a systematically unreliable individual worker.
- The real bottleneck is test-set curation (representative, reliable ground truth), not the eval-running mechanics — build test sets iteratively from real production failures.
- Evaluation must run continuously (drift happens even with no code changes), tiered by cost (Day 16) — cheap smoke tests on every change, expensive full suites on a schedule.

---
*Next: Day 19 — State & Context Management at Scale (context window pressure, summarization strategies, checkpointing).*
