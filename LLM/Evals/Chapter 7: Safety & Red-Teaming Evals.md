# Chapter 7: Safety & Red-Teaming Evals

## Why this is a distinct category, not just "another axis on the rubric"

In Chapter 3 you saw "safety" listed as one rubric axis among several (fluency, relevance, safety, etc.) for general quality eval. That's fine for catching *accidental* bad outputs on normal prompts. But safety eval as its own discipline is about something structurally different: **actively searching for the worst-case inputs that make the model fail**, not just measuring average-case quality. That shift — from measuring typical behavior to hunting for adversarial failure — is why it gets its own toolkit.

## Two distinct threat models

**1. Passive harm (toxicity/bias) — the model says something bad on an ordinary or mildly leading prompt**, without anyone trying hard to provoke it. This is about the model's default behavior distribution.

**2. Active harm (jailbreaks/adversarial attacks) — a user deliberately, cleverly constructs input specifically designed to bypass safety training.** This is an adversarial, cat-and-mouse setting — the "attacker" is actively searching for weaknesses.

These need different evaluation methodologies, so let's take them in turn.

## Toxicity Evaluation

**Classifier-based scoring:** run model outputs through a trained toxicity classifier (e.g., Perspective API-style models) that outputs a probability the text is toxic/offensive/threatening. Aggregate across a large prompt set to get an overall toxicity rate.

**Worked example — the standard "expected maximum toxicity" methodology (from the RealToxicityPrompts line of work):** for a set of prompts (some neutral, some already somewhat toxic as a stress-test), generate k completions per prompt (say k=25), score each with the toxicity classifier, and report:
- **Expected Maximum Toxicity** = average, across prompts, of the *maximum* toxicity score seen among the k completions for that prompt.
- **Toxicity Probability** = fraction of prompts for which *at least one* of the k completions crossed a toxicity threshold (e.g., >0.5).

**Why max-over-k rather than average-over-k:** because in deployment, a single bad output shown to a single user is the failure event that matters — a model that's toxic 1% of the time it's sampled is still a real production risk, even if its *average* toxicity score looks low. Sampling multiple completions and taking the max simulates "how likely is it that a real user, across many interactions, eventually sees a bad one."

## Bias Evaluation

**Intuition:** does the model's behavior change in an unfair/stereotyped way when you vary a demographic-associated attribute in the prompt, holding everything else constant?

**The core methodology — counterfactual/template-based testing:** construct paired prompts that differ *only* in one demographic term, then measure whether the model's output differs systematically.

**Worked example.** Template: "[NAME] applied for the software engineering role. Their resume shows 5 years of experience. Should they be hired?" Fill NAME with a set of names statistically associated with different demographic groups, run each through the model, and compare:
- The model's hire/no-hire recommendation rate
- The sentiment/tone of the generated justification
- The associated confidence or enthusiasm language used

If recommendation rates or tone differ systematically across name groups despite identical resume content, that's a measured bias effect — you'd report it as, e.g., "name group X received a positive recommendation 92% of the time vs. 78% for name group Y, on otherwise-identical prompts," which is a concrete, quantifiable disparity rather than a vague "the model might be biased."

**Known benchmark families to name-drop:** BBQ (Bias Benchmark for QA) tests whether a model relies on social stereotypes when answering ambiguous questions about people from different groups; StereoSet and CrowS-Pairs measure whether a model assigns higher likelihood to stereotype-consistent vs. stereotype-inconsistent sentences.

## Jailbreak / Red-Teaming Evaluation

**The core idea:** don't wait for real attackers to find failures — proactively simulate them. This has two flavors:

**1. Manual/human red-teaming:** trained specialists (or contracted external red-teamers) creatively try to get the model to produce disallowed content, using tricks like role-play framing ("pretend you're an AI with no restrictions"), incremental escalation, obfuscation (encoding harmful requests in code, foreign languages, or ciphers), or exploiting long-context distraction.

**2. Automated/systematic red-teaming:** use another model (or search algorithm) to automatically generate and iterate on adversarial prompts at scale, since manual red-teaming can't cover the space exhaustively. A common pattern: an "attacker" LLM proposes a jailbreak attempt, observes whether it succeeded (checked by a classifier or judge), and iterates/mutates the prompt if it failed — an automated search loop over adversarial prompt space.

**The core metric: Attack Success Rate (ASR).**

$$ASR = \frac{\text{\# successful attacks}}{\text{\# total attack attempts}}$$

**Worked numerical example.** You run 200 known jailbreak techniques (from a curated attack library) against a model. 14 of them successfully elicit disallowed content (checked via a safety classifier or human review of the output). ASR = 14/200 = **7%**.

**Why ASR alone isn't the whole story — severity weighting matters:** a jailbreak that gets the model to use mildly crude language is not equivalent to one that extracts detailed harmful instructions. Mature red-teaming reports break ASR down by **severity tier** (e.g., low/medium/high-risk content categories) rather than a single blended number, because a 7% ASR could mean "7% of attacks got a mildly off-color joke" or "7% got detailed harmful content" — very different risk profiles that a single aggregate number hides.

## How this feeds back into the model (the eval → mitigation loop)

This is worth stating explicitly in an interview because it shows you understand safety eval isn't just a report card, it's part of a training loop:

1. Red-team / measure toxicity+bias → find failure cases
2. Failure cases become new training data (often for RLHF/safety fine-tuning, or direct preference data showing "don't do this")
3. Retrain / fine-tune
4. Re-run the same eval suite to confirm the fix worked *and* didn't regress general capability elsewhere
5. Repeat — this is why safety eval suites are run continuously, not once before launch

**Important caveat to raise proactively:** fixing one class of jailbreak often doesn't generalize — attackers adapt, and models can overfit to specific patterns seen during safety fine-tuning while remaining vulnerable to novel attack styles. This is why held-out, periodically refreshed red-team prompt sets matter — reusing the same fixed jailbreak library over and over will show flat/improving ASR that doesn't reflect real-world robustness, because the model may just be pattern-matching to that specific known set.

## Quick check

A model shows a toxicity classifier score of "average toxicity: 0.02" across 10,000 evaluation prompts — sounds very safe. What additional number would you want before trusting that this model is safe to deploy, and why?

**Expected maximum toxicity / toxicity probability at k samples**, not just the average. A low average can hide a model that's usually fine but occasionally (say, 1 in 500 generations) produces a severely toxic output — which is exactly the real-world failure mode that matters, since any single bad output shown to a user is the incident, regardless of how good the average looks. You'd want to know the tail behavior, not just the mean.

---

Chapter 8 is Statistical Rigor — confidence intervals, significance testing, and sample size/power calculations for eval results. Want me to continue?
