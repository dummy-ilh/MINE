# Chapter 10 — Practical Synthesis

This is the last chapter, and it's designed to be a rehearsal, not new material. Everything here reuses the loan-screening example and numbers from Chapters 2, 3, and 7 — the goal is to walk it end-to-end, the way an interviewer would actually push you through it in a single extended conversation, then to give you a general decision framework and a set of practice questions with strong answer structures.

## 10.1 End-to-end worked case: diagnose → choose → mitigate → document

**Setup (from Chapter 2):** a loan-screening model, two groups (A and B), 100 people each. Group A's true default-avoidance rate ("positive" = will repay) is 40%; Group B's is 20% — genuinely different base rates. At the default global threshold of 0.50: TPR(A)=0.75, TPR(B)=0.60, FPR(A)=0.20, FPR(B)=0.10, PPV(A)≈0.71, PPV(B)=0.60.

**Step 1 — Diagnose (Chapters 2–4).** Don't start with "is this model biased?" — start by computing the full per-group confusion-matrix breakdown (Chapter 2) and checking every metric from Chapter 3: demographic parity (42% vs 20% approval — a 22-point gap), equalized odds (15-point TPR gap, 10-point FPR gap), and predictive parity (11-point PPV gap). All three are violated, which is expected given the base-rate difference (Chapter 3, §3.5) — the diagnostic step isn't to find *the* problem, it's to lay out *all* the gaps so the next step can be a deliberate choice rather than a reflexive fix of whichever number looks worst.

**Step 2 — Choose a target metric (Chapter 4, §4.1).** For lending specifically: a missed qualified borrower (false negative) denies someone a real opportunity and, at scale, reinforces the exact historical exclusion pattern from Chapter 1; a false positive costs the lender money but doesn't compound the same way across a protected group. That reasoning points toward prioritizing **equal opportunity** (equal TPR) as the primary target, while still monitoring FPR and PPV rather than ignoring them — this is a stated, defensible choice, not "the fair option," and it should be written down as a decision, not left implicit.

**Step 3 — Mitigate (Chapters 5–7).** Given that this is an already-trained model with a retraining cost, and the identified gap is specifically a TPR gap, **post-processing threshold adjustment** (Chapter 7, §7.1) is the fastest lever: lowering Group B's threshold from 0.50 to ~0.44 closes the TPR gap (both groups land at TPR=0.75), at the cost of a small increase in Group B's FPR that must now be explicitly re-measured and reported, not assumed away. If the resulting FPR gap turns out to be unacceptably large, that's the point where you'd escalate to an in-processing approach (Chapter 6) — e.g., an equalized-odds-targeted adversarial setup that lets the training objective search a broader space of models than a fixed model's threshold alone can reach, potentially finding a better joint TPR/FPR tradeoff than post-processing can.

**Step 4 — Document (Chapter 9).** The resulting Model Card states plainly: intended use (preliminary screening with human review, not automated final approval), per-group metrics after mitigation (TPR now equal at 0.75; FPR gap of ~X points remaining and explicitly flagged as unmitigated), and the reasoning for choosing equal opportunity over full equalized odds. This goes through a governance review before deployment, and the whole chain — data, chosen metric, mitigation method, remaining known gaps — is versioned together, so a year from now, anyone asking "why does this model behave this way" has an answer that doesn't depend on anyone's memory.

**Before/after summary:**

| Metric | Before (threshold=0.50 for both) | After (Group B threshold ≈0.44) |
|---|---|---|
| TPR(A) / TPR(B) | 0.75 / 0.60 | 0.75 / 0.75 ✓ |
| FPR(A) / FPR(B) | 0.20 / 0.10 | 0.20 / ~0.13 (small new gap, documented) |
| Approval rate(A) / (B) | 42% / 20% | 42% / ~23% (still a large gap — demographic parity was never the target here, and that's a stated choice) |

## 10.2 A general decision framework

Boiled down to a repeatable sequence, for any new scenario an interviewer hands you:

1. **Measure first, broadly.** Compute demographic parity, equalized odds/equal opportunity, and predictive parity/calibration gaps — all of them — before deciding anything (Chapters 2–4). Slice by intersections where sample size allows (Chapter 4, §4.2–4.3).
2. **Ask what a false positive costs, and who bears it. Ask the same for a false negative.** This single question, more than any formula, determines which metric to prioritize (Chapter 4, §4.1).
3. **Ask whether the score will be interpreted by a human downstream**, which favors prioritizing calibration, versus used purely as an automated gate, which favors TPR/FPR-based metrics.
4. **Pick a mitigation stage based on constraints, not preference:** need a fast fix with no retraining budget → post-processing (Chapter 7). Need finer control over the tradeoff and retraining is feasible → in-processing (Chapter 6). Fixing it at the root, cheaply, and it's a data/proxy problem specifically → pre-processing (Chapter 5).
5. **Re-measure after mitigating — all metrics, not just the one you targeted** — because fixing one (per Chapter 3's impossibility result) can move another.
6. **Document the choice, not just the outcome:** which metric, why, what tradeoff was accepted, what gaps remain unresolved and why (Chapter 9). This step is what separates a defensible decision from a lucky-looking number.

## 10.3 Practice interview questions with strong answer structures

**"How would you detect and fix bias in a hiring model?"**
Structure your answer in the same six-step order as 10.2: start with measurement (per-group and intersectional TPR/FPR/approval-rate breakdown), name the specific cost asymmetry for hiring (missed candidates vs. wasted interview slots — Chapter 4, §4.1), pick equal opportunity as a likely default with justification, pick a mitigation stage based on stated constraints (if the interviewer says "the model is already in production and we need something fast," go straight to post-processing), then close with documentation. Don't jump straight to "I'd use adversarial debiasing" — naming a technique before establishing the diagnosis and the chosen metric reads as pattern-matching rather than reasoning.

**"The model is calibrated but has an FPR gap — is it fair?"**
This is testing whether you know the impossibility result (Chapter 3, §3.5) cold. Answer: it depends which definition of "fair" you mean, and if base rates differ across groups, you should *expect* calibration and equalized-FPR to disagree — that's not a bug, it's the mathematical structure of the problem. Then name COMPAS explicitly as the real-world instance of exactly this disagreement (Chapter 1, §1.3).

**"How would you fix a model without retraining it?"**
Post-processing (Chapter 7) — walk through per-group threshold adjustment concretely, and proactively flag the disparate-treatment legal question (Chapter 7, §7.4) before the interviewer has to ask about it. Volunteering that caveat signals you understand the topic isn't purely technical.

**"What's the difference between demographic parity and equalized odds, and when would you use each?"**
Give the one-line formal definition of each (Chapter 3, §3.1 and §3.2), then immediately ground it in a use-case contrast — e.g., "demographic parity for equal exposure in ad delivery, where the harm is about access to information; equalized odds for loan screening, where the harm is about being wrongly denied or wrongly approved" — definitions alone, without a use-case anchor, tend to sound memorized rather than understood.

**"Walk me through how you'd write documentation for a fairness-sensitive model."**
Model Card (what it's for, per-group metrics, known caveats) plus Datasheet (where the data came from, how it was collected/labeled, known collection biases) as two distinct documents covering the model and the data respectively (Chapter 9, §9.1–9.2) — and mention that the Model Card should explicitly state unmitigated, known gaps rather than only reporting favorable numbers, since that honesty is exactly what a governance reviewer or regulator is checking for.

---

**That's all ten chapters.** You now have the full arc: why bias enters a pipeline (Ch1) → the notation everything is built on (Ch2) → the formal metrics and why they conflict (Ch3) → measuring them properly in practice (Ch4) → three mitigation stages (Ch5–7) → reasoning about the cost (Ch8) → documenting and governing the decision (Ch9) → and pulling it all together under interview conditions (Ch10). If you want, I can also put together a condensed one-page cheat sheet of formulas and definitions for last-minute review before an interview.
