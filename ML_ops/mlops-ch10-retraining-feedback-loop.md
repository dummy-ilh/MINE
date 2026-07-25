# Chapter 10 — Retraining & the Feedback Loop

*(Module 10 of the syllabus)*

---

## 1. Closing the loop from Chapter 1

Go back to the lifecycle diagram from Chapter 1: Monitoring feeds back into Retraining, which feeds back into Data collection, restarting the whole cycle. Every chapter since then has been about individual pieces of that loop — versioning, packaging, deployment, latency, monitoring, governance. This chapter is about the piece that actually *closes* it: **when and how do you decide to train a new model, and get it safely back into production?**

This matters because everything you've built — drift detection (Ch7), deployment strategies (Ch5), governance (Ch9) — exists in service of this loop running continuously and reliably, not as a one-time launch.

---

## 2. What triggers a retrain?

Three distinct trigger types, and you should be able to name the situation each one fits:

### Scheduled retraining
**What it is:** retrain on a fixed cadence — e.g., every week, every month — regardless of whether monitoring has flagged a specific problem.

**When this makes sense:** when you expect gradual drift as a matter of course (the world naturally shifts over time, per Chapter 7's data drift discussion) and a predictable, regular refresh is simpler to operate and reason about than trying to precisely detect the exact moment retraining becomes necessary. Simple to build, easy to reason about, but not responsive to sudden problems between scheduled runs.

### Performance-triggered retraining
**What it is:** retrain when a monitored metric (from Chapter 7's Layer 3 or Layer 4) crosses a pre-defined degradation threshold — not on a calendar, but reactively, exactly when needed.

**When this makes sense:** for models where drift is unpredictable in timing (per Chapter 7's concept drift discussion — e.g., adversarial fraud patterns that can shift suddenly, not gradually), waiting for a scheduled window could mean running a degraded model for longer than necessary. This connects directly to the alerting design from Chapter 7 — the same thresholds that trigger a human alert can also trigger an automated retraining pipeline.

### Data-volume-triggered retraining
**What it is:** retrain once a meaningful amount of *new* labeled data has accumulated since the last training run, regardless of a fixed calendar or a detected performance drop.

**When this makes sense:** for models where more recent, representative data is inherently valuable to incorporate (e.g., a new product category launched, generating a wave of new-but-relevant training examples) — the trigger is "there's enough new signal worth learning from," independent of whether current performance has degraded yet.

**How to choose in an interview answer:** state which trigger type fits based on how the domain actually drifts — steady/gradual → scheduled is reasonable; sudden/adversarial → performance-triggered is essential; data-availability-driven → volume-triggered fits. A strong answer can also combine multiple trigger types (e.g., scheduled as a baseline safety net, plus performance-triggered as an early-response layer on top) rather than treating it as an exclusive choice.

---

## 3. Online learning vs. batch retraining

This is a genuinely important distinction, and interviewers often test whether you understand *why* one is riskier than the other, not just that they exist.

### Batch retraining
**What it is:** the standard approach covered implicitly throughout this syllabus — periodically (on whatever trigger) train a new model from a batch of accumulated data, evaluate it offline, and deploy it through the normal pipeline (registry approval, canary/shadow, monitoring).

### Online learning
**What it is:** the model continuously updates its weights in near-real-time as new data/feedback arrives, rather than being retrained in discrete batches.

**Why online learning is appealing:** it can adapt to change much faster than waiting for the next batch retraining cycle — genuinely valuable for fast-moving domains like adversarial fraud detection.

**Why online learning is dangerous, and specifically why:** because updates happen continuously and immediately, there's no equivalent of the offline evaluation checkpoint (Chapter 1) or the approval gate (Chapter 9) sitting between "new data arrives" and "the model's behavior changes in production." This creates two specific risks worth naming:

- **Feedback loops** — if the model's own predictions influence future data (a very common situation — e.g., a recommendation model's picks shape what users click on, which becomes training data for the next update), the model can reinforce and amplify its own biases or mistakes over time, in a self-reinforcing spiral, with nothing forcing a pause to catch it.
- **Poisoning** — if any bad or manipulated data reaches the model (whether malicious, e.g. an adversary intentionally feeding bad inputs to corrupt the model, or just a data pipeline bug), online learning incorporates it into the live model *immediately*, with no offline evaluation step to catch the problem before it affects real users. Batch retraining's offline evaluation and approval gate act as a natural checkpoint that would likely catch a poisoning attempt before it ever reaches production — online learning removes that checkpoint by design.

**The interview-ready framing:** online learning trades safety/reviewability for responsiveness. It's the right call only when the domain's rate of change genuinely can't wait for a batch cycle *and* you've built specific safeguards to compensate for the missing offline checkpoint (e.g., automated sanity bounds on how much any single update is allowed to shift the model). Proposing online learning without acknowledging this tradeoff is a weak answer; batch retraining is the safer, more common default for good reason.

---

## 4. Human-in-the-loop review before promotion

Even for automatically-triggered retraining, a well-governed pipeline (connecting back to Chapter 9) typically still includes a **human review checkpoint** before a freshly retrained model is promoted to full production — the automated trigger decides *when* to retrain, but a human (or at minimum, automated criteria a human has pre-approved) still decides whether the *result* is good enough to actually replace the current production model.

**Why this matters even when everything is "automated":** an automated pipeline can automatically produce a *bad* model just as easily as a good one — automation removes manual toil from the process, but it shouldn't remove judgment about the outcome. This is the same principle as Chapter 9's approval workflow, just applied specifically at the retraining step of the loop rather than treated as a one-time launch gate.

---

## 5. Closing the loop: production logs become the next training set

Here's the piece that makes this genuinely a *loop* rather than a one-way pipeline: the predictions and outcomes logged during monitoring (Chapter 7) — what the model predicted, and eventually what actually happened (ground truth) — become the raw material for the *next* round of training data. This is literally the arrow from "Monitoring" back to "Data collection" in the Chapter 1 diagram, now made concrete: it's not an abstract feedback arrow, it's specifically **production logs flowing back into the training data pipeline.**

**Why this deserves being stated explicitly in an interview:** it shows you understand that a production ML system isn't just "trained once, deployed, monitored for problems" — the monitoring output is *itself* an input to keeping the whole system healthy going forward, which is precisely why every earlier chapter's practices (careful logging, versioning, lineage) all pay off here: none of this works if you can't trust the quality and traceability of the data flowing back into retraining.

---

## 6. Common pitfall interviewers listen for

Proposing online learning as a default "more advanced/better" answer without acknowledging feedback loops and poisoning risk. Online learning sounds sophisticated, but reaching for it reflexively — rather than as a deliberate choice justified by a domain that genuinely needs that responsiveness, with real safeguards in place — reads as not having thought through the risk it introduces. Batch retraining with a solid trigger strategy (Section 2) and a human review gate (Section 4) is the correct default answer for the large majority of production systems, and should be your starting point unless the scenario specifically demands otherwise.

---

## Comprehension check

1. In your own words, explain why online learning removes a safety checkpoint that batch retraining naturally has — name the checkpoint specifically.
2. Give an example (can be different from the ones above) of a feedback loop where a model's own predictions influence its future training data, and explain how that could go wrong over time.
3. A fraud detection team says "we want to retrain automatically whenever our fraud-catch-rate metric drops below a threshold, with no human review before the new model deploys, so we can respond as fast as possible." What risk are they introducing, and what would you suggest instead?

Say "c11" when ready for **Chapter 11: System Design Synthesis** — the capstone chapter that ties everything together into full whiteboard-style answers.
