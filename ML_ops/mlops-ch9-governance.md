# Chapter 9 — Model Governance & Responsible Deployment

*(Module 9 of the syllabus)*

---

## 1. Why governance is its own topic, not just "process overhead"

Everything so far has been about *making the system work* — deploy safely, serve fast, monitor well. Governance is about a different question: **can you prove, after the fact, that the system did what it was supposed to, and can you control what's allowed to reach production in the first place?** This matters most in exactly the moments things go wrong — an incident, a complaint, a regulatory inquiry — when "trust me, it's fine" isn't good enough and you need an actual, verifiable answer.

This is often underweighted by candidates who only think about ML as a technical problem, which is exactly why bringing it up unprompted signals seniority.

---

## 2. Approval workflows

**What it is:** a required review/sign-off step before a model can move from "trained and evaluated" to "actually serving production traffic" — connecting directly back to the model registry's "status" field from Chapter 2 (staging → approved → production).

**Why this matters beyond bureaucracy:** without a required checkpoint, there's nothing stopping an unvalidated or poorly-tested model from reaching real users, especially as a team grows and more people can trigger deployments. The approval step is the enforcement mechanism that makes all your evaluation/validation work (Chapter 1's Evaluation stage) actually binding, rather than optional.

**What a real approval step typically checks:** the required evaluation metrics were met, the model passed any fairness/bias checks (below), and — increasingly — that a canary/shadow validation (Chapter 5) was actually run and looked healthy before full rollout.

---

## 3. Auditability

**Core question governance needs to answer at any time:** *"Which model version made this specific prediction, on what data, and when?"*

Why this specific question, and not something vaguer: if a decision is later challenged (a user disputes a fraud flag, a regulator asks about a lending decision, an incident review needs to find the root cause), you need to trace that *exact* prediction back to an exact model version — and from there, back to that model's exact lineage (data version, code version — Chapter 2). Without this chain, you're left guessing, which is both an operational risk and, in regulated domains, a compliance failure on its own.

**What this requires in practice, pulling threads from earlier chapters together:**
- Every serving request is logged with which model version handled it (Chapter 7's prediction logging)
- Every model version's registry entry has full lineage back to its data and code (Chapter 2)
- Deployment events themselves are logged (when did version X start receiving traffic, when did it stop) — this is the piece that's easy to overlook, since it's not about the model or the data, it's about the *deployment timeline* itself.

---

## 4. Fairness/bias monitoring in production — not just at training time

**The gap this addresses:** it's common for teams to check a model for fairness/bias issues once, during initial evaluation, and then never again. But per Chapter 7's whole premise — models can drift — a model that passed a fairness check at launch is not guaranteed to still be fair six months later, especially if concept drift is occurring unevenly across different user subgroups.

**What "ongoing" fairness monitoring looks like conceptually:** tracking model performance metrics (not just overall, but broken out across relevant subgroups) on a continuing basis, the same way you'd track any other output metric in Chapter 7 — rather than treating fairness as a one-time gate passed at initial launch.

**Why to mention this even briefly in an interview:** it demonstrates you understand that responsible deployment isn't a checkbox exercise done once before shipping — it's part of the same continuous monitoring loop as everything else in this syllabus, connecting the governance topic back to the ongoing lifecycle from Chapter 1 rather than treating it as a separate, one-time process.

---

## 5. Compliance considerations

**The core point to convey, without needing deep domain-specific legal knowledge:** different domains carry very different regulatory bars, and a strong answer acknowledges this rather than proposing one-size-fits-all governance. A recommendation model for a shopping app and a model influencing a lending or healthcare decision face very different scrutiny and requirements — the second category typically demands far stronger auditability, explainability, and human review before deployment.

**What's safe and useful to say in an interview, without overclaiming specific legal expertise:** "the level of governance rigor should scale with the domain's regulatory and human-impact stakes — a system affecting people's access to credit or healthcare needs a stronger audit trail and more human oversight than a low-stakes recommendation system, even though the underlying MLOps mechanics are similar." This shows judgment without pretending to be a compliance expert.

---

## 6. Explainability requirements

**Why this connects to governance specifically:** in some deployment contexts — often exactly the higher-stakes ones from the compliance point above — it's not enough for a model to be accurate; you may be required to explain *why* it made a specific decision, in terms a human (a regulator, an affected user, an internal reviewer) can understand.

**Why this is a real tradeoff, not just an add-on feature:** some of the most accurate model types are also the hardest to explain in human-understandable terms, so explainability requirements can genuinely constrain what kind of model architecture is even viable for a given use case — this is worth mentioning if a question touches on model selection for a regulated or high-stakes domain, since it shows you understand explainability as a real design constraint, not a nice-to-have bolted on afterward.

---

## 7. Common pitfall interviewers listen for

Treating governance as separate from, or in tension with, engineering velocity — a weak answer implies governance is just red tape that slows shipping down. A strong answer frames it as *infrastructure*, the same category as monitoring or the model registry: something that, done well, is largely automated (approval gates checking pre-defined criteria, automatic lineage logging) rather than a slow manual process bolted on at the end. Good governance is built into the pipeline, not layered on top of it after the fact.

---

## Comprehension check

1. In your own words, why is "which model made this prediction, on what data" a question you need to be able to answer *after the fact*, not just something nice to have logged?
2. Explain why fairness checks done only once, at initial launch, aren't sufficient — connect this back to a concept from an earlier chapter.
3. A model is highly accurate but very hard to explain in human-understandable terms. Describe a realistic scenario where this tradeoff would actually block you from using that model, versus a scenario where it wouldn't matter much.

Say "c10" when ready for **Chapter 10: Retraining & the Feedback Loop**.
