# Model Deployment & MLOps — Interview Prep Syllabus
### Target: Google / Apple MLE interviews · Assumes zero prior knowledge

---

## How this syllabus is structured

Each module below is a future "chapter" (matching the format of your other prep material — numbered, .md, plain-language, diagrams where useful). Modules are ordered so each one builds on the last. Rough time estimates assume you're studying alongside your other prep tracks, not full-time.

By the end, you should be able to: (a) design an end-to-end deployment pipeline for a given ML system on a whiteboard, (b) explain every tradeoff an interviewer probes on, and (c) answer "how would you know if this model broke in production" without hesitating.

---

## Module 0 — Why MLOps exists (30 min, context-setting)
- What's different about deploying ML vs. deploying regular software (the model is a function learned from data, not written by a human — so its "correctness" can silently decay)
- The ML system lifecycle end to end: data → training → evaluation → packaging → deployment → serving → monitoring → retraining (the loop)
- Why interviewers ask MLOps questions at all — they're testing whether you've shipped something real, not just trained a notebook model

## Module 1 — Model & Data Versioning
- Why versioning ML artifacts is harder than versioning code (data + code + hyperparameters + environment all need to be reproducible together)
- Model registries — what they are, what metadata they track (lineage, metrics, approval status)
- Data versioning tools and concepts (DVC-style content-addressed data, dataset snapshots)
- Experiment tracking vs. model registry vs. artifact store — the three layers and how they connect
- Reproducibility: pinning environments, random seeds, deterministic training

## Module 2 — Packaging & Containerization
- Why models get containerized (environment consistency, dependency isolation, portability across dev/staging/prod)
- Docker basics as they apply to ML serving (image layers, base images for GPU workloads)
- Model serialization formats and why format choice matters (framework lock-in, portability, ONNX as an interchange format)
- Model servers: TensorFlow Serving, TorchServe, NVIDIA Triton, and what problem each solves (batching, multi-model serving, hardware acceleration)

## Module 3 — CI/CD for ML (the biggest conceptual jump from normal CI/CD)
- Why ML CI/CD needs an extra dimension: code changes AND data changes AND model changes each need their own pipeline trigger
- What "continuous training" (CT) means and how it differs from continuous integration/deployment
- Automated testing for ML: unit tests for code, data validation tests, model quality gates (won't ship if metric drops below threshold)
- Pipeline orchestration concepts (DAGs, dependency graphs between data prep → train → eval → deploy steps)
- Feature stores — what problem they solve (train/serve consistency, feature reuse across teams) and how they fit into the pipeline

## Module 4 — Training-Serving Skew (a favorite interview topic)
- Definition: when the data/features seen at training time differ from what the model sees at serving time
- The three classic causes: (1) different code paths computing features offline vs. online, (2) data distribution shift over time, (3) subtle bugs like time-travel leakage (using future info at training that isn't available at serving)
- How feature stores mitigate this (single source of truth for feature computation)
- How to detect skew in an interview answer: compare offline vs. online feature distributions, log serving-time features and diff against training data
- Worked example: recommend a debugging process for "model works great offline, degrades in production"

## Module 5 — Deployment Strategies
- **Shadow deployment**: new model runs in parallel on live traffic, predictions logged but not served — used to validate before any user sees it
- **Canary release**: new model serves a small % of real traffic, monitored closely, rolled out gradually if healthy
- **Blue-green deployment**: two full environments, instant traffic switch, instant rollback
- **A/B testing**: comparing model versions on business/product metrics, not just ML metrics — statistical significance, guardrail metrics, novelty effects
- When to use which strategy — this is a classic "design this rollout" interview prompt, so be ready to justify tradeoffs (risk tolerance, traffic volume, how fast you need signal)
- Rollback strategy: what triggers an automatic rollback, and why rollback plans must be decided *before* deployment, not improvised after

## Module 6 — Latency vs. Accuracy Tradeoffs
- Why serving latency is a first-class constraint (SLAs, user experience, cost)
- Model compression techniques and what each trades away: quantization, pruning, distillation
- Batching strategies for throughput vs. per-request latency
- Caching predictions vs. always computing fresh
- Hardware tradeoffs: CPU vs GPU vs specialized accelerators, and when each makes sense
- How to frame this in an interview: always state the *product* constraint first (real-time fraud detection vs. overnight batch scoring need wildly different answers)

## Module 7 — Monitoring & Observability
- What to monitor beyond "is the server up": prediction distribution, feature distribution, latency percentiles (p50/p95/p99), business metrics
- **Data drift** vs **concept drift** — the distinction interviewers love to probe (input distribution changing vs. the relationship between inputs and target changing)
- Drift detection methods conceptually (statistical distance between distributions, population stability index)
- Alerting design: what threshold, who gets paged, what's the escalation path
- Logging predictions for later auditing/retraining — and the privacy/PII considerations that come with it

## Module 8 — Scaling & Infrastructure
- Horizontal vs vertical scaling for model serving
- Load balancing across model replicas
- Autoscaling based on traffic/queue depth
- Multi-region serving and the consistency/latency tradeoffs it introduces
- Cost considerations — why "just use the biggest GPU" is usually the wrong interview answer

## Module 9 — Model Governance & Responsible Deployment
- Approval workflows before a model can ship (who signs off, what evidence is required)
- Auditability — being able to answer "which model version made this prediction, on what data, when"
- Fairness/bias monitoring in production, not just at training time
- Compliance considerations (varies by domain — healthcare, finance, ads all have different bars)
- Explainability requirements for certain deployment contexts

## Module 10 — Retraining & the Feedback Loop
- Trigger types for retraining: scheduled (e.g. weekly), performance-triggered (metric drops below threshold), data-volume-triggered
- Online learning vs. batch retraining — when continuous updates make sense vs. when they're dangerous (feedback loops, poisoning)
- Human-in-the-loop review before promoting a retrained model
- Closing the loop: how production logs become the next training set

## Module 11 — System Design Synthesis (capstone)
- Practice designing full deployment architectures for classic prompts:
  - "Design the deployment pipeline for a fraud detection model that needs sub-100ms latency"
  - "Design an A/B testing framework for a recommendation model"
  - "How would you detect and respond to a model silently degrading in production?"
  - "Design a system where a new model can be rolled back within 60 seconds of a bad metric"
- Framework for structuring any MLOps system design answer: requirements → constraints → data flow → deployment strategy → monitoring → failure modes

---

## Suggested study order
1. Modules 0–3 (foundations — versioning, packaging, CI/CD) — these give you vocabulary for everything else
2. Modules 4–6 (skew, deployment strategies, latency/accuracy) — these are the most commonly asked *conceptual* interview topics
3. Modules 7–8 (monitoring, scaling) — these come up heavily in system design rounds
4. Modules 9–10 (governance, retraining) — less common but signal seniority when you bring them up unprompted
5. Module 11 last, once the vocabulary is solid — this is where it all gets tested together

## What "mastery" looks like for this topic in an interview
- You can draw the full lifecycle diagram from memory in under 2 minutes
- For any deployment strategy question, you name the strategy, explain the tradeoff, and pick one based on stated constraints (not just list all four)
- You can explain training-serving skew with a concrete example, not just the definition
- You default to asking about latency/scale/risk-tolerance requirements before answering a system design question, rather than jumping to a solution

---

*Next step: I can expand any module into a full standalone chapter — same depth/style as your RNN and optimization chapters — with worked examples and diagrams. Just tell me which one to start with.*
