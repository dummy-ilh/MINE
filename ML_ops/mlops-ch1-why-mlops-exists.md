# Chapter 1 — Why MLOps Exists & The ML System Lifecycle

*(Module 0 of the syllabus)*

---

## 1. Start with the core problem

Imagine you write a normal piece of software — say, a function that calculates tax owed given income. You write the logic yourself, line by line. If it's wrong, you go read the code and find the bug. The code *is* the logic. As long as the code doesn't change, the behavior doesn't change. Ever.

Now imagine a machine learning model that predicts "will this customer churn?" You didn't write the logic. You showed the model thousands of past examples, and it *learned* a pattern from that data. Nobody — not even you — can point to a single line and say "this is the rule that decides churn." The "logic" is baked into millions of numbers (weights) that were shaped by the data you happened to train on.

This one difference — **the logic comes from data, not from a human writing rules** — is the root cause of almost everything in MLOps. Let's pull on that thread.

### Consequence 1: the model can be "correct" today and "wrong" tomorrow, with no code change at all

Your tax function will compute the same answer forever, unless someone edits the code. But your churn model learned patterns from *last year's* customer behavior. If customer behavior shifts — say, a competitor launches a cheaper plan and churn patterns change — your model is now making decisions based on a world that no longer exists. Nobody touched the code. Nobody touched the model file. It just quietly stopped being accurate.

This never happens with regular software. It happens *constantly* with ML systems. This single fact is why "monitoring" and "retraining" are entire disciplines in ML that don't really exist in normal backend engineering.

### Consequence 2: reproducing a result requires much more than the code

If a teammate says "the tax calculator gave a wrong number," you just need the code and the input. You can reproduce it instantly.

If a teammate says "the model made a bad prediction," reproducing that requires: the exact model weights used, the exact version of the training data, the exact preprocessing code, the exact library versions, and often the exact hardware/random seed. Miss any one of these and you may never reproduce the issue. This is why **versioning** in ML means versioning data + code + config + environment together, not just code (we'll go deep on this in Chapter 2).

### Consequence 3: testing "is this correct" is fundamentally fuzzy

You can write a unit test that says `calculate_tax(50000) == 7500` and it will pass or fail with certainty forever. You cannot write a test that says `predict_churn(customer_X) == True` with that same certainty — the model is probabilistic, and "correct" is a statistical property measured over many examples, not a single deterministic fact. This changes how testing, deployment gates, and rollback decisions all have to work.

---

## 2. So what is "MLOps," actually?

Strip away the buzzword and MLOps is just this:

> **The set of practices and infrastructure that let you deploy a model to production, know whether it's still working, and safely update it — over and over, forever — given that the model's correctness silently depends on the real world matching the world it was trained on.**

It borrows heavily from DevOps (the discipline of reliably shipping and running normal software), but adds an extra dimension that pure code doesn't have: **data**. Every practice in MLOps is really answering one of these three questions:

1. **Can I reliably get a new/updated model into production?** → versioning, packaging, CI/CD, deployment strategies
2. **How do I know it's still working right now?** → monitoring, drift detection, observability
3. **What do I do when it stops working?** → rollback, retraining, governance

Everything in this syllabus is a more detailed answer to one of those three questions. Keep this three-question framework in your head — interviewers structure their probing around exactly this, even if they don't say so explicitly.

---

## 3. The full ML system lifecycle

Here's the loop, stage by stage. This is the diagram you should be able to redraw from memory.

```
 ┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌───────────┐
 │   Data       │ --> │ Training │ --> │ Evaluation │ --> │ Packaging │
 │ collection & │     │          │     │            │     │           │
 │ preparation  │     └──────────┘     └────────────┘     └───────────┘
 └─────────────┘                                                 |
        ^                                                        v
        |                                                 ┌────────────┐
 ┌──────────────┐                                          │ Deployment │
 │  Retraining  │                                          └────────────┘
 │  (trigger)   │                                                 |
 └──────────────┘                                                 v
        ^                                                 ┌────────────┐
        |                                                  │  Serving   │
 ┌──────────────┐                                          └────────────┘
 │  Monitoring  │ <----------------------------------------------┘
 └──────────────┘
```

Walk through each stage in plain language:

- **Data collection & preparation** — gathering raw data, cleaning it, computing features. This is where training-serving skew usually gets born (Chapter 4).
- **Training** — the model learns weights from the prepared data.
- **Evaluation** — checking the trained model's quality against held-out data *before* anyone trusts it. This is the last checkpoint before the model touches anything real.
- **Packaging** — turning the raw model weights into something that can actually be run as a service (containerizing it, wrapping it with a serving API). Chapter 2/3 territory.
- **Deployment** — the act of getting the packaged model in front of real traffic, using a strategy (canary, shadow, blue-green) that limits risk. Chapter 5.
- **Serving** — the model is now live, answering real requests, under real latency/throughput constraints. Chapter 6.
- **Monitoring** — watching the live model's inputs, outputs, and downstream business metrics to catch degradation. Chapter 7.
- **Retraining (trigger)** — when monitoring says "this model is stale or degrading," it feeds back into a new round of data collection and training. The loop closes. Chapter 10.

Notice: this is a **loop**, not a one-time pipeline. A model is deployed once and then lives through this cycle repeatedly, often automatically. That's the mental shift from "I trained a model" (a project) to "I run a model in production" (an ongoing system) — and that shift is exactly what separates an ML *researcher* mindset from an ML *engineer* mindset, which is precisely what these interviews are testing for.

---

## 4. Why interviewers actually ask this stuff

A quick but important point: when a Google or Apple MLE interviewer asks an MLOps question, they are rarely testing whether you memorized a tool name. They're testing whether you've internalized that **a model is a living system, not a finished artifact** — and whether you instinctively think about failure modes, staleness, and rollback rather than just "does it have good accuracy." A candidate who only talks about model architecture and never mentions monitoring or rollback reads as someone who has only ever worked in a notebook. That's the gap this whole syllabus is designed to close.

---

## Comprehension check — answer these before we move to Chapter 2

1. In your own words, why is "correctness" a moving target for an ML model but not for a normal function like a tax calculator?
2. If a teammate asked you "why can't we just unit-test the model like we test regular code?" — what would you say?
3. Looking at the lifecycle diagram: if a model's live accuracy starts dropping, which two stages of the loop are most directly responsible for catching that and fixing it?

Take a shot at these — I'll tell you where you're right, fill in any gaps, and then we'll move to **Chapter 2: Model & Data Versioning**.
