# Chapter 2 — Model & Data Versioning

*(Module 1 of the syllabus)*

---

## 1. Why this is harder than it sounds

In Chapter 1 we established: an ML model's behavior comes from data + code + config, all combined. So "versioning the model" can't just mean "put the code in git." Let's see exactly why, with a concrete scenario.

Say your churn model in production starts misbehaving. To investigate, you need to answer: *what exactly produced this specific model file?* That means you need to know, simultaneously:

- Which exact snapshot of the training data was used (not "the customers table" — that table changes every day; you need the exact rows as they existed at training time)
- Which exact version of the feature engineering code transformed that raw data into features
- Which exact version of the training code and hyperparameters (learning rate, number of epochs, etc.)
- Which exact library versions (a different version of a math library can change floating-point results slightly, and sometimes not-so-slightly)
- Which random seed was used, if the training process has any randomness

Miss any one of these five and you cannot reliably reproduce the model. This is the core difficulty: **normal software versioning (git) only solves one of these five — the code.** ML versioning needs to solve all five, together, as one linked unit.

---

## 2. The three layers of ML versioning

It helps to think of ML versioning as three separate but connected systems. Interviewers often ask "how would you version a model?" expecting you to name and connect these three:

### Layer 1: Data versioning
The idea: treat datasets like code — every meaningful version of a dataset gets a unique identifier, and you can always retrieve *exactly* the data that existed at that point in time.

The core technique underlying most tools here is **content-addressed storage**: instead of naming a file version "v1", "v2" (which is ambiguous and easy to overwrite), you compute a hash of the actual data content. If even one row changes, the hash changes. This gives you a tamper-evident, unambiguous fingerprint for "this exact dataset." Tools like DVC apply this same idea that git uses for code, but built for large data files that don't belong in git directly (data files are often huge; git handles that badly).

Plain-language mental model: **git tracks the history of your code's text. A data versioning tool tracks the history of your dataset's content, using the same "snapshot + hash" idea, but built to handle gigabytes of files efficiently.**

### Layer 2: Experiment tracking
While you're actively training and experimenting — trying different hyperparameters, different feature sets, different architectures — you generate *dozens or hundreds* of training runs. Experiment tracking is the system that logs, for every single run: what data version was used, what hyperparameters, what code version, and what the resulting metrics were.

Why this matters for interviews: without experiment tracking, "what's our best model?" becomes an untraceable question the moment more than one person is training models. Experiment tracking is what lets you answer "show me every run where we used learning rate 0.001 and got validation accuracy above 92%."

### Layer 3: Model registry
Once a training run produces a model you actually want to consider for production, it gets promoted from "just another experiment" into the **model registry** — a catalog of models that are candidates for or actively serving in production.

A model registry entry typically tracks:
- **Lineage** — which data version, code version, and experiment run produced this exact model (linking back to Layers 1 and 2)
- **Metrics** — the evaluation numbers that justified promoting it
- **Approval status** — is this model in "staging," "production," "archived," "rejected"?
- **Artifact location** — where the actual serialized model file lives

Think of the registry as the **single source of truth for "what is allowed to be deployed."** Training produces many candidate models; only ones that pass through registry approval should ever reach a serving system. This is also your audit trail: if someone asks "which model was serving predictions on March 3rd," the registry is where that question gets answered (this connects directly to Module 9 — governance).

---

## 3. How the three layers connect (the full picture)

```
 Data version (hash: abc123)
        │
        ▼
 Training run (tracked by experiment tracker)
   - code version: git commit xyz789
   - hyperparams: lr=0.001, epochs=50
   - metrics: val_accuracy=0.94
        │
        ▼
 Model artifact produced
        │
        ▼
 Promoted to Model Registry
   - status: "staging" → (after validation) → "production"
   - lineage: points back to data version abc123 + code commit xyz789
```

The key insight to say out loud in an interview: **the registry entry is not just a file pointer — it's a lineage record.** Given a production incident, you should be able to walk backward from "the model that's currently serving" all the way to "the exact rows of data it was trained on," with nothing in between left to guesswork.

---

## 4. Reproducibility — the practical checklist

Pulling this together, "reproducibility" in ML means you can answer yes to all of these:

1. Can I retrieve the *exact* data used, byte-for-byte, even if the source table has since changed?
2. Can I retrieve the *exact* code (including feature engineering, not just the model architecture)?
3. Do I know every hyperparameter and config value used?
4. Is the environment (library versions, hardware type where relevant) pinned somewhere?
5. If training involves randomness, was the seed logged?

A common interview framing: *"Your model was retrained and accuracy dropped 3 points overnight — walk me through how you'd debug it."* A strong answer explicitly uses this checklist: pull up the registry entry for both the old and new model, compare their linked data versions and hyperparameters, and isolate what changed. A weak answer vaguely says "I'd look at the logs."

---

## 5. Common pitfall interviewers listen for

Watch out for conflating these three terms, since interviewers will notice if you use them interchangeably:

| Term | What it actually tracks |
|---|---|
| Data versioning | Historical snapshots of datasets |
| Experiment tracking | Every training run's config + results, promoted or not |
| Model registry | Only the *candidate/approved* models, with lineage + deployment status |

A dataset version and an experiment run are not the same as a registered model — a registry entry is a *curated subset* of all experiments, specifically the ones judged worth deploying.

---

## Comprehension check

1. Explain in your own words why "content-addressed storage" (hashing data content) solves a problem that simply naming files "v1, v2, v3" does not.
2. What's the difference between something living in your experiment tracker vs. something living in your model registry?
3. Suppose two data scientists both trained a "final" churn model, and your team can't tell which one is actually running in production right now. Which of the three versioning layers failed, and what would you put in place to prevent this from recurring?

Answer when ready, or say "c3" to move straight to **Chapter 3: Packaging & Containerization**.
