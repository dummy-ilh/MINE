# Chapter 1 — Why Fairness Is Hard (Motivation & Framing)

## 1.1 What do we even mean by "bias"?

In everyday language, "bias" usually means prejudice — someone treating a person unfairly because of who they are. In machine learning, the word gets used for something related but more specific and more slippery:

> **Statistical bias**, in the ML sense: a systematic difference in how a model behaves for one group of people versus another, where "behaves" is measured by some numeric quantity (an error rate, an approval rate, a predicted score).

Notice what's missing from that definition: intent. A model doesn't need anyone to be prejudiced for it to end up biased. Nobody at a bank sits down and writes "deny more loans to group X." Bias creeps in through the data, the objective function, and the way the system is built — often while every individual step looks completely reasonable in isolation.

This is the first mental shift you need for this topic: **fairness problems are usually emergent, not intentional.** That's actually what makes them hard — there's no single villain to catch, just a pipeline that quietly accumulates distortion at every stage.

## 1.2 Where does bias actually enter the pipeline?

Think of an ML system as a pipeline with several stages. Bias can be introduced at *any* of them, and the fixes are different at each stage (this is why Chapters 5–7 split mitigation into pre/in/post-processing).

```
 [Real World]
      │
      │  historical inequities already exist here
      ▼
 [Data Collection] ── who gets measured, who gets left out
      │
      ▼
 [Labeling] ───────── whose judgment defines "ground truth"
      │
      ▼
 [Feature Engineering] ── which signals are kept, which proxy for sensitive attributes
      │
      ▼
 [Model Training] ──── the objective function optimizes average performance,
      │                 not fairness, unless you tell it to
      ▼
 [Deployment] ──────── model decisions affect the world
      │
      ▼
 [Feedback Loop] ───── the model's own decisions become tomorrow's training data
      │
      └──────────────► back to [Real World]
```

Let's walk through each stage with a concrete example: a model that screens loan applications.

**1. Historical inequities already exist in the real world.**
Before any data scientist touches anything, the world already has unequal outcomes — unequal access to credit history, generational wealth gaps, redlining's long tail. This isn't something ML created. But ML trained on historical outcomes will faithfully learn and reproduce these patterns unless someone actively intervenes.

**2. Data collection.**
Suppose loan applications are only well-documented for people who already have a banking relationship. Anyone outside the traditional banking system is underrepresented in your training data — not because you excluded them on purpose, but because your data collection process only "sees" people who were already inside the system.

**3. Labeling.**
Your label might be "did this person default?" — but what if the *real* signal you want is "would this person have repaid, if given the loan?" Historical labels only exist for people who were *approved*. People who were denied never get a chance to prove they'd repay. This is called **selection bias in labeling**, and it's subtle: your ground truth itself is a product of past (possibly biased) decisions.

**4. Feature engineering.**
Suppose you drop "race" as a feature because that seems like the responsible thing to do. But zip code, school attended, or first name can act as a **proxy** — a feature that's highly correlated with a protected attribute even though it isn't literally that attribute. Removing the "obvious" sensitive feature does very little if ten other features encode the same information.

**5. Model training.**
By default, a model trained with plain accuracy (or log-loss) as its objective is optimizing for *overall* performance. If one group is 80% of your data and another is 20%, the model can get great overall accuracy mostly by doing well on the majority group, even if it does poorly on the minority group — because the loss function has no idea groups exist unless you tell it.

**6. Deployment.**
A model with a bias baked in now makes real decisions — approvals, denials, interest rates — for real people. The scale problem kicks in here: a human loan officer's individual bias affects dozens of applicants; a deployed model's bias affects millions, instantly and consistently.

**7. Feedback loop.**
Here's the part that makes things compound over time. Suppose the model under-approves group X. Next year, when you retrain, group X has even less repayment history in your dataset (because they were denied loans and never got the chance to build that history). The model "confirms" its own prior belief that group X is riskier — not because it's true, but because the model's own past decisions shaped the data it's now learning from. This is sometimes called a **runaway feedback loop**, and it's one of the reasons bias can get *worse* over successive retraining cycles if nobody intervenes.

## 1.3 Three real-world cases (just enough to motivate the math)

You don't need deep legal knowledge of these for an interview — you need to be able to name-drop them accurately and connect them to a concept.

- **COMPAS (criminal recidivism risk scores):** ProPublica's 2016 analysis found that Black defendants who did *not* reoffend were flagged as high-risk far more often than white defendants who didn't reoffend — a false positive rate disparity. The tool's maker countered that the scores were well-*calibrated* within each group (a high-risk score meant roughly the same reoffense probability regardless of race). **Both sides were right about their own metric** — and this is the single best real-world illustration of the impossibility result you'll meet in Chapter 3: you can satisfy equalized-FPR fairness or calibration fairness, but generally not both at once when base rates differ across groups.

- **Amazon's scrapped hiring tool:** trained on ~10 years of resumes submitted to Amazon, which were mostly from men (reflecting the tech industry's existing gender skew). The model learned to penalize resumes containing the word "women's" (as in "women's chess club captain") and downgraded graduates of two all-women's colleges. This is a clean example of **historical data bias**: the training labels ("who succeeded here historically") were themselves shaped by a non-representative past.

- **Facial recognition accuracy gaps:** Buolamwini & Gebru's "Gender Shades" study found commercial facial-analysis systems had much higher error rates on darker-skinned women than on lighter-skinned men. This traces back to **data collection bias** — benchmark datasets were overwhelmingly composed of lighter-skinned faces, so models optimized for average performance on those datasets simply never had to get darker-skinned faces right.

## 1.4 The central tension (preview of Chapter 3)

Here's the idea to hold onto going into the next chapter, stated informally:

> If two groups have different **base rates** (different true proportions of the outcome you're predicting — e.g., different true default rates, different true reoffense rates), then it is mathematically impossible, in general, for a model to simultaneously have: (a) equal positive-prediction rates across groups, (b) equal error rates across groups, and (c) equal calibration across groups.

You don't need to prove this yet — Chapter 3 will walk through a small worked numeric example that makes it concrete. For now, the point is: **fairness in ML is not a single checkbox.** It's a set of different, individually reasonable definitions that mathematically compete with each other. Picking "the fair one" is actually picking which tradeoff you're willing to accept — and that choice depends on the use case, not on the math alone.

## 1.5 What this means for how you should think about the rest of this topic

Three mental habits to carry forward through the rest of the chapters:

1. **Always ask "fair according to which metric?"** before agreeing that a model is or isn't biased. "The model is biased" is an incomplete sentence — "the model has a 15-point FPR gap between groups" is a complete one.
2. **Always ask "biased compared to what?"** — a model can be an *improvement* over the human process it replaced while still having a measurable group disparity. Both facts can be true at once.
3. **Always locate the fix in the pipeline.** "Bias" isn't one bug in one place — it's a property that can enter at data collection, labeling, features, training, or deployment, and the right fix depends on where the problem actually originates.

---

**Next: Chapter 2 — Defining Groups and Setting Up Notation**, where we'll formalize protected attributes, true labels, predictions, and scores, and refresh the confusion matrix through a per-group lens — the notation every metric in Chapter 3 is built on.
