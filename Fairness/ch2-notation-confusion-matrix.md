# Chapter 2 — Defining Groups and Setting Up Notation

Chapter 1 gave you the intuition. Now we need precise notation, because every metric in Chapter 3 is really just "take a familiar quantity, compute it separately for each group, and compare." If the notation is solid, Chapter 3 becomes much easier — it's mostly just plugging into formulas you already understand.

## 2.1 The four quantities everything is built from

For any individual in your dataset, define four things:

- **A (the protected/sensitive attribute):** which group someone belongs to — e.g., A = 0 or A = 1 for a simple two-group case (gender, race, age bracket, etc.). Real problems often have more than two groups, or even overlapping/intersectional groups, but we'll start with two groups because the definitions are cleanest there, and generalize later.
- **Y (the true label):** what actually happened / the ground truth — did this person actually default, actually reoffend, actually deserve the loan. Y = 1 or Y = 0.
- **Ŷ ("Y-hat," the predicted label):** what the model decided — approve or deny, flag as high-risk or not. Ŷ = 1 or Y = 0. This is the *hard* decision, after any threshold has been applied.
- **S (the score):** the raw number the model outputs before thresholding — a probability like 0.73, or a risk score like 640. Ŷ is usually just "S ≥ threshold."

Keep these four apart in your head. A huge number of fairness questions in interviews come down to: *is this a metric about Y, Ŷ, or S?* Demographic parity is about Ŷ. Equalized odds is about Ŷ compared to Y. Calibration is about S compared to Y. Getting this distinction right is most of the battle.

## 2.2 The confusion matrix, refreshed

You've seen the confusion matrix before, but let's rebuild it in words, because fairness metrics are just this table computed twice (once per group) and compared.

For a given group, split every individual into one of four buckets based on (Y, Ŷ):

| | Ŷ = 1 (predicted positive) | Ŷ = 0 (predicted negative) |
|---|---|---|
| **Y = 1 (actually positive)** | True Positive (TP) | False Negative (FN) |
| **Y = 0 (actually negative)** | False Positive (FP) | True Negative (TN) |

"Positive" here just means "the outcome the model is trying to flag" — reoffend, default, has the disease, is a fraud case. It's a modeling convention, not a moral judgment about the group.

From these four counts, four rates matter most for fairness:

- **True Positive Rate (TPR)**, also called **recall** or **sensitivity**:
  TPR = TP / (TP + FN) — *"of the people who are actually positive, what fraction did the model correctly catch?"*

- **False Positive Rate (FPR)**:
  FPR = FP / (FP + TN) — *"of the people who are actually negative, what fraction did the model wrongly flag as positive?"*

- **Positive Predictive Value (PPV)**, also called **precision**:
  PPV = TP / (TP + FP) — *"of the people the model flagged as positive, what fraction actually were positive?"*

- **Negative Predictive Value (NPV)**:
  NPV = TN / (TN + FN) — *"of the people the model cleared as negative, what fraction actually were negative?"*

If any of these four feel shaky, sit with them before moving on — Chapter 3's fairness definitions are literally just "compute one of these per group and require them to match" (or in the case of calibration, a related-but-distinct condition on S rather than Ŷ, covered in 2.4 below).

## 2.3 A tiny worked numeric example

Say we have a screening model and two groups, Group A and Group B, each with 100 people.

**Group A (100 people):**
- 40 are actually positive (Y=1), 60 are actually negative (Y=0)
- Of the 40 positives: model catches 30 (TP=30), misses 10 (FN=10)
- Of the 60 negatives: model wrongly flags 12 (FP=12), correctly clears 48 (TN=48)

TPR(A) = 30 / (30+10) = 30/40 = **0.75**
FPR(A) = 12 / (12+48) = 12/60 = **0.20**
PPV(A) = 30 / (30+12) = 30/42 ≈ **0.71**

**Group B (100 people):**
- 20 are actually positive (Y=1), 80 are actually negative (Y=0) — a **different base rate** than Group A
- Of the 20 positives: model catches 12 (TP=12), misses 8 (FN=8)
- Of the 80 negatives: model wrongly flags 8 (FP=8), correctly clears 72 (TN=72)

TPR(B) = 12 / (12+8) = 12/20 = **0.60**
FPR(B) = 8 / (8+72) = 8/80 = **0.10**
PPV(B) = 12 / (12+8) = 12/20 = **0.60**

Just sit with these two tables for a second: Group A and Group B have **different base rates** (40% vs 20% actually positive) and the model, computed group-by-group, already shows a **15-point TPR gap** (0.75 vs 0.60) and a **10-point FPR gap** (0.20 vs 0.10) and an **11-point PPV gap** (0.71 vs 0.60) — even without anyone having done anything to the model on purpose.

This one example is the seed of everything in Chapter 3. Every fairness definition is a rule about which of these gaps you're allowed to have.

## 2.4 Base rates: the variable that causes all the trouble

The **base rate** of a group is simply P(Y=1 | A=a) — the true proportion of positives in that group. In the example above, Group A's base rate is 40%, Group B's is 20%.

Base rates differing across groups is the single most important fact in this whole topic, because (as Chapter 3 will show rigorously) **when base rates differ, you cannot generally equalize TPR/FPR and PPV/calibration at the same time.** Every "gotcha" fairness question in an interview — "why can't you just make the model fair?" — traces back to this one fact.

It's worth being precise about *why* base rates differ in the real world, because "the groups are just different" is not a satisfying or complete answer. Base rates can differ because of:
- genuine underlying differences (rare, and hard to establish causally),
- **measurement differences** — e.g., one group is policed/monitored/audited more, so more of their true positives get *recorded*,
- **historical feedback loops** (Chapter 1, section 1.2) — past biased decisions shape who has a chance to generate a "positive" outcome at all,
- or **label bias** — the label itself imperfectly proxies for what you actually care about (Chapter 1, section 1.2, the "defaulted vs. would-have-repaid" issue).

Keep this list handy — a strong interview answer distinguishes "the base rate differs" from "the base rate differs *because of ongoing measurement/historical distortion*," since the appropriate response is very different in each case.

## 2.5 Calibration: a metric about S, not Ŷ

One more piece of notation before Chapter 3, because calibration works differently from TPR/FPR/PPV.

**Calibration** asks: among all the people a model gave a score of, say, S=0.7, do about 70% of them actually turn out positive? Formally, for every score value s:

P(Y=1 | S=s) = s

**Calibration across groups** (sometimes called "predictive parity" when applied at a single threshold) asks whether this holds *equally well within each group* — i.e., a score of 0.7 means the same real-world probability of being positive whether you're in Group A or Group B.

This is subtly different from PPV: PPV is calibration evaluated at one specific threshold, collapsed into a single number. Calibration (properly speaking) is a claim about *the whole score distribution*, at every threshold, not just the one you happened to pick for your Ŷ decision. Both TPR/FPR and calibration matter, they just describe different failure modes — and (as you now have the tools to guess) they can conflict with each other whenever base rates differ across groups.

## 2.6 Quick self-check before Chapter 3

Before moving on, make sure you can answer these without looking back:
- What's the difference between TPR and PPV in plain language, not formulas?
- Why does a base-rate difference between groups matter so much for fairness metrics?
- Is calibration a statement about Y vs Ŷ, or about Y vs S?

If those are solid, Chapter 3 will feel like a natural continuation rather than new material — it's the same confusion-matrix machinery, formalized into named definitions, plus the numeric proof of why they conflict.

---

**Next: Chapter 3 — Group Fairness Metrics**, where we formally define demographic parity, equalized odds, equal opportunity, and predictive parity/calibration, and walk through the impossibility result using an extension of the Group A / Group B example above.
