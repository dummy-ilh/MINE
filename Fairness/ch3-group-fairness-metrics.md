# Chapter 3 — Group Fairness Metrics

This is the core chapter of the whole topic. Almost every interview question about fairness is really asking you to (a) name the right metric for a scenario, or (b) explain why you can't satisfy two metrics at once. Everything here builds directly on the A/Y/Ŷ/S notation and the Group A / Group B numbers from Chapter 2 — keep those numbers close, we're going to reuse and extend them.

## 3.1 Demographic Parity (Statistical Parity)

**Definition (plain language):** The model approves/flags people at the same *rate* in every group, regardless of whether those approvals are actually correct.

**Formal:** P(Ŷ=1 | A=a) is the same for every group a.

**Worked check against Chapter 2's numbers.** In our Group A / Group B example, let's compute the overall positive-prediction rate for each group:

- Group A: Ŷ=1 happens for TP+FP = 30+12 = 42 people out of 100 → P(Ŷ=1|A) = **42%**
- Group B: Ŷ=1 happens for TP+FP = 12+8 = 20 people out of 100 → P(Ŷ=1|B) = **20%**

That's a 22-point gap — demographic parity is badly violated here.

**What it captures:** a clean, easy-to-explain notion of "equal treatment at the gate" — useful when you want equal *representation* in outcomes (e.g., equal advancement rate through a hiring funnel), independent of whether the underlying qualification rates differ.

**What it ignores:** it says nothing about whether the approvals were *correct*. A model that approves 42% of Group A and 42% of Group B satisfies demographic parity even if it's approving the *wrong* 42% in one group and the *right* 42% in the other. This is demographic parity's most common criticism: it can force equal outcome rates onto groups with genuinely different base rates, which may mean approving less-qualified people in one group and rejecting more-qualified people in another, just to hit the same rate.

## 3.2 Equalized Odds

**Definition (plain language):** Among people who are *actually* positive, the model catches them at the same rate in every group (equal TPR) — *and* among people who are *actually* negative, the model wrongly flags them at the same rate in every group (equal FPR).

**Formal:** P(Ŷ=1 | Y=y, A=a) is the same across groups a, for both y=1 and y=0. Equivalently: TPR(A) = TPR(B) **and** FPR(A) = FPR(B).

**Worked check.** From Chapter 2: TPR(A)=0.75 vs TPR(B)=0.60 (15-point gap), FPR(A)=0.20 vs FPR(B)=0.10 (10-point gap). Both conditions are violated — this model does not satisfy equalized odds.

**What it captures:** "equal error rates" — no group is disproportionately burdened by false positives (wrongly flagged) or false negatives (wrongly cleared/missed). This is the metric at the center of the COMPAS controversy from Chapter 1 — ProPublica's critique was specifically an FPR-gap critique.

**What it ignores:** it says nothing about the raw approval rate, and (as you'll see in 3.5) it generally can't be satisfied alongside calibration when base rates differ.

## 3.3 Equal Opportunity

**Definition:** a relaxation of equalized odds that only requires the TPR condition, not the FPR one.

**Formal:** P(Ŷ=1 | Y=1, A=a) equal across groups (equal TPR only).

**Why this relaxation exists:** sometimes you care much more about one type of error than the other. In a "should we interview this candidate" model, missing a qualified candidate (a false negative) is arguably worse than accidentally interviewing an unqualified one (a false positive) — interviews are cheap, missed talent is expensive. Equal opportunity lets you focus fairness effort on the error type that matters most for the use case, rather than forcing both TPR and FPR to match, which is a strictly harder constraint to satisfy.

## 3.4 Predictive Parity and Calibration Across Groups

**Predictive Parity (a single-threshold version):** PPV is equal across groups.

**Formal:** P(Y=1 | Ŷ=1, A=a) equal across groups.

**Worked check.** From Chapter 2: PPV(A) ≈ 0.71, PPV(B) = 0.60 — an 11-point gap. Predictive parity is violated too.

**Calibration (the full-distribution version, from Chapter 2 §2.5):** for every score value s, P(Y=1 | S=s, A=a) = s, for every group a. This is the property that COMPAS's maker pointed to in their defense — "when we say 70% risk, it really is about 70% for both groups."

**What predictive parity/calibration captures:** "when the model says positive, it means the same thing regardless of group" — this matters enormously whenever a human downstream is going to *interpret* the score or decision as meaningful (a judge reading a risk score, a doctor reading a diagnostic probability).

## 3.5 The Impossibility Result — why you can't have all three

Here is the key theorem of this whole topic (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016), stated informally:

> **If two groups have different base rates, then no model (except a perfect one) can simultaneously satisfy: (1) equal FPR and TPR across groups (equalized odds), and (2) equal PPV / calibration across groups — unless the model is a perfect predictor.**

Let's see *why*, using algebra you already have the pieces for. There's an exact relationship between PPV, TPR, FPR, and the base rate (call the base rate p = P(Y=1|A)):

**PPV = (TPR × p) / (TPR × p + FPR × (1 − p))**

Look at what this formula says: PPV is *entirely determined* by TPR, FPR, and the base rate p. Now suppose you insist TPR and FPR are equal across both groups (equalized odds). If Group A and Group B have *different* base rates p_A ≠ p_B, then plugging different p values into an otherwise-identical formula **must** produce different PPVs. There's no way around it algebraically — the only way to keep PPV equal too would be for the formula to be insensitive to p, which it isn't (except in the degenerate case of a perfect classifier, where TPR=1 and FPR=0 for everyone, and the tradeoff disappears because there are no errors to distribute unevenly in the first place).

**Concretely, plugging in Chapter 2's Group A numbers:**
PPV(A) = (0.75 × 0.40) / (0.75 × 0.40 + 0.20 × 0.60) = 0.30 / (0.30 + 0.12) = 0.30/0.42 ≈ **0.71** ✓ matches Chapter 2

**And if we forced Group B to have the exact same TPR (0.75) and FPR (0.20) as Group A, but kept Group B's base rate at 0.20:**
PPV(B, forced-equal-odds) = (0.75 × 0.20) / (0.75 × 0.20 + 0.20 × 0.80) = 0.15 / (0.15 + 0.16) = 0.15/0.31 ≈ **0.48**

Compare that to Group A's 0.71 — even after we *forced* TPR and FPR to match exactly, PPV still ends up 23 points apart, purely because the base rates differ. That's the impossibility result made concrete: equalized odds and predictive parity are pulling against each other, and the size of the pull is driven directly by the base-rate gap.

**The interview-ready one-liner:** *"You generally can't equalize error rates and calibration at the same time across groups with different base rates — picking a fairness metric is really picking which kind of disparity you're willing to accept, not eliminating disparity altogether."*

## 3.6 Individual Fairness (brief contrast)

Everything above is a **group fairness** definition — it's a statistical property of an entire group, and says nothing about any specific person. **Individual fairness** takes the opposite approach:

**Definition (Dwork et al. 2012):** similar individuals should receive similar predictions. Formally, for a distance metric d(x, x′) measuring how similar two individuals' inputs are, and a distance measure D on outcomes:

D(Ŷ(x), Ŷ(x′)) ≤ L · d(x, x′)

for some Lipschitz constant L — i.e., the model's output can't change faster than the inputs do (this is the same "Lipschitz" idea from smoothness/optimization, just applied to fairness instead of convergence rates).

**Why it's hard to use in practice:** it requires you to define a similarity metric d(x, x′) between people — and *that itself* is where all the fairness judgment silently goes. Choosing what counts as "similar" is doing the same ethical work that choosing a group fairness metric does, just hidden one level deeper. This is why group fairness metrics dominate in practice, even though individual fairness is arguably the philosophically cleaner idea.

## 3.7 Summary table

| Metric | What's equalized | Statement about | Blind to |
|---|---|---|---|
| Demographic Parity | P(Ŷ=1) | Ŷ only | correctness of predictions |
| Equalized Odds | TPR and FPR | Ŷ vs Y | approval rate, calibration |
| Equal Opportunity | TPR only | Ŷ vs Y | FPR, approval rate, calibration |
| Predictive Parity / Calibration | PPV / P(Y=1\|S=s) | Y vs Ŷ or S | approval rate, error-rate gaps |
| Individual Fairness | similar Ŷ for similar x | Ŷ vs x | requires a similarity metric — the hard part is hidden here |

## 3.8 Quick self-check before Chapter 4

- Can you re-derive, without looking, why different base rates break equalized-odds + calibration simultaneously?
- Given a scenario (e.g., "false negatives are much more costly than false positives here"), can you pick equal opportunity over full equalized odds and explain why?
- Can you state in one sentence what individual fairness is quietly assuming?

---

**Next: Chapter 4 — Measuring Fairness in Practice**, where we go from "here's the definition" to "here's how you'd actually compute and report this on a real model" — metric selection by use case, intersectional slicing, and the small-sample-size problem that shows up the moment you slice finely enough to be useful.
