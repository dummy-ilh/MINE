# Chapter 1: The Potential Outcomes Framework

## 1. Explanation

Start with the most basic question in causal inference: **what does it even mean to say "X causes Y"?**

Imagine you take an aspirin (D=1) for a headache, and one hour later your headache (Y) is gone. Did the aspirin *cause* the headache to go away? You genuinely don't know — maybe it would have gone away anyway. To make "caused" precise, we need to compare two *versions of the world*:
- The world where you took the aspirin → your headache outcome is $Y(1)$
- The world where you didn't → your headache outcome is $Y(0)$

The causal effect of aspirin **for you, specifically** is:
```
τ_you = Y_you(1) − Y_you(0)
```

This is called the **potential outcomes framework** (Neyman-Rubin causal model). Every unit (person, user, region) has *two* potential outcomes lurking — one for each treatment state — but reality only lets one of them happen. This single fact — that we can only ever witness one branch of a fork — is the entire reason causal inference is hard and is its own field, separate from ordinary statistics/ML.

**Formalizing:** Let $D_i \in \{0,1\}$ be the treatment unit $i$ receives, and let $Y_i(0), Y_i(1)$ be their two potential outcomes. The **observed outcome** follows the "switching equation":
```
Y_i = D_i · Y_i(1) + (1 − D_i) · Y_i(0)
```
This is just saying: "you see Y(1) if treated, Y(0) if not" — but written as a formula so we can manipulate it algebraically.

**The Fundamental Problem of Causal Inference** (Holland, 1986): for any specific unit $i$, you can never observe both $Y_i(1)$ and $Y_i(0)$. You know $D_i$ (which branch happened) and $Y_i$ (the outcome on that branch), but the *other* branch — the counterfactual — is permanently missing data, not just "hard to measure." This reframes causal inference as a **missing data problem**: almost every method you'll learn is a clever way of estimating the missing potential outcome (or an average of it across many units), using assumptions to justify the substitution.

**Why not just solve it at the individual level?** You can't — no matter how much data you collect on person X, you cannot re-run history for that exact same person at the exact same moment with the opposite treatment. This is why almost all of causal inference gives you **average** effects across groups of similar units, not individual-level truths (with some exceptions in heterogeneous treatment effect modeling, which still relies on the same underlying assumptions).

### A second layer: SUTVA, quietly baked into the notation above

Notice the switching equation above assumes $Y_i$ depends only on $D_i$ — unit $i$'s own treatment — not on anyone else's treatment. This is the **Stable Unit Treatment Value Assumption (SUTVA)**, and it's baked in from the very first formula you write down in this field. It has two parts:
1. **No interference**: my outcome doesn't depend on your treatment assignment.
2. **No hidden variation in treatment**: "treatment" means the same thing for every unit (e.g., "1 unit of aspirin" isn't secretly a different dose for different people).

This assumption will come back with a vengeance in Chapter 11 (Interference) — for now, just notice that it's already silently present the moment you write $Y_i(D_i)$ instead of $Y_i(D_1, D_2, ..., D_n)$.

## 2. Example

**Setup:** 8 patients get a new pain medication (D) or a placebo. We *hypothetically* know both potential outcomes (pain reduction score, 0-10, higher=better) — something researchers never actually observe, but useful for teaching:

| Patient | Y(0) placebo-outcome | Y(1) drug-outcome | Individual effect τ_i |
|---|---|---|---|
| 1 | 2 | 6 | 4 |
| 2 | 3 | 5 | 2 |
| 3 | 1 | 7 | 6 |
| 4 | 4 | 4 | 0 |
| 5 | 2 | 8 | 6 |
| 6 | 3 | 3 | 0 |
| 7 | 5 | 9 | 4 |
| 8 | 1 | 5 | 4 |

True ATE = mean(τ_i) = (4+2+6+0+6+0+4+4)/8 = 26/8 = **3.25**

Now, in reality, each patient gets *either* drug or placebo — say patients 1,3,5,7 got the drug and 2,4,6,8 got placebo (assume this was random). What you'd actually observe:

| Patient | D | Y observed |
|---|---|---|
| 1 | 1 | 6 |
| 2 | 0 | 3 |
| 3 | 1 | 7 |
| 4 | 0 | 4 |
| 5 | 1 | 8 |
| 6 | 0 | 3 |
| 7 | 1 | 9 |
| 8 | 0 | 1 |

Estimated ATE = mean(Y|D=1) − mean(Y|D=0) = (6+7+8+9)/4 − (3+4+3+1)/4 = 7.5 − 2.75 = **4.75**

This differs from the true 3.25 purely due to **sampling variability** in which 4 of the 8 happened to be randomized to treatment (with only 8 units, luck plays a big role — this small-sample noise is exactly why real experiments need hundreds/thousands of units, not 8, to get a stable estimate close to the true ATE).

**A second, smaller example to build the missing-data intuition directly:** Consider just Patient 1. We *know* (hypothetically) that Y(0)=2 and Y(1)=6, so τ_1=4. But in real life, once Patient 1 takes the drug (D=1), we only ever observe Y_1=6. The value "2" — what would have happened without the drug — vanishes into the counterfactual and is never recorded anywhere, no matter how carefully we measure Patient 1 afterward. No amount of follow-up bloodwork, surveys, or monitoring recovers it. This is the concrete, single-patient version of "the fundamental problem."

## 3. Interview Q&A

**Q: In your own words, why is the "fundamental problem of causal inference" fundamental — i.e., why can't better data collection fix it?**
A: It's a logical/definitional limitation, not a measurement limitation. To see both $Y_i(1)$ and $Y_i(0)$ for the same unit at the same moment in history, you'd need to rewind time and change only the treatment — physically impossible. No amount of additional data collected *going forward* changes this; you can only ever collect the realized branch. That's why we shift the goal from "know τ_i exactly" to "estimate averages of τ_i using assumptions that make groups comparable."

**Q: What is the "switching equation" and why is it useful to write down explicitly?**
A: $Y_i = D_iY_i(1) + (1-D_i)Y_i(0)$. It's useful because it lets you *algebraically* decompose the observed data into the causal effect plus a bias term, rather than just intuiting it — it's the bridge between the "two hidden worlds" framework and the single observed dataset you actually have.

**Q: Someone says "we have perfect data — millions of user logs — so we don't need to worry about missing counterfactuals." Respond.**
A: Volume of data doesn't solve the *structural* missingness — for any individual user, you still only see one treatment branch. Millions of rows help you estimate *average* effects precisely (assuming the right identifying assumptions hold), but they don't let you observe any single user's counterfactual. In fact, large N without a valid comparison group (e.g., all confounded observational data) just gives you a very *precise* estimate of the *wrong* (biased) quantity — precision isn't the same as correctness.

**Q: If τ_i varies a lot across units (heterogeneous treatment effects), is knowing the ATE still useful?**
A: Yes for population-level decisions (e.g., "should we ship this to everyone"), but it can hide important structure — some subgroups might have large positive effects, others negative, averaging to a small/no net ATE. For personalization or targeting decisions you'd want CATE (Chapter 2) or a full distribution of effects, not just the mean.

**Q: What is SUTVA, and where does it silently show up already in the very first formula of this framework?**
A: SUTVA (Stable Unit Treatment Value Assumption) requires that a unit's outcome depends only on its own treatment, not on others', and that "treatment" is a well-defined, consistent thing across units. It shows up the moment you write $Y_i = D_iY_i(1) + (1-D_i)Y_i(0)$ — this notation only has *one* treatment index ($D_i$) feeding into unit $i$'s outcome; if interference existed, you'd technically need $Y_i(D_1,...,D_n)$, a potential outcome depending on everyone's treatment, which is far harder to work with (and is exactly the subject of Chapter 11).

**Q: A friend says "if we just had a perfect predictive model of the counterfactual, we could compute individual treatment effects directly and skip all this framework." What's the flaw?**
A: Any "predictive model of the counterfactual" still has to be trained/validated on data — and by the fundamental problem, we never have ground-truth counterfactual labels to check it against for any individual. Such a model's accuracy on individual-level counterfactuals is fundamentally unverifiable; you can only validate it indirectly at the aggregate level (e.g., does it recover known RCT-based ATEs), which brings you right back to the same identification assumptions this framework already requires.

---
**Next: Chapter 2 — Causal Estimands (ATE, ATT, ATC, CATE, LATE)**
