# Chapter 15 — Variable Selection & Overfitting

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations. Uses Chapter 5's full dataset ($x_1,x_2,y$; $n=5$) to work a complete leave-one-out cross-validation by hand — small enough that every one of the 5 folds can be solved explicitly.*

---

## 15.1 The Motivating Question

Every fit statistic used so far — $R^2$, $SSE$, even the AIC/BIC/$C_p$ criteria from Chapter 14 — is computed **on the same data used to fit the model.** This is called **in-sample** performance, and it has a fundamental, unavoidable optimism problem: a model has, in some sense, already "seen the answers" for every point it's being judged on. **Out-of-sample** performance — how well the model predicts data it never saw during fitting — is what actually matters for real-world use, and it can look meaningfully worse than in-sample statistics suggest. This gap between in-sample and out-of-sample performance **is overfitting**, and this chapter builds the tools to measure it honestly.

**Plain-language framing before anything else:** imagine grading a student using the exact same questions they used to study from. Of course they'll do well — they've already seen the answers. That's exactly the problem with judging a model using the same data it was fit on: $R^2$ and SSE are like grading a student on their own study sheet. What you actually want to know is how the model performs on *new* questions it's never seen — and that's a fundamentally different, usually less flattering, number. This chapter is about honestly measuring that gap.

---

## 15.2 The Train/Test Split — The Simplest Fix, and Its Limitation

Split the data into a **training set** (used to fit the model) and a **test set** (held out, used only to evaluate). The test-set error is an honest, unbiased estimate of out-of-sample performance, precisely because the model never saw those points during fitting.

**The limitation, especially acute with small datasets like our 5-observation running example:** a single train/test split wastes data (the test portion contributes nothing to fitting), and with few points, the *particular* random split chosen can swing the estimated test error substantially — you might get lucky or unlucky with which points land in the test set. This motivates **cross-validation**, which reuses the data far more efficiently.

**In plain words:** the simplest fix is to lock away part of the data as a genuine "final exam" the model never studies from. That's honest, but it has a real cost: whatever data you locked away is data the model *didn't* get to learn from, which hurts especially badly when you only have 5 data points to begin with. And with so few points, whichever ones happen to land in the "exam" pile versus the "study" pile is somewhat down to luck — a different random split could easily paint a different, possibly misleading picture. Cross-validation is the fix for both problems at once.

---

## 15.3 K-Fold Cross-Validation and Its Extreme Case: LOOCV

**K-fold CV:** split the data into $k$ roughly equal folds; for each fold in turn, train on the other $k-1$ folds and test on the held-out one; average the $k$ resulting test errors.

**Leave-One-Out Cross-Validation (LOOCV)** is the extreme special case where $k=n$ — each fold holds out exactly **one** observation, trains on the remaining $n-1$, and predicts the held-out point. With only $n=5$ observations in our running dataset, LOOCV is both the natural choice and small enough to work through by hand in full.

**Plain-language framing:** instead of one single train/test split, why not run the "final exam" test multiple times, each time swapping out who gets tested and who gets studied from — and then average the results? That's k-fold cross-validation: everyone gets a turn being the "held-out exam" once, and everyone else gets a turn as "study material" for the others. LOOCV is just the most thorough version of this idea: instead of grouping data into a few chunks, you hold out data points **one at a time** — refit the model each time on everyone else, and see how well it predicts the one person you left out. With only 5 students, this means doing the whole "refit and predict" dance 5 separate times, once per student.

---

## 15.4 Worked Example — Full LOOCV on the 5-Student Dataset

Recall the full model from Chapter 5: $x_1$ (hours), $x_2$ (practice tests), $y$ (score) $=50,55,65,70,83$. For each student, refit the model on the **other four** and predict the held-out one.

**Fold 1 (hold out student 1):** refitting on students 2–5 gives $\hat{\beta}=(37.75,\ 5,\ 6.5)$. Predicted for student 1 ($x_1=1,x_2=1$): $37.75+5+6.5=49.25$. Actual $=50$. Squared error $=0.75^2=0.5625$.

**Fold 2 (hold out student 2):** refitting on students 1,3,4,5 gives $\hat{\beta}=(37,\ 4,\ 8.5)$. Predicted for student 2 ($x_1=2,x_2=1$): $37+8+8.5=53.5$. Actual $=55$. Squared error $=1.5^2=2.25$.

**Fold 3 (hold out student 3):** refitting on students 1,2,4,5 gives $\hat{\beta}=(38.2,\ 4.1,\ 8.0)$. Predicted for student 3 ($x_1=3,x_2=2$): $38.2+12.3+16=66.5$. Actual $=65$. Squared error $=(-1.5)^2=2.25$.

**Fold 4 (hold out student 4):** refitting on students 1,2,3,5 gives $\hat{\beta}=(38.5,\ 5.5,\ 5.5)$. Predicted for student 4 ($x_1=4,x_2=2$): $38.5+22+11=71.5$. Actual $=70$. Squared error $=(-1.5)^2=2.25$.

**Fold 5 (hold out student 5):** refitting on students 1–4 gives the **exact-fit** relationship from Chapter 4, $\hat{\beta}=(40,5,5)$ (with $SSE=0$ on those 4 points, as established in Chapter 8). Predicted for student 5 ($x_1=5,x_2=3$): $40+25+15=80$. Actual $=83$. Squared error $=3^2=9$.

**LOOCV estimate of test MSE:**

$$ CV_{(5)} = \frac{0.5625+2.25+2.25+2.25+9}{5} = \frac{16.3125}{5} \approx 3.26 $$

**Compare to in-sample MSE** (Chapter 5): $SSE_{full}/n = 2.4/5 = 0.48$. **The out-of-sample (LOOCV) error estimate, 3.26, is nearly 7 times larger than the optimistic in-sample estimate, 0.48** — a concrete, hand-verified illustration of exactly the gap §15.1 warned about.

**Walking through what actually happened, in plain words:** each time, we pretended one student didn't exist, let the model learn from the remaining four, and then asked it to guess the missing student's score. Four of the five guesses were reasonably close (off by 0.75 to 1.5 points). But one guess — student 5 — was off by a full 3 points, contributing the single largest chunk of error by far. Averaging all five of these "honest, never-seen-this-student" guesses gives a test error of about 3.26 — nearly 7 times worse than what the in-sample statistic (0.48) claimed. That 0.48 number was flattering the model by letting it grade itself on data it had already memorized; 3.26 is a much more honest picture of how well it would do on a genuinely new student.

---

## 15.5 A Direct Callback to Chapter 8 — Influence and Predictability Are the Same Story

**Look at which fold produced by far the largest error: student 5, with squared error 9 — more than three times any other fold's error.** This is the *same* student flagged with the largest Cook's distance ($D_5=1.833$) back in Chapter 8. This is not a coincidence: **a highly influential point (Chapter 8) is, by the same underlying logic, a hard point to predict when it's excluded from fitting** — the rest of the data has comparatively little information about what that point's value "should" be, precisely because that point wasn't similar enough to the others to be well-predicted by them. Influence diagnostics and cross-validation error are two different lenses onto the same underlying fact about a dataset's structure.

**Plain-language version of this connection:** student 5 was flagged back in Chapter 8 as the point that most strongly *pulled* the fitted line toward itself when it was included. It makes intuitive sense, then, that the same student is the hardest one to *predict* when excluded — both symptoms come from the same root cause: student 5 sits somewhere unusual relative to the rest of the group, so the other four students simply don't carry much information about what student 5's score "should" look like. When student 5 is in the room, it drags the fit toward itself; when student 5 is sent out of the room, nobody left behind can guess where it went. Same underlying oddity, showing up as two different symptoms.

---

## 15.6 The Bias-Variance Tradeoff in Choosing $k$

- **LOOCV** ($k=n$): uses almost all the data for training each time (low bias in the error estimate — each training set is very close in size to the full dataset), but the $n$ resulting test errors are highly correlated with each other (each training set overlaps with every other training set in all but one point), giving a **higher-variance** estimate of true test error.
- **Small $k$** (e.g., $k=5$ or $k=10$, standard choices with larger datasets): each training set is smaller (slightly more biased error estimate, since the model is trained on less data than the full sample), but the $k$ resulting test errors are less correlated with each other, giving a **lower-variance** overall estimate.

**Plain-language version of this tradeoff:** LOOCV's training sets are almost identical to each other every single time (they only ever differ by one swapped-out point), so its 5 error measurements aren't really 5 *independent* pieces of evidence — they're 5 very similar retellings of nearly the same story, which makes the overall average a bit shaky (high variance) even though each individual training run was about as accurate as you could hope for (low bias). Smaller-$k$ CV (like 5-fold or 10-fold on bigger datasets) uses noticeably different, less-overlapping training sets each time — giving you genuinely more varied evidence (lower variance in the final average), at the small cost of each individual training run seeing a bit less data than the full sample (slightly more bias).

**Practical guidance (standard in ML practice, reflected especially in ESL/ISL):** $k=5$ or $k=10$ is the common default for datasets of moderate-to-large size, balancing this tradeoff; LOOCV remains attractive mainly for very small datasets (like ours) where every observation is precious, or for models where the leave-one-out fit has a fast closed-form shortcut (linear regression, in fact, has exactly such a shortcut — see the "PRESS statistic" note below — avoiding the need to literally refit $n$ times).

**A computational note worth having ready:** for ordinary linear regression specifically, LOOCV doesn't actually require refitting the model $n$ separate times — there's a closed-form shortcut using the leverage values from Chapter 3/7:

$$ CV_{(n)} = \frac{1}{n}\sum_{i=1}^n\left(\frac{e_i}{1-h_{ii}}\right)^2 $$

This is called the **PRESS statistic** (Predicted Residual Sum of Squares) when left unaveraged, and it gives the exact same answer as literally refitting $n$ times (as done by hand above) — a valuable efficiency fact for an interview, even though we did the full manual version above specifically to make the mechanism transparent.

**In plain words, why this shortcut is such a nice fact:** normally, "leave one out and refit" sounds like it should require literally redoing the entire regression $n$ separate times — painfully slow for big datasets. But for plain linear regression specifically, it turns out you can get the *exact same* answer using only the leverage values ($h_{ii}$) you already computed back in Chapter 7 — no refitting required at all. This is a case where a clever piece of algebra saves you an enormous amount of actual computation, which is exactly the kind of fact that's worth having ready to mention in an interview.

---

## 15.7 Where the Textbooks Differ

- **Kutner** covers this material lightly, mostly in the context of the PRESS statistic as an extension of the diagnostic tools from Chapters 7–8, rather than as a full cross-validation framework.
- **Montgomery** similarly treats PRESS as the primary tool, consistent with the book's overall diagnostics-heavy orientation.
- **Sheather** bridges toward the ML perspective more explicitly, introducing train/test splits and k-fold CV as standard practice alongside the classical diagnostics.
- **ESL/ISL** treat cross-validation as the central, default tool for model evaluation and selection throughout the entire book — Chapter 14's AIC/BIC/$C_p$ are, from ESL/ISL's perspective, mainly fast approximations to what cross-validation would tell you directly; this chapter's material is arguably ESL/ISL's true starting point, with everything in Chapters 1–14 serving as necessary background.

---

## 15.8 Interview Q&A

**Q: Why is in-sample $R^2$ or SSE an unreliable guide to real-world model performance?**
A: The model was fit using that same data, so its apparent fit is optimistic — it's being evaluated on data it already "learned from." Out-of-sample (held-out) performance is what actually reflects real-world predictive ability, and it's typically worse than in-sample statistics suggest.
*(Simple version: grading a student on their own study sheet always looks good — you need a fresh test to know what they've actually learned.)*

**Q: What's the difference between k-fold CV and LOOCV?**
A: LOOCV is the special case of k-fold CV where $k=n$ — each fold holds out exactly one observation. Smaller $k$ (e.g., 5 or 10) trades a bit more bias (smaller training sets) for lower variance in the overall test-error estimate, since the folds' errors are less correlated with each other than LOOCV's are.
*(Simple version: LOOCV tests one point at a time using almost-identical training sets each round; smaller-$k$ CV uses more genuinely different training sets, giving a steadier overall estimate.)*

**Q: Does linear regression require literally refitting the model $n$ times to compute LOOCV?**
A: No — there's a closed-form shortcut using leverage values, $CV_{(n)}=\frac{1}{n}\sum(e_i/(1-h_{ii}))^2$ (the PRESS statistic, averaged), giving the exact same result without ever refitting.
*(Simple version: there's a mathematical shortcut that gives the exact same answer as refitting $n$ times, using numbers you likely already have on hand.)*

**Q: If a point has a high Cook's distance, would you also expect it to have a high leave-one-out prediction error?**
A: Generally yes — both are measuring related aspects of the same underlying fact: a point that strongly influences the fitted model when included is, by the same logic, hard to predict accurately when excluded, since the rest of the data carries comparatively little information about it.
*(Simple version: a point that "pulls hard" on the line when present is usually also a point nobody can guess correctly when it's absent — same underlying oddity, two different symptoms.)*

**Q: Why isn't a single train/test split always sufficient, especially with a small dataset?**
A: It wastes data (the test portion doesn't contribute to fitting) and the specific random split can substantially swing the estimated test error with few observations — cross-validation reuses the data more efficiently and averages over multiple splits to reduce that variance.
*(Simple version: with a single split, you're both wasting data and gambling on which points happen to land in which pile — cross-validation removes both problems by giving every point a turn on both sides.)*

---

*End of Chapter 15. Next: Chapter 16 — Ridge Regression (the L2 penalty, its closed-form solution, and how it directly addresses the multicollinearity instability from Chapter 9 by deliberately introducing bias in exchange for reduced variance — the tradeoff previewed all the way back in Chapter 6).*
