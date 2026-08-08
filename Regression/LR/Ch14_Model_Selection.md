# Chapter 14 — Model Selection

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Compares Chapter 5's full model ($x_1,x_2$; $SSE=2.4$) against the reduced model ($x_1$ only; $SSE=17.1$) — the same two models from the partial F-test in Chapter 5, §5.5 — across four different selection criteria.*

---

## 14.1 The Motivating Question

Chapter 4, §4.7 flagged the core problem this chapter solves: $R^2$ **mechanically never decreases** when you add a predictor, even a useless one, so it can never by itself tell you whether a predictor is worth keeping. This chapter builds four different tools that all penalize model complexity in some way, so that adding a predictor only "wins" if it improves fit by more than the penalty for the added complexity.

**Recall the two competing models:** Full ($x_1,x_2$): $SSE=2.4$, $p=3$ parameters. Reduced ($x_1$ only): $SSE=17.1$, $p=2$ parameters. $n=5$, $SST=673.2$ in both cases.

---

## 14.2 Adjusted $R^2$

$$ R^2_{adj} = 1-(1-R^2)\frac{n-1}{n-p-1} $$

**Plain-English reading:** ordinary $R^2$ never penalizes adding predictors; adjusted $R^2$ explicitly does, via the $(n-p-1)$ term shrinking as $p$ grows — so adjusted $R^2$ can actually **decrease** if an added predictor doesn't earn its keep.

**Worked numbers:**

Full model: $R^2 = SSR/SST = 670.8/673.2 = 0.9964$.
$$ R^2_{adj,full} = 1-(1-0.9964)\frac{4}{2} = 1-0.00357(2) = 0.9929 $$

Reduced model: $R^2 = 656.1/673.2 = 0.9746$.
$$ R^2_{adj,reduced} = 1-(1-0.9746)\frac{4}{3} = 1-0.0254(1.333) = 0.9661 $$

**Verdict: adjusted $R^2$ favors the full model (0.9929 vs. 0.9661)** — even though Chapter 5 found $x_2$'s individual t-test insignificant. This is an important, sometimes-confusing result worth stating plainly: adjusted $R^2$ judges *overall explanatory power relative to parameter cost*, which is a different question from "is this one coefficient individually distinguishable from zero." A predictor can fail an individual significance test yet still improve adjusted $R^2$, especially under multicollinearity (Chapter 9) where individual t-tests are underpowered but the *joint* contribution remains real.

---

## 14.3 AIC (Akaike Information Criterion)

$$ AIC = n\ln\left(\frac{SSE}{n}\right)+2p $$

**Convention warning worth stating in an interview:** some sources add an extra $+2$ for the estimated error variance $\sigma^2$ itself (treating it as a parameter), so absolute AIC values can differ by a constant across software/textbooks — what matters for model *comparison* is the relative difference between candidate models' AIC values, not any single AIC value in isolation. **Lower AIC is better.**

**Worked numbers:**

Full model: $SSE/n = 2.4/5=0.48$, $\ln(0.48)=-0.734$.
$$ AIC_{full} = 5(-0.734)+2(3) = -3.67+6 = 2.33 $$

Reduced model: $SSE/n=17.1/5=3.42$, $\ln(3.42)=1.230$.
$$ AIC_{reduced} = 5(1.230)+2(2) = 6.15+4 = 10.15 $$

**Verdict: AIC strongly favors the full model** (2.33 vs. 10.15 — a difference of about 7.8, well above the conventional "substantially better" threshold of roughly 2).

---

## 14.4 BIC (Bayesian Information Criterion)

$$ BIC = n\ln\left(\frac{SSE}{n}\right)+p\ln(n) $$

**The key difference from AIC:** BIC's penalty term is $p\ln(n)$ instead of $2p$ — for any $n>7$ (since $\ln(n)>2$ once $n>e^2\approx7.39$), BIC penalizes each additional parameter **more harshly** than AIC does. This means **BIC tends to favor smaller models than AIC does, especially as $n$ grows** — a frequently tested conceptual point.

**Worked numbers** ($\ln(5)=1.609$):

$$ BIC_{full} = -3.67+3(1.609) = -3.67+4.83 = 1.16 $$

$$ BIC_{reduced} = 6.15+2(1.609) = 6.15+3.22 = 9.37 $$

**Verdict: BIC also favors the full model** here (1.16 vs. 9.37) — with $n=5$ still below the $n>7$ threshold where BIC's penalty exceeds AIC's, so the two criteria happen to agree in this small example; with larger $n$, BIC's stronger penalty could plausibly flip a borderline decision toward the simpler model even when AIC still favors the more complex one.

---

## 14.5 Mallows' $C_p$

$$ C_p = \frac{SSE_p}{MSE_{full}}+2p-n $$

where $SSE_p$ is the SSE of the candidate model being evaluated, and $MSE_{full}$ is the mean squared error from the **largest** model under consideration (here, the full $x_1,x_2$ model) — used as the best available estimate of the true $\sigma^2$.

**The rule of thumb:** a well-specified model (one with negligible bias) should have $C_p\approx p$. A model with $C_p\gg p$ suggests important predictors are missing (the model is biased, underfitting).

**Worked numbers.** $MSE_{full}=SSE_{full}/(n-p_{full})=2.4/2=1.2$.

**Full model, evaluated against itself:**
$$ C_{p,full} = \frac{2.4}{1.2}+2(3)-5 = 2+6-5 = 3 $$

Notice $C_{p,full}=3$ **exactly equals** $p_{full}=3$ — this is not a coincidence specific to our numbers; **it's a mathematical identity that always holds when a model is evaluated against itself as the reference "full" model** (since $SSE_p/MSE_{full}=n-p$ exactly in that case, making $C_p=(n-p)+2p-n=p$ always).

**Reduced model:**
$$ C_{p,reduced} = \frac{17.1}{1.2}+2(2)-5 = 14.25+4-5 = 13.25 $$

**Verdict:** $C_{p,reduced}=13.25 \gg p_{reduced}=2$ — a dramatic red flag that the reduced model is badly underfitting (missing important explanatory structure, namely $x_2$), while the full model's $C_p=3=p$ signals no detectable lack of fit. **All four criteria (adjusted $R^2$, AIC, BIC, $C_p$) agree here: keep $x_2$ in the model,** despite its individually insignificant t-test from Chapter 5 — a clean illustration of why model-selection criteria and individual-coefficient significance tests are answering genuinely different questions (echoing Chapter 5, §5.6's decision guide).

---

## 14.6 Stepwise Selection Procedures (Briefly)

- **Forward selection**: start with no predictors, add the one that most improves the criterion at each step, stop when no further addition helps.
- **Backward elimination**: start with all candidate predictors, remove the least useful one at each step, stop when no further removal helps.
- **Stepwise (both directions)**: alternates adding and removing at each step, allowing a predictor added earlier to be dropped later if it becomes redundant once other predictors are included.

**Well-known criticisms (worth raising proactively in an interview, since they're commonly asked about):** stepwise procedures perform many implicit hypothesis tests in sequence without correcting for multiple comparisons, so reported p-values and even $R^2$ values from the final selected model are **overoptimistic** (this is a form of data dredging). They can also be unstable — small changes in the data can lead to a different sequence of inclusions/exclusions and a different final model entirely. Modern practice increasingly prefers regularization (Chapters 16–18) or cross-validation-based selection (Chapter 15) over classical stepwise procedures for exactly these reasons.

---

## 14.7 Where the Textbooks Differ

- **Kutner** gives the most complete derivation of Mallows' $C_p$ and its bias-detection interpretation, tying it closely to the SSE-decomposition framework used throughout the book.
- **Montgomery** covers stepwise procedures in the greatest practical depth, including detailed guidance on entry/exit significance thresholds, reflecting its more traditional applied-statistics orientation.
- **Sheather** emphasizes AIC/BIC as computed and compared directly via software (`AIC()`, `BIC()` in R), treating them as the practical default over $C_p$ or manual stepwise procedures.
- **ESL/ISL** are notably skeptical of all four classical criteria as the *primary* selection tool, preferring cross-validation (Chapter 15) as the gold standard for prediction-focused model selection — AIC/BIC/$C_p$ are presented mainly as fast, computation-free approximations to what cross-validation would tell you, useful when refitting many times is expensive.

---

## 14.8 Interview Q&A

**Q: Why can't you use plain $R^2$ to decide whether to add a predictor?**
A: $R^2$ mechanically never decreases when a predictor is added, even a completely useless one — it provides no penalty for the added complexity, so it can't distinguish a genuinely useful predictor from pure noise.

**Q: How does BIC's penalty differ from AIC's, and what's the practical consequence?**
A: BIC penalizes each parameter by $\ln(n)$ instead of AIC's flat $2$. Once $n>7$ (roughly), $\ln(n)>2$, so BIC penalizes complexity more heavily than AIC — meaning BIC tends to select smaller/simpler models than AIC, especially in larger datasets.

**Q: What does it mean if a candidate model's Mallows' $C_p$ is much larger than its number of parameters $p$?**
A: It signals the model is likely missing important predictors (underfitting/biased) — a well-specified model should have $C_p$ close to $p$.

**Q: Why might a predictor be worth keeping in a model-selection sense (AIC/BIC/adjusted $R^2$/$C_p$) even if its individual t-test isn't significant?**
A: These criteria evaluate the model's overall fit-versus-complexity tradeoff, which is a different question from an individual coefficient's statistical significance — especially under multicollinearity, where individual t-tests are underpowered even though the predictor contributes real joint explanatory value.

**Q: What's a major criticism of stepwise selection procedures?**
A: They perform many sequential hypothesis tests without correcting for multiple comparisons, producing overoptimistic fit statistics and unstable model choices that can vary substantially with small changes in the data — modern practice generally prefers regularization or cross-validation instead.

---

*End of Chapter 14. Next: Chapter 15 — Variable Selection & Overfitting (the train/test split, why in-sample fit statistics like $R^2$ can be misleading, and a first introduction to cross-validation as the more robust alternative to Chapter 14's classical criteria).*
