# Chapter 14 — Model Selection

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations. Compares Chapter 5's full model ($x_1,x_2$; $SSE=2.4$) against the reduced model ($x_1$ only; $SSE=17.1$) — the same two models from the partial F-test in Chapter 5, §5.5 — across four different selection criteria.*

---

## 14.1 The Motivating Question

Chapter 4, §4.7 flagged the core problem this chapter solves: $R^2$ **mechanically never decreases** when you add a predictor, even a useless one, so it can never by itself tell you whether a predictor is worth keeping. This chapter builds four different tools that all penalize model complexity in some way, so that adding a predictor only "wins" if it improves fit by more than the penalty for the added complexity.

**Recall the two competing models:** Full ($x_1,x_2$): $SSE=2.4$, $p=3$ parameters. Reduced ($x_1$ only): $SSE=17.1$, $p=2$ parameters. $n=5$, $SST=673.2$ in both cases.

**Plain-language framing before anything else:** $R^2$ has a sneaky flaw — you can throw in a totally random, meaningless column of numbers as an extra "predictor," and $R^2$ will never go down, only stay flat or creep up. That's because OLS is mathematically guaranteed to use *any* extra flexibility it's given to shave off at least a tiny bit of error, even if that extra flexibility is pure noise. So $R^2$ alone can never tell you "was this new predictor actually worth adding, or did I just give the model more room to memorize coincidences?" This chapter is about four different referees, each with its own way of asking that same question: "does the improvement in fit actually earn back the cost of the extra complexity?"

---

## 14.2 Adjusted $R^2$

$$ R^2_{adj} = 1-(1-R^2)\frac{n-1}{n-p-1} $$

**Plain-English reading:** ordinary $R^2$ never penalizes adding predictors; adjusted $R^2$ explicitly does, via the $(n-p-1)$ term shrinking as $p$ grows — so adjusted $R^2$ can actually **decrease** if an added predictor doesn't earn its keep.

**How the penalty mechanically works, in plain words:** as you add more parameters $p$, the denominator $(n-p-1)$ gets smaller, which makes the whole fraction $\frac{n-1}{n-p-1}$ get *bigger*. That bigger fraction then gets multiplied against $(1-R^2)$ — effectively "amplifying" whatever leftover unexplained variation still exists. If a new predictor only shrinks $(1-R^2)$ by a tiny sliver, that tiny improvement can get swallowed up by the amplification from the growing penalty — and adjusted $R^2$ actually drops. If the new predictor shrinks $(1-R^2)$ by a lot, it overcomes the penalty, and adjusted $R^2$ rises.

**Worked numbers:**

Full model: $R^2 = SSR/SST = 670.8/673.2 = 0.9964$.
$$ R^2_{adj,full} = 1-(1-0.9964)\frac{4}{2} = 1-0.00357(2) = 0.9929 $$

Reduced model: $R^2 = 656.1/673.2 = 0.9746$.
$$ R^2_{adj,reduced} = 1-(1-0.9746)\frac{4}{3} = 1-0.0254(1.333) = 0.9661 $$

**Verdict: adjusted $R^2$ favors the full model (0.9929 vs. 0.9661)** — even though Chapter 5 found $x_2$'s individual t-test insignificant. This is an important, sometimes-confusing result worth stating plainly: adjusted $R^2$ judges *overall explanatory power relative to parameter cost*, which is a different question from "is this one coefficient individually distinguishable from zero." A predictor can fail an individual significance test yet still improve adjusted $R^2$, especially under multicollinearity (Chapter 9) where individual t-tests are underpowered but the *joint* contribution remains real.

**Why this result feels surprising, and why it isn't a contradiction:** it can feel odd that a predictor "isn't significant on its own" (Chapter 5) but "is worth keeping" (this chapter) — but these two tools are simply answering different questions. The t-test asks: "can I confidently say *this one coefficient alone* is different from zero?" Adjusted $R^2$ asks: "does the model as a whole explain enough extra variation to justify its extra complexity?" Under multicollinearity (Chapter 9), a predictor's *individual* signal can get muddied and hard to isolate, even while its *combined* contribution alongside the other predictor remains real and valuable. Different questions, different tools, no contradiction.

---

## 14.3 AIC (Akaike Information Criterion)

$$ AIC = n\ln\left(\frac{SSE}{n}\right)+2p $$

**Convention warning worth stating in an interview:** some sources add an extra $+2$ for the estimated error variance $\sigma^2$ itself (treating it as a parameter), so absolute AIC values can differ by a constant across software/textbooks — what matters for model *comparison* is the relative difference between candidate models' AIC values, not any single AIC value in isolation. **Lower AIC is better.**

**Plain-English reading before the numbers:** AIC has two ingredients fighting each other. The first term, $n\ln(SSE/n)$, gets *smaller* (more negative) the better your model fits — it rewards accuracy. The second term, $2p$, gets *bigger* the more parameters you use — it punishes complexity. AIC is essentially asking: "does the accuracy gain from adding this parameter outweigh the flat penalty of $2$ points I'm charging you for using it?" Lower total AIC = better trade-off.

**Worked numbers:**

Full model: $SSE/n = 2.4/5=0.48$, $\ln(0.48)=-0.734$.
$$ AIC_{full} = 5(-0.734)+2(3) = -3.67+6 = 2.33 $$

Reduced model: $SSE/n=17.1/5=3.42$, $\ln(3.42)=1.230$.
$$ AIC_{reduced} = 5(1.230)+2(2) = 6.15+4 = 10.15 $$

**Verdict: AIC strongly favors the full model** (2.33 vs. 10.15 — a difference of about 7.8, well above the conventional "substantially better" threshold of roughly 2).

**In plain words:** the full model has one more parameter, which costs it 2 extra "penalty points" compared to the reduced model. But its fit is *so* much better (SSE of 2.4 vs. 17.1) that the accuracy term swings massively in its favor — dropping AIC from 6.15 down to -3.67, a shift of nearly 10.8 points, which utterly dwarfs the mere 2-point complexity penalty being paid. The full model wins comfortably, not by a hair.

---

## 14.4 BIC (Bayesian Information Criterion)

$$ BIC = n\ln\left(\frac{SSE}{n}\right)+p\ln(n) $$

**The key difference from AIC:** BIC's penalty term is $p\ln(n)$ instead of $2p$ — for any $n>7$ (since $\ln(n)>2$ once $n>e^2\approx7.39$), BIC penalizes each additional parameter **more harshly** than AIC does. This means **BIC tends to favor smaller models than AIC does, especially as $n$ grows** — a frequently tested conceptual point.

**Plain-English version of the AIC-vs-BIC distinction:** AIC always charges the exact same flat toll (2 points) for every extra parameter, no matter how much data you have. BIC's toll grows with your sample size — the more data you have, the more BIC assumes you *should* be able to tell a genuinely useful predictor apart from a coincidental one, so it demands a stronger justification (a bigger fit improvement) before allowing extra complexity in. That's why BIC is often described as the "stricter, more skeptical" cousin of AIC, especially in large datasets.

**Worked numbers** ($\ln(5)=1.609$):

$$ BIC_{full} = -3.67+3(1.609) = -3.67+4.83 = 1.16 $$

$$ BIC_{reduced} = 6.15+2(1.609) = 6.15+3.22 = 9.37 $$

**Verdict: BIC also favors the full model** here (1.16 vs. 9.37) — with $n=5$ still below the $n>7$ threshold where BIC's penalty exceeds AIC's, so the two criteria happen to agree in this small example; with larger $n$, BIC's stronger penalty could plausibly flip a borderline decision toward the simpler model even when AIC still favors the more complex one.

**Why $n=5$ is a special case worth noting:** with such a tiny sample, $\ln(5)\approx1.609$ is actually *smaller* than AIC's flat penalty of 2 — meaning BIC is, unusually, being slightly *gentler* than AIC here, not stricter. This flips once you cross roughly $n=8$ data points, after which BIC's penalty overtakes AIC's and starts being the stricter of the two, as described above. It's a useful reminder that BIC's "usually stricter" behavior is an asymptotic (large-sample) tendency, not a rule that applies at every single sample size.

---

## 14.5 Mallows' $C_p$

$$ C_p = \frac{SSE_p}{MSE_{full}}+2p-n $$

where $SSE_p$ is the SSE of the candidate model being evaluated, and $MSE_{full}$ is the mean squared error from the **largest** model under consideration (here, the full $x_1,x_2$ model) — used as the best available estimate of the true $\sigma^2$.

**The rule of thumb:** a well-specified model (one with negligible bias) should have $C_p\approx p$. A model with $C_p\gg p$ suggests important predictors are missing (the model is biased, underfitting).

**Plain-English framing before the numbers:** $C_p$ works a little differently from AIC/BIC — instead of just balancing fit against complexity in the abstract, it directly asks "if this candidate model were actually correct and complete, how big *should* its error be, given how noisy the data genuinely is (estimated from the biggest model available)? And how does that compare to how big its error *actually* is?" If a model's actual error matches what you'd expect from a well-specified model, $C_p$ lands right around $p$. If the actual error is way bigger than expected, that's a sign the model is missing something important — it's *systematically* wrong (biased), not just noisy.

**Worked numbers.** $MSE_{full}=SSE_{full}/(n-p_{full})=2.4/2=1.2$.

**Full model, evaluated against itself:**
$$ C_{p,full} = \frac{2.4}{1.2}+2(3)-5 = 2+6-5 = 3 $$

Notice $C_{p,full}=3$ **exactly equals** $p_{full}=3$ — this is not a coincidence specific to our numbers; **it's a mathematical identity that always holds when a model is evaluated against itself as the reference "full" model** (since $SSE_p/MSE_{full}=n-p$ exactly in that case, making $C_p=(n-p)+2p-n=p$ always).

**Why the full model automatically scores $C_p=p$, in plain words:** the full model is, by definition, the yardstick everything else is measured against — so of course it "passes its own test" perfectly. This isn't a discovery about the full model being good; it's just baked into the arithmetic. The genuinely useful part of $C_p$ is what happens when you check a *smaller* model against that same yardstick, which is exactly what's done next.

**Reduced model:**
$$ C_{p,reduced} = \frac{17.1}{1.2}+2(2)-5 = 14.25+4-5 = 13.25 $$

**Verdict:** $C_{p,reduced}=13.25 \gg p_{reduced}=2$ — a dramatic red flag that the reduced model is badly underfitting (missing important explanatory structure, namely $x_2$), while the full model's $C_p=3=p$ signals no detectable lack of fit. **All four criteria (adjusted $R^2$, AIC, BIC, $C_p$) agree here: keep $x_2$ in the model,** despite its individually insignificant t-test from Chapter 5 — a clean illustration of why model-selection criteria and individual-coefficient significance tests are answering genuinely different questions (echoing Chapter 5, §5.6's decision guide).

**In plain words, the size of the red flag:** a well-behaved 2-parameter model "should" score around $C_p\approx2$. Instead, the reduced model scores 13.25 — more than six times higher than expected. That enormous gap is $C_p$ loudly signaling "this model is missing something real," and in this case we already know exactly what's missing: $x_2$, the predictor the reduced model left out.

---

## 14.6 Stepwise Selection Procedures (Briefly)

- **Forward selection**: start with no predictors, add the one that most improves the criterion at each step, stop when no further addition helps.
- **Backward elimination**: start with all candidate predictors, remove the least useful one at each step, stop when no further removal helps.
- **Stepwise (both directions)**: alternates adding and removing at each step, allowing a predictor added earlier to be dropped later if it becomes redundant once other predictors are included.

**Plain-English one-liner for each:** forward selection builds the model up from nothing, one useful predictor at a time. Backward elimination starts with everything thrown in and trims away whatever isn't pulling its weight. The combined version does both — it's willing to "change its mind" and remove something it added earlier, if a later addition makes that earlier one redundant.

**Well-known criticisms (worth raising proactively in an interview, since they're commonly asked about):** stepwise procedures perform many implicit hypothesis tests in sequence without correcting for multiple comparisons, so reported p-values and even $R^2$ values from the final selected model are **overoptimistic** (this is a form of data dredging). They can also be unstable — small changes in the data can lead to a different sequence of inclusions/exclusions and a different final model entirely. Modern practice increasingly prefers regularization (Chapters 16–18) or cross-validation-based selection (Chapter 15) over classical stepwise procedures for exactly these reasons.

**In plain words, why this is a real problem and not just a technicality:** imagine flipping a coin 20 times and only reporting the one run where you got 15 heads in a row — that "impressive" result is much less impressive once you remember you tried 20 times and cherry-picked the best one. Stepwise selection does something structurally similar: it silently runs many "is this predictor worth adding" tests in a row, and the final model you're left with is effectively the "best-looking result" out of many attempts. Any p-values or fit statistics reported on that final model haven't accounted for all those hidden attempts — so they look more impressive than they honestly should.

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
*(Simple version: $R^2$ always rewards more flexibility, even flexibility spent on nonsense — it has no built-in sense of "was this actually worth it.")*

**Q: How does BIC's penalty differ from AIC's, and what's the practical consequence?**
A: BIC penalizes each parameter by $\ln(n)$ instead of AIC's flat $2$. Once $n>7$ (roughly), $\ln(n)>2$, so BIC penalizes complexity more heavily than AIC — meaning BIC tends to select smaller/simpler models than AIC, especially in larger datasets.
*(Simple version: AIC always charges the same flat fee per extra parameter; BIC's fee grows with your sample size, making it stricter as your data grows.)*

**Q: What does it mean if a candidate model's Mallows' $C_p$ is much larger than its number of parameters $p$?**
A: It signals the model is likely missing important predictors (underfitting/biased) — a well-specified model should have $C_p$ close to $p$.
*(Simple version: the model's real-world error is bigger than a "complete" model of that size should produce — a sign something important got left out.)*

**Q: Why might a predictor be worth keeping in a model-selection sense (AIC/BIC/adjusted $R^2$/$C_p$) even if its individual t-test isn't significant?**
A: These criteria evaluate the model's overall fit-versus-complexity tradeoff, which is a different question from an individual coefficient's statistical significance — especially under multicollinearity, where individual t-tests are underpowered even though the predictor contributes real joint explanatory value.
*(Simple version: "is this one coefficient significant alone" and "does the whole model earn its complexity" are two different questions, and they can disagree — especially when two predictors overlap.)*

**Q: What's a major criticism of stepwise selection procedures?**
A: They perform many sequential hypothesis tests without correcting for multiple comparisons, producing overoptimistic fit statistics and unstable model choices that can vary substantially with small changes in the data — modern practice generally prefers regularization or cross-validation instead.
*(Simple version: it's like reporting your best result out of many hidden attempts — the final numbers look better than they honestly are.)*

---

*End of Chapter 14. Next: Chapter 15 — Variable Selection & Overfitting (the train/test split, why in-sample fit statistics like $R^2$ can be misleading, and a first introduction to cross-validation as the more robust alternative to Chapter 14's classical criteria).*
