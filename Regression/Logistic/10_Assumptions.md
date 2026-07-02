# Module 10 — Assumptions & Diagnostics

## 1. WHY

Every model makes implicit assumptions about the data it's trained on. If those assumptions are badly violated, the model can still *run* and produce numbers — but those numbers become unreliable in ways that are easy to miss unless you know what to check. This module answers: **what does logistic regression silently assume about your data, and how do you catch it if those assumptions are broken?**

**What breaks if you ignore this:** You could ship a model with seemingly reasonable coefficients that are actually unstable, misleading, or wildly overconfident — not because the math is wrong, but because the data violated an assumption the math depends on. This is a favorite area for L5 interviewers because it separates "I can call `.fit()`" from "I understand what I'm fitting."

## 2. INTUITION

Think of logistic regression's assumptions like the fine print on a ladder: *"Only use on flat, stable ground."* The ladder still physically exists and you can still climb it on uneven ground — but the manufacturer's safety guarantee no longer applies, and it might wobble or tip in ways you didn't expect. Assumptions in statistics work the same way: the model still fits and produces output even when assumptions are violated, but the guarantees about that output (reliable coefficients, valid interpretations, stable predictions) quietly stop holding.

## 3. THE THREE MAIN ASSUMPTIONS

### Assumption 1 — Linearity in the Log-Odds

**In words:** logistic regression assumes that each feature has a **constant, straight-line relationship with the LOG-ODDS** of the outcome (not with the probability itself — remember Module 3, the S-curve means the relationship with raw probability is never straight-line). If you plotted log-odds against a feature, it should look like a straight line, not a curve.

**What breaks if violated:** if the true relationship is actually curved (e.g., churn risk first drops, then rises again as tenure increases — a U-shape), a plain linear term will fit an "average" straight line through that curve, systematically **mispredicting both ends** — underestimating risk for very new AND very long-tenured customers, if the truth is U-shaped.

**How to check:** plot each continuous feature against the empirical log-odds of the outcome (bucket the feature into groups, compute log-odds of the target within each group, plot it). If it looks like a smooth curve rather than a line, you have a linearity violation.

**Fix if violated:** add polynomial terms (x², x³) or bin the continuous feature into categories, or add interaction terms — anything that lets the model capture curvature it otherwise can't express (this connects directly back to Module 6's discussion of linear decision boundaries).

### Assumption 2 — Independence of Observations

**In words:** each row (data point) in your dataset is assumed to be **independent of every other row** — knowing the outcome for one customer should give you no information about another customer's outcome, beyond what's captured in the features themselves.

**What breaks if violated:** a classic violation is **repeated measurements from the same entity** — e.g., 10 rows for the same customer across 10 different months, or in a medical context, multiple visits from the same patient. If you treat these as fully independent, the model (and any statistical inference like confidence intervals) becomes overconfident — it thinks it has more independent evidence than it actually does, since those 10 rows aren't really 10 independent "opinions," they're really more like 1 customer's story told 10 times.

**How to check:** think about your data collection process — are there natural groupings (same user, same store, same time period) that could create correlated rows? Time-series data and panel data (repeated entities over time) are the most common red flags.

**Fix if violated:** use models designed for this structure (mixed-effects/hierarchical models), aggregate to one row per entity, or at minimum, use cluster-robust standard errors to avoid overstating your confidence.

### Assumption 3 — No Severe Multicollinearity

**In words:** the features you feed into the model shouldn't be **too strongly correlated with each other**. Each feature should bring genuinely distinct information to the table.

**What breaks if violated:** if two features are highly correlated (e.g., "monthly income" and "annual income" — literally the same information, just scaled differently), the model struggles to figure out how much credit to assign to EACH one individually. The coefficients become unstable — tiny changes in the data can cause big, erratic swings in the individual coefficient values (even flipping their sign), even though the model's overall PREDICTIONS might still be fine. This directly undermines coefficient interpretation (Module 3) — if you can't trust individual coefficients, you can't reliably say "this feature causes a 2x increase in odds."

**How to check:** compute a **correlation matrix** between features (quick, simple check) or, more rigorously, compute **VIF (Variance Inflation Factor)** for each feature — a common rule of thumb is VIF > 5 or 10 signals a problem worth addressing.

**Fix if violated:** drop one of the correlated features, combine them into a single composite feature, or use regularization (Module 7 — specifically L2/Ridge is known to help stabilize coefficients under multicollinearity, since it shrinks correlated features' weights together rather than letting one dominate erratically).

## 4. WORKED NUMERIC EXAMPLE — Checking Linearity in Log-Odds

Let's check whether "months as customer" (tenure) has a linear relationship with log-odds of churn, using binned data:

| Tenure bucket | Customers in bucket | Churned | Empirical p (churned/total) | Empirical log-odds |
|---|---|---|---|---|
| 0-6 months | 100 | 40 | 0.40 | log(0.40/0.60) = -0.405 |
| 6-12 months | 100 | 25 | 0.25 | log(0.25/0.75) = -1.099 |
| 12-24 months | 100 | 15 | 0.15 | log(0.15/0.85) = -1.735 |
| 24-36 months | 100 | 10 | 0.10 | log(0.10/0.90) = -2.197 |

**Check the differences between consecutive log-odds values:**
```
-1.099 - (-0.405) = -0.694
-1.735 - (-1.099) = -0.636
-2.197 - (-1.735) = -0.462
```

These differences are roughly similar in size (-0.69, -0.64, -0.46) — not identical, but reasonably close, suggesting an **approximately linear** relationship (each additional tenure bucket reduces log-odds by a roughly similar amount). If instead these differences were wildly different in size (say, -0.69, -0.05, -3.2), or changed sign (some positive, some negative), that would be a red flag suggesting a genuinely curved (non-linear) relationship that a straight-line model would badly misrepresent.

## 5. INTERPRETATION

In real terms: before trusting your model's coefficients enough to make business claims ("every additional complaint triples churn odds"), it's worth a quick sanity pass on these three assumptions. A 10-minute correlation matrix check or log-odds binning plot can save you from shipping a model whose individual coefficients are misleading, even if its overall predictions still look "fine" on a dashboard. This is especially important when the model's *coefficients themselves* — not just its predictions — are being used to drive decisions (e.g., "which lever should the business pull?").

## 6. FAANG L5 ANGLE

**Common interview question:** *"What assumptions does logistic regression make, and how would you check them?"*
Strong answer: name all three — linearity in log-odds (check via binned log-odds plots), independence of observations (check via understanding data collection/entity structure), no severe multicollinearity (check via correlation matrix or VIF) — and pair each with its practical fix.

**Common follow-up:** *"If two features are highly correlated, does that hurt the model's PREDICTIONS or just its coefficients?"*
Sharp answer: usually just the coefficients (and their interpretability/stability) — the model can often still predict reasonably well even under multicollinearity, since it can distribute "credit" between the correlated features in many different ways and still land on similar overall predictions. The real danger is trusting an individual coefficient's value or sign when multicollinearity is present.

**Common follow-up:** *"How is 'independence of observations' different from 'no multicollinearity'?"*
Good answer: independence is about **rows** (are your data points/observations independent of each other), while multicollinearity is about **columns** (are your features independent of each other). Candidates who confuse these two show they're pattern-matching on jargon rather than understanding the underlying assumption.

**Common trap:** treating "logistic regression assumes a linear relationship between features and the outcome" as true — this is imprecise and will be flagged. The correct, precise statement is "linear relationship between features and the LOG-ODDS of the outcome," not the probability or the raw outcome itself. This exact wording distinction is a favorite trip-up.

## 7. QUICK PYTHON CHECK

```python
import numpy as np

# Empirical probabilities by tenure bucket
p_values = np.array([0.40, 0.25, 0.15, 0.10])
log_odds = np.log(p_values / (1 - p_values))

print("Log-odds per bucket:", np.round(log_odds, 3))
print("Differences between buckets:", np.round(np.diff(log_odds), 3))
```

## 8. CHECK — before Module 11

1. You compute a correlation matrix and find two features have a correlation of 0.95. Your model's overall accuracy still looks fine. Should you be concerned? Why or why not, and what specifically would you be worried about?
2. A colleague says "logistic regression assumes a linear relationship between the features and churn." What's imprecise about this statement, and how would you correct it?

Check 1 — confirmed
Right: high correlation (0.95) between two features is a real problem for coefficient reliability — you can't trust either individual coefficient's magnitude or even its sign, since the model can't cleanly separate credit between two nearly-identical features. But it's usually not a problem for overall prediction quality — the model can still combine the two correlated features in some workable way and produce accurate predictions, even if the individual weights look strange or unstable. The practical takeaway: if you're using this model purely to generate predictions (a score, a ranking), multicollinearity is a lower priority. If you're using it to explain WHY (e.g., "which feature drives churn most"), it becomes a serious, non-negotiable concern.
Check 2 — confirmed
Exactly right. The precise statement is: logistic regression assumes a linear relationship between features and the log-odds of the outcome — not the outcome itself, and not the probability. Saying "linear relationship with churn" is ambiguous and technically wrong, since we know from Module 3 that the relationship with raw probability is curved (the S-shape), and "churn" alone doesn't specify which scale (probability vs. log-odds) you mean. Precise language here is exactly the kind of thing that separates a strong L5 answer from a vague one.
