# Chapter 5: Metric Selection — Offline

## 1. Why This Chapter Exists

By now you have a target (Ch. 3), a labeling strategy (Ch. 4), and a task type (Ch. 2). This chapter answers: **how do you numerically judge whether a model is good, before it ever sees production traffic?** The core message: the "right" offline metric isn't a universal choice — it's derived from the cost structure of the decision the model feeds, and picking the wrong one silently optimizes for the wrong tradeoff even with a perfectly-formulated target.

## 2. Classification Metrics and What They Actually Assume

Start from the confusion matrix: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).

$$
\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P+R}
$$

**These are not interchangeable, and the choice encodes an implicit cost assumption.** $F_1$ specifically assumes false positives and false negatives are *equally costly* — which is almost never true in real decisions.

### Worked Numerical Example: Fraud Detection Threshold Choice

Suppose a fraud model on 10,000 transactions/day produces this confusion matrix at threshold $\tau = 0.5$:

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actually Fraud** | TP = 80 | FN = 20 |
| **Actually Legit** | FP = 200 | TN = 9,700 |

Precision $= 80/280 = 0.286$, Recall $= 80/100 = 0.80$, $F_1 = 2(0.286)(0.80)/(0.286+0.80) = 0.421$.

Now suppose the business cost structure is: a missed fraud (FN) costs the company an average \$500 (the fraud loss itself), while a false alarm (FP) costs \$5 (a customer service review + minor friction, most customers aren't lost over one declined-then-approved transaction). Total cost at this threshold:

$$
\text{Cost} = 20 \times \$500 + 200 \times \$5 = \$10{,}000 + \$1{,}000 = \$11{,}000/\text{day}
$$

If we raise the threshold to $\tau = 0.7$ (more conservative, flags fewer transactions) and get FN=45, FP=60: Cost $= 45(500) + 60(5) = \$22{,}500 + \$300 = \$22{,}800$ — *worse*, despite fewer false alarms, because in this cost structure a missed fraud is 100x costlier than a false alarm, so recall matters far more than precision. $F_1$ alone, treating FP and FN symmetrically, would not surface this — you need the explicit dollar-cost calculation to pick the right operating point, not just an aggregate metric that hides the asymmetry inside its harmonic mean.

**The generalizable lesson**: whenever FP and FN have meaningfully different real-world costs (nearly always), report and optimize a cost-weighted metric or explicitly choose your threshold via a cost curve, rather than defaulting to $F_1$ or accuracy.

## 3. Threshold-Free Metrics: AUC-ROC and AUC-PR

**AUC-ROC** measures ranking quality across *all* thresholds simultaneously: the probability that a randomly chosen positive example is ranked above a randomly chosen negative one.

$$
\text{AUC} = P(\hat{y}_{\text{positive example}} > \hat{y}_{\text{negative example}})
$$

**Critical pitfall**: AUC-ROC is computed against the *negative class rate*, which makes it dangerously optimistic under severe class imbalance. Consider fraud again with 1% true positive rate: even a mediocre model can achieve AUC-ROC $\approx 0.90$ while producing an unusable number of false positives in absolute terms, because the ROC curve's false-positive-rate axis is normalized by the (huge) negative class, hiding how many *absolute* false positives you rack up on the way to a high true-positive rate.

**AUC-PR** (precision-recall curve area) is far more informative under imbalance because both axes are anchored to the (small) positive class — it directly shows you the precision cost of achieving a given recall level, which is exactly the tradeoff Section 2 quantified in dollars. As a rule of thumb: **for rare-event problems (fraud, disease detection, rare-defect detection), prefer AUC-PR over AUC-ROC as your headline offline metric**, and always report a raw confusion matrix at a realistic operating threshold alongside either.

## 4. Regression Metrics

$$
\text{MSE} = \frac{1}{n}\sum (\hat y_i - y_i)^2, \qquad \text{MAE} = \frac{1}{n}\sum |\hat y_i - y_i|
$$

**MSE** penalizes large errors quadratically — appropriate when large errors are disproportionately costly (e.g., wildly underpricing a house). **MAE** penalizes linearly — appropriate when cost is roughly proportional to error size, and it's notably more robust to outliers (a single wildly-off prediction doesn't dominate the metric the way it does under MSE, because squaring an outlier's error amplifies its contribution far more than a linear term does).

**Worked numeric contrast**: predictions vs. true values $[(\hat y, y)] = [(10,12), (20,22), (5,45)]$ (last one is a bad miss). MAE $= (2+2+40)/3 = 14.67$. MSE $= (4+4+1600)/3 = 536$. If you optimize MSE, that one bad outlier (error 40) dominates the objective far more than in MAE-space (its squared contribution, 1600, is ~400x the other two points' combined 8, whereas under MAE it's only ~10x) — meaning an MSE-trained model will contort itself disproportionately to reduce that single large error, potentially at the cost of accuracy on the bulk of "normal" points. Whether that's desirable depends entirely on whether large errors really are disproportionately costly in the business sense (e.g., mispricing a house by \$400k is catastrophic in a way 10 separate \$4k errors aren't) — if not, MAE or Huber loss (which blends the two, quadratic near zero and linear beyond a threshold) is usually the better production choice.

## 5. Ranking Metrics: NDCG

For ranking, use **NDCG@K** (Normalized Discounted Cumulative Gain), which rewards placing highly-relevant items near the top of a list more than placing them lower:

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}, \qquad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

where $IDCG@K$ is the DCG of the ideal (perfectly sorted) ordering, used to normalize into $[0,1]$.

**Worked numeric example**: true relevances (graded 0–3) for the top 3 slots returned by a model: $[3, 0, 2]$ (best item first, then an irrelevant one, then a good one).

$$
DCG@3 = \frac{2^3-1}{\log_2 2} + \frac{2^0-1}{\log_2 3} + \frac{2^2-1}{\log_2 4} = \frac{7}{1} + \frac{0}{1.585} + \frac{3}{2} = 7 + 0 + 1.5 = 8.5
$$

Ideal ordering would be $[3, 2, 0]$: $IDCG@3 = \frac{7}{1} + \frac{3}{1.585} + \frac{0}{2} = 7 + 1.89 + 0 = 8.89$.

$$
NDCG@3 = 8.5 / 8.89 = 0.956
$$

Notice the position discount ($\log_2(i+1)$ in the denominator) is exactly why the metric prefers correct items at the top: the same relevance score contributes less to DCG the later it appears, mirroring the reality that users rarely look past the first few results — a relevant item at position 10 does far less good than the same item at position 1, and NDCG is constructed specifically to reflect that in the number itself.

## 6. Production Considerations

- **Offline metric choice should mirror the true decision cost function whenever that cost is knowable** (as in the fraud dollar-cost example) — if it's knowable, compute it and use it as (or alongside) your headline metric, don't default to a generic proxy like $F_1$ out of habit.
- **Class imbalance makes metric choice a first-order decision, not a footnote** — defaulting to accuracy or AUC-ROC on a 1%-positive-rate problem can make a nearly-useless model look excellent (Section 3).
- **NDCG@K's choice of $K$ should match the actual UI surface** (e.g., NDCG@10 for a page showing 10 results is meaningful; NDCG@1000 is not, since users never see position 1000).
- **A single scalar metric is rarely sufficient for a launch decision** — pairing a headline metric with a raw confusion matrix (classification) or a calibration plot (regression) at a realistic operating point catches issues an aggregate number hides.

## 7. Common Interview Traps

- **Defaulting to accuracy on an imbalanced problem** without noting that a trivial "always predict majority class" baseline could already score 99% accuracy on a 1%-positive-rate task, making accuracy nearly meaningless there.
- **Quoting AUC-ROC as if it's automatically the right choice for any classification problem**, without checking class balance (Section 3's core point).
- **Using MSE without asking whether large errors really are disproportionately costly** in the business sense, rather than defaulting to it out of familiarity.
- **Choosing $F_1$ reflexively** for problems with clearly asymmetric FP/FN costs, without computing (or at least gesturing at) the actual cost-weighted alternative from Section 2.

## 8. L5-Differentiating Talking Points

- Derive or sketch a cost-weighted metric from the actual dollar/harm costs of FP vs. FN whenever those costs are estimable, rather than reaching for a generic classification metric by default — this is the single clearest signal of moving from "I know the metric formulas" to "I know how to choose one."
- Explicitly flag AUC-ROC's optimism under class imbalance and proactively suggest AUC-PR instead for rare-event problems.
- Tie the choice of $K$ in NDCG@K back to the actual UI surface being optimized, rather than treating $K$ as an arbitrary hyperparameter.
- Note that offline metrics are a necessary but not sufficient gate for launch — foreshadowing Chapter 6's point that an offline win doesn't guarantee an online one.

## 9. Comprehension Checks

1. A medical screening test has FN cost (a missed disease case) far higher than FP cost (an unnecessary follow-up test). Using the style of Section 2's fraud example, explain qualitatively why $F_1$ is a poor headline metric here and what alternative you'd propose.
2. Compute AUC-PR's core advantage over AUC-ROC in your own words using a scenario with 0.5% positive rate — why does AUC-ROC's false-positive-rate axis understate the real-world false-positive burden?
3. For a housing price regression model, argue for MSE vs. MAE using a concrete scenario where large errors carry disproportionate real-world cost, and a second scenario where they don't.
4. Using the NDCG@3 formula, compute NDCG@3 for a ranking that returns relevances $[0, 3, 2]$ (worst item first) instead of $[3, 0, 2]$, and explain numerically why the score drops even though the *same* set of items with the *same* relevances is being returned, just reordered.

---

*End of Chapter 5. Chapter 6 will cover online and proxy metrics — surrogate/guardrail metrics, and why offline metric gains don't always translate to online business impact.*
