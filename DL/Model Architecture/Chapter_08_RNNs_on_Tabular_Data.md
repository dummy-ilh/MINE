# Chapter 8: RNNs on Tabular Data

## Why this pairing is unusual

Tabular data — rows in a spreadsheet, each independent, each with the same fixed set of columns — is normally the territory of MLPs and gradient-boosted trees (XGBoost, LightGBM), not RNNs. The whole premise of an RNN is *sequence*: order matters, length varies, and each step depends on what came before. Most tabular datasets have none of that — row 5 and row 500 are unrelated customers, and shuffling the rows changes nothing.

So when does an RNN actually make sense on tabular data? Two genuine cases, and one dubious trick worth knowing about for interviews.

## Case 1: the rows aren't actually independent — they're a history

This is the real, common case. If your "tabular" data is really **panel data** — repeated snapshots of the same entity over time — then each entity's rows form a genuine sequence, even though the data is stored as a flat table.

**Example.** Predicting customer churn from monthly account snapshots:

| customer | month | spend | visits |
|---|---|---|---|
| A | 1 | 50 | 2 |
| A | 2 | 80 | 3 |
| A | 3 | 30 | 1 |
| B | 1 | 20 | 1 |
| B | 2 | 25 | 1 |

Naively, this looks like 5 independent tabular rows. But rows for customer A form a real sequence: month 1 → month 2 → month 3, in order, with month 3's low activity plausibly *caused by* or *following from* what happened in month 2. This is exactly a **many-to-one** RNN (Chapter 6, architecture #3): feed in the sequence of monthly feature vectors, get out one churn probability at the end.

**Reshaping into a sequence.** For customer A:
$$x_1 = [50, 2], \quad x_2 = [80, 3], \quad x_3 = [30, 1]$$

Each $x_t$ is now a small feature vector (spend, visits) at timestep $t$, instead of a flat row. Run it through the same forward pass from Chapter 2 — $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1}+b_h)$ — and read out a churn probability from $h_3$, the final hidden state.

**Why bother, instead of just using a normal model?** Two real advantages:

1. **Variable-length histories.** Customer A has 3 months of data, customer B has 2. An RNN naturally handles both with the same weights — no padding tricks needed beyond the usual sequence-length handling. A standard MLP or XGBoost model needs a *fixed* number of input columns, so you'd have to pick a fixed window (say, "always use the last 3 months") and pad or truncate everyone to fit.
2. **Weight sharing across time** means the model learns one "how does this customer's trajectory tend to evolve" rule, applied at every month, rather than needing separate features engineered for "month 1 pattern" vs. "month 2 pattern."

**Why you might *not* bother:** on most real tabular benchmarks, gradient-boosted trees (XGBoost, LightGBM, CatBoost) outperform RNNs and even MLPs, especially on small-to-medium datasets — trees handle mixed feature types, missing values, and non-smooth interactions better out of the box. The common practical baseline is to **flatten** the history into one row (spend_month1, visits_month1, spend_month2, visits_month2, ...) and feed that to XGBoost. This works well when the number of time periods is small and fixed. The RNN approach earns its keep specifically when histories are long, variable-length, or numerous enough that flattening becomes impractical.

## Case 2: no natural time axis, but the same entity has grouped multi-row structure

Sometimes rows aren't monthly snapshots but still belong together — e.g., all the line items in one insurance claim, or all the products in one shopping cart. There's a "sequence" in the sense of "a variable-length group of related rows," even without a literal time axis. The same many-to-one RNN setup applies: feed in the group's rows one at a time (in whatever order they're naturally stored, e.g. by item ID or timestamp of add-to-cart), and predict one outcome for the whole group (e.g., "will this cart convert to a purchase?"). The RNN's variable-length handling is the main draw here, not any real temporal dependency.

## Case 3 (the dubious one): treating columns themselves as a sequence

You'll sometimes see this in older Kaggle-style tricks: instead of feeding a normal tabular row `[age, income, credit_score]` into an MLP all at once, feed it into an RNN **one column at a time**, treating "column 1, column 2, column 3" as a fake timestep sequence.

**Why this is questionable:** columns in a tabular dataset usually have **no natural order**. Age, income, and credit score aren't sequential the way months are — there's no reason column 2 should "follow from" column 1 the way $h_2$ follows from $h_1$ in a real sequence. An RNN's core assumption (order carries information, and the same update rule should apply at every step) doesn't hold. This technique occasionally shows up as a regularization or feature-interaction trick and can work by accident, but it's not principled the way Case 1 is — flag it as a curiosity if it comes up, not a recommended approach.

## Decision rule

| Situation | Recommended approach |
|---|---|
| Rows are genuinely independent (no entity history) | MLP or gradient-boosted trees — skip RNNs entirely |
| Repeated snapshots per entity, fixed & small number of periods | Flatten history into one row, use XGBoost/MLP (usually wins) |
| Repeated snapshots per entity, variable-length or long histories | Many-to-one RNN (or LSTM/GRU) over the per-entity sequence |
| Variable-length groups of related rows, no real time axis | Many-to-one RNN, mainly for the variable-length handling |
| Feeding individual columns as fake timesteps | Avoid unless you have specific evidence it helps — not principled |

## What's ahead

Chapter 9 — the last chapter — builds a vanilla RNN from scratch: first in raw NumPy (so every line maps back to the equations from Chapter 2 and the gradients from Chapter 3), then the equivalent in PyTorch.

---

**One-line summary:** RNNs earn their place on tabular data only when rows genuinely form a sequence — usually repeated snapshots of the same entity over time — and even then, only when the history is variable-length or too long to simply flatten into one row for a tree-based model.
