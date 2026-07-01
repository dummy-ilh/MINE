# Module 1 — Odds, Log-Odds, and the Logit

## 1. WHY

We just established: we can't model probability directly with a straight line, because probability is trapped between 0 and 1, and a straight line refuses to stay trapped anywhere.

So here's the trick statisticians came up with: **don't model probability directly. Model something else — something that CAN range from −∞ to ∞ — and then translate it back into a probability at the very end.**

## 2. INTUITION — What are "odds"?

You've heard this in a betting context: *"The odds of that horse winning are 3 to 1."*

That sentence means: **for every 1 time the horse loses, we expect it to win 3 times.** Or: winning is 3 times more likely than losing.

Odds is just a **ratio**: 

**odds = (chances something happens) / (chances it doesn't happen)**

Compare this to probability, which is:

**probability = (chances something happens) / (all possible outcomes)**

Probability is "wins out of total races." Odds is "wins compared to losses." Same underlying data, different way of expressing it.

**Why does this matter for us?** Because probability is stuck in [0, 1], but odds can go much higher — all the way up to +∞ (if something is nearly certain, wins vastly outnumber losses). That's progress — we've unlocked one side of the number line. But odds still can't go negative — it stops at 0 (if something never happens). We need one more step to unlock the *other* side (negative numbers). That's what "log" will do, in a moment.

## 3. SIMPLE FORMULA — Odds

**In words:**
> Odds of an event = (probability the event happens) divided by (probability the event does NOT happen)

**In simple notation:**

```
odds = p / (1 - p)
```

- `p` = probability the event happens (a number between 0 and 1)
- `1 - p` = probability the event does NOT happen
- `odds` = the ratio between the two

## 4. WORKED NUMERIC EXAMPLE — Odds

Say a customer has a **p = 0.75** (75%) probability of churning.

```
odds = 0.75 / (1 - 0.75)
odds = 0.75 / 0.25
odds = 3.0
```

**In plain English:** "This customer is 3 times more likely to churn than to stay." That matches the horse-racing intuition — "3 to 1."

Let's do a couple more, so the pattern sticks:

| Probability (p) | 1 - p | Odds = p / (1-p) | Plain English |
|---|---|---|---|
| 0.50 | 0.50 | 1.0 | Equally likely to happen or not |
| 0.90 | 0.10 | 9.0 | 9x more likely to happen than not |
| 0.10 | 0.90 | 0.111 | About 9x more likely NOT to happen |
| 0.99 | 0.01 | 99.0 | Extremely likely |

Notice something: probability only moves between 0 and 1, but odds can shoot up toward infinity (as p approaches 1) or shrink toward 0 (as p approaches 0). We've stretched out one side of the ruler. But it still can't go negative — the smallest odds can ever be is 0. **We're halfway to an unbounded scale.**

## 5. NOW — the log-odds (the "logit")

**WHY:** Odds solved half our problem (it can now go up to infinity), but it's still stuck at a floor of 0 — it can never be negative. We want a number that can be truly unbounded in *both* directions, so a straight-line model can predict it freely without breaking.

**INTUITION:** Take the **logarithm** of the odds. Here's the plain-English reason logs help: odds between 0 and 1 (meaning the event is less likely than not) become *negative* numbers once you take their log. Odds above 1 (event is more likely than not) become *positive* numbers. And odds of exactly 1 (50/50) become exactly 0. That's exactly the symmetric, unbounded number line we wanted!

**SIMPLE FORMULA:**

**In words:**
> Log-odds = the natural logarithm of the odds

**In simple notation:**

```
logit = log(odds) = log( p / (1-p) )
```

- `p` = probability the event happens
- `odds` = p / (1-p), as computed above
- `log(...)` = natural logarithm (log base e) — don't worry about "why base e" yet, just think of it as "the standard log button"
- `logit` = just a shorter name for "log-odds" — it's the name mathematicians gave this specific quantity

## 6. WORKED NUMERIC EXAMPLE — Log-Odds

Continuing our churn example, where **p = 0.75**, so **odds = 3.0**:

```
logit = log(3.0)
logit = 1.0986
```

Let's also compute the flip side — what if p = 0.25 (25% chance of churning)?

```
odds = 0.25 / 0.75 = 0.333
logit = log(0.333)
logit = -1.0986
```

**Notice the symmetry:** p = 0.75 gives logit = +1.0986. p = 0.25 (the mirror image) gives logit = **−1.0986** — same magnitude, flipped sign. That's the unbounded, symmetric number line we were after.

And the midpoint check: p = 0.50 →

```
odds = 0.50 / 0.50 = 1.0
logit = log(1.0) = 0
```

Log-odds of exactly 0 means "totally undecided, 50/50." Positive logit = "more likely than not." Negative logit = "less likely than not." This is a clean, intuitive number line.

Here's the full picture as a table:

| p | odds | logit = log(odds) |
|---|---|---|
| 0.01 | 0.0101 | -4.60 |
| 0.10 | 0.111 | -2.20 |
| 0.25 | 0.333 | -1.10 |
| 0.50 | 1.0 | 0.00 |
| 0.75 | 3.0 | +1.10 |
| 0.90 | 9.0 | +2.20 |
| 0.99 | 99.0 | +4.60 |

Now you can see it end to end: **p is squeezed into [0,1]... odds stretches that to [0, ∞)... and log(odds) stretches that further into (−∞, +∞).** We've built a fully unbounded scale, one step at a time, starting from a bounded probability.

## 7. So what IS a "link function"?

Now the payoff — the term you said trips you up.

**A link function is just the name for the math operation that translates between "probability world" (bounded, 0 to 1) and "straight line world" (unbounded, −∞ to ∞).**

In our case, the link function is exactly what we just built: **logit(p) = log(p / (1-p))**. That's it. That's the whole mystery. "Link function" is just formal vocabulary for "the translator function." Logistic regression's specific translator is called the **logit link**.

Here's the mental model to keep forever:

```
PROBABILITY WORLD  --[link function: logit]-->  STRAIGHT-LINE WORLD
   (0 to 1)                                          (−∞ to +∞)
```

Once we're in "straight-line world," we CAN do ordinary linear regression math (weighted sum of features) without breaking anything — because that world has no boundaries to violate. Then, at the end, we need a way to translate *back* from straight-line world to probability world so we get a usable prediction. That "un-translator" is the **sigmoid function** — which is exactly what Module 2 covers next.

## 8. INTERPRETATION

In real terms: when you fit a logistic regression model, internally it's predicting **log-odds**, not probability directly. A coefficient in the model tells you "how much does the log-odds change when this feature increases by 1 unit" —

## 9. FAANG L5 ANGLE

**Common interview question:** *"What is a link function, and why does logistic regression use the logit link?"*

Strong answer: "A link function maps the bounded probability space to an unbounded real-number space so we can model it linearly. Logistic regression uses the logit — log-odds — because it's the natural link that comes out of assuming the data follows a Bernoulli/binomial distribution" (you don't need the distributional theory yet, just know this phrase exists).

**Common follow-up:** *"Why not just use log(p) directly as the link, instead of log(p/(1-p))?"*
Good answer: log(p) alone is only bounded on one side — it still can't exceed 0 no matter what p is (since p ≤ 1 means log(p) ≤ 0), so it never becomes unbounded on the positive side. You need the *odds ratio* first (which stretches things toward +∞), and only then log gives you full two-sided unboundedness.

**Common trap:** Candidates memorize "logit = log(p/(1-p))" as a formula but can't explain *why* each step (odds, then log) is necessary. Interviewers at L5 want you to justify the construction, not just recite it.

## 10. QUICK PYTHON CHECK

```python
import numpy as np

def odds(p):
    return p / (1 - p)

def logit(p):
    return np.log(odds(p))

# verify our table
for p in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
    print(f"p={p:.2f}  odds={odds(p):.3f}  logit={logit(p):.3f}")
```

## 11. CHECK — before Module 2

1. If a feature's odds ratio is 2.0, what does that mean in plain English about the event happening?
Odds of 2.0 = the event is twice as likely to happen as not happen. ✓
2. Why can't we stop at "odds" and skip taking the log — what specifically would still be broken if we tried to model raw odds with a straight line?
Odds is bounded at 0 on the low end (can never go negative), so it's only "half-unbounded" — that's precisely why we needed the log step to get a truly symmetric, unbounded number line. ✓

## Additional

p: bounds [0, 1], why sigmoid enforces this

Odds: p/(1-p), bounds [0, ∞), interpretation as "ratio of probabilities"

Log-odds (logit): ln(p/(1-p)), bounds (-∞, ∞), why this is the natural parameter for Bernoulli

Emphasize that linear layers output logits (unbounded), and sigmoid maps to probabilities









Excellent! Let's break this down step-by-step, both mathematically and with a concrete example.

---

# 1. The Mathematical Derivation

We start with the **log-odds** of probability `p`:

$$
\text{log-odds}(p) = \ln\left(\frac{p}{1-p}\right)
$$

Now, consider the **log-odds** of the complement probability `1 - p`:

$$
\text{log-odds}(1-p) = \ln\left(\frac{1-p}{p}\right)
$$

We want to prove:

$$
\ln\left(\frac{1-p}{p}\right)
=
-\ln\left(\frac{p}{1-p}\right)
$$

---

## Step 1: Rewrite the Fraction

Notice that:

$$
\frac{1-p}{p}
=
\frac{1}{\frac{p}{1-p}}
$$

So,

$$
\ln\left(\frac{1-p}{p}\right)
=
\ln\left(\frac{1}{\frac{p}{1-p}}\right)
$$

---

## Step 2: Apply the Logarithm Rule

Recall the logarithm identity:

$$
\ln\left(\frac{1}{x}\right)
=
-\ln(x)
$$

Let

$$
x=\frac{p}{1-p}
$$

Then,

$$
\ln\left(\frac{1}{\frac{p}{1-p}}\right)
=
-\ln\left(\frac{p}{1-p}\right)
$$

---

## Step 3: Conclusion

Therefore,

$$
\boxed{
\ln\left(\frac{1-p}{p}\right)
=
-\ln\left(\frac{p}{1-p}\right)
}
$$

Or, in log-odds notation,

$$
\boxed{
\text{log-odds}(1-p)
=
-\text{log-odds}(p)
}
$$

This property is called **anti-symmetry**.

---

# 2. Why This Makes the Sigmoid Symmetric

The sigmoid function is the inverse of the log-odds function:

$$
p
=
\sigma(z)
=
\frac{1}{1+e^{-z}}
$$

where

$$
z=\text{log-odds}(p)
$$

From the anti-symmetry result,

$$
\text{log-odds}(1-p)
=
-z
$$

Therefore,

If

$$
\sigma(z)=p
$$

then

$$
\sigma(-z)=1-p
$$

This means the sigmoid curve has **rotational symmetry** about the point

$$
(0,\;0.5)
$$

Moving the same distance to the left or right of zero produces complementary probabilities.

---

# 3. Concrete Example

Let's choose

$$
p=0.75
$$

### Step 1: Compute the Log-Odds of 0.75

$$
\text{log-odds}(0.75)
=
\ln\left(\frac{0.75}{0.25}\right)
=
\ln(3)
\approx
1.0986
$$

---

### Step 2: Compute the Log-Odds of 0.25

Since

$$
1-p=0.25
$$

we get

$$
\text{log-odds}(0.25)
=
\ln\left(\frac{0.25}{0.75}\right)
=
\ln\left(\frac13\right)
=
-\ln(3)
\approx
-1.0986
$$

---

### Step 3: Verify Anti-Symmetry

We have

$$
\text{log-odds}(0.75)
=
1.0986
$$

and

$$
\text{log-odds}(0.25)
=
-1.0986
$$

So,

$
1.0986
=
-
(-1.0986)
\quad\checkmark
$

or equivalently,

$
\ln\left(\frac{0.25}{0.75}\right)
=
-
\ln\left(\frac{0.75}{0.25}\right)
$

Exactly as predicted.

---

# 4. Verify Using the Sigmoid Function

Now evaluate the sigmoid.

For

$
z=1.0986
$

$
\sigma(1.0986)
=
0.75
$

For

$
z=-1.0986
$

$
\sigma(-1.0986)
=
0.25
=
1-0.75
$

So,
$
\sigma(-z)
=
1-\sigma(z)
$

which confirms the symmetry property.

---

# Key Insight

The log-odds function is **anti-symmetric** because exchanging `p` and `1-p` simply takes the reciprocal of the odds ratio, and taking the logarithm of a reciprocal flips the sign:

$
\ln\left(\frac{1}{x}\right)
=
-\ln(x)
$

Since the sigmoid is the inverse of the log-odds function, this immediately implies

$
\boxed{
\sigma(-z)
=
1-\sigma(z)
}
$

This is why the sigmoid curve is perfectly symmetric around the point **(0, 0.5)**.
