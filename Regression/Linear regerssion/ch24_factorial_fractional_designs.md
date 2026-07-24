# Chapter 24 — Two-Level Factorial and Fractional Factorial Designs
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Every design so far assumed you could afford to run every combination of every factor level. This chapter covers the classical industrial-statistics answer to "what if you have many candidate factors and can't afford a full factorial?" — **screening designs**, built around factors held at just **two levels** each, and their more economical cousin, **fractional factorial designs**, which deliberately trade away some information (via **confounding/aliasing**) to cut the number of required runs dramatically.

---

## 24.1 Two-Level Factorial Designs and Coded Notation

**Plain English.** Rather than modeling factors on their natural scale, two-level designs code every factor as $-1$ (the "low" setting) or $+1$ (the "high" setting) — e.g., Button Color: Red$=-1$, Blue$=+1$. This coding makes every factor comparable in scale and makes the arithmetic underlying "effect estimation" (below) remarkably clean.

### The Worked Example: A Full $2^3$ Factorial

Three factors, each at 2 levels: **Button Color** (A: Red$=-1$/Blue$=+1$), **Headline Length** (B: Short$=-1$/Long$=+1$), **Hero Image** (C: Absent$=-1$/Present$=+1$). A **full factorial** runs all $2^3=8$ combinations. Response: conversion rate.

| Run | A | B | C | Y |
|---|---|---|---|---|
| 1 | −1 | −1 | −1 | 15.5 |
| 2 | +1 | −1 | −1 | 18.5 |
| 3 | −1 | +1 | −1 | 16.5 |
| 4 | +1 | +1 | −1 | 25.5 |
| 5 | −1 | −1 | +1 | 17.5 |
| 6 | +1 | −1 | +1 | 20.5 |
| 7 | −1 | +1 | +1 | 18.5 |
| 8 | +1 | +1 | +1 | 27.5 |

---

## 24.2 Estimating Effects via Contrasts

### The Core Technique

**Plain English.** In a two-level design, every effect — main effect or interaction — is estimated the same simple way: **take the average response at the "+1" runs for that column, subtract the average response at the "−1" runs.** For interactions, you first construct a new column by **multiplying** the relevant factor columns together, then apply the exact same subtraction.

### Worked Example: Main Effects

**Effect of A:** average $Y$ where $A=+1$ (runs 2,4,6,8: $18.5,25.5,20.5,27.5$, average $23.0$) minus average $Y$ where $A=-1$ (runs 1,3,5,7: $15.5,16.5,17.5,18.5$, average $17.0$):
$$
\text{Effect}_A = 23.0-17.0=6.0
$$
**Effect of B:** ($B{=}{+}1$ avg $22.0$) − ($B{=}{-}1$ avg $18.0$) $=4.0$.
**Effect of C:** ($C{=}{+}1$ avg $21.0$) − ($C{=}{-}1$ avg $19.0$) $=2.0$.

### Worked Example: An Interaction Effect

**Construct the AB column** by multiplying each run's $A$ and $B$ values: Run 1 $(-1)(-1)=+1$; Run 2 $(+1)(-1)=-1$; Run 3 $(-1)(+1)=-1$; Run 4 $(+1)(+1)=+1$; Run 5 $+1$; Run 6 $-1$; Run 7 $-1$; Run 8 $+1$.

$$
\text{Effect}_{AB} = \underbrace{\frac{15.5+25.5+17.5+27.5}{4}}_{AB=+1,\ \text{avg}=21.5} - \underbrace{\frac{18.5+16.5+20.5+18.5}{4}}_{AB=-1,\ \text{avg}=18.5} = 3.0
$$

**Checking against the other possible interactions (AC, BC):** constructing the AC and BC columns the same way and applying the same subtraction gives $\text{Effect}_{AC}=0.0$ (the average at $AC=+1$ and $AC=-1$ both come out to exactly $20.0$) — correctly signaling **no** real AC interaction, exactly as this data was built to demonstrate. (The same clean-zero result holds for BC and the three-way ABC interaction, omitted here for space.)

**Why this contrast method is so elegant and computationally cheap.** No matrix inversion, no iterative fitting — every effect in a full two-level factorial is just a simple average-difference over an appropriately constructed column. This is *why* two-level designs became the industrial standard for cheaply screening many candidate factors: the arithmetic scales beautifully even with dozens of factors, as long as you can afford the (rapidly growing) $2^k$ number of runs.

---

## 24.3 Fractional Factorial Designs: Trading Information for Runs

### The Motivation

**Plain English.** A full factorial's run count **doubles with every additional factor** ($2^3=8$, $2^4=16$, $2^7=128\ldots$). With many candidate factors (common in real screening scenarios — dozens of possible product tweaks), a full factorial quickly becomes infeasible. **Fractional factorial designs deliberately run only a fraction** (e.g., a half, $2^{k-1}$) of the full design — at the cost of no longer being able to distinguish certain effects from each other.

### Constructing a Half-Fraction: Adding a 4th Factor to Our 8 Runs

Suppose we want to screen a **4th** factor — **Urgency Banner** (D: Absent$=-1$/Present$=+1$) — but can only afford **8 runs total** (a half-fraction of the full $2^4=16$, i.e., $2^{4-1}=8$). **The classic construction: define the new factor's column as the product of some existing columns** — here, set $D=ABC$ (the three-way interaction column from our existing design):

| Run | A | B | C | D=ABC |
|---|---|---|---|---|
| 1 | −1 | −1 | −1 | −1 |
| 2 | +1 | −1 | −1 | +1 |
| 3 | −1 | +1 | −1 | +1 |
| 4 | +1 | +1 | −1 | −1 |
| 5 | −1 | −1 | +1 | +1 |
| 6 | +1 | −1 | +1 | −1 |
| 7 | −1 | +1 | +1 | −1 |
| 8 | +1 | +1 | +1 | +1 |

**We get a 4th factor "for free," using the exact same 8 runs already collected** — but this is not actually free at all, as the next section shows.

---

## 24.4 The Defining Relation and Alias Structure

### Why $D=ABC$ Creates Confounding

**The defining relation** for this design is $I=ABCD$ (read: "the identity/mean column equals the ABCD product column" — a direct algebraic consequence of setting $D=ABC$, since multiplying both sides by $D$ gives $ABCD=D^2=I$, using the fact that any column times itself equals the all-$+1$ identity column in this coding).

**The alias structure** — which effects are confounded with which — is found by multiplying any effect by the defining relation ($I=ABCD$) and simplifying (any letter squared becomes $I$ and drops out):
$$
A\times(ABCD) = A^2BCD = BCD \quad\Rightarrow\quad A \text{ is aliased with } BCD
$$
Applying the same logic to every effect:
$$
B\leftrightarrow ACD, \qquad C\leftrightarrow ABD, \qquad D\leftrightarrow ABC
$$
$$
AB\leftrightarrow CD, \qquad AC\leftrightarrow BD, \qquad AD\leftrightarrow BC
$$

**What this means, concretely and consequentially.** When you compute "the effect of D" in this design, you are **not** measuring D's true effect in isolation — you're measuring **D's true main effect plus the true ABC three-way interaction, combined, with no way to separate the two using this data alone.** Similarly, "the AB interaction effect" you compute is really (true AB effect) + (true CD effect), combined.

### Verifying This Numerically

**Computing "Effect$_D$" using the D=ABC column against our original Y values:** $D{=}{+}1$ runs (2,3,5,8) average $20.0$; $D{=}{-}1$ runs (1,4,6,7) average $20.0$. $\text{Effect}_D=0.0$.

**Since our original data had no true D effect and no true ABC interaction (both were genuinely zero), this comes out cleanly as zero — but this is only because both aliased quantities happened to be zero.** If a real $D$ effect (or a real $ABC$ interaction) had been present, this same computed number would reflect **some blend of both**, and the design gives you **no way whatsoever to determine how much of it to attribute to each.** This is the honest, unavoidable price of fractionation: **you gain a 4th factor for free in terms of runs, but you lose the ability to cleanly separate D from ABC, and AB from CD, and AC from BD** — permanently, for this specific design.

**Interview question:** *"You run a half-fraction factorial and find a large effect on the column defined as D=ABC. What can you actually conclude?"*
**Ideal answer:** You can conclude that *either* D has a real main effect, *or* the three-way interaction ABC is real, *or* some combination of both — but the design itself cannot distinguish these possibilities, since D and ABC are perfectly aliased (confounded) given the defining relation used to construct this fraction. To resolve the ambiguity, you would need to run a follow-up experiment (e.g., a "fold-over" design that flips the sign of D, which breaks this particular alias) or fall back on prior domain knowledge about which explanation is more plausible (three-way interactions are generally rarer and smaller than main effects, per Chapter 20's discussion, so D's main effect is often the more likely explanation — but this is a judgment call, not something the data alone can settle).

---

## 24.5 Design Resolution: How Bad Is the Confounding?

**Definition: a design's resolution equals the length of the shortest "word" in its defining relation.** Our example's defining relation is $I=ABCD$ — a 4-letter word — making this a **Resolution IV** design.

**Why resolution is the single number practitioners use to judge a fractional design's usefulness:**
- **Resolution III:** main effects are aliased with **two-factor interactions**. Very risky — if any two-factor interaction is real, it directly contaminates a main effect estimate, and you can't tell which is which. Appropriate only for pure screening under a strong assumption that interactions are negligible.
- **Resolution IV** (our example): main effects are aliased only with **three-factor** (and higher) interactions — generally safer, since three-factor interactions are typically smaller and rarer (directly recalling Chapter 20's point about higher-order interactions being both less common and harder to detect). **However, two-factor interactions are aliased with each other** (AB with CD, etc.), so you still can't cleanly separate those.
- **Resolution V:** main effects and two-factor interactions are both completely clear of each other (aliased only with three-factor-or-higher terms) — the safest common practical choice when interactions matter, at the cost of requiring more runs than a Resolution III or IV design for the same number of factors.

**The practical tradeoff, stated plainly:** higher resolution (less problematic confounding) always costs more runs for a given number of factors — there is no free lunch; you're always trading information for economy somewhere along this spectrum.

---

## 24.6 Practical Use: Screening, Then Follow-Up

**The standard industrial workflow this chapter's tools support:** (1) use a **low-resolution fractional factorial** to cheaply screen many candidate factors (accepting that some interactions will be confounded, on the reasonable assumption per Chapter 20 that higher-order interactions are less likely to matter); (2) identify the handful of factors (and any surviving, cleanly-estimated low-order interactions) that appear genuinely important; (3) run a smaller, focused **full factorial** or move to **response surface methodology** (optimizing a continuous response over the most promising factors, found in Kutner's next chapter) on just those few factors, now affordable since you've screened out the rest.

**Direct connection to modern practice:** this is conceptually identical to feature-screening workflows in ML (cheaply identifying which of many candidate features/hyperparameters matter before investing in expensive, fully-crossed tuning), and to multivariate testing (MVT) platforms in industry experimentation — Chapter 20's warning about the difficulty of interpreting higher-order interactions is precisely why most fractional designs deliberately target Resolution IV or V (protecting main effects and often two-factor interactions), rather than trying to resolve every possible high-order interaction from the start.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from itertools import product

# --- Full 2^3 factorial with contrast-based effect estimation ---
runs = list(product([-1,1], repeat=3))
df = pd.DataFrame(runs, columns=['A','B','C'])
df['Y'] = [15.5, 18.5, 16.5, 25.5, 17.5, 20.5, 18.5, 27.5]
# reorder to match standard run order (A,B,C combinations as listed above)
df = pd.DataFrame({
    'A': [-1,1,-1,1,-1,1,-1,1],
    'B': [-1,-1,1,1,-1,-1,1,1],
    'C': [-1,-1,-1,-1,1,1,1,1],
    'Y': [15.5,18.5,16.5,25.5,17.5,20.5,18.5,27.5]
})
df['AB'] = df['A']*df['B']
df['AC'] = df['A']*df['C']
df['BC'] = df['B']*df['C']
df['ABC'] = df['A']*df['B']*df['C']

for col in ['A','B','C','AB','AC','BC','ABC']:
    effect = df.loc[df[col]==1, 'Y'].mean() - df.loc[df[col]==-1, 'Y'].mean()
    print(f"Effect of {col}: {effect:.2f}")

# --- Fractional factorial: add D = ABC (half fraction of 2^4) ---
df['D'] = df['ABC']  # the generator
print("\nHalf-fraction design (D=ABC):")
print(df[['A','B','C','D','Y']])

effect_D = df.loc[df['D']==1, 'Y'].mean() - df.loc[df['D']==-1, 'Y'].mean()
print(f"\nEffect of D (aliased with ABC): {effect_D:.2f}")
# Verify CD column equals AB column (confirming the AB<->CD alias)
df['CD'] = df['C']*df['D']
print("AB and CD columns identical:", (df['AB']==df['CD']).all())
```

---

## Interview Question Bank — Chapter 24

**Conceptual:**
1. Why does a two-level factorial design's run count double with every additional factor, and why does this motivate fractional designs?
2. What does it mean for two effects to be "aliased," in plain terms?
3. What's the practical difference between a Resolution III, IV, and V design?

**Derivation:**
4. Derive the alias structure for a design with defining relation $I=ABCD$, showing why $A$ becomes aliased with $BCD$.
5. Show why the contrast-based effect estimate (average at +1 minus average at −1) is mathematically equivalent to a regression coefficient in a coded ±1 design.

**ML/Statistics:**
6. Why does a Resolution IV design's confounding pattern (main effects clear of 2-way interactions, but 2-way interactions confounded with each other) make it a common practical sweet spot?
7. Connect fractional factorial screening to how you might approach a large hyperparameter or feature screening problem in ML with a limited experimentation budget.
8. Why is it often reasonable to assume three-factor-and-higher interactions are negligible when interpreting an aliased Resolution IV design, and what's the risk in that assumption?

**Coding:**
9. Implement contrast-based effect estimation for a full 2^k factorial from scratch in NumPy/pandas.
10. Construct a half-fraction design by defining a new factor as a product column, and verify the resulting alias structure algebraically and numerically.

**Traps:**
11. "We found a big effect on column D in our fractional design, so D definitely matters." — what's the more precise, careful conclusion?
12. "A Resolution III design is strictly worse than a Resolution V design, so you should always use the highest resolution possible." — what's the practical tradeoff this ignores?
13. Someone runs a fractional factorial and, upon finding a large aliased effect, simply assumes it's due to the main effect rather than the confounded interaction, without further justification. What should they do instead to resolve the ambiguity?

---

*This file covers Kutner Ch. 24 — full two-level factorial designs and the contrast-based effect estimation method (worked completely by hand), fractional factorial designs and the defining relation / alias structure (worked with a concrete D=ABC half-fraction, showing exactly what's lost), design resolution (III/IV/V) as the key summary metric for a fractional design's quality, and the standard screen-then-follow-up industrial workflow, connected to modern ML feature-screening and MVT practice. Chapter 25 (Response Surface Methodology) is the last chapter in this arc if you'd like to continue — optimizing a continuous response over a handful of promising factors identified through screening.*
