# Chapter 6: Lagrange Multipliers & Constrained Optimization — Interview Notes (Beginner-Friendly)

This chapter builds on Chapter 1 (convexity, gradients). Same style as always: plain English first, formulas second, every formula translated back into words.

---

## 1. The Big Picture — What's Different About This Chapter?

Every chapter so far asked: "find the lowest point on a landscape." This chapter asks a harder, more realistic question: **"find the lowest point on a landscape, but you're only allowed to walk along a specific path or stay within a specific region."**

This "only allowed to..." part is called a **constraint**. Examples in real ML: "find the model weights that minimize error, **but the weights must sum to 1**" (like a probability distribution), or "find the decision boundary that maximizes margin, **but every training point must be correctly classified**" (this is literally how SVMs are defined — more on that below).

**The key insight that makes this whole topic click:** *the unconstrained minimum of your landscape might sit somewhere you're not allowed to go. So instead, the best point you're allowed to reach is often not where the landscape is flattest overall — it's where the landscape's downhill pull exactly cancels out against the constraint boundary pushing back.* Everything in this chapter is really just a precise way of finding that balance point.

---

## 2. A Concrete Picture First (Before Any Formulas)

Imagine you're trying to get to the lowest point of a valley, but you're tied to a fence (the constraint) — you can only stand somewhere along that fence line, not anywhere in the valley.

![constrained optimization picture — objective contours and a constraint curve touching tangentially](Lagrange multiplier constrained optimization diagram)

The picture above shows the idea: the rings are the "how good is this point" contours (like a topographic map of the objective function — same idea as Chapter 2's contour plots), and the curved line is the constraint (the fence — the only points you're allowed to be at). The best point along the fence is where a contour ring just barely **touches** the fence line without crossing it — like a rubber band pulled taut, touching the fence at exactly one point.

**Why does "just barely touching" matter?** If a contour ring actually *crosses* the fence at some point, that means you could slide a little further along the fence and land on a *lower* contour ring (a better score) — so that point wasn't the best one yet. The best point along the fence is where sliding in *either* direction along the fence would only make things worse — which happens exactly where the contour ring and the fence are tangent (touching, not crossing).

---

## 3. Refresher: What Does "Tangent" Mean Here, and Why Gradients?

Recall from Chapter 2: **the gradient always points in the direction of steepest increase**, and it's always **perpendicular (at a right angle) to the contour line** passing through that point (this is a property of gradients we haven't needed until now: if you moved *along* a contour line, the function's value wouldn't change at all, since a contour line is defined as "everywhere the value is equal" — so the gradient, which points toward the fastest *change*, has to point directly *away* from the contour line, i.e., perpendicular to it).

**The key geometric fact this chapter relies on:** at the point where the objective's contour line is tangent to the constraint line, **their two gradient directions point along the exact same line** (though possibly in opposite directions, and possibly different lengths) — because "tangent" means the two curves are momentarily running parallel to each other at that point, so the directions perpendicular to each of them (their gradients) must also line up with each other.

This single geometric fact — **the two gradients point along the same line at the optimal constrained point** — is the entire mathematical content of the Lagrange multiplier method. Everything below just turns that picture into an equation you can solve.

---

## 4. Equality Constraints — The Lagrange Multiplier Method

**Setup:** minimize some function $f(x)$ (the thing you care about — e.g., prediction error) subject to a constraint $g(x) = 0$ (the "fence" — e.g., "the weights must sum to exactly 1," rewritten as "weights-sum-minus-1 must equal zero").

**The core equation, in plain English first:** *at the best point, the gradient of your objective and the gradient of your constraint must point along the same line.* We write "point along the same line" mathematically as one gradient being a **scaled copy** of the other:

$$\nabla f(x) = \lambda \cdot \nabla g(x)$$

Translating every symbol:
- $\nabla f(x)$ = the gradient of the thing you're minimizing (which direction would improve your score fastest, ignoring the fence).
- $\nabla g(x)$ = the gradient of the constraint (which direction moves you off the fence fastest).
- $\lambda$ (the Greek letter "lambda," the **Lagrange multiplier**) = just a scaling number that makes the two gradient vectors match up in length as well as direction. It can be positive or negative — its sign and size will fall out of solving the equations, and (bonus, covered more in Chapter 7 with inequality constraints) it also tells you *how sensitive* your best achievable score is to loosening the constraint slightly.

**How you actually solve a problem like this, mechanically:** package $f$, $g$, and $\lambda$ into a single new function called the **Lagrangian**:
$$\mathcal{L}(x, \lambda) = f(x) - \lambda \cdot g(x)$$

Then just take the gradient of $\mathcal{L}$ with respect to *everything* (both $x$ and $\lambda$) and set it all to zero. Setting the gradient with respect to $x$ to zero recovers exactly the "gradients point along the same line" equation above. Setting the gradient with respect to $\lambda$ to zero recovers exactly the original constraint $g(x)=0$ (this is a nice, almost magical trick: differentiating with respect to $\lambda$ just spits the constraint back out, ensuring you never forget to enforce it).

**Why this trick is genuinely useful, not just notational sugar:** it converts a *constrained* problem (which is awkward to search directly) into an *unconstrained* problem (just find where $\mathcal{L}$'s gradient is zero, mechanically identical to everything from Chapters 2–4) — you've folded the fence directly into the function you're optimizing, rather than having to special-case "and also stay on the fence" as a separate rule.

### 4.1 A simple numeric example (do this by hand)

**Problem:** minimize $f(x,y) = x^2+y^2$ (distance from the origin, squared — this is convex, a bowl, from Chapter 1), subject to the constraint $x+y=4$ (you must stay on this line).

Rewrite the constraint as $g(x,y) = x+y-4 = 0$.

**Build the Lagrangian:**
$$\mathcal{L}(x,y,\lambda) = x^2+y^2 - \lambda(x+y-4)$$

**Take derivatives and set to zero:**
- $\frac{\partial \mathcal L}{\partial x} = 2x - \lambda = 0 \implies x = \frac{\lambda}{2}$
- $\frac{\partial \mathcal L}{\partial y} = 2y - \lambda = 0 \implies y = \frac{\lambda}{2}$
- $\frac{\partial \mathcal L}{\partial \lambda} = -(x+y-4) = 0 \implies x+y=4$ (just the constraint again, as promised)

**Solve:** since $x=y=\frac{\lambda}{2}$, substitute into $x+y=4$: $\frac{\lambda}{2}+\frac{\lambda}{2} = 4 \implies \lambda = 4$. So $x = y = 2$.

**Sanity check the picture:** the unconstrained minimum of $x^2+y^2$ is at $(0,0)$ — but $(0,0)$ isn't on the line $x+y=4$, so it's not reachable. The constrained answer $(2,2)$ is the closest point on that line to the origin — which matches simple geometric intuition (the closest point on a line to the origin is where the line from the origin hits it perpendicularly, and $(2,2)$ is indeed the perpendicular foot from the origin onto $x+y=4$). The math and the geometric picture agree, which is a good habit to double-check when solving these by hand.

---

## 5. Inequality Constraints — The KKT Conditions

Equality constraints ("must sum to exactly 1") are one thing, but many real problems have **inequality** constraints instead — "must be less than or equal to some bound," rather than "must equal exactly." Example: "minimize error, subject to every training point being correctly classified **or better**" (an SVM's constraint, covered below) — that's a "greater than or equal to" condition, not an exact equality.

The **KKT conditions** (Karush-Kuhn-Tucker) are the generalization of the Lagrange multiplier idea to handle inequality constraints. There are four conditions, and each one has a simple plain-English meaning:

**Setup:** minimize $f(x)$ subject to $h(x) \le 0$ (an inequality constraint — you must stay on one *side* of a boundary, not necessarily exactly *on* it).

1. **Stationarity:** $\nabla f(x) = \mu \cdot \nabla h(x)$ — same idea as before: at the optimum, the objective's gradient and the constraint's gradient line up (this only really "activates" and matters when the constraint boundary is actually in play — see point 4).

2. **Primal feasibility:** $h(x) \le 0$ — obvious, but worth stating explicitly: your answer actually has to satisfy the constraint. ("Primal" just means "the original problem's variables," as opposed to the $\mu$/$\lambda$ multiplier variables.)

3. **Dual feasibility:** $\mu \ge 0$ — unlike the equality case, the multiplier here is **required to be non-negative**. Plain-language reason: the multiplier represents "how hard the constraint is pushing back against you," and a boundary can only ever push you *away* from violating it, never pull you *toward* violating it — so the direction of that push has a required sign.

4. **Complementary slackness:** $\mu \cdot h(x) = 0$ — this is the cleverest of the four, and the one most worth understanding rather than memorizing. It says: **either the constraint is exactly "tight" (you're sitting right on the boundary, $h(x)=0$), or the multiplier is zero (the constraint isn't actually doing anything at this point, so it might as well not exist).** You can't have both a "loose" constraint (not touching the boundary) *and* a nonzero push from it — that wouldn't make sense; a boundary you're not touching can't be pushing on you at all.

**Plain-language summary of all four together:** *find a point where you can't improve without either leaving the allowed region, or where you're pressed right up against a boundary that's actively pushing back with a valid (non-negative) amount of force — and any boundary you're not touching contributes exactly zero force.*

---

## 6. Worked Example: The SVM Dual (A Favorite Interview Question)

This is one of the most commonly asked derivations in ML interviews, and it's a direct, real application of everything above.

**The SVM primal problem, in plain English:** find a decision boundary (defined by weights $w$ and bias $b$) that (a) has the **largest possible margin** (the widest gap between the boundary and the nearest points of either class — a smaller $\|w\|$ corresponds to a wider margin, so "maximize margin" becomes "minimize $\|w\|^2$"), while (b) **every training point is correctly classified with at least that margin.**

In symbols:
$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^Tx_i+b) \ge 1 \text{ for every training point } i$$

Translating: $\frac{1}{2}\|w\|^2$ is just "minimize the size of $w$" (written with the $\frac12$ and squared purely to make the calculus cleaner later — doesn't change *where* the minimum is). The constraint says "every point, correctly signed by its label $y_i$ (which is $+1$ or $-1$), must land on the correct side of the boundary by a margin of at least 1."

**Build the Lagrangian** (one multiplier $\alpha_i \ge 0$ per training point, since there's one inequality constraint per point):
$$\mathcal{L}(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i\big[y_i(w^Tx_i+b)-1\big]$$

**Apply stationarity** (take derivatives w.r.t. the *primal* variables $w,b$ and set to zero — this is exactly Section 4's method, just with a sum over many constraints instead of one):
- $\frac{\partial \mathcal L}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \implies w = \sum_i \alpha_i y_i x_i$
- $\frac{\partial \mathcal L}{\partial b} = -\sum_i \alpha_i y_i = 0 \implies \sum_i \alpha_i y_i = 0$

**Substitute these back into $\mathcal{L}$** (this is the "dual" trick — eliminate $w$ and $b$ entirely, leaving an optimization problem purely in terms of the $\alpha_i$ multipliers):
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_iy_j (x_i^Tx_j) \quad \text{subject to } \alpha_i \ge 0,\ \sum_i \alpha_i y_i = 0$$

**Why anyone bothers with this "dual" version instead of the original:** two big practical payoffs — (1) the data only ever appears as dot products $x_i^Tx_j$, never as raw individual points, which is exactly what makes the **kernel trick** possible (you can swap in a fancier similarity measure without ever explicitly computing the transformed features), and (2) by **complementary slackness** (Section 5, point 4), $\alpha_i$ can only be nonzero for points sitting *exactly* on the margin boundary — meaning most training points end up with $\alpha_i=0$ and don't matter at all to the final decision boundary. **Those are exactly the "support vectors"** the algorithm is named after — literally the points whose constraint is "tight," in the KKT sense from Section 5.

---

## 7. Quick Summary Table

| Concept | Plain-language meaning |
|---|---|
| Constrained optimization | Minimize something, but only among points that satisfy some restriction (the "fence") |
| Tangency condition | At the best allowed point, the objective's and constraint's gradients point along the same line |
| Lagrange multiplier $\lambda$ | The scaling factor that makes those two gradients match up exactly |
| Lagrangian $\mathcal{L}$ | A trick that folds the constraint directly into the function you optimize, turning a constrained problem into an unconstrained one |
| KKT stationarity | Same gradient-alignment idea, generalized to inequality constraints |
| KKT dual feasibility ($\mu \ge 0$) | A boundary can only push you away from violating it, never pull you toward violating it |
| KKT complementary slackness | Either you're touching the boundary, or that boundary's multiplier is zero — you can't have a "loose" constraint still pushing on you |
| SVM dual | A direct real-world application: only points touching the margin boundary (the support vectors) end up with nonzero multipliers and matter to the final answer |
