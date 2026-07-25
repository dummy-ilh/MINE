# Chapter 7 (Bonus): Second-Order Methods — Newton's Method & L-BFGS — Interview Notes (Beginner-Friendly)

This chapter builds on Chapter 1 (Hessian/curvature) and Chapter 2 (gradient descent). Same style as always: plain English first, formulas second, every formula translated back into words.

---

## 1. The Big Picture — What's "Second-Order" About This?

Every method in Chapters 2–4 (plain GD, momentum, Adam) is called a **first-order method** — it only ever looks at the **gradient** (first derivative — the slope) to decide which way to step. It never looks at **curvature** (second derivative — how the slope itself is changing, from Chapter 1) to decide *how big* a step to take. That step size is instead controlled by the learning rate $\eta$, a number you have to pick and tune by hand.

**The core idea of this chapter, in one sentence:** *if you're willing to compute curvature information (the Hessian) as well as the gradient, you can figure out mathematically exactly how big a step to take — no manually-tuned learning rate needed — and often reach the minimum in far fewer steps.*

---

## 2. Building the Intuition: Why Would Curvature Tell You Step Size?

Recall the plain gradient descent numeric example from Chapter 2: on the bowl $f(x)=x^2$, starting at $x=6$ with $\eta=0.1$, it took many small steps ($6 \to 4.8 \to 3.84 \to 3.072 \to \dots$), each one just a fraction of the way to the true minimum at $0$.

Here's the frustrating part: for this simple bowl, we *know* the curvature is constant ($f''(x)=2$ everywhere, from Chapter 1). If we knew how to use that number, we could in principle jump straight to the minimum in **one step**, instead of crawling there gradually. Newton's method is exactly this idea, made precise and generalized to any (well-behaved) function.

**The intuition:** gradient descent only asks "which way is downhill?" and takes a small step. Newton's method asks a much more ambitious question: **"if I pretend the function is a perfect bowl shape locally (matching both the current slope AND the current curvature), where exactly is the bottom of *that* bowl?"** — and then jumps straight there in one move.

---

## 3. Newton's Method — The Update Rule

**In 1 dimension**, in plain English: *use the current slope and the current curvature together to directly estimate where the flat bottom is, and jump there.*

$$x_{new} = x_{old} - \frac{f'(x_{old})}{f''(x_{old})}$$

Translating: instead of subtracting "learning rate times gradient" (Chapter 2's rule), you subtract "gradient divided by curvature." **Dividing by the curvature is doing the job the learning rate used to do** — but now it's computed exactly from the function's own shape, rather than guessed by you. Steep curvature (a narrow bowl) automatically produces a smaller step; gentle curvature (a wide, shallow bowl) automatically produces a bigger step — precisely the adaptive behavior we had to hand-tune with learning rate schedules in Chapter 2.

### 3.1 A numeric example — same bowl as Chapter 2, but with Newton's method

$f(x)=x^2$, so $f'(x)=2x$ and $f''(x)=2$ (a constant, as noted above). Start at $x=6$:

$$x_{new} = 6 - \frac{f'(6)}{f''(6)} = 6 - \frac{12}{2} = 6 - 6 = 0$$

**One step. Done.** We land exactly on the true minimum, immediately — compare this to gradient descent's slow crawl ($6\to4.8\to3.84\to\dots$) over many steps in Chapter 2. This is the entire appeal of Newton's method: on a function that's genuinely bowl-shaped (or well-approximated as one locally), it can converge dramatically faster.

**Why this worked in exactly one step here:** $f(x)=x^2$ *is* a perfect bowl everywhere — Newton's method's "pretend it's a perfect bowl locally" assumption was exactly true, not just an approximation, so the one-step jump landed exactly right. On more complicated functions, the local "pretend it's a bowl" approximation is only approximately true at any single point, so Newton's method takes more than one step — but usually still far fewer than gradient descent, because it re-estimates a fresh, better bowl-approximation at every step.

### 3.2 The Multi-Dimensional Version

With many variables (real model weights), "divide by curvature" becomes "multiply by the **inverse of the Hessian matrix**" (matrix inversion is the multi-dimensional equivalent of division — from Chapter 1, recall the Hessian is just a table of curvature numbers in every direction):

$$w_{new} = w_{old} - [\nabla^2 f(w_{old})]^{-1} \nabla f(w_{old})$$

Translating: $\nabla^2 f(w_{old})$ is the Hessian (curvature table) at your current position, from Chapter 1. The $^{-1}$ means "matrix inverse" — conceptually the same idea as "divide by," just generalized to work with a whole table of numbers instead of a single one. $\nabla f(w_{old})$ is the ordinary gradient, same as always.

---

## 4. Why Don't We Just Always Use Newton's Method, Then?

If it's so much faster, why did Chapters 2–4 spend so much time on gradient descent, momentum, and Adam instead? Two big, very practical reasons:

**Reason 1 — Computing and storing the Hessian is expensive.** If you have $n$ weights, the Hessian is an $n \times n$ table of numbers (every pair of directions needs its own curvature entry — recall the cross-terms from Chapter 1, Section 3.1). A modern neural network can easily have **billions** of weights. A Hessian for a billion weights would need roughly $(10^9)^2 = 10^{18}$ numbers stored — completely infeasible, both to compute and to store in memory. Gradient-only methods only ever need $n$ numbers (one gradient value per weight), which is why they're the only realistic option at large scale.

**Reason 2 — Inverting that huge matrix is also expensive.** Even if you could store the Hessian, computing its inverse (needed for the update rule above) is computationally very costly for large matrices — the cost grows roughly with the *cube* of the number of weights, which becomes hopeless very quickly as models grow.

**Reason 3 — Newton's method doesn't handle non-convexity gracefully.** Recall Chapter 5: near a saddle point, the Hessian has *mixed* positive and negative curvature. Newton's method, which blindly trusts the local "pretend it's a bowl" approximation, can actually get **actively pulled toward** a saddle point in this situation, rather than away from it — the "jump straight to where the local bowl-approximation says the bottom is" logic backfires badly when the local shape isn't really a bowl at all.

**The honest one-line interview summary:** *"Newton's method converges in far fewer steps when it works, because it uses real curvature information instead of a guessed learning rate — but computing and inverting the full Hessian is computationally infeasible at the scale of modern neural networks, and it can behave badly near saddle points, so it's essentially never used directly for deep learning."*

---

## 5. Quasi-Newton Methods & L-BFGS — Getting Some of the Benefit More Cheaply

**The idea, in plain English:** *what if, instead of exactly computing the full, expensive Hessian at every step, we cleverly estimated an approximate version of it, built up gradually from the gradients we were computing anyway?*

This family of methods is called **quasi-Newton** methods ("quasi" = "sort of, approximately"). The most well-known one is **L-BFGS** ("Limited-memory BFGS" — BFGS being the names of the four mathematicians who devised the original, more memory-hungry version).

**How L-BFGS builds its approximate curvature information:** every time you take a step, you naturally learn two things "for free" — how much your position changed, and how much your gradient changed. Comparing those two things across recent steps tells you something about the local curvature, **without ever computing the full Hessian directly.** L-BFGS keeps track of just a handful of the most recent such comparisons (that's the "limited-memory" part — it deliberately only remembers, say, the last 10–20 steps' worth of information, rather than the full exact Hessian) and uses that limited history to build a good-enough curvature estimate on the fly.

**Why "limited memory" matters:** storing the full history needed for the *original*, non-limited BFGS method still scales badly with the number of weights ($n^2$ storage, same problem as the true Hessian). L-BFGS's trick of keeping only a small fixed window of recent history keeps the memory cost down to roughly $n$ times a small constant — genuinely usable at reasonably large scale, though still generally reserved for small-to-medium-sized problems, not today's largest neural networks.

---

## 6. Quick Comparison Table

| Method | Uses curvature info? | Cost per step | Learning rate needed? | Typical use case |
|---|---|---|---|---|
| Plain GD / momentum / Adam (Ch. 2–4) | No (first-order only) | Cheap — proportional to $n$ | Yes, must be tuned | Deep learning at any scale |
| Newton's method | Yes — full, exact Hessian | Very expensive — roughly $n^3$ (inverting an $n\times n$ matrix) | No — step size derived from curvature | Small problems, or as a theoretical benchmark |
| L-BFGS (quasi-Newton) | Yes — cheap approximate estimate | Moderate — roughly $n$ times a small constant | Mostly no (though often paired with a simple line search) | Medium-sized problems (classical ML, smaller models); rare in large-scale deep learning |

---

## 7. Common Interview Follow-Ups

**"Why don't we use Newton's method to train neural nets?"** (A very common question, and the direct answer to Section 4 above.) Two reasons dominate: the Hessian is far too large to store or invert at the scale of modern models (billions of weights), and it can behave badly (getting pulled toward) saddle points, which Chapter 5 established are the dominant obstacle in high-dimensional non-convex landscapes — exactly the setting Newton's method is worst-suited for.

**"If Newton's method has trouble with saddle points, is there a fix?"** Yes — a known variant called the **saddle-free Newton method** deliberately flips the sign of any negative curvature directions before taking the step (so it always moves *away* from downward-curving directions, rather than blindly trusting the raw Hessian). This is more of a research-literature answer than a widely-deployed practical technique, worth mentioning briefly if the conversation goes deep on saddle points, but not something to over-claim as standard practice.

**"Does Adam (Chapter 4) use any curvature information?"** Not exactly, but it's a fair connection to draw: Adam's per-weight scaling by the running average of *squared gradients* is a much cheaper, rougher proxy for "how much does this weight typically move," which behaves somewhat similarly in spirit to using curvature to set step size — without ever computing an actual second derivative. Good answer if an interviewer asks you to connect Chapter 4 and Chapter 7's ideas.

---

## 8. Quick Summary Table

| Concept | Plain-language meaning |
|---|---|
| First-order methods | Only use the gradient (slope); step size set by a manually-tuned learning rate |
| Second-order methods | Also use curvature (the Hessian) to mathematically derive the step size directly |
| Newton's method | Pretend the function is a perfect bowl locally (matching slope and curvature), jump straight to that bowl's bottom |
| Why it's fast when it works | No guessing the learning rate — the step size comes directly from the function's own shape |
| Why it's rarely used in deep learning | Storing/inverting the full Hessian is computationally infeasible at billions of weights; can misbehave near saddle points |
| Quasi-Newton / L-BFGS | Build a cheap, approximate curvature estimate from recent gradient/position changes, instead of computing the true Hessian |
| "Limited memory" in L-BFGS | Only keep a small fixed window of recent history, instead of the full expensive history BFGS would otherwise need |
