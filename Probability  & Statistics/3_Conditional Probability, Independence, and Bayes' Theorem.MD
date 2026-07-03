# Conditional Probability — Complete Interview Guide (L5 Google/Apple)

**Reference texts** (cite these if asked "where did you learn this" or want deeper study):
- Sheldon Ross, *A First Course in Probability* (9th/10th ed.) — Ch. 3
- Blitzstein & Hwang, *Introduction to Probability* (Harvard Stat 110) — Ch. 2
- DeGroot & Schervish, *Probability and Statistics* (4th ed.) — Ch. 2
- Casella & Berger, *Statistical Inference* — Ch. 1 (more rigorous, measure-flavored)

---

## 1. Foundations

### 1.1 Definition
For events $A$, $B$ with $P(B) > 0$:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Intuition (Ross's framing):** $B$ becomes the new sample space. You're asking "of the outcomes where $B$ happened, what fraction also had $A$ happen?"

This is a *definition*, not a derived fact — everything else (Bayes, chain rule, total probability) follows from it.


"How does the probability of an event change when we have extra information?".


### 1.2 Multiplication Rule (Chain Rule)
Rearranging the definition:

$$P(A \cap B) = P(A\mid B)\,P(B) = P(B \mid A)\,P(A)$$

Generalizes to $n$ events:
$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1)\,P(A_2\mid A_1)\,P(A_3 \mid A_1 A_2) \cdots P(A_n \mid A_1 \cdots A_{n-1})$$

This is the workhorse for sequential/"without replacement" problems.

### 1.3 Law of Total Probability (LOTP)
If $B_1, \dots, B_n$ partition the sample space ($\bigcup B_i = \Omega$, mutually exclusive, $P(B_i)>0$):

$$P(A) = \sum_{i=1}^n P(A \mid B_i)\,P(B_i)$$
[
**This is how you compute an "unconditional" probability when you naturally know things conditionally.** Extremely common in interviews — anytime a problem says "there are two types of X" or "with probability p it's a fair coin, else biased," LOTP is your first move.
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-f89ImFsaysYE-AhYZWRjYMQ1Q9ulWDzmC_KA5q_tow&s=10)
### 1.4 Bayes' Theorem
$$P(B_i \mid A) = \frac{P(A \mid B_i)\,P(B_i)}{\sum_j P(A \mid B_j)\,P(B_j)} = \frac{P(A\mid B_i)P(B_i)}{P(A)}$$

Vocabulary you should use fluently in an interview:
- $P(B_i)$ = **prior**
- $P(A \mid B_i)$ = **likelihood**
- $P(B_i \mid A)$ = **posterior**
- $P(A)$ = **evidence / normalizing constant** (computed via LOTP)

**Interview tip:** When an interviewer asks a Bayes question, say the words "prior," "likelihood," "posterior" out loud. It signals fluency and helps you organize the algebra before you touch numbers.

---

## 2. Independence

$A$ and $B$ are independent iff:
$$P(A \cap B) = P(A)P(B) \iff P(A\mid B) = P(A) \iff P(B \mid A) = P(B)$$

**Common trap:** Independence ≠ mutually exclusive. In fact, if $P(A), P(B) > 0$ and $A, B$ are mutually exclusive, they are *maximally dependent* (knowing one happened tells you the other definitely didn't).

### Conditional Independence
$A \perp B \mid C$ means $P(A \cap B \mid C) = P(A\mid C)P(B \mid C)$.

**Critical point:** $A \perp B$ does **not** imply $A \perp B \mid C$, and vice versa. This is the mathematical seed of **Simpson's Paradox** (Section 6) — a favorite "gotcha" at L5 interviews because it tests whether you actually understand conditioning or just plug into formulas.

---

## 3. Worked Numerical Examples

### Example 1 — Basic conditional probability (2 dice)
Roll two fair dice. Let $A$ = "sum is 8", $B$ = "first die shows 3". Find $P(A \mid B)$.

$P(B) = 6/36$ (first die = 3, second die anything).
$P(A \cap B)$: first die = 3 AND sum = 8 → second die = 5. Exactly 1 outcome out of 36.

$$P(A\mid B) = \frac{1/36}{6/36} = \frac{1}{6}$$

Sanity check: given first die is 3, the second die is uniform on {1..6}, and we need it to be 5 → probability 1/6. Matches.

---

### Example 2 — Classic disease testing (Bayes)
A disease has prevalence 1%. A test has 99% sensitivity ($P(\text{+}\mid \text{disease})=0.99$) and 95% specificity ($P(-\mid\text{no disease})=0.95$, so false positive rate = 5%). A random person tests positive. What's $P(\text{disease} \mid +)$?

**Setup with LOTP + Bayes:**
- Prior: $P(D) = 0.01$, $P(D^c) = 0.99$
- Likelihood: $P(+\mid D) = 0.99$, $P(+\mid D^c) = 0.05$

$$P(+) = P(+\mid D)P(D) + P(+\mid D^c)P(D^c) = 0.99(0.01) + 0.05(0.99) = 0.0099 + 0.0495 = 0.0594$$

$$P(D\mid +) = \frac{0.0099}{0.0594} \approx 0.1667 \;(\approx 16.7\%)$$

**Interview point:** This is the base-rate fallacy demo. Most candidates guess ~90%+. The correct intuition: because the disease is rare, false positives from the huge healthy population outnumber true positives even with a "good" test. Always say this intuition out loud — it's exactly what interviewers are probing for.

---

### Example 3 — Monty Hall (conditional probability under selection)
3 doors, 1 car, 2 goats. You pick door 1. Host (who knows where the car is) opens a door with a goat, always different from your pick and never revealing the car. Should you switch?

**Bayesian solution.** Let $C_i$ = car behind door $i$. WLOG you pick door 1, host opens door 3.

$$P(C_1) = P(C_2) = P(C_3) = 1/3$$

We want $P(C_1 \mid \text{host opens 3})$ vs $P(C_2 \mid \text{host opens 3})$.

- If car is at 1: host picks randomly between doors 2,3 → $P(\text{opens }3\mid C_1) = 1/2$
- If car is at 2: host is forced to open 3 → $P(\text{opens }3\mid C_2) = 1$
- If car is at 3: host can't open 3 → $P(\text{opens }3\mid C_3) = 0$

By Bayes:
$$P(C_1\mid \text{opens }3) = \frac{(1/2)(1/3)}{(1/2)(1/3)+(1)(1/3)+0} = \frac{1/6}{1/2} = \frac{1}{3}$$
$$P(C_2 \mid \text{opens }3) = \frac{(1)(1/3)}{1/2} = \frac{2}{3}$$

**Switching wins 2/3 of the time.** The key subtlety (interviewers love probing this): the host's *behavior/constraint* (he must open a goat door, never your door) is what breaks symmetry. If the host opened doors randomly and *happened* to reveal a goat, the answer changes to 1/2 — because then the fact he revealed a goat is informative differently. Being able to articulate this distinction is what separates L5-level understanding from rote memorization.

---

### Example 4 — Sequential draws without replacement (multiplication rule)
An urn has 5 red, 3 blue balls. Draw 3 without replacement. What's $P(\text{R, B, R in that order})$?

$$P(R_1)\,P(B_2\mid R_1)\,P(R_3 \mid R_1 B_2) = \frac{5}{8}\cdot\frac{3}{7}\cdot\frac{4}{6} = \frac{5\cdot3\cdot4}{8\cdot7\cdot6} = \frac{60}{336} = \frac{5}{28} \approx 0.1786$$

---

### Example 5 — Two-child problem (classic conditioning subtlety)
A family has 2 children. 

**(a)** Given at least one is a boy, what's $P(\text{both boys})$?
Sample space (equally likely): {BB, BG, GB, GG}. Condition on "at least one boy" → {BB, BG, GB}.
$$P(\text{BB} \mid \text{at least one boy}) = 1/3$$

**(b)** Given the **older** child is a boy, what's $P(\text{both boys})$?
Condition on older=boy → {BB, BG}.
$$P(\text{BB}\mid \text{older is boy}) = 1/2$$

**Why the interviewer asks this:** it demonstrates that *how* information arrives (which conditioning event, precisely) changes the answer — "at least one boy" is a weaker, differently-structured condition than "the specific older child is a boy." Precision about the conditioning event is the whole game in conditional probability.

---

### Example 6 — A/B testing style problem (common at Google/Apple for applied roles)
You have two ad variants. Variant A converts at 5%, Variant B at 7%. A user is randomly shown A or B with equal probability (50/50), and converts. What's the probability they were shown variant B?

$$P(B\mid \text{convert}) = \frac{P(\text{conv}\mid B)P(B)}{P(\text{conv}\mid A)P(A) + P(\text{conv}\mid B)P(B)} = \frac{0.07 \times 0.5}{0.05\times0.5 + 0.07\times0.5} = \frac{0.035}{0.06} = 0.5833$$

This exact structure (attribution / "which source produced this observed event") shows up constantly in applied-ML and analytics interviews — spam classification, fraud detection, funnel attribution.

---

### Example 7 — Naive Bayes flavor (spam filter), ties to ML interviews
Word "free" appears in 40% of spam emails and 1% of non-spam. 20% of all email is spam. An email contains "free." $P(\text{spam}\mid\text{"free"})$?

$$P(\text{free}) = 0.4(0.2) + 0.01(0.8) = 0.08 + 0.008 = 0.088$$
$$P(\text{spam}\mid\text{free}) = \frac{0.08}{0.088} \approx 0.909$$

This is literally the mechanism behind Naive Bayes classifiers — worth mentioning if the interview drifts toward ML fundamentals, since it shows you can connect probability theory to the models you'd actually build.

---

## 4. Odds Form of Bayes (fast mental math trick)

Instead of full normalization, use **odds**:
$$\frac{P(D\mid +)}{P(D^c\mid +)} = \frac{P(+\mid D)}{P(+\mid D^c)} \times \frac{P(D)}{P(D^c)}$$

Redo Example 2: prior odds $= 0.01/0.99 \approx 1/99$. Likelihood ratio $= 0.99/0.05 = 19.8$.
Posterior odds $\approx 19.8/99 = 0.2$. So $P(D\mid+) = 0.2/1.2 = 1/6 \approx 16.7\%$. ✓ matches.

**Why this matters for interviews:** doing Bayes via odds is faster under time pressure and signals you know multiple ways to manipulate the same identity — good for whiteboard settings where mental arithmetic speed matters.

---

## 5. Common Pitfalls / "Gotchas" Interviewers Test

1. **Confusing $P(A\mid B)$ with $P(B\mid A)$** — the prosecutor's fallacy. E.g., "probability of the evidence given innocence" is not "probability of innocence given the evidence." This is exactly the base-rate structure in Example 2.

2. **Forgetting the base rate** — treating a high sensitivity/specificity test as if it directly gives you the posterior.

3. **Simpson's Paradox** — a treatment can look better than another *overall*, but worse in *every subgroup*, when subgroup sizes differ. Classic example (Ross, and also DeGroot): UC Berkeley admissions, aggregated across departments, appeared to favor men; but within nearly every individual department, women had equal or higher admit rates. The lurking variable was which departments each gender applied to (women disproportionately applied to more competitive departments).
   Formally: it's possible that $P(A\mid B, C) < P(A\mid B^c, C)$ and $P(A\mid B, C^c) < P(A\mid B^c, C^c)$ but $P(A\mid B) > P(A\mid B^c)$.
   **Interview signal:** always ask "conditioned on what?" before trusting an aggregate comparison.

4. **Assuming independence without justification** — always verify $P(A\cap B) = P(A)P(B)$ rather than assuming it because events "feel" unrelated.

5. **Order of conditioning in sequential draws** — with/without replacement changes everything (Example 4).

6. **Ignoring how the conditioning event was generated** (Monty Hall) — the *mechanism* generating your observation matters, not just the raw fact observed.

---

## 6. How L5 Interviews Usually Frame These Questions

- **Data Science / Analytics (Google, Apple):** funnel attribution, A/B test interpretation, fraud/anomaly detection base-rate questions, "what's wrong with this metric" (often a Simpson's Paradox trap).
- **ML/Applied Science:** Naive Bayes derivations, calibration of classifiers (precision vs. recall framed as conditional probabilities: $P(\text{true}\mid\text{predicted true})$ = precision = a posterior, essentially a Bayes problem), confusion matrix reasoning.
- **General SWE/Systems (Google especially):** classic brainteaser style — Monty Hall variants, dice/card problems, "two children" style traps, expected-value-under-conditioning questions.

**Meta-tip:** In all these interviews, verbalize your setup before computing: define events precisely, state what's given (prior/likelihood), then write the formula symbolically before plugging numbers. Interviewers are grading your *process*, not just the final number.

---

## 7. Practice Problems (try these — answers below the line)

1. A bag has 2 fair coins and 1 two-headed coin. You pick one at random and flip it twice; both flips are heads. What's the probability you picked the two-headed coin?
2. In a class, 60% of students are in engineering, 40% in liberal arts. 25% of engineering students and 10% of liberal arts students take a specific stats elective. If a random student is in the elective, what's the probability they're an engineering student?
3. Three independent alarms each detect an intrusion with probability 0.9. Given at least one alarm triggered, what's the probability all three triggered? (Assume intrusion actually happened, alarms independent.)

<details>
<summary>Answers (click to expand mentally / scroll)</summary>

**1.** $P(\text{2-head}) = 1/3$. $P(HH\mid \text{fair}) = 1/4$, $P(HH\mid\text{2-head})=1$.
$P(HH) = 1/3(1) + 2/3(1/4) = 1/3+1/6 = 1/2$.
$P(\text{2-head}\mid HH) = \frac{1/3}{1/2} = 2/3$.

**2.** $P(\text{elective}) = 0.6(0.25)+0.4(0.10) = 0.15+0.04=0.19$.
$P(\text{eng}\mid\text{elective}) = 0.15/0.19 \approx 0.789$.

**3.** $P(\ge 1) = 1-0.1^3 = 0.999$. $P(\text{all 3}) = 0.9^3=0.729$.
$P(\text{all 3}\mid \ge1) = 0.729/0.999 \approx 0.7297$.

</details>

---

## 8. One-Page Formula Cheat Sheet

| Concept | Formula |
|---|---|
| Definition | $P(A\mid B) = P(A\cap B)/P(B)$ |
| Multiplication | $P(A\cap B) = P(A\mid B)P(B)$ |
| Chain rule ($n$ events) | $P(A_1\cdots A_n)=\prod_i P(A_i\mid A_1\cdots A_{i-1})$ |
| Total probability | $P(A)=\sum_i P(A\mid B_i)P(B_i)$ |
| Bayes | $P(B_i\mid A)=\dfrac{P(A\mid B_i)P(B_i)}{\sum_j P(A\mid B_j)P(B_j)}$ |
| Odds-form Bayes | posterior odds = likelihood ratio × prior odds |
| Independence | $P(A\cap B)=P(A)P(B)$ |
| Conditional independence | $P(A\cap B\mid C)=P(A\mid C)P(B\mid C)$ |

Study path suggestion: Ross Ch. 3 problems (especially the "self-test problems" at chapter end) are the closest in flavor to what shows up in tech interviews. Blitzstein's Stat 110 lecture notes/problem sets (freely available, Harvard) are excellent for the Bayes/LOTP intuition-building — his "Bayes' billiards" and story-proof style is very close to interview reasoning.

---

## 9. Additional Concepts & Examples (from Blitzstein & Hwang, *Introduction to Probability*, Stat 110)

These supplement everything above — nothing above is removed, this extends the guide with material you flagged as useful.

### 9.1 Two Cards (multiplication rule + naive definition, symmetry argument)

A standard deck is shuffled. Two cards are drawn without replacement. $A$ = first card is a heart, $B$ = second card is red. Find $P(A\mid B)$ and $P(B\mid A)$.

$$P(A\cap B) = \frac{13}{52}\cdot\frac{25}{51} = \frac{25}{204}$$

($13$ ways to pick a heart first, then $25$ remaining red cards for the second.)

$P(A) = 1/4$ (suits equally likely).

$$P(B) = \frac{26}{52}\cdot\frac{51}{51} = \frac12$$

**Symmetry shortcut for $P(B)$:** before doing the experiment, the second card is equally likely to be *any* card in the deck — so $P(B) = 26/52 = 1/2$ directly, no need to condition on the first draw.

$$P(A\mid B) = \frac{25/204}{1/2} = \frac{25}{102}, \qquad P(B\mid A) = \frac{25/204}{1/4} = \frac{25}{51}$$

**Key lesson:** $P(A\mid B) \ne P(B\mid A)$ — confusing the two is the **prosecutor's fallacy**. Always be deliberate about which event is "given" and which is being updated.

### 9.2 Frequentist Interpretation of Conditioning
Imagine repeating the experiment $n$ times. Let $n_A, n_B, n_{AB}$ be counts of $A$, $B$, $A\cap B$ across trials. Then

$$P(A\mid B) \approx \frac{n_{AB}/n}{n_B/n} = \frac{n_{AB}}{n_B}$$

i.e., **circle all trials where $B$ happened, then ask what fraction of those circled trials also had $A$**. This is the cleanest mental model to fall back on if you freeze on a formula mid-interview — just narrate this instead.

### 9.3 The Two-Children Problem, done fully rigorously (Gardner's original phrasing)

Assumptions needed (state these explicitly in an interview — it shows rigor): gender is binary, $P(\text{boy})=P(\text{girl})=1/2$ for each child, and the two children's genders are independent.

**(a)** "The older child is a girl." Find $P(\text{both girls})$.
$$P(\text{both girls}\mid \text{older is girl}) = \frac{1/4}{1/2} = \frac12$$

**(b)** "At least one is a girl." Find $P(\text{both girls})$.
$$P(\text{both girls}\mid \text{at least one girl}) = \frac{1/4}{3/4} = \frac13$$

**Why these differ (the intuition interviewers want):** conditioning on "older is a girl" designates a *specific* child, leaving the other child's gender genuinely 50/50 — it knocks 2 of the 4 equally-likely outcomes {GG, GB, BG, BB} out (keeps GG, GB). Conditioning on "at least one girl" does **not** designate a specific child — it only removes BB, leaving 3 outcomes {GG, GB, BG}, of which only 1 is "both girls." By symmetry, $P(\text{both girls}\mid\text{younger is girl}) = 1/2$ as well — but there's no such symmetry between "older is a girl" and "at least one is a girl," because those are structurally different conditioning events.

### 9.4 Random Child Is a Girl (sampling mechanism matters)

You run into **one** of the two children at random (equally likely either child), and she's a girl. $P(\text{both girls})$?

Let $G_1, G_2, G_3$ = elder is girl, younger is girl, the randomly-encountered child is a girl. $P(G_1)=P(G_2)=P(G_3)=1/2$, $P(G_1\cap G_2) = 1/4$.

$$P(G_1\cap G_2 \mid G_3) = \frac{P(G_1\cap G_2\cap G_3)}{P(G_3)} = \frac{P(G_1\cap G_2)}{P(G_3)} = \frac{1/4}{1/2} = \frac12$$

(since $G_1 \cap G_2 \subseteq G_3$: if both children are girls, the randomly-met child is certainly a girl).

**Critical takeaway — this is a statistics interview favorite:** the answer here (1/2) differs from the "at least one girl" answer (1/3) even though both start with "we know one child is a girl." The difference is entirely about **how the information was sampled**. "At least one is a girl" is a fact about the *pair*; "the specific child I randomly encountered is a girl" designates one child, like the "older is a girl" case. **Always ask "how was this observation generated?" before conditioning** — this is the single most important meta-skill in applied conditional-probability reasoning (funnel data, survivorship bias, selection bias all reduce to this).

### 9.5 A Girl Born in Winter (extreme illustration of "how specific is the conditioning event?")

Family has 2 children. Given **at least one is a girl born in winter** (seasons equally likely, independent of gender), find $P(\text{both girls})$.

$P(\text{a specific child is a winter girl}) = 1/2 \times 1/4 = 1/8$.

Denominator: $P(\text{at least one winter girl}) = 1-(7/8)^2 = 15/64$.

Numerator: "both girls AND at least one winter girl" = "both girls AND at least one winter child" (since if both are girls, "winter child" = "winter girl"):
$$P(\text{both girls}, \ge 1 \text{ winter child}) = \frac14\left(1-\left(\frac34\right)^2\right) = \frac14\cdot\frac{7}{16} = \frac{7}{64}$$

$$P(\text{both girls}\mid \ge1\text{ winter girl}) = \frac{7/64}{15/64} = \frac{7}{15} \approx 0.4667$$

**This sits between 1/3 (the "at least one girl" example) and 1/2 (the "specific child is a girl" example).** Adding the seemingly-irrelevant "born in winter" detail makes the conditioning event closer to identifying a *specific* child, nudging the answer toward 1/2. Push this to the limit — "at least one is a girl born on March 31 at 8:20pm" — and the answer approaches 1/2 exactly, because that description essentially pins down one particular child. **This is the cleanest possible demonstration that the informativeness/specificity of the conditioning event, not just its truth value, determines the posterior.**

### 9.6 Odds — Formal Definition and Odds-Form Bayes (restated precisely)

$$\text{odds}(A) = \frac{P(A)}{P(A^c)}, \qquad P(A) = \frac{\text{odds}(A)}{1+\text{odds}(A)}$$

**Odds form of Bayes' rule:**
$$\frac{P(A\mid B)}{P(A^c\mid B)} = \frac{P(B\mid A)}{P(B\mid A^c)}\cdot\frac{P(A)}{P(A^c)}$$

i.e. **posterior odds = likelihood ratio × prior odds.** (This matches Section 4 above — here is the fully formal statement with proof-ready notation.)

### 9.7 Random Coin (fair vs. biased) — full Bayes + LOTP with extra conditioning

One fair coin, one biased coin ($P(H)=3/4$). Pick one at random, flip 3 times, get HHH. Find $P(\text{fair}\mid HHH)$.

Let $F$ = fair coin chosen, $A$ = three Heads observed.
$$P(F\mid A) = \frac{P(A\mid F)P(F)}{P(A\mid F)P(F) + P(A\mid F^c)P(F^c)} = \frac{(1/2)^3(1/2)}{(1/2)^3(1/2) + (3/4)^3(1/2)} \approx 0.23$$

**Follow-up — LOTP with extra conditioning (this pattern shows up constantly in sequential-prediction interview questions):** given HHH already observed, what's $P(\text{4th flip is Heads})$?

We don't know which coin we have, but conditioning on $F$ makes it easy:
$$P(H_4\mid A) = P(H_4\mid F,A)P(F\mid A) + P(H_4\mid F^c,A)P(F^c\mid A) = \tfrac12(0.23) + \tfrac34(0.77) \approx 0.69$$

**Pattern to remember:** whenever direct computation of $P(\text{event}\mid \text{data})$ is hard, but computation is easy *conditional on some latent variable* (which coin, which population, which regime), condition on that latent variable, compute both pieces, then weight by the *posterior* probability of the latent variable (not the prior!) using LOTP with extra conditioning:
$$P(X\mid \text{data}) = \sum_i P(X \mid Z=i, \text{data})\, P(Z=i \mid \text{data})$$

### 9.8 "Conditionitis" — Disease Testing with Symmetric 95%/95% Accuracy

(A cleaner numeric variant of Example 2 above — same phenomenon, different exact numbers, worth having both in your back pocket.) Prevalence 1%, $P(T\mid D) = 0.95$ (sensitivity), $P(T^c\mid D^c)=0.95$ (specificity), so $P(T\mid D^c) = 0.05$.

$$P(D\mid T) = \frac{0.95(0.01)}{0.95(0.01) + 0.05(0.99)} = \frac{0.0095}{0.0095+0.0495} \approx 0.16$$

**Intuition via a population of 10,000 (great way to explain this on a whiteboard without any algebra):** 100 people have the disease → 95 test positive, 5 negative. 9,900 don't have it → 495 false-positives, 9,405 true-negatives. So among all positives ($95+495=590$), only $95/590 \approx 16\%$ actually have the disease. **This "natural frequencies" framing (Gigerenzer & Hoffrage) is often the fastest way to convince a non-technical interviewer/stakeholder — worth having ready as a plain-English explanation, not just the formula.**

### 9.9 Independence — Full Formal Treatment

**Two events:** $A\perp B \iff P(A\cap B)=P(A)P(B) \iff P(A\mid B)=P(A) \iff P(B\mid A)=P(B)$ (given positive probabilities). Independence is symmetric.

**Disjointness vs. independence (restated formally):** if $A,B$ disjoint with $P(A),P(B)>0$, then $P(A\cap B)=0 \ne P(A)P(B)$, so disjoint events with positive probability are **never** independent — knowing $A$ happened tells you for certain $B$ did not.

**Proposition:** if $A\perp B$, then $A\perp B^c$, $A^c\perp B$, and $A^c\perp B^c$ also hold. (Proof: $P(B^c\mid A) = 1-P(B\mid A) = 1-P(B) = P(B^c)$ when $P(A)>0$; independence "propagates" to complements.)

**Three events — independence requires ALL FOUR conditions:**
$$P(A\cap B)=P(A)P(B),\;\; P(A\cap C)=P(A)P(C),\;\; P(B\cap C)=P(B)P(C),\;\; P(A\cap B\cap C)=P(A)P(B)P(C)$$
If only the first three hold, that's called **pairwise independence** — a strictly weaker condition.

**Pairwise independence does NOT imply full independence — classic counterexample (memorize this one, it's a favorite theoretical-depth question):**
Two independent fair coin tosses. $A$ = first is Heads, $B$ = second is Heads, $C$ = both tosses match. Check: $P(A)=P(B)=P(C)=1/2$. $P(A\cap B) = 1/4 = P(A)P(B)$ ✓. $P(A\cap C)$: need first=H and both match → HH only → $1/4 = P(A)P(C)$ ✓. Similarly $B,C$ pairwise independent ✓. But
$$P(A\cap B\cap C) = P(HH) = \frac14 \ne P(A)P(B)P(C) = \frac18$$
So $A,B,C$ are **pairwise independent but not mutually independent** — knowing $A$ and $B$ together fully determines $C$, even though knowing just $A$ or just $B$ alone tells you nothing about $C$.

*(Converse also fails: $P(A\cap B\cap C)=P(A)P(B)P(C)$ does not imply pairwise independence — trivially true if any single event has probability 0.)*

### 9.10 Conditional Independence — Where Most People Get Tripped Up

**Definition:** $A,B$ conditionally independent given $E$ iff $P(A\cap B\mid E) = P(A\mid E)P(B\mid E)$.

**Three distinct failure modes to keep straight (all real, all interview-relevant):**

**(i) Conditionally independent given $E$, but NOT given $E^c$.**
*Good/bad classes example:* $G$ = class is "good," $W$ = student works hard, $A$ = student gets an A. In good classes, hard work strongly predicts an A ($W,A$ dependent given $G$). In bad classes, grades are assigned randomly regardless of effort ($W \perp A \mid G^c$). So conditional independence can hold given $G^c$ but fail given $G$ — there's no reason independence given one conditioning event should transfer to its complement.

**(ii) Conditionally independent given $E$, but NOT unconditionally independent.**
*Random coin, revisited:* $F$ = fair coin chosen. Given $F$, all flips are independent (each $P(H)=1/2$); given $F^c$, all flips are independent (each $P(H)=3/4$). But **unconditionally**, flip 1 and flip 2 are *not* independent — observing many Heads on flip 1 raises your belief that you hold the biased coin, which shifts your prediction for flip 2. **This is exactly the mechanism behind hierarchical/mixture models, and why "the data conditional on the model parameter is i.i.d., but the raw data isn't" is a standard phrase in Bayesian ML** — directly relevant if the interview drifts toward ML systems design.

**(iii) Unconditionally independent, but NOT conditionally independent given $E$.**
*Alice and Bob calls example:* $A$ = Alice calls Friday, $B$ = Bob calls Friday, independent unconditionally, each with positive probability. Let $C$ = exactly one call received Friday. Then given $C$, the call is from Alice **iff** it's not from Bob — so $A$ and $B$ become perfectly (negatively) dependent given $C$: $P(B\mid A,C)=0$ even though $P(B\mid C)>0$. **This is explosively important**: conditioning on a *collider* (an event caused by both $A$ and $B$) can manufacture dependence between variables that were originally unrelated — this is the formal seed of **"explaining away"** in Bayesian networks / causal inference, a concept that comes up a lot in applied-ML and causal-inference-flavored interviews.

**Worked numerical version — crying baby (ties (i)-(iii) together with actual algebra):**
Baby cries iff hungry ($H$) or tired ($T$) or both. $P(H)=h$, $P(T)=t$, $H\perp T$ unconditionally. Let $C$ = crying.

$$P(C) = P(H\cup T) = h+t-ht = c \quad\text{(inclusion-exclusion, using independence for } P(H\cap T)=ht\text{)}$$

By Bayes:
$$P(H\mid C) = \frac{P(C\mid H)P(H)}{P(C)} = \frac hc,\qquad P(T\mid C) = \frac tc$$

$$P(H\cap T\mid C) = \frac{P(C\mid H\cap T)P(H\cap T)}{P(C)} = \frac{ht}{c}$$

Check conditional independence given $C$: is $\dfrac{ht}{c} \stackrel{?}{=} \dfrac hc\cdot\dfrac tc = \dfrac{ht}{c^2}$? Since $0<c<1$, $\dfrac{1}{c} < \dfrac{1}{c^2}$, so $P(H\cap T\mid C) < P(H\mid C)P(T\mid C)$. **So $H$ and $T$ are NOT conditionally independent given $C$** — they're *negatively* associated given crying.

**Intuition:** if the baby is crying but you learn she is **not** hungry, she must be tired (since crying requires hungry OR tired) — so learning $H^c$ given $C$ makes $T$ *more* likely. That's the signature of **"explaining away"**: two independent causes of a common observed effect become negatively correlated once you condition on the effect having occurred. This exact pattern is the theoretical basis for why, e.g., two independent risk factors for a diagnosis can appear (spuriously) to trade off against each other once you condition on diagnosis-positive patients only — a subtle form of selection bias worth flagging in any analytics interview involving conditioned/filtered cohorts.


---

## 10. Branching Processes, Legal Fallacies, and Simpson's Paradox in Full (Blitzstein & Hwang)

### 10.1 Branching Process — Amoeba Extinction (self-referential conditioning)

A single amoeba: each minute, with probability 1/3 each, it dies, stays the same, or splits into two amoebas; all descendants behave identically and independently. Find $P(\text{population eventually dies out})$.

**Setup:** condition on the outcome of the *first* step. Let $D$ = eventual extinction, $B_i$ = "splits into $i$ amoebas after minute 1," $i=0,1,2$.

- $P(D\mid B_0) = 1$ (already extinct)
- $P(D\mid B_1) = P(D)$ (back to the exact same problem — self-similarity)
- $P(D\mid B_2) = P(D)^2$ (need **both** independent lineages to separately die out)

By LOTP:
$$P(D) = 1\cdot\tfrac13 + P(D)\cdot\tfrac13 + P(D)^2\cdot\tfrac13$$

Multiply by 3: $3P(D) = 1 + P(D) + P(D)^2 \Rightarrow P(D)^2 - 2P(D) + 1 = 0 \Rightarrow (P(D)-1)^2=0 \Rightarrow P(D)=1$.

**The population dies out with probability 1**, even though it could in principle grow forever. Counterintuitive, and a great worked example of the technique: **when a problem is self-similar (splitting creates independent copies of the original problem), set up an equation in $P(D)$ itself using LOTP, rather than trying to compute an infinite sum directly.** This exact pattern (branching processes) underlies population genetics, epidemic modeling, and — very relevant to Google/Apple ML interviews — the theory behind random forests' tree-growth analysis and certain queueing/load-balancing models.

### 10.2 The Prosecutor's Fallacy — Sally Clark Case (real, consequential misuse of conditional probability)

In the Sally Clark UK murder trial (1998), two of her infant sons died shortly after birth. An expert witness testified that the probability of a single SIDS death was about 1/8500, so the probability of two SIDS deaths in one family was roughly $(1/8500)^2 \approx 1$ in 73 million — and concluded this was the probability of Clark's innocence.

**Two separate errors:**

1. **Independence was wrongly assumed.** Squaring $1/8500$ requires SIDS deaths within a family to be independent — false if there are shared genetic or environmental risk factors that elevate risk for *all* children in that family.

2. **The core prosecutor's fallacy:** the witness conflated $P(\text{evidence}\mid\text{innocence})$ (low) with $P(\text{innocence}\mid\text{evidence})$ (what actually matters). By Bayes:
$$P(\text{innocence}\mid\text{evidence}) = \frac{P(\text{evidence}\mid\text{innocence})\,P(\text{innocence})}{P(\text{evidence}\mid\text{innocence})P(\text{innocence}) + P(\text{evidence}\mid\text{guilt})P(\text{guilt})}$$
Double SIDS deaths are rare — but double **infanticide** is far rarer still, i.e. $P(\text{innocence})$ (the prior) is extremely high. If the guilt term in the denominator is negligible relative to the innocence term, $P(\text{innocence}\mid\text{evidence})$ stays close to 1 despite the small numerator. The expert's number captured only the likelihood, never the prior.

Clark was convicted and served over three years before the conviction was overturned; the case triggered review of many other prosecutions built on the same faulty statistical reasoning.

**Interview-relevance:** this is the single cleanest real-world cautionary tale for "don't skip the prior" — extremely quotable if an interviewer asks "why do people get Bayes' theorem wrong in practice?"

### 10.3 The Defense Attorney's Fallacy — Failing to Condition on All the Evidence

A murdered woman's husband is on trial; evidence shows a history of domestic abuse. The defense argues this evidence is irrelevant, citing that only 1 in 10,000 men who abuse their wives go on to murder them — i.e. $P(\text{guilt}\mid\text{abuse}) = P(G\mid A) = 1/10{,}000$.

**The flaw:** we already know the wife was murdered — $P(G\mid A)$ is the wrong conditional probability. We need $P(G\mid A, M)$ where $M$ = "wife was murdered."

Given assumptions: $P(A\mid G,M) = 0.5$ (half of husbands who murder their wives had abused them), $P(G\mid M) = 0.2$ (1 in 5 murdered wives were killed by their husband), $P(A\mid G^c, M) = 0.1$ (base rate of abuse, applying to husbands who didn't do it).

$$P(G\mid A,M) = \frac{P(A\mid G,M)P(G\mid M)}{P(A\mid G,M)P(G\mid M) + P(A\mid G^c,M)P(G^c\mid M)} = \frac{0.5\times0.2}{0.5\times0.2+0.1\times0.8} = \frac{0.10}{0.18} = \frac59 \approx 0.556$$

So conditioning on the murder having happened, prior history of abuse raises the probability of guilt from a base rate of $P(G\mid M)=0.2$ all the way to $\approx 0.56$ — over 5,000× larger than the misleading $1/10{,}000$ figure, because that figure never conditioned on the murder actually occurring.

**Interview-relevance:** this is the mirror-image mistake to the prosecutor's fallacy, and a great one to bring up if asked "what's a common Bayesian reasoning mistake in industry?" — analogous to computing $P(\text{fraud}\mid\text{risk factor})$ using population base rates instead of conditioning on the fact that you're already looking at a flagged/reviewed case (a very common real mistake in trust & safety / fraud analytics work at Google/Apple-scale companies).

### 10.4 Simpson's Paradox — Full Worked Numerical Example (Doctors)

Two surgeons, Dr. Hibbert and Dr. Nick, each perform heart surgery and Band-Aid removal:

| | Heart Success | Heart Fail | Band-Aid Success | Band-Aid Fail |
|---|---|---|---|---|
| **Hibbert** | 70 | 20 | 10 | 0 |
| **Nick** | 2 | 8 | 81 | 9 |

- Heart surgery success rate: Hibbert $70/90 \approx 77.8\%$ vs Nick $2/10=20\%$ — **Hibbert better**
- Band-Aid success rate: Hibbert $10/10=100\%$ vs Nick $81/90=90\%$ — **Hibbert better**
- **Aggregate** success rate: Hibbert $80/100=80\%$ vs Nick $83/100=83\%$ — **Nick better overall!**

**Formal definition of the paradox:** for events $A$ (success), $B$ (Nick is surgeon), $C$ (heart surgery):
$$P(A\mid B,C) < P(A\mid B^c,C) \quad\text{and}\quad P(A\mid B,C^c) < P(A\mid B^c,C^c)$$
$$\text{but} \quad P(A\mid B) > P(A\mid B^c)$$

**Mechanism via LOTP:**
$$P(A\mid B) = P(A\mid C,B)P(C\mid B) + P(A\mid C^c,B)P(C^c\mid B) = \tfrac{2}{10}(0.1) + \tfrac{81}{90}(0.9) = 0.83$$
$$P(A\mid B^c) = P(A\mid C,B^c)P(C\mid B^c) + P(A\mid C^c,B^c)P(C^c\mid B^c) = \tfrac{70}{90}(0.9) + \tfrac{10}{10}(0.1) = 0.80$$

Both are weighted averages of the same two per-surgery-type success rates — but the **weights differ**: Nick does heart surgery only 10% of the time ($P(C\mid B)=0.1$) while Hibbert does it 90% of the time ($P(C\mid B^c)=0.9$). Since heart surgery is inherently riskier, Hibbert's aggregate rate is dragged down purely by case-mix, not skill.

**Real-world instances (name the confounder in an interview to show you really get it):**
- **UC Berkeley admissions (1970s):** aggregate data suggested bias against women; within almost every department, women were admitted at equal or higher rates than men. The confound: women disproportionately applied to more competitive/lower-acceptance-rate departments.
- **Baseball batting averages:** a player can have a higher average than a rival in both halves of a season yet a lower average overall, if at-bat counts differ sharply between halves.
- **Smoking mortality (Cochran):** within every age group, cigarette smokers had higher mortality than cigar smokers, but aggregated mortality was lower for cigarette smokers — because cigarette smokers were on average younger, and age is the confound.

**Interview-relevance — this is probably the single most commonly cited "gotcha" concept for analytics/DS interviews at Google, Meta, and Apple**, because it directly models real production mistakes: comparing two product variants, ML models, or ad campaigns in aggregate without controlling for a confound (traffic source, user segment, cohort) that differs in mix between the two groups. **Always ask "could the subgroup composition differ between what I'm comparing?" before trusting an aggregate A/B test or model-comparison result.**

---

## 11. Real Conditional-Probability Interview Questions Asked at Google, Apple & Peer Companies

Sourced from public interview-question aggregators (Glassdoor, Exponent, DataLemur/Nick Singh's "Ace the Data Science Interview," StrataScratch, Interview Query) and community write-ups. Company attributions reflect what's publicly reported by candidates; treat them as representative of the *style and difficulty* you'll face rather than guaranteed-verbatim questions, since interview banks get reused and re-attributed across companies over time.

**1. (Google) Seven-game series.** Two teams play a best-of-7 series; each team has a 50% chance of winning any individual game (no draws). What's the probability the series goes the full 7 games?
*Approach:* the series reaches game 7 iff after 6 games each team has exactly 3 wins — this is a conditioning/binomial-counting question dressed up as a probability question: $P(\text{3 wins in 6 games}) = \binom{6}{3}(0.5)^6 = \frac{20}{64} = 0.3125$.

**2. (Reported at Facebook/Meta, very close in spirit to what Google/Apple ask) Coin fairness test.** You have a fair coin and a two-tailed (unfair) coin. You pick one uniformly at random, flip it 5 times, and observe tails all 5 times. What's the probability you picked the unfair coin?
*Approach:* textbook Bayes. $P(\text{unfair}\mid TTTTT) = \dfrac{1\times 0.5}{1\times 0.5 + (0.5)^5\times 0.5} = \dfrac{0.5}{0.5+0.015625} \approx 0.9697$. This is structurally identical to the Random Coin example in Section 9.7 above — recognize the pattern.

**3. (Lyft, frequently recycled at other companies incl. Google-adjacent DS interviews) HH vs. TH race.** You and a friend flip a coin repeatedly. You win if HH appears before TH; your friend wins if TH appears before HH. Find your win probability.
*Approach:* condition on the first flip. If first flip is T, then TH is now "waiting" for just one more H, and effectively TH will appear before HH almost certainly once a T has occurred (a classic Markov-chain-via-conditioning problem) — the surprising final answer is $P(\text{you win}) = 1/4$, much worse than the naive 50/50 guess. This is a great one to practice because most candidates badly misjudge the intuition (it exploits the fact that once a T shows up, TH is "locked in" ahead of HH).

**4. (Classic, shows up across Google/Amazon/quant interviews under different wrappers) Monty Hall variants.** Beyond the basic 3-door version (Section 3, Example 3), interviewers sometimes twist it: host doesn't know where the car is and opens a random door that happens to reveal a goat (answer becomes 1/2, not 2/3); or there are more doors/more revealed goats. **Always identify exactly what the host's strategy/knowledge is before answering** — that's the crux of every variant.

**5. (Two dice, standard warm-up at Google and most quant/DS phone screens) Given the first die shows 3, what's the probability the sum is 8?** (Worked in Section 3, Example 1 above — $1/6$.) Frequently the very first "can this candidate handle a conditional probability definition at all" filter question.

**6. (Google — probability/estimation style) Archery / relative performance question.** Two players with identical skill (each hits target with probability 0.5) have fired a different, close number of shots (e.g., 201 vs. 200 arrows). What's the probability the player with one more shot has strictly more hits than the other? *Approach:* by symmetry on the first 200 shots each, this reduces to whether the extra shot's single flip breaks a tie in the leader's favor — answer works out to $1/2$. Tests whether you can find symmetry shortcuts instead of brute-force computing.

**7. (General DS-interview staple, tests LOTP+Bayes fluency, close cousin of the "conditionitis" problem) Sequential test / rare-event Bayes problems.** A rare disease/fraud/defect with a known base rate, plus a test/flag with known sensitivity and specificity — compute the posterior. This exact template (Sections 9.8 and Example 2) is, by far, the most repeated conditional-probability interview question across Google, Apple, Meta, and quant shops, because it's the cleanest test of whether a candidate reflexively reaches for the prior or falls for the base-rate fallacy.

**8. (Apple-specific flavor, per public interview guides) Applied/experimentation framing rather than pure brainteasers.** Public write-ups (Interview Query, Exponent, DataInterview) consistently describe Apple's DS loop as leaning toward **applied statistics: A/B testing, experiment design, causal inference, and how you'd reason about a metric under confounding** — rather than isolated math-puzzle-style conditional probability questions. In that context, **Simpson's Paradox (Section 10.4) is exactly the concept most likely to appear**, e.g. framed as "we launched a feature, engagement went up in every user segment but down overall — what happened?" Practicing how to *articulate* the confounding-variable explanation out loud is more valuable prep for Apple than memorizing more brainteasers.

**9. (Common "hint-giving" Monty-Hall-style variant, shows up in DS interview compilations) Multi-box hint problem.** With several boxes and a party who *knows* where a prize is and gives you a truthful "not in this box" hint after you've committed to a guess, compute the posterior probability the prize is in a specific remaining box. Same core lesson as Monty Hall: the *hint-giver's knowledge and constraints* are what make the hint informative — treat it as a Bayes problem where the likelihood of the specific hint given each hypothesis is the key computation, exactly like Section 3 Example 3.

**Meta-takeaway for your prep:** across Google, Apple, and peers, the actual distribution of conditional-probability questions clusters around four recurring templates — **(a) rare-event Bayes/base-rate problems, (b) Monty-Hall-style "informative revealing" problems, (c) sequential/race-condition coin or game problems solved by conditioning on the first step, and (d) Simpson's-Paradox-style aggregation traps.** If you can solve one clean example of each from scratch, on a blank page, while narrating prior/likelihood/posterior out loud, you're prepared for the large majority of what actually gets asked.


---

## 12. More Worked Examples (Sheldon Ross, *A First Course in Probability*)

### 12.1 Celine's Course Choice (simple LOTP application)

Celine estimates $P(A\mid \text{French}) = 1/2$, $P(A\mid\text{chemistry}) = 2/3$. She flips a fair coin to choose. Find $P(\text{chemistry AND gets an A})$.

Let $C$ = chooses chemistry, $A$ = gets an A.
$$P(C\cap A) = P(C)\,P(A\mid C) = \frac12\cdot\frac23 = \frac13$$

**Note the phrasing:** this asks for a *joint* probability $P(C\cap A)$, not $P(A\mid C)$ — a good reminder to read the question precisely before reaching for a formula. Straightforward application of the multiplication rule (Section 1.2), included here mainly as a clean warm-up rep.

### 12.2 Multiple-Choice Guessing (Bayes with a clean closed form)

A student either knows the answer (probability $p$) or guesses (probability $1-p$), and a guess is correct with probability $1/m$ ($m$ = number of choices). Given the student answered correctly, what's the probability they actually knew it?

Let $K$ = knew the answer, $C$ = answered correctly.
$$P(K\mid C) = \frac{P(C\mid K)P(K)}{P(C\mid K)P(K) + P(C\mid K^c)P(K^c)} = \frac{p}{p + \frac1m(1-p)} = \frac{mp}{1+(m-1)p}$$

**Plug in $m=5, p=1/2$:** $P(K\mid C) = \dfrac{5(0.5)}{1+4(0.5)} = \dfrac{2.5}{3} = \dfrac56 \approx 0.833$.

**Why this is a great one to have memorized as a closed form:** it's the general template behind any "did the model actually detect signal, or did it get lucky by chance" question — e.g. in an ML context, distinguishing whether a classifier's correct prediction reflects real learned signal versus a high base rate of lucky guesses. The formula $\frac{mp}{1+(m-1)p}$ is worth deriving once by hand so you can rebuild it live rather than memorize it.

### 12.3 Flashlight Types (3-way partition, LOTP + Bayes together)

Bin has type 1 (20% of stock, 70% chance of lasting 100+ hrs), type 2 (30% of stock, 40% chance), type 3 (50% of stock, 30% chance).

**(a) Unconditional probability of lasting 100+ hours** — condition on type via LOTP:
$$P(A) = (.7)(.2)+(.4)(.3)+(.3)(.5) = .14+.12+.15 = .41$$

**(b) Given it lasted 100+ hours, which type was it?** — Bayes, using the same numerator terms computed above:
$$P(F_1\mid A) = \frac{.14}{.41}=\frac{14}{41}\approx.341,\quad P(F_2\mid A)=\frac{12}{41}\approx.293,\quad P(F_3\mid A)=\frac{15}{41}\approx.366$$

Sanity check: these three posteriors sum to 1 ✓. **Notice the shift**: type 1 was only 20% of stock but jumps to 34.1% of the posterior because it's disproportionately likely to survive — while type 3 drops from 50% prior share to 36.6% posterior share despite still being the modal type, because it's the least reliable. **This "three-or-more-way Bayes" pattern (compute all LOTP terms once, then divide each by the same denominator) is exactly how you'd do multi-class Bayesian classification / Naive Bayes by hand — good to practice beyond the two-outcome disease-test template.**

### 12.4 DNA Forensic Match (a genuinely hard, multi-layered Bayes problem — great for testing depth in a longer interview)

A crime scene DNA sample matches on 5 strands; an innocent person matches by chance with probability $10^{-5}$, independently. Of 1,000,000 town residents, 10,000 are ex-convicts with DNA on file. The DA assumes each ex-convict is $c$ times as likely to be guilty as each of the other 990,000 residents (prior odds ratio $c$). Checking the database, exactly one person — A.J. Jones — matches. Find $P(\text{A.J. guilty}\mid\text{unique match})$.

**Setup:** with $\alpha$ = prior guilt probability per ex-convict, $\beta$ = prior guilt probability per non-convict, $\alpha = c\beta$, and probabilities summing to 1 across all 1,000,000 residents:
$$10{,}000\alpha + 990{,}000\beta = 1 \;\Rightarrow\; \beta = \frac{1}{10{,}000c+990{,}000},\quad \alpha = \frac{c}{10{,}000c+990{,}000}$$

Let $G$ = A.J. guilty, $M$ = A.J. is the *unique* match among the 10,000 on file.

- $P(M\mid G) = (1-10^{-5})^{9999}$ — A.J. matches automatically (he's guilty), and none of the other 9999 ex-convicts (innocent) can also match.
- $P(M\mid G^c)$ requires: A.J. (innocent) matches by chance ($10^{-5}$), **and** all other 9999 ex-convicts are innocent, **and** none of them match either:
$$P(M\mid G^c) = 10^{-5}\cdot\underbrace{\frac{1-10{,}000\alpha}{1-\alpha}}_{P(\text{all others innocent}\mid \text{A.J. innocent})}\cdot(1-10^{-5})^{9999}$$

Combining via Bayes and simplifying (using $P(G)=\alpha$) collapses to the clean closed form:
$$P(G\mid M) = \frac{\alpha}{\alpha + 10^{-5}(1-10{,}000\alpha)} = \frac{1}{0.9 + 10^{-5}/\alpha}$$

**Plugging in different prior beliefs about $c$ (how much more suspicious an ex-convict is a priori):**

| DA's prior belief $c$ | $\alpha$ | $P(G\mid M)$ |
|---|---|---|
| $c=100$ (ex-convict 100× more likely a priori) | $1/19{,}900$ | $\approx 0.910$ |
| $c=10$ | $1/109{,}000$ | $\approx 0.503$ |
| $c=1$ (no prior suspicion — uniform over all 1M residents) | $10^{-6}$ | $\approx 0.092$ |

**Why this example is worth the effort to work through fully:** it's a real, high-stakes illustration of how **the final posterior can swing from ~9% to ~91% guilt purely based on a subjective prior assumption ($c$)**, even with the *exact same DNA evidence*. This is precisely the kind of "the math is right, but the answer depends entirely on an assumption nobody agreed on" issue that comes up constantly in applied Bayesian modeling, risk scoring, and fraud detection at scale — a strong example to cite if an interviewer probes "what are the limitations of a Bayesian approach in production?"

---

## 13. Tree-Diagram Method for Bayes' Theorem

A probability tree is often the fastest way to *derive* — and to explain out loud on a whiteboard — a Bayes computation without getting lost in the formula. **Method:** draw branches for the prior partition first, then sub-branches for the conditioning event under each; multiply along each root-to-leaf path to get joint probabilities; sum the leaves matching your evidence to get $P(\text{evidence})$; divide the specific leaf(s) you want by that sum.

### 13.1 Disease Testing via Tree (prevalence 1%, sensitivity 99%, specificity 95%)

```
                         START
                 ┌────────────┴────────────┐
            D (0.01)                   Dᶜ (0.99)
         ┌────┴────┐                ┌────┴────┐
     + (0.99)   − (0.01)         + (0.05)   − (0.95)
       │            │               │            │
  0.01×0.99     0.01×0.01      0.99×0.05     0.99×0.95
   = 0.0099      = 0.0001       = 0.0495      = 0.9405
```

Leaves matching "tested positive": $0.0099$ (top-left) and $0.0495$ (bottom-left). Sum = $P(+) = 0.0594$.

$$P(D\mid +) = \frac{0.0099}{0.0594} \approx 0.1667$$

**Why the tree is more effective on a whiteboard than the formula:** the interviewer can *see* both routes to a positive test — the tiny true-positive branch and the much fatter false-positive branch — which visually communicates the base-rate fallacy far faster than the algebra does. Multiply along the branches, then just look at the ratio of the branch you care about to the sum of all matching branches.

### 13.2 Monty Hall via Tree (you pick door 1; host opens a goat door, never door 1, never the car)

**Level 1 — where's the car (prior, all equal):**

```
                                START
              ┌───────────────────┼───────────────────┐
        Car@1 (1/3)          Car@2 (1/3)          Car@3 (1/3)
```

**Level 2 — which door does the host open** (host must avoid door 1 and the car door):

```
Car@1 (1/3)                    Car@2 (1/3)              Car@3 (1/3)
  ┌────┴────┐                       │                        │
opens 2   opens 3               opens 3                   opens 2
 (1/2)     (1/2)                 (1)                        (1)
  │          │                    │                          │
1/3×1/2   1/3×1/2              1/3×1                       1/3×1
 =1/6      =1/6                 =1/3                        =1/3
```

(When the car is at 1, the host has a genuine free choice between doors 2 and 3, so each gets weight 1/2. When the car is at 2, the host is *forced* to open 3 — the only remaining goat door not door 1. Symmetric logic when the car is at 3.)

**Now condition on the observed evidence "host opened door 3"** — pick out only the leaves under "opens 3":

| Path | Leaf probability |
|---|---|
| Car@1 → opens 3 | $1/6$ |
| Car@2 → opens 3 | $1/3$ |
| Car@3 → opens 3 | (impossible — host can't open the car door) |

Sum of matching leaves: $1/6+1/3 = 1/2 = P(\text{host opens door 3})$.

$$P(\text{Car@1}\mid\text{opens }3) = \frac{1/6}{1/2}=\frac13 \qquad P(\text{Car@2}\mid\text{opens }3)=\frac{1/3}{1/2}=\frac23$$

**Switching (moving to door 2) wins with probability 2/3; staying (door 1) wins with probability 1/3** — same answer as the algebraic Bayes derivation in Section 3, Example 3, but the tree makes the *asymmetric branching* (host has a free 50/50 choice only when you're already right, and zero choice otherwise) visually obvious, which is exactly the fact that breaks the naive "50/50 after one door is removed" intuition. **This is the single best way to explain Monty Hall live in an interview**: draw the tree, show that the "Car@2" branch has no competing branch to split its probability with (host is forced), while the "Car@1" branch has to split 1/3 in half — so the surviving door-2 leaf ends up twice as large as the surviving door-1 leaf.
