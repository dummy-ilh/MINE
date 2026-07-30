# Chapter 4: Confounding, DAGs, and the Backdoor Criterion

## 1. Explanation

### Why you need a picture, not just a formula

When you can't randomize, you rely on **observed covariates** to try to recreate the conditions of an RCT within subgroups. To do this rigorously, it helps enormously to draw a **Directed Acyclic Graph (DAG)** — a picture with arrows showing assumed causal relationships between variables. DAGs aren't a nice-to-have visualization; they let you *mechanically derive*, using graph-theoretic rules, exactly which variables you should and shouldn't control for. Without a DAG, "control for confounders" is vague advice that different people will interpret differently, often incorrectly.

### What is a confounder, precisely?

A confounder $Z$ is a variable that causally affects **both** the treatment $D$ and the outcome $Y$. It creates a spurious statistical association between $D$ and $Y$ even if $D$ has *zero* true causal effect on $Y$. Classic picture:
```
      Z (confounder)
     /              \
    v                v
    D  ------------> Y
   (treatment)     (outcome)
```
Ice cream sales (D) and drowning deaths (Y) are correlated — not because ice cream causes drowning, but because hot weather (Z) independently increases both. If you don't control for Z, you'll see a strong D-Y correlation that has nothing to do with a direct causal arrow between them.

### The three canonical structures — you must recognize these instantly

**1. Chain**: $A \rightarrow B \rightarrow C$. Here $B$ is a **mediator** — part of the causal *pathway* from A to C. If you're trying to measure A's total effect on C, you should generally **not** control for B, because doing so blocks (statistically removes) exactly the mechanism you're trying to measure, biasing your estimate of the total effect toward zero.

**2. Fork**: $A \leftarrow B \rightarrow C$. Here $B$ is a **confounder** of A and C — a genuine common cause. You generally **should** control for B, because it creates a non-causal, spurious path between A and C that needs to be closed off to isolate any real causal relationship.

**3. Collider**: $A \rightarrow B \leftarrow C$. Here $B$ is a **collider** — a common *effect* of A and C, not a common cause. Critically: A and C are *already statistically independent* (no confounding path exists between them) *unless* you condition on B. Conditioning on (or sample-restricting by) a collider **opens up** a spurious association between A and C that didn't exist before you touched it. This is the single most counterintuitive fact in this chapter: adding more control variables is not automatically "more rigorous" — sometimes it actively creates bias that wasn't there.

### The backdoor criterion, explained conceptually then formally

You want to isolate the *causal* arrow from D to Y and shut down every *non-causal* ("backdoor") path connecting them. A set of variables $X$ satisfies the **backdoor criterion** relative to (D, Y) if:
1. No variable in $X$ is a descendant of $D$ (i.e., nothing in your control set was itself *caused by* the treatment — controlling for a post-treatment variable is exactly the mediator/collider trap above), and
2. $X$ blocks every "backdoor path" between D and Y — every path that starts with an arrow *pointing into* D (these are inherently non-causal, confounding-type paths, as opposed to the causal path that starts by pointing *out of* D).

If $X$ satisfies this, then conditioning on $X$ achieves **conditional ignorability** (also called "unconfoundedness," "no unmeasured confounders," or the **Conditional Independence Assumption, CIA**):
```
(Y(1), Y(0)) ⊥ D | X
```
This is the linchpin assumption underneath essentially every observational-data method you'll learn (regression adjustment, matching, propensity scores). It is, crucially, **untestable from data alone** — you can never prove there's no hidden confounder lurking outside your DAG; you can only argue plausibility using domain knowledge, and quantify fragility using sensitivity analysis (Chapter 12).

### Selection bias as a special, related case

Selection bias is closely related to collider bias but broader: it arises whenever the *sample itself* is restricted or filtered in a way tied to both treatment and outcome (or a descendant of both) — e.g., only studying "users who completed onboarding," when onboarding-completion is itself influenced by both the treatment and unrelated factors correlated with the outcome. It's essentially collider bias induced by how the data was collected/filtered, rather than by a variable you explicitly chose to control for in a model.

## 2. Example

### Example A — Full DAG walkthrough (confounder vs mediator vs collider, all in one scenario)

**Scenario:** Google wants to know if "using the Android widget" (D) causes higher "daily app opens" (Y). Candidate variables: `prior_engagement` (heavy users more likely to set up widgets AND already open apps a lot, independent of the widget), `notification_count` (a variable that occurs *after* someone starts using the widget, since the widget itself can trigger extra notifications), and `contacted_support` (a variable plausibly caused by *both* having widget-related bugs and by being a highly engaged/vocal user).

DAG:
```
prior_engagement --> D (widget use)
prior_engagement --> Y (app opens)
D --> notification_count --> Y
D --> contacted_support <-- engagement_style (unobserved)
```
- `prior_engagement` is a **confounder** (fork: it independently drives both D and Y) → **should control for it**.
- `notification_count` is a **mediator** (chain: D → notification_count → Y, part of the true causal pathway) → **should NOT control for it** if the goal is the *total* effect of D on Y; controlling for it would only recover the narrower "direct effect not through notifications," a different question.
- `contacted_support` is a **collider** (both D and an unobserved "engagement_style" factor cause it) → **should NOT control for it**; restricting analysis to "contacted support" users (or controlling for this variable) would open a spurious path between D and the unobserved engagement style, potentially inducing a fake association between widget use and app opens that has nothing to do with the widget itself.

### Example B — Quantifying omitted-variable bias numerically

Suppose (a simplified linear world) the *true* causal model is:
```
Y = 2 + 3D + 5Z + ε        (Z drives Y directly, coefficient 5)
D = 0.1Z + ν               (Z also drives D, coefficient 0.1)
```
with $Z \sim \text{Uniform}(0,1)$ representing "prior engagement," and $\varepsilon, \nu$ independent mean-zero noise. The *true* causal effect of D on Y is 3.

If you naively regress Y on D **without** controlling for Z, the well-known omitted-variable-bias formula applies:
```
Bias = β_Z × Cov(D,Z) / Var(D)
```
With $Z \sim U(0,1)$: $\text{Var}(Z) = 1/12 \approx 0.0833$.
```
Cov(D,Z) = Cov(0.1Z + ν, Z) = 0.1 × Var(Z) = 0.1 × 0.0833 = 0.00833
```
Suppose $\text{Var}(\nu) = 0.02$ (some independent noise in D). Then:
```
Var(D) = 0.1² × Var(Z) + Var(ν) = 0.01×0.0833 + 0.02 = 0.000833 + 0.02 ≈ 0.02083
```
```
Bias = 5 × (0.00833 / 0.02083) = 5 × 0.40 = 2.0
```
So the naive regression coefficient on D would be approximately **3 (true) + 2.0 (bias) = 5.0** — overstating the true causal effect of 3 by 67%, purely because Z confounds both D and Y and was left out of the model. This numerically demonstrates: even a "small-looking" confounding relationship (Z only weakly drives D here, coefficient 0.1) can produce substantial bias if Z's effect on Y (coefficient 5) is large — bias depends on the *product* of both relationships, not just one.

## 3. Interview Q&A

**Q: Why is "just add every available variable as a control" a dangerous default strategy?**
A: Because some available variables are colliders or mediators, not confounders. Controlling for a mediator blocks part of the true causal effect (biasing toward zero, or answering a narrower "direct effect" question instead of the total effect you wanted). Controlling for a collider can *create* spurious associations that didn't exist in the true causal structure. You need a DAG (or at minimum, explicit reasoning about temporal order and causal roles) before deciding what to control for — "more controls = more rigorous" is a common but incorrect intuition, and interviewers specifically probe for whether you know this.

**Q: Give a Google-relevant example of a collider that a naive analyst might mistakenly control for.**
A: Studying whether "video quality" (D) affects "user complaint rate" (Y) using only sessions where the user contacted support about *something* — support contact is a collider potentially caused jointly by video quality issues and by unrelated account issues. Restricting the sample to "contacted support" (or controlling for it) can induce a spurious relationship between video quality and complaint content that doesn't reflect the true, unrestricted population relationship.

**Q: How do you handle the fact that you can never be 100% sure your DAG (and thus your control set) is correctly specified?**
A: Be transparent that the DAG encodes domain assumptions, not proven facts. Use sensitivity analysis (E-values, Rosenbaum bounds — Chapter 12) to quantify how strong an *unmodeled* confounder would need to be to overturn your conclusion. Also, seek out several independent, DAG-consistent specifications (different plausible control sets, informed by different domain experts) and check whether conclusions are stable across them — convergence across reasonable alternative specifications increases confidence more than any single DAG can on its own.

**Q: A colleague wants to control for "customer support ticket volume" when estimating the effect of a product bug (D) on churn (Y). Good idea?**
A: Likely a bad idea if ticket volume is *caused by* the bug (chain: D → tickets → churn) — it's a mediator on the causal pathway from bug to churn (frustrated users file tickets and then churn). Controlling for it would strip out exactly the mechanism you're trying to measure, biasing the estimated total effect toward zero. You'd first need to check the timing and causal story: is ticket volume measured *before* the bug's effects could manifest (then it might be a legitimate pre-treatment confounder), or after/concurrently (then it's a mediator to avoid controlling for)?

**Q: What's the practical first step you take on a new observational causal question, before choosing an estimator?**
A: Draw the DAG (even informally, on a whiteboard, with domain experts) — decide what's a plausible confounder (fork, control for it), what's a mediator (chain, don't control for it if you want the total effect), and what's a collider (common effect, don't control for it). This determines your control set and estimator strategy *before* touching code, and is exactly the reasoning an interviewer wants to hear verbalized rather than jumping straight to "let's run a regression with all available features."

**Q: What's the difference between confounding bias and selection bias, in terms of when each one is introduced?**
A: Confounding bias exists in the underlying data-generating process itself — a real, uncontrolled common cause of D and Y exists in the population, whether or not you've done anything to your sample. Selection bias is typically introduced by *how the sample was constructed or filtered* (e.g., restricting to users who did X, or differential dropout) — it's a property of your data collection/analysis choices, often functioning like collider bias, opening a spurious path that wouldn't exist in the full, unrestricted population.

**Q: If Z has only a very weak effect on D (say, correlation of 0.1) can it still meaningfully bias your estimate if left uncontrolled? Walk through why or why not.**
A: Yes — as the worked numerical shows, omitted variable bias is proportional to the *product* of Z's effect on D and Z's effect on Y (Bias = β_Z × Cov(D,Z)/Var(D)). A weak D-side relationship can still produce large bias if Z's effect on Y is large enough, since the bias formula multiplies both effects together rather than depending on either one alone. This is why you can't dismiss a "weak" confounder just by looking at how strongly it predicts treatment — you also need to know how strongly it independently predicts the outcome.

---
**Previous: Chapter 3 — Randomized Experiments (RCTs)**
**Next: Chapter 5 — Regression Adjustment**
