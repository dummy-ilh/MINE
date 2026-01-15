
**Part Four: Design and Analysis of Single-Factor Studies** introduces the crucial foundation of how data is collected for statistical analysis. It moves beyond simply analyzing existing data (as in much of regression) to understanding how to design studies that allow for valid causal inference and efficient estimation.

-----

## Chapter 15: Introduction to the Design of Experimental and Observational Studies


### 15.1 Experimental Studies, Observational Studies, and Causation 


Understanding **how data is collected** is fundamental to understanding **what conclusions we are allowed to draw**, especially when the goal is to infer **cause-and-effect**. This section clarifies *why some studies can support causal claims while others cannot*, even if they analyze large datasets with sophisticated models.

---

## 🔹 Experimental Studies

### 📌 What Is an Experimental Study?

An **experimental study** is one in which the researcher **actively intervenes** in the system by assigning subjects to different treatments and then measuring outcomes.

In simple terms:
> The researcher *decides who gets what*.

---

### 🧠 Core Characteristics (Why Experiments Are Powerful)

#### 1️⃣ Manipulation of the Independent Variable  
The researcher **controls** the explanatory variable (treatment, factor).

- Example: Assigning Drug A vs Drug B vs Placebo
- Example: Setting different prices to study consumer demand
- Example: Assigning different teaching methods to classrooms

This eliminates ambiguity about *what caused what*.

---

#### 2️⃣ Random Assignment (The Key to Causality)  
Subjects are randomly assigned to treatment groups.

**Why this matters:**
- Randomization balances **both observed and unobserved confounders**
- On average, groups are similar in:
  - Age
  - Health
  - Motivation
  - Socioeconomic status
  - Any unknown variables

Mathematically:
> Random assignment makes the treatment independent of all confounders.

---

#### 3️⃣ Control of Extraneous Variables  
Researchers often use:
- **Control groups**
- **Blinding / double-blinding**
- **Blocking** (grouping similar subjects before randomization)
- **Standardized protocols**

These reduce bias and noise.

---

### 🎯 What Can We Conclude?

✅ **Causal relationships**

If groups differ *only* in treatment, then:
> Differences in outcomes **must be caused by the treatment**.

---

### 📘 Example: Randomized Drug Trial

- Subjects randomly assigned to:
  - Drug group
  - Placebo group
- Outcome: Blood pressure reduction

Because of random assignment:
- Diet, exercise, genetics, stress → balanced across groups
- Any systematic difference in blood pressure is attributable to the drug

✅ Valid causal claim:  
> “The drug causes a reduction in blood pressure.”

---

### ⚠️ Limitations of Experimental Studies

- Ethical constraints (cannot randomly assign smoking, poverty, trauma)
- Costly and time-consuming
- Sometimes unrealistic or artificial settings
- Limited external validity in some cases

---

## 🔹 Observational Studies

### 📌 What Is an Observational Study?

An **observational study** measures variables **as they naturally occur**, without intervention or random assignment.

In simple terms:
> The researcher *observes what already happened*.

---

### 🧠 Core Characteristics

#### 1️⃣ No Manipulation  
The researcher does **not assign treatments**.

- Smoking status
- Income level
- Education
- Diet
- Exercise habits

These arise naturally.

---

#### 2️⃣ No Random Assignment  
Subjects **self-select** into exposure groups.

This introduces **systematic differences** between groups.

---

#### 3️⃣ Limited Control Over Confounding  
Researchers attempt to adjust using:
- Regression models
- Covariate adjustment
- Matching
- Stratification

But:
> You can only adjust for variables you measured — **unobserved confounders remain**.

---

### 🎯 What Can We Conclude?

❌ **No direct causation**  
✅ **Association / correlation**

Because alternative explanations cannot be ruled out.

---

### 📘 Example: Smoking and Lung Cancer

- Observe smokers vs non-smokers
- Smokers have higher lung cancer rates

Possible explanations:
- Smoking causes cancer ✅
- Smokers differ in lifestyle, occupation, pollution exposure ❌
- Genetic factors influence both smoking and cancer risk ❌

Even with regression adjustment:
> We can reduce bias, but not eliminate it completely.

Thus, the correct statement is:
> “Smoking is strongly associated with lung cancer.”

(Not: “Smoking causes lung cancer” — unless supported by experimental or quasi-experimental evidence.)

---

### 🧠 Why “Correlation ≠ Causation” Matters Here

Observed relationship:
\[
X \leftrightarrow Y
\]

Could be:
- \( X \rightarrow Y \) (causal)
- \( Y \rightarrow X \) (reverse causality)
- \( Z \rightarrow X \) and \( Z \rightarrow Y \) (confounding)

Observational studies cannot fully distinguish among these.

---

### ⚠️ Strengths and Limitations

**Strengths**
- Ethical feasibility
- Large sample sizes
- Real-world relevance
- Long-term outcomes

**Limitations**
- Confounding bias
- Selection bias
- Measurement error
- Cannot definitively establish causality

---

## 🔹 Mixed Experimental and Observational Studies

### 📌 What Are Mixed Studies?

These designs contain **both experimental and observational components**.

Some variables are:
- **Randomized** → causal inference possible
- **Observed** → associational inference only

---

### 📘 Example: Clinical Trial with Patient Characteristics

- Drug dosage: randomized ✅
- Patient age, gender, lifestyle: observed ❌

Interpretation:
- Effect of dosage → causal
- Effect of age → associational

The analysis must:
- Respect randomization for experimental factors
- Adjust cautiously for observed covariates

---

### 📘 Example: Education Intervention

- Schools randomly assigned a new curriculum
- Students’ socioeconomic status is observed

Valid conclusions:
- Curriculum effect → causal
- SES effect → correlational

---

## 🔑 Big Picture Summary

| Study Type        | Manipulation | Random Assignment | Causal Inference |
|-------------------|-------------|-------------------|------------------|
| Experimental      | ✅ Yes      | ✅ Yes            | ✅ Strong        |
| Observational     | ❌ No       | ❌ No             | ❌ Limited       |
| Mixed             | ⚠️ Partial  | ⚠️ Partial        | ⚠️ Depends       |

---

## 🧠 Key Takeaway

> **Causation comes from design, not from statistical sophistication.**

No amount of modeling can fully replace:
- Random assignment
- Experimental control

Understanding this distinction is essential for:
- Statistical inference
- Machine learning interpretation
- Policy evaluation
- Scientific reasoning



### 15.2 Experimental Studies: Basic Concepts 

*A unified, intuitive, and design-focused treatment*

This section develops the **core building blocks of experimental design**, explaining *what decisions must be made*, *why they matter*, and *how they affect causal conclusions*. The emphasis is on **design logic**, not just terminology.

---

## 🔹 What Is an Experimental Design?

The **design of an experiment** refers to the *entire structural plan* governing how data are generated. Specifically, it includes decisions about:

1. **Which explanatory factors** are studied  
2. **Which treatments** are included  
3. **What constitutes an experimental unit**  
4. **How treatments are randomly assigned**  
5. **What outcomes are measured and how**

These decisions determine:
- Whether causal inference is valid
- How precise estimates will be
- Whether results generalize beyond the study

---

## 🔹 Factors

### 📌 Definition
A **factor** is an explanatory variable whose effect on a response is of interest.

In regression language:
- Factors ≈ predictors / independent variables

---

### 🔸 Experimental vs Observational Factors

#### Experimental Factor
- Levels are **assigned at random** by the investigator  
- Supports **cause-and-effect inference**

**Example**
- Baking temperature in a bread-volume experiment
- Drug dosage in a clinical trial

#### Observational Factor
- Levels are **not controlled** by the investigator  
- No causal interpretation allowed

**Example**
- Training center location
- Patient age
- Instructor preference

⚠️ Even in experiments, **observational factors may appear**. Effects of such factors must be interpreted associationally.

---

### 🔸 Qualitative vs Quantitative Factors

#### Qualitative (Categorical) Factors
Levels differ by type, not magnitude.

Examples:
- Advertisement type
- Brand
- Television program
- Teaching method

Modeled using indicator (dummy) variables.

#### Quantitative Factors
Levels are numerical with meaningful intervals.

Examples:
- Temperature (°F or °C)
- Price ($)
- Time (minutes)
- Dosage (mg)

---

### 🔸 Factor Levels

A **factor level** is a specific value or category of a factor.

**Example (Bread Volume Study)**
- Factor: Baking temperature  
- Levels: 320°F, 340°F, 360°F, 380°F  

A factor with \( r \) levels can generate:
- \( r \) treatments (single-factor study)
- Or many treatment combinations (multifactor study)

---

## 🔹 Single-Factor vs Multifactor Studies

### 🔸 Single-Factor Study
Only one factor varies.

**Example**
- Four baking temperatures → 4 treatments

Useful for:
- Isolated effects
- Simpler interpretation

---

### 🔸 Multifactor Study
Two or more factors vary simultaneously.

**Example**
- Temperature (Low, Medium, High)
- Solvent concentration (Low, High)

This produces:
\[
3 \times 2 = 6 \text{ treatment combinations}
\]

Multifactor studies allow:
- Estimation of **main effects**
- Detection of **interactions**

---

## 🔹 Crossed vs Nested Factors

### 🔸 Crossed Factors

Two factors are **crossed** if *every level of one appears with every level of the other*.

**Conceptual Grid**

| Temperature | Low | Medium | High |
|------------|-----|--------|------|
| Solvent Low | X | X | X |
| Solvent High | X | X | X |

**Why important**
- Enables interaction analysis
- Common in factorial experiments

---

### 🔸 Nested Factors

A factor is **nested** within another if its levels are unique to one level of the higher factor.

**Example**
- Operators nested within plants  
- Each operator works in only one plant

**Structure**


Plant 1 → Operators 1, 2, 3
Plant 2 → Operators 4, 5, 6
Plant 3 → Operators 7, 8, 9



⚠️ Nested factors:
- Do **not** allow interaction estimation
- Often arise due to logistical constraints

---

## 🔹 Treatments

### 📌 Definition
A **treatment** is the specific condition applied to an experimental unit.

- Single-factor study → treatment = factor level
- Multifactor study → treatment = **combination of factor levels**

**Example**
- Price ($0.25, $0.29)
- Package color (Red, Blue)

Treatments:
- ($0.25, Red)
- ($0.25, Blue)
- ($0.29, Red)
- ($0.29, Blue)

---

### ⚠️ Treatment Definition Pitfalls

Defining treatments incorrectly can **confound results**.

**Programming Language Example**
- Is the treatment the language?
- Or language + instructor preference?
- Or language × instructor?

Design must ensure:
- Treatment effects are not entangled with instructor effects
- Randomization is meaningful

---

## 🔹 Choice of Treatments

### 🔸 Number of Factors
- Early-stage studies often identify *many candidate factors*
- Too many factors → infeasible experiment

**Tool**
- Cause-and-effect (Ishikawa / fishbone) diagrams

Goal:
- Screen down to the most influential factors

---

### 🔸 Number of Levels

#### Qualitative Factors
- Often dictated by context
- May be reduced to save cost

#### Quantitative Factors
Chosen based on expected response shape:

| Expected Relationship | Recommended Levels |
|----------------------|--------------------|
| Linear | 2 |
| Quadratic / curvature | 3 |
| Complex / asymptotic | 4+ |

---

### 🔸 Range of Levels (Quantitative Factors)

Choosing the range is critical:

- Too narrow → effect undetectable
- Too wide → miss important structure

**Bread Temperature Example**
- True response peaks near 400°F
- Using only 250°F and 450°F misses the maximum
- Using only 250–300°F suggests no effect

Good ranges require **subject-matter knowledge**.

---

### 🔸 Control Treatments

A **control treatment**:
- Uses identical procedures
- Applies *no active treatment*

**Why needed**
- When effectiveness is unknown
- When baseline comparison is required

⚠️ Controls must be run **inside the experiment**, not externally.

Laboratory vs home ratings example shows how failure to do this leads to misleading conclusions due to **context effects**.

---

## 🔹 Experimental Units

### 📌 Definition
> The **experimental unit** is the smallest unit to which a treatment is independently assigned.

This is determined by **randomization**, not measurement.

---

### 🔸 Examples

| Scenario | Experimental Unit |
|--------|------------------|
| Incentive pay by plant | Plant |
| Fertilizer per pot | Pot |
| Drug per patient | Patient |
| Commercial shown weekly | Time period |

Misidentifying the unit leads to:
- Pseudoreplication
- Invalid inference

---

### 🔸 Representativeness

Experimental units should reflect the population of interest.

**Caution**
- Students ≠ managers
- Lab behavior ≠ field behavior

External validity depends on unit representativeness.

---

## 🔹 Sample Size and Replication

### 🔸 Replication

**Replication = repeated application of the same treatment**

Benefits:
1. Estimates experimental (pure) error
2. Increases precision
3. Enables hypothesis testing

Example:
- 4 treatments × 2 replicates = 8 units

---

### 🔸 Experimental Error
Differences among units receiving the same treatment reflect:
- Measurement noise
- Uncontrolled variability

Low error → high reproducibility  
High error → noisy response

---

## 🔹 Randomization

### 📌 Why Randomize?

Randomization:
- Breaks links between treatments and confounders
- Eliminates selection bias
- Justifies probability-based inference

It is an **insurance policy against unknown bias**.

---

### 🔸 What Should Be Randomized?

- Treatment assignment
- Order of runs
- Timing of treatments
- Subject order

Any process susceptible to systematic effects should be randomized.

---

### 🔸 How Randomization Works (Conceptual)

1. List treatments with replication
2. Generate random numbers
3. Sort by random number
4. Assign in sorted order to experimental units

This ensures:
- No hidden structure
- Fair comparison

---

## 🔹 Blocking (Preview)

Blocking is **restricted randomization**:
- Units are grouped by a nuisance factor
- Randomization occurs *within blocks*

Purpose:
- Reduce unexplained variability
- Increase precision

Example:
- Soil fertility blocks in agriculture
- Time blocks in behavioral studies

(Developed in detail in later sections.)

---

## 🔑 Core Design Principle

> **Causality is earned by design, not by analysis.**

Randomization, replication, and proper definition of units and treatments are what make statistical conclusions meaningful.

Poor design cannot be rescued by sophisticated modeling.



### 15.3 An Overview of Standard Experimental Designs (Page 658)
  
*Models, Blocking, Nesting, Repeated Measures, and Advanced Factorial Designs*

This section moves beyond **basic experimental structure** and explains how **statistical models reflect design choices**, and how more sophisticated designs improve efficiency, precision, and interpretability.

The unifying idea is simple:

> **Design determines the model, and the model determines what questions you can answer.**

---

## 🔹 General Linear Model for Designed Experiments

Most experimental designs can be represented by a **linear statistical model** of the form:

\[
y = \text{Overall Constant}
+ \text{First-Order Treatment Effects}
+ \text{Interaction Effects}
+ \text{Experimental Error}
\]

This decomposition clarifies *where variation in the response comes from*.

---

### 🔸 First-Order (Main) Effects

These represent the **individual effects** of each factor.

In factorial experiments, main effects are modeled using **indicator (dummy) variables**:

- \(X_i = 1\) if treatment \(i\) is present  
- \(X_i = 0\) otherwise  

Each coefficient answers:
> *What happens to the mean response when this factor changes, holding others constant?*

---

### 🔸 Interaction Effects

Interaction terms capture **non-additive behavior**:

\[
X_1X_2,\; X_1X_3,\; X_2X_3,\; X_1X_2X_3
\]

These answer questions like:
> *Does the effect of temperature depend on pressure?*  
> *Does one factor amplify or dampen another?*

Interactions are **central to factorial experiments** and are impossible to detect using one-factor-at-a-time designs.

---

## 🔹 Randomized Complete Block Designs (RCBD)

### 📌 Motivation
Experimental units are often **heterogeneous**, but can be grouped into **homogeneous subsets**.

Blocking removes known nuisance variability, improving precision.

---

### 🔸 Structure of a Blocked Design

1. Divide experimental units into **blocks** (similar units)
2. Randomize treatments **within each block**
3. Compare treatments using **within-block contrasts**

---

### 🔸 Example: Quick Bread Volume (Two Plants)

- Factor of interest: Oven temperature  
- Nuisance factor: Manufacturing plant (A vs B)  
- Blocks: Plants  
- Treatments: Four temperatures  

Each plant receives **all treatments exactly once**.

---

### 🔸 Statistical Model for RCBD

\[
y = \mu + \text{Treatment Effect} + \text{Block Effect} + \varepsilon
\]

- Treatment effects → effects of scientific interest  
- Block effects → variability we want to remove  
- Errors assumed independent \(N(0, \sigma^2)\)

---

### 🔸 Why Blocking Works

If plant B consistently produces higher volumes than plant A:
- A completely randomized design inflates error variance
- A blocked design *explains* this variation explicitly

➡️ **Higher power, tighter confidence intervals**

---

## 🔹 Nested Designs

### 📌 When Do Nested Factors Occur?

A factor is **nested** within another when its levels are **unique** to a higher-level factor.

---

### 🔸 Example: Operators within Plants

- Operators 1–3 → Plant 1  
- Operators 4–6 → Plant 2  
- Operators 7–9 → Plant 3  

Operators do **not** cross plants.

---

### 🔸 Key Consequences

- Operator effects cannot be separated from plant context
- No interaction estimation between nested and parent factors
- Common in industrial, biological, and organizational studies

---

### 🔸 Crossed–Nested Designs

Some factors may be crossed, others nested.

**Example**
- SPC (Yes/No): crossed with plants and operators  
- Operators: nested within plants  

These hybrid designs are common and powerful but require careful modeling.

---

## 🔹 Repeated Measures Designs

### 📌 Core Idea
The **same experimental unit receives multiple treatments**.

This increases efficiency by removing between-unit variability.

---

### 🔸 Example: Taste Testing (Sweetener Levels)

- Subjects rate **low, medium, high** sweetness
- Each subject acts as their **own block**

This dramatically reduces noise caused by individual taste differences.

---

### 🔸 Statistical Implication

Responses from the same subject are **correlated**.
Models must account for:
- Subject effects
- Within-subject dependence

---

## 🔹 Split-Plot (Repeated Measures + Between-Subjects)

Some treatments apply to **subjects**, others to **measurements within subjects**.

---

### 🔸 Example: Sweetness × Perceived Wholesomeness

- Wholesomeness: applied to consumers (between-subjects)
- Sweetness: applied within each consumer

This creates **two experimental units**:
- Consumers → wholesomeness comparisons
- Tastings → sweetness comparisons

⚠️ Different error terms apply to different effects.

---

## 🔹 Incomplete Block Designs

### 📌 Why Needed?

Sometimes blocks cannot accommodate all treatments.

**Example**
- Five products
- Consumers can taste only three

Each consumer becomes an **incomplete block**.

---

### 🔸 Balanced Incomplete Block Designs (BIBDs)

A BIBD ensures:
- Every pair of treatments appears together equally often
- Fair within-block comparisons

This preserves comparability while respecting practical limits.

---

### 🔸 Key Benefit

Removes block-to-block heterogeneity **without requiring full replication**.

---

## 🔹 Two-Level Factorial Designs

### 📌 Motivation
The number of treatments explodes with many factors.

| Factors | Levels | Treatments |
|-------|--------|-----------|
| 3 | 3 | 27 |
| 3 | 2 | 8 |

Two-level designs are ideal for **screening**.

---

### 🔸 Strengths

- Efficient
- Excellent for identifying important main effects
- Detect low-order interactions

Used extensively in:
- Process optimization
- Industrial experimentation
- Early-stage scientific studies

---

## 🔹 Fractional Factorial Designs

### 📌 When Full Factorials Are Too Large

Instead of all \(2^k\) combinations, use a **carefully chosen subset**.

---

### 🔸 Tradeoff

| Gain | Cost |
|----|-----|
| Fewer runs | Some effects become aliased |
| Lower cost | Higher-order interactions sacrificed |

Assumption:
> High-order interactions are negligible.

---

## 🔹 Response Surface Experiments

### 📌 When Two Levels Are Not Enough

Use when:
- All factors are quantitative
- Curvature is expected
- Optimization is the goal

---

### 🔸 Model Assumption

The true response can be approximated by a **second-order polynomial**:

\[
y = \beta_0 + \sum \beta_i x_i + \sum \beta_{ii} x_i^2 + \sum \beta_{ij} x_ix_j + \varepsilon
\]

---

### 🔸 What You Gain

- Locate maxima or minima
- Visualize response surfaces
- Fine-tune factor settings

Common tools:
- Contour plots
- 3D surface plots
- Conditional effect plots

---

## 🔑 Big Picture Summary

| Design Feature | Purpose |
|---------------|--------|
| Blocking | Reduce nuisance variability |
| Nesting | Reflect hierarchical structure |
| Repeated measures | Remove subject-to-subject noise |
| Factorial designs | Study interactions efficiently |
| Fractional factorials | Screen many factors cheaply |
| Response surfaces | Optimize quantitatively |

---

> **Good experimental design is about asking the right question with the right structure.**  
> Statistics then becomes the language that answers it.



### 15.4 Design of Observational Studies (Page 666)
Below is a **clean, structured, high-yield explanation** of the material you pasted, written as **study notes** you can directly use. I’ll keep it conceptual, comparative, and exam/interview ready.

---

# 📘 Observational Studies — Structured Notes

## 🔹 What Are Observational Studies?

Observational studies differ fundamentally from **experimental studies** because:

* **No random assignment** of treatments/exposures to units
* Researchers **observe** existing conditions rather than manipulate them
* **Causality cannot be directly established**
* They establish **association**, not cause–effect

To argue causality:

* Confounders must be identified
* Subgroup analysis, matching, or regression adjustment is required
* Still weaker than randomized experiments

> ❗ Experiments → causation
> ❗ Observational studies → association

---

## 🔹 Goals of Observational Studies

1. Describe relationships between variables
2. Generate hypotheses for causal mechanisms
3. Study situations where experiments are unethical, impractical, or impossible

---

## 🔹 Types of Observational Studies

Observational studies are commonly classified into **three major types**:

| Type            | Time Direction   | Question Answered        |
| --------------- | ---------------- | ------------------------ |
| Cross-sectional | Present          | *What is happening now?* |
| Prospective     | Forward in time  | *What will happen?*      |
| Retrospective   | Backward in time | *What has happened?*     |

---

## 🔹 Cross-Sectional Studies

### 📌 Definition

* Measurements taken **at a single point in time**
* Exposure and outcome observed **simultaneously**
* Provides a **snapshot** of population characteristics

### 📌 Characteristics

* No temporal ordering → weak for causality
* Useful for:

  * Descriptive analysis
  * Group comparisons
  * Prevalence estimation

### 📌 Examples

* Household income by zip code
* Road traffic volume vs road characteristics
* Health surveys

### 📌 Stratification

* **Pre-stratified**: groups defined *before* sampling
* **Post-stratified**: groups formed *after* data collection

### 📌 Analysis Methods

* ANOVA
* Regression
* Group comparisons

---

## 🔹 Prospective Observational Studies (Cohort Studies)

### 📌 Definition

* Groups formed **based on exposure**
* Outcomes observed **in the future**
* Treatment precedes response

### 📌 Key Question

> “What is going to happen?”

### 📌 Characteristics

* Stronger causal suggestion than cross-sectional
* Still no randomization → confounding possible
* Often large and time-consuming

### 📌 Examples

* Teaching workshop attendance → later teaching effectiveness
* Estrogen therapy → heart disease outcomes

### 📌 Analysis Methods

* Regression
* ANOVA
* Survival analysis

---

## 🔹 Retrospective Observational Studies

### 📌 Definition

* Groups formed **based on outcome**
* Past exposure is examined
* Time direction is reversed

### 📌 Key Question

> “What has happened?”

### 📌 Characteristics

* Efficient for **rare outcomes**
* Vulnerable to **bias**
* Often cheaper and faster than prospective studies

### 📌 Examples

* Lung cancer patients vs non-patients → smoking history
* Manufacturing failures → historical process conditions
* Surgical survival studies

### 📌 Terminology

* **Cases**: subjects with outcome
* **Controls**: subjects without outcome

### 📌 Bias Risks

* Recall bias (memory-based histories)
* Selection bias

### 📌 Archival Studies

* Use existing records
* Less recall bias
* Common in manufacturing and medical databases

---

## 🔹 Comparison: Prospective vs Retrospective

| Aspect        | Prospective        | Retrospective      |
| ------------- | ------------------ | ------------------ |
| Direction     | Exposure → Outcome | Outcome → Exposure |
| Cost          | High               | Low                |
| Time          | Long               | Short              |
| Bias          | Lower recall bias  | Higher recall bias |
| Rare outcomes | Inefficient        | Efficient          |

---

## 🔹 Matching in Observational Studies

### 📌 Why Matching?

* No randomization → confounding
* Matching reduces **variance** and **bias**
* Analogous to **blocking** in experiments

### 📌 Example

Teaching effectiveness study:

* Match faculty who attended workshop
* With similar age, gender, department, prior performance

Each matched pair acts like a **block of size 2**

---

## 🔹 Matching Methods

### 1️⃣ Within-Class Matching

* Categorical confounders (e.g., gender)
* Match if same category

### 2️⃣ Multi-Factor Matching

* Match on multiple categorical variables
* E.g., gender + department

### 3️⃣ Categorized Matching

* Continuous variable → binned
* E.g., age groups, test score ranges

### 4️⃣ Caliper (Interval) Matching

* Match if:
  [
  |x_1 - x_2| \leq \text{caliper}
  ]
* Example: age difference ≤ 5 years
* Trade-off:

  * Small caliper → fewer matches
  * Large caliper → more bias

### 5️⃣ Other Methods

* Nearest-neighbor matching
* Mean balancing

---

## 🔹 Matching vs Covariance Analysis

| Approach            | When Used      |
| ------------------- | -------------- |
| Matching            | Design stage   |
| ANCOVA / regression | Analysis stage |

* Regression adjusts for confounders statistically
* Matching removes imbalance structurally

---

## 🔹 Key Takeaways

* Observational studies ≠ experiments
* Association ≠ causation
* Temporal ordering strengthens causal claims
* Retrospective studies excel for rare events
* Matching is critical for variance reduction
* Regression adjustment complements design-stage controls

---




### 15.5 Case Study: Paired-Comparison Experiment (Page 669)

## 🔹 What Is a Paired-Comparison Design?

A **paired-comparison (matched-pairs) design** is:

* The **simplest randomized complete block design (RCBD)**
* Exactly **two treatments**
* **Block size = 2**
* Each block contains **one observation per treatment**
* Often implemented as:

  * **Repeated measures** (same subject receives both treatments)
  * **Matched observational study** (matched subjects)

> Key idea: **Control nuisance variability by comparing treatments within the same block**

---

## 🔹 Why Use Paired Comparisons?

When:

* Experimental units differ substantially
* Between-unit variability is large
* Within-unit comparisons are more precise

Blocking (or matching) **removes subject-to-subject variability** from the error term, dramatically increasing statistical power.

---

## 🔹 Case Study Overview — Skin Sensitivity Experiment

### 🎯 Objective

Determine whether a **new allergen formulation** reduces skin sensitivity compared to the standard allergen.

### 🧪 Treatments

* Control allergen
* Experimental allergen

### 🧍 Blocks

* **Subjects**
* Each subject receives **both treatments**

### 🧬 Experimental Units

* **Arms of the subjects**
* One arm gets control, the other gets experimental

### 🎲 Randomization

* For each subject, treatment assignment to **left/right arm is randomized**

---

## 🔹 Why This Is a Powerful Design

| Feature                   | Benefit                              |
| ------------------------- | ------------------------------------ |
| Same subject              | Eliminates inter-subject variability |
| Randomized arm assignment | Prevents systematic bias             |
| Paired structure          | Enables within-subject comparison    |

---

## 🔹 Response Variable

* **Skin sensitivity**
* Measured as **diameter (cm)** of redness around injection site

---

## 🔹 Visual Evidence (Slope Plot)

In the summary plot:

* Each line connects the two arms of a subject
* **Negative slope** ⇒ experimental < control
* Majority of slopes are negative → strong visual evidence of reduction

> Slope plots are diagnostic tools for paired designs.

---

## 🔹 Statistical Model

This is a **linear model with treatment + block effects**:

[
Y_{ij} = \beta_0 + \beta_1 X_{i1} + \sum_{j=2}^{20} \beta_j X_{ij} + \varepsilon_{ij}
]

### Components Explained

#### Treatment Indicator

[
X_{i1} =
\begin{cases}
1 & \text{experimental treatment} \
0 & \text{control}
\end{cases}
]

#### Block (Subject) Indicators

[
X_{ij} =
\begin{cases}
1 & \text{if response from subject } j-1 \
0 & \text{otherwise}
\end{cases}
\quad j = 2, \dots, 20
]

#### Parameters

* (\beta_1): **treatment effect**
* (\beta_2, \dots, \beta_{20}): **subject effects**
* (\varepsilon_{ij} \sim N(0, \sigma^2))

---

## 🔹 Hypothesis Test (Primary Interest)

Dermatologists allowed for **increase or decrease**, so a **two-sided test**:

[
H_0: \beta_1 = 0 \
H_a: \beta_1 \neq 0
]

---

## 🔹 Results Interpretation

### Estimated Treatment Effect

[
\hat{\beta}_1 = -0.1915
]

➡ Experimental allergen **reduces redness by ~0.19 cm on average**

---

### Test Statistic

[
t^* = -17.10
]

Critical value:
[
t(0.975, 19) = 2.093
]

Decision:
[
|t^*| \gg 2.093 \Rightarrow \text{Reject } H_0
]

### ✅ Conclusion

The experimental allergen **significantly reduces skin sensitivity**.

---

## 🔹 Role of Blocking (Subjects)

* Investigators **expected** large subject-to-subject variation
* Subject effects were **not of scientific interest**
* Blocking was used purely to **reduce error variance**

> This is a critical philosophical point:
> **You block on nuisance factors, not because you care about them, but because you want them out of the way.**

---

## 🔹 Was Blocking Effective?

Yes.

### Hypothesis for Block Effects

[
H_0: \beta_2 = \beta_3 = \dots = \beta_{20} = 0
]

Result:

* Blocking **significantly reduced residual variance**
* Precision of treatment estimate improved

---

## 🔹 Key Statistical Insight

This paired-comparison analysis is **mathematically equivalent** to:

* A **one-sample t-test** on **within-subject differences**
* Or a **matched-pairs t-test**

Where:
[
D_i = Y_{i,\text{exp}} - Y_{i,\text{control}}
]

This equivalence is foundational and reappears in:

* ANOVA
* Repeated measures
* Mixed models
* Causal inference

---

## 🔹 Design Classifications (All Apply!)

This single study is simultaneously:

| Design Type                          | Why                         |
| ------------------------------------ | --------------------------- |
| Randomized complete block            | Subjects = blocks           |
| Paired-comparison                    | Two treatments per block    |
| Repeated measures                    | Same subject measured twice |
| Matched observational (conceptually) | Within-unit comparison      |

---

## 🔹 Key Takeaways

* Paired designs **maximize power** when units are heterogeneous
* Randomization still matters (arm assignment)
* Treatment inference comes from **within-block differences**
* Regression, ANOVA, and paired t-tests are **unified by the linear model**
* Blocking is about **precision**, not hypothesis testing on blocks

---

## 🔹 What This Sets You Up For Next

This case study is the gateway to:

* ANOVA (categorical factors)
* Repeated measures models
* Mixed-effects models
* Matched observational causal studies
* Multiple comparison procedures

---

Below are **FAANG-style interview questions with strong, interview-ready answers** from **Chapter 15: Design of Experimental and Observational Studies**.

These are **conceptual + applied + edge-case questions** that show *statistical maturity*, not just textbook recall.

---

# 📘 FAANG Interview Q&A — Experimental & Observational Studies (Chapter 15)

## 🔹 Core Conceptual Questions

---

### **Q1. Why are randomized experiments considered the gold standard for causal inference?**

**Answer:**

Randomized experiments allow causal inference because **random assignment breaks the link between treatment and confounders**.

* Randomization ensures treatment groups are, *in expectation*, identical on:

  * observed confounders
  * unobserved confounders
* Any systematic difference in outcomes can therefore be attributed to the treatment.

Formally:
[
E[Y(1) - Y(0)] \text{ is identifiable because } T \perp !!! \perp (Y(1), Y(0))
]

Observational studies lack this independence.

---

### **Q2. Why can’t observational studies prove causation?**

**Answer:**

Because **treatment is not randomly assigned**, so:

* Confounding variables may affect both exposure and outcome
* Direction of causality may be unclear
* Reverse causation is possible

Even with regression adjustment:

* Only *measured* confounders are controlled
* Unmeasured confounding remains

Thus observational studies establish **association**, not causation.

---

### **Q3. Give examples where observational studies are preferable to experiments.**

**Answer:**

Observational studies are preferred when experiments are:

* **Unethical** (e.g., smoking exposure)
* **Impractical** (long-term disease incidence)
* **Too costly** or infeasible

Examples:

* Studying rare diseases (retrospective case-control)
* Studying long-term medication effects
* Analyzing production failures using historical logs

---

## 🔹 Observational Study Types

---

### **Q4. Differentiate cross-sectional, prospective, and retrospective studies.**

| Study Type      | Time Direction | Groups Defined By | Key Question        |
| --------------- | -------------- | ----------------- | ------------------- |
| Cross-sectional | Same time      | Exposure          | “What exists now?”  |
| Prospective     | Forward        | Exposure          | “What will happen?” |
| Retrospective   | Backward       | Outcome           | “What happened?”    |

---

### **Q5. Why are retrospective studies efficient for rare outcomes?**

**Answer:**

Because:

* You **start with cases** (rare outcome already observed)
* No need to follow a large population over time
* Dramatically reduces required sample size

Example:

* Lung cancer → compare smokers vs non-smokers retrospectively

This is why epidemiology relies heavily on **case-control designs**.

---

### **Q6. What is recall bias and where does it arise?**

**Answer:**

Recall bias occurs when:

* Subjects reconstruct past exposures from memory
* Cases remember exposures differently than controls

Most common in:

* Retrospective, non-archival studies

Archival studies (medical records, logs) reduce recall bias.

---

## 🔹 Blocking, Matching, and Variance Reduction

---

### **Q7. What is blocking and why is it used?**

**Answer:**

Blocking groups **similar experimental units** together to:

* Remove nuisance variability
* Reduce experimental error variance
* Increase precision of treatment comparisons

Blocking does **not** introduce bias—it improves efficiency.

---

### **Q8. Why can’t we technically block in observational studies?**

**Answer:**

Because:

* Treatments are not assigned
* You cannot control treatment allocation within blocks

Instead, observational studies use **matching**, which is:

* Conceptually analogous to blocking
* Applied at the design stage

---

### **Q9. Explain matching and its purpose.**

**Answer:**

Matching pairs treated and untreated units with similar confounders.

Goal:
[
\text{Reduce } Var(\hat{\tau}) \text{ by eliminating confounder imbalance}
]

After matching:

* Comparisons mimic within-block experimental comparisons
* Remaining differences are attributed to treatment + residual confounding

---

### **Q10. What are common matching methods?**

**Answer:**

| Method                | Description              |         |     |
| --------------------- | ------------------------ | ------- | --- |
| Within-class matching | Exact categorical match  |         |     |
| Coarsened matching    | Continuous → categorical |         |     |
| Caliper matching      |                          | X₁ − X₂ | < δ |
| Nearest-neighbor      | Closest match            |         |     |
| Mean balancing        | Match distributions      |         |     |

Tradeoff: **precision vs sample size**

---

## 🔹 Paired-Comparison / Matched-Pairs Designs

---

### **Q11. What is a paired-comparison design?**

**Answer:**

A paired-comparison design:

* Has **two treatments**
* Uses **blocks of size two**
* Often uses the **same subject** for both treatments

It is simultaneously:

* Randomized complete block design
* Repeated measures design
* Matched design

---

### **Q12. Why are paired designs more powerful than independent designs?**

**Answer:**

Because they eliminate between-unit variability.

Let:
[
Y_{ij} = \mu + \tau_i + b_j + \varepsilon_{ij}
]

Subtracting within block:
[
Y_{1j} - Y_{2j} = \tau_1 - \tau_2 + (\varepsilon_{1j} - \varepsilon_{2j})
]

Block effect cancels → smaller variance.

---

### **Q13. What statistical test is equivalent to a paired-comparison regression?**

**Answer:**

A **paired t-test**.

Regression with block indicators and treatment indicator:
[
Y = \beta_0 + \beta_1 T + \text{block dummies} + \varepsilon
]

is algebraically equivalent to:

* One-sample t-test on within-pair differences

---

### **Q14. When would a paired design be inappropriate?**

**Answer:**

* Carryover effects (learning, fatigue)
* Treatment permanently alters subject
* Order effects not controllable
* Subject dropouts after first treatment

In such cases, use:

* Parallel group designs
* Washout periods
* Mixed-effects models

---

## 🔹 Factorial and Advanced Designs

---

### **Q15. Why are two-level factorial designs popular in screening experiments?**

**Answer:**

Because they:

* Minimize number of runs
* Estimate main effects efficiently
* Identify key drivers quickly

For (k) factors:
[
2^k \text{ instead of } 3^k \text{ or more}
]

Used before response surface optimization.

---

### **Q16. What is a fractional factorial design?**

**Answer:**

A fractional factorial:

* Uses a **subset** of full factorial runs
* Sacrifices information on higher-order interactions
* Retains main effects and low-order interactions

Assumption:

> Higher-order interactions are negligible

---

### **Q17. What problem do response surface designs solve?**

**Answer:**

They model **curvature** and find **optima**.

Two-level designs:

* Capture linear trends only

Response surface designs:

* Fit second-order models
* Identify maxima/minima
* Enable contour and surface plots

---

## 🔹 Meta / FAANG-Style Judgment Questions

---

### **Q18. How would you decide between experimentation and observation in industry?**

**Answer:**

Decision criteria:

* Can I randomize?
* Is intervention ethical?
* Is speed or certainty more important?
* What is the cost of wrong inference?

FAANG preference:

* Experiment when possible
* Observe when necessary
* Validate with multiple methods

---

### **Q19. How does ANOVA relate to regression in these designs?**

**Answer:**

They are the **same linear model**.

* ANOVA = regression with categorical predictors
* Balanced designs → ANOVA is simpler
* Unbalanced designs → regression preferred

---

### **Q20. What signals “statistical maturity” in experimental design answers?**

**Answer:**

Mentioning:

* Confounding vs variance
* Design before analysis
* Blocking vs adjustment
* Identifiability vs estimation
* Tradeoffs, not absolutes

---

## 🔥 Final FAANG Tip

Interviewers don’t want formulas.

They want:

* **Why** the design works
* **When** it fails
* **What tradeoffs** exist
* **How you’d decide in practice**

---




