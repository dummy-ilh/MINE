# Collecting Engineering Data


## 🔷 1. Three Ways to Collect Data (Deep Understanding)

### 1️⃣Retrospective Study (Using Historical Data)
**Idea**
You look backward at already existing data.

* No control over variables
* No control over how data was collected

**Example (Engineering)**
* Analyze past machine failure logs
* Study website traffic logs
* Look at past medical records

**Example:**
You analyze last 5 years of server downtime logs to identify patterns.

**Limitations**
* Missing variables
* Bias in data collection
* Cannot establish causation
*  Only correlation, not cause-effect

### 2️⃣ Observational Study
**Idea**
You observe the system in real time but do not intervene.

**Example**
* Observe how users interact with an app
* Monitor traffic flow at a junction
* Study patient outcomes without assigning treatment

**Example:**
You observe two groups of users: those who use dark mode vs light mode — but you don’t assign them.

**Limitation**
* Confounding variables
*  Maybe dark mode users are younger → affects results

### Designed Experiment (MOST POWERFUL)
**Idea**
You actively control variables and assign treatments.

** Example**
* Test 3 algorithms under controlled input
* Run A/B testing on a website
* Manufacturing: test temperature levels on output quality

** Example:**
You randomly assign users to version A or B of a feature.

** Key Advantage**
*  Can establish causality

---

##  2. What are “Factors”?
This is a core concept in Design of Experiments (DOE).

** Definition**
A factor is any variable that affects the outcome (response).

** Examples**
* ** Manufacturing:** Temperature, Pressure, Machine speed
* ** Software / ML:** Learning rate, Batch size, Model architecture
* ** Product Experiment:** UI design, Button color, Recommendation algorithm

** Factor Levels**
Each factor has levels (values it can take).
*  **Example:** Temperature → 100°C, 150°C, 200°C; Button color → Red, Blue, Green

** Response Variable**
What you measure.
*  **Example:** Output quality, Conversion rate, Accuracy

---

##  3. Designed Experiment — Key Principles
Montgomery emphasizes 3 pillars:

### 1️⃣ Randomization
** Idea**
Assign treatments randomly.

** Why?**
Removes bias and unknown effects.

**Example**
* **Instead of:** First 100 users → Version A; Next 100 → Version B ❌
* **Do:** Randomly assign each user to A or B ✅

** Without Randomization**
* Time effects
* User behavior patterns
* Hidden bias

### ️ Replication
** Idea**
Repeat experiment multiple times.

** Why?**
Reduces noise and improves reliability.

**Example**
* **Test algorithm performance on:** 1 dataset ❌; 20 datasets ✅

### 3️⃣ Blocking
** Idea**
Control known sources of variation.

** Example**
Suppose machines differ slightly.
 **Block by machine:** Compare treatments within same machine.

**Real Example**
Website test: Block by device type (mobile vs desktop).

---

##  4. Putting It All Together (Clear Comparison)

| Method | Control | Causality | Example |
| :--- | :--- | :--- | :--- |
| **Retrospective** | ❌ None | ❌ No | Analyze past logs |
| **Observational** | ❌ None | ❌ Weak | Observe user behavior |
| **Designed Experiment** | ✅ Full | ✅ Strong | A/B testing |

---

##  5. Real-World Engineering Scenario

**Problem:** Improve battery life of a device.

* ** Retrospective:** Look at past battery data → Limited insights.
* ** Observational:** Observe how users use devices → Confounding variables.
* ** Designed Experiment:** Control CPU frequency, Screen brightness, and Background apps. Measure battery life. 👉 This gives true cause-effect.

---

##  6. Common Pitfalls (VERY IMPORTANT)

* ** Mistake 1:** Confusing observation with experiment.
* ** Mistake 2:** Ignoring randomization.
* ** Mistake 3:** Changing multiple factors blindly (leads to confounding).
* ** Mistake 4:** Too few samples (no replication).

** Final Insight (Professor Level)**
* Retrospective → “What happened?”
* Observational → “What is happening?”
* Designed Experiment → “What causes what?”

# Core Terms in Designed Experiments

## 1. Control
### Idea
“Control” means keeping conditions stable so the effect of a factor can be isolated.

### Two Meanings of Control
**(A) Control Group**
A baseline group that does not receive the treatment.
*Example (A/B Testing):*
* Group A → Old UI (control group)
* Group B → New UI (treatment group)
* Comparison is made against the control group.

**(B) Controlled Variables**
Variables that are held constant during the experiment.
*Example (Battery Experiment):*
* Factor being tested: CPU frequency
* Keep constant: Screen brightness, Background apps, Network usage
* These are controlled variables.

## 2. Treatment
### Idea
A treatment is a specific combination of factor levels.
*Example:*
* Factors: Temperature → {100, 150}; Pressure → {10, 20}
* Treatments: (100, 10), (100, 20), (150, 10), (150, 20)
* Each combination is one treatment.

## 3. Experimental Unit
### Idea
The smallest unit to which a treatment is applied.
*Examples:*
* A user (in A/B testing)
* A machine component
* A patient

## 4. Response Variable
### Idea
The output that is measured.
*Examples:*
* Conversion rate
* Model accuracy
* Battery life
* Product quality

## 5. Factors (Recap)
### Idea
Variables that influence the response.
*Examples:*
* Temperature
* Learning rate
* UI design

## 6. Factor Levels
### Idea
The values that a factor can take.
*Examples:*
* Temperature → 100°C, 150°C
* Button color → Red, Blue

## Summary Table

| Concept | Meaning | Example |
| :--- | :--- | :--- |
| **Factor** | Input variable | Temperature |
| **Level** | Value of a factor | 100°C |
| **Treatment** | Combination of levels | (100°C, 10 psi) |
| **Experimental Unit** | Where treatment is applied | Machine part |
| **Response Variable** | Output measured | Quality |
| **Control Group** | Baseline group | Old UI |
| **Controlled Variables** | Held constant | Brightness |

## End-to-End Example
### Problem
Improve website conversion rate

### Setup
* **Factor:** Button color
* **Levels:** Red, Blue

### Design
* Randomly assign users to groups
* Keep other variables constant (layout, speed, content)

### Groups
* **Control group** → Existing button color
* **Treatment group** → New button color

### Response
* Conversion rate

### Goal
Determine whether button color causes a change in conversion.
