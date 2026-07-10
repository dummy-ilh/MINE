# Partial Dependence Plots (PDP) Explained

Partial Dependence Plots (PDPs) are a powerful **model-agnostic** visualization tool that helps us understand how a single feature (or a pair of features) affects the predictions of a machine learning model.

***

## What PDPs Show
- PDPs illustrate the **average predicted output** of the model as we vary the value of one feature while keeping all other features constant.
- This helps uncover the overall relationship between a feature and the target prediction.
- You can see if the relationship is **linear, monotonic, non-linear, or more complex**.
- They also reveal the **direction and strength** of influence—a rising curve indicates a positive relationship and a falling curve indicates a negative one.

## How PDPs Work (Step-by-Step)
1. Select the feature of interest.
2. For a range of values of this feature:
   - Replace the feature values in your dataset with the fixed value.
   - Use the model to predict the output for each modified instance.
3. Average the predicted outputs over all instances for each fixed feature value.
4. Plot these average predictions against the feature values.

This averaging marginalizes out effects of all other features.

## What PDPs Tell You
- **Feature Effect:** How changing the feature affects predictions on average.
- **Feature Importance:** Helps highlight which features have strong effects (steep slopes) on predictions.
- **Interaction Effects:** Two-way PDPs show how the joint values of two features influence predictions and if interactions exist.

## Limitations
- PDP assumes features are independent; correlated features can produce misleading plots because unrealistic data combinations are averaged.

## Practical Uses
- Interpreting complex, black-box models (e.g., random forests, gradient boosting, neural networks).
- Exploring both classification (e.g., probability curves) and regression models.
- Complementing global feature importance scores by showing the feature’s detailed relationship with predictions.

***
### Mathematical Explanation and Numerical Example of Partial Dependence Plots (PDP)

***

#### Mathematical Formula

Given a machine learning model $$ f $$ that predicts the output based on a feature vector $$\mathbf{X} = (X_1, X_2, \ldots, X_p) $$, we split the features into:

- $$ X_S $$: The feature(s) of interest (e.g., $$ X_j $$)
- $$ X_C $$: The complement, i.e., all other features not in $$ S $$

The **Partial Dependence function** for $$ X_S = \mathbf{x}_S $$ is defined as the expected model output over the marginal distribution of $$ X_C $$, i.e.:

$$
\text{PD}_S(\mathbf{x}_S) = \mathbb{E}_{X_C}[f(\mathbf{x}_S, X_C)] = \int f(\mathbf{x}_S, \mathbf{x}_C) \, dP(\mathbf{x}_C)
$$

In practice, this is estimated by averaging over the training data instances:

$$
\hat{\text{PD}}_S(\mathbf{x}_S) = \frac{1}{n} \sum_{i=1}^n f(\mathbf{x}_S, \mathbf{x}_C^{(i)})
$$

where $$ \mathbf{x}_C^{(i)} $$ are the observed values of the other features for each instance $$ i $$.

***

#### Numerical Example

Assume a simple dataset with two features $$ X_1 $$ and $$ X_2 $$, and a model $$ f $$ predicting the target $$ y $$.

| Instance | $$ X_1 $$ | $$ X_2 $$ | $$ y $$ (true) |
|----------|-----------|-----------|----------------|
| 1        | 1         | 5         | 20             |
| 2        | 2         | 6         | 22             |
| 3        | 3         | 7         | 24             |

The trained model predicts:

$$
f(X_1, X_2) = 10 + 3 \times X_1 + 2 \times X_2
$$

***

##### Calculate PDP for $$ X_1 = 2 $$:

1. Fix $$ X_1 = 2 $$ for all instances.
2. For each original instance, keep $$ X_2 $$ values and predict:

- For instance 1: $$ f(2, 5) = 10 + 3 \times 2 + 2 \times 5 = 10 + 6 + 10 = 26 $$
- For instance 2: $$ f(2, 6) = 10 + 6 + 12 = 28 $$
- For instance 3: $$ f(2, 7) = 10 + 6 + 14 = 30 $$

3. Average predicted values:

$$
\text{PD}_{X_1}(2) = \frac{26 + 28 + 30}{3} = \frac{84}{3} = 28
$$

***

##### Calculate PDP for $$ X_1 = 3 $$:

1. Fix $$ X_1 = 3 $$ for all instances.
2. Predictions:

- $$ f(3, 5) = 10 + 9 + 10 = 29 $$
- $$ f(3, 6) = 10 + 9 + 12 = 31 $$
- $$ f(3, 7) = 10 + 9 + 14 = 33 $$

3. Average:

$$
\text{PD}_{X_1}(3) = \frac{29 + 31 + 33}{3} = \frac{93}{3} = 31
$$

***

#### Interpretation

The PDP values for $$ X_1 $$ at points 2 and 3 are 28 and 31, respectively, showing on average how predictions increase as $$ X_1 $$ moves from 2 to 3, **marginalizing over $$ X_2 $$**.

Plotting these points (and others) creates the PDP curve reflecting $$ X_1 $$'s effect on the model output.

***

In summary, **Partial Dependence Plots average model predictions over the dataset for fixed values of chosen features, revealing the global relationship between those features and the predicted outcome**. This helps visualize if a feature affects predictions linearly, non-linearly, or not at all.### Mathematical Explanation and Numerical Example of Partial Dependence Plot (PDP)

#### Mathematical Formulation:

Given a model prediction function $$ f(\mathbf{X}) $$ where $$\mathbf{X} = (X_1, X_2, ..., X_p) $$ are the features, we split the features into:

- $$ X_S $$: The feature(s) of interest (e.g., $$ X_j $$)
- $$ X_C $$: The complementary features (all others)

The partial dependence function for $$ X_S = \mathbf{x}_S $$ is defined as:

$$
\text{PD}_S(\mathbf{x}_S) = \mathbb{E}_{X_C} [ f(\mathbf{x}_S, X_C) ] = \int f(\mathbf{x}_S, \mathbf{x}_C) dP(\mathbf{x}_C)
$$

In practice, it is estimated by averaging over the dataset of $$ n $$ instances:

$$
\hat{\text{PD}}_S(\mathbf{x}_S) = \frac{1}{n} \sum_{i=1}^n f(\mathbf{x}_S, \mathbf{x}_C^{(i)})
$$

where $$ \mathbf{x}_C^{(i)} $$ are observed values of other features for instance $$ i $$.

***

#### Numerical Example:

Suppose a model

$$
f(X_1, X_2) = 10 + 3 \times X_1 + 2 \times X_2
$$

with dataset instances:

| Instance | $$X_1$$ | $$X_2$$ |
|----------|---------|---------|
| 1        | 1       | 5       |
| 2        | 2       | 6       |
| 3        | 3       | 7       |

***

**Calculate PDP for $$ X_1 = 2 $$:**

- Fix $$ X_1=2 $$ and combine with original $$ X_2 $$ values:

$$
f(2,5)=10+6+10=26, \quad f(2,6)=10+6+12=28, \quad f(2,7)=10+6+14=30
$$

- Average prediction:

$$
\text{PD}_{X_1}(2) = \frac{26 + 28 + 30}{3} = 28
$$

***

**Calculate PDP for $$ X_1 = 3 $$:**

- Fix $$ X_1=3 $$:

$$
f(3,5)=10+9+10=29, \quad f(3,6)=10+9+12=31, \quad f(3,7)=10+9+14=33
$$

- Average:

$$
\text{PD}_{X_1}(3) = \frac{29 + 31 + 33}{3} = 31
$$

***

### Interpretation:
The PDP value increases from 28 to 31 as $$ X_1 $$ goes from 2 to 3, showing the average effect of changing $$ X_1 $$ on model predictions while marginalizing over $$ X_2 $$.

Plotting these points across a range of $$ X_1 $$ values forms the Partial Dependence Plot, which helps visualize the global influence of $$ X_1 $$ on model output.

This averaging approach also smooths out interactions from other features, revealing the marginal dependency of the chosen feature(s).
### Two-Feature Partial Dependence Plot (PDP) with Interaction Effects: Concept & Example

Partial Dependence Plots (PDPs) visualize the **marginal effect of one or two features** on a model's prediction. When considering two features together (2D PDP), we can observe **feature interaction effects**, i.e., how the combined values of both features influence predictions beyond their individual effects.

#### Mathematical Concept
Given features $$ X_j $$ and $$ X_k $$, the 2D partial dependence function is:

$$
PD_{jk}(x_j, x_k) = \mathbb{E}_{X_{-jk}}[f(x_j, x_k, X_{-jk})] = \frac{1}{N} \sum_{i=1}^N f(x_j, x_k, x_{-jk}^{(i)})
$$

where $$ X_{-jk} $$ are all other features except $$ j, k $$, and $$ N $$ is the number of data instances.

The shape of this 2D PDP surface reveals whether the two features interact: if the effect of one feature changes depending on the value of the other, their interaction is strong.

### Numerical Example
Consider a simple model:

$$
f(X_1,X_2) = 5 + 2X_1 + 3X_2 + 4X_1X_2
$$

for which there is an interaction term $$4X_1X_2$$.

| Instance | $$ X_1 $$ | $$ X_2 $$ |
| -------- | --------- | --------- |
| 1        | 1         | 10        |
| 2        | 2         | 20        |
| 3        | 3         | 30        |

##### Step 1: Fix $$X_1=2$$ and vary $$X_2$$ over observed values:
- $$f(2,10) = 5 + 4 + 30 + 80 = 119$$
- $$f(2,20) = 5 + 4 + 60 + 160 = 229$$
- $$f(2,30) = 5 + 4 + 90 + 240 = 339$$

Average over these:\
$$PD_{12}(2, x_2) = \frac{119 + 229 + 339}{3} = 229$$

##### Step 2: Fix $$X_2=20$$ and vary $$X_1$$:
- $$f(1,20) = 5 + 2 + 60 + 80 = 147$$
- $$f(2,20) = 5 + 4 + 60 + 160 = 229$$
- $$f(3,20) = 5 + 6 + 60 + 240 = 311$$

Average over these:\
$$PD_{12}(x_1,20) = \frac{147 + 229 + 311}{3} = 229$$

##### Step 3: Calculate combined PDP at $$X_1=2, X_2=20$$:
$$f(2,20) = 5 + 4 + 60 + 160 = 229$$

#### Interpretation
- The 2D PDP surface varies significantly because of the $$4X_1X_2$$ interaction term.
- The combined effect of $$X_1$$ and $$X_2$$ is **not additive** — changing one modifies the impact of the other.
- Plotting these averages in a heatmap or contour plot highlights areas where predictions are high (e.g., large $$X_1$$ and $$X_2$$) due to this interaction.

Partial Dependence Plots (PDPs) visually show how a feature (or pair of features) affects a machine learning model’s predictions on average.

### Common Types of PDP Visualizations

1. **One-Way PDP (Single Feature)**
   - **Plot:** X-axis is the feature value, Y-axis is average predicted outcome.
   - **Interpretation:** Shows the marginal effect of this feature on the prediction, averaging out other features.
   - Example: Plot of temperature vs predicted demand, showing demand increases with temperature.

2. **Two-Way PDP (Feature Interaction)**
   - **Plot:** 3D surface or contour plot with two features on the X and Y axes, predicted values as height or color.
   - **Interpretation:** Reveals how combinations of the two features influence predictions and whether they interact.
   - Example: Plot showing bike rentals influenced jointly by temperature and humidity.

3. **Comparison Across Groups**
   - PDPs can be drawn separately for subgroups (e.g., male vs female) to observe differences.

### Types of Interpretation from PDPs
- **Monotonic relationship:** Increasing feature value always increases/decreases prediction.
- **Non-linear relationship:** Feature effect changes non-linearly (e.g., rises then falls).
- **Interaction presence:** Non-additive effects in two-way PDP indicating that one feature's impact depends on the other's value.

### How to Read PDPs
- Look for slope and shape of curves.
- Identify ranges where feature matters more or less.
- Use as a complement to global importance metrics like permutation importance.

Partial Dependence Plots (PDPs) are a core concept in machine learning model interpretability. Here are some conceptual interview questions and detailed answers regarding PDPs.

***

## 1. What is a Partial Dependence Plot (PDP)?

**Answer:** A **Partial Dependence Plot (PDP)** is a model-agnostic visualization technique that shows the marginal effect one or two features have on the predicted outcome of a machine learning model. It illustrates the average relationship between the feature(s) of interest and the model's prediction, essentially answering the question: "How does the predicted outcome change when we vary the value of this feature, while averaging out the effects of all other features?"

**Key Takeaway:** PDPs help visualize the **global** relationship a feature has with the predicted target, even for complex "black-box" models.

***

## 2. How is a PDP calculated conceptually?

**Answer:** The PDP is calculated after the model has been trained. For a single feature of interest, say $X_s$, the calculation involves the following conceptual steps:

1.  **Select a set of values** for the feature of interest $X_s$ (e.g., all unique values or a grid of values for continuous features).
2.  For each value $z$ in the selected set:
    * **Impute/Substitute:** Replace the value of $X_s$ with $z$ for *every observation* in the dataset. All other features ($X_c$, the complement set) retain their original values.
    * **Predict:** Use the trained model to make a prediction for every modified observation.
    * **Average:** Calculate the average of all these predictions. This average prediction is the partial dependence value for the feature value $z$.
3.  **Plot:** The PDP is a line graph where the **feature value** ($z$) is on the x-axis and the **average predicted outcome** is on the y-axis.

**Mathematically (for a model $\hat{f}$ and feature set $X_s$):**
$$\hat{f}_{s}(x_{s}) = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(x_{s}, x_{c}^{(i)})$$
where $n$ is the number of observations, $x_{c}^{(i)}$ is the complement feature vector for the $i^{th}$ instance, and $\hat{f}(x_{s}, x_{c}^{(i)})$ is the prediction for the $i^{th}$ instance with $X_s$ fixed at $x_s$.

***

## 3. What are the main benefits of using PDPs?

**Answer:**

* **Model Interpretability:** They provide a straightforward, graphical way to understand the marginal effect of individual features on model output, making "black-box" models like Random Forests or Neural Networks more transparent.
* **Model Debugging/Sanity Check:** They allow data scientists to verify if the model is learning expected relationships (e.g., house price increasing with size) or if it has captured counter-intuitive or nonsensical patterns, potentially indicating data leakage or poor feature engineering.
* **Model Agnostic:** They can be applied to *any* machine learning model, regardless of its underlying complexity or structure.

***

## 4. What is the main limitation of PDPs, and how might it lead to misleading results?

**Answer:**

The main limitation is the **assumption of feature independence**.

* **Problem:** The PDP calculation works by substituting a feature value $z$ across *all* instances, even those where that value is highly unlikely or impossible given the values of other, correlated features. For example, in a model predicting income, the PDP might show the effect of a very low **age** and a very high **years of experience** combination, even though these features are highly correlated and that combination is impossible in the real world.
* **Misleading Results:** When features are highly correlated, the PDP averages model predictions over these *unrealistic* or *extrapolated* data points. This can make the resulting average relationship on the plot appear smooth and simple, potentially **masking complex interactions** or presenting a relationship that doesn't accurately reflect the true conditional impact of the feature on the prediction in realistic scenarios.

***

## 5. How do Individual Conditional Expectation (ICE) plots relate to PDPs?

**Answer:**

* **ICE Plots (Individual Conditional Expectation):** An ICE plot shows how the predicted outcome changes for a **single instance** as the feature of interest varies. It plots a separate line for every instance in the dataset.
* **Relationship to PDP:** The **PDP is simply the average of all the ICE curves.**
* **Advantage of ICE over PDP:** If the feature of interest interacts with other features, the individual ICE lines will vary in shape and spread out. This variation reveals the presence of an **interaction effect**, which is completely masked by the single, averaged line of the PDP. PDPs are great for a global view, but ICE plots are better for identifying **heterogeneity** in the feature-prediction relationship.

***

## 6. How would you use a PDP to detect an interaction effect between two features?

**Answer:** A standard one-way PDP (for a single feature) **cannot** detect an interaction because it averages out the effects of all other features.

To detect an interaction, you must use a **two-way PDP** (also called a 2D PDP).

* **Method:** A two-way PDP plots the average predicted outcome as two features, say $X_1$ and $X_2$, are varied simultaneously. This is usually visualized as a **heatmap** or a 3D surface plot.
* **Interpretation:**
    * **No Interaction:** If the effect of $X_1$ on the prediction is the *same* regardless of the value of $X_2$, there is no interaction. The contours of the heatmap would appear in parallel lines.
    * **Interaction:** If the effect of $X_1$ on the prediction *depends* on the value of $X_2$ (i.e., the lines are non-parallel or the heatmap patterns are non-uniform), then a non-linear interaction is present.

***

## 7. What does a flat line on a PDP signify?

**Answer:** A flat, horizontal line on a PDP indicates that the feature of interest **has little to no average marginal effect** on the model's prediction, across all its values.

* **Significance:** This suggests that the model is either not using this feature or that the effect of the feature on the prediction is highly dependent on the values of other features (i.e., a strong interaction effect is present, which is masked by the averaging).
* **Action:** If a feature has a flat PDP, it might be a candidate for removal (since it doesn't contribute much to the global prediction) or it might prompt a follow-up with ICE plots or two-way PDPs to see if its impact is hidden within a conditional/interaction effect.

