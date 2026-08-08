Understood — here is the **fully expanded version**, with every arithmetic operation shown as its own step (no combined formulas, no skipped multiplications).

# Backpropagation Walkthrough: 2-2-1 Network (Fully Expanded)

## Setup

| Parameter | Value |
|---|---|
| X1, X2 | 0.3, 0.7 |
| Y (target) | 0.9 |
| η (learning rate) | 0.1 |
| Activation | Sigmoid: $\sigma(z)=\dfrac{1}{1+e^{-z}}$, $\sigma'(z)=\sigma(z)(1-\sigma(z))$ |
| Loss | $L=\tfrac12(Y-\hat Y)^2$ |

*(Note: your labels say "[2-2-2-1]" but only one hidden layer's weights were given, so this is a 2-2-1 network. `b1=0.1` is treated as shared by both hidden neurons.)*

---

## 1. Forward Propagation

### 1.1 Hidden neuron h1

**Step 1 — weighted sum, term by term:**

| Term | Calculation | Value |
|---|---|---|
| $W_{11}\cdot X_1$ | 0.4 × 0.3 | 0.12 |
| $W_{21}\cdot X_2$ | 0.6 × 0.7 | 0.42 |
| $+\,b_1$ | + 0.1 | 0.1 |
| **Sum $Z_{h1}$** | 0.12 + 0.42 + 0.1 | **0.64** |

**Step 2 — apply sigmoid, term by term:**

| Sub-step | Calculation | Value |
|---|---|---|
| $-Z_{h1}$ | −0.64 | −0.64 |
| $e^{-0.64}$ | (Euler's number to this power) | 0.527292 |
| $1+e^{-Z_{h1}}$ | 1 + 0.527292 | 1.527292 |
| $a_{h1}=1/(1.527292)$ | division | **0.654737** |

### 1.2 Hidden neuron h2

**Step 1 — weighted sum:**

| Term | Calculation | Value |
|---|---|---|
| $W_{12}\cdot X_1$ | 0.2 × 0.3 | 0.06 |
| $W_{22}\cdot X_2$ | 0.8 × 0.7 | 0.56 |
| $+\,b_1$ | + 0.1 | 0.1 |
| **Sum $Z_{h2}$** | 0.06 + 0.56 + 0.1 | **0.72** |

**Step 2 — apply sigmoid:**

| Sub-step | Calculation | Value |
|---|---|---|
| $-Z_{h2}$ | −0.72 | −0.72 |
| $e^{-0.72}$ | — | 0.486752 |
| $1+e^{-Z_{h2}}$ | 1 + 0.486752 | 1.486752 |
| $a_{h2}=1/(1.486752)$ | division | **0.672642** |

### 1.3 Output neuron

**Step 1 — weighted sum, term by term:**

| Term | Calculation | Value |
|---|---|---|
| $W_{1o}\cdot a_{h1}$ | 0.5 × 0.654737 | 0.327369 |
| $W_{2o}\cdot a_{h2}$ | 0.3 × 0.672642 | 0.201793 |
| $+\,b_2$ | + 0.2 | 0.2 |
| **Sum $Z_o$** | 0.327369 + 0.201793 + 0.2 | **0.729161** |

**Step 2 — apply sigmoid:**

| Sub-step | Calculation | Value |
|---|---|---|
| $-Z_o$ | −0.729161 | −0.729161 |
| $e^{-0.729161}$ | — | 0.482318 |
| $1+e^{-Z_o}$ | 1 + 0.482318 | 1.482318 |
| $a_o = \hat Y = 1/(1.482318)$ | division | **0.674616** |

**Forward pass result: ŷ = 0.674616** (target Y = 0.9)

---

## 2. Loss Calculation

| Step | Calculation | Value |
|---|---|---|
| $Y-\hat Y$ | 0.9 − 0.674616 | 0.225384 |
| $(Y-\hat Y)^2$ | 0.225384 × 0.225384 | 0.050798 |
| $L=\tfrac12(Y-\hat Y)^2$ | 0.5 × 0.050798 | **0.025399** |

---

## 3. Backward Propagation

This is the chain rule applied one link at a time. For every weight we need:
$$\frac{\partial L}{\partial W}=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial Z}\cdot\frac{\partial Z}{\partial W}$$
Each of these three factors is computed **separately**, then multiplied in sequence — nothing is combined into a shortcut formula.

### 3.1 Output layer — δ_o

**Factor 1: $\dfrac{\partial L}{\partial a_o}$**

Since $L=\tfrac12(Y-a_o)^2$, its derivative w.r.t. $a_o$ is $-(Y-a_o) = a_o - Y$.

| Step | Calculation | Value |
|---|---|---|
| $a_o - Y$ | 0.674616 − 0.9 | **−0.225384** |

**Factor 2: $\dfrac{\partial a_o}{\partial Z_o} = a_o(1-a_o)$**

| Step | Calculation | Value |
|---|---|---|
| $1-a_o$ | 1 − 0.674616 | 0.325384 |
| $a_o\times(1-a_o)$ | 0.674616 × 0.325384 | **0.219513** |

**Multiply Factor 1 × Factor 2 → δ_o**

| Step | Calculation | Value |
|---|---|---|
| $\delta_o=\dfrac{\partial L}{\partial a_o}\times\dfrac{\partial a_o}{\partial Z_o}$ | −0.225384 × 0.219513 | **−0.049474** |

### 3.2 Gradients for output-layer weights

Now multiply δ_o by **Factor 3** ($\partial Z_o/\partial W$) individually for each weight.

**W1o:** $\partial Z_o/\partial W_{1o} = a_{h1}$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{1o}}=\delta_o \times a_{h1}$ | −0.049474 × 0.654737 | **−0.032389** |

**W2o:** $\partial Z_o/\partial W_{2o} = a_{h2}$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{2o}}=\delta_o \times a_{h2}$ | −0.049474 × 0.672642 | **−0.033281** |

**b2:** $\partial Z_o/\partial b_2 = 1$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial b_2}=\delta_o \times 1$ | −0.049474 × 1 | **−0.049474** |

### 3.3 Hidden layer — δ_h1 and δ_h2

Error must first be pushed backward from the output neuron to each hidden neuron's **activation**, before converting to that neuron's **pre-activation** δ.

**For h1:**

**Factor 1: $\dfrac{\partial L}{\partial a_{h1}} = \delta_o \times W_{1o}$** (chain rule through the output neuron)

| Step | Calculation | Value |
|---|---|---|
| $\delta_o \times W_{1o}$ | −0.049474 × 0.5 | **−0.024737** |

**Factor 2: $\dfrac{\partial a_{h1}}{\partial Z_{h1}} = a_{h1}(1-a_{h1})$**

| Step | Calculation | Value |
|---|---|---|
| $1-a_{h1}$ | 1 − 0.654737 | 0.345263 |
| $a_{h1}\times(1-a_{h1})$ | 0.654737 × 0.345263 | **0.226031** |

**Multiply → δ_h1**

| Step | Calculation | Value |
|---|---|---|
| $\delta_{h1}=$ Factor1 × Factor2 | −0.024737 × 0.226031 | **−0.005591** |

**For h2:**

**Factor 1: $\dfrac{\partial L}{\partial a_{h2}} = \delta_o \times W_{2o}$**

| Step | Calculation | Value |
|---|---|---|
| $\delta_o \times W_{2o}$ | −0.049474 × 0.3 | **−0.014842** |

**Factor 2: $\dfrac{\partial a_{h2}}{\partial Z_{h2}} = a_{h2}(1-a_{h2})$**

| Step | Calculation | Value |
|---|---|---|
| $1-a_{h2}$ | 1 − 0.672642 | 0.327358 |
| $a_{h2}\times(1-a_{h2})$ | 0.672642 × 0.327358 | **0.220229** |

**Multiply → δ_h2**

| Step | Calculation | Value |
|---|---|---|
| $\delta_{h2}=$ Factor1 × Factor2 | −0.014842 × 0.220229 | **−0.003269** |

### 3.4 Gradients for hidden-layer weights

Multiply each δ_h by **Factor 3** ($\partial Z_h/\partial W$ = the relevant input) individually.

**W11:** $\partial Z_{h1}/\partial W_{11} = X_1$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{11}}=\delta_{h1}\times X_1$ | −0.005591 × 0.3 | **−0.001677** |

**W21:** $\partial Z_{h1}/\partial W_{21} = X_2$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{21}}=\delta_{h1}\times X_2$ | −0.005591 × 0.7 | **−0.003914** |

**W12:** $\partial Z_{h2}/\partial W_{12} = X_1$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{12}}=\delta_{h2}\times X_1$ | −0.003269 × 0.3 | **−0.000981** |

**W22:** $\partial Z_{h2}/\partial W_{22} = X_2$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{22}}=\delta_{h2}\times X_2$ | −0.003269 × 0.7 | **−0.002288** |

**b1 (shared by both hidden neurons):** contributions from both paths are summed

| Step | Calculation | Value |
|---|---|---|
| $\delta_{h1}\times 1$ | −0.005591 | −0.005591 |
| $\delta_{h2}\times 1$ | −0.003269 | −0.003269 |
| $\dfrac{\partial L}{\partial b_1}=$ sum | −0.005591 + (−0.003269) | **−0.008860** |

---

## 4. Parameter Updates

Rule applied individually to every parameter: $W_{new}=W_{old}-\eta\times\text{gradient}$

**Output layer:**

| Weight | $W_{old}$ | gradient | $\eta\times$grad | $W_{old}-\eta\times$grad | $W_{new}$ |
|---|---|---|---|---|---|
| W1o | 0.5 | −0.032389 | 0.1×(−0.032389)=−0.003239 | 0.5−(−0.003239) | **0.503239** |
| W2o | 0.3 | −0.033281 | 0.1×(−0.033281)=−0.003328 | 0.3−(−0.003328) | **0.303328** |
| b2 | 0.2 | −0.049474 | 0.1×(−0.049474)=−0.004947 | 0.2−(−0.004947) | **0.204947** |

**Hidden layer:**

| Weight | $W_{old}$ | gradient | $\eta\times$grad | $W_{old}-\eta\times$grad | $W_{new}$ |
|---|---|---|---|---|---|
| W11 | 0.4 | −0.001677 | 0.1×(−0.001677)=−0.0001677 | 0.4−(−0.0001677) | **0.400168** |
| W21 | 0.6 | −0.003914 | 0.1×(−0.003914)=−0.0003914 | 0.6−(−0.0003914) | **0.600391** |
| W12 | 0.2 | −0.000981 | 0.1×(−0.000981)=−0.0000981 | 0.2−(−0.0000981) | **0.200098** |
| W22 | 0.8 | −0.002288 | 0.1×(−0.002288)=−0.0002288 | 0.8−(−0.0002288) | **0.800229** |
| b1 | 0.1 | −0.008860 | 0.1×(−0.008860)=−0.000886 | 0.1−(−0.000886) | **0.100886** |

---

## 5. Verify Loss Decreased — full re-run of Step 1 with new weights

### 5.1 Hidden neuron h1 (new)

| Term | Calculation | Value |
|---|---|---|
| $W_{11,new}\cdot X_1$ | 0.400168 × 0.3 | 0.120050 |
| $W_{21,new}\cdot X_2$ | 0.600391 × 0.7 | 0.420274 |
| $+\,b_{1,new}$ | + 0.100886 | 0.100886 |
| $Z_{h1,new}$ | sum | **0.641210** |

| Sub-step | Calculation | Value |
|---|---|---|
| $e^{-0.641210}$ | — | 0.526708 |
| $1+e^{-Z}$ | 1+0.526708 | 1.526708 |
| $a_{h1,new}$ | 1/1.526708 | **0.654994** |

### 5.2 Hidden neuron h2 (new)

| Term | Calculation | Value |
|---|---|---|
| $W_{12,new}\cdot X_1$ | 0.200098 × 0.3 | 0.060029 |
| $W_{22,new}\cdot X_2$ | 0.800229 × 0.7 | 0.560160 |
| $+\,b_{1,new}$ | + 0.100886 | 0.100886 |
| $Z_{h2,new}$ | sum | **0.721075** |

| Sub-step | Calculation | Value |
|---|---|---|
| $e^{-0.721075}$ | — | 0.486221 |
| $1+e^{-Z}$ | 1+0.486221 | 1.486221 |
| $a_{h2,new}$ | 1/1.486221 | **0.672882** |

### 5.3 Output neuron (new)

| Term | Calculation | Value |
|---|---|---|
| $W_{1o,new}\cdot a_{h1,new}$ | 0.503239 × 0.654994 | 0.329617 |
| $W_{2o,new}\cdot a_{h2,new}$ | 0.303328 × 0.672882 | 0.204076 |
| $+\,b_{2,new}$ | + 0.204947 | 0.204947 |
| $Z_{o,new}$ | sum | **0.738640** |

| Sub-step | Calculation | Value |
|---|---|---|
| $e^{-0.738640}$ | — | 0.477813 |
| $1+e^{-Z}$ | 1+0.477813 | 1.477813 |
| $\hat Y_{new}=a_{o,new}$ | 1/1.477813 | **0.676679** |

### 5.4 New loss

| Step | Calculation | Value |
|---|---|---|
| $Y-\hat Y_{new}$ | 0.9 − 0.676679 | 0.223321 |
| $(Y-\hat Y_{new})^2$ | 0.223321 × 0.223321 | 0.049872 |
| $L_{new}=\tfrac12(\cdot)$ | 0.5 × 0.049872 | **0.024936** |

### Comparison

| | Before | After | Change |
|---|---|---|---|
| ŷ | 0.674616 | 0.676679 | +0.002063 (moved toward Y=0.9) |
| **Loss** | **0.025399** | **0.024936** | **−0.000463 ✅** |

Understood — here is the **fully expanded version**, with every arithmetic operation shown as its own step (no combined formulas, no skipped multiplications).

# Backpropagation Walkthrough: 2-2-1 Network (Fully Expanded)

## Setup

| Parameter | Value |
|---|---|
| X1, X2 | 0.3, 0.7 |
| Y (target) | 0.9 |
| η (learning rate) | 0.1 |
| Activation | Sigmoid: $\sigma(z)=\dfrac{1}{1+e^{-z}}$, $\sigma'(z)=\sigma(z)(1-\sigma(z))$ |
| Loss | $L=\tfrac12(Y-\hat Y)^2$ |

*(Note: your labels say "[2-2-2-1]" but only one hidden layer's weights were given, so this is a 2-2-1 network. `b1=0.1` is treated as shared by both hidden neurons.)*

---

## 1. Forward Propagation

### 1.1 Hidden neuron h1

**Step 1 — weighted sum, term by term:**

| Term | Calculation | Value |
|---|---|---|
| $W_{11}\cdot X_1$ | 0.4 × 0.3 | 0.12 |
| $W_{21}\cdot X_2$ | 0.6 × 0.7 | 0.42 |
| $+\,b_1$ | + 0.1 | 0.1 |
| **Sum $Z_{h1}$** | 0.12 + 0.42 + 0.1 | **0.64** |

**Step 2 — apply sigmoid, term by term:**

| Sub-step | Calculation | Value |
|---|---|---|
| $-Z_{h1}$ | −0.64 | −0.64 |
| $e^{-0.64}$ | (Euler's number to this power) | 0.527292 |
| $1+e^{-Z_{h1}}$ | 1 + 0.527292 | 1.527292 |
| $a_{h1}=1/(1.527292)$ | division | **0.654737** |

### 1.2 Hidden neuron h2

**Step 1 — weighted sum:**

| Term | Calculation | Value |
|---|---|---|
| $W_{12}\cdot X_1$ | 0.2 × 0.3 | 0.06 |
| $W_{22}\cdot X_2$ | 0.8 × 0.7 | 0.56 |
| $+\,b_1$ | + 0.1 | 0.1 |
| **Sum $Z_{h2}$** | 0.06 + 0.56 + 0.1 | **0.72** |

**Step 2 — apply sigmoid:**

| Sub-step | Calculation | Value |
|---|---|---|
| $-Z_{h2}$ | −0.72 | −0.72 |
| $e^{-0.72}$ | — | 0.486752 |
| $1+e^{-Z_{h2}}$ | 1 + 0.486752 | 1.486752 |
| $a_{h2}=1/(1.486752)$ | division | **0.672642** |

### 1.3 Output neuron

**Step 1 — weighted sum, term by term:**

| Term | Calculation | Value |
|---|---|---|
| $W_{1o}\cdot a_{h1}$ | 0.5 × 0.654737 | 0.327369 |
| $W_{2o}\cdot a_{h2}$ | 0.3 × 0.672642 | 0.201793 |
| $+\,b_2$ | + 0.2 | 0.2 |
| **Sum $Z_o$** | 0.327369 + 0.201793 + 0.2 | **0.729161** |

**Step 2 — apply sigmoid:**

| Sub-step | Calculation | Value |
|---|---|---|
| $-Z_o$ | −0.729161 | −0.729161 |
| $e^{-0.729161}$ | — | 0.482318 |
| $1+e^{-Z_o}$ | 1 + 0.482318 | 1.482318 |
| $a_o = \hat Y = 1/(1.482318)$ | division | **0.674616** |

**Forward pass result: ŷ = 0.674616** (target Y = 0.9)

---

## 2. Loss Calculation

| Step | Calculation | Value |
|---|---|---|
| $Y-\hat Y$ | 0.9 − 0.674616 | 0.225384 |
| $(Y-\hat Y)^2$ | 0.225384 × 0.225384 | 0.050798 |
| $L=\tfrac12(Y-\hat Y)^2$ | 0.5 × 0.050798 | **0.025399** |

---

## 3. Backward Propagation

This is the chain rule applied one link at a time. For every weight we need:
$$\frac{\partial L}{\partial W}=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial Z}\cdot\frac{\partial Z}{\partial W}$$
Each of these three factors is computed **separately**, then multiplied in sequence — nothing is combined into a shortcut formula.

### 3.1 Output layer — δ_o

**Factor 1: $\dfrac{\partial L}{\partial a_o}$**

Since $L=\tfrac12(Y-a_o)^2$, its derivative w.r.t. $a_o$ is $-(Y-a_o) = a_o - Y$.

| Step | Calculation | Value |
|---|---|---|
| $a_o - Y$ | 0.674616 − 0.9 | **−0.225384** |

**Factor 2: $\dfrac{\partial a_o}{\partial Z_o} = a_o(1-a_o)$**

| Step | Calculation | Value |
|---|---|---|
| $1-a_o$ | 1 − 0.674616 | 0.325384 |
| $a_o\times(1-a_o)$ | 0.674616 × 0.325384 | **0.219513** |

**Multiply Factor 1 × Factor 2 → δ_o**

| Step | Calculation | Value |
|---|---|---|
| $\delta_o=\dfrac{\partial L}{\partial a_o}\times\dfrac{\partial a_o}{\partial Z_o}$ | −0.225384 × 0.219513 | **−0.049474** |

### 3.2 Gradients for output-layer weights

Now multiply δ_o by **Factor 3** ($\partial Z_o/\partial W$) individually for each weight.

**W1o:** $\partial Z_o/\partial W_{1o} = a_{h1}$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{1o}}=\delta_o \times a_{h1}$ | −0.049474 × 0.654737 | **−0.032389** |

**W2o:** $\partial Z_o/\partial W_{2o} = a_{h2}$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{2o}}=\delta_o \times a_{h2}$ | −0.049474 × 0.672642 | **−0.033281** |

**b2:** $\partial Z_o/\partial b_2 = 1$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial b_2}=\delta_o \times 1$ | −0.049474 × 1 | **−0.049474** |

### 3.3 Hidden layer — δ_h1 and δ_h2

Error must first be pushed backward from the output neuron to each hidden neuron's **activation**, before converting to that neuron's **pre-activation** δ.

**For h1:**

**Factor 1: $\dfrac{\partial L}{\partial a_{h1}} = \delta_o \times W_{1o}$** (chain rule through the output neuron)

| Step | Calculation | Value |
|---|---|---|
| $\delta_o \times W_{1o}$ | −0.049474 × 0.5 | **−0.024737** |

**Factor 2: $\dfrac{\partial a_{h1}}{\partial Z_{h1}} = a_{h1}(1-a_{h1})$**

| Step | Calculation | Value |
|---|---|---|
| $1-a_{h1}$ | 1 − 0.654737 | 0.345263 |
| $a_{h1}\times(1-a_{h1})$ | 0.654737 × 0.345263 | **0.226031** |

**Multiply → δ_h1**

| Step | Calculation | Value |
|---|---|---|
| $\delta_{h1}=$ Factor1 × Factor2 | −0.024737 × 0.226031 | **−0.005591** |

**For h2:**

**Factor 1: $\dfrac{\partial L}{\partial a_{h2}} = \delta_o \times W_{2o}$**

| Step | Calculation | Value |
|---|---|---|
| $\delta_o \times W_{2o}$ | −0.049474 × 0.3 | **−0.014842** |

**Factor 2: $\dfrac{\partial a_{h2}}{\partial Z_{h2}} = a_{h2}(1-a_{h2})$**

| Step | Calculation | Value |
|---|---|---|
| $1-a_{h2}$ | 1 − 0.672642 | 0.327358 |
| $a_{h2}\times(1-a_{h2})$ | 0.672642 × 0.327358 | **0.220229** |

**Multiply → δ_h2**

| Step | Calculation | Value |
|---|---|---|
| $\delta_{h2}=$ Factor1 × Factor2 | −0.014842 × 0.220229 | **−0.003269** |

### 3.4 Gradients for hidden-layer weights

Multiply each δ_h by **Factor 3** ($\partial Z_h/\partial W$ = the relevant input) individually.

**W11:** $\partial Z_{h1}/\partial W_{11} = X_1$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{11}}=\delta_{h1}\times X_1$ | −0.005591 × 0.3 | **−0.001677** |

**W21:** $\partial Z_{h1}/\partial W_{21} = X_2$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{21}}=\delta_{h1}\times X_2$ | −0.005591 × 0.7 | **−0.003914** |

**W12:** $\partial Z_{h2}/\partial W_{12} = X_1$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{12}}=\delta_{h2}\times X_1$ | −0.003269 × 0.3 | **−0.000981** |

**W22:** $\partial Z_{h2}/\partial W_{22} = X_2$

| Step | Calculation | Value |
|---|---|---|
| $\dfrac{\partial L}{\partial W_{22}}=\delta_{h2}\times X_2$ | −0.003269 × 0.7 | **−0.002288** |

**b1 (shared by both hidden neurons):** contributions from both paths are summed

| Step | Calculation | Value |
|---|---|---|
| $\delta_{h1}\times 1$ | −0.005591 | −0.005591 |
| $\delta_{h2}\times 1$ | −0.003269 | −0.003269 |
| $\dfrac{\partial L}{\partial b_1}=$ sum | −0.005591 + (−0.003269) | **−0.008860** |

---

## 4. Parameter Updates

Rule applied individually to every parameter: $W_{new}=W_{old}-\eta\times\text{gradient}$

**Output layer:**

| Weight | $W_{old}$ | gradient | $\eta\times$grad | $W_{old}-\eta\times$grad | $W_{new}$ |
|---|---|---|---|---|---|
| W1o | 0.5 | −0.032389 | 0.1×(−0.032389)=−0.003239 | 0.5−(−0.003239) | **0.503239** |
| W2o | 0.3 | −0.033281 | 0.1×(−0.033281)=−0.003328 | 0.3−(−0.003328) | **0.303328** |
| b2 | 0.2 | −0.049474 | 0.1×(−0.049474)=−0.004947 | 0.2−(−0.004947) | **0.204947** |

**Hidden layer:**

| Weight | $W_{old}$ | gradient | $\eta\times$grad | $W_{old}-\eta\times$grad | $W_{new}$ |
|---|---|---|---|---|---|
| W11 | 0.4 | −0.001677 | 0.1×(−0.001677)=−0.0001677 | 0.4−(−0.0001677) | **0.400168** |
| W21 | 0.6 | −0.003914 | 0.1×(−0.003914)=−0.0003914 | 0.6−(−0.0003914) | **0.600391** |
| W12 | 0.2 | −0.000981 | 0.1×(−0.000981)=−0.0000981 | 0.2−(−0.0000981) | **0.200098** |
| W22 | 0.8 | −0.002288 | 0.1×(−0.002288)=−0.0002288 | 0.8−(−0.0002288) | **0.800229** |
| b1 | 0.1 | −0.008860 | 0.1×(−0.008860)=−0.000886 | 0.1−(−0.000886) | **0.100886** |

---

## 5. Verify Loss Decreased — full re-run of Step 1 with new weights

### 5.1 Hidden neuron h1 (new)

| Term | Calculation | Value |
|---|---|---|
| $W_{11,new}\cdot X_1$ | 0.400168 × 0.3 | 0.120050 |
| $W_{21,new}\cdot X_2$ | 0.600391 × 0.7 | 0.420274 |
| $+\,b_{1,new}$ | + 0.100886 | 0.100886 |
| $Z_{h1,new}$ | sum | **0.641210** |

| Sub-step | Calculation | Value |
|---|---|---|
| $e^{-0.641210}$ | — | 0.526708 |
| $1+e^{-Z}$ | 1+0.526708 | 1.526708 |
| $a_{h1,new}$ | 1/1.526708 | **0.654994** |

### 5.2 Hidden neuron h2 (new)

| Term | Calculation | Value |
|---|---|---|
| $W_{12,new}\cdot X_1$ | 0.200098 × 0.3 | 0.060029 |
| $W_{22,new}\cdot X_2$ | 0.800229 × 0.7 | 0.560160 |
| $+\,b_{1,new}$ | + 0.100886 | 0.100886 |
| $Z_{h2,new}$ | sum | **0.721075** |

| Sub-step | Calculation | Value |
|---|---|---|
| $e^{-0.721075}$ | — | 0.486221 |
| $1+e^{-Z}$ | 1+0.486221 | 1.486221 |
| $a_{h2,new}$ | 1/1.486221 | **0.672882** |

### 5.3 Output neuron (new)

| Term | Calculation | Value |
|---|---|---|
| $W_{1o,new}\cdot a_{h1,new}$ | 0.503239 × 0.654994 | 0.329617 |
| $W_{2o,new}\cdot a_{h2,new}$ | 0.303328 × 0.672882 | 0.204076 |
| $+\,b_{2,new}$ | + 0.204947 | 0.204947 |
| $Z_{o,new}$ | sum | **0.738640** |

| Sub-step | Calculation | Value |
|---|---|---|
| $e^{-0.738640}$ | — | 0.477813 |
| $1+e^{-Z}$ | 1+0.477813 | 1.477813 |
| $\hat Y_{new}=a_{o,new}$ | 1/1.477813 | **0.676679** |

### 5.4 New loss

| Step | Calculation | Value |
|---|---|---|
| $Y-\hat Y_{new}$ | 0.9 − 0.676679 | 0.223321 |
| $(Y-\hat Y_{new})^2$ | 0.223321 × 0.223321 | 0.049872 |
| $L_{new}=\tfrac12(\cdot)$ | 0.5 × 0.049872 | **0.024936** |

### Comparison

| | Before | After | Change |
|---|---|---|---|
| ŷ | 0.674616 | 0.676679 | +0.002063 (moved toward Y=0.9) |
| **Loss** | **0.025399** | **0.024936** | **−0.000463 ✅** |

Loss decreased, confirming every gradient pointed the right direction. The step is small because η=0.1 and the gradients themselves are small (the output neuron isn't very "wrong" yet, and error shrinks further as it's split across two hidden paths) — this is expected single-step gradient descent behavior, not an error.
