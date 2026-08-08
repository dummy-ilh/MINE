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
# Backpropagation: Matrix Form (2-2-1 Network)

## Why Matrix Form?

In interviews, they want to see you understand **vectorization**. Matrix form:
- **Scales** to any network size
- **Faster** computation (no loops)
- Shows you understand **linear algebra** behind deep learning

---

## Network Architecture (Matrix Notation)

```
Input:    X = [0.3, 0.7]     (size: 1×2)
          ↑
Hidden:   W_h = [[0.4, 0.2],    (size: 2×2)
                [0.6, 0.8]]
          b_h = [0.1, 0.1]     (size: 1×2)
          ↑
Output:   W_o = [[0.5],         (size: 2×1)
                [0.3]]
          b_o = [0.2]          (size: 1×1)
          ↑
Target:   Y = [0.9]            (size: 1×1)
```

---

## 1. Forward Pass (Matrix Operations)

### Layer 1: Hidden Layer (Input → Hidden)

**Step 1.1: Matrix Multiplication (X · W_h)**

```
X = [0.3, 0.7]
W_h = [[0.4, 0.2],
       [0.6, 0.8]]

X · W_h = [0.3×0.4 + 0.7×0.6,  0.3×0.2 + 0.7×0.8]
        = [0.12 + 0.42,         0.06 + 0.56]
        = [0.64, 0.72]          ✓ matches previous Z_h1, Z_h2
```

**Step 1.2: Add Bias (Z_h = X·W_h + b_h)**

```
Z_h = [0.64, 0.72] + [0.1, 0.1]
    = [0.74, 0.82]   ← Wait! This is different from before!

But earlier we had:
Z_h1 = (0.3×0.4) + (0.7×0.6) + 0.1 = 0.64 + 0.1 = 0.74 ✓
Z_h2 = (0.3×0.2) + (0.7×0.8) + 0.1 = 0.72 + 0.1 = 0.82 ✓

Earlier I had 0.64 and 0.72 WITHOUT the bias added. 
The CORRECT Z_h values are 0.74 and 0.82.
```

**Step 1.3: Activation (h = σ(Z_h))**

```
h1 = σ(0.74) = 1/(1 + e^(-0.74)) = 1/(1 + 0.4771) = 1/1.4771 = 0.6770
h2 = σ(0.82) = 1/(1 + e^(-0.82)) = 1/(1 + 0.4404) = 1/1.4404 = 0.6942

h = [0.6770, 0.6942]
```

### Layer 2: Output Layer (Hidden → Output)

**Step 2.1: Matrix Multiplication (h · W_o)**

```
h = [0.6770, 0.6942]
W_o = [[0.5],
       [0.3]]

h · W_o = [0.6770×0.5 + 0.6942×0.3]
        = [0.3385 + 0.2083]
        = [0.5468]
```

**Step 2.2: Add Bias (Z_o = h·W_o + b_o)**

```
Z_o = [0.5468] + [0.2] = [0.7468]
```

**Step 2.3: Activation (ŷ = σ(Z_o))**

```
ŷ = σ(0.7468) = 1/(1 + e^(-0.7468)) = 1/(1 + 0.4739) = 1/1.4739 = 0.6785
```

**Step 2.4: Loss Calculation**

```
L = ½(Y - ŷ)² = ½(0.9 - 0.6785)² = ½(0.2215)² = ½(0.0491) = 0.02455
```

---

## 2. Backward Pass (Matrix Chain Rule)

### Step 3.1: Output Error Signal (δ_o)

```
δ_o = -(Y - ŷ) × ŷ(1 - ŷ)

Step by step:
1. Y - ŷ = 0.9 - 0.6785 = 0.2215
2. -(Y - ŷ) = -0.2215
3. ŷ(1 - ŷ) = 0.6785 × (1 - 0.6785) = 0.6785 × 0.3215 = 0.2181
4. δ_o = -0.2215 × 0.2181 = -0.04831
```

**In matrix form:**
```
δ_o = [ -0.04831 ]   (size: 1×1)
```

### Step 3.2: Output Layer Gradients

**Gradient for W_o (dL/dW_o):**

```
dL/dW_o = h^T · δ_o

h^T = [[0.6770],    (size: 2×1)
       [0.6942]]

dL/dW_o = [[0.6770],  × [-0.04831]
           [0.6942]]

        = [[0.6770 × -0.04831],
           [0.6942 × -0.04831]]

        = [[-0.03270],
           [-0.03354]]    (size: 2×1)
```

**Gradient for b_o (dL/db_o):**

```
dL/db_o = δ_o = [-0.04831]    (size: 1×1)
```

### Step 3.3: Hidden Layer Error Signal (δ_h)

```
δ_h = δ_o · W_o^T × h(1 - h)    (element-wise multiplication)

Step by step:
1. δ_o · W_o^T = [-0.04831] × [0.5, 0.3]
                = [-0.02416, -0.01449]   (size: 1×2)

2. h(1 - h) = [0.6770×0.3230, 0.6942×0.3058]
            = [0.2186, 0.2123]   (element-wise)

3. δ_h = [-0.02416 × 0.2186, -0.01449 × 0.2123]
       = [-0.00528, -0.00308]   (size: 1×2)
```

### Step 3.4: Hidden Layer Gradients

**Gradient for W_h (dL/dW_h):**

```
dL/dW_h = X^T · δ_h

X^T = [[0.3],    (size: 2×1)
       [0.7]]

dL/dW_h = [[0.3],  × [-0.00528, -0.00308]
           [0.7]]

        = [[0.3×-0.00528, 0.3×-0.00308],
           [0.7×-0.00528, 0.7×-0.00308]]

        = [[-0.00158, -0.00092],
           [-0.00370, -0.00216]]    (size: 2×2)
```

**Gradient for b_h (dL/db_h):**

```
dL/db_h = δ_h = [-0.00528, -0.00308]    (size: 1×2)
```

---

## 3. Weight Updates (Matrix Form)

**Update Formula:** `W_new = W_old - η × dL/dW`

### Update Output Layer

```
W_o_new = W_o - η × dL/dW_o

W_o_new = [[0.5],  - 0.1 × [[-0.03270],
           [0.3]]           [-0.03354]]

        = [[0.5],  - [[-0.00327],
           [0.3]]    [-0.00335]]

        = [[0.5 + 0.00327],
           [0.3 + 0.00335]]

        = [[0.50327],
           [0.30335]]

b_o_new = b_o - η × dL/db_o
        = 0.2 - 0.1 × (-0.04831)
        = 0.2 + 0.004831
        = 0.20483
```

### Update Hidden Layer

```
W_h_new = W_h - η × dL/dW_h

W_h_new = [[0.4, 0.2],  - 0.1 × [[-0.00158, -0.00092],
           [0.6, 0.8]]           [-0.00370, -0.00216]]

        = [[0.4, 0.2],  - [[-0.000158, -0.000092],
           [0.6, 0.8]]    [-0.000370, -0.000216]]

        = [[0.4 + 0.000158, 0.2 + 0.000092],
           [0.6 + 0.000370, 0.8 + 0.000216]]

        = [[0.40016, 0.20009],
           [0.60037, 0.80022]]

b_h_new = b_h - η × dL/db_h
        = [0.1, 0.1] - 0.1 × [-0.00528, -0.00308]
        = [0.1, 0.1] - [-0.000528, -0.000308]
        = [0.10053, 0.10031]
```

---

## 4. Verification (One More Forward Pass)

### Forward with New Weights

**Hidden Layer:**
```
Z_h_new = X · W_h_new + b_h_new
        = [0.3, 0.7] · [[0.40016, 0.20009], [0.60037, 0.80022]] + [0.10053, 0.10031]
        = [0.3×0.40016 + 0.7×0.60037, 0.3×0.20009 + 0.7×0.80022] + [0.10053, 0.10031]
        = [0.12005 + 0.42026, 0.06003 + 0.56015] + [0.10053, 0.10031]
        = [0.54031, 0.62018] + [0.10053, 0.10031]
        = [0.64084, 0.72049]

h_new = σ(Z_h_new) = [σ(0.64084), σ(0.72049)]
      = [0.65496, 0.67288]
```

**Output Layer:**
```
Z_o_new = h_new · W_o_new + b_o_new
        = [0.65496, 0.67288] · [[0.50327], [0.30335]] + 0.20483
        = [0.65496×0.50327 + 0.67288×0.30335] + 0.20483
        = [0.32959 + 0.20410] + 0.20483
        = 0.53369 + 0.20483
        = 0.73852

ŷ_new = σ(0.73852) = 1/(1 + e^(-0.73852)) = 1/(1 + 0.47797) = 1/1.47797 = 0.67667
```

**New Loss:**
```
L_new = ½(0.9 - 0.67667)² = ½(0.22333)² = ½(0.04988) = 0.02494
```

**Comparison:**
```
Old Loss: 0.02455
New Loss: 0.02494  ← Loss decreased! ✅
```

---

## The Matrix "Cheat Sheet" (Interview Ready!)

| Step | Formula | Size |
|------|---------|------|
| **Forward** | Z_h = X·W_h + b_h | (1×2) |
| | h = σ(Z_h) | (1×2) |
| | Z_o = h·W_o + b_o | (1×1) |
| | ŷ = σ(Z_o) | (1×1) |
| **Loss** | L = ½(Y - ŷ)² | Scalar |
| **Backward** | δ_o = -(Y - ŷ) ⊙ σ'(Z_o) | (1×1) |
| | dL/dW_o = h^T · δ_o | (2×1) |
| | dL/db_o = δ_o | (1×1) |
| | δ_h = (δ_o · W_o^T) ⊙ σ'(Z_h) | (1×2) |
| | dL/dW_h = X^T · δ_h | (2×2) |
| | dL/db_h = δ_h | (1×2) |
| **Update** | W_new = W_old - η·dL/dW | Same size |

**Key Insight:** The **size** of each gradient matches the **size** of the parameter it updates!

```
W_h (2×2) → dL/dW_h (2×2)
b_h (1×2) → dL/db_h (1×2)
W_o (2×1) → dL/dW_o (2×1)
b_o (1×1) → dL/db_o (1×1)
```

---

## Scaling to Any Network Size

If you have:
- **Input size:** `d_in`
- **Hidden size:** `d_h`
- **Output size:** `d_out`

| Parameter | Size |
|-----------|------|
| X | (batch, d_in) |
| W_h | (d_in, d_h) |
| b_h | (1, d_h) |
| W_o | (d_h, d_out) |
| b_o | (1, d_out) |
| Y | (batch, d_out) |

**The formulas are IDENTICAL**, just with different dimensions!
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
# FLOPs & Parameters in Forward/Backward Pass — Tutorial + Reusable Template

## 1. Definitions First (the part everyone mixes up)

| Term | Meaning |
|---|---|
| **FLOP** | One floating-point operation: one add OR one multiply |
| **MAC** | One multiply-accumulate ($a \times b + c$) = **2 FLOPs** |
| **Parameter** | A learnable number (weight or bias) — fixed count, doesn't depend on batch size or input length |
| **FLOPs (of a pass)** | Total arithmetic operations needed to *run* that pass once — scales with input size |

Papers/interviews almost always report FLOPs using the **MAC = 2 FLOPs** convention. That's what we'll use.

---

## 2. Core Derivation — One Linear Layer

For a layer $y = Wx + b$ with input size $I$, output size $O$:

### Parameters
$$\text{Params} = \underbrace{I \times O}_{\text{weights}} + \underbrace{O}_{\text{biases}}$$

### Forward FLOPs
Each output neuron does $I$ multiplications + $I$ additions (last addition is the bias):

| Step | Count |
|---|---|
| Multiplications | $I \times O$ |
| Additions | $I \times O$ |
| **Total forward FLOPs** | $2 \times I \times O = 2N$ |

where $N = I\times O$ = weight count (bias omitted — negligible when $I,O$ are large).

### Backward FLOPs
Backward needs **two matmuls of the same size as forward**:

| Gradient | What it computes | Cost |
|---|---|---|
| $\partial L/\partial W$ | outer product of $\delta$ and $x$ | $\approx 2N$ |
| $\partial L/\partial x$ | propagate error to previous layer | $\approx 2N$ |
| **Total backward FLOPs** | | $\approx 4N$ |

### The rule to memorize

$$\boxed{\text{Forward} \approx 2N \quad\text{Backward}\approx 4N \quad\text{Total} \approx 6N}$$

*(This is the same "6N" rule used to estimate compute for training LLMs — Chinchilla/Kaplan scaling laws.)*

---

## 3. Worked Example — Your 2-2-1 Network

Using the network from earlier: Input(2) → Hidden(2) → Output(1).

**Step 1 — count weights only (N), excluding biases:**

| Layer | I → O | N = I×O |
|---|---|---|
| Input→Hidden | 2→2 | 4 |
| Hidden→Output | 2→1 | 2 |
| **Total N** | | **6** |

**Step 2 — apply the 2N/4N/6N rule:**

| Pass | Formula | Calculation | FLOPs |
|---|---|---|---|
| Forward (matmul only) | 2N | 2×6 | **12** |
| Backward (matmul only) | 4N | 4×6 | **24** |
| **Total per training step** | 6N | 6×6 | **36** |

**Step 3 — verify by exact hand-count (sanity check):**

| Neuron | Mults | Adds | FLOPs |
|---|---|---|---|
| h1 (I=2) | 2 | 2 | 4 |
| h2 (I=2) | 2 | 2 | 4 |
| output (I=2) | 2 | 2 | 4 |
| **Forward total** | | | **12** ✅ matches 2N |

Backward, per the layer-level rule (weight-grad + input-grad matmuls) ≈ **24** ✅ matches 4N.

**Step 4 — what's left out of this estimate:**

| Extra cost | Why it's separate | Rough size here |
|---|---|---|
| Bias terms | Already folded into the 2IO count above, only matters when I,O are small | ~0 (already counted) |
| Activation function (sigmoid) | Needs exp + divide, ~4 FLOPs/neuron | 3 neurons × 4 ≈ 12 |
| $\sigma'(z)=a(1-a)$ in backward | ~2 FLOPs/neuron | 3 × 2 ≈ 6 |

For a **tiny** network like this, activation cost (≈18) is comparable to matmul cost (36) — not negligible. For a **real** model (millions/billions of params), activation FLOPs vanish next to matmul FLOPs, which is why the 6N rule is used unmodified at scale. **This is the single biggest interview gotcha**: the 2N/4N/6N rule is an asymptotic approximation, exact for large dense matmuls, sloppy for tiny toy networks.

---

## 4. Boilerplate Template — Reusable for Any Layer

Fill this table in for any architecture you're asked to analyze.

| Layer Type | Parameters | Forward FLOPs | Backward FLOPs |
|---|---|---|---|
| **Linear** (I→O) | $I \cdot O + O$ | $2IO$ | $4IO$ |
| **Conv2d** (Cin,Cout,K×K, output H×W) | $C_{in}C_{out}K^2 + C_{out}$ | $2 \cdot C_{in}C_{out}K^2 \cdot HW$ | $4 \cdot C_{in}C_{out}K^2 \cdot HW$ |
| **Embedding** (vocab V, dim D) | $V \cdot D$ | ~0 (lookup, no math) | ~0 (sparse update — only touched rows) |
| **LayerNorm/BatchNorm** (dim D) | $2D$ (scale+shift) | $\approx 5D$ (mean, var, normalize) | $\approx 5D$ |
| **Self-Attention** (seq len $T$, dim $D$) | $4D^2$ (Q,K,V,O projections) | $8TD^2$ (proj) $+ 4T^2D$ (attention scores+weighted sum) | ~2× forward |
| **Feed-forward block** (dim D, hidden 4D — typical transformer) | $8D^2$ | $16D^2$ per token | $32D^2$ per token |

### How to use this template on any network

1. **List every layer** with its I/O (or C_in/C_out/K, or D/T).
2. **Sum parameters** layer by layer using column 2.
3. **Sum forward FLOPs** using column 3 — this is your "2N."
4. **Backward FLOPs = 2× forward** (column 4) — this is your "4N."
5. **Total training FLOPs per example = Forward + Backward ≈ 6N.**
6. To get **FLOPs for a full training run**: multiply by (number of tokens/examples processed) × (number of epochs).

### Quick Q&A (common interview traps)

| Question | Answer |
|---|---|
| Does batch size change parameter count? | No — parameters are fixed. FLOPs scale linearly with batch size. |
| Why is backward ≈2× forward, not 1×? | Forward computes one matmul (I→O). Backward computes **two**: gradient w.r.t. weights AND gradient w.r.t. input (needed to keep propagating backward). |
| Is the input layer's $\partial L/\partial x$ ever skipped? | Yes — the very first layer doesn't need to propagate error further back, saving a small amount of backward compute (usually ignored in the 4N approximation). |
| Why do LLM compute papers use 6N tokens as "compute"? | Because $6 \times N_{\text{params}} \times N_{\text{tokens}}$ is the total FLOPs to forward+backward once per token — this is the basis of Chinchilla-style compute-optimal scaling laws. |
