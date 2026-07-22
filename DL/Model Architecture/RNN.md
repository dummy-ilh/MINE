Got it—we'll keep LSTMs and GRUs out for now and focus purely on **Vanilla Recurrent Neural Networks (RNNs)**.

By grounding ourselves deeply in the standard RNN first, you'll see exactly how it processes sequences, how backpropagation through time works under the hood, and the precise mathematical breakdown of *why* it fails on long sequences (the vanishing gradient problem).

Here is the updated roadmap focusing strictly on Vanilla RNNs for both time series and text:

---

## 📚 Vanilla RNN Mastery Curriculum

### Module 1: Core Fundamentals & Vanilla RNN Mechanics

* **Why Traditional Networks Fail:** Fixed inputs vs. sequential dependencies (why feedforward models struggle with order).
* **Unrolling the RNN:** The hidden state concept ($h_t$), recurrent connections, and weight sharing across time ($W_{hh}, W_{xh}, W_{hy}$).
* **The Forward Pass:** Step-by-step mathematical breakdown of calculating hidden states and outputs.
* **Vector & Matrix Shapes:** Getting comfortable with batch size, sequence length, and feature dimensions in memory.

---

### Module 2: Training Vanilla RNNs & Gradient Dynamics

* **Backpropagation Through Time (BPTT):** How gradients flow backward through unrolled time steps.
* **Truncated BPTT:** Managing memory constraints on long sequences.
* **The Vanishing & Exploding Gradient Problem:**
* Mathematical cause: Repeated matrix multiplication ($\mathbf{W}_{hh}^T$) during BPTT.
* Eigenvalues and exploding vs. vanishing regimes.


* **Mitigation Strategies:** Gradient clipping, specialized initializations (e.g., identity matrix, orthogonal weights), and learning rate schedules.

---

### Module 3: Vanilla RNNs for Time Series Analysis

* **Data Preparation:** Structuring 1D/2D time series into sequence windows (`[batch_size, sequence_length, features]`).
* **Architectures:**
* **Many-to-One:** Forecasting single/multivariate future values from past window sequences.
* **Many-to-Many:** Continuous sequence predictions (auto-regressive forecasting).


* **Hands-on Practice:** Building a time-series Vanilla RNN in PyTorch/TensorFlow from scratch without built-in abstractions.

---

### Module 4: Vanilla RNNs for Text & Natural Language Processing

* **Text as Sequences:** Tokenization, vocabulary building, and character vs. word-level representations.
* **Embeddings + RNN:** Feeding embedded tokens into the hidden state loop.
* **Classic NLP Applications:**
* **Text Classification / Sentiment Analysis** (Many-to-One).
* **Character-Level Text Generation** (Many-to-Many / Auto-regressive language modeling).


* **Hands-on Practice:** Training a character-level Vanilla RNN text generator.

---

### Module 5: Limitations & Diagnostics

* **Measuring Memory Loss:** How far back can a Vanilla RNN *actually* remember in practice?
* **Diagnostic Tools:** Inspecting gradient norms during training to catch vanishing/exploding gradients in real-time.

---

Let's dive straight into **Module 1: Core Fundamentals & Vanilla RNN Mechanics**.

---

## 1. Why Feedforward Neural Networks Fail on Sequences

Standard Feedforward Neural Networks (ANNs) assume that **all inputs are independent of each other**.

Think of reading a sentence:

* *"I grew up in France, so I speak fluent ______."*

To guess the missing word (**French**), your brain connects "France" from earlier in the sentence to the word "speak" near the end.

```
Standard Feedforward (ANN):
  Input [X] ───► [ Hidden Layer ] ───► Output [Y]
  (Processes entire input at once. No memory of what came before.)

Recurrent Neural Network (RNN):
  Input [X_t] ───► [ Hidden State (h_t) ] ───► Output [Y_t]
                           │ ↺ (Passes memory h_t-1 to the next step)

```

A standard ANN cannot easily handle this because:

1. **Fixed Input Size:** It expects a fixed number of inputs (e.g., 50 features), but text and time series can vary in length (10 words vs. 500 words).
2. **No Temporal Memory:** It looks at each data point in isolation and forgets what it saw a step ago.

**The RNN Solution:** It loops! An RNN processes a sequence one step at a time ($t = 1, 2, 3...$) and maintains a **Hidden State ($h_t$)**, which acts as its short-term memory.

---

## 2. Unrolling an RNN Through Time

To understand an RNN, it helps to "unroll" the loop across time steps $t=1, t=2, t=3$:

```
          Output y_1               Output y_2               Output y_3
              ▲                        ▲                        ▲
              │ W_hy                   │ W_hy                   │ W_hy
          ┌───────┐      W_hh      ┌───────┐      W_hh      ┌───────┐
h_0 ────► │  h_1  │ ─────────────► │  h_2  │ ─────────────► │  h_3  │
          └───────┘                └───────┘                └───────┘
              ▲                        ▲                        ▲
              │ W_xh                   │ W_xh                   │ W_xh
          Input x_1                Input x_2                Input x_3

```

Notice three critical weight matrices:

* **$W_{xh}$ (Input-to-Hidden):** Transforms the current input $x_t$ into hidden memory space.
* **$W_{hh}$ (Hidden-to-Hidden):** Transforms the *previous* memory $h_{t-1}$ into current memory space.
* **$W_{hy}$ (Hidden-to-Output):** Transforms the current memory $h_t$ into a final prediction $y_t$.

> **Key Takeaway:** The exact same weight matrices ($W_{xh}, W_{hh}, W_{hy}$) are reused at **every single time step**. This is called **weight sharing**.

---

## 3. The Core Equations Illustrated

At any time step $t$, the RNN does two calculations:

### Equation 1: Calculate New Memory (Hidden State $h_t$)

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

* **$x_t$**: Current input vector.
* **$h_{t-1}$**: Previous hidden state (memory from time step $t-1$).
* **$b_h$**: Bias term for the hidden state.
* **$\tanh$**: Activation function squeezing values between $-1$ and $1$ to keep memory from exploding.

### Equation 2: Calculate Output ($y_t$)

$$\hat{y}_t = \text{Softmax}(W_{hy} \cdot h_t + b_y) \quad \text{or} \quad \hat{y}_t = W_{hy} \cdot h_t + b_y$$

*(Use Softmax for classification, linear for regression/time series).*

---

## 4. Concrete Numerical Walkthrough

Let's do the math step-by-step with real numbers!

### Setup Parameters:

* Input dimension: $1$ (e.g., stock price or single token value)
* Hidden state dimension: $1$ (for ultra-simple math)
* Initial memory $h_0 = 0$ (starting fresh)

**Fixed Weights & Biases:**

* $W_{xh} = 0.5$ (Weight for input)
* $W_{hh} = 0.8$ (Weight for previous memory)
* $W_{hy} = 2.0$ (Weight for output prediction)
* $b_h = 0.0$, $b_y = 0.0$

**Our Input Sequence:** $x = [1.0, 2.0]$ (Length = 2)

---

### Step $t = 1$ (Processing $x_1 = 1.0$)

**1. Calculate Hidden State $h_1$:**


$$h_1 = \tanh(W_{xh} \cdot x_1 + W_{hh} \cdot h_0 + b_h)$$

$$h_1 = \tanh((0.5 \times 1.0) + (0.8 \times 0.0) + 0.0)$$

$$h_1 = \tanh(0.5) \approx 0.462$$

**2. Calculate Output $\hat{y}_1$:**


$$\hat{y}_1 = W_{hy} \cdot h_1 = 2.0 \times 0.462 = \mathbf{0.924}$$

---

### Step $t = 2$ (Processing $x_2 = 2.0$)

Now, $x_2 = 2.0$, and we pass our calculated memory $h_1 = 0.462$ forward!

**1. Calculate Hidden State $h_2$:**


$$h_2 = \tanh(W_{xh} \cdot x_2 + W_{hh} \cdot h_1 + b_h)$$

$$h_2 = \tanh((0.5 \times 2.0) + (0.8 \times 0.462) + 0.0)$$

$$h_2 = \tanh(1.0 + 0.3696) = \tanh(1.3696) \approx \mathbf{0.878}$$

Notice how $h_2$ carries information from **both** $x_2$ ($1.0$) AND $x_1$ (via $h_1 = 0.3696$).

**2. Calculate Output $\hat{y}_2$:**


$$\hat{y}_2 = W_{hy} \cdot h_2 = 2.0 \times 0.878 = \mathbf{1.756}$$

---

## 5. Shape Guide (Dimensions Cheat Sheet)

When writing code in PyTorch/TensorFlow, dimension mismatch is the #1 headache. Here is how matrix shapes fit together:

Assuming:

* $B$ = Batch size (e.g., 32 samples)
* $D_{in}$ = Input feature dimension (e.g., 10 features per time step)
* $D_{out}$ = Hidden state dimension (e.g., 64 hidden units)

| Variable | Shape | Description |
| --- | --- | --- |
| **$x_t$** | $(B, D_{in})$ | Input tensor at time step $t$ |
| **$h_{t-1}$** | $(B, D_{out})$ | Previous hidden state tensor |
| **$W_{xh}$** | $(D_{in}, D_{out})$ | Weight matrix mapping input to hidden space |
| **$W_{hh}$** | $(D_{out}, D_{out})$ | Weight matrix mapping hidden state to hidden space |
| **$b_h$** | $(D_{out})$ | Bias vector for hidden state |
| **$h_t$** | $(B, D_{out})$ | Calculated current hidden state tensor |

---

## Summary Checklist

1. **Memory:** RNNs maintain a hidden state $h_t$ that carries past context to future steps.
2. **Reuse:** The weights $W_{xh}, W_{hh}, W_{hy}$ stay identical across all time steps.
3. **Combination:** $h_t$ combines current input ($x_t$) with prior state ($h_{t-1}$) squeezed through $\tanh$.

---

Let's make these two concepts super concrete! First, we'll re-run our numerical example using an **input dimension of 3** ($D_{in} = 3$). Then, we'll look at the **Shape Guide** with real batch dimensions.

---

## 1. Numerical Walkthrough: Input Dimension = 3 ($D_{in} = 3$)

Instead of a single number at each time step (like just temperature), imagine our input at time step $t$ is a vector of **3 features**: e.g., `[Temperature, Humidity, Wind Speed]`.

### Setup Parameters:

* **Input Dimension ($D_{in}$):** $3$
* **Hidden Dimension ($D_{out}$):** $2$ (we'll keep 2 hidden units so you can see matrix-vector multiplication in action)
* **Initial Hidden State ($h_0$):** $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ (dimension: $2 \times 1$)

### Weights and Biases:

* **$W_{xh}$ (Input-to-Hidden):** Shape $(D_{out}, D_{in}) = (2, 3)$

$$W_{xh} = \begin{bmatrix} 0.2 & 0.1 & -0.3 \\ 0.5 & 0.0 & 0.4 \end{bmatrix}$$


* **$W_{hh}$ (Hidden-to-Hidden):** Shape $(D_{out}, D_{out}) = (2, 2)$

$$W_{hh} = \begin{bmatrix} 0.6 & -0.1 \\ 0.2 & 0.8 \end{bmatrix}$$


* **$b_h$ (Hidden Bias):** Shape $(D_{out}, 1) = (2, 1)$

$$b_h = \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix}$$



---

### Time Step $t = 1$

Our input vector is $x_1 = \begin{bmatrix} 1.0 \\ 2.0 \\ 0.5 \end{bmatrix}$ (dimension $3 \times 1$).

#### **Step 1: Compute $W_{xh} \cdot x_1$**

Multiply $(2 \times 3)$ by $(3 \times 1) \rightarrow$ Output is $(2 \times 1)$:

$$W_{xh} \cdot x_1 = \begin{bmatrix} (0.2 \times 1.0) + (0.1 \times 2.0) + (-0.3 \times 0.5) \\ (0.5 \times 1.0) + (0.0 \times 2.0) + (0.4 \times 0.5) \end{bmatrix} = \begin{bmatrix} 0.2 + 0.2 - 0.15 \\ 0.5 + 0.0 + 0.2 \end{bmatrix} = \begin{bmatrix} 0.25 \\ 0.70 \end{bmatrix}$$

#### **Step 2: Compute $W_{hh} \cdot h_0$**

Since $h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$:


$$W_{hh} \cdot h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

#### **Step 3: Combine and Apply $\tanh$**

$$h_1 = \tanh \left( W_{xh} x_1 + W_{hh} h_0 + b_h \right)$$

$$h_1 = \tanh \left( \begin{bmatrix} 0.25 \\ 0.70 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix} \right) = \tanh \left( \begin{bmatrix} 0.35 \\ 0.70 \end{bmatrix} \right)$$

Applying $\tanh$ element-wise:

* $\tanh(0.35) \approx 0.336$
* $\tanh(0.70) \approx 0.604$

$$\mathbf{h_1 = \begin{bmatrix} 0.336 \\ 0.604 \end{bmatrix}}$$

---

### Time Step $t = 2$

Now let's process the next input vector: $x_2 = \begin{bmatrix} 0.5 \\ 1.0 \\ 2.0 \end{bmatrix}$.
We pass forward our previous hidden state memory: $h_1 = \begin{bmatrix} 0.336 \\ 0.604 \end{bmatrix}$.

#### **Step 1: Compute $W_{xh} \cdot x_2$**

$$W_{xh} \cdot x_2 = \begin{bmatrix} (0.2 \times 0.5) + (0.1 \times 1.0) + (-0.3 \times 2.0) \\ (0.5 \times 0.5) + (0.0 \times 1.0) + (0.4 \times 2.0) \end{bmatrix} = \begin{bmatrix} 0.1 + 0.1 - 0.6 \\ 0.25 + 0.0 + 0.8 \end{bmatrix} = \begin{bmatrix} -0.40 \\ 1.05 \end{bmatrix}$$

#### **Step 2: Compute $W_{hh} \cdot h_1$** (Memory carrying forward)

$$W_{hh} \cdot h_1 = \begin{bmatrix} (0.6 \times 0.336) + (-0.1 \times 0.604) \\ (0.2 \times 0.336) + (0.8 \times 0.604) \end{bmatrix} = \begin{bmatrix} 0.2016 - 0.0604 \\ 0.0672 + 0.4832 \end{bmatrix} = \begin{bmatrix} 0.1412 \\ 0.5504 \end{bmatrix}$$

#### **Step 3: Combine and Apply $\tanh$**

$$h_2 = \tanh \left( \begin{bmatrix} -0.40 \\ 1.05 \end{bmatrix} + \begin{bmatrix} 0.1412 \\ 0.5504 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix} \right) = \tanh \left( \begin{bmatrix} -0.1588 \\ 1.6004 \end{bmatrix} \right)$$

Applying $\tanh$ element-wise:

* $\tanh(-0.1588) \approx -0.1575$
* $\tanh(1.6004) \approx 0.9217$

$$\mathbf{h_2 = \begin{bmatrix} -0.1575 \\ 0.9217 \end{bmatrix}}$$

Notice how $h_2$ blends the new input $x_2$ with the old memory $h_1$!

---

## 2. Shape Guide with Batches (PyTorch/TensorFlow Context)

In real code, we never pass inputs one vector at a time. We pass **batches of sequences** all at once to leverage GPU parallel computing.

### Concrete Example Dimensions:

* **Batch Size ($B$):** `32` (32 independent sequences processed in parallel)
* **Sequence Length ($T$):** `10` (each sequence has 10 time steps)
* **Input Dimension ($D_{in}$):** `3` (3 features per time step, e.g., Temp, Humidity, Wind)
* **Hidden Dimension ($D_{out}$):** `64` (64 hidden memory units)
* **Output Dimension ($K$):** `1` (predicting 1 value, e.g., next hour's temperature)

---

### Step-by-Step Tensor Dimensions:

#### 1. Input Tensor ($X$)

* **Full Batch Shape:** `(Batch Size, Sequence Length, Input Dim)` $\rightarrow$ **`(32, 10, 3)`**
* **At a single time step $t$ ($x_t$):** We slice along the sequence axis $\rightarrow$ **`(32, 3)`**

#### 2. Weight Matrices (Shared across all batch items & time steps)

* **$W_{xh}$ (Input-to-Hidden):** `(Input Dim, Hidden Dim)` $\rightarrow$ **`(3, 64)`**
* **$W_{hh}$ (Hidden-to-Hidden):** `(Hidden Dim, Hidden Dim)` $\rightarrow$ **`(64, 64)`**
* **$b_h$ (Bias):** `(Hidden Dim)` $\rightarrow$ **`(64)`**
* **$W_{hy}$ (Hidden-to-Output):** `(Hidden Dim, Output Dim)` $\rightarrow$ **`(64, 1)`**

#### 3. Hidden State Tensor ($h_t$)

* **Initial Hidden State ($h_0$):** `(Batch Size, Hidden Dim)` $\rightarrow$ **`(32, 64)`** (all zeros)
* **Updated Hidden State ($h_t$):** `(Batch Size, Hidden Dim)` $\rightarrow$ **`(32, 64)`**

---

### The Matrix Equation with Batches:

Using matrix multiplication (`@` in Python):

$$h_t = \tanh(\underbrace{x_t}_{(32, 3)} @ \underbrace{W_{xh}}_{(3, 64)} \;+\; \underbrace{h_{t-1}}_{(32, 64)} @ \underbrace{W_{hh}}_{(64, 64)} \;+\; \underbrace{b_h}_{(64)})$$

* Matrix multiplication term 1: $(32, 3) \times (3, 64) \rightarrow (32, 64)$
* Matrix multiplication term 2: $(32, 64) \times (64, 64) \rightarrow (32, 64)$
* Output shape of $h_t$: **`(32, 64)`**

---
Now let's move into **Module 2: Training Vanilla RNNs & Gradient Dynamics**.

This is where we peel back the hood on how an RNN actually **learns** using **Backpropagation Through Time (BPTT)**, and derive the exact mathematical reason why Vanilla RNNs struggle with long-term memory.

---

## 1. What is Backpropagation Through Time (BPTT)?

In a standard Feedforward Neural Network, backpropagation calculates the gradient of a loss function with respect to weights by moving backward layer-by-layer.

In an RNN, we do the exact same thing, but we **unroll the network across time steps** first.

```
          Time step t=1          Time step t=2          Time step t=3
             Loss L_1               Loss L_2               Loss L_3
                ▲                      ▲                      ▲
                │                      │                      │
            Output y_1             Output y_2             Output y_3
                ▲                      ▲                      ▲
                │ W_hy                 │ W_hy                 │ W_hy
  h_0 ─────► [ h_1 ] ─── W_hh ─────► [ h_2 ] ─── W_hh ─────► [ h_3 ]
                ▲                      ▲                      ▲
                │ W_xh                 │ W_xh                 │ W_xh
             Input x_1              Input x_2              Input x_3

```

### Total Loss Calculation

At each time step $t$, the network makes a prediction $\hat{y}_t$ and incurs a loss $L_t$ (e.g., Mean Squared Error or Cross-Entropy).

The **total loss ($L$)** for the entire sequence of length $T$ is simply the sum of losses at each step:

$$L = \sum_{t=1}^{T} L_t$$

To update our weight matrices ($W_{hy}, W_{xh}, W_{hh}$), we need to find their partial derivatives relative to total loss $L$:


$$\frac{\partial L}{\partial W_{hy}}, \quad \frac{\partial L}{\partial W_{xh}}, \quad \frac{\partial L}{\partial W_{hh}}$$

---

## 2. Deriving the Gradients (Using the Chain Rule)

Let's focus on updating **$W_{hh}$** (the weight matrix connecting hidden states across time), because this is where the famous **Vanishing Gradient** problem occurs.

### Step 1: Loss at a single time step ($t=3$)

Suppose we want to see how $W_{hh}$ affected the loss at time $t=3$ ($L_3$).

By the chain rule:


$$\frac{\partial L_3}{\partial W_{hh}} = \frac{\partial L_3}{\partial h_3} \cdot \frac{\partial h_3}{\partial W_{hh}}$$

### Step 2: The Temporal Chain Reaction

Here is the catch: $h_3$ depends on $h_2$, $h_2$ depends on $h_1$, and $h_1$ depends on $h_0$.

So $W_{hh}$ didn't just affect $h_3$ directly; it affected $h_3$ **indirectly through all previous hidden states**!

To compute $\frac{\partial h_3}{\partial W_{hh}}$, we must sum up the contributions across all past time steps $k \le 3$:

$$\frac{\partial L_3}{\partial W_{hh}} = \sum_{k=1}^{3} \frac{\partial L_3}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

---

## 3. The Root of Vanishing & Exploding Gradients

Look closely at the term $\frac{\partial h_3}{\partial h_k}$. How do we compute how state $h_3$ changes with respect to state $h_1$?

By expanded chain rule:


$$\frac{\partial h_3}{\partial h_1} = \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1}$$

For a sequence of length $T$, computing the gradient at step $T$ back to step $1$ requires multiplying a chain of jacobian matrices:

$$\frac{\partial h_T}{\partial h_1} = \prod_{j=2}^{T} \frac{\partial h_j}{\partial h_{j-1}}$$

### What is inside each term $\frac{\partial h_j}{\partial h_{j-1}}$?

Recall the hidden state equation:


$$h_j = \tanh(W_{xh} x_j + W_{hh} h_{j-1} + b_h)$$

Taking the derivative with respect to $h_{j-1}$:


$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - \tanh^2(...)) \cdot W_{hh}^T$$

Notice that at every backward step, **we multiply by the matrix $W_{hh}^T$ again and again!**

---

## 4. Simple Numerical Intuition for Vanishing/Exploding Gradients

Imagine $W_{hh}$ is just a single scalar number instead of a matrix, and ignore $\tanh$ derivative for a second.

If sequence length $T = 100$:

### Case A: Vanishing Gradient ($W_{hh} = 0.8$)

If $W_{hh} = 0.8$ (less than 1):


$$(0.8)^{100} \approx 0.000000000203$$

* **Result:** The gradient vanishes to almost zero! The network gets **zero feedback** on how input $x_1$ affected loss at time $t=100$. It completely forgets distant past information.

### Case B: Exploding Gradient ($W_{hh} = 1.2$)

If $W_{hh} = 1.2$ (greater than 1):


$$(1.2)^{100} \approx 82,817,974.5$$

* **Result:** The gradient explodes into massive numbers! Weights become `NaN`, leading to unstable training or crashes.

---

## 5. Practical Mitigations for Vanilla RNNs

Since Vanilla RNNs suffer from this mathematical bottleneck, how do we train them in practice?

### 1. Truncated BPTT

Instead of propagating gradients back through the *entire* sequence (e.g., 1000 time steps), we cut off the gradient after $k$ steps (e.g., 20 or 30 steps).

* **Pros:** Saves memory and prevents vanishing/exploding gradients over huge sequences.
* **Cons:** Limits the model's memory horizon to $k$ steps.

### 2. Gradient Clipping (Solves Exploding Gradients)

If the norm of the gradient vector exceeds a threshold $c$, scale it down:

$$\text{if } \Vert{}\mathbf{g}\Vert{} > c \implies \mathbf{g} \leftarrow c \cdot \frac{\mathbf{g}}{\Vert{}\mathbf{g}\Vert{}}$$

```python
# In PyTorch:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

```

### 3. Proper Weight Initialization

* Initialize $W_{hh}$ as an **Identity Matrix ($I$)** or an **Orthogonal Matrix**. This prevents repeated matrix multiplications from scaling eigenvalues up or down prematurely.

---

## Summary Checklist

1. **BPTT:** Unrolls the network across time and sums loss over time steps.
2. **Chain Rule Matrix Multiplication:** Backpropagation requires multiplying by $W_{hh}^T$ at every step backward.
3. **Vanishing Gradient:** Repeated multiplication by values less than 1 causes gradients to disappear $\implies$ **Short-term memory loss**.
4. **Exploding Gradient:** Repeated multiplication by values greater than 1 causes gradients to blow up $\implies$ **Instability** (fixed via gradient clipping).

---

Now let's move into **Module 3: Vanilla RNNs for Time Series Analysis**.

In this module, we will cover:

1. How to transform raw sequential data into **sliding windows**.
2. How to choose between **Many-to-One** and **Many-to-Many** forecasting.
3. How to write a clean, complete **PyTorch implementation** of a Vanilla RNN from first principles.

---

## 1. Structuring Time Series Data (Sliding Windows)

Time series data usually starts as a continuous 1D sequence (e.g., daily temperatures or stock prices over time).

```
Raw Data: [10, 12, 15, 18, 20, 22, 25, 28, 30]

```

An RNN cannot consume a single long vector directly during batch training. We must reframe this into a **supervised learning dataset** using a **Lookback Window ($L$)**:

Suppose $L = 3$ (we use the past 3 days to predict the next day):

| Window Index | Input Sequence ($X$) | Target Label ($y$) |
| --- | --- | --- |
| **Window 0** | `[10, 12, 15]` | `18` |
| **Window 1** | `[12, 15, 18]` | `20` |
| **Window 2** | `[15, 18, 20]` | `22` |
| **Window 3** | `[18, 20, 22]` | `25` |

### Tensor Dimension Check

When batched into PyTorch, our input tensor $X$ will have shape:


$$\text{Shape: } (B, L, D_{in}) = (\text{Batch Size}, \text{Lookback Window Length}, \text{Features per Step})$$

For example, with batch size $32$, $L = 30$ days, and $3$ features (Temp, Humidity, Wind):


$$\text{Input Shape} = (32, 30, 3)$$

---

## 2. Many-to-One vs. Many-to-Many Forecasting

Depending on your business problem, you will configure your RNN output layer in one of two ways:

```
Many-to-One (Predicting Single Step ahead):
  x_1 ──► [RNN] ──► h_1
  x_2 ──► [RNN] ──► h_2
  x_3 ──► [RNN] ──► h_3 ──► [Linear Layer] ──► Prediction (y_4)

Many-to-Many (Predicting Entire Sequence ahead):
  x_1 ──► [RNN] ──► h_1 ──► [Linear] ──► Prediction (y_2)
  x_2 ──► [RNN] ──► h_2 ──► [Linear] ──► Prediction (y_3)
  x_3 ──► [RNN] ──► h_3 ──► [Linear] ──► Prediction (y_4)

```

1. **Many-to-One:** Process the whole sequence, ignore intermediate hidden states, and pass **only the final hidden state ($h_T$)** into a linear output layer.
2. **Many-to-Many:** Pass **every hidden state ($h_1, h_2, ..., h_T$)** into the linear output layer to output a prediction at every time step.

---

## 3. Building a Custom Vanilla RNN in PyTorch

Let's build a **Many-to-One Vanilla RNN** from scratch using pure PyTorch tensors (implementing $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ manually), so you can see every step under the hood!

```python
import torch
import torch.nn as nn

class CustomVanillaRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(CustomVanillaRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Input-to-Hidden weights & bias
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        
        # 2. Hidden-to-Hidden weights & bias
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        
        # 3. Hidden-to-Output weights & bias
        self.W_hy = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (batch_size, seq_len, input_dim)
        Output shape:  (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize h_0 with zeros: shape (batch_size, hidden_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Unroll loop across all time steps t
        for t in range(seq_len):
            x_t = x[:, t, :]  # Slice at step t -> shape (batch_size, input_dim)
            
            # Equation: h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
            h_t = torch.tanh(
                torch.matmul(x_t, self.W_xh) + 
                torch.matmul(h_t, self.W_hh) + 
                self.b_h
            )
            
        # Many-to-One: Use ONLY the final hidden state h_T for output prediction
        y_pred = torch.matmul(h_t, self.W_hy) + self.b_y
        return y_pred

```

---

## 4. End-to-End Training Loop Example

Now let's synthesize synthetic time series data and train our custom RNN using **Gradient Clipping**:

```python
# Setup hyperparameters
batch_size = 16
seq_len = 20    # 20 lookback time steps
input_dim = 3   # 3 time-series features
hidden_dim = 32 # 32 hidden units
output_dim = 1  # Predict 1 target scalar

# 1. Create dummy dataset
X_dummy = torch.randn(batch_size, seq_len, input_dim)
y_dummy = torch.randn(batch_size, output_dim)

# 2. Instantiate Model, Loss Function, and Optimizer
model = CustomVanillaRNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Training Step
model.train()
optimizer.zero_grad()

# Forward pass
y_pred = model(X_dummy)
loss = criterion(y_pred, y_dummy)

# Backward pass (BPTT)
loss.backward()

# Gradient Clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Update weights
optimizer.step()

print(f"Training Loss: {loss.item():.4f}")

```

---

## Summary Checklist

1. **Windowing:** Convert raw sequential series into sliding windows of shape `(Batch, Seq_Len, Features)`.
2. **Hidden Loop:** Iterate step-by-step over `Seq_Len` updating state $h_t$.
3. **Many-to-One:** Use the final step state $h_T$ for prediction.
4. **Gradient Clipping:** Clip gradients before `optimizer.step()` to protect training stability.

---

Yes! How Backpropagation Through Time (BPTT) works **depends directly on the sequence architecture** you are using (Many-to-One, Many-to-Many, One-to-Many).

Let's look at how loss is calculated and how gradients flow backward in each layout.

---

## The 4 Sequence Architectures & Their BPTT Flow

```
1. Many-to-One             2. Many-to-Many (Synchronized)      3. Many-to-Many (Unsynchronized / Seq2Seq)
   Loss L_T                    Loss L_1   Loss L_2   Loss L_3        Encoder          Decoder
      ▲                           ▲          ▲          ▲                         Loss L_1  Loss L_2
      │                           │          │          │                            ▲         ▲
    y_pred                     y_pred_1   y_pred_2   y_pred_3                   y_pred_1  y_pred_2
      │                           │          │          │                            │         │
   [ h_T ]                     [ h_1 ] ──► [ h_2 ] ──► [ h_3 ]             [ h_enc ] ──► [ h_1 ] ──► [ h_2 ]
      ▲                           ▲          ▲          ▲                         ▲            ▲         ▲
  (Only 1 Loss)                (Loss at EVERY time step)                 (Read input, then predict output)

```

---

### 1. Many-to-One BPTT

* **Examples:** Sentiment Analysis (*"This movie was great"* $\rightarrow$ Positive), Time Series Single-Step Prediction (Past 30 days $\rightarrow$ Tomorrow's price).
* **Loss Calculation:** There is **only ONE loss term** $L_T$ generated at the final time step $T$:

$$L_{total} = L_T(y_{pred}, y_{true})$$


* **BPTT Flow:**
1. The gradient starts **only at the final hidden state $h_T$**: $\frac{\partial L_T}{\partial h_T}$.
2. Gradients then travel backward through time: $h_T \rightarrow h_{T-1} \rightarrow h_{T-2} \dots \rightarrow h_1$.
3. Every step updates the shared weights ($W_{hh}, W_{xh}$) based on how past states influenced that **single final output**.



---

### 2. Many-to-Many (Synchronized) BPTT

* **Examples:** Named Entity Recognition / POS Tagging (labeling every word in a sentence), Continuous Time Series Forecasting.
* **Loss Calculation:** Every time step generates its own prediction and loss. The **total loss is the sum** of losses at every step:

$$L_{total} = \sum_{t=1}^{T} L_t$$


* **BPTT Flow:**
1. At step $t=3$, gradient enters from **both** its local loss $L_3$ AND the incoming gradient from step $t=4$:

$$\text{Total Gradient at } h_3 = \frac{\partial L_3}{\partial h_3} + \left( \frac{\partial L_{later}}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \right)$$


2. Gradients **accumulate at every step** as they flow backward through time.



---

### 3. Many-to-Many (Unsynchronized / Seq2Seq) BPTT

* **Examples:** Machine Translation (*"How are you"* [3 words] $\rightarrow$ *"Comment allez-vous"* [3 words], but sentence lengths can differ).
* **Two Phases:**
* **Encoder:** Reads input sequence $(x_1, \dots, x_N)$ and produces a summary vector $h_{enc}$. (No local loss generated here).
* **Decoder:** Takes $h_{enc}$ and generates output sequence $(y_1, \dots, y_M)$ step by step, producing losses $(L_1, \dots, L_M)$.


* **BPTT Flow:**
1. Gradients flow backward through the **Decoder** time steps ($M \dots 1$).
2. The accumulated gradient passes through the context vector into the **Encoder** and flows backward through the input time steps ($N \dots 1$).



---

### 4. One-to-Many BPTT

* **Examples:** Image Captioning (1 image $\rightarrow$ sequence of words describing it), Music Generation (1 seed key $\rightarrow$ sequence of notes).
* **Loss Calculation:** Losses $(L_1, L_2, \dots, L_M)$ are accumulated during the output sequence generation.
* **BPTT Flow:** Gradients flow backward step-by-step through the generated sequence, ending at the initial input/embedding vector.

---

## Comparison Summary Table

| Architecture | Loss Points | How Gradients Enter BPTT | Primary Use Case |
| --- | --- | --- | --- |
| **Many-to-One** | Single loss at time step $T$ | Enters *only* at step $T$, flows backward to $1$. | Classification, Next-step forecasting |
| **Many-to-Many (Sync)** | Loss at *every* step ($1 \dots T$) | Enters at *every* step, accumulating as it flows backward. | Sequence labeling, POS tagging |
| **Seq2Seq** | Loss at *decoder* steps ($1 \dots M$) | Flows through Decoder, then crosses into Encoder. | Translation, Summarization |
| **One-to-Many** | Loss at *output* steps ($1 \dots M$) | Enters across output steps, flows back to original input. | Image captioning, Music generation |

---

Now let's dive into **Module 4: Vanilla RNNs for Text & Natural Language Processing (NLP)**.

Here, we'll see how unstructured text is turned into mathematical representations that an RNN can process, and build a complete character-level text generation model in PyTorch.

---

## 1. Representing Text as Tensors

Computers don't understand words or characters directly; they only work with numbers. To feed text into an RNN, we follow a strict pipeline:

```
Raw Text: "hello"
   │
   ▼
1. Vocabulary Building: {'h': 0, 'e': 1, 'l': 2, 'o': 3}
   │
   ▼
2. Tokenization / Numericalization: [0, 1, 2, 2, 3]
   │
   ▼
3. Vector Representation: One-Hot Vectors OR Learned Embeddings

```

### Approach A: One-Hot Encoding

If our vocabulary size is $V$:

* `'h'` (index 0) $\rightarrow [1, 0, 0, 0]$
* `'e'` (index 1) $\rightarrow [0, 1, 0, 0]$
* **Pros:** Simple, no training required.
* **Cons:** Sparse, high-dimensional for large word vocabularies, and assumes all words are equally distant (no semantic similarity).

### Approach B: Learned Embeddings (Word / Character Embeddings)

Instead of sparse vectors, we map each token index to a dense vector of size $E$ (e.g., $E = 64$ or $300$).
An **Embedding Layer** acts as a trainable lookup table:

* Index $0 \rightarrow [-0.24, 0.81, 0.05, \dots]$ (Dense vector of dimension $E$)

---

## 2. Text Tensor Dimensions

When processing text in PyTorch, your input batch looks like this:

$$\text{Token Indices Batch Shape: } (B, T) = (\text{Batch Size}, \text{Sequence Length})$$

After passing through an **Embedding Layer**:


$$\text{Embedded Input Shape: } (B, T, E) = (\text{Batch Size}, \text{Sequence Length}, \text{Embedding Dim})$$

Notice how $(B, T, E)$ matches our time series shape $(B, T, D_{in})$! In text, $E$ is simply our feature dimension at each time step.

---

## 3. Classic NLP Tasks with Vanilla RNNs

### Task 1: Text Classification / Sentiment Analysis (Many-to-One)

* **Input:** Token sequence $x_1, x_2, \dots, x_T$ (*"This movie was great"*)
* **Process:** Pass tokens step-by-step through the RNN to update hidden state $h_t$.
* **Output:** Use final hidden state $h_T$ $\rightarrow$ Linear Layer $\rightarrow$ Softmax $\rightarrow$ Class probabilities (`Positive` / `Negative`).

### Task 2: Language Modeling & Text Generation (Many-to-Many Auto-regressive)

* **Input:** Sequence of characters/words.
* **Target:** Predict the **very next token** at every single time step.
* **Auto-regressive Generation:** Feed the model's predicted output at step $t$ back in as the input at step $t+1$ to generate brand new text character-by-character!

```
Target:      'e'        'l'        'l'        'o'
              ▲          ▲          ▲          ▲
          Softmax    Softmax    Softmax    Softmax
              │          │          │          │
h_0 ─────► [ h_1 ] ───► [ h_2 ] ───► [ h_3 ] ───► [ h_4 ]
              ▲          ▲          ▲          ▲
            Input      Input      Input      Input
             'h'        'e'        'l'        'l'

```

---

## 4. Hands-On PyTorch: Character-Level Language Model

Let's write a complete, runnable PyTorch model for character-level text generation from scratch!

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharVanillaRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super(CharVanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Embedding layer converts char indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Vanilla RNN cell weights
        self.W_xh = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # 3. Output layer maps hidden state to vocabulary scores (logits)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None) -> tuple:
        """
        Input x shape:      (batch_size, seq_len)
        Output logits:      (batch_size, seq_len, vocab_size)
        Final hidden state: (batch_size, hidden_dim)
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embeds = self.embedding(x)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            
        logits_list = []
        h_t = h_prev
        
        # Process sequence time step by time step
        for t in range(seq_len):
            x_t = embeds[:, t, :] # Shape: (batch_size, embed_dim)
            
            # h_t = tanh(W_xh @ x_t + W_hh @ h_t + b)
            h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_t))
            
            # Compute character scores for step t
            out_t = self.fc_out(h_t) # Shape: (batch_size, vocab_size)
            logits_list.append(out_t)
            
        # Stack logits along sequence dimension: (batch_size, seq_len, vocab_size)
        logits = torch.stack(logits_list, dim=1)
        return logits, h_t

```

---

## 5. Training & Generating Text (Step-by-Step Execution)

```python
# Setup Vocabulary
text = "hello world"
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
embed_dim = 16
hidden_dim = 32

model = CharVanillaRNN(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Input sequence:  "hello worl"
# Target sequence: "ello world"
input_text = text[:-1]
target_text = text[1:]

input_tensor = torch.tensor([[char_to_idx[c] for c in input_text]])   # Shape (1, 10)
target_tensor = torch.tensor([[char_to_idx[c] for c in target_text]]) # Shape (1, 10)

# Single Training Step
model.train()
optimizer.zero_grad()

logits, _ = model(input_tensor)

# Reshape for CrossEntropyLoss: (Batch * Seq_Len, Vocab_Size) vs (Batch * Seq_Len)
loss = criterion(logits.view(-1, vocab_size), target_tensor.view(-1))
loss.backward()

# Clip gradients to prevent explosion during BPTT
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

print(f"Loss: {loss.item():.4f}")

```

### Auto-Regressive Text Sampling

To generate text from a prompt (e.g., seed character `'h'`):

1. Pass `'h'` into the model to get $h_1$ and character probabilities for step 2.
2. Sample the next character (e.g., `'e'`) from the probability distribution.
3. Pass `'e'` and previous hidden state $h_1$ back into the model to predict step 3!
4. Repeat until desired text length is generated.

---

## Summary Checklist

1. **Embeddings:** Map integer token IDs to dense vectors of shape $(B, T, E)$.
2. **Text Generation Setup:** Input is shifted right by 1 character relative to target.
3. **Loss Function:** `CrossEntropyLoss` evaluated over all time steps (Many-to-Many).
4. **Sampling:** Feed generated predictions iteratively back as inputs to generate text auto-regressively.

---

Welcome to **Module 5: Diagnostic Tools & Practical Limits of Vanilla RNNs**.

In this final module, we will explore the real-world operational bounds of Vanilla RNNs: how far back they can *actually* remember, how to diagnose vanishing/exploding gradients during training, and where Vanilla RNNs still fit in the modern machine learning landscape.

---

## 1. Measuring Memory Capacity: How Far Can a Vanilla RNN See?

While a Vanilla RNN theoretically passes information across arbitrarily long sequences, in practice, its **effective memory horizon** is strictly limited by gradient dynamics and non-linear compression ($\tanh$).

```
Effective Context Memory Window:
Step 1    Step 2    ...    Step 10    Step 11    ...    Step 50
  │         │                 │          │                 │
  ▼         ▼                 ▼          ▼                 ▼
[High Impact on Memory]    [Fading Context]    [Effectively Zero Context]

```

### Empirical Memory Horizon Rule of Thumb:

* **Vanilla RNN:** Effective memory spans only **5 to 10 time steps**. Beyond ~10 steps, gradients vanish near-completely, and initial inputs ($x_1$) lose influence over predictions at step $T$.
* **Why?** The repeated multiplication by $W_{hh}^T$ coupled with the derivative of $\tanh$ (which is always $\le 1.0$) causes exponential decay of information.

---

## 2. Diagnostic Tools: Monitoring Gradient Norms

To detect vanishing or exploding gradients before your model fails silently or throws `NaN` errors, you can track the **global gradient norm** during training.

### Tracking Gradient Health in PyTorch

```python
def compute_grad_norm(model: torch.nn.Module) -> float:
    """Computes the total L2 norm of model parameter gradients."""
    total_sq_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_sq_norm += param_norm.item() ** 2
    return total_sq_norm ** 0.5

# Inside your training loop:
loss.backward()

# 1. Inspect gradient norm BEFORE clipping
raw_grad_norm = compute_grad_norm(model)

# 2. Apply Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Diagnostic Logging
if raw_grad_norm < 1e-4:
    print(f"[Warning] Potential Vanishing Gradient: norm = {raw_grad_norm:.6f}")
elif raw_grad_norm > 10.0:
    print(f"[Warning] Potential Exploding Gradient: norm = {raw_grad_norm:.6f}")

```

---

## 3. The Vanilla RNN Diagnostic Matrix

When debugging Vanilla RNN training on time series or text tasks, use this checklist to isolate issues:

| Symptom | Mathematical Cause | Diagnostic Sign | Primary Fix |
| --- | --- | --- | --- |
| **Loss becomes `NaN` or `Inf**` | Exploding Gradients | Gradient norm $> 100$ or rapidly increasing | Apply gradient clipping (`max_norm=1.0`). |
| **Model ignores long-range context** | Vanishing Gradients | Gradient norm drops to $\approx 0$ for early steps | Reduce sequence length / Use Truncated BPTT. |
| **Training loss stagnates early** | Poor weight initialization | Hidden state activations collapse to 0 | Initialize $W_{hh}$ as an **Orthogonal** matrix. |
| **Model overfits small sequences** | High parameter density | Validation loss diverges from training loss | Add Dropout or weight decay ($L_2$ regularization). |

---

## 4. Modern Perspective: Where Vanilla RNNs Still Shine

While gated architectures (LSTMs, GRUs) and Transformers dominate long-context NLP and complex time series forecasting, Vanilla RNNs maintain several distinct advantages:

1. **Ultra-Low Latency Inference:** A Vanilla RNN cell requires only **one matrix multiplication per time step** (compared to 4 in LSTMs and 3 in GRUs).
2. **Edge & Microcontroller Deployment:** Extremely tiny memory footprint ($W_{xh}, W_{hh}, W_{hy}$), making them ideal for tinyML and low-power IoT sensors.
3. **Streaming Inputs:** Naturally suited for continuous online processing where inputs arrive one sample at a time without needing full-sequence re-computation.

---

## 🎓 Complete Course Summary & Key Formulas

To wrap up our Vanilla RNN mastery journey, here are the core mathematical backbones to retain:

1. **Forward State Update:**

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$


2. **Output Prediction:**

$$\hat{y}_t = W_{hy} h_t + b_y$$


3. **BPTT Gradient Decay:**

$$\frac{\partial h_T}{\partial h_1} = \prod_{j=2}^{T} \frac{\partial h_j}{\partial h_{j-1}}$$


4. **Gradient Clipping Formula:**

$$\mathbf{g} \leftarrow \text{min}\left(1, \frac{c}{\Vert{}\mathbf{g}\Vert{}}\right) \mathbf{g}$$



---
