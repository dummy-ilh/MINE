# Recurrent Neural Networks (RNN) - Complete Deep Dive

## 1. Core Design & Architecture

### What Makes RNNs Special?
RNNs process **sequential data** by maintaining a **hidden state** that acts as memory. Unlike feedforward networks, RNNs have loops that allow information to persist.

### Basic RNN Cell Structure

```
Input at time t: x_t
Hidden state (memory): h_t
Output at time t: y_t

The same cell is reused across time steps!
```

**Visual representation:**
```
    x₀        x₁        x₂        x₃
     ↓         ↓         ↓         ↓
    [RNN] → [RNN] → [RNN] → [RNN]
     ↓         ↓         ↓         ↓
    y₀        y₁        y₂        y₃
```

---

## 2. Components & Their Importance

### **A. Input Vector (x_t)**
- **What**: Current input at time step t
- **Shape**: (input_size,)
- **Importance**: Brings new information at each time step
- **Example**: Word embedding, sensor reading, stock price

### **B. Hidden State (h_t)**
- **What**: Memory that carries information from previous time steps
- **Shape**: (hidden_size,)
- **Importance**: 
  - Captures context from the past
  - Enables the network to "remember"
  - Gets updated at each time step
- **Initialized**: Usually to zeros at t=0

### **C. Weight Matrices**

**W_xh (Input-to-Hidden weights)**
- Shape: (hidden_size × input_size)
- Purpose: Transforms current input to hidden space
- Importance: Learns what features of input are relevant

**W_hh (Hidden-to-Hidden weights)**
- Shape: (hidden_size × hidden_size)
- Purpose: Transforms previous hidden state
- Importance: Learns how to combine past information with present
- **CRITICAL**: These are the "recurrent" weights that create memory

**W_hy (Hidden-to-Output weights)**
- Shape: (output_size × hidden_size)
- Purpose: Produces output from hidden state
- Importance: Maps internal representation to desired output

### **D. Bias Vectors**
- **b_h**: Bias for hidden state (hidden_size,)
- **b_y**: Bias for output (output_size,)
- **Importance**: Allows shifting of activation functions

### **E. Activation Functions**

**tanh (for hidden state)**
- Range: [-1, 1]
- Importance: 
  - Keeps values bounded
  - Allows both positive and negative signals
  - Stronger gradients than sigmoid near 0

**softmax/sigmoid (for output)**
- For classification tasks
- Produces probability distributions

---

## 3. Mathematical Formulation

### **Core RNN Equations**

At each time step t:

**1. Hidden State Update:**
```
h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)
```

**2. Output Calculation:**
```
y_t = W_hy · h_t + b_y
```

**3. Final Prediction (for classification):**
```
ŷ_t = softmax(y_t)
```

### **Detailed Breakdown**

```
Step 1: Linear transformation of input
  z_x = W_xh · x_t

Step 2: Linear transformation of previous hidden state
  z_h = W_hh · h_(t-1)

Step 3: Combine and add bias
  z = z_x + z_h + b_h

Step 4: Apply non-linearity
  h_t = tanh(z)

Step 5: Generate output
  y_t = W_hy · h_t + b_y
```

### **Why These Equations Matter**

The key insight: **h_t depends on h_(t-1), which depends on h_(t-2), and so on...**

This creates a chain:
```
h_t = f(x_t, h_(t-1))
    = f(x_t, f(x_(t-1), h_(t-2)))
    = f(x_t, f(x_(t-1), f(x_(t-2), ...)))
```

So h_t contains information from ALL previous inputs!

---

## 4. Hand Calculation Example

Let's work through a complete example with **real numbers**.

### **Setup**
```
Task: Sentiment analysis
Sequence: "I love AI" → 3 words
Input size: 2 (word embedding dimension)
Hidden size: 3
Output size: 2 (positive/negative)
```

### **Initialize Parameters**

**Weight matrices:**
```
W_xh = [[ 0.5, -0.3],
        [ 0.2,  0.8],
        [-0.4,  0.1]]
Shape: (3, 2)

W_hh = [[ 0.6, -0.2,  0.3],
        [ 0.1,  0.7, -0.1],
        [-0.3,  0.4,  0.5]]
Shape: (3, 3)

W_hy = [[ 0.9,  0.2, -0.4],
        [-0.5,  0.6,  0.8]]
Shape: (2, 3)
```

**Biases:**
```
b_h = [0.1, -0.2, 0.0]
b_y = [0.0, 0.0]
```

**Initial hidden state:**
```
h_0 = [0.0, 0.0, 0.0]
```

**Input embeddings:**
```
x_1 (I)    = [0.8, 0.3]
x_2 (love) = [0.9, 0.7]
x_3 (AI)   = [0.6, 0.4]
```

---

### **TIME STEP 1: Process "I"**

**Input:** x_1 = [0.8, 0.3], h_0 = [0.0, 0.0, 0.0]

**Step 1: Compute W_xh · x_1**
```
W_xh · x_1 = [[ 0.5, -0.3],     [0.8]
              [ 0.2,  0.8],  ×  [0.3]
              [-0.4,  0.1]]

Row 1: 0.5×0.8 + (-0.3)×0.3 = 0.4 - 0.09 = 0.31
Row 2: 0.2×0.8 + 0.8×0.3    = 0.16 + 0.24 = 0.40
Row 3: -0.4×0.8 + 0.1×0.3   = -0.32 + 0.03 = -0.29

Result: [0.31, 0.40, -0.29]
```

**Step 2: Compute W_hh · h_0**
```
W_hh · h_0 = [[ 0.6, -0.2,  0.3],     [0.0]
              [ 0.1,  0.7, -0.1],  ×  [0.0]
              [-0.3,  0.4,  0.5]]     [0.0]

Result: [0.0, 0.0, 0.0]  (since h_0 is all zeros)
```

**Step 3: Add bias and combine**
```
z_1 = W_xh·x_1 + W_hh·h_0 + b_h
    = [0.31, 0.40, -0.29] + [0.0, 0.0, 0.0] + [0.1, -0.2, 0.0]
    = [0.41, 0.20, -0.29]
```

**Step 4: Apply tanh activation**
```
h_1 = tanh(z_1) = tanh([0.41, 0.20, -0.29])

tanh(0.41) ≈ 0.388
tanh(0.20) ≈ 0.197
tanh(-0.29) ≈ -0.282

h_1 = [0.388, 0.197, -0.282]
```

**Step 5: Compute output (optional, if needed at each step)**
```
y_1 = W_hy · h_1 + b_y

W_hy · h_1 = [[ 0.9,  0.2, -0.4],     [0.388]
              [-0.5,  0.6,  0.8]]  ×  [0.197]
                                       [-0.282]

Row 1: 0.9×0.388 + 0.2×0.197 + (-0.4)×(-0.282)
     = 0.349 + 0.039 + 0.113 = 0.501

Row 2: -0.5×0.388 + 0.6×0.197 + 0.8×(-0.282)
     = -0.194 + 0.118 - 0.226 = -0.302

y_1 = [0.501, -0.302] + [0.0, 0.0] = [0.501, -0.302]
```

---

### **TIME STEP 2: Process "love"**

**Input:** x_2 = [0.9, 0.7], h_1 = [0.388, 0.197, -0.282]

**Step 1: Compute W_xh · x_2**
```
W_xh · x_2 = [[ 0.5, -0.3],     [0.9]
              [ 0.2,  0.8],  ×  [0.7]
              [-0.4,  0.1]]

Row 1: 0.5×0.9 + (-0.3)×0.7 = 0.45 - 0.21 = 0.24
Row 2: 0.2×0.9 + 0.8×0.7    = 0.18 + 0.56 = 0.74
Row 3: -0.4×0.9 + 0.1×0.7   = -0.36 + 0.07 = -0.29

Result: [0.24, 0.74, -0.29]
```

**Step 2: Compute W_hh · h_1**
```
W_hh · h_1 = [[ 0.6, -0.2,  0.3],     [0.388]
              [ 0.1,  0.7, -0.1],  ×  [0.197]
              [-0.3,  0.4,  0.5]]     [-0.282]

Row 1: 0.6×0.388 + (-0.2)×0.197 + 0.3×(-0.282)
     = 0.233 - 0.039 - 0.085 = 0.109

Row 2: 0.1×0.388 + 0.7×0.197 + (-0.1)×(-0.282)
     = 0.039 + 0.138 + 0.028 = 0.205

Row 3: -0.3×0.388 + 0.4×0.197 + 0.5×(-0.282)
     = -0.116 + 0.079 - 0.141 = -0.178

Result: [0.109, 0.205, -0.178]
```

**Step 3: Add bias and combine**
```
z_2 = [0.24, 0.74, -0.29] + [0.109, 0.205, -0.178] + [0.1, -0.2, 0.0]
    = [0.449, 0.745, -0.468]
```

**Step 4: Apply tanh**
```
h_2 = tanh([0.449, 0.745, -0.468])

tanh(0.449) ≈ 0.422
tanh(0.745) ≈ 0.633
tanh(-0.468) ≈ -0.437

h_2 = [0.422, 0.633, -0.437]
```

**Step 5: Compute output**
```
W_hy · h_2 = [[ 0.9,  0.2, -0.4],     [0.422]
              [-0.5,  0.6,  0.8]]  ×  [0.633]
                                       [-0.437]

Row 1: 0.9×0.422 + 0.2×0.633 + (-0.4)×(-0.437)
     = 0.380 + 0.127 + 0.175 = 0.682

Row 2: -0.5×0.422 + 0.6×0.633 + 0.8×(-0.437)
     = -0.211 + 0.380 - 0.350 = -0.181

y_2 = [0.682, -0.181]
```

---

### **TIME STEP 3: Process "AI"**

**Input:** x_3 = [0.6, 0.4], h_2 = [0.422, 0.633, -0.437]

**Step 1: Compute W_xh · x_3**
```
W_xh · x_3 = [[ 0.5, -0.3],     [0.6]
              [ 0.2,  0.8],  ×  [0.4]
              [-0.4,  0.1]]

Row 1: 0.5×0.6 + (-0.3)×0.4 = 0.3 - 0.12 = 0.18
Row 2: 0.2×0.6 + 0.8×0.4    = 0.12 + 0.32 = 0.44
Row 3: -0.4×0.6 + 0.1×0.4   = -0.24 + 0.04 = -0.20

Result: [0.18, 0.44, -0.20]
```

**Step 2: Compute W_hh · h_2**
```
W_hh · h_2 = [[ 0.6, -0.2,  0.3],     [0.422]
              [ 0.1,  0.7, -0.1],  ×  [0.633]
              [-0.3,  0.4,  0.5]]     [-0.437]

Row 1: 0.6×0.422 + (-0.2)×0.633 + 0.3×(-0.437)
     = 0.253 - 0.127 - 0.131 = -0.005

Row 2: 0.1×0.422 + 0.7×0.633 + (-0.1)×(-0.437)
     = 0.042 + 0.443 + 0.044 = 0.529

Row 3: -0.3×0.422 + 0.4×0.633 + 0.5×(-0.437)
     = -0.127 + 0.253 - 0.219 = -0.093

Result: [-0.005, 0.529, -0.093]
```

**Step 3: Add bias and combine**
```
z_3 = [0.18, 0.44, -0.20] + [-0.005, 0.529, -0.093] + [0.1, -0.2, 0.0]
    = [0.275, 0.769, -0.293]
```

**Step 4: Apply tanh**
```
h_3 = tanh([0.275, 0.769, -0.293])

tanh(0.275) ≈ 0.268
tanh(0.769) ≈ 0.646
tanh(-0.293) ≈ -0.285

h_3 = [0.268, 0.646, -0.285]
```

**Step 5: Compute final output**
```
W_hy · h_3 = [[ 0.9,  0.2, -0.4],     [0.268]
              [-0.5,  0.6,  0.8]]  ×  [0.646]
                                       [-0.285]

Row 1: 0.9×0.268 + 0.2×0.646 + (-0.4)×(-0.285)
     = 0.241 + 0.129 + 0.114 = 0.484

Row 2: -0.5×0.268 + 0.6×0.646 + 0.8×(-0.285)
     = -0.134 + 0.388 - 0.228 = 0.026

y_3 = [0.484, 0.026]
```

**Apply softmax for final prediction:**
```
exp(y_3) = [exp(0.484), exp(0.026)] = [1.622, 1.026]
sum = 2.648

Probabilities = [1.622/2.648, 1.026/2.648]
              = [0.612, 0.388]

Prediction: Class 0 (Positive) with 61.2% confidence
```

---

## 5. Summary of Computation Flow

```
t=0: h_0 = [0, 0, 0]

t=1: x_1=[0.8,0.3] → h_1=[0.388, 0.197, -0.282]
t=2: x_2=[0.9,0.7] → h_2=[0.422, 0.633, -0.437]
t=3: x_3=[0.6,0.4] → h_3=[0.268, 0.646, -0.285]

Final output: y_3 = [0.484, 0.026]
After softmax: [0.612, 0.388] → Positive sentiment
```

---

## 6. Key Insights

### **Why W_hh is Critical**
Notice how h_2 depends on h_1, which carries information from x_1. The W_hh matrix is what allows the network to **propagate information** through time.

### **Information Flow**
```
"I" → affects h_1
"love" → combines with memory of "I" in h_2
"AI" → combines with memory of "I love" in h_3
```

The final hidden state h_3 contains compressed information about the entire sequence!

### **Gradient Issues (Vanishing/Exploding)**
When we backpropagate through time, gradients multiply by W_hh repeatedly:
```
∂Loss/∂W_hh involves products like W_hh × W_hh × W_hh × ...
```

If eigenvalues of W_hh are:
- < 1: Gradients vanish (can't learn long dependencies)
- > 1: Gradients explode (unstable training)

This is why LSTM and GRU were invented!

---

## 7. Training (Brief Overview)

**Loss Function** (for classification):
```
L = -Σ y_true × log(y_pred)
```

**Backpropagation Through Time (BPTT):**
1. Forward pass through all time steps
2. Compute loss at final step
3. Backpropagate gradients backward through time
4. Update W_xh, W_hh, W_hy, biases using gradients


1. Show the backpropagation calculations?
2. Explain LSTM in similar depth?
3. Provide code implementation?
