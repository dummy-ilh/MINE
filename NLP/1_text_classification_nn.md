# Forward Pass Simulation: Text Classification

Let me walk through a **concrete example** with actual numbers!

## Input Text
**"I love this movie"**

---

## Step-by-Step Forward Pass

### **Step 1: Tokenization**
```
Text → Tokens: ["I", "love", "this", "movie"]
```

### **Step 2: Convert to Integer IDs**
Using a vocabulary dictionary:
```python
vocab = {"<PAD>": 0, "I": 1, "love": 2, "this": 3, "movie": 4, ...}

Input IDs: [1, 2, 3, 4]
Shape: (4,)  # sequence length = 4
```

### **Step 3: Embedding Layer**
Each ID is mapped to a dense vector (let's use embedding_dim = 3)

```
Embedding Matrix shape: (vocab_size × embedding_dim) = (10000 × 3)

Token "I" (ID=1)    → [0.2, -0.5, 0.8]
Token "love" (ID=2) → [0.9, 0.1, -0.3]
Token "this" (ID=3) → [-0.1, 0.6, 0.4]
Token "movie" (ID=4)→ [0.7, -0.2, 0.5]

Embedded Input shape: (4, 3)
```

**Matrix form:**
```
[[0.2, -0.5,  0.8],
 [0.9,  0.1, -0.3],
 [-0.1, 0.6,  0.4],
 [0.7, -0.2,  0.5]]
```

### **Step 4: LSTM Layer** (hidden_size = 2)
Processes sequence one token at a time:

```
Initial hidden state h₀: [0.0, 0.0]
Initial cell state c₀:   [0.0, 0.0]

Time step 1: Input [0.2, -0.5, 0.8]
  → h₁ = [0.15, -0.23]
  → c₁ = [0.31, -0.12]

Time step 2: Input [0.9, 0.1, -0.3]
  → h₂ = [0.42, 0.08]
  → c₂ = [0.67, 0.15]

Time step 3: Input [-0.1, 0.6, 0.4]
  → h₃ = [0.28, 0.51]
  → c₃ = [0.45, 0.89]

Time step 4: Input [0.7, -0.2, 0.5]
  → h₄ = [0.61, 0.33]  ← Final hidden state
  → c₄ = [0.92, 0.71]
```

**Output: Final hidden state**
```
h₄ = [0.61, 0.33]
Shape: (2,)
```

### **Step 5: Dense (Fully Connected) Layer**
Maps hidden state to number of classes (let's say 2 classes: Positive/Negative)

```
Weight matrix W: (2 × 2)
Bias b: (2,)

W = [[0.8, -0.4],
     [0.3,  0.9]]
     
b = [0.1, -0.2]

Output = W × h₄ + b
       = [[0.8, -0.4],   [0.61]   [0.1]
          [0.3,  0.9]] × [0.33] + [-0.2]
       
       = [0.8×0.61 + (-0.4)×0.33 + 0.1,
          0.3×0.61 + 0.9×0.33 + (-0.2)]
       
       = [0.488 - 0.132 + 0.1,
          0.183 + 0.297 - 0.2]
       
       = [0.456, 0.280]

Logits shape: (2,)
```

### **Step 6: Softmax Activation**
Convert logits to probabilities:

```
Softmax(z) = exp(z) / sum(exp(z))

exp([0.456, 0.280]) = [1.578, 1.323]
sum = 1.578 + 1.323 = 2.901

Probabilities = [1.578/2.901, 1.323/2.901]
              = [0.544, 0.456]
```

**Final Output:**
```
Positive: 54.4%
Negative: 45.6%

Prediction: Positive (argmax)
```

---

## Summary Table

| Layer | Input Shape | Output Shape | Values |
|-------|-------------|--------------|--------|
| Input IDs | (4,) | (4,) | [1, 2, 3, 4] |
| Embedding | (4,) | (4, 3) | 4 vectors of dim 3 |
| LSTM | (4, 3) | (2,) | [0.61, 0.33] |
| Dense | (2,) | (2,) | [0.456, 0.280] |
| Softmax | (2,) | (2,) | [0.544, 0.456] |

**Predicted Class: Positive (54.4% confidence)**
