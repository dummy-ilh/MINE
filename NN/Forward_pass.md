

# 📘 Forward Pass in Artificial Neural Networks (ANNs)

---

## 1. **What Is the Forward Pass?**

The **forward pass** is the process of taking an input, passing it through the layers of a neural network, and producing an output (the prediction).

Mathematically, this is nothing more than a sequence of **linear transformations** followed by **nonlinear activations**.

---

## 2. **Single Neuron Recap**

For one neuron, the computation is:

$$
z = \sum_{i=1}^n w_i x_i + b \quad \text{(linear combination)}
$$

$$
a = f(z) \quad \text{(activation)}
$$

Where:

* $x_i$ = inputs
* $w_i$ = weights
* $b$ = bias
* $f$ = activation function (e.g., ReLU, sigmoid, tanh)

---

## 3. **One Layer (Vectorized Form)**

For a full layer with $m$ neurons, we write:

$$
\mathbf{z} = W \mathbf{x} + \mathbf{b}
$$

$$
\mathbf{a} = f(\mathbf{z})
$$

Where:

* $\mathbf{x} \in \mathbb{R}^n$: input vector
* $W \in \mathbb{R}^{m \times n}$: weight matrix
* $\mathbf{b} \in \mathbb{R}^m$: bias vector
* $f$: activation function applied elementwise

---

## 4. **Multi-Layer Forward Pass**

For a 3-layer fully connected network:

* Input: size $d$
* Hidden layer 1: size $h_1$
* Hidden layer 2: size $h_2$
* Output: size $k$

The forward pass equations are:

$$
\begin{aligned}
\mathbf{z}^{[1]} &= W^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \\
\mathbf{a}^{[1]} &= f(\mathbf{z}^{[1]}) \\[6pt]
\mathbf{z}^{[2]} &= W^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]} \\
\mathbf{a}^{[2]} &= f(\mathbf{z}^{[2]}) \\[6pt]
\mathbf{z}^{[3]} &= W^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]} \\
\mathbf{a}^{[3]} &= g(\mathbf{z}^{[3]}) \quad \text{(final output, often softmax/sigmoid)}
\end{aligned}
$$

Where:

* $W^{[l]}$, $b^{[l]}$: parameters of layer $l$
* $f$: hidden activation (e.g., ReLU, tanh)
* $g$: output activation (softmax, sigmoid, or identity for regression)

---

## 5. **Worked Example (Toy Network)**

### Setup:

* Input: 2 features $(x_1, x_2)$
* Hidden layer: 2 neurons, ReLU activation
* Output: 1 neuron, sigmoid activation

---

### Step 1. Input Vector

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

---

### Step 2. Hidden Layer

$$
z^{[1]} = W^{[1]} \mathbf{x} + b^{[1]}
$$

With:

$$
W^{[1]} = \begin{bmatrix} 0.5 & -0.3 \\ 0.8 & 0.2 \end{bmatrix}, 
\quad b^{[1]} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

So:

$$
z^{[1]} = \begin{bmatrix} 0.5x_1 - 0.3x_2 + 0.1 \\ 0.8x_1 + 0.2x_2 - 0.1 \end{bmatrix}
$$

Apply ReLU:

$$
a^{[1]} = \max(0, z^{[1]})
$$

---

### Step 3. Output Layer

$$
z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}
$$

With:

$$
W^{[2]} = \begin{bmatrix} 1.0 & -1.0 \end{bmatrix}, 
\quad b^{[2]} = 0.2
$$

So:

$$
z^{[2]} = 1.0 \cdot a_1^{[1]} - 1.0 \cdot a_2^{[1]} + 0.2
$$

Apply sigmoid:

$$
a^{[2]} = \frac{1}{1+e^{-z^{[2]}}}
$$

This is the final prediction.

---

## 6. **Key Intuition**

* The forward pass is **matrix multiplications + nonlinearities**.
* Each layer transforms inputs into a new feature space.
* By the output layer, the network has reshaped raw input into a form that is easier to separate (for classification) or map (for regression).

---

✅ **Summary:**
Forward pass = input → linear transformation (weights + biases) → activation → repeat for each layer → output prediction.
