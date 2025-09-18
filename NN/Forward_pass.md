Perfect — you’re ready to get into the **mechanics**.
Let’s walk carefully through the **forward pass** of an artificial neural network.

---

# 📘 The Forward Pass in ANNs

---

## 1. **Definition**

The **forward pass** is the process of taking an input, pushing it through the network (layer by layer), and producing an output (prediction).

It is literally just applying **linear algebra + nonlinearities** step by step.

---

## 2. **One Neuron Recap**

For a single neuron:

$$
z = \sum_{i=1}^n w_i x_i + b \quad \text{(linear combination)}
$$

$$
a = f(z) \quad \text{(activation)}
$$

* $x_i$ = inputs
* $w_i$ = weights
* $b$ = bias
* $f$ = activation (ReLU, sigmoid, etc.)

---

## 3. **One Layer (Vectorized)**

If you have **m neurons** in a layer, we can write:

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

Suppose we have a 3-layer fully connected network:

* Input layer: size $d$
* Hidden layer 1: size $h_1$
* Hidden layer 2: size $h_2$
* Output layer: size $k$

Then the forward pass is:

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

* $W^{[l]}$, $b^{[l]}$ = parameters of layer $l$
* $f$ = hidden activation (ReLU, tanh, etc.)
* $g$ = output activation (softmax, sigmoid, identity)

---

## 5. **Toy Example**

Imagine:

* Input: 2 features ($x_1, x_2$)
* Hidden layer: 2 neurons with ReLU
* Output: 1 neuron with sigmoid

Step-by-step:

1. **Input vector**:

   $$
   \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
   $$

2. **Hidden layer computation**:

   $$
   z^{[1]} = W^{[1]} \mathbf{x} + b^{[1]}
   $$

   If

   $$
   W^{[1]} = \begin{bmatrix} 0.5 & -0.3 \\ 0.8 & 0.2 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
   $$

   then

   $$
   z^{[1]} = \begin{bmatrix} 0.5x_1 - 0.3x_2 + 0.1 \\ 0.8x_1 + 0.2x_2 - 0.1 \end{bmatrix}
   $$

   Apply ReLU:

   $$
   a^{[1]} = \max(0, z^{[1]})
   $$

3. **Output layer computation**:
   Suppose output weights and bias are:

   $$
   W^{[2]} = \begin{bmatrix} 1.0 & -1.0 \end{bmatrix}, \quad b^{[2]} = 0.2
   $$

   Then

   $$
   z^{[2]} = 1.0 \cdot a_1^{[1]} - 1.0 \cdot a_2^{[1]} + 0.2
   $$

   Apply sigmoid:

   $$
   a^{[2]} = \frac{1}{1+e^{-z^{[2]}}}
   $$

That’s the prediction.

---

## 6. **Intuition**

* Forward pass is **just matrix multiplications + nonlinear functions**.
* Each layer maps the input into a new “feature space.”
* By the end, the network has transformed raw data into something linearly separable for classification (or into a regression output).

---

✅ **Summary**:
The forward pass = input → linear transform (weights+biases) → nonlinearity → repeat across layers → final prediction.

---

Do you want me to **code a forward pass from scratch in NumPy** (no frameworks) so you can see the actual numbers flow through?
