
---

# Recurrent Neural Networks (RNNs)
A Recurrent Neural Network (RNN) is a type of neural network designed for sequential data that processes inputs step-by-step while maintaining a hidden state (memory), and its main components are the input layer, hidden layer with recurrent connections, and output layer.
## 1. Introduction

A Recurrent Neural Network (RNN) is a type of neural network designed to handle sequential data like text, time series, or speech. Unlike a traditional feedforward neural network, which treats each input independently, RNNs have a "memory" that allows them to use information from previous steps in a sequence to influence the current output. This is achieved through a hidden state that is passed from one time step to the next, creating a loop in the network's structure.

A Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data by using an internal memory. Unlike traditional feedforward networks that treat inputs independently, RNNs leverage information from previous steps to influence the current output. This is made possible by a hidden state that is passed along from one time step to the next, forming a loop in the network's architecture.

---

## 2. How RNNs Work

At each time step RNNs process units with a fixed activation function. These units have an internal hidden state that acts as memory that retains information from previous time steps. This memory allows the network to store past knowledge and adapt based on new inputs.

Updating the Hidden State in RNNs:

1. **State Update:**
   $h_t = f(h_{t-1}, x_t)$
2. **Activation Function Application:**
   $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$
3. **Output Calculation:**
   $y_t = W_{hy}h_t$

How RNNs Work (alternate explanation):

* At each step, it takes the current input and the hidden state from the previous step to produce a new hidden state.
* The output at each time step is a function of the current hidden state.
* The same set of weights and biases are used across all time steps, a concept known as parameter sharing.

Formulas for a simple RNN:

* $h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$
* $y_t = g(W_{hy}h_t + b_y)$

---

## 3. Key Components of RNNs

1. **Recurrent Neurons** – The fundamental processing unit in RNNs that maintains information about previous inputs.
2. **RNN Unfolding** – Expanding the recurrent structure over time steps, enabling visualization and Backpropagation Through Time (BPTT).

Recurrent Neural Network Architecture (alternate formulation):

1. Hidden State: $h = \sigma(U \cdot X + W \cdot h_{t-1} + B)$
2. Output: $Y = O(V \cdot h + C)$
3. Overall: $Y = f(X, h, W, U, V, B, C)$

---

## 4. Training RNNs

### Backpropagation Through Time (BPTT)

Since RNNs process sequential data, Backpropagation Through Time (BPTT) is used to update parameters.

Steps:

1. Unroll the network across time steps.
2. Propagate gradients backward through each time step.
3. Compute gradients:
   $\frac{\partial L(\theta)}{\partial W} = \frac{\partial L(\theta)}{\partial h_3} \cdot \frac{\partial h_3}{\partial W}$
4. Handle dependencies explicitly and implicitly across hidden states.

---

## 5. Challenges: Vanishing and Exploding Gradient Problem

* **Vanishing Gradient:** Gradients shrink exponentially (sigmoid/tanh), making it hard to learn long-term dependencies.
* **Exploding Gradient:** Gradients grow uncontrollably, destabilizing training.

**Solution:** Use advanced architectures such as **LSTM** and **GRU**, which employ gates to selectively remember or forget information.

---

## 6. Types of RNNs

### Based on Input–Output Structure

1. **One-to-One** – Single input, single output (e.g., basic classification).
2. **One-to-Many** – Single input, multiple outputs (e.g., image captioning).
3. **Many-to-One** – Multiple inputs, single output (e.g., sentiment analysis).
4. **Many-to-Many** – Multiple inputs, multiple outputs (e.g., machine translation).

---

## 7. Variants of RNNs

1. **Vanilla RNN** – Basic form, limited by vanishing gradients.
2. **Bidirectional RNNs** – Capture both past and future context.
3. **LSTMs** – Use input, forget, and output gates to model long-term dependencies.
4. **GRUs** – Simplified LSTMs with combined gates, more efficient.

---

## 8. Why RNNs Outperform Old Techniques

* **Handling Context and Order:** Unlike feedforward models, RNNs process sequences with memory.
* **Parameter Sharing:** Same weights across time steps allow handling variable-length input.
* **Memory and Long-Term Dependencies:** LSTMs/GRUs overcome vanishing gradients to capture long-term relations.

---

## 9. RNN vs Feedforward Neural Networks

* **FNNs:** Process data in one direction, no memory, suitable for independent input tasks (e.g., image classification).
* **RNNs:** Incorporate feedback loops to remember past inputs, suitable for sequential tasks.

---

## 10. Applications

**Implementing a Text Generator Using RNNs**
RNNs can be used to build character-level text generators in TensorFlow/Keras, learning sequential patterns to generate new text character by character.

---
# Detailed Notes

An **Artificial Neural Network (ANN)** is not suitable for sequential data because it lacks a built-in concept of order or memory. It processes each input independently, treating a sequence of inputs as a "bag of words" without considering their temporal relationship. A **Recurrent Neural Network (RNN)**, on the other hand, is specifically designed for sequential data because it has a **hidden state** that acts as a memory, allowing it to use information from previous steps to inform its processing of the current input.

***

### Why ANNs Won't Work on Sequential Data

* **No Memory:** ANNs process each input vector in isolation. When you feed it a sequence of words, it has no way to remember the words it has already seen. For a sentence like "I am not happy," an ANN would analyze each word individually without understanding that the word "not" modifies the meaning of "happy."
* **Fixed Input Size:** ANNs require a fixed-size input. This poses a major problem for sequential data like sentences, which can vary greatly in length. To make the data fit, you would have to either truncate long sequences or pad short ones with zeros. Both methods result in a loss of information or the introduction of meaningless data.
* **Disregards Context:** Without memory, an ANN cannot understand how the context of a sequence builds over time. It can't learn long-range dependencies, which are critical for tasks like language translation or time-series forecasting. The output for a given input is not influenced by any previous inputs.

***

### Why RNNs Will Work on Sequential Data

* **Memory through Hidden State:** RNNs have a feedback loop that allows them to pass a **hidden state** from one time step to the next. This hidden state essentially functions as the network's short-term memory, carrying a summary of the sequence processed so far. 
* **Variable-Length Inputs:** Because an RNN processes data one element at a time, it can handle sequences of varying lengths without needing to pad or truncate the data. The same weights are applied at each time step, making the model scalable and flexible.
* **Captures Temporal Dependencies:** The recurrent nature of the hidden state allows an RNN to capture the **temporal relationships** between elements in a sequence. This means it can understand how the word "not" in a sentence modifies the meaning of a later word, which is essential for tasks like sentiment analysis and language modeling. Advanced RNNs like **LSTMs** and **GRUs** further improve on this by using gating mechanisms to control what information to remember or forget, effectively solving the vanishing gradient problem and allowing them to learn very long-term dependencies.



The concept of **I.I.D. (Independent and Identically Distributed)** is fundamentally at odds with the core purpose and design of a **Recurrent Neural Network (RNN)**. While many traditional machine learning models assume that data points are I.I.D., RNNs are explicitly built to handle the **non-I.I.D.** nature of sequential data.

***

### Why the I.I.D. Assumption is Violated in Sequential Data

Sequential data, by its very definition, is not independent. The value of an element in a sequence is highly dependent on the values of the elements that came before it. Consider these examples:

* **Language:** In the sentence "The weather is very **hot** today," the word "hot" is contextually dependent on the words "weather is very." A different set of preceding words, like "The food is very...", would likely lead to a different word like "tasty."
* **Time Series:** The stock price of a company on Tuesday is not independent of its price on Monday. It is directly influenced by the previous day's value.
* **Genetics:** A specific gene sequence is not a random collection of bases; the probability of a certain base appearing is dependent on the bases that precede it.

In all these cases, the independence component of the I.I.D. assumption is violated. The data points are **correlated** or **dependent** on each other.

***

### How RNNs Explicitly Work with Non-I.I.D. Data

RNNs are designed to model these dependencies, making them a perfect fit for non-I.I.D. sequential data. They achieve this through a **recurrent loop** and a **hidden state**, which acts as a form of memory.



1.  **Memory:** At each time step, an RNN processes the current input and the hidden state from the previous time step. This hidden state encapsulates information about the entire history of the sequence up to that point. This allows the network to learn and leverage the temporal dependencies in the data.
2.  **Parameter Sharing:** The same set of weights is used across all time steps. This is a crucial feature that allows the model to generalize patterns learned from one part of a sequence to another, regardless of the sequence's length. This is in contrast to a traditional feedforward network, which would need a new set of weights for every possible sequence length.

Essentially, RNNs were developed precisely because the I.I.D. assumption is not applicable to real-world sequential data. Their architecture, with its built-in memory and parameter sharing, allows them to model the dependencies and context that traditional models, which rely on the I.I.D. assumption, cannot.


---

# Backpropagation (BP) vs Backpropagation Through Time (BPTT)

## 🔹 1. Backpropagation (BP)

* **Where it’s used:**
  Feedforward Neural Networks (e.g., MLPs, CNNs).
* **Process:**

  * Error is calculated at the output layer.
  * Gradients are computed by applying the chain rule backward through the network’s layers.
  * Parameters (weights, biases) are updated using these gradients.
* **Key point:**
  BP is applied on **static architectures**, where inputs do not depend on time or previous states.

**Formula (for a single layer):**

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
$$

---

## 🔹 2. Backpropagation Through Time (BPTT)

* **Where it’s used:**
  Recurrent Neural Networks (RNNs) and their variants (LSTM, GRU).
* **Process:**

  * Since RNNs reuse the same weights across time steps, the network is **unrolled** across all time steps (like a deep feedforward network where each layer is a time step).
  * Gradients are computed not only across layers but also across **time**.
  * Errors from later time steps flow backward through earlier time steps.
* **Key point:**
  BPTT is BP **extended to handle sequences**, accounting for temporal dependencies.

**Formula (simplified for RNN):**

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W}
$$

---

## 🔹 3. Differences at a Glance

| Aspect            | Backpropagation (BP)                            | Backpropagation Through Time (BPTT)                           |
| ----------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| **Use case**      | Feedforward Neural Networks (FNNs, CNNs)        | Recurrent Neural Networks (RNNs, LSTMs, GRUs)                 |
| **Structure**     | Static (fixed layers, no recurrence)            | Unrolled in time (same weights repeated across steps)         |
| **Gradient flow** | Backward through layers only                    | Backward through layers **and** time steps                    |
| **Dependencies**  | Each input independent                          | Each input depends on past hidden states                      |
| **Main issues**   | Vanishing/exploding gradients in very deep nets | Vanishing/exploding gradients are worse due to long sequences |
| **Computation**   | Gradients per layer                             | Sum of gradients over all time steps                          |

---

## 🔹 4. Variants of BPTT

* **Full BPTT:** Backpropagate through the entire sequence (computationally expensive).
* **Truncated BPTT:** Backpropagate through only a fixed number of steps (reduces cost, often used in practice).

---

✅ In short:

* **BP** → Standard method for static networks.
* **BPTT** → Special case of BP adapted to RNNs, where the network is unrolled through time.

---


***

# Backpropagation Through Time (BPTT) Simplified

**What is BPTT?**  
Backpropagation Through Time (BPTT) is a method used to train Recurrent Neural Networks (RNNs). Unlike regular neural networks, RNNs process sequences of data by maintaining a memory of past inputs. BPTT extends the standard backpropagation algorithm by unfolding the RNN across time, allowing the network to learn dependencies that span multiple steps.

**What makes BPTT unique?**  
The key feature of BPTT is its *unfolding* of the RNN over several time steps, transforming it into a deep feedforward network where each layer equals a time step. This approach captures how early inputs influence later outputs, essential for sequence tasks like language modeling or time series prediction.

**The Forward Pass**  
At each time step $$ t $$:
- The hidden state $$ S_t $$ updates based on the current input $$ X_t $$ and previous state $$ S_{t-1} $$:
  $$
  S_t = g_1(W_x X_t + W_s S_{t-1})
  $$
- The output $$ Y_t $$ is computed from $$ S_t $$:
  $$
  Y_t = g_2(W_y S_t)
  $$

**The Backward Pass (Training)**  
To improve the model, BPTT computes gradients for all weights by considering how errors at time $$ t $$ depend on earlier states:
- Gradient w.r.t output weights $$ W_y $$ depends only on the current output.
- Gradients w.r.t hidden $$ W_s $$ and input weights $$ W_x $$ are sums over all previous states and inputs affecting $$ Y_t $$.

This means errors are propagated backward through all relevant time steps, enabling the network to learn long-term dependencies.

**Why use BPTT?**  
- **Captures temporal relationships:** crucial for sequential data like speech or language.
- **Precursor to advanced models:** used in training LSTM and GRU networks that handle longer sequences better.

**Challenges & Solutions**  
- **Vanishing gradients:** gradients shrink over many steps, causing the network to forget early inputs. Solutions include architectures like LSTM and techniques like gradient clipping.
- **Exploding gradients:** gradients grow too large, destabilizing training. Gradient clipping can address this.

***
in rnn do a calcultaion for this.

hand calculation

i love this book its amazing -1
i hate this book - 0

forward pass and full bptt.

size 3 embediing


Great! Here's a step-by-step hand calculation outline for the given sentences using an RNN with embedding size 3, covering the forward pass and full BPTT.

### Setup:
- Vocabulary includes words: "i", "love", "this", "book", "its", "amazing", "hate"
- Embedding size: 3
- Dummy random initial embeddings (3 values per word)
- Vanilla RNN with hidden size (for simplicity) = 3
- Activation function: tanh
- Output layer is a simple linear + sigmoid for binary classification (sentiment 1 or 0)
- Loss: Binary cross-entropy

### Sentences:
1. "i love this book its amazing" → label 1  
2. "i hate this book" → label 0

***

## Step 1: Initialize embeddings and parameters

Assume these embeddings (random example):

| Word    | Embedding (3-dim)    |
|---------|----------------------|
| i       | [0.1, 0.2, 0.3]      |
| love    | [0.4, 0.1, 0.5]      |
| this    | [0.3, 0.2, 0.1]      |
| book    | [0.6, 0.1, 0.4]      |
| its     | [0.1, 0.3, 0.6]      |
| amazing | [0.5, 0.4, 0.2]      |
| hate    | [0.7, 0.8, 0.1]      |

RNN weight matrices to initialize:

- $$ W_{xh} $$: input to hidden (3x3)
- $$ W_{hh} $$: hidden to hidden (3x3)
- $$ b_h $$: hidden bias (3x1)
- $$ W_{ho} $$: hidden to output (1x3)
- $$ b_o $$: output bias (scalar)

Random values:

$$
W_{xh} = \begin{bmatrix}0.2 & 0.4 & 0.1 \\ 0.1 & 0.3 & 0.5 \\ 0.4 & 0.2 & 0.3\end{bmatrix}
\quad
W_{hh} = \begin{bmatrix}0.3 & 0.2 & 0.1 \\ 0.5 & 0.4 & 0.2 \\ 0.1 & 0.3 & 0.4\end{bmatrix}
\quad
b_h = \begin{bmatrix}0.1 \\ 0.1 \\ 0.1 \end{bmatrix}
$$

$$
W_{ho} = \begin{bmatrix}0.3 & 0.2 & 0.4\end{bmatrix}
\quad
b_o = 0.1
$$

***

## Step 2: Forward Pass (sentence 1: "i love this book its amazing")

Initialize hidden state $$ h_0 = ^T $$.

For each timestep $$t$$, calculate:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = \sigma(W_{ho} h_t + b_o)
$$

Since full sentence sentiment is output at last timestep, only $$ y_T $$ is used for loss.

***
Continuing with the forward pass step-by-step for sentence 1: "i love this book its amazing" (6 tokens).

***

### Timestep 1: word = "i", embedding $$x_1 = [0.1, 0.2, 0.3]^T$$, hidden $$h_0 = ^T$$

$$
z_1 = W_{xh} x_1 + W_{hh} h_0 + b_h
= W_{xh} x_1 + b_h
$$

Calculate $$W_{xh} x_1$$:

$$
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\
0.1 & 0.3 & 0.5 \\
0.4 & 0.2 & 0.3
\end{bmatrix}
\begin{bmatrix}
0.1 \\ 0.2 \\ 0.3
\end{bmatrix}
=
\begin{bmatrix}
0.2 \times 0.1 + 0.4 \times 0.2 + 0.1 \times 0.3 \\
0.1 \times 0.1 + 0.3 \times 0.2 + 0.5 \times 0.3 \\
0.4 \times 0.1 + 0.2 \times 0.2 + 0.3 \times 0.3
\end{bmatrix}
=
\begin{bmatrix}
0.02 + 0.08 + 0.03 \\
0.01 + 0.06 + 0.15 \\
0.04 + 0.04 + 0.09
\end{bmatrix}
=
\begin{bmatrix}
0.13 \\ 0.22 \\ 0.17
\end{bmatrix}
$$

Add bias $$b_h$$:

$$
z_1 = \begin{bmatrix}0.13 \\ 0.22 \\ 0.17\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.1 \\ 0.1\end{bmatrix} = \begin{bmatrix}0.23 \\ 0.32 \\ 0.27\end{bmatrix}
$$

Apply $$\tanh$$ to get $$h_1$$:

$$
h_1 = \tanh(z_1) \approx \begin{bmatrix} \tanh(0.23) \\ \tanh(0.32) \\ \tanh(0.27) \end{bmatrix} 
\approx \begin{bmatrix}0.226 \\ 0.309 \\ 0.263\end{bmatrix}
$$

***

### Timestep 2: word = "love", $$x_2 = [0.4, 0.1, 0.5]^T$$

$$
z_2 = W_{xh} x_2 + W_{hh} h_1 + b_h
$$

Calculate $$W_{xh} x_2$$:

$$
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\
0.1 & 0.3 & 0.5 \\
0.4 & 0.2 & 0.3
\end{bmatrix}
\begin{bmatrix}
0.4 \\ 0.1 \\ 0.5
\end{bmatrix}
=
\begin{bmatrix}
0.2 \times 0.4 + 0.4 \times 0.1 + 0.1 \times 0.5 \\
0.1 \times 0.4 + 0.3 \times 0.1 + 0.5 \times 0.5 \\
0.4 \times 0.4 + 0.2 \times 0.1 + 0.3 \times 0.5
\end{bmatrix}
=
\begin{bmatrix}
0.08 + 0.04 + 0.05 \\
0.04 + 0.03 + 0.25 \\
0.16 + 0.02 + 0.15
\end{bmatrix}
=
\begin{bmatrix}
0.17 \\ 0.32 \\ 0.33
\end{bmatrix}
$$

Calculate $$W_{hh} h_1$$:

$$
\begin{bmatrix}
0.3 & 0.2 & 0.1 \\
0.5 & 0.4 & 0.2 \\
0.1 & 0.3 & 0.4
\end{bmatrix}
\begin{bmatrix}
0.226 \\ 0.309 \\ 0.263
\end{bmatrix}
=
\begin{bmatrix}
0.3 \times 0.226 + 0.2 \times 0.309 + 0.1 \times 0.263 \\
0.5 \times 0.226 + 0.4 \times 0.309 + 0.2 \times 0.263 \\
0.1 \times 0.226 + 0.3 \times 0.309 + 0.4 \times 0.263
\end{bmatrix}
\approx
\begin{bmatrix}
0.0678 + 0.0618 + 0.0263 \\
0.113 + 0.1236 + 0.0526 \\
0.0226 + 0.0927 + 0.1052
\end{bmatrix}
=
\begin{bmatrix}
0.156 \\ 0.289 \\ 0.221
\end{bmatrix}
$$

Add all + bias:

$$
z_2 = \begin{bmatrix}0.17 \\ 0.32 \\ 0.33\end{bmatrix} + \begin{bmatrix}0.156 \\ 0.289 \\ 0.221\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.1 \\ 0.1\end{bmatrix} = \begin{bmatrix}0.426 \\ 0.709 \\ 0.651\end{bmatrix}
$$

Apply $$\tanh$$:

$$
h_2 = \tanh(z_2) \approx \begin{bmatrix}0.402 \\ 0.61 \\ 0.572 \end{bmatrix}
$$

***
Continuing the forward pass for sentence 1:

***

### Timestep 3: word = "this", $$x_3 = [0.3, 0.2, 0.1]^T$$

Calculate $$W_{xh} x_3$$:

$$
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\
0.1 & 0.3 & 0.5 \\
0.4 & 0.2 & 0.3
\end{bmatrix}
\begin{bmatrix}
0.3 \\ 0.2 \\ 0.1
\end{bmatrix}
=
\begin{bmatrix}
0.2 \times 0.3 + 0.4 \times 0.2 + 0.1 \times 0.1 \\
0.1 \times 0.3 + 0.3 \times 0.2 + 0.5 \times 0.1 \\
0.4 \times 0.3 + 0.2 \times 0.2 + 0.3 \times 0.1
\end{bmatrix}
=
\begin{bmatrix}
0.06 + 0.08 + 0.01 \\
0.03 + 0.06 + 0.05 \\
0.12 + 0.04 + 0.03
\end{bmatrix}
=
\begin{bmatrix}
0.15 \\ 0.14 \\ 0.19
\end{bmatrix}
$$

Calculate $$W_{hh} h_2$$:

$$
\begin{bmatrix}
0.3 & 0.2 & 0.1 \\
0.5 & 0.4 & 0.2 \\
0.1 & 0.3 & 0.4
\end{bmatrix}
\begin{bmatrix}
0.402 \\ 0.61 \\ 0.572
\end{bmatrix}
=
\begin{bmatrix}
0.3 \times 0.402 + 0.2 \times 0.61 + 0.1 \times 0.572 \\
0.5 \times 0.402 + 0.4 \times 0.61 + 0.2 \times 0.572 \\
0.1 \times 0.402 + 0.3 \times 0.61 + 0.4 \times 0.572
\end{bmatrix}
\approx
\begin{bmatrix}
0.1206 + 0.122 + 0.0572 \\
0.201 + 0.244 + 0.114 \\
0.0402 + 0.183 + 0.229
\end{bmatrix}
=
\begin{bmatrix}
0.300 \\ 0.559 \\ 0.452
\end{bmatrix}
$$

Add all + bias:

$$
z_3 = \begin{bmatrix}0.15 \\ 0.14 \\ 0.19\end{bmatrix} + \begin{bmatrix}0.300 \\ 0.559 \\ 0.452\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.1 \\ 0.1\end{bmatrix} = \begin{bmatrix}0.55 \\ 0.799 \\ 0.742\end{bmatrix}
$$

Apply $$\tanh$$:

$$
h_3 = \tanh(z_3) \approx \begin{bmatrix}0.501 \\ 0.663 \\ 0.630\end{bmatrix}
$$

***

### Timestep 4: word = "book", $$x_4 = [0.6, 0.1, 0.4]^T$$

Calculate $$W_{xh} x_4$$:

$$
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\
0.1 & 0.3 & 0.5 \\
0.4 & 0.2 & 0.3
\end{bmatrix}
\begin{bmatrix}
0.6 \\ 0.1 \\ 0.4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \times 0.6 + 0.4 \times 0.1 + 0.1 \times 0.4 \\
0.1 \times 0.6 + 0.3 \times 0.1 + 0.5 \times 0.4 \\
0.4 \times 0.6 + 0.2 \times 0.1 + 0.3 \times 0.4
\end{bmatrix}
=
\begin{bmatrix}
0.12 + 0.04 + 0.04 \\
0.06 + 0.03 + 0.20 \\
0.24 + 0.02 + 0.12
\end{bmatrix}
=
\begin{bmatrix}
0.20 \\ 0.29 \\ 0.38
\end{bmatrix}
$$

Calculate $$W_{hh} h_3$$:

$$
\begin{bmatrix}
0.3 & 0.2 & 0.1 \\
0.5 & 0.4 & 0.2 \\
0.1 & 0.3 & 0.4
\end{bmatrix}
\begin{bmatrix}
0.501 \\ 0.663 \\ 0.630
\end{bmatrix}
=
\begin{bmatrix}
0.3 \times 0.501 + 0.2 \times 0.663 + 0.1 \times 0.630 \\
0.5 \times 0.501 + 0.4 \times 0.663 + 0.2 \times 0.630 \\
0.1 \times 0.501 + 0.3 \times 0.663 + 0.4 \times 0.630
\end{bmatrix}
\approx
\begin{bmatrix}
0.1503 + 0.1326 + 0.063 \\
0.2505 + 0.2652 + 0.126 \\
0.0501 + 0.1989 + 0.252
\end{bmatrix}
=
\begin{bmatrix}
0.346 \\ 0.642 \\ 0.501
\end{bmatrix}
$$

Add all + bias:

$$
z_4 = \begin{bmatrix}0.20 \\ 0.29 \\ 0.38\end{bmatrix} + \begin{bmatrix}0.346 \\ 0.642 \\ 0.501\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.1 \\ 0.1\end{bmatrix} = \begin{bmatrix}0.646 \\ 1.032 \\ 0.981\end{bmatrix}
$$

Apply $$\tanh$$:

$$
h_4 = \tanh(z_4) \approx \begin{bmatrix}0.570 \\ 0.775 \\ 0.755\end{bmatrix}
$$

***

Continuing with timestep 5 and 6:

***

### Timestep 5: word = "its", $$x_5 = [0.1, 0.3, 0.6]^T$$

Calculate $$W_{xh} x_5$$:

$$
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\
0.1 & 0.3 & 0.5 \\
0.4 & 0.2 & 0.3
\end{bmatrix}
\begin{bmatrix}
0.1 \\ 0.3 \\ 0.6
\end{bmatrix}
=
\begin{bmatrix}
0.2 \times 0.1 + 0.4 \times 0.3 + 0.1 \times 0.6 \\
0.1 \times 0.1 + 0.3 \times 0.3 + 0.5 \times 0.6 \\
0.4 \times 0.1 + 0.2 \times 0.3 + 0.3 \times 0.6
\end{bmatrix}
=
\begin{bmatrix}
0.02 + 0.12 + 0.06 \\
0.01 + 0.09 + 0.30 \\
0.04 + 0.06 + 0.18
\end{bmatrix}
=
\begin{bmatrix}
0.20 \\ 0.40 \\ 0.28
\end{bmatrix}
$$

Calculate $$W_{hh} h_4$$:

$$
\begin{bmatrix}
0.3 & 0.2 & 0.1 \\
0.5 & 0.4 & 0.2 \\
0.1 & 0.3 & 0.4
\end{bmatrix}
\begin{bmatrix}
0.570 \\ 0.775 \\ 0.755
\end{bmatrix}
=
\begin{bmatrix}
0.3 \times 0.570 + 0.2 \times 0.775 + 0.1 \times 0.755 \\
0.5 \times 0.570 + 0.4 \times 0.775 + 0.2 \times 0.755 \\
0.1 \times 0.570 + 0.3 \times 0.775 + 0.4 \times 0.755
\end{bmatrix}
\approx
\begin{bmatrix}
0.171 + 0.155 + 0.0755 \\
0.285 + 0.31 + 0.151 \\
0.057 + 0.232 + 0.302
\end{bmatrix}
=
\begin{bmatrix}
0.4015 \\ 0.746 \\ 0.591
\end{bmatrix}
$$

Add all + bias:

$$
z_5 = \begin{bmatrix}0.20 \\ 0.40 \\ 0.28\end{bmatrix} + \begin{bmatrix}0.4015 \\ 0.746 \\ 0.591\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.1 \\ 0.1\end{bmatrix} = \begin{bmatrix}0.7015 \\ 1.246 \\ 0.971\end{bmatrix}
$$

Apply $$\tanh$$:

$$
h_5 = \tanh(z_5) \approx \begin{bmatrix}0.605 \\ 0.847 \\ 0.75\end{bmatrix}
$$

***

### Timestep 6: word = "amazing", $$x_6 = [0.5, 0.4, 0.2]^T$$

Calculate $$W_{xh} x_6$$:

$$
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\
0.1 & 0.3 & 0.5 \\
0.4 & 0.2 & 0.3
\end{bmatrix}
\begin{bmatrix}
0.5 \\ 0.4 \\ 0.2
\end{bmatrix}
=
\begin{bmatrix}
0.2 \times 0.5 + 0.4 \times 0.4 + 0.1 \times 0.2 \\
0.1 \times 0.5 + 0.3 \times 0.4 + 0.5 \times 0.2 \\
0.4 \times 0.5 + 0.2 \times 0.4 + 0.3 \times 0.2
\end{bmatrix}
=
\begin{bmatrix}
0.10 + 0.16 + 0.02 \\
0.05 + 0.12 + 0.10 \\
0.20 + 0.08 + 0.06
\end{bmatrix}
=
\begin{bmatrix}
0.28 \\ 0.27 \\ 0.34
\end{bmatrix}
$$

Calculate $$W_{hh} h_5$$:

$$
\begin{bmatrix}
0.3 & 0.2 & 0.1 \\
0.5 & 0.4 & 0.2 \\
0.1 & 0.3 & 0.4
\end{bmatrix}
\begin{bmatrix}
0.605 \\ 0.847 \\ 0.75
\end{bmatrix}
=
\begin{bmatrix}
0.3 \times 0.605 + 0.2 \times 0.847 + 0.1 \times 0.75 \\
0.5 \times 0.605 + 0.4 \times 0.847 + 0.2 \times 0.75 \\
0.1 \times 0.605 + 0.3 \times 0.847 + 0.4 \times 0.75
\end{bmatrix}
\approx
\begin{bmatrix}
0.1815 + 0.169 + 0.075 \\
0.3025 + 0.339 + 0.15 \\
0.0605 + 0.254 + 0.3
\end{bmatrix}
=
\begin{bmatrix}
0.426 \\ 0.791 \\ 0.615
\end{bmatrix}
$$

Add all + bias:

$$
z_6 = \begin{bmatrix}0.28 \\ 0.27 \\ 0.34\end{bmatrix} + \begin{bmatrix}0.426 \\ 0.791 \\ 0.615\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.1 \\ 0.1\end{bmatrix} = \begin{bmatrix}0.806 \\ 1.161 \\ 1.055\end{bmatrix}
$$

Apply $$\tanh$$:

$$
h_6 = \tanh(z_6) \approx \begin{bmatrix}0.667 \\ 0.821 \\ 0.784\end{bmatrix}
$$

***

### Final output $$y_6$$:

$$
y_6 = \sigma(W_{ho} h_6 + b_o) = \sigma\left(0.3 \times 0.667 + 0.2 \times 0.821 + 0.4 \times 0.784 + 0.1\right)
$$

Calculate:

$$
0.3 \times 0.667 = 0.200, \quad 0.2 \times 0.821 = 0.164, \quad 0.4 \times 0.784 = 0.314
$$

Sum:

$$
0.200 + 0.164 + 0.314 + 0.1 = 0.778
$$

Apply sigmoid:

$$
y_6 = \frac{1}{1+e^{-0.778}} \approx 0.685
$$

Output prediction: $$0.685$$

***

Now, label is 1, so loss:

$$
\text{Loss} = -\log(y_6) = -\log(0.685) \approx 0.377
$$

***

Shall I continue with full BPTT calculation for this sentence now?I will proceed with detailed backpropagation through time (BPTT) for the sentence "i love this book its amazing" using the forward pass results. We will compute gradients for all parameters and embeddings step-by-step.

***

### Output error derivative:

Label: 1  
Output $$y_6 = 0.685$$

$$
\frac{\partial L}{\partial y_6} = -\frac{1}{y_6} = -\frac{1}{0.685} \approx -1.46
$$
Since the output activation is sigmoid with output $$y_6$$:

$$
\frac{\partial y_6}{\partial z_o} = y_6 (1 - y_6) = 0.685 \times 0.315 = 0.216
$$

Let $$z_o = W_{ho} h_6 + b_o$$.

Chain rule:

$$
\delta_o = \frac{\partial L}{\partial z_o} = \frac{\partial L}{\partial y_6} \times \frac{\partial y_6}{\partial z_o} = -1.46 \times 0.216 = -0.316
$$

***

### Gradients at output layer:

$$
\frac{\partial L}{\partial W_{ho}} = \delta_o \times h_6^T = -0.316 \times [0.667, 0.821, 0.784] = [-0.211, -0.259, -0.248]
$$

$$
\frac{\partial L}{\partial b_o} = \delta_o = -0.316
$$

***

### Gradient with respect to $$h_6$$:

$$
\delta h_6 = W_{ho}^T \times \delta_o = \begin{bmatrix}0.3 \\ 0.2 \\ 0.4\end{bmatrix} \times -0.316 = \begin{bmatrix} -0.095 \\ -0.063 \\ -0.126 \end{bmatrix}
$$

***

### Backpropagate through tanh at $$h_6$$:

Use $$h_6 = \tanh(z_6)$$, derivative $$\frac{\partial h}{\partial z} = 1 - h^2$$:

$$
1 - h_6^2 = 1 - \begin{bmatrix}0.667^2 \\ 0.821^2 \\ 0.784^2\end{bmatrix} = \begin{bmatrix}0.555 \\ 0.325 \\ 0.386\end{bmatrix}
$$

Elementwise multiply:

$$
\delta z_6 = \delta h_6 \odot (1 - h_6^2) = \begin{bmatrix} -0.095 \\ -0.063 \\ -0.126 \end{bmatrix} \odot \begin{bmatrix}0.555 \\ 0.325 \\ 0.386\end{bmatrix} = \begin{bmatrix} -0.053 \\ -0.020 \\ -0.049 \end{bmatrix}
$$

***

### Calculate gradients for parameters at timestep 6:

$$
\frac{\partial L}{\partial W_{xh}} += \delta z_6 \times x_6^T
$$

$$
\frac{\partial L}{\partial W_{hh}} += \delta z_6 \times h_5^T
$$

$$
\frac{\partial L}{\partial b_h} += \delta z_6
$$

Calculate each:

$$
\delta z_6 = \begin{bmatrix} -0.053 \\ -0.020 \\ -0.049 \end{bmatrix}, \quad x_6^T = [0.5, 0.4, 0.2]
$$

$$
\delta z_6 x_6^T =
\begin{bmatrix}
-0.053 \times 0.5 & -0.053 \times 0.4 & -0.053 \times 0.2 \\
-0.020 \times 0.5 & -0.020 \times 0.4 & -0.020 \times 0.2 \\
-0.049 \times 0.5 & -0.049 \times 0.4 & -0.049 \times 0.2
\end{bmatrix}
=
\begin{bmatrix}
-0.0265 & -0.0212 & -0.0106 \\
-0.0100 & -0.0080 & -0.0040 \\
-0.0245 & -0.0196 & -0.0098
\end{bmatrix}
$$

Similarly:

$$
h_5^T = [0.605, 0.847, 0.75]
$$

$$
\delta z_6 h_5^T = 
\begin{bmatrix}
-0.053 \times 0.605 & -0.053 \times 0.847 & -0.053 \times 0.75 \\
-0.020 \times 0.605 & -0.020 \times 0.847 & -0.020 \times 0.75 \\
-0.049 \times 0.605 & -0.049 \times 0.847 & -0.049 \times 0.75
\end{bmatrix}
$$

Calculating each:

$$
=
\begin{bmatrix}
-0.032 & -0.045 & -0.040 \\
-0.012 & -0.017 & -0.015 \\
-0.030 & -0.041 & -0.037
\end{bmatrix}
$$

Bias gradient:

$$
\frac{\partial L}{\partial b_h} = \delta z_6 = [-0.053, -0.020, -0.049]
$$

***

This completes gradient computation for timestep 6. Next, propagate $$\delta z_6$$ back to timestep 5's hidden state and continue similarly for all previous timesteps through BPTT.

Would you like me to continue through all timesteps?Continuing full BPTT for the previous timesteps, we backpropagate the error through the hidden states.

***

### Backpropagation to timestep 5:

Hidden state error combines:

- $$\delta h_5$$ from output layer's backprop via $$W_{hh}^T \delta z_6$$
- Plus propagated error from future states

Calculate:

$$
\delta h_5 = W_{hh}^T \delta z_6 + \text{previous } \delta h_5
$$

Using

$$
W_{hh}^T = \begin{bmatrix}0.3 & 0.5 & 0.1 \\ 0.2 & 0.4 & 0.3 \\ 0.1 & 0.2 & 0.4\end{bmatrix}
$$

and $$\delta z_6 = [-0.053, -0.020, -0.049]^T$$:

$$
\delta h_5 = W_{hh}^T \delta z_6 = 
\begin{bmatrix}
0.3 & 0.5 & 0.1 \\
0.2 & 0.4 & 0.3 \\
0.1 & 0.2 & 0.4
\end{bmatrix}
\begin{bmatrix}
-0.053 \\ -0.020 \\ -0.049
\end{bmatrix}
= 
\begin{bmatrix}
0.3 \times -0.053 + 0.5 \times -0.020 + 0.1 \times -0.049 \\
0.2 \times -0.053 + 0.4 \times -0.020 + 0.3 \times -0.049 \\
0.1 \times -0.053 + 0.2 \times -0.020 + 0.4 \times -0.049
\end{bmatrix}
\approx
\begin{bmatrix}
-0.016 - 0.010 - 0.005 \\
-0.011 - 0.008 - 0.015 \\
-0.005 - 0.004 - 0.020
\end{bmatrix}
= 
\begin{bmatrix}
-0.031 \\ -0.034 \\ -0.029
\end{bmatrix}
$$

***

### Backprop through tanh at timestep 5 using $$h_5$$:

Recall:

$$
1 - h_5^2 = 1 - \begin{bmatrix}0.605^2 \\ 0.847^2 \\ 0.75^2\end{bmatrix} \approx \begin{bmatrix}0.634 \\ 0.281 \\ 0.438\end{bmatrix}
$$

Elementwise multiply:

$$
\delta z_5 = \delta h_5 \odot (1 - h_5^2) = \begin{bmatrix} -0.031 \\ -0.034 \\ -0.029 \end{bmatrix} \odot \begin{bmatrix}0.634 \\ 0.281 \\ 0.438 \end{bmatrix} = \begin{bmatrix} -0.020 \\ -0.010 \\ -0.013 \end{bmatrix}
$$

***

### Repeat gradient calculations for timestep 5 parameters, then propagate $$\delta z_5$$ to timestep 4.

***

This pattern continues through each timestep by propagating errors backward, calculating parameter gradients, and updating.

Here is a summary of the full BPTT process for the sentence "i love this book its amazing," following the detailed example of timestep 6 and 5 backpropagation:

***

### BPTT summary:

1. **Initialize** at last timestep (6):
   - Compute output layer error $$\delta_o$$ from loss.
   - Backpropagate through output weights to get $$\delta h_6$$.
   - Backpropagate through tanh to get $$\delta z_6$$.
   - Compute gradients $$ \partial L / \partial W_{xh}, W_{hh}, b_h $$ for timestep 6.
2. **For each previous timestep $$t = 5,4,3,2,1$$**:
   - Backpropagate $$ \delta z_{t+1} $$ through $$ W_{hh}^T $$ to get $$\delta h_t$$.
   - Add any gradient contributions from further in the network (if multiple output steps).
   - Backpropagate through tanh to get $$\delta z_t$$.
   - Compute gradients $$ \partial L / \partial W_{xh}, W_{hh}, b_h $$ for timestep $$t$$.
3. **Also compute gradients wrt inputs (embedding vectors) for each word $$x_t$$ as:**
   $$
   \frac{\partial L}{\partial x_t} = W_{xh}^T \delta z_t
   $$
   This is useful if embeddings are trainable.
4. **Sum gradients over all timesteps** for each parameter.
5. **Update parameters** using gradient descent or other optimization techniques.

***

### Key points:

- This is a chain rule application through time, propagating error backwards from output to each previous hidden state.
- The gradients at one timestep depend on gradients from the next timestep via $$W_{hh}^T$$.
- The nonlinearity derivative (tanh) controls gradient flow to prevent exploding or vanishing gradients.

***

If desired, this full BPTT can be done for the second sentence "i hate this book" (label 0) similarly, and gradients accumulated over both.

Please let me know if a full numerical step-by-step for each timestep and parameter is preferred or a conceptual walkthrough like above suffices.For the sentence "i love this book its amazing" with label 1, using embedding size 3 and a simple vanilla RNN setup, the forward pass calculates the hidden states at each time step via $$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$ and produces an output $$y_6 = \sigma(W_{ho} h_6 + b_o) \approx 0.685$$, yielding a cross-entropy loss of about 0.377.

Backpropagation through time (BPTT) involves computing gradients starting from the output error, propagating back through the output layer, the tanh activations, and the hidden states across all time steps. At each time step, gradients for weights $$W_{xh}$$, $$W_{hh}$$, biases $$b_h$$, as well as the embeddings $$x_t$$ are calculated using the chain rule.

The process sums gradients over all time steps and updates parameters accordingly. This example was demonstrated in detail for the last two steps and a conceptual summary provided for the full BPTT.


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

