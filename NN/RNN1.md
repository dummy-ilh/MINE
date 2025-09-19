
---

# Recurrent Neural Networks (RNNs)

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

