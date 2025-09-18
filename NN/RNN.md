
What is a Recurrent Neural Network (RNN)?

A Recurrent Neural Network (RNN) is a type of neural network designed to handle sequential data or time series data, like text, audio, or video. Unlike a traditional Feedforward Neural Network where information flows in a single direction, an RNN has a "loop" that allows information to persist and be used in future calculations. This gives it a form of "memory." 🧠
Think of it like a human reading a sentence: you understand the meaning of the current word by also remembering the words that came before it. An RNN does the same, using the output from a previous step as an additional input for the current step.
Image of recurrent neural network architecture diagram

To better understand this "loop," we often unroll the RNN over time. This makes it look like a very deep network where each layer corresponds to a time step.
Image of unrolled recurrent neural network diagram


The Math Behind an RNN

The core of an RNN is its ability to update a hidden state at each time step, which acts as the network's memory.
At each time step t, the RNN takes two inputs:
The current input, xt​.
The hidden state from the previous time step, ht−1​.
The new hidden state, ht​, is calculated using a set of learned weights:
ht​=tanh(Whh​ht−1​+Wxh​xt​+bh​)
Where:
Whh​ is the weight matrix for the recurrent hidden state.
Wxh​ is the weight matrix for the input.
bh​ is the bias vector.
The tanh function (or sometimes ReLU) is the activation function.
The output, yt​, at time step t is then calculated based on the new hidden state:
yt​=Why​ht​+by​
This output can be a prediction, a word, or whatever the task requires.

Training an RNN

Training an RNN is similar to training a regular neural network, but it uses a specialized version of backpropagation called Backpropagation Through Time (BPTT). The error at the final time step is propagated backward through the unrolled network, allowing the model to update the weights based on the loss from all time steps.

Challenges with RNNs

Vanishing Gradient Problem: During BPTT, the gradients (used to update weights) can become extremely small. This makes it difficult for the network to learn long-term dependencies, as the updates from early time steps are effectively "lost."
Exploding Gradient Problem: Conversely, the gradients can become extremely large, leading to unstable learning and the network's weights "exploding" to a value of NaN (Not a Number). This is less common but can be managed by gradient clipping, which caps the maximum value of the gradients.

Types of RNNs and Their Evolution

The challenges of the vanishing/exploding gradient problem led to the development of more advanced architectures.

1. Long Short-Term Memory (LSTM)

An LSTM is a specialized RNN that was explicitly designed to handle long-term dependencies. It achieves this by using a "cell state" that runs through the entire chain and a series of "gates" that regulate the flow of information. The gates are special neural networks themselves that use a sigmoid function to output values between 0 and 1, acting as a filter.
Forget Gate: Decides what information to throw away from the cell state.
Input Gate: Decides what new information to store in the cell state.
Output Gate: Decides what part of the cell state to output to the hidden state.
![]("Image of http://googleusercontent.com/image_collection/image_retrieval/735203521162558814")

2. Gated Recurrent Unit (GRU)

A GRU is a simplified version of an LSTM. It combines the forget and input gates into a single update gate and also merges the cell state and hidden state. GRUs are simpler to compute and train but offer comparable performance to LSTMs on many tasks.
Image of gated recurrent unit (GRU) diagram


Applications of RNNs

RNNs, and more specifically LSTMs and GRUs, are used for a wide range of tasks involving sequential data:
Natural Language Processing (NLP): Machine translation, text generation (e.g., creating summaries), and sentiment analysis.
Speech Recognition: Converting audio signals into text.
Time Series Prediction: Forecasting stock prices or weather patterns.
Video Analysis: Understanding actions in video frames.
It looks like your text file was corrupted. I've re-formatted your numerical example of an RNN into a clean, markdown-friendly format. Here is the full text, ready for use.

***

## 📘 A Numerical Example of an RNN

Let's walk through a simple, two-word sequence ("hi there") to see how an RNN's internal calculations work. Our goal is to predict the next word in the sequence.

### 1. The Setup

* **Vocabulary**: "hi", "there", "class", "!"
* **One-hot encoding**:
    * `hi` = $[1, 0, 0, 0]$
    * `there` = $[0, 1, 0, 0]$
    * `class` = $[0, 0, 1, 0]$
    * `!` = $[0, 0, 0, 1]$
* **Initial Hidden State ($h_0$)**: $[0, 0]^T$
* **Assigned Weights and Biases**:
    * Input-to-hidden weights ($W_{xh}$): $\begin{pmatrix} 0.5 & 0.1 \\ -0.2 & 0.8 \end{pmatrix}$
    * Hidden-to-hidden weights ($W_{hh}$): $\begin{pmatrix} 0.3 & 0.4 \\ 0.6 & -0.1 \end{pmatrix}$
    * Output-to-hidden weights ($W_{hy}$): $\begin{pmatrix} 0.1 & 0.7 \\ -0.3 & 0.2 \\ 0.5 & 0.4 \\ -0.6 & -0.1 \end{pmatrix}$
    * Hidden bias ($b_h$): $[0.1, 0.1]^T$
    * Output bias ($b_y$): $[0, 0, 0, 0]^T$
* **Activation Function**: $\tanh$ for the hidden state.

---

### 2. Time Step 1: Processing "hi"

The network processes the first word, "hi". Since this is the first step, the previous hidden state, $h_0$, is a vector of zeros.

* **Input**: $x_1 = [1, 0, 0, 0]^T$
* **Previous hidden state**: $h_0 = [0, 0]^T$

We calculate the new hidden state, $h_1$:

$$h_1 = \tanh(W_{hh} h_0 + W_{xh} x_1 + b_h)$$$$h_1 = \tanh\left(\begin{pmatrix} 0.3 & 0.4 \\ 0.6 & -0.1 \end{pmatrix}\begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.5 & 0.1 \\ -0.2 & 0.8 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix}\right)$$$$h_1 = \tanh\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.5 \\ -0.2 \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix}\right)$$$$h_1 = \tanh\left(\begin{pmatrix} 0.6 \\ -0.1 \end{pmatrix}\right)$$$$h_1 \approx \begin{pmatrix} 0.537 \\ -0.099 \end{pmatrix}$$

This vector, $h_1$, is the network's "memory" after seeing "hi".

---

### 3. Time Step 2: Processing "there"

Now, the network processes "there" and, critically, uses the **hidden state from the previous step** ($h_1$).

* **Input**: $x_2 = [0, 1, 0, 0]^T$
* **Previous hidden state**: $h_1 = [0.537, -0.099]^T$

We calculate the next hidden state, $h_2$:

$$h_2 = \tanh(W_{hh} h_1 + W_{xh} x_2 + b_h)$$$$h_2 = \tanh\left(\begin{pmatrix} 0.3 & 0.4 \\ 0.6 & -0.1 \end{pmatrix}\begin{pmatrix} 0.537 \\ -0.099 \end{pmatrix} + \begin{pmatrix} 0.5 & 0.1 \\ -0.2 & 0.8 \end{pmatrix}\begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix}\right)$$$$h_2 = \tanh\left(\begin{pmatrix} (0.3 \times 0.537) + (0.4 \times -0.099) \\ (0.6 \times 0.537) + (-0.1 \times -0.099) \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix}\right)$$$$h_2 = \tanh\left(\begin{pmatrix} 0.121 \\ 0.332 \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix}\right)$$$$h_2 = \tanh\left(\begin{pmatrix} 0.321 \\ 1.232 \end{pmatrix}\right)$$
$$h_2 \approx \begin{pmatrix} 0.311 \\ 0.842 \end{pmatrix}$$

The calculation of $h_2$ directly incorporates information from the word "hi" through the $h_1$ vector.

---

### 4. Prediction

Finally, we use the final hidden state, $h_2$, to predict the next word. We'll use the $W_{hy}$ weights to get the output scores before applying a softmax function.

$$y_{pred} = W_{hy} h_2 + b_y$$$$y_{pred} = \begin{pmatrix} 0.1 & 0.7 \\ -0.3 & 0.2 \\ 0.5 & 0.4 \\ -0.6 & -0.1 \end{pmatrix} \begin{pmatrix} 0.311 \\ 0.842 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$$$y_{pred} = \begin{pmatrix} (0.1 \times 0.311) + (0.7 \times 0.842) \\ (-0.3 \times 0.311) + (0.2 \times 0.842) \\ (0.5 \times 0.311) + (0.4 \times 0.842) \\ (-0.6 \times 0.311) + (-0.1 \times 0.842) \end{pmatrix}$$
$$y_{pred} = \begin{pmatrix} 0.620 \\ 0.075 \\ 0.492 \\ -0.271 \end{pmatrix}$$

These values are the raw scores for each possible next word. The network would then pass this vector through a softmax function to get a probability distribution, which in this case would likely point to "hi" or "class" as the most probable next word based on these learned weights.

# Numerical RNN worked example (with your one-hot vocabulary)

Excellent — you gave the vocabulary and one-hot encodings. I’ll treat this as a next-word prediction RNN and show **(A)** a forward pass (hidden states → logits → softmax probabilities → loss) and **(B)** a full BPTT gradient pass across the whole sequence, then show a single SGD parameter update. I’ll explain each step and show the actual numbers so you can follow end-to-end.

---

# 1. Model & data (choices I made to keep numbers small & interpretable)

Vocabulary and one-hots (you already gave):

* `hi` = \[1,0,0,0]
* `there` = \[0,1,0,0]
* `class` = \[0,0,1,0]
* `!` = \[0,0,0,1]

I use a **very small RNN** with scalar hidden state (hidden size = 1) so the math is simple but still shows recurrence and BPTT.

Model equations:

$$
h_t = \tanh(W_x[x_t] + W_h \, h_{t-1} + b_h)
$$

$$
z_t = W_y \cdot h_t + b_y \qquad(\text{logits, size }V)
$$

$$
p_t = \operatorname{softmax}(z_t)
$$

$$
\mathcal{L} = -\sum_{t} \log p_t[y_t] \quad(\text{cross-entropy; we will average later})
$$

Chosen parameters (easy numbers):

* initial hidden: $h_0 = 0$
* $W_x$ (one scalar per input word):

  * $W_x[\text{hi}]=0.5$
  * $W_x[\text{there}]=-0.3$
  * $W_x[\text{class}]=0.8$
  * $W_x[!]=0.0$
* $W_h = 0.2$ (scalar)
* $b_h = 0$
* $W_y$ (maps scalar $h_t$ → 4 logits): $[1.0,\; 0.5,\; -0.5,\; 0.0]$ (one entry per vocab word)
* $b_y = [0,0,0,0]$

Sequence (inputs):
$\;x = [\text{hi},\; \text{there},\; \text{class},\; !]$

We train as a standard language model (predict next word). So targets (shifted by one) I use here are:

* target at t=1 is `there`
* target at t=2 is `class`
* target at t=3 is `!`
* target at t=4 is `!` (we just set a last target for the example)

---

# 2. Forward pass — compute hidden states, logits, softmax probabilities, per-step loss

I will show the arithmetic step-by-step.

### Time step t=1 (input = `hi`)

* pre-activation: $a_1 = W_x[\text{hi}] + W_h h_0 = 0.5 + 0.2\cdot 0 = 0.5$
* hidden: $h_1 = \tanh(0.5) = 0.46211715726000974$
* logits: $z_1 = h_1 \cdot W_y = [0.46211716,\; 0.23105858,\; -0.23105858,\; 0]$
* softmax probabilities $p_1$ (normalize $z_1$):

  $$
  p_1 \approx [0.34204087,\; 0.27147543,\; 0.17101555,\; 0.21546814]
  $$
* target = `there` (index 1) → cross-entropy loss at t=1:

  $$
  \ell_1 = -\log p_1[\text{there}] \approx 1.3038836248184211
  $$

---

### Time step t=2 (input = `there`)

* pre-activation: $a_2 = W_x[\text{there}] + W_h h_1 = -0.3 + 0.2\cdot 0.462117157 = -0.2075846$
* hidden: $h_2 = \tanh(-0.2075846) = -0.20464571679388185$
* logits: $z_2 = h_2 \cdot W_y = [-0.20464572,\; -0.10232286,\; +0.10232286,\; 0]$
* softmax $p_2 \approx [0.21303204,\; 0.23598434,\; 0.28957408,\; 0.26140954]$
* target = `class` (index 2) → $\ell_2 = -\log p_2[\text{class}] \approx 1.2393441229841993$

---

### Time step t=3 (input = `class`)

* pre-activation: $a_3 = W_x[\text{class}] + W_h h_2 = 0.8 + 0.2\cdot(-0.2046457168) = 0.75907085664$
* hidden: $h_3 = \tanh(0.75907085664) = 0.6405293508298561$
* logits: $z_3 = h_3\cdot W_y = [0.64052935,\; 0.32026468,\; -0.32026468,\; 0]$
* softmax $p_3 \approx [0.37942612,\;0.27544699,\;0.14516425,\;0.19996264]$
* target = `!` (index 3) → $\ell_3 = -\log p_3[!]\approx 1.609624739367793$

---

### Time step t=4 (input = `!`)

* pre-activation: $a_4 = W_x[!]+ W_h h_3 = 0.0 + 0.2\cdot 0.64052935083 = 0.12810587$
* hidden: $h_4 = \tanh(0.12810587) = 0.12740965340755267$
* logits: $z_4 = h_4\cdot W_y = [0.12740965,\;0.06370483,\;-0.06370483,\;0]$
* softmax $p_4 \approx [0.27437154,\;0.25743785,\;0.22664132,\;0.24154928]$
* target = `!` (index 3) → $\ell_4 = -\log p_4[!]\approx 1.4206817583023288$

---

### Loss summary

Per-step losses:

* $\ell_1 \approx 1.3038836248$
* $\ell_2 \approx 1.23934412298$
* $\ell_3 \approx 1.60962473937$
* $\ell_4 \approx 1.42068175830$

Total (sum) loss = 5.57353424547. Average loss per time step:

$$
\bar{\mathcal{L}} = \frac{1}{4}\sum_{t=1}^4 \ell_t \approx 1.3933835613681855
$$

(These numbers are exact to the digits shown.)

---

# 3. Backpropagation Through Time (BPTT) — compute gradients of the loss w\.r.t. parameters

We’ll compute gradients for: $W_y$ (vector length 4), $W_h$ (scalar), and the four entries of $W_x$ (one per vocab word). Because hidden is scalar and the input is one-hot, gradients with respect to $W_x$ go into the column corresponding to the input word at that time.

### Key identities used

* For softmax + cross-entropy: if $z$ are logits and $p=\text{softmax}(z)$ and the one-hot target is $y$, then

  $$
  \frac{\partial \ell}{\partial z} = p - y.
  $$
* With $z_t = W_y \, h_t$ (elementwise: $z_{t,j}=W_{y,j} h_t$), we have

  $$
  \frac{\partial \ell}{\partial W_{y,j}} = \sum_t (p_{t,j} - y_{t,j}) \cdot h_t.
  $$
* Backprop to hidden: $\frac{\partial \ell}{\partial h_t} = \sum_j (p_{t,j} - y_{t,j})W_{y,j} + \text{(from future timesteps through }W_h\text{)}$.
* For tanh pre-activation $a_t$: $\frac{\partial h_t}{\partial a_t} = 1 - h_t^2$.
* For $W_x$ (the scalar associated with the input word at time $t$): $\partial \ell/\partial W_x[x_t] $ accumulates the scalar $\partial \ell/\partial a_t$.

I computed these exactly across the whole 4-step sequence (full BPTT). Here are the resulting gradients.

### Gradients (numbers)

* Gradient w\.r.t. $W_y$ (4 entries for vocab order $[hi,\;there,\;class,\;!]$):

$$
\nabla_{W_y} \approx [\;0.39245801,\; -0.17572493,\; 0.34627310,\; -0.56300617\;]
$$

Interpretation: for example, the positive 0.392... for the first component means increasing $W_y[\text{hi}]$ (the logit weight for the word `hi`) would increase the total loss by that amount times a small parameter change (so gradient descent would reduce that component).

* Gradient w\.r.t. $W_h$ (scalar):

$$
\nabla_{W_h} \approx 0.45209451101463705
$$

* Gradients w\.r.t. $W_x$ (one scalar per input word): these are accumulated over all time steps where that word appeared:

$$
\nabla_{W_x}[\text{hi}] \approx 0.027609473508349962
$$

$$
\nabla_{W_x}[\text{there}] \approx 0.7141787364790343
$$

$$
\nabla_{W_x}[\text{class}] \approx 0.2957932315495087
$$

$$
\nabla_{W_x}[!] \approx 0.2850659086801632
$$

Note: each of these was computed by summing the backpropagated scalar $\partial\ell/\partial a_t$ at the time(s) that word was the input.

(full derivation steps were applied exactly as per the identities above).

---

# 4. Single SGD update example (learning rate $\eta=0.1$)

Perform a simple gradient descent step: parameter := parameter − $\eta$ \* gradient.

* Updated $W_y$:

  $$
  W_y \leftarrow W_y - 0.1 \cdot \nabla_{W_y} \approx
  [0.96075420,\; 0.51757249,\; -0.53462731,\; 0.05630062]
  $$

* Updated $W_h$:

  $$
  W_h \leftarrow 0.2 - 0.1\cdot 0.452094511 \approx 0.1547905489
  $$

* Updated $W_x$ entries:

  * $W_x[\text{hi}] \leftarrow 0.5 - 0.1\cdot 0.02760947 \approx 0.49723905$
  * $W_x[\text{there}] \leftarrow -0.3 - 0.1\cdot 0.71417874 \approx -0.37141787$
  * $W_x[\text{class}] \leftarrow 0.8 - 0.1\cdot 0.29579323 \approx 0.77042068$
  * $W_x[!] \leftarrow 0.0 - 0.1\cdot 0.28506591 \approx -0.02850659$

So you can see how each parameter shifts after one pass.

---

# 5. Intuition & key teaching points (professor mode)

1. **Why hidden size = 1 is helpful here.**
   It keeps every quantity scalar so you see clearly how each gradient component accumulates across time. With vector hidden states you do the same math matrix-wise.

2. **Softmax + CE gives very simple local derivative.**
   $\partial \ell/\partial z = p - y$. That little identity is what makes language modeling tight and efficient.

3. **BPTT accumulates gradients over time.**
   Gradients to recurrent weight $W_h$ and to earlier $W_x$ come from contributions at many time steps and must flow backwards through the chain of $\tanh$ derivatives and multiply by $W_h$ repeatedly (that's why vanishing/exploding gradients can occur for long sequences).

4. **One update is not learning yet.**
   The average loss \~1.39 indicates the model is not yet predicting next words strongly; repeated passes, proper initialization, learning rate schedules, and usually larger hidden sizes will be needed.

---

# 6. If you want to go deeper (next options I can do now)

* Show the **full algebraic derivation** of each gradient term (I used the standard identities above; I can expand every line if you want).
* Run **multiple SGD steps** and show loss evolution numerically (I can simulate several epochs with these equations).
* Repeat the same example with **hidden size = 2** so you can see matrix forms and how the one-hot picks columns of $W_x$.
* Replace $\tanh$ with **ReLU** or use a **GRU/LSTM** step and compare the gradients numerically.
* Show **vectorized formulas** to implement this easily in NumPy / PyTorch (with code).

Tell me which of those you want and I’ll produce it next — or if you want, I can **walk through the exact derivation of the BPTT recursion** line-by-line on the numbers above.

