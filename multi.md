# Zero-Redundancy LLM Arbitration System Using Probabilistic Early Rejection and Reward-Guided Selection

## Abstract

We introduce a novel arbitration system that efficiently selects the best-performing Large Language Models (LLMs) for classification, minimizing redundancy by:

- Using a fast early-rejection step on “easy” examples
- Employing a reward-driven multi-armed bandit algorithm to assign future examples to strong-performing models
- Optionally routing examples to specialized models using learned input features

This design significantly reduces computation from \( \mathcal{O}(N \times D) \) to approximately \( \mathcal{O}(N) + \mathcal{O}(D) \), making it viable for deployment in real-time, compute-constrained environments.

---

## 1. Background

Running all \( N \) LLMs on \( D \) examples incurs a cost of:

\[
\mathcal{O}(N \times D)
\]

Ensemble methods provide robustness but are computationally expensive and often redundant. Our approach reduces redundancy by early-dropping weak models and assigning examples intelligently using adaptive reward mechanisms.

---

## 2. Architecture Overview

The system proceeds in three distinct stages:

### Stage 1: Probabilistic Early Rejection (PER)

Let:

- \( \mathcal{E} = \{x_1, x_2, \ldots, x_k\} \) be a set of easy examples  
- \( M_i \) be the \( i \)-th model out of \( N \)  
- \( y_j \) be the true label for input \( x_j \)

We compute the accuracy of model \( M_i \) on the easy set:

\[
A_i = \frac{1}{k} \sum_{j=1}^{k} \mathbf{1}[M_i(x_j) = y_j]
\]

If:

\[
A_i < \tau
\]

where \( \tau \in (0, 1) \), model \( M_i \) is dropped from future consideration.

This filters out models that fail to classify trivial or low-entropy examples correctly.

---

### Stage 2: Reward-Guided Bandit Allocation (RGBA)

Remaining models (say \( M' \ll N \)) are modeled as arms in a contextual multi-armed bandit. At each time step \( t \), an input \( x_t \) is assigned to a model \( M_i \) based on a policy \( \pi(t) \).

Each model receives a reward defined as:

\[
r_i(t) = \alpha \cdot \mathbf{1}[M_i(x_t) = y_t] + \beta \cdot \mathrm{conf}(M_i(x_t))
\]

Where:

- \( \alpha, \beta \) are hyperparameters  
- \( \mathrm{conf}(M_i(x_t)) \in [0, 1] \) is model confidence  

The cumulative reward is:

\[
R_T = \sum_{t=1}^{T} r_{\pi(t)}(t)
\]

We update the value estimate of each model using:

\[
Q_i(t+1) = Q_i(t) + \eta \cdot \left(r_i(t) - Q_i(t)\right)
\]

Where:

- \( Q_i \) is the estimated value of model \( M_i \)  
- \( \eta \) is the learning rate  

---

### Stage 3 (Optional): Task-Specific Router (TSR)

Train a lightweight classifier \( R(x) \) using features \( \phi(x) \) (e.g., sentence embeddings) to route inputs to the most suitable model:

\[
R(x) = \arg\max_i \; P(M_i \mid \phi(x))
\]

Router models may include logistic regression, SVMs, or shallow neural networks.

---

## 3. Component Overview

| Component         | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| EasySetSelector   | Selects low-entropy examples \( H(p(y|x)) < H_0 \)                                            |
| FailureDetector   | Removes models \( M_i \) with \( A_i < \tau \)                                               |
| BanditAllocator   | Selects models via reward-maximizing bandit policy                                           |
| RewardEngine      | Computes: \( r_i(t) = \alpha \cdot \text{accuracy} + \beta \cdot \text{confidence} \)       |
| ModelRouter (TSR) | Learns to route inputs via \( R(x) = \arg\max_i P(M_i \mid \phi(x)) \)                        |

---

## 4. Complexity Comparison

| Method           | Complexity                   |
|------------------|------------------------------|
| Traditional      | \( \mathcal{O}(N \times D) \) |
| This Method      | \( \mathcal{O}(k \times N) + \mathcal{O}(M' \times D') \), where \( k \ll D \), \( M' \ll N \) |

Net effective cost:  

\[
\mathcal{O}(N) + \mathcal{O}(D)
\]

---

## 5. Novelty & Inventive Step

- **Probabilistic Early Rejection:** dynamically filters weak models early based on easy samples.  
- **Bandit-Driven Selection:** incorporates model confidence and reward signals for efficient selection.  
- **Task-Specific Routing:** optional downstream router personalizes model assignment.

No prior art combines:

- Early rejection on easy samples,  
- Bandit-based selection using model confidence,  
- Task-aware dynamic routing.

---

