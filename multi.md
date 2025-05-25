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


Uniqueness and Novelty
1. Probabilistic Early Rejection Based on “Easy” Examples
Unlike existing ensemble or multi-model selection techniques that treat all data points equally, this system introduces a data-driven early rejection step using a carefully curated subset of "easy" examples.

This ensures models that fail on trivial or low-entropy cases are immediately discarded, saving computational resources upfront.

The dynamic threshold 
𝜏
τ is adaptive and learned, enabling flexible model pruning customized per deployment environment.

This pre-filtering concept based on example difficulty combined with model performance is a novel use in LLM arbitration.

2. Reward-Guided Multi-Armed Bandit Model Selection
While multi-armed bandits have been widely used in model selection and online learning, our system integrates:

Model confidence scores into the reward signal, not just accuracy, reflecting the certainty of predictions, which improves stability and decision granularity.

An adaptive learning rate mechanism that balances exploration and exploitation in selecting models per incoming example stream.

This fine-grained reward design with confidence-augmented feedback for model arbitration is unique in the LLM domain.

3. Dynamic Model Dropout Driven by Easy Example Performance
The novel integration of early rejection with bandit-based allocation enables dynamic dropout of underperforming models after the initial easy example evaluation, which contrasts with static ensembling or manual selection in prior art.

The approach reduces the model pool without multiple passes over the entire dataset.

This efficiency gain is crucial for large-scale LLM systems, where each model evaluation is costly.

4. Optional Task-Specific Router for Input-Conditional Model Assignment
The introduction of a lightweight input router trained on learned input features to map examples to the most competent model introduces:

Fine-grained, input-aware specialization beyond simple global model ranking.

A modular architecture allowing plug-and-play of any classification routing mechanism, including logistic regression, SVM, or shallow neural nets.

This hybrid combination of bandit-based arbitration with routing adds flexibility and further efficiency.

5. Computational Efficiency and Scalability
The system achieves a near-linear computational cost in dataset size and model count compared to quadratic cost in naïve multi-model evaluation.

This significant reduction in inference cost while maintaining accuracy creates practical viability for real-time, large-scale deployments.

6. Patent-Worthy Combinatorial Approach
While individual components like multi-armed bandits and confidence scoring exist in the literature, the specific combination and sequencing of early rejection, confidence-augmented reward, and input-aware routing applied to LLM arbitration is novel and non-obvious.

The architecture also incorporates dynamic model dropout based on easy example performance in a single/few-pass approach, which is not previously known.


