# Zero-Redundancy LLM Arbitration System Using Probabilistic Early Rejection and Reward-Guided Selection
Summary Statement Example for Your Patent or Paper
“Our Zero-Redundancy LLM Arbitration System is a novel, highly efficient approach to selecting the best Large Language Model for classification tasks. By combining early model dropout based on easy examples with a reward-driven multi-armed bandit for dynamic input assignment, our system optimally balances accuracy and computational cost. The inclusion of an optional task-specific routing mechanism further enhances assignment precision. This architecture uniquely minimizes redundant evaluations, adapts in real time, and scales gracefully, establishing a new state-of-the-art for efficient LLM model selection.”


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



Rethought Workflow for Multi-LLM Model Selection Using Multi-Armed Bandit with Early Dropout
Input
𝑁
N candidate LLMs: 
𝑀
1
,
𝑀
2
,
.
.
.
,
𝑀
𝑁
M 
1
​
 ,M 
2
​
 ,...,M 
N
​
 

Dataset with 
𝑀
M rows/examples: 
𝐷
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑀
D={(x 
i
​
 ,y 
i
​
 )} 
i=1
M
​
 

Step 1: Initial Dataset Split and Early Performance Monitoring
Split dataset 
𝐷
D into two parts:

Phase 1: First 20% of data, 
𝐷
𝑖
𝑛
𝑖
𝑡
D 
init
​
  (e.g., first 
0.2
×
𝑀
0.2×M rows)

Phase 2: Remaining 80%, 
𝐷
𝑟
𝑒
𝑠
𝑡
D 
rest
​
 

Step 2: Parallel Model Evaluation on 
𝐷
𝑖
𝑛
𝑖
𝑡
D 
init
​
 
Run all 
𝑁
N models on 
𝐷
𝑖
𝑛
𝑖
𝑡
D 
init
​
  in parallel (or batched to minimize runtime).

Track model performance accuracy 
𝐴
𝑖
A 
i
​
  over 
𝐷
𝑖
𝑛
𝑖
𝑡
D 
init
​
  incrementally, i.e., update accuracy after each example or small batch.

Step 3: Early Dropout with Patience
Define a patience parameter 
𝑝
=
3
p=3 (number of consecutive failures allowed).

For each model 
𝑀
𝑖
M 
i
​
 , monitor performance in sliding windows or per example:

If 
𝑀
𝑖
M 
i
​
  fails to predict correctly on 
𝑝
p consecutive examples while at least one other model performs correctly on those examples, mark 
𝑀
𝑖
M 
i
​
  as a candidate for dropout.

Drop 
𝑀
𝑖
M 
i
​
  only if this failure pattern persists (no recovery) within the first 20% data.

This eliminates clearly underperforming models early, relying on relative performance, not absolute thresholds.

Step 4: Narrowed Down Model Set
After Phase 1, retain models that survived dropout.

If more than 5 models remain, keep top 5 based on accuracy or aggregate performance metrics on 
𝐷
𝑖
𝑛
𝑖
𝑡
D 
init
​
 .

Step 5: Multi-Armed Bandit on Remaining Data 
𝐷
𝑟
𝑒
𝑠
𝑡
D 
rest
​
 
Initialize bandit values 
𝑄
𝑖
Q 
i
​
  for each retained model 
𝑀
𝑖
M 
i
​
 .

For each incoming example 
𝑥
𝑡
x 
t
​
  in 
𝐷
𝑟
𝑒
𝑠
𝑡
D 
rest
​
 :

Select a model 
𝑀
𝑖
M 
i
​
  using the multi-armed bandit policy (e.g., UCB, Thompson Sampling) balancing exploration-exploitation.

Model 
𝑀
𝑖
M 
i
​
  predicts 
𝑦
^
𝑡
y
^
​
  
t
​
 .

Reward 
𝑟
𝑖
=
1
r 
i
​
 =1 if 
𝑦
^
𝑡
=
𝑦
𝑡
y
^
​
  
t
​
 =y 
t
​
 , else 0.

Update 
𝑄
𝑖
Q 
i
​
  based on reward to improve model selection policy.

Step 6: Optional Router Training (Post Bandit Phase)
Use data from bandit assignments and outcomes to train a router 
𝑅
R that predicts best model per input features.

This router can accelerate or replace bandit decisions on new/unseen data.

Step 7: Output
Final best subset of models selected by performance and bandit optimization.

Model value estimates 
𝑄
𝑖
Q 
i
​
  reflecting per-model utility.

Optional trained router 
𝑅
R for input-driven model assignment.

Summary:
Early phase dropout uses relative performance with patience on 20% data.

Patience ensures transient failures don’t unfairly drop models, but persistent lagging models are pruned.

Final selection uses multi-armed bandit on reduced model pool and remaining data, dynamically optimizing model assignment.

--------
Algorithm: Multi-LLM Selection with Early Dropout and Bandit Optimization
Input:

Models 
𝑀
=
{
𝑀
1
,
𝑀
2
,
.
.
.
,
𝑀
𝑁
}
M={M 
1
​
 ,M 
2
​
 ,...,M 
N
​
 }

Dataset 
𝐷
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑀
D={(x 
i
​
 ,y 
i
​
 )} 
i=1
M
​
 

Patience parameter 
𝑝
p (e.g., 3)

Max retained models 
𝑘
k (e.g., 5)

Output:

Selected subset of models 
𝑀
𝑓
𝑖
𝑛
𝑎
𝑙
M 
final
​
 

Model quality estimates 
𝑄
Q

Optional router 
𝑅
R

Procedure:

Split dataset:

𝐷
𝑖
𝑛
𝑖
𝑡
←
D 
init
​
 ← first 20% of 
𝐷
D

𝐷
𝑟
𝑒
𝑠
𝑡
←
D 
rest
​
 ← remaining 80% of 
𝐷
D

Initialize:

For each model 
𝑀
𝑖
M 
i
​
 , set:

𝑐
𝑜
𝑛
𝑠
𝑒
𝑐
𝑢
𝑡
𝑖
𝑣
𝑒
_
𝑓
𝑎
𝑖
𝑙
𝑢
𝑟
𝑒
𝑠
𝑖
←
0
consecutive_failures 
i
​
 ←0

𝑑
𝑟
𝑜
𝑝
𝑝
𝑒
𝑑
𝑖
←
𝐹
𝑎
𝑙
𝑠
𝑒
dropped 
i
​
 ←False

𝑐
𝑜
𝑟
𝑟
𝑒
𝑐
𝑡
𝑖
←
0
correct 
i
​
 ←0

𝑡
𝑜
𝑡
𝑎
𝑙
𝑖
←
0
total 
i
​
 ←0

Early evaluation & dropout:
For each example 
(
𝑥
𝑡
,
𝑦
𝑡
)
∈
𝐷
𝑖
𝑛
𝑖
𝑡
(x 
t
​
 ,y 
t
​
 )∈D 
init
​
 :
  a. Collect set 
𝐶
=
{
}
C={} of models predicting correctly on 
𝑥
𝑡
x 
t
​
 .
  b. For each 
𝑀
𝑖
M 
i
​
  not dropped:
    i. Predict 
𝑦
^
𝑡
=
𝑀
𝑖
(
𝑥
𝑡
)
y
^
​
  
t
​
 =M 
i
​
 (x 
t
​
 )
    ii. Update 
𝑡
𝑜
𝑡
𝑎
𝑙
𝑖
←
𝑡
𝑜
𝑡
𝑎
𝑙
𝑖
+
1
total 
i
​
 ←total 
i
​
 +1
    iii. If 
𝑦
^
𝑡
=
𝑦
𝑡
y
^
​
  
t
​
 =y 
t
​
 , then
      - 
𝑐
𝑜
𝑟
𝑟
𝑒
𝑐
𝑡
𝑖
←
𝑐
𝑜
𝑟
𝑟
𝑒
𝑐
𝑡
𝑖
+
1
correct 
i
​
 ←correct 
i
​
 +1
      - 
𝑐
𝑜
𝑛
𝑠
𝑒
𝑐
𝑢
𝑡
𝑖
𝑣
𝑒
_
𝑓
𝑎
𝑖
𝑙
𝑢
𝑟
𝑒
𝑠
𝑖
←
0
consecutive_failures 
i
​
 ←0
      - Add 
𝑀
𝑖
M 
i
​
  to 
𝐶
C
    Else
      - 
𝑐
𝑜
𝑛
𝑠
𝑒
𝑐
𝑢
𝑡
𝑖
𝑣
𝑒
_
𝑓
𝑎
𝑖
𝑙
𝑢
𝑟
𝑒
𝑠
𝑖
←
𝑐
𝑜
𝑛
𝑠
𝑒
𝑐
𝑢
𝑡
𝑖
𝑣
𝑒
_
𝑓
𝑎
𝑖
𝑙
𝑢
𝑟
𝑒
𝑠
𝑖
+
1
consecutive_failures 
i
​
 ←consecutive_failures 
i
​
 +1
  c. For each 
𝑀
𝑖
M 
i
​
  not dropped:
    i. If 
𝑐
𝑜
𝑛
𝑠
𝑒
𝑐
𝑢
𝑡
𝑖
𝑣
𝑒
_
𝑓
𝑎
𝑖
𝑙
𝑢
𝑟
𝑒
𝑠
𝑖
≥
𝑝
consecutive_failures 
i
​
 ≥p AND 
𝑀
𝑖
∉
𝐶
M 
i
​
 ∈
/
C AND 
∣
𝐶
∣
>
0
∣C∣>0 then
      Set 
𝑑
𝑟
𝑜
𝑝
𝑝
𝑒
𝑑
𝑖
←
𝑇
𝑟
𝑢
𝑒
dropped 
i
​
 ←True

Select survivors:

Let 
𝑆
=
{
𝑀
𝑖
∣
𝑑
𝑟
𝑜
𝑝
𝑝
𝑒
𝑑
𝑖
=
𝐹
𝑎
𝑙
𝑠
𝑒
}
S={M 
i
​
 ∣dropped 
i
​
 =False}

If 
∣
𝑆
∣
>
𝑘
∣S∣>k, keep top 
𝑘
k models with highest 
𝑐
𝑜
𝑟
𝑟
𝑒
𝑐
𝑡
𝑖
𝑡
𝑜
𝑡
𝑎
𝑙
𝑖
total 
i
​
 
correct 
i
​
 
​
 

Initialize bandit:

For each 
𝑀
𝑖
∈
𝑆
M 
i
​
 ∈S, set 
𝑄
𝑖
←
0
Q 
i
​
 ←0, 
𝑁
𝑖
←
0
N 
i
​
 ←0

Bandit-based selection on 
𝐷
𝑟
𝑒
𝑠
𝑡
D 
rest
​
 :
For each 
(
𝑥
𝑡
,
𝑦
𝑡
)
∈
𝐷
𝑟
𝑒
𝑠
𝑡
(x 
t
​
 ,y 
t
​
 )∈D 
rest
​
 :
  a. Select 
𝑀
𝑗
∈
𝑆
M 
j
​
 ∈S via bandit policy (e.g., UCB)
  b. Predict 
𝑦
^
𝑡
=
𝑀
𝑗
(
𝑥
𝑡
)
y
^
​
  
t
​
 =M 
j
​
 (x 
t
​
 )
  c. Compute reward 
𝑟
=
1
r=1 if 
𝑦
^
𝑡
=
𝑦
𝑡
y
^
​
  
t
​
 =y 
t
​
  else 0
  d. Update 
𝑁
𝑗
←
𝑁
𝑗
+
1
N 
j
​
 ←N 
j
​
 +1
  e. Update 
𝑄
𝑗
←
𝑄
𝑗
+
1
𝑁
𝑗
(
𝑟
−
𝑄
𝑗
)
Q 
j
​
 ←Q 
j
​
 + 
N 
j
​
 
1
​
 (r−Q 
j
​
 )

Optional router training:

Train router 
𝑅
R to map input 
𝑥
x to best 
𝑀
𝑖
M 
i
​
  using accumulated data

Return:

𝑀
𝑓
𝑖
𝑛
𝑎
𝑙
=
𝑆
M 
final
​
 =S

Quality estimates 
𝑄
Q

Optional 
𝑅
R
