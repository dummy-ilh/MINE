ML Problem Formulation — 12 Chapters

From Business Objective to ML Task — translating a vague ask ("increase engagement") into a well-posed prediction problem; identifying what's actually learnable
Task Taxonomy — classification vs. regression vs. ranking vs. recommendation vs. clustering vs. anomaly detection vs. generative; picking the right one given the objective
Defining the Prediction Target — what to predict, at what granularity (user/session/item), and over what time window
Label Design & Ground Truth — implicit vs. explicit labels, delayed feedback, label leakage, noisy labels
Metric Selection: Offline — accuracy/precision/recall/AUC/NDCG/RMSE — matching metric to task and to business cost asymmetries
Metric Selection: Online & Proxy Metrics — surrogate metrics, guardrail metrics, the gap between offline gains and online impact
Objective Function Design — translating business cost into a loss function; asymmetric losses, weighted objectives
Multi-Objective & Constrained Formulations — combining objectives (e.g., relevance + diversity + revenue), Lagrangian/constraint framing, Pareto tradeoffs
Data Availability & Cold-Start Framing — reformulating the problem when data is sparse, delayed, or biased by existing systems
Baseline Design — non-ML baselines, heuristic baselines, and what "beating baseline" should mean
Feedback Loops & Selection Bias — how deployed models reshape their own training data; counterfactual framing
Case Studies — 3–4 worked formulations (e.g., "design the ranking objective for a news feed," "formulate churn prediction," "formulate fraud detection") with L5 vs. lower-level answer contrasts
