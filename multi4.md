INPUT:
    Models = {M1, M2, ..., Mn}          # Candidate pool (e.g., 20 models)
    Data = {d1, d2, ..., dK}            # Evaluation samples
    Thresholds = {dropout_limit, reward_limit, cost_limit}

OUTPUT:
    BestModels                         # Surviving top performers
    Router (optional)

---------------------------------------------------------
# Stage 0: Consensus Dropout
---------------------------------------------------------
Initialize Survivors = Models
Initialize SampleCount = 0

FOR each sample in Data (small fraction first):
    Collect predictions from Survivors
    Determine majority consensus label
    FOR each model in Survivors:
        IF model prediction ≠ consensus:
            model.error_count += 1
        IF model.error_count > dropout_limit:
            Remove model from Survivors
    
    SampleCount += 1
    IF SampleCount ≥ 10% of Data OR
       Survivors ≤ 50% of Models:
        BREAK

---------------------------------------------------------
# Stage 1: Reward-Guided Bandit Allocation
---------------------------------------------------------
Initialize Reward[model] = 0
Initialize PullCount[model] = 0

WHILE evaluation budget not exhausted:
    SELECT model using bandit rule (UCB / Thompson)
    Assign next sample to chosen model
    Compute reward:
        reward = (Correctness * weight1)
               + (Confidence * weight2)
               + (BenchmarkScore * weight3)
               - (Cost * penalty)
    Update Reward[model]
    PullCount[model] += 1

    IF Reward[model] < reward_limit:
        Remove model from Survivors

---------------------------------------------------------
# Stage 2: Final Selection & Router
---------------------------------------------------------
BestModels = Top 1–3 Survivors ranked by reward

IF multiple BestModels remain:
    Train Router:
        Input features → select appropriate model
        Example rule:
            IF input_length < T → small model
            ELSE → large model

RETURN BestModels, Router
