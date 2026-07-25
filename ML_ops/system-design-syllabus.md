# System Design Fundamentals — API Design, Database Choices, AWS/Cloud Service Choices
### Target: Google / Apple MLE interviews · Assumes zero prior knowledge

These three topics aren't ML-specific, but they show up constantly in MLE system design rounds — the moment your MLOps answer (from the last syllabus) needs an actual API for the model, an actual place to store feature/prediction data, and actual cloud infrastructure to run on. This syllabus fills that gap.

---

## Part A — API Design (how a model gets called)

### A1. What an API actually is, and why design choices matter
- The basic contract: a client sends a request, a service returns a response — API design is about making that contract clear, stable, and hard to misuse
- Why sloppy API design causes real production pain (breaking changes, ambiguous errors, clients doing the wrong thing)

### A2. REST fundamentals
- Resources, HTTP verbs (GET/POST/PUT/DELETE) and what each *means*, not just does
- Status codes and why picking the right one matters (200 vs 202 vs 4xx vs 5xx)
- Statelessness — why REST APIs don't rely on server-side session memory between requests, and why that matters for scaling (ties to Chapter 8 horizontal scaling)

### A3. Request/response design for ML specifically
- Sync vs. async prediction endpoints — when a client can't wait for a slow model (batch scoring, large models) and needs a "submit job, poll for result" pattern instead of a direct response
- Versioning an API (so you can change a model/endpoint without breaking existing clients) — connects directly to model versioning from the MLOps syllabus
- Idempotency — why it matters that retrying a request (e.g., after a network blip) doesn't cause duplicate side effects

### A4. gRPC vs REST
- What gRPC is and why it's common in internal, performance-sensitive ML serving (binary protocol, lower overhead, strongly-typed contracts) vs. REST's simplicity and universal compatibility for public-facing APIs
- When to pick which — a concrete tradeoff interviewers like to probe

### A5. Rate limiting & authentication basics
- Why a public or shared API needs to protect itself from being overwhelmed by one client
- API keys / auth tokens at a conceptual level — enough to reason about who's allowed to call what

---

## Part B — Database Choices (where the data actually lives)

### B1. The fundamental split: SQL vs. NoSQL
- What "relational" actually means (structured tables, defined relationships, strong consistency guarantees)
- What NoSQL trades away and gains (flexible/unstructured schema, horizontal scalability, often weaker consistency guarantees)
- Why this isn't "NoSQL is newer/better" — it's a real tradeoff based on your access patterns

### B2. CAP theorem (plain language)
- Consistency, Availability, Partition tolerance — why you can't have all three perfectly at once in a distributed system, and what that means practically when picking a database
- How this shows up concretely in an ML system (e.g., a feature store choosing to serve slightly stale data rather than becoming unavailable during a network partition)

### B3. Common database types and what each is *for*
- Relational (e.g., structured transactional data — orders, users, anything needing strong consistency/joins)
- Key-value stores (fast lookups by ID — a natural fit for an online feature store's low-latency reads, tying back to Chapter 4 of the MLOps syllabus)
- Document stores (flexible, semi-structured records)
- Wide-column stores (huge volumes of data with fast writes — logging/time-series-like workloads)
- Time-series databases (naturally suited to monitoring data — metrics over time, tying back to Chapter 7)
- Vector databases — the one most specific to modern ML: storing embeddings and doing similarity search (relevant for recommendation/search/RAG-style systems)

### B4. Choosing a database for an ML system, concretely
- Feature store's online store → needs low-latency key-value lookups
- Feature store's offline store → needs to handle large-scale historical batch reads, less latency-sensitive
- Prediction/monitoring logs → high write volume, time-series-friendly access pattern
- Model registry metadata → relational fits well (structured, relationships between models/versions/lineage)
- This mapping — "which workload needs which database shape" — is exactly what interviewers want to see you reason through live, not memorize.

### B5. Indexing & query performance basics
- Why an index speeds up lookups (conceptually — a sorted/structured shortcut instead of scanning everything)
- The tradeoff: indexes speed up reads but slow down writes and cost storage — another "not free" tradeoff, matching the pattern from the MLOps syllabus

---

## Part C — AWS / Cloud Service Choices

### C1. Why cloud service choice is itself a design decision
- Interviewers aren't testing AWS trivia — they're testing whether you can map a system requirement to the right *category* of managed service, understanding what problem each category solves

### C2. Compute options
- Virtual machines (EC2) — full control, you manage everything, good for custom/long-running services
- Managed container orchestration (ECS/EKS) — for running your containerized model-serving setup (Chapter 3 of MLOps syllabus) at scale without managing raw servers
- Serverless functions (Lambda) — good for sporadic, short-lived, event-triggered work; a poor fit for a large model needing GPU and low cold-start latency
- Managed ML-specific serving (SageMaker endpoints) — purpose-built for hosting models with built-in autoscaling/monitoring hooks, trading some flexibility for less operational overhead

### C3. Storage options
- Object storage (S3) — the default for large files: datasets, model artifacts, logs; cheap, durable, not a database
- Block storage (EBS) — attached disk for a running instance, when you need low-latency local-like access
- Managed relational DB (RDS) vs. managed NoSQL (DynamoDB) — mapping back to Part B's SQL/NoSQL tradeoffs, now as concrete managed services

### C4. Data/streaming pipeline services
- Managed streaming (Kinesis) — for real-time data flowing into feature computation or monitoring pipelines
- Managed orchestration (Step Functions / managed Airflow) — for coordinating the multi-stage pipelines from Module 3 of the MLOps syllabus (data prep → train → eval → deploy)

### C5. Putting it together — mapping a full ML system to AWS services
- Worked-style thinking: raw data lands in S3 → streaming ingestion via Kinesis for real-time features → online feature store backed by DynamoDB → model artifacts versioned in S3 + tracked via a registry → served via SageMaker endpoints or ECS/EKS behind a load balancer → logs/metrics flow to a monitoring service
- The goal isn't memorizing this exact stack — it's being able to justify *why* each piece was chosen, the same "requirements → constraints → choice" reasoning from the MLOps capstone chapter

---

## How this fits with what you've already built
This isn't really a separate topic from MLOps — it's the "what's actually running underneath" layer. Every chapter in the MLOps syllabus (versioning, packaging, deployment, monitoring, retraining) needs an API in front of it, a database underneath it, and cloud infrastructure to run on. When you get a full system design prompt in an interview, these three areas are what you're filling in during Step 3 (data flow) and Step 4 (deployment strategy) of the five-step framework from Chapter 11.

---

*Say "go" and I'll start Chapter A1 the same way as the MLOps chapters — explained from scratch, plain language, with a comprehension check at the end of each.*
