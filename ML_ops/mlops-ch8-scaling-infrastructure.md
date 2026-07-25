# Chapter 8 — Scaling & Infrastructure

*(Module 8 of the syllabus)*

---

## 1. The problem this chapter solves

Everything so far assumed a model serving *some* amount of traffic. This chapter asks: what happens as traffic grows — from a handful of requests per second to potentially millions — and how does your serving infrastructure need to adapt without breaking latency budgets (Chapter 6) or blowing up cost?

---

## 2. Horizontal vs. vertical scaling

Two fundamentally different ways to handle more load:

**Vertical scaling** — make each machine bigger/more powerful (more CPU, more RAM, a bigger/faster GPU). Simple to reason about, but has a hard ceiling: eventually there's no bigger machine to move to, and the biggest machines are disproportionately expensive.

**Horizontal scaling** — instead of making one machine bigger, run *more* machines (replicas), each running the same model, and distribute incoming traffic across them. This is the dominant approach for production ML serving, because it scales much further than vertical scaling and offers a critical side benefit: **redundancy**. If one replica fails, the others keep serving — vertical scaling gives you no such safety net, since there's only ever one machine.

**Why horizontal scaling is the default answer in interviews:** beyond just "it scales further," it directly buys you fault tolerance, which vertical scaling structurally cannot provide (a single bigger machine is still a single point of failure). If asked to design a scalable serving system, defaulting to horizontal scaling (many replicas) and explaining *why* over vertical is a strong, complete answer.

---

## 3. Load balancing

Once you have multiple replicas of your model running, something needs to decide which replica handles each incoming request. That's a **load balancer** — sitting in front of your replicas, distributing incoming requests across them.

Why this matters beyond "just split it evenly": a load balancer also plays a role in **fault tolerance** — if a replica becomes unhealthy (crashes, or starts responding too slowly), a well-configured load balancer detects this (via health checks) and stops routing traffic to it, automatically routing only to healthy replicas instead. This connects directly back to the redundancy benefit of horizontal scaling — the redundancy is only actually useful in practice if something is actively detecting and routing around failures.

---

## 4. Autoscaling

Running a fixed number of replicas is wasteful in both directions: too few replicas during peak traffic causes overload and latency blowups; too many replicas during quiet periods wastes money on idle compute. **Autoscaling** is the practice of automatically adjusting the number of running replicas based on real-time demand.

**What autoscaling typically reacts to:**
- **Traffic volume / request rate** — more incoming requests per second → spin up more replicas.
- **Queue depth** — if requests are piling up waiting to be processed faster than they're being drained, that's a direct, very responsive signal that more capacity is needed right now.

**The tradeoff to be aware of:** spinning up a new replica isn't instantaous — the container needs to start, the model needs to load into memory, and only then can it actually start serving requests. This startup delay means autoscaling needs to react *ahead* of demand, not purely reactively, or you risk a period of overload while new capacity is still coming online. This is a legitimate, specific thing to mention if asked about autoscaling challenges — it's a good signal that you understand this isn't a trivial "just add more machines" switch.

---

## 5. Multi-region serving

For a global product, running your ML service in only one geographic region means users far from that region experience meaningfully higher latency (physical distance directly adds network travel time), even if the model itself is fast. **Multi-region serving** means running full copies of your serving infrastructure in multiple geographic locations, and routing each user's traffic to the nearest one.

**The tradeoffs this introduces, worth naming explicitly:**
- **Consistency vs. latency** — if different regions need to stay in sync (e.g., a shared feature store, or a model update that needs to roll out everywhere), keeping all regions perfectly consistent in real time can itself add latency or complexity. Often, production systems accept some acceptable staleness between regions (eventual consistency) in exchange for keeping each region's actual serving latency low — this is a classic distributed-systems tradeoff that also shows up in ML infrastructure specifically.
- **Operational complexity** — deploying a model update now means coordinating rollout across multiple regions (do they update simultaneously? staggered, so a bad rollout only affects one region at a time — which itself is a form of the canary idea from Chapter 5, applied at the region level rather than the traffic-percentage level).

---

## 6. Cost considerations

A recurring theme worth stating directly in interviews: **infrastructure decisions in ML serving are never purely technical — they're always also cost decisions**, and a good answer reflects that explicitly rather than only optimizing for raw performance.

- Bigger/more replicas, bigger GPUs, multi-region redundancy — all of these improve some combination of latency, throughput, and reliability, but all of them cost more.
- "Just use the biggest GPU" (flagged as a pitfall in Chapter 6) is the same failure mode here at the infrastructure level: defaulting to maximal scale/redundancy without weighing it against actual requirements and cost is a weak answer.
- The right framing, again: state the actual requirement (expected peak traffic, latency SLA, acceptable downtime/redundancy level) *first*, then size infrastructure to meet that requirement efficiently — not by defaulting to "as much as possible."

---

## 7. Common pitfall interviewers listen for

Treating scaling as purely "add more compute" without addressing **how requests get distributed and rerouted around failure** (load balancing) or **how capacity adjusts to changing demand** (autoscaling) is an incomplete answer. A complete system design answer names all three together — horizontal scaling for capacity, load balancing for distribution and fault tolerance, autoscaling for cost-efficient adjustment to demand — rather than treating "add more replicas" as the whole story.

---

## Comprehension check

1. In your own words, why does horizontal scaling provide fault tolerance that vertical scaling structurally cannot?
2. Explain why autoscaling needs to anticipate demand somewhat, rather than purely react to current load — what specifically causes the lag?
3. You're designing serving infrastructure for a global product with a strict 100ms latency SLA. Briefly sketch which concepts from this chapter you'd combine, and why multi-region matters specifically for this constraint.

Say "c9" when ready for **Chapter 9: Model Governance & Responsible Deployment**.
