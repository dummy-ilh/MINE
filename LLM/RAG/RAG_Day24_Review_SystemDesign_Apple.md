# RAG Interview Prep — Day 24
## Review Day: System Design & Apple-Specific (Days 22–23) — Closed Book

---

## 📋 How to run this review

No notes. This is the last review day before your two mock interviews (Days 25 and 27) — treat it as a dry run: answer out loud, at conversational pace, not just in your head. If you find yourself needing to pause and think for more than a few seconds on a Section C question, that's exactly the kind of hesitation a live interview will expose, so flag it honestly in the tracker.

---

## Section A — System Design Methodology (Day 22)

**A1.** What are the four phases of a system design interview answer, and roughly how should time be split across them?

<details>
<summary>Show answer</summary>
Requirements clarification (~5 min) → High-level architecture (~10 min) → Deep dives (~20-25 min, interviewer-directed) → Trade-offs/wrap-up (~5 min). The bulk of the time and the actual signal comes from the deep-dive phase, but skipping the first phase is the most common and costly mistake.
</details>

**A2.** Name the seven requirements-clarification dimensions to ask about before proposing an architecture.

<details>
<summary>Show answer</summary>
Scale (documents/QPS), latency (target p95/p99), freshness (how often content changes), consistency (staleness tolerance), multi-tenancy (isolation needs), budget/cost sensitivity, and accuracy/stakes (cost of a wrong answer).
</details>

**A3 (calculation).** A system runs 8,000 queries/day, reranking 25 candidates each at $0.0012/candidate. Compute daily reranking cost, and state why this line item is easy to underestimate.

<details>
<summary>Show answer</summary>

```
8,000 × 25 = 200,000 rerank calls/day
200,000 × $0.0012 = $240/day
```
It's easy to underestimate because attention naturally gravitates to generation cost (the most "visible" LLM cost), but reranking cost scales with candidate-set size and can dominate total cost depending on per-candidate pricing — exactly the counter-intuitive result Day 22's worked cost model demonstrated.
</details>

**A4.** Why should faithfulness monitoring in production typically use sampling rather than 100% coverage?

<details>
<summary>Show answer</summary>
Running a full faithfulness check (claim decomposition + NLI/LLM-judge verification) on every single response adds meaningful cost and latency at scale — the same trade-off as Day 15's runtime guardrail, now applied to aggregate monitoring. Sampling a representative percentage (or oversampling high-risk categories) gives an ongoing quality signal at a fraction of the cost.
</details>

**A5.** Why should retrieval-stage and generation-stage metrics be tracked as separate dashboard lines rather than one aggregate quality score?

<details>
<summary>Show answer</summary>
A single aggregate score can't tell you which stage regressed when it drops — separating Recall@k/nDCG (retrieval-stage) from faithfulness/answer-relevance (generation-stage) lets a sudden drop in one but not the other immediately localize where to investigate, mirroring Module 7's foundational argument for why retrieval and generation need separate metrics, now applied at the monitoring-dashboard level instead of offline evaluation.
</details>

---

## Section B — Apple-Specific Framing (Day 23)

**B1.** Describe Apple's three-tier routing architecture in one or two sentences.

<details>
<summary>Show answer</summary>
Simple requests run entirely on-device (AFM 3 Core / Core Advanced, ~3B parameters, 2-bit quantized, ~4K context, fully offline, no request limits); moderately complex requests escalate to Private Cloud Compute (larger ~32K context, supports reasoning, requires connectivity, daily limits, strong non-retention privacy guarantees); the most demanding requests can escalate further to AFM 3 Cloud Pro running on NVIDIA GPUs in Google Cloud.
</details>

**B2.** Why does RAG matter more, not less, for the on-device tier specifically?

<details>
<summary>Show answer</summary>
A ~3B-parameter, heavily quantized model has much less capacity to memorize broad world knowledge than a large cloud model, so retrieved context becomes proportionally more load-bearing for correctness — RAG compensates for the model's inherently limited parametric knowledge, making it more structurally necessary, not an optional accuracy boost.
</details>

**B3.** Why is vector index compression "mandatory" on-device even at a much smaller corpus size than would trigger it on a cloud system?

<details>
<summary>Show answer</summary>
On-device, the trigger is an absolute, tight memory ceiling shared with the OS, the generator model itself, and every other app — not a relative "getting large" judgment call the way it is at cloud scale (where PQ is typically adopted only once corpus size reaches hundreds of millions of vectors). Even a modest personal corpus can exceed what's feasible at full precision within that tight shared budget.
</details>

**B4.** How does Apple's on-device privacy model differ fundamentally from Day 5's retrieval-layer filtering approach for multi-tenant systems?

<details>
<summary>Show answer</summary>
Day 5's filtering is a mitigation applied to a shared, centralized index, with correctness dependent on the filter being applied correctly. Apple's on-device tier for personal data avoids the problem architecturally — the data often never leaves the device or enters any shared system at all for on-device requests, so there's no filtering-correctness problem to get right in the first place, a stronger guarantee than any access-control filter provides.
</details>

---

## Section C — Cross-Synthesis: Applying the Methodology to Apple Constraints (the actual mock-interview simulation)

**C1.** Apply Day 22's four-phase methodology to this prompt: "Design a feature that lets Siri answer questions using a user's on-device Notes app." What would you ask in Phase 1, specifically informed by Day 23's constraints (not generic Day 22 questions)?

<details>
<summary>Show answer</summary>
Beyond the generic Day 22 checklist, Apple-specific clarifying questions would include: is this feature required to work fully offline, or is escalation to Private Cloud Compute acceptable for complex queries? What's the expected size of a typical user's Notes corpus (dozens vs. thousands of notes) — this determines whether on-device index compression is even necessary at this specific scale? Are there any notes explicitly marked sensitive/locked that need special handling beyond the default on-device privacy guarantee? Is there a target latency that assumes purely on-device response (near-instant) or is some escalation latency acceptable for harder queries? These questions specifically probe the on-device/PCC boundary and offline requirement that a generic (non-Apple) system design prompt wouldn't need to ask.
</details>

**C2.** Using Day 22's cost-modeling approach and Day 23's architecture, explain why a "cost model" for this Notes-search feature looks fundamentally different from Day 22's worked cloud example.

<details>
<summary>Show answer</summary>
Day 22's worked cost model priced per-token/per-candidate API costs (embedding, reranking, generation, infrastructure) — costs that scale with query volume on a metered cloud service. For an on-device Siri/Notes feature, the on-device tier has effectively zero marginal per-query cost to Apple (no API calls, no token metering — it's local compute the user's own device already has), so the "cost" consideration shifts entirely to device resources (battery, memory, latency) rather than dollars, until a query escalates to Private Cloud Compute, at which point Apple does bear real infrastructure cost (though not directly metered to the developer in the same way a third-party LLM API is, per the developer-facing PCC free-tier model). This is a fundamentally different cost framing than a typical cloud RAG system design question, and worth stating explicitly rather than defaulting to Day 22's dollar-per-token framing on an Apple-context prompt.
</details>

**C3.** A system design prompt asks you to design monitoring (Day 22) for the Apple on-device Notes-search feature (Day 23). What's different about what you'd propose to monitor, given the offline/on-device constraint?

<details>
<summary>Show answer</summary>
Standard cloud RAG monitoring (Day 22) assumes centralized logging is straightforward — every request passes through servers you control. For an on-device feature, most requests never touch Apple's servers at all (by design, for privacy), so centralized real-time monitoring of individual request quality isn't available the same way. Monitoring would need to rely on privacy-preserving aggregate telemetry (e.g., opt-in, anonymized/aggregated signals about escalation rates, on-device latency distributions, or crash/error rates) rather than the kind of per-query faithfulness sampling described in Day 22, since sampling actual on-device query content for a centralized faithfulness check would itself violate the privacy guarantee the whole feature is built around. This tension — wanting quality visibility vs. preserving the privacy guarantee that's the point of the on-device tier — is itself worth naming explicitly as a real design trade-off, not something to gloss over.
</details>

**C4.** Combine Day 16 (agentic/multi-hop RAG) with Day 23's context constraints: would you ever run a multi-hop ReAct-style loop entirely on-device? Why or why not?

<details>
<summary>Show answer</summary>
Likely not, or only in a very constrained form. Day 16's latency worked example showed even a moderate 3-hop agentic loop roughly doubling or tripling latency relative to a single-shot pipeline, and each hop consumes both a retrieval call and a reasoning/generation call — on a ~4K-token context budget with a ~3B-parameter model, running multiple full reasoning hops on-device would consume a disproportionate share of an already-tiny context budget across multiple turns, and likely exceed reasonable on-device latency/battery expectations for what's supposed to feel like an instant, Siri-style interaction. A more realistic design: detect (cheaply, likely via a lightweight on-device classifier) whether a query looks like it needs multi-hop reasoning at all, and if so, escalate the *entire* multi-hop process to Private Cloud Compute's larger context and more capable reasoning tier, rather than attempting to run an iterative agentic loop within the on-device tier's tight resource envelope.
</details>

---

## 📊 Weak Spot Tracker

| Section | Topic | Got it cold? | Needs repair? |
|---|---|---|---|
| A1–A5 | System design methodology | ☐ | ☐ |
| B1–B4 | Apple-specific architecture | ☐ | ☐ |
| C1–C4 | Applying methodology to Apple constraints | ☐ | ☐ |

**This is the last review before your mock interviews.** Section C is the actual dress rehearsal for what a real Apple RAG interview will feel like — if any of those answers felt shaky, spend tonight specifically re-reading Days 22 and 23 together, side by side, rather than reviewing them in isolation. The mock interview on Day 25 will not be forgiving of hesitation on exactly this kind of synthesis.

---

*Systems week complete — Days 1-24 fully covered and reviewed. Next up — Day 25: Mock Interview #1 (full 45-minute simulation, mixed conceptual + calculation + system design).*
