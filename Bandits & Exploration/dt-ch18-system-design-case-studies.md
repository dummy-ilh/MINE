# Chapter 18 — System Design Case Studies

*(Two full mock interviews, dialogue format, each with an L5-vs-L6 breakdown.)*

---

## 18.1 Mock Interview 1: "Design the distributed training setup for a large language model on 1,000 GPUs"

**Interviewer**: "We want to train a 70-billion-parameter language model. We have 1,000 GPUs, 8 per server, connected via NVLink within a server and standard InfiniBand between servers. Each GPU has 80GB of memory. Walk me through your training setup."

**Candidate**: "Let me start with the memory budget, since that determines almost everything else. 70B parameters in bf16 for the forward/backward pass is $70\times10^9 \times 2$ bytes = 140 GB just for parameters — already bigger than one GPU's 80GB, before gradients or optimizer states. Adding gradients (another 140 GB in bf16) and Adam's optimizer states, which I'd keep in fp32 for numerical stability — $70\times10^9 \times 4$ bytes $\times 2 = 560$ GB — the unsharded total is well over 800 GB. This model has no chance of fitting on one GPU, or even a handful, without serious sharding.

So I'd reach for ZeRO Stage 3 or FSDP first, since it captures most of the memory benefit with less engineering complexity than manually writing tensor-parallel splits for every layer. But even with 8-way sharding within one server, 800GB / 8 is still 100GB per GPU — more than the 80GB available. So I'd combine ZeRO sharding with actual model parallelism.

Given 8 GPUs per server with NVLink, I'd set tensor-parallel degree to 8 — keeping that communication-heavy all-reduce traffic on the fastest connection, per the communication-locality principle. That uses a full server per tensor-parallel group. From there, I'd add pipeline parallelism to further split the model's layers across several of these 8-GPU tensor-parallel groups — say, pipeline-parallel degree 4, meaning 4 servers (32 GPUs) form one complete model replica. With 1,000 GPUs total, that leaves roughly 31 complete replicas for data parallelism — I'd round to a clean number, say 1,024 GPUs conceptually or adjust the pipeline depth slightly to divide evenly, but the shape is: 8 tensor × 4 pipeline × ~31 data.

I'd layer ZeRO Stage 1 or 2 sharding on top of the data-parallel dimension specifically — sharding optimizer states and gradients across those ~31 replicas — to squeeze out further memory headroom for larger micro-batches or longer sequences, without touching the tensor/pipeline dimensions that are already handling the raw model-fitting problem.

For precision, bf16 throughout, given how much more forgiving it is of underflow than fp16, with fp32 master optimizer states as I mentioned. For memory headroom, I'd also turn on activation checkpointing at transformer-layer boundaries — likely necessary at this scale given how much sequence length affects activation memory.

Operationally: checkpoint asynchronously, at a frequency informed by our expected failure rate — at 1,000 GPUs, we should expect failures close to daily, so I'd want relatively frequent checkpointing and elastic training support so a single device failure doesn't require restarting the whole job."

**Interviewer**: "Why pipeline-parallel degree 4 specifically, and not something else?"

**Candidate**: "Honestly, that number needs to come from the model's actual layer count and per-layer memory footprint, which I don't have exact numbers for here — I'd want to compute how many layers can fit per pipeline stage given the 80GB budget after tensor-parallel sharding, and pick the smallest pipeline depth that makes each stage fit, since deeper pipelines mean a worse bubble, per the memory-vs-bubble tradeoff. I picked 4 as an illustrative placeholder, not a derived number — I'd actually work that out from the model's real memory profile."

---

### Breakdown

- **L5 answer**: identifies ZeRO and some form of model parallelism are needed, and gives a rough shape.
- **L6 answer (the one above)**: additionally (a) walks through the actual memory-budget arithmetic before proposing any solution, (b) explicitly justifies tensor-parallel degree via the communication-locality principle rather than asserting it, (c) explains *why* ZeRO is layered specifically onto the data-parallel dimension rather than replacing tensor/pipeline parallelism, (d) proactively raises checkpointing frequency and elastic training tied to the expected failure-rate math from Chapter 15, and (e) when pressed on an unjustified number, honestly flags it as illustrative rather than defending an arbitrary guess — a mark of real engineering judgment over performative confidence.

---

## 18.2 Mock Interview 2: "Your training job keeps crashing every 6 hours — diagnose and fix"

**Interviewer**: "We're training a large model on 512 GPUs. The job reliably crashes roughly every 6 hours, and each time, we lose significant progress. Walk me through how you'd diagnose and address this."

**Candidate**: "First, I'd separate two questions: why is it crashing, and why are we losing so much progress each time — those call for different fixes.

On the crash frequency: at 512 GPUs, some baseline rate of hardware or network failure is expected — I'd want actual failure logs to see whether this is a consistent hardware issue on specific nodes (which would suggest a bad GPU or a flaky network link that should be physically replaced or excluded) versus a more systemic issue like a memory leak or a numerical instability (say, NaN gradients from insufficient loss-scaling headroom) that would show up regardless of which node happens to hit it. A 6-hour periodicity that's suspiciously consistent makes me lean toward looking for something systemic and reproducible rather than pure random hardware bad luck — that kind of regularity is a real clue worth investigating before assuming it's just normal-scale hardware attrition.

On the lost-progress problem, regardless of root cause: this points directly at checkpointing. If we're losing 'significant progress' every 6 hours, that strongly suggests our checkpoint frequency is too infrequent relative to our failure rate — I'd want to know our current checkpoint interval, and I'd push to shorten it, ideally using asynchronous checkpointing so the more-frequent saves don't themselves eat meaningfully into training throughput.

I'd also want to verify checkpoint *correctness*, not just frequency — is the checkpoint actually restoring cleanly, with optimizer state, data-loader position, and RNG state intact? If any of those are silently missing, we might be resuming from checkpoints that are themselves subtly flawed even when the crash-recovery mechanism itself is technically working.

If the crashes turn out to be genuinely random hardware failures rather than systemic, I'd push for elastic training support, so a single node failure doesn't force a full job restart — letting the remaining healthy nodes continue while a replacement is brought online, rather than treating every failure as an all-stop event."

**Interviewer**: "What would make you suspect a numerical instability specifically, versus hardware?"

**Candidate**: "A few signals: if the crash logs show NaN or Inf values in the loss or gradients right before the crash, that's a strong pointer toward numerical instability rather than hardware — possibly insufficient loss-scaling headroom if we're in fp16, or a learning rate that's too aggressive for the current batch size, especially if we recently scaled up batch size without correspondingly adjusting the learning rate and warmup schedule. I'd check whether the 6-hour mark correlates with a specific training step or a specific point in a learning-rate schedule, rather than assuming it's purely a wall-clock-time coincidence."

---

### Breakdown

- **L5 answer**: identifies checkpointing frequency as likely relevant, and mentions hardware failures as a possible cause.
- **L6 answer (the one above)**: additionally (a) explicitly separates the "why crashing" and "why losing progress" questions as requiring different fixes, rather than treating it as one undifferentiated problem, (b) uses the suspicious 6-hour regularity as a genuine diagnostic clue pointing toward systemic causes over random hardware attrition, (c) checks checkpoint *correctness*, not just frequency, connecting back to Chapter 14's four-component checkpoint completeness requirement, and (d) when pressed, gives specific, checkable numerical-instability signals (NaN/Inf in logs, correlation with LR schedule or a recent batch-size change) rather than a vague "could be numerical issues."

---

## 18.3 Comprehension checks

1. In Mock 1, why did the candidate compute the memory budget before proposing any parallelism strategy?
2. In Mock 1, why was ZeRO sharding applied specifically to the data-parallel dimension rather than replacing tensor/pipeline parallelism?
3. In Mock 2, why did the candidate treat "why is it crashing" and "why are we losing so much progress" as two separate questions?
4. In Mock 2, what specific detail about the crash pattern made the candidate lean toward a systemic cause over random hardware failure?
5. Try applying Mock 1's memory-budget-first approach to a hypothetical: a 30B-parameter model, 256 GPUs, 8 per server — walk through the reasoning yourself before checking against the pattern used above.

---

*Next: Chapter 19 — Rapid-Fire Review & L5-vs-L6 Differentiators, the final chapter — consolidated comparison tables, the most likely follow-up questions with model answers, and a full traps checklist.*
