# Chapter 14 — Checkpointing Strategies

*(Plain language first — opening Phase 5, the operational realities of long-running training jobs.)*

---

## 14.1 Why "just save the weights" is the wrong mental model at scale

For a small model on one GPU, checkpointing is trivially simple: periodically write the model's weights to disk. At the scale this course has been building toward — hundreds or thousands of devices, ZeRO-sharded state, 3D-parallel layouts — checkpointing becomes a genuinely harder systems problem, for reasons this chapter walks through one at a time.

---

## 14.2 What actually needs to be saved — more than just the weights

**A complete, resumable checkpoint needs to capture**:

- **Model parameters** — the obvious one.
- **Optimizer state** — recall from Chapter 1 that Adam keeps two running-average buffers per parameter. If you resume training with a freshly-reset optimizer state instead of the real, accumulated one, you lose all of Adam's accumulated momentum/variance information — the training dynamics after resuming won't match what would have happened if the run had never stopped, even though the weights themselves are identical.
- **Data-loader position** — exactly which examples/batches have already been consumed, so resumed training doesn't accidentally skip data or repeat the same data twice.
- **RNG (random number generator) state** — anything relying on randomness (dropout, data shuffling order, and in this course's earlier material, things like loss-scaling's dynamic adjustment) needs its random state saved too, or resumed training silently follows a different random trajectory than a true, uninterrupted continuation would have.

**The single most important point in this section**: **missing any one of these pieces doesn't cause an obvious crash — it silently produces a resumed run that's subtly different from an uninterrupted one**, which is a genuinely dangerous, hard-to-detect failure mode (you might not notice for days that your "resumed" run has been quietly training somewhat differently than intended).

---

## 14.3 Why this is much harder with sharded state (ZeRO/FSDP)

Recall Chapter 9: under ZeRO Stage 2/3 or FSDP, the optimizer states (and possibly gradients and parameters) are **sharded** — no single GPU holds the complete picture. **Saving a checkpoint now means coordinating across every GPU that holds a shard of the state**, gathering (or separately writing) each piece, and doing so in a way that produces a complete, consistent, reloadable checkpoint — not just each GPU independently dumping its own local shard to disk with no coordination, which could easily produce a corrupted or incomplete result if, say, one GPU's shard reflects a slightly different training step than another's.

**This is directly analogous to a database consistency problem**, if that framing helps: you need a **consistent snapshot** across many distributed pieces of state, not several independently-taken snapshots that might not actually correspond to the same overall moment in training.

---

## 14.4 Synchronous vs. asynchronous checkpointing

**Synchronous checkpointing**: pause all training computation, have every device write out its piece of the state, wait for every write to complete, and only then resume training. **Simple and safe** (you know the checkpoint is fully complete and consistent before continuing) — but it **directly costs wall-clock training time**, since the GPUs sit idle during the write.

**Asynchronous checkpointing**: kick off the checkpoint-writing process, but **allow training computation to continue in the background** while the (typically CPU-driven, or otherwise separately-resourced) checkpoint write happens concurrently — the idea being to hide most of the checkpointing cost behind ongoing useful computation, similar in spirit to the communication-computation overlap idea mentioned back in Chapter 4. **The complication**: you need to be careful that the state being written doesn't get modified by the still-running training process partway through the write (a genuine, real synchronization challenge) — typically handled by first quickly copying the relevant state to a separate buffer, then writing that frozen copy out asynchronously while training proceeds on the "live" copy.

**The practical tradeoff, stated plainly**: synchronous checkpointing is simpler to reason about but wastes training time; asynchronous checkpointing recovers that wasted time but adds real implementation complexity to avoid corrupting the in-progress write — a genuine engineering tradeoff, not a strictly-better-strictly-worse comparison.

---

## 14.5 Checkpoint frequency — another real tradeoff

**Checkpoint too rarely**: if training crashes (and, per the next chapter, it eventually will, at large enough scale), you lose all the progress made since the last checkpoint — potentially many hours or even days of expensive compute, wasted.

**Checkpoint too often**: each checkpoint costs real time (Section 14.4) and real storage — checkpointing constantly would meaningfully slow down overall training throughput, even with asynchronous checkpointing softening the direct time cost.

**The practical way this gets decided**: weigh the *expected* cost of lost progress from a crash (which depends on how *likely* a crash is in a given time window — directly connecting to the next chapter's fault-tolerance material) against the *certain*, recurring cost of each checkpoint itself — a classic expected-value tradeoff, not a fixed, one-size-fits-all number.

---

## 14.6 Recovering from a crash — putting it all together

**When a training job crashes and needs to resume from the most recent checkpoint**, the recovery process needs to correctly restore *every* piece from Section 14.2 — reload the (possibly still-sharded) model parameters and optimizer state onto the correct devices in the correct layout, reposition the data loader to the correct point, and restore the RNG state — and do so in a way that's faithful to the original 3D-parallelism layout (Chapter 8) the job was using, which might even need to be reconstructed if, say, you're resuming on a different number of devices than the original run used (a genuinely harder version of this problem, sometimes called **elastic checkpoint resumption**, briefly worth knowing the name of even without full mechanical depth here).

---

## 14.7 Production considerations

- **Checkpoint frequency for large frontier-model training runs is a real, carefully-tuned operational parameter** — not an afterthought — precisely because both failure likelihood (next chapter) and per-checkpoint cost (this chapter) are significant, well-understood quantities that infrastructure teams actively reason about and tune.
- **Frameworks like DeepSpeed and Megatron-LM have built-in, tested support for sharded checkpoint save/load**, correctly handling the Section 14.3 coordination problem — this is not something most teams reimplement from scratch, and knowing these frameworks handle it is itself a useful, concrete fact.
- **The RNG-state and data-loader-position details (Section 14.2) are exactly the kind of "easy to forget, hard to debug" details that separate a genuinely production-ready checkpointing system from a naive one** — worth naming specifically and proactively, since they're commonly overlooked even by people who understand the broader checkpointing idea correctly.

---

## 14.8 Interview traps

- **Describing checkpointing as "just saving the weights."** As shown in Section 14.2, a complete, safely-resumable checkpoint needs optimizer state, data-loader position, and RNG state too — omitting any of these produces a subtly, silently different resumed run, not an obvious failure.
- **Not recognizing why sharded state (ZeRO/FSDP) makes checkpointing meaningfully harder** — a candidate should be able to explain the distributed-consistent-snapshot problem from Section 14.3, not just say "you save each GPU's piece."
- **Presenting asynchronous checkpointing as a strictly free win over synchronous checkpointing**, without acknowledging the real synchronization complexity it introduces (making sure the state being written isn't mutated mid-write) — this is a genuine engineering tradeoff, not a one-directional improvement.

---

## 14.9 L5-vs-L6 differentiating talking points

- **L5 bar**: knows checkpointing needs to save more than just weights, and knows synchronous vs. asynchronous checkpointing are both options.
- **L6 bar**:
  - Can list all four components from Section 14.2 (parameters, optimizer state, data-loader position, RNG state) unprompted, and explain specifically *why* omitting each one causes a silent, hard-to-detect discrepancy rather than an obvious crash.
  - Frames sharded-state checkpointing explicitly as a distributed-consistent-snapshot problem (Section 14.3), rather than a simple "everyone saves their own piece" description.
  - Reasons about checkpoint frequency as a genuine expected-value tradeoff (Section 14.5) — connecting failure likelihood to checkpointing cost — rather than citing a fixed rule of thumb with no underlying justification.

---

## 14.10 Comprehension checks

1. List the four components a complete, resumable checkpoint needs to save, and explain what silently goes wrong if you omit each one.
2. Why does ZeRO/FSDP's sharded state make checkpointing meaningfully harder than in plain (unsharded) data parallelism?
3. What's the core tradeoff between synchronous and asynchronous checkpointing?
4. How would you reason about how frequently to checkpoint a given training job?
5. What does "elastic checkpoint resumption" refer to, at a high level?

---

*Next: Chapter 15 — Fault Tolerance and Elastic Training, covering why failures are a certainty (not an edge case) at large scale, elastic training, and straggler mitigation.*
