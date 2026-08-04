# RAG Interview Prep — Day 18
## Review Day: Generation Week (Days 13–17) — Closed Book

---

## 📋 How to run this review

1. No notes. Close Days 13–17 (and ideally everything before).
2. Answer, then check `<details>`. Log gaps in the Weak Spot Tracker.
3. Target: 60–90 minutes for all 26 questions.
4. This is the last review before Day 19's Evaluation recap and Day 20's Diagnosis day — both build directly on your ability to reason across the *entire* pipeline, not just one week at a time, so the cross-week synthesis section here matters more than usual.

---

## Section A — Day 13: Context Construction & Lost-in-the-Middle

**A1.** Describe the lost-in-the-middle effect and its practical implication for ordering retrieved chunks.

<details>
<summary>Show answer</summary>
LLMs attend less reliably to information placed in the middle of a long context vs. the start or end — a U-shaped accuracy curve by position. Practically, this means the most relevant chunks (as determined by reranking) should be placed near the start and/or end of the prompt, not simply left in whatever order retrieval/reranking produced them, since a naive top-to-bottom listing can waste a high-attention end position on a weaker chunk.
</details>

**A2.** Why might increasing k hurt generation quality even while improving Recall@k?

<details>
<summary>Show answer</summary>
More chunks means more content pushed into the vulnerable middle zone and more dilution from marginal/irrelevant content — both can hurt context relevance and faithfulness even though Recall@k (which only checks presence anywhere in top-k) technically improves. Optimal k for generation should be tuned against generation-stage metrics, not retrieval recall alone.
</details>

---

## Section B — Day 14: Context Compression & Caching

**B1.** Why is chunk deduplication considered nearly risk-free compared to abstractive compression?

<details>
<summary>Show answer</summary>
Deduplication only removes near-duplicate content (no unique information lost by definition) using cheap similarity comparisons, with no LLM call and no rewriting involved. Abstractive compression is itself a generation step that can introduce hallucination or subtle information drift, compounding faithfulness risk on top of the final generation step.
</details>

**B2.** What's the key difference between prefix/KV caching and semantic caching, including their respective risks?

<details>
<summary>Show answer</summary>
Prefix/KV caching reuses computed attention states for an exact, identical prompt prefix (e.g., a fixed system prompt) — risk-free, purely an infrastructure optimization. Semantic caching serves a cached answer for queries that are similarity-close (not identical) to a previously-cached query — it requires threshold tuning (too low risks false-positive cache hits, too high minimizes hit rate) and carries a genuine staleness risk if not tied to cache invalidation on knowledge-base updates.
</details>

---

## Section C — Day 15: Citation & Faithfulness Enforcement

**C1.** Why can post-hoc attribution be more reliable than inline citations generated during the same pass as the answer?

<details>
<summary>Show answer</summary>
Inline citations are self-reported by the same model generating the claim, risking fabricated citations that look plausible but don't actually correspond to a supporting source. Post-hoc attribution decouples claim generation from citation assignment — an independent verification step (NLI/embedding-based) checks each claim against retrieved chunks and assigns citations based on that check, not the model's self-report.
</details>

**C2.** Frame refusal calibration as a two-sided error problem.

<details>
<summary>Show answer</summary>
False refusal (declining a question actually answerable from context — unhelpful) vs. false answer (confidently answering when context is insufficient — worse, especially in high-stakes domains). The correct threshold trades these off and is domain-specific, not universal — requires a golden eval set with both known-sufficient and known-insufficient examples to measure both error rates separately.
</details>

---

## Section D — Day 16: Multi-Hop & Agentic RAG

**D1.** Why can't static decomposition (Day 11) handle a question like "who is the CEO of the company that acquired the maker of X"?

<details>
<summary>Show answer</summary>
The second sub-question ("who acquired the maker of X") can't be formulated until you know the answer to the first ("who made X") — static decomposition commits to all sub-questions upfront, before any retrieval happens, so it structurally cannot handle a question where a later sub-question's content depends on an earlier retrieval's actual result.
</details>

**D2.** What is error propagation in multi-hop retrieval, and why is static decomposition largely immune to it?

<details>
<summary>Show answer</summary>
If an early hop retrieves incorrect information, later hops' queries are built directly on that faulty premise, compounding the error rather than being independently correctable. Static decomposition's sub-questions are independent of each other — a bad retrieval on one doesn't feed into or corrupt a different sub-question's retrieval.
</details>

---

## Section E — Day 17: Failure Modes Catalog

**E1.** Distinguish generic hallucination from over-reliance on parametric knowledge.

<details>
<summary>Show answer</summary>
Generic hallucination is inventing content absent from the context entirely. Over-reliance on parametric knowledge is subtler — correct information WAS in the context, but the model answered from a strong pretrained default instead, typically because that fact was learned very consistently during training and competes with/overrides the provided (correct but perhaps less-reinforced) context.
</details>

**E2.** Why doesn't a standard golden eval set reliably surface knowledge-conflict failures?

<details>
<summary>Show answer</summary>
This failure only manifests when context contradicts a strong parametric prior — typical eval questions usually have context and parametric knowledge in agreement, giving no opportunity for the conflict to surface. Detecting it requires a deliberately-constructed counterfactual eval slice engineered specifically to create that contradiction.
</details>

---

## Section F — Cross-Week Synthesis (the hardest section)

**F1.** A RAG system uses agentic multi-hop retrieval (Day 16) with context sandwiching (Day 13) applied within each hop's context construction. One hop's context accidentally places its most relevant chunk in the middle instead of sandwiched. How does this specific context-assembly mistake compound with multi-hop's error propagation risk (Day 16), compared to if the same mistake happened in a single-shot RAG pipeline?

<details>
<summary>Show answer</summary>
In a single-shot pipeline, a lost-in-the-middle mistake risks one wrong or incomplete final answer — a contained failure. In a multi-hop pipeline, that same mistake at an early hop risks the model failing to properly register or use the relevant evidence at that hop, potentially leading it to reason forward from an incomplete or wrong intermediate conclusion — exactly Day 16's error propagation mechanism, except the root cause here is a context-assembly failure (Day 13) rather than a retrieval failure. This shows failure modes from different "layers" (context assembly vs. multi-hop orchestration) can compound with each other, not just occur independently — a lost-in-the-middle mistake is more costly inside a multi-hop loop than in a single-shot pipeline, because it can corrupt everything built on top of that hop.
</details>

**F2.** How would you distinguish, in production, between a refusal miscalibration failure (Day 17) and an over-reliance-on-parametric-knowledge failure (Day 17), given that both can produce "wrong answer when context suggested something else" symptoms?

<details>
<summary>Show answer</summary>
Refusal miscalibration (specifically the false-answer type) occurs when context is genuinely INSUFFICIENT and the model should have declined but didn't — the model is essentially guessing without adequate grounding. Over-reliance on parametric knowledge occurs when context is genuinely SUFFICIENT and correct, but the model ignored it in favor of a conflicting pretrained belief. Distinguishing them requires checking whether the retrieved context, if properly used, actually contained the correct answer: if yes, and the model still gave a different (wrong, parametric-matching) answer, that's over-reliance on parametric knowledge; if the context genuinely didn't contain enough to answer correctly and the model answered anyway, that's a false-answer refusal-miscalibration failure. This distinction directly determines the fix — recency signaling/fine-tuning on conflict examples vs. refusal-threshold recalibration — so conflating them would lead to the wrong intervention.
</details>

**F3.** Connect Day 14's semantic caching staleness risk to Day 17's over-reliance-on-parametric-knowledge failure mode — how could stale caching actively make this specific failure mode worse?

<details>
<summary>Show answer</summary>
If a semantic cache serves a stale cached answer from before a policy/fact change (Day 14's staleness risk), the served answer effectively reverts to outdated information — functionally similar in symptom to the model ignoring updated context in favor of an old "default," except the root cause here is caching infrastructure, not model behavior. Worse, if the cache was populated back when the model itself exhibited over-reliance on parametric knowledge (giving the old, "common" answer even when context had already been updated at that time), the cache would now be permanently serving that already-wrong answer, compounding two separate failure modes (stale cache + original knowledge-conflict generation failure) into one entrenched, harder-to-detect error that persists even after the underlying model behavior might otherwise self-correct on a fresh (uncached) query.
</details>

**F4.** A system passes all offline evaluation (high faithfulness, high nDCG, good Recall@k) but production users report occasional confidently-wrong answers specifically on questions about recently-changed information. Using the full taxonomy from Days 1-17, name the most likely failure mode and explain why offline eval missed it, tying together at least two specific days' concepts.

<details>
<summary>Show answer</summary>
Most likely: over-reliance on parametric knowledge (Day 17), specifically triggered by recently-changed information — exactly the scenario knowledge conflict manifests in. Offline eval likely missed it because the golden eval set (Module 7 §7.6 / Day 3's chunk-tuning eval set discussion) probably wasn't deliberately constructed with counterfactual examples contradicting strong parametric priors — a standard eval set built from typical, non-adversarial questions would show high faithfulness/recall/nDCG scores while this narrow, high-impact failure mode goes completely undetected, since it only manifests under the specific and easily-overlooked condition of context contradicting a well-known fact. This ties Day 17's failure mode directly to Module 7's broader lesson that eval set construction quality determines what failures you can even see — a system can look excellent on paper while having a real, specific blind spot that simply wasn't tested for.
</details>

---

## 📊 Weak Spot Tracker

| Question # | Topic | Got it cold? | Needs repair? |
|---|---|---|---|
| A1–A2 | Context construction / lost-in-middle | ☐ | ☐ |
| B1–B2 | Compression & caching | ☐ | ☐ |
| C1–C2 | Citation & faithfulness enforcement | ☐ | ☐ |
| D1–D2 | Multi-hop / agentic RAG | ☐ | ☐ |
| E1–E2 | Failure modes catalog | ☐ | ☐ |
| F1–F4 | Cross-week synthesis | ☐ | ☐ |

**Reminder:** Section F misses matter most — by this point in the curriculum, an interviewer will be actively probing whether you can connect concepts across the whole pipeline, not just recite individual days.

---

*Generation week complete. Next up — Day 19: Evaluation Deep Recap (cold review of your existing Module 7 notes).*
