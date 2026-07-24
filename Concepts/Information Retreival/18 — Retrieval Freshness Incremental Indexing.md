# Chapter 18 — Retrieval Freshness & Incremental Indexing

## What is it?

Every retrieval structure covered so far — inverted indexes (Ch. 2-3), dense ANN indexes (Ch. 13) — was implicitly discussed as if built once, offline, over a static corpus. Real corpora are never static: news breaks, prices change, products go out of stock, social content is posted every second. **Retrieval freshness** is the problem of keeping an index correct and up-to-date as the underlying corpus changes continuously, without paying the full cost of rebuilding the index from scratch every time.

This chapter closes the loop on two things flagged as open problems in earlier chapters: Chapter 13 noted that HNSW graphs are expensive to update incrementally, and Chapter 16 noted that Query Understanding assumes a query correctly reflects current intent — freshness is about making sure the *documents* side of the equation is equally current.

---

## The intuition

**Observation 1 — there's a fundamental tension between index quality and update cost.** The best-quality indexes (a perfectly balanced IVF clustering, a perfectly-tuned HNSW graph) are usually the most expensive to modify incrementally, because their quality comes precisely from having been built holistically over the full, final dataset. A structure optimized for batch-build quality is rarely also optimized for cheap single-item insertion — this is the direct continuation of the tension flagged in Chapter 13.

**Observation 2 — different content has different acceptable staleness**, and treating everything the same is wasteful. A breaking-news query needs sub-minute freshness; a product's core description rarely changes and can tolerate being reindexed once a week; a user's saved documents change whenever they act, requiring near-immediate reflection. Good freshness architecture matches update mechanism to the actual staleness tolerance of the content, rather than one-size-fits-all.

**Observation 3 — the standard solution is a two-tier (or multi-tier) index, not one index that does everything well.** Instead of trying to make one index structure both perfectly-optimized *and* cheaply-updatable — a genuine architectural contradiction — production systems split the index into a **large, high-quality, infrequently-rebuilt main index** and a **small, cheaply-updatable delta/real-time index** that captures everything since the last main rebuild, merging results from both at query time.

---

## The core architectural patterns

### 1. Main + delta index (the standard pattern)

```
Main index:  built via full batch process (e.g., HNSW build, IVF-PQ clustering)
             rebuilt on a schedule (hourly / daily / weekly, depending on content type)
             optimized purely for query-time quality and speed

Delta index: small, simple structure (often flat/brute-force, or a lightweight
             append-friendly structure) holding everything added/changed
             since the last main rebuild
             optimized purely for cheap, fast insertion

Query time: search BOTH indexes, merge results (often via RRF, Chapter 17)
            Deletions handled via a "tombstone" set checked against both
```

Because the delta index only needs to hold recent changes, it stays small — brute-force search over it is cheap even though brute-force over the *main* corpus would not be (this directly reuses the "why brute-force is sometimes fine" reasoning from Chapter 13, just scoped to a small subset).

### 2. Tombstoning for deletions

Deleting a vector from HNSW or an IVF bucket is expensive (rewiring graph edges, or leaving fragmented buckets). Instead, maintain a separate **tombstone set** of deleted document IDs; at query time, filter out any tombstoned ID from the results *before* returning them, and only physically remove tombstoned entries during the next full index rebuild.

```
query_results = search(index, q, k = k_requested + expected_tombstone_overlap)
final_results = [r for r in query_results if r.id not in tombstone_set][:k_requested]
```

Note the `k_requested + expected_tombstone_overlap` — you need to over-fetch slightly, since some of the top-k raw results may get filtered out by the tombstone check, and you still need to return a full k results to the user.

### 3. Content-tiered update SLAs

Rather than one global freshness target, define explicit SLAs per content type based on actual staleness tolerance, and route each type to update mechanisms that match:

```
Breaking news:       delta index, near real-time ingestion, sub-minute target
Product availability: delta index or direct DB lookup at query time, minutes target
Product descriptions: main index, daily rebuild, hours-to-a-day tolerance acceptable
Archived/static docs: main index, weekly (or less frequent) rebuild
```

---

## Worked numeric example

```
Main index: 100,000,000 documents, HNSW, rebuilt daily
  Full rebuild cost: ~4 hours of cluster compute time (fixed, regardless of how many
                      documents actually changed since yesterday)

Daily document churn: 500,000 new/changed documents (0.5% of corpus)

Delta index: flat/brute-force over just the day's churned documents
  Delta search cost per query: 500,000 vectors × 768 dims ≈ 384M FLOPs
  (compare to main index ANN search: ~O(log 100,000,000) ≈ 27 "hops", each touching
   a small constant number of candidates — dramatically cheaper than 384M FLOPs,
   but the delta index doesn't need ANN speed since it's 200× smaller than the main corpus)
```

**Without a delta index (naive full rebuild on every change):**

```
500,000 changed documents/day → if each change triggered a full rebuild:
  500,000 × 4 hours = 2,000,000 compute-hours/day  ← completely infeasible
```

**With a delta index (main rebuilt once/day, delta absorbs everything in between):**

```
1 main rebuild/day: 4 compute-hours
Delta index: absorbs all 500,000 changes with O(1) append cost each ≈ negligible
Query-time cost: 1 main ANN search (~log-scale) + 1 delta brute-force search (384M FLOPs,
                  still cheap in absolute terms since it's a bounded, small set)

Total: 4 compute-hours/day (fixed) + negligible per-change cost + small added query latency
```

**The core arithmetic that matters:** rebuilding the full 100M-document main index costs the same fixed ~4 hours whether 1 document changed or 500,000 did — the delta index exists specifically to absorb that volume of change *between* rebuilds without incurring the full rebuild cost per change, at the price of a slightly more expensive (but still cheap in absolute terms, because it's scoped to a small recent-changes set) per-query search that touches two indexes instead of one.

---

## Why it works / why it fails

**Why it works:**
- The main+delta split lets each tier be architecturally specialized for what it needs to be good at — the main index for query-time quality/speed at massive scale, the delta index for cheap, simple, fast writes — rather than forcing one structure to compromise on both simultaneously, which (as Chapter 13 established) is close to a structural contradiction for graph/cluster-based ANN indexes.
- Tombstoning avoids the expensive graph-rewiring or bucket-fragmentation cost of true deletion, deferring the actual cleanup to the next full rebuild when it can be done holistically and cheaply (a rebuild has to touch every document anyway, so folding in "don't include tombstoned IDs" costs nothing extra).
- Tiering update SLAs by content type avoids wasting engineering and compute effort chasing sub-minute freshness for content that genuinely doesn't need it, freeing that effort/budget for the content types (breaking news, live availability) where staleness is actually user-visible and costly.

**Why it fails / risks:**
- **Delta index grows unboundedly between rebuilds if rebuild cadence slips.** If the main rebuild is delayed (compute contention, a pipeline failure), the delta index keeps absorbing changes and can grow large enough that its brute-force search stops being cheap — the entire architecture's cost model assumes the delta stays small, which depends on rebuilds actually happening on schedule.
- **Query-time complexity increases.** Every query now searches two (or more) indexes and merges results (often via RRF, Chapter 17), adding latency and engineering surface area compared to a single-index query — and results need careful deduplication, since a document that was updated may appear in both the (stale) main index and the (fresh) delta index simultaneously, requiring the delta version to take precedence.
- **Tombstone set growth.** If deletions are frequent and rebuilds infrequent, the tombstone set itself grows and must be checked on every query — an increasingly expensive filter step until the next rebuild physically removes those entries, meaning the tombstone approach has the same "must actually rebuild on schedule" dependency as delta index growth.
- **Some content categories genuinely resist tiering.** For a source with continuously, uniformly critical freshness needs across its entire corpus (not just a small hot subset), the "small delta absorbs the hot changes" assumption breaks down — the delta index itself would need to be nearly as large as the main index, defeating the purpose.

---

## The one thing to remember

Freshness architectures exist because rebuilding a high-quality index from scratch has a fixed cost regardless of how much actually changed, so the standard solution is to decouple "how good does the index need to be" (main index, rebuilt on a schedule) from "how fast does a change need to be reflected" (delta index, cheap and immediate), matching each content type's actual staleness tolerance rather than optimizing one global freshness target for everything.

---

## Formulas used in this chapter

| Formula | Meaning |
|---|---|
| Full rebuild cost is ~fixed regardless of churn volume | Motivates decoupling update frequency from rebuild cost via a delta index |
| `final_results = filter(search_results, not in tombstone_set)[:k]` with over-fetch | Handling deletions cheaply without touching the index structure itself |

---

## Interview Q&A

**Q1. Why not just make the main index update incrementally instead of maintaining a separate delta index?**

Because the structures that give the main index its query-time quality and speed (well-balanced IVF clusters, a well-connected HNSW graph) get that quality specifically from being built holistically over the full dataset — incremental insertion into an already-mature HNSW graph requires expensive per-insert neighbor search and potential edge rewiring (as established in Chapter 13), and incremental insertion into IVF clusters can gradually unbalance cluster sizes as the data distribution drifts from what the original k-means clustering assumed. Rather than accepting degraded main-index quality in exchange for cheap updates, the standard architecture keeps the main index's build process untouched (batch, periodic) and absorbs the update-cost problem entirely in a separate, architecturally simpler delta structure whose only job is cheap writes.

**Q2. Why is tombstoning preferred over immediately deleting a vector from the index?**

True deletion from a graph-based (HNSW) or cluster-based (IVF) index requires actually modifying the structure — removing a node from HNSW can fragment its neighbors' connectivity and require re-establishing alternate paths; removing from an IVF bucket is cheaper but still touches the live structure mid-operation, right when it's also serving queries. Tombstoning instead just adds the ID to a separate "excluded" set checked at query time, leaving the underlying index structure completely untouched until the next full rebuild, at which point removing tombstoned entries is essentially free (the rebuild has to enumerate every document anyway). This trades a small, bounded query-time filtering cost (and a slight over-fetch to compensate) for avoiding expensive, structure-modifying deletions on the hot path.

**Q3. In the worked example, why does the delta index's brute-force search stay cheap even though brute-force search was called infeasible for the main index in Chapter 13?**

Brute-force cost scales with corpus size, and the delta index is deliberately kept small by design — in the example, 500,000 documents versus the main index's 100,000,000, roughly 200x smaller. Brute-force search over 500,000 vectors (384M FLOPs) is entirely tractable at typical query latencies, whereas brute-force over 100,000,000 would not be. The entire architecture depends on this size asymmetry holding — which is precisely why "delta index growing too large because rebuilds are delayed" is a real risk: if the delta grows toward main-index scale, its brute-force search stops being cheap and the architecture's core assumption breaks.

**Q4. How would you decide the rebuild cadence for the main index, and does it need to be the same for every content type?**

It should not be uniform — the right approach is to define freshness SLAs per content type based on actual staleness tolerance (as in the tiering example: sub-minute for breaking news, hours-to-daily for product descriptions, weekly-or-less for static archives) and choose rebuild cadence and update mechanism (delta index vs. direct database lookup vs. full rebuild) to match each tier's requirement, rather than picking one global cadence. The cost trade-off to balance explicitly: more frequent main rebuilds mean fresher content and a smaller, cheaper delta index, but higher fixed compute cost paid more often; less frequent rebuilds are cheaper in aggregate compute but push more load and staleness risk onto the delta tier.

**Q5. What happens at query time if a document exists in both the main index (stale version) and the delta index (updated version) — how should the system handle that?**

The system needs explicit deduplication logic that prefers the delta index's version whenever a document ID appears in both, since the delta index by construction holds the more recent state. This typically means: after searching both indexes and before (or during) fusion/merging (often via RRF, Chapter 17), deduplicate by document ID and keep only the delta version's score/content for any ID present in both, discarding the main index's stale copy. This is an important detail because naively merging both search results without this precedence rule could either double-count the same underlying document under two different states, or — worse — surface the stale main-index version in the final ranked results despite a fresher version being available in the delta index.
