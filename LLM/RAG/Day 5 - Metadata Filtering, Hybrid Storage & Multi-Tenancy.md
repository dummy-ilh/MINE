# RAG Interview Prep — Day 5
## Metadata Filtering, Hybrid Storage & Multi-Tenancy

---

## 🚀 Quick Summary

Pure vector similarity search answers "what's semantically closest to this query" — but real production systems almost always need to combine that with hard structured constraints: "...but only documents from this region," "...only from the last 90 days," "...only what this specific user is allowed to see." Today covers how metadata filtering actually interacts with ANN index structures (it's not a free `WHERE` clause), how to architect hybrid storage that supports both fuzzy and exact constraints efficiently, and how to design multi-tenant vector systems that balance cost, isolation, and performance — including the access-control and data-deletion implications that come up constantly in real enterprise RAG systems.

**Think of it like a library with locked sections and patron records.** Pure similarity search is "find me books about this topic." Real usage is "find me books about this topic — that were published after 2023, that are in the medical wing, and that this specific patron's library card is authorized to check out." The card-check and section-lock aren't optional add-ons bolted onto search after the fact — they change *how* the librarian searches the shelves in the first place, and getting that wrong either leaks restricted books to the wrong patron or makes the search reader unusably slow for popular filters.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Metadata filtering** | Combining vector similarity search with structured field constraints (date, category, tenant, permission) |
| **Pre-filtering** | Narrowing the candidate set by metadata *before* similarity search runs |
| **Post-filtering** | Running similarity search first, then discarding non-matching results afterward |
| **Selectivity** | What fraction of the corpus survives a given filter — highly selective = few matches, low selectivity = most things match |
| **Filtered ANN index** | An ANN implementation designed to efficiently combine graph/cluster traversal with filter constraints mid-search |
| **Multi-tenancy** | Serving multiple distinct customers/orgs/users from shared infrastructure |
| **Namespace / partition** | A logical (sometimes physical) subdivision of an index, often used to isolate tenants |
| **Noisy neighbor problem** | One tenant's heavy usage degrading performance for other tenants sharing the same infrastructure |
| **Row-level security (RLS)** | Access control enforced at the level of individual records/vectors, not just at the table/index level |
| **Right to be forgotten** | A legal/compliance requirement (e.g., GDPR) that a specific user's data can be fully deleted on request |

---

# PHASE 1 — Intuition & Visual Map

## Why filtering isn't "just a WHERE clause"

```
   NAIVE MENTAL MODEL                    WHAT ACTUALLY HAPPENS

   [vector search] ──▶ [SQL-style       [vector search] ◀──filter-aware──▶
                         WHERE filter]    [ANN index structure]
                                                    │
                                          The filter changes HOW the
                                          graph/cluster traversal
                                          happens, not just what
                                          gets shown afterward
```

In a relational database, adding a `WHERE` clause to a query is nearly free — indexes on structured columns are cheap and composable. In an ANN vector index, the graph (HNSW) or cluster structure (IVF) was built assuming a *particular* corpus is fully eligible for search. A filter that suddenly makes 99% of that corpus ineligible doesn't just "narrow the results" — it can break the assumptions the index's shortcuts were built on, forcing something close to a much slower search within the eligible subset unless the index itself was designed to be filter-aware.

## When to reach for which pattern

- ✅ **Post-filtering** — filters with **low selectivity** (most of the corpus matches, e.g. `language = "en"` on a mostly-English corpus) — cheap, simple, minimal recall risk since most top-k results will already pass the filter.
- ✅ **Pre-filtering / filter-aware ANN** — filters with **high selectivity** (only a small fraction of the corpus matches, e.g. `tenant_id = X` in a large shared multi-tenant index) — post-filtering here would discard most of your top-k, leaving too few results.
- ✅ **Per-tenant / per-partition indexes** — when isolation, predictable performance, or compliance requirements matter more than infrastructure cost efficiency.
- ❌ Avoid pure post-filtering on any filter you know in advance will be highly selective — it's a very common, very avoidable production bug (retrieving top-10, having 8 discarded by the filter, silently returning only 2 results).

---

# PHASE 2 — Deep Dive: Mechanics, Math, and Architecture

## Pre-filtering vs. Post-filtering — Quantified

**Worked numerical example — why selectivity determines which approach works:**

Suppose you request the top-10 most similar documents, and a metadata filter matches only **2% of the corpus** (a highly selective filter, e.g., a specific tenant in a large shared index of many tenants).

**Post-filtering approach:**
```
Retrieve top-10 by similarity (ignoring filter) → apply filter afterward

Expected number of the unfiltered top-10 that also match a 2% filter
≈ 10 × 0.02 = 0.2 results on average

→ You'd typically get ZERO results back, unless you retrieve a much
  larger candidate set (e.g., top-500) before filtering down — which
  means paying the cost of retrieving 500 candidates just to end up
  with usable results, and even then, you're not guaranteed 10.
```

**Pre-filtering (naive, non-filter-aware index) approach:**
```
Restrict candidate pool to the matching 2% first, THEN search
→ Correct results guaranteed if 10+ matches exist, BUT if the ANN
  index (e.g., HNSW graph) wasn't built with this filter in mind,
  searching only within an arbitrary 2% subset of the graph can
  degrade toward brute-force-like behavior within that subset,
  since the graph's shortcuts assumed the full corpus was eligible.
```

**Filter-aware ANN index:** designed so the graph/cluster traversal itself is aware of the filter and can efficiently skip ineligible regions *during* traversal, rather than either discarding results afterward or falling back to a slow scan of the filtered subset — this is the actual production-grade solution once selectivity gets high, and it's what modern vector databases increasingly implement (e.g., Qdrant's filtered-HNSW approach, or partition/namespace-based designs that sidestep the problem architecturally).

**Rule of thumb to state in an interview:** as filter selectivity gets *tighter* (fewer matches), the case for pre-filtering (or a filter-aware/partitioned architecture) gets *stronger* — because post-filtering's "waste rate" scales directly with how restrictive the filter is.

---

## Hybrid Storage Architectures

**The core idea:** combine a vector index (for fuzzy semantic search) with structured storage (for exact/range constraints) so a single logical query can use both — without requiring two separate systems and a slow, error-prone application-level join.

### Common architectural patterns

| Pattern | How it works | Trade-offs |
|---|---|---|
| **Native filtering inside the vector DB** | The vector database itself stores metadata alongside vectors and supports filter-aware ANN search natively (most modern managed vector DBs: Pinecone, Weaviate, Qdrant, Milvus) | Simplest to operate — one system, one query. Filter capability and performance at high selectivity varies by product/index implementation. |
| **Vector search + separate structured DB, app-level join** | Run vector search to get candidate IDs, then query a separate relational/structured DB to filter/enrich, join results in application code | More flexible for complex structured queries (joins across many tables), but adds latency (two round trips) and complexity, and risks the post-filtering "too few results" problem if not careful about candidate set size |
| **Postgres + pgvector** | Vector search and structured filtering live in the *same* database, using normal SQL with a vector extension | Attractive when you're already SQL-invested and want ACID guarantees, joins, and vector search in one place — but generally has a lower ceiling on raw ANN performance/scale compared to purpose-built vector databases at very large corpus sizes |

> **Why This Matters callout:** A common Apple-style follow-up is "would you build this as one system or two?" The strong answer isn't a fixed preference — it's naming the actual trade-off: a unified system (native filtering, or pgvector) reduces latency and operational complexity but may cap out on ANN performance/scale sooner; a split architecture (vector DB + separate structured store) scales each piece independently but adds cross-system latency and consistency complexity. Choose based on scale requirements and how complex your structured-query needs actually are.

---

## Multi-Tenancy Architecture — Deep Dive

**The central tension:** shared infrastructure is cheaper and simpler to operate, but isolation, security, and predictable performance get harder to guarantee as tenants share more.

### Isolation strategies, from least to most isolated

**1. Shared index + metadata filter (`tenant_id` field)**
- **Pros:** Cheapest — one index, pooled storage/compute efficiency across all tenants; simplest to operate and monitor.
- **Cons:** A filtering bug is a cross-tenant data leak — a serious, high-severity incident class. Performance for small tenants can suffer if the shared index isn't filter-aware and their filter is highly selective (see the math above). Also exposed to the **noisy neighbor problem** — one tenant running a burst of heavy queries can degrade latency for every other tenant sharing the same underlying compute.

**2. Shared infrastructure, per-tenant namespace/partition**
- **Pros:** Logical separation without fully separate infrastructure — many managed vector DBs (Pinecone namespaces, Milvus partitions) support this natively, giving much of the isolation benefit at a fraction of the operational cost of fully separate indexes.
- **Cons:** Still shares underlying compute/storage resources, so noisy-neighbor effects and blast-radius of infrastructure-level bugs aren't fully eliminated — it's a middle ground, not full isolation.

**3. Fully separate index per tenant**
- **Pros:** Strongest isolation by construction — no filtering logic to get wrong, predictable per-tenant performance regardless of what other tenants are doing, cleanest story for strict compliance/contractual requirements (common in enterprise/regulated industries).
- **Cons:** Infrastructure and operational overhead scale roughly linearly with tenant count — index build/maintenance, monitoring, and cost multiply per tenant, which becomes expensive and operationally heavy at large tenant counts (thousands of small tenants would be impractical this way).

### Decision framework

| Factor | Leans toward shared+filter | Leans toward per-tenant isolation |
|---|---|---|
| **Tenant count** | Many (hundreds–thousands) small tenants | Few (tens) large, high-value tenants |
| **Compliance requirements** | Low/moderate | Strict (healthcare, finance, government contracts) |
| **Cost sensitivity** | High | Lower (isolation cost is acceptable) |
| **Filter selectivity per tenant** | Low-to-moderate (tenants have meaningfully-sized data) | Very high (each tenant is a tiny slice of a huge shared pool) |
| **Noisy-neighbor tolerance** | Higher tolerance / less critical workload | Low tolerance — predictable performance is a hard requirement |

> **Business example:** A B2B enterprise RAG SaaS product with 5 large healthcare-system customers (each with strict HIPAA-adjacent compliance requirements and large document volumes) is a clear case for **per-tenant indexes** — few tenants, high compliance stakes, cost of isolation easily justified. A consumer-facing app with 50,000 individual small users each uploading a handful of personal documents is a clear case for a **shared index with per-user filtering** (or per-user namespaces if the platform supports them cheaply) — per-tenant indexes at that count would be operationally absurd.

---

## Access Control & Row-Level Security in RAG

**Why this matters specifically for RAG (not just generic app security):** In a RAG system, the retrieved context is what the *generator* conditions its answer on — if access control is enforced only at the UI layer (e.g., hiding certain buttons) but not at the retrieval layer, a user could receive an answer that was generated using content they were never authorized to see, simply because it was retrieved and fed into the prompt. **Access control for RAG must be enforced at the retrieval/filtering layer, not just the presentation layer** — this is a frequently-missed, high-severity design mistake worth calling out proactively in an interview.

**Practical implementation:** every retrieval query should include the requesting user's permission scope as a mandatory filter (e.g., `visible_to: [user_id, user's_groups...]`) applied *before* or *during* the ANN search, not as an afterthought — treating it with the same rigor as the tenant-isolation filtering discussion above, since the failure mode (unauthorized content leaking into a generated answer) is arguably worse than a generic search bug, because it's silent and only surfaces in the LLM's fluent, confident-sounding output.

---

## Data Deletion & "Right to Be Forgotten" — A Practical Gotcha

**The problem:** Compliance regimes (GDPR and similar) can require that a specific user's data be **fully and verifiably deleted** on request — including from vector indexes, not just source document stores.

**Why this is harder than it sounds for vector indexes:**
- Simply "marking as deleted" in metadata isn't sufficient for compliance in many regimes — the actual vector, and often cached copies (replicas, backups, PQ-compressed shards, etc.), need to be genuinely removed.
- Some ANN index structures (notably graph-based ones like HNSW) don't support cheap, fully clean deletion — removing a node from a graph can require re-wiring its neighbors' connections, and naive "tombstone" deletion (marking deleted but leaving the node in the graph, filtering it out at query time) leaves the actual data present in the index, which may not satisfy a compliance deletion requirement.
- Replication and backups mean the same vector may physically exist in multiple places, all of which need the deletion to propagate.

**Practical mitigation patterns:** periodic index rebuilds that physically purge tombstoned/deleted vectors, per-tenant or per-user partitioning (which makes "delete this user" as simple as "drop this partition" — a strong argument for partitioning even outside pure performance reasons), and treating deletion SLAs as a first-class design requirement from the start rather than a bolt-on afterthought.

> **Why This Matters callout:** This is exactly the kind of "beyond the leaderboard metrics" depth that separates strong systems candidates — most people can explain HNSW; fewer people proactively raise that graph-based ANN indexes have real, nontrivial deletion semantics with compliance implications.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** Why can't metadata filtering on a vector index be treated the same as a `WHERE` clause on a relational database?

<details>
<summary>Show answer</summary>

A relational database's indexes on structured columns are cheap and composable with each other by design. An ANN vector index's graph (HNSW) or cluster (IVF) structure is built assuming a particular corpus is fully eligible for traversal — a filter that makes most of that corpus ineligible doesn't just narrow the output, it can break the assumptions the index's shortcuts rely on, forcing degraded (sometimes near-brute-force) search performance within the eligible subset unless the index itself is specifically designed to be filter-aware.
</details>

---

**Q2 (Easy — calculation).** A filter matches 5% of a corpus. If you retrieve the top-20 results by similarity and then post-filter, roughly how many results would you expect to survive on average, and what does this tell you about when post-filtering is safe to use?

<details>
<summary>Show answer</summary>

```
Expected survivors ≈ 20 × 0.05 = 1 result on average
```
This shows post-filtering is unsafe for highly selective filters — you'd need to retrieve a much larger candidate set (or switch to pre-filtering / a filter-aware index) to reliably get a full top-20 after filtering. Post-filtering is generally safe only when the filter has low selectivity (most of the corpus matches), where the expected survivor count stays close to the requested k.
</details>

---

**Q3 (Medium — conceptual).** Compare a shared vector index with per-tenant metadata filtering vs. fully separate per-tenant indexes. When would you choose each?

<details>
<summary>Show answer</summary>

Shared index + filter is cheaper and simpler to operate (pooled resources, one system to maintain), but carries meaningful risk: a filtering bug becomes a cross-tenant data leak, small tenants can suffer poor performance if the index isn't filter-aware, and all tenants share exposure to the noisy-neighbor problem. Fully separate per-tenant indexes give the strongest isolation by construction (no filtering logic to fail, predictable per-tenant performance, cleanest compliance story) but infrastructure/operational cost scales roughly linearly with tenant count. Choose shared+filter for many small, low-compliance-stakes tenants where cost efficiency matters most; choose per-tenant isolation for a smaller number of large, high-value, or compliance-sensitive tenants (e.g., healthcare, finance) where isolation cost is easily justified.
</details>

---

**Q4 (Medium — conceptual, security-focused).** A team implements user-level access control for a RAG system only in the front-end UI (hiding restricted documents from search result displays), while the retrieval layer itself queries the full corpus with no permission filter. What's the risk, and how should it actually be implemented?

<details>
<summary>Show answer</summary>

This is a serious security gap: even though the UI hides restricted documents from *display*, the retrieval layer can still retrieve unauthorized content and feed it into the generator's context — meaning the LLM's generated answer can be conditioned on, and potentially reveal, content the user was never authorized to see, entirely bypassing the UI-level restriction. This is worse than a typical search bug because it's silent and surfaces through fluent, confident-sounding generated text rather than an obvious visible document. Access control must be enforced at the retrieval/filtering layer itself — every query should apply the requesting user's permission scope as a mandatory filter during or before the ANN search, the same way tenant isolation filtering should be enforced, not layered on afterward at the presentation layer.
</details>

---

**Q5 (Hard — conceptual, compliance).** Why is deleting a specific user's data from a graph-based ANN index (like HNSW) harder than deleting a row from a relational database, and how would you design around this from the start?

<details>
<summary>Show answer</summary>

In a relational database, deleting a row is a simple, well-supported operation. In a graph-based ANN index like HNSW, a vector isn't an isolated record — it's a node wired into a graph via multiple edges to its neighbors, and removing it cleanly can require re-wiring those neighbors' connections to maintain graph quality. A common shortcut, "tombstoning" (marking as deleted and filtering it out at query time without actually removing it from the graph), leaves the actual vector data physically present in the index — which may not satisfy strict compliance deletion requirements (e.g., GDPR's right to be forgotten), and doesn't address copies in replicas or backups either. To design around this from the start: consider per-user or per-tenant partitioning (making "delete this user" as simple as "drop this partition"), plan for periodic index rebuilds that physically purge tombstoned vectors, and treat deletion SLAs as a first-class requirement in the initial architecture rather than an afterthought bolted on after a compliance request arrives.
</details>

---

**Q6 (Hard — system design synthesis).** Design the retrieval-and-filtering layer for a RAG system serving a B2B platform with ~200 enterprise customers, where each customer's documents must never be visible to another customer, latency must stay under 100ms p95, and customer sizes vary widely (some have 500 documents, some have 2 million). Walk through your reasoning.

<details>
<summary>Show answer</summary>

Given ~200 tenants with widely varying sizes and a hard isolation requirement (data must *never* cross tenants — a compliance-grade requirement, not just a nice-to-have), I'd lean toward **per-tenant namespaces/partitions within a shared infrastructure layer** rather than either a single fully-shared filtered index or 200 fully separate physical indexes: namespaces give strong logical isolation (no cross-tenant filtering logic that could fail) while still pooling underlying compute/storage more efficiently than 200 entirely separate deployments, which would be excessive at this tenant count. For the largest tenants (2 million documents), I'd evaluate whether their namespace alone needs its own dedicated resources/sharding to hit the 100ms p95 target, effectively treating very large tenants closer to "isolated index" while smaller tenants share pooled compute more freely — a tiered approach rather than one-size-fits-all. I'd also apply the mandatory-filter-at-retrieval-layer pattern within each namespace for any additional user-level permissions inside a given tenant, and design deletion/offboarding (a tenant churns) around dropping their namespace/partition cleanly, sidestepping the graph-deletion problem raised above.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Treating metadata filtering as architecturally free, like a relational `WHERE` clause — it interacts with (and can degrade) ANN index performance, especially at high selectivity.
- ❌ Using post-filtering on a highly selective filter and being surprised when far fewer than k results come back.
- ❌ Enforcing access control only at the UI/presentation layer instead of the retrieval layer — a silent, high-severity RAG-specific security gap.
- ❌ Assuming vector deletion is as simple as a relational row delete — graph-based ANN indexes have real, nontrivial deletion semantics with compliance implications.
- ❌ Picking a multi-tenancy isolation strategy based on "what's easiest to build" rather than actual tenant count, compliance requirements, and cost sensitivity.
- ❌ Ignoring the noisy-neighbor problem when justifying a shared-index multi-tenancy design.

---

# 📌 Cheat Sheet (Day 5)

**Filtering:** Post-filter only when selectivity is low (most of the corpus matches). Pre-filter or use a filter-aware ANN index when selectivity is high — naive pre-filtering on a non-filter-aware index can degrade toward slow scans within the filtered subset.

**Hybrid storage:** Native filtering in the vector DB (simplest, product-dependent ceiling) vs. vector DB + separate structured store with app-level join (more flexible, more latency/complexity) vs. pgvector (unified SQL + vectors, lower ceiling at extreme scale).

**Multi-tenancy:** Shared+filter (cheap, filter-bug = data leak, noisy neighbors) → shared infra + per-tenant namespace (good middle ground, native in most managed vector DBs) → fully separate per-tenant index (strongest isolation, cost scales linearly with tenant count). Choose based on tenant count, compliance stakes, and cost sensitivity — often a tiered mix for widely varying tenant sizes.

**Access control:** Must be enforced at the retrieval/filtering layer, not just the UI — otherwise unauthorized content can leak into generated answers silently.

**Deletion/compliance:** Graph-based ANN indexes (HNSW) don't support cheap clean deletion — tombstoning isn't true deletion. Design for compliance-grade deletion from the start (partitioning by user/tenant makes this dramatically simpler).

---

*End of Day 5 — Foundations week complete. Next up — Day 6: Review Day (cold Q&A across Days 1-5, no new content).*
