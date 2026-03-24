# 🔥 DAY 25 — Domain-Specific RAG

**Legal • Medical • Finance • Code**

This is where RAG becomes *real-world critical infrastructure*.

Generic RAG ≠ Production RAG.

Each domain changes:

* Retrieval strategy
* Chunking strategy
* Evaluation metrics
* Risk tolerance
* Architecture

We’ll go domain by domain with deep system-level thinking.

---

# 1️⃣ Legal RAG

## Why Legal Is Hard

Legal documents are:

* Long (50–500 pages)
* Hierarchical (Sections → Subsections → Clauses)
* Citation-dependent
* Precedent-driven
* High liability

---

## 📚 Example Legal Sources

![Image](https://substackcdn.com/image/fetch/%24s_%21tamh%21%2Cw_1200%2Ch_675%2Cc_fill%2Cf_jpg%2Cq_auto%3Agood%2Cfl_progressive%3Asteep%2Cg_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F423a47f8-adcf-46b4-be38-13077d0e8678_850x590.jpeg)

![Image](https://images.template.net/518695/One-Page-Legal-Contract-Format-Template-edit-online.png)

![Image](https://p2.piqsels.com/preview/33/483/771/law-book-page.jpg)

![Image](https://open.umn.edu/rails/active_storage/representations/redirect/eyJfcmFpbHMiOnsiZGF0YSI6ODIsInB1ciI6ImJsb2JfaWQifX0%3D--43dc105fd8eac304344f0d306100a96b05ae0591/eyJfcmFpbHMiOnsiZGF0YSI6eyJmb3JtYXQiOiJ3ZWJwIiwicmVzaXplIjoiNDkweDEwMDAifSwicHVyIjoidmFyaWF0aW9uIn19--421ab82856ce4ced5bdaa37f96bcb9060315a4c7/0000BasicLega.png?disposition=inline)

---

## Core Design Changes

### 1. Hierarchical Chunking (Critical)

Instead of naive fixed tokens:

```
Section 4
  → 4.1
     → 4.1(a)
```

Store metadata:

```json
{
  "section": "4.1(a)",
  "case": "Smith v. Jones",
  "jurisdiction": "CA",
  "year": 2019
}
```

Retrieval becomes:

[
P(doc \mid query, jurisdiction, year)
]

---

### 2. Citation Preservation

Legal users require:

* Direct quotes
* Exact citations
* No paraphrased hallucinations

Prompt constraint:

> Answer ONLY using retrieved passages. Include citation IDs.

---

### 3. Risk Profile

| Failure Type      | Severity     |
| ----------------- | ------------ |
| Missing citation  | Medium       |
| Wrong statute     | Critical     |
| Hallucinated case | Catastrophic |

So:

* Strict refusal policies
* Groundedness scoring mandatory

---

## Evaluation Focus

* Citation correctness
* Faithfulness
* Retrieval recall@k
* Jurisdiction filtering accuracy

---

# 2️⃣ Medical RAG

## Why Medical Is Hard

* Life-critical answers
* Rapid knowledge updates
* Evidence hierarchy (RCT > meta-analysis > blog)

---

## 📚 Example Medical Sources

![Image](https://www.nlm.nih.gov/bsd/policy/graphics/structured_abs_fig1a.gif)

![Image](https://m.media-amazon.com/images/I/81xrSH92RWL._AC_UF1000%2C1000_QL80_.jpg)

![Image](https://imgv2-2-f.scribdassets.com/img/document/506184974/original/e0524484cf/1?v=1)

![Image](https://nhsrcindia.org/sites/default/files/stg.jpg)

---

## Core Design Changes

### 1. Evidence Weighting

Not all documents equal.

Add metadata:

```json
{
  "evidence_level": "RCT",
  "year": 2023,
  "peer_reviewed": true
}
```

Ranking:

[
Score = similarity + \lambda_1 evidence_weight + \lambda_2 recency
]

---

### 2. Strict Refusal

If retrieval confidence < threshold:

Refuse.

Medical systems must prefer abstention over hallucination.

---

### 3. Differential Diagnosis Pattern

Often multiple possibilities.

RAG should retrieve multiple sources and generate:

* Likely causes
* Supporting evidence
* Contraindications

---

## Risk Profile

| Failure            | Severity     |
| ------------------ | ------------ |
| Mild dosage error  | High         |
| Fabricated trial   | Catastrophic |
| Outdated guideline | Critical     |

---

## Evaluation

* Groundedness
* Calibration
* Abstention accuracy
* Temporal validity

---

# 3️⃣ Finance RAG

## Why Finance Is Unique

* Numbers matter
* Real-time data matters
* Regulatory compliance
* Multi-document synthesis

---

## 📊 Example Finance Sources

![Image](https://www.investopedia.com/thmb/7VhbIdLJZqkNiUzYAjeVQRrjjDQ%3D/1500x0/filters%3Ano_upscale%28%29%3Amax_bytes%28150000%29%3Astrip_icc%28%29/Microsoft202310-KFinancialStatements-2401d7d8a55e4d5c9c2a3283fe65fdf2.png)

![Image](https://www.investopedia.com/thmb/hZ3xeJmbLmUBZtzunJrrMCmT5QI%3D/1500x0/filters%3Ano_upscale%28%29%3Amax_bytes%28150000%29%3Astrip_icc%28%29/s1-5bfd932246e0fb0051865e0f)

![Image](https://s3.amazonaws.com/thumbnails.venngage.com/template/ff356a68-3ca0-4696-a175-0e5b3354d9b3.png)

![Image](https://cdn.corporatefinanceinstitute.com/assets/equity-research-report-cover.png)

---

## Core Design Changes

### 1. Structured + Unstructured Hybrid

Finance needs:

* RAG (reports, filings)
* Tool-use (live prices, calculations)

Never let LLM compute NAV manually.

---

### 2. Numeric Verification

After generation:

* Extract numbers
* Cross-check with source
* Recompute with tool

Answer verification loop.

---

### 3. Compliance Constraints

Must avoid:

* Unauthorized advice
* Regulatory violations

Prompt must embed compliance policy.

---

## Risk Profile

| Failure               | Severity |
| --------------------- | -------- |
| Minor estimate error  | Medium   |
| Wrong earnings figure | High     |
| Regulatory violation  | Severe   |

---

## Evaluation

* Numeric faithfulness
* Calculation accuracy
* Compliance adherence

---

# 4️⃣ Code RAG

Most exciting domain.

---

## 🖥 Example Code Sources

![Image](https://www.researchgate.net/publication/327191473/figure/fig13/AS%3A961445533585418%401606237886125/A-schematic-overview-of-the-file-structure-of-GitHub-left-and-the-representation-of.png)

![Image](https://miro.medium.com/1%2A45obi6lIPEGUefPLw3Pk-g.png)

![Image](https://cdn.prod.website-files.com/6320e912264435aca2ab0351/644ae20b7ea56560ab87af68_stripe-docs-1.jpg)

![Image](https://s3.us-west-1.wasabisys.com/idbwmedia.com/images/api/sample_api_diagram.png)

---

## Why Code Is Different

* Syntax-sensitive
* Execution-verifiable
* Graph structured (call graph, dependency tree)

---

## Core Design Changes

### 1. Chunk by AST, Not Tokens

Instead of 512-token chunks:

Chunk by:

* Function
* Class
* Module

Preserve imports + dependencies.

---

### 2. Dependency-Aware Retrieval

If function A calls B:

Retrieval should pull both.

Graph-aware retrievers outperform vanilla vector search.

---

### 3. Compile/Test Loop

After generation:

* Run unit tests
* Static analysis
* Lint

Feedback → regenerate.

---

## Risk Profile

| Failure                | Severity |
| ---------------------- | -------- |
| Minor syntax error     | Low      |
| Silent logic bug       | High     |
| Security vulnerability | Critical |

---

# Cross-Domain Differences

| Dimension                | Legal     | Medical   | Finance   | Code      |
| ------------------------ | --------- | --------- | --------- | --------- |
| Recency sensitivity      | Medium    | High      | Very High | Medium    |
| Citation importance      | Very High | High      | Medium    | Low       |
| Deterministic validation | Low       | Medium    | High      | Very High |
| Refusal criticality      | High      | Very High | High      | Medium    |

---

# Architecture Evolution by Domain

Generic RAG:

```
Embed → Retrieve → Generate
```

Legal RAG:

```
Metadata filter → Hierarchical retrieve → Quote-constrained generation
```

Medical RAG:

```
Evidence-weighted retrieval → Confidence threshold → Refusal gate
```

Finance RAG:

```
Hybrid retrieval → Tool verification → Compliance filter
```

Code RAG:

```
Graph retrieval → Generate → Compile → Test → Retry
```

---

# Common Interview Question

### Q: Why can’t we use the same RAG for all domains?

Because:

* Risk tolerance differs.
* Retrieval signals differ.
* Validation mechanisms differ.
* Evaluation metrics differ.

Domain dictates architecture.

---

# Advanced Insight

Domain-specific RAG systems eventually become:

> Retrieval + policy engine + verification loop + domain-aware ranking.

Pure semantic similarity is never enough.

---

# Mastery Check

You should now understand:

* How RAG changes per industry.
* Why evaluation differs.
* Why refusal policies vary.
* Why numeric and execution verification matter.

---

