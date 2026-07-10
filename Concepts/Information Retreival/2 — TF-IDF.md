# Chapter 2 — TF-IDF

## What is it?

TF-IDF (Term Frequency — Inverse Document Frequency) is a scoring function that answers the question: **how important is a word to a specific document, relative to the whole corpus?**

It is the first real answer to `score(q, d)` from Chapter 1. Instead of Boolean match/no-match, every document gets a number. The higher the number, the more relevant the document to the query.

TF-IDF is the foundation of almost every classical IR system built between the 1970s and 2010s. Even today it's a strong baseline and appears constantly in FAANG interviews as the "explain from scratch" starting point.

---

## The intuition

Two observations that any reasonable person would make:

**Observation 1 — frequency matters.** A document that mentions "neural network" 20 times is probably more about neural networks than one that mentions it once. So words that appear more in a document should score higher. That's **TF**.

**Observation 2 — rarity matters.** The word "the" appears in every document. It tells you nothing about what the document is about. The word "backpropagation" appears in very few documents — if your query contains it, matching docs are probably very relevant. Rare words should be worth more. That's **IDF**.

**TF-IDF = multiply both.** A word that appears often in this document AND rarely across the corpus is the most discriminative signal you have.

---

## The equations

### Term Frequency

```
TF(t, d) = count(t in d) / |d|
```

Where:
- `t` = the term (word)
- `d` = the document
- `count(t in d)` = how many times t appears in d
- `|d|` = total number of words in d (document length)

**Why divide by document length?** Without normalization, longer documents score higher just by being longer. A 10,000-word book mentioning "cat" 5 times is less "cat-focused" than a 50-word abstract mentioning it 3 times. Dividing by length fixes this.

**What happens at the extremes?**
- `TF = 0` → term not in document → contributes nothing to score
- `TF → 1` → term is nearly every word in the document (unusual)
- Typical values are small decimals: 0.01, 0.05, 0.12

---

### Inverse Document Frequency

```
IDF(t) = log(N / df(t))
```

Where:
- `N` = total number of documents in the corpus
- `df(t)` = number of documents containing term t
- `log` = natural log or log base 10 (convention varies, doesn't matter as long as consistent)

**Why log?** Without it, IDF grows too aggressively. A term in 1 out of 1,000,000 docs would get a raw ratio of 1,000,000 — completely dominating TF. The log compresses this into a reasonable range.

**What happens at the extremes?**
- `df(t) = N` → term appears in every doc → `log(N/N) = log(1) = 0` → IDF = 0 → word is useless (e.g. "the")
- `df(t) = 1` → term appears in only one doc → `log(N/1) = log(N)` → maximum IDF → very discriminative
- `df(t) = sqrt(N)` → term appears in ~0.1% of docs → `log(sqrt(N)) = 0.5 * log(N)` → moderate signal

---

### TF-IDF score

```
TFIDF(t, d) = TF(t, d) × IDF(t)
```

For a multi-term query, sum across all query terms:

```
score(q, d) = Σ TFIDF(t, d)   for each t in q
```

---

## Worked numeric example

```
corpus (3 documents, each ~10 words):
  D1 = "the cat sat on the mat the cat"       → |D1| = 8
  D2 = "the cat wore a hat to the party"      → |D2| = 8
  D3 = "the mat and the hat sat on the floor" → |D3| = 9

N = 3

query: "cat mat"
```

### Step 1 — compute TF for each term in each doc

```
TF("cat", D1) = 2/8 = 0.250
TF("cat", D2) = 1/8 = 0.125
TF("cat", D3) = 0/9 = 0.000

TF("mat", D1) = 1/8 = 0.125
TF("mat", D2) = 0/8 = 0.000
TF("mat", D3) = 1/9 = 0.111
```

### Step 2 — compute IDF for each query term

```
df("cat") = 2  (appears in D1, D2)
df("mat") = 2  (appears in D1, D3)

IDF("cat") = log(3/2) = log(1.5) = 0.405
IDF("mat") = log(3/2) = log(1.5) = 0.405
```

Both terms have the same IDF here because they appear in the same number of docs. In a real corpus they'd differ.

### Step 3 — compute TF-IDF per term per doc

```
TFIDF("cat", D1) = 0.250 × 0.405 = 0.101
TFIDF("cat", D2) = 0.125 × 0.405 = 0.051
TFIDF("cat", D3) = 0.000 × 0.405 = 0.000

TFIDF("mat", D1) = 0.125 × 0.405 = 0.051
TFIDF("mat", D2) = 0.000 × 0.405 = 0.000
TFIDF("mat", D3) = 0.111 × 0.405 = 0.045
```

### Step 4 — sum across query terms to get final score

```
score(q, D1) = 0.101 + 0.051 = 0.152  ← ranked 1st
score(q, D2) = 0.051 + 0.000 = 0.051  ← ranked 2nd
score(q, D3) = 0.000 + 0.045 = 0.045  ← ranked 3rd
```

D1 wins because it's the only document containing both query terms, and "cat" appears twice in it.

---

## Why it works / why it fails

**Why it works:**
- Simple, fast, and interpretable — you can explain every number
- Penalizes common words automatically (IDF kills "the", "is", "and")
- Length normalization makes short and long docs comparable
- Strong baseline — often 80% as good as much more complex methods on keyword-heavy queries

**Why it fails:**
- **No TF saturation.** If "cat" appears 100 times vs 1 time, the score is 100× higher. Does a document really become 100× more relevant? No — there's a point of diminishing returns. BM25 (Chapter 3) fixes this.
- **Vocabulary mismatch.** "Car" and "automobile" are completely different terms. TF-IDF has zero way to know they're related. Dense retrieval (Chapter 4) fixes this.
- **No word order.** "Dog bites man" and "man bites dog" are identical under TF-IDF. Both are treated as the bag of words {dog, bite, man}.
- **IDF is a corpus-level statistic.** If your corpus changes significantly, IDF values go stale and the whole index needs rebuilding.

---

## The one thing to remember

TF-IDF rewards words that appear **often in this document** but **rarely in the corpus** — the two properties that make a word a good discriminator for that document.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `TF(t,d) = count(t,d) / \|d\|` | How often term t appears in doc d, normalized by doc length |
| `IDF(t) = log(N / df(t))` | How rare term t is across the corpus. Rare = high IDF |
| `TFIDF(t,d) = TF(t,d) × IDF(t)` | Combined importance of term t for document d |
| `score(q,d) = Σ TFIDF(t,d)` | Final document score = sum of TF-IDF over all query terms |

---

## Interview Q&A

**Q1. Why do we take the log in IDF instead of using the raw ratio N/df(t)?**

Without log, the IDF values explode. A term appearing in 1 out of 1,000,000 documents gets a raw ratio of 1,000,000. Multiplied by TF, this term would dominate the score completely, drowning out all other query terms. The log compresses the range — log(1,000,000) = 13.8 in natural log — making all IDF values comparable and keeping the score well-behaved. It also has a nice property: a term in every document gets IDF = log(1) = 0, which is exactly what we want.

**Q2. What happens to the score of the word "the" under TF-IDF?**

"The" appears in almost every document, so `df("the") ≈ N`. Therefore `IDF("the") = log(N/N) = log(1) = 0`. No matter how high TF("the") is, multiplying by 0 gives a score of 0. "The" contributes nothing — which is exactly correct. This is TF-IDF's elegant built-in stopword handling. You don't even need an explicit stopword list if your corpus is large enough.

**Q3. Why does TF-IDF fail when a term appears 100 times vs once?**

TF grows linearly with count — 100 occurrences gives 100× the TF of 1 occurrence. But intuitively, a document mentioning "cat" 100 times isn't 100× more relevant than one mentioning it once. There are sharply diminishing returns after the first few occurrences. This unbounded linear growth is called the TF saturation problem and is the primary motivation for BM25 (Chapter 3), which caps TF growth asymptotically.

**Q4. Can TF-IDF handle synonyms? Why or why not?**

No. TF-IDF operates on exact term matching. "Car" and "automobile" are different strings and map to different entries in the inverted index. A query for "car" will not retrieve a document about "automobiles" even if they're identical in meaning. Fixing this requires either query expansion (manually add synonyms to the query) or dense retrieval (Chapter 4), where both words map to nearby vectors in embedding space regardless of surface form.

**Q5. If the corpus doubles in size overnight, what happens to your TF-IDF scores?**

TF scores are unaffected — they're computed per document. But IDF scores change because N doubles. `IDF(t) = log(2N / df(t)) = log(N/df(t)) + log(2)` — every IDF value increases by a constant `log(2) ≈ 0.69`. The relative ranking between documents for the same query is mostly preserved, but absolute scores shift. If df(t) also changes (because the new docs contain term t), rankings can shift too. In production systems this means IDF needs periodic recomputation — another reason Elasticsearch segment merges matter.

