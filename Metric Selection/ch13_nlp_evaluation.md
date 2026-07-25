# Chapter 13: NLP Evaluation — BLEU, ROUGE, BERTScore, METEOR, and Human Eval

> *"Evaluating language is hard because language is not a function. The same meaning can be expressed in a thousand ways, and a thousand different meanings can be expressed with the same words. No metric captures this fully — and knowing which metric to distrust is as important as knowing which to trust."*

---

## 13.1 The NLP Evaluation Problem

Image classification has a ground truth: the image either is or isn't a cat. Regression has a ground truth: the house either sold for $350K or it didn't. NLP rarely has a single ground truth.

```
Source sentence (French): "Le chat est sur le tapis."

Valid translations:
  "The cat is on the mat."
  "The cat sits on the mat."
  "On the mat sits the cat."
  "There is a cat on the mat."
  "A cat is lying on the carpet."

All correct. All different. Most metrics penalize all but the first.
```

This is the **reference problem**: automatic metrics compare against one or a few reference outputs, but valid outputs form a vast space. Any metric that measures distance from a fixed reference underestimates model quality.

### The Evaluation Stack for NLP

```
Level 4: Task performance (downstream)    ← Most valid; slowest
Level 3: Human evaluation                 ← Ground truth; expensive
Level 2: Learned metrics (BERTScore)      ← Better than n-gram; not perfect
Level 1: N-gram metrics (BLEU, ROUGE)     ← Fast and cheap; often misleading
```

No level replaces the one above it. You use lower levels as proxies during development and validate with higher levels before deployment.

---

## 13.2 BLEU (Bilingual Evaluation Understudy)

The oldest and most widely used automatic MT metric. Introduced by Papineni et al. (2002). Still a standard benchmark despite well-documented flaws.

### Core Idea

BLEU measures how many n-grams in the hypothesis (model output) also appear in the reference (human translation).

### Modified N-gram Precision

For each n-gram order (1 to 4), compute **modified precision** — clip the count of each n-gram in the hypothesis by its maximum count in any reference:

```
Hypothesis: "the the the the the"
Reference:  "the cat is on the mat"

Unclipped precision: 5/5 = 1.0  ← wrong; this cheats by repeating "the"
Clipped count of "the": min(5, 2) = 2  ← max count in reference is 2
Modified precision: 2/5 = 0.4   ← correct
```

### BLEU Formula

```
BLEU = BP × exp(Σₙ wₙ × log pₙ)

Where:
  pₙ   = modified n-gram precision for order n
  wₙ   = weight per n-gram order (typically 1/4 each for n=1..4)
  BP   = brevity penalty (penalizes short hypotheses)

Brevity Penalty:
  BP = 1                         if c > r
  BP = exp(1 - r/c)              if c ≤ r

  c = hypothesis length
  r = reference length
```

### BLEU Worked Example

```
Hypothesis: "the cat sat on the mat"
Reference:  "the cat is on the mat"

Unigram matches: the(2), cat(1), on(1), the(duplicate), mat(1)
  Clipped: the(2), cat(1), on(1), mat(1) → 5 matches / 6 words = 0.833

Bigram matches: "the cat"✓, "on the"✓, "the mat"✓
  3/5 = 0.600

Trigram matches: "on the mat"✓
  1/4 = 0.250

4-gram matches: none
  0/3 = 0.000

BLEU-4 = BP × exp(0.25 × log(0.833) + 0.25 × log(0.600)
                 + 0.25 × log(0.250) + 0.25 × log(0.000001))
       ≈ 0  (log(0) → -∞; smoothing required)
```

**Standard practice:** Use smoothing (add-1 or Chen & Cherry smoothing) when n-gram counts are zero.

### BLEU Variants

| Variant | Description |
|---|---|
| BLEU-1 | Unigram precision only; measures word-level overlap |
| BLEU-4 | Standard: 1–4 gram weighted geometric mean |
| corpus-BLEU | Computed over full corpus; more stable than sentence BLEU |
| sentence-BLEU | Per-sentence; high variance; not recommended without smoothing |
| sacreBLEU | Standardized implementation; reproducible across papers |

**Always use sacreBLEU for reproducibility.** Raw BLEU scores vary with tokenization, which makes paper comparisons unreliable.

### BLEU's Well-Known Failures

**1. No semantic understanding**
```
Reference:  "The cat sat on the mat."
Hypothesis: "The feline rested on the rug."

Semantically identical. BLEU score: ~0.
```

**2. Recall-insensitive**
Standard BLEU only measures precision (n-grams in hypothesis that appear in reference). A hypothesis that omits half the content but uses precise words scores well.

**3. Fails for short hypotheses**
The brevity penalty helps but doesn't fully solve the problem.

**4. Poor correlation with human judgment at sentence level**
At corpus level, BLEU correlates moderately with human judgment. At sentence level, the correlation nearly vanishes.

**5. Not comparable across languages or domains**
A BLEU of 30 in Chinese MT and a BLEU of 30 in French MT do not mean the same thing.

---

## 13.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Developed for summarization (Lin, 2004). While BLEU emphasizes precision, ROUGE emphasizes **recall** — did the hypothesis cover the content of the reference?

### ROUGE-N

N-gram recall between hypothesis and reference:

```
ROUGE-N = (# n-gram matches) / (# n-grams in reference)
```

**ROUGE-1:** Unigram recall. Measures content word coverage.
**ROUGE-2:** Bigram recall. Measures phrase-level coverage.

```
Hypothesis: "the cat sat on the mat"
Reference:  "the cat is on the mat"

ROUGE-1 recall:
  Reference unigrams: {the, cat, is, on, the, mat} → 6 total
  Matches: the(2), cat(1), on(1), mat(1) → 5
  ROUGE-1 = 5/6 = 0.833

ROUGE-2 recall:
  Reference bigrams: {the cat, cat is, is on, on the, the mat} → 5
  Matches in hypothesis: {the cat, on the, the mat} → 3
  ROUGE-2 = 3/5 = 0.600
```

### ROUGE-L (Longest Common Subsequence)

Measures the longest common subsequence (LCS) between hypothesis and reference, allowing for non-contiguous matches:

```
ROUGE-L = (LCS length) / (reference length)

Hypothesis: "the cat sat on mat"
Reference:  "the cat is on the mat"

LCS: "the cat on mat" (length 4)  ← "is" and "the" skipped
ROUGE-L = 4/6 = 0.667
```

LCS captures sentence-level structure better than bigrams — it doesn't require exact phrase matches.

### ROUGE-W and ROUGE-S

**ROUGE-W (Weighted LCS):** Penalizes fragmented matches; rewards consecutive matches more.

**ROUGE-S (Skip-bigram):** Counts co-occurring word pairs with gaps allowed:
```
"the mat" matches in "the cat sat on the mat" even though not adjacent
```

### ROUGE in Practice

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(
    target="the cat is on the mat",
    prediction="the cat sat on the mat"
)
# Returns Precision, Recall, F1 for each variant
print(scores['rouge1'])   # Score(precision=0.833, recall=0.833, fmeasure=0.833)
print(scores['rouge2'])   # Score(precision=0.600, recall=0.600, fmeasure=0.600)
print(scores['rougeL'])   # Score(precision=0.667, recall=0.667, fmeasure=0.667)
```

### ROUGE Failures

Same semantic blindness as BLEU:
- Synonyms not rewarded
- Valid paraphrases penalized
- Order matters less (recall-focused) but coherence not measured
- Easy to game: long extracts from source score high ROUGE-1

---

## 13.4 METEOR (Metric for Evaluation of Translation with Explicit Ordering)

Designed to fix BLEU's recall insensitivity and synonym blindness.

### Key Innovations over BLEU

1. **Combines precision and recall** via F-mean (weighted, F₁₀ with recall weighted 9×)
2. **Synonym matching**: uses WordNet to match synonyms
3. **Stemming**: matches word stems (running = run)
4. **Paraphrase tables**: matches known paraphrases
5. **Chunk penalty**: penalizes fragmented matches

### Formula

```
METEOR = Fmean × (1 - Penalty)

Fmean = (10 × Precision × Recall) / (Recall + 9 × Precision)

Penalty = 0.5 × (# chunks / # matched unigrams)³
```

The chunk penalty: if all matched words appear in one contiguous chunk, penalty = 0. If matches are scattered (many short chunks), penalty increases.

### METEOR vs. BLEU

| Property | BLEU | METEOR |
|---|---|---|
| N-gram order | 1–4 grams | Unigrams (with alignment) |
| Recall | BP only | Explicit recall |
| Synonyms | No | Yes (WordNet) |
| Stemming | No | Yes |
| Chunk penalty | No | Yes |
| Correlation with human judgment | Moderate | Higher (especially sentence-level) |
| Language support | Any | English-best; multilingual tables exist |

METEOR correlates better with human judgment than BLEU in most studies, especially at the sentence level. However, it is slower and language-specific (synonym tables required).

---

## 13.5 BERTScore

The most significant leap in automatic NLP evaluation since BLEU. Uses contextual embeddings instead of n-gram overlap.

### Core Idea

Instead of counting matching words, compare meaning by computing cosine similarity between contextual BERT embeddings.

```
Reference tokens:   [the] [cat] [is]  [on]  [the] [mat]
Hypothesis tokens:  [the] [cat] [sat] [on]  [the] [mat]

For each hypothesis token, find the most similar reference token:
  "sat" ↔ "is"  → cosine similarity = 0.72  (semantically related)

BERTScore = f(precision similarities, recall similarities)
```

### BERTScore Components

**Precision (P):** For each hypothesis token, match to most similar reference token:
```
P = (1/|ŷ|) × Σᵢ max_j cos(ŷᵢ, yⱼ)
```

**Recall (R):** For each reference token, match to most similar hypothesis token:
```
R = (1/|y|) × Σⱼ max_i cos(yⱼ, ŷᵢ)
```

**F1:**
```
F1 = 2PR / (P + R)
```

### Why BERTScore Is Better

```
Reference:   "The cat is on the mat."
Hypothesis:  "The feline rests on the rug."

BLEU:       ≈ 0.10  (few n-gram matches)
BERTScore:  ≈ 0.92  (high semantic similarity via embeddings)
```

BERTScore captures:
- Synonyms ("cat" ↔ "feline", "is" ↔ "rests")
- Paraphrases ("on the mat" ↔ "on the rug")
- Meaning, not just surface form

### BERTScore Variants

The choice of base model matters:

| Model | Use case |
|---|---|
| `bert-base-uncased` | English general; fast |
| `roberta-large` | English; better correlation with human judgment |
| `xlm-roberta-large` | Multilingual |
| `microsoft/deberta-xlarge-mnli` | Highest correlation; slow |

```python
from bert_score import score

P, R, F1 = score(
    cands=["The feline rests on the rug."],
    refs=["The cat is on the mat."],
    lang="en",
    model_type="roberta-large"
)
print(f"BERTScore F1: {F1.mean():.4f}")
```

### BERTScore Limitations

- **Computationally expensive**: requires running BERT for every evaluation
- **Not interpretable**: score is a real number without clear units
- **Embedding space biases**: BERT embeddings encode world knowledge that may mismatch human judgment in some domains
- **Domain sensitivity**: A model fine-tuned on medical text may give different similarity scores than a general model
- **Still imperfect for diversity**: can reward repetition of reference phrasing over creative valid alternatives

---

## 13.6 Human Evaluation

The gold standard. No automatic metric replaces it; all automatic metrics are proxies for it.

### Human Evaluation Dimensions

Different tasks require different evaluation criteria:

| Task | Dimensions to Evaluate |
|---|---|
| Machine translation | Adequacy (meaning preserved?), Fluency (grammatically natural?) |
| Summarization | Faithfulness (factually correct?), Coverage (key info included?), Conciseness |
| Dialogue | Coherence, Relevance, Engagingness, Humanness |
| Story generation | Creativity, Coherence, Interestingness |
| Code generation | Correctness, Readability, Efficiency |

### Direct Assessment (DA)

Rate each output on a continuous scale (0–100):

```
Rate the following translation for adequacy (0–100):
Source:     "Le chat est sur le tapis."
Translation: "The cat is on the mat."

Rater score: 95
```

Standardize scores across raters (z-score normalization) to correct for rater scale differences.

### Comparative Evaluation (Pairwise)

Show raters two outputs; ask which is better:

```
System A: "The cat is on the mat."
System B: "The cat sat upon the carpet."

Which is a better translation? A / B / Tie
```

More reliable than absolute scoring because raters are better at relative judgments. Foundation of Chatbot Arena and RLHF preference collection.

### MOS (Mean Opinion Score)

Used in speech and dialogue: rate output quality on a 1–5 Likert scale. Average across raters = MOS.

```
MOS > 4.0  → Excellent
MOS 3–4    → Good
MOS 2–3    → Fair
MOS < 2    → Poor
```

### Inter-Annotator Agreement

Always measure agreement between raters. Low agreement means the evaluation task is poorly defined or subjective.

| Metric | Use |
|---|---|
| Cohen's κ | Two raters, categorical labels |
| Fleiss' κ | Multiple raters, categorical labels |
| Krippendorff's α | Multiple raters, any scale |
| Pearson/Spearman | Continuous scores |

**Guideline:** κ < 0.4 = poor agreement; 0.4–0.6 = moderate; > 0.6 = substantial.

If κ is low, before improving the model, improve the annotation guidelines.

### Human Evaluation Pitfalls

| Pitfall | Mitigation |
|---|---|
| Rater fatigue | Limit sessions to 30–60 min; randomize order |
| Rater bias toward verbosity | Blind raters to output length |
| Position bias | Randomize A/B order; counterbalance |
| Anchoring | Use practice examples; calibrate raters |
| Narrow rater pool | Diverse raters; domain experts when needed |
| Gaming by systems | Blind raters to which system produced which output |

---

## 13.7 Metric Correlation with Human Judgment

Which automatic metrics actually predict human preference? Meta-evaluation on standard benchmarks:

### WMT Findings (Translation)

| Metric | Segment-level ρ | System-level ρ |
|---|---|---|
| BLEU | 0.35 | 0.90 |
| METEOR | 0.45 | 0.92 |
| BERTScore | 0.58 | 0.96 |
| COMET | 0.72 | 0.98 |
| chrF | 0.52 | 0.94 |

**Key insight:** At the **system level** (comparing two models), BLEU is reasonably correlated with human judgment. At the **segment level** (evaluating a single sentence), BLEU is weakly correlated. This is why BLEU remains useful for tracking model progress but should not be used to evaluate individual outputs.

### COMET (Crosslingual Optimized Metric for Evaluation of Translation)

A learned metric trained to predict human DA scores. Consistently outperforms BLEU, METEOR, and BERTScore in correlation with human judgment.

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [{"src": "Le chat est sur le tapis.",
         "mt":  "The cat is on the mat.",
         "ref": "The cat is on the rug."}]

scores = model.predict(data, batch_size=8)
# Returns per-segment and corpus-level score
```

---

## 13.8 Task-Specific Metrics

### Code Generation: pass@k

```
pass@k = probability that at least one of k generated samples passes unit tests
```

```python
def pass_at_k(n, c, k):
    """
    n: total generated samples
    c: samples that pass tests
    k: samples considered (k ≤ n)
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)])

# pass@1: single attempt
# pass@10: best of 10 attempts
# pass@100: best of 100 attempts
```

Used in HumanEval, MBPP, and coding benchmarks. More informative than accuracy because it reflects the sample efficiency of the model.

### Question Answering: Exact Match and F1

```
Exact Match: 1 if normalized prediction exactly equals normalized answer
F1: token-level overlap between prediction and answer
```

Normalization: lowercase, remove punctuation and articles (a, an, the), strip whitespace.

**SQuAD F1:**
```
Precision = # shared tokens / # tokens in prediction
Recall    = # shared tokens / # tokens in answer
F1        = 2PR / (P + R)
```

### Factual Consistency: FactScore and QAEval

For summarization and generation, factual consistency is critical:

**FactScore:** Decomposes generated text into atomic claims; verifies each claim against a knowledge source.

**QAEval:** Generates questions from the reference; checks if the hypothesis answers them correctly.

---

## 13.9 Choosing the Right NLP Metric

```
Task: Machine Translation
  Development:    sacreBLEU (fast, reproducible)
  Paper results:  COMET + sacreBLEU (both)
  Final eval:     Human DA

Task: Summarization
  Development:    ROUGE-1, ROUGE-2, ROUGE-L
  Content check:  BERTScore
  Factual check:  FactScore or QAEval
  Final eval:     Human (faithfulness, coverage, conciseness)

Task: Dialogue / Chatbot
  Automatic:      Perplexity, BERTScore, BLEU (weak signal)
  Behavioral:     Task completion rate (goal-oriented)
  Final eval:     Human MOS / pairwise preference

Task: Code Generation
  Unit tests:     pass@k (primary)
  Style:          Code style linters
  Final eval:     Human code review

Task: Open-ended Generation
  Automatic:      BERTScore, MAUVE (distribution similarity)
  Final eval:     Human pairwise preference (required)
```

---

## 13.10 The Reference Problem Revisited

Every automatic metric discussed assumes you have a high-quality reference. But:

- References are expensive to collect
- References from one human reflect that human's style
- Multiple references help but don't solve the problem
- Reference-free metrics (BLEURT, COMET-QE) evaluate without references

**Reference-free evaluation** is an active research frontier. COMET-QE (Quality Estimation) scores translations using only source and hypothesis — no reference — and achieves strong correlation with human judgment.

For LLM evaluation, reference-free learned metrics and LLM-as-judge (Chapter 14) are becoming the primary approach.

---

## Summary

| Metric | Type | Measures | Best For | Main Weakness |
|---|---|---|---|---|
| BLEU | N-gram precision | Word overlap | MT benchmarks, fast dev | No semantics; sentence-level fails |
| ROUGE | N-gram recall | Content coverage | Summarization | No semantics; gameable |
| METEOR | N-gram + alignment | Precision + recall + synonyms | MT with synonym matching | Language-specific resources |
| BERTScore | Embedding similarity | Semantic similarity | Any text generation | Expensive; not perfectly calibrated |
| COMET | Learned | Human DA prediction | MT production eval | Requires reference (DA) or source (QE) |
| Human DA | Human | Quality ground truth | Any; final validation | Expensive, slow |
| pass@k | Unit tests | Code correctness | Code generation | Requires test suite |

---

## Further Reading

- Papineni et al. — *BLEU: A Method for Automatic Evaluation of MT* (ACL 2002) — original BLEU
- Lin — *ROUGE: A Package for Automatic Evaluation of Summaries* (ACL 2004)
- Zhang et al. — *BERTScore: Evaluating Text Generation with BERT* (ICLR 2020)
- Rei et al. — *COMET: A Neural Framework for MT Evaluation* (EMNLP 2020)
- Kocmi et al. — *To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for MT* (WMT 2021)
- Chen et al. — *Evaluating Large Language Models Trained on Code* (HumanEval / pass@k, 2021)

---

*Next: Chapter 14 — Generative & LLM Evaluation*
