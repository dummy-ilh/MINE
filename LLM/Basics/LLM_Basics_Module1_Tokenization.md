# Module 1 — Tokenization (Master Notes, Expanded)

## 0. Why tokenization exists — the actual tradeoff

A neural net's input layer is a fixed-size lookup table (embedding matrix). You must decide: what is one "unit" of text?

| Granularity | Vocab size | Sequence length | Problem |
|---|---|---|---|
| Character-level | ~100-1000 | Very long (4-6x more tokens than words) | Attention cost is O(n²) — sequences this long make training/inference painfully slow, and the model must relearn "cat" = 3 characters every time instead of treating it as one concept |
| Word-level | Millions (every inflection, typo, name) | Short | OOV problem: any word not seen during vocab-building (typos, new slang, rare names) becomes an `<UNK>` token — the model loses all information about it |
| **Subword-level** | 30k–128k (tunable) | Moderate | The actual sweet spot — common words stay whole, rare/unseen words decompose into meaningful, previously-seen pieces |

**Practical example**: the word "unhappiness" — 
- Character-level: `u-n-h-a-p-p-i-n-e-s-s` (11 tokens, no shared meaning learned)
- Word-level: if "unhappiness" wasn't in training vocab → `<UNK>` (all information lost)
- Subword-level (typical BPE output): `un` + `happi` + `ness` (3 tokens) — the model can reuse what it learned about "un-" (negation prefix, seen in "unhappy", "unable", "undo") and "-ness" (noun suffix, seen in "happiness", "kindness") even if it never saw "unhappiness" as a whole word during training. This is the entire point: **subword tokenization lets the model compose meaning from parts it has seen elsewhere.**

**Interview one-liner**: "Subword tokenization is a compression + generalization trick — it shrinks vocab size to avoid OOV, while decomposing rare words into reusable morphological pieces, at a smaller sequence-length cost than characters."

---

## 1. Byte-Pair Encoding (BPE)

### Core idea, in plain words
BPE is dead simple: **look at the whole training corpus, find the two symbols that sit next to each other most often, glue them into one new symbol, and repeat.** No probability theory — just "what pair shows up together the most, over and over."

### Algorithm (training time)
1. Split corpus into words; represent each word as a sequence of characters + an end-of-word marker (so the model knows where a word ends, e.g. `low` → `l o w </w>`).
2. Count frequency of every adjacent symbol pair across the *entire* corpus.
3. Merge the single most frequent pair everywhere it occurs → this becomes one new symbol/token.
4. Repeat steps 2–3 for `k` iterations. `k` is a hyperparameter you choose upfront — it directly controls final vocab size.

### Full worked numerical example (carried through to the end)
Corpus (word: frequency count in the corpus):
```
low     : 5
lower   : 2
newest  : 6
widest  : 3
```
Initial representation (characters + end-of-word marker `</w>`):
```
l o w </w>          x5
l o w e r </w>       x2
n e w e s t </w>     x6
w i d e s t </w>     x3
```

**Merge 1**: count all adjacent pairs.
- (e,s): from "newest"(6) + "widest"(3) = 9 ← winner
- (s,t): 6+3 = 9 (tie — pick (e,s) first by convention)
- (l,o): 5+2 = 7
- (o,w): 5+2 = 7
- (w,e): 2 (from "lower") + 6 (from "newest") = 8
→ Merge (e,s) → **"es"**. Vocab so far: base chars + {"es"}.

**Merge 2**: recount with "es" as a unit.
```
l o w </w>            x5
l o w e r </w>          x2
n e w es t </w>       x6
w i d es t </w>       x3
```
- (es,t): 6+3=9 ← winner
→ Merge → **"est"**.

**Merge 3**:
```
n e w est </w>   x6
w i d est </w>   x3
l o w </w>       x5
l o w e r </w>     x2
```
- (l,o): 5+2=7 ← winner
→ Merge → **"lo"**.

**Merge 4**:
```
lo w </w>       x5
lo w e r </w>     x2
```
- (lo,w): 5+2=7 ← winner
→ Merge → **"low"**.

**Result after 4 merges**: "low" is now a single atomic token (matches human intuition — it's a common whole word). "newest" and "widest" have been compressed to `n e w est </w>` and `w i d est </w>` — the model gets a reusable "est" superlative-suffix token, exactly like a linguist would segment it.

**The stored artifact**: the ordered list of merges — `(e,s)→es`, `(es,t)→est`, `(l,o)→lo`, `(lo,w)→low` — *is* the trained tokenizer. At inference time on brand-new text, you apply these exact merges in this exact order to segment the new word.

### Byte-level BPE (this is what GPT-2/3/4 actually use)
Instead of starting from Unicode *characters*, start from raw **bytes** (256 possible values, 0-255). Why this matters practically:
- A Unicode character-based vocab can still hit unseen characters (obscure scripts, emoji combos, corrupted text) → still needs an `<UNK>` fallback.
- A byte-level vocab is **mathematically complete** — every possible string, in every language, including binary garbage, is representable, because everything is bytes underneath. Zero `<UNK>` tokens, ever.
- Cost: non-Latin scripts (Chinese, Korean, Hindi) use multi-byte UTF-8 encodings per character, so those languages end up needing *more* tokens per character than English does. This is a real, measured inefficiency in GPT-family models for non-English text.

**Numbers to know**: GPT-2 vocab = 50,257 (256 base bytes + 50,000 merges + 1 special `<|endoftext|>` token).

### Where BPE is used standalone in practice
- **GPT-2, GPT-3, GPT-4** (byte-level BPE) — OpenAI's `tiktoken` library implements this.
- **RoBERTa** — byte-level BPE, same family as GPT-2's.
- Standard, off-the-shelf BPE (character-level, not byte-level) is what the original `subword-nmt` library did for machine translation before this became standard in LLMs.

---

## 2. WordPiece (used in BERT)

### Core idea, in plain words
Same merge-loop skeleton as BPE, but instead of asking "which pair appears together most often in raw counts," it asks **"which pair appears together *far more than you'd statistically expect* given how common each symbol individually is?"**

### The formula, explained term by term
```
score(a, b) = freq(a, b) / (freq(a) × freq(b))
```
- `freq(a, b)` = how often `a` and `b` sit next to each other.
- `freq(a) × freq(b)` = what you'd expect if `a` and `b` were totally independent/unrelated (this is just basic probability: P(A and B) = P(A)×P(B) if independent).
- So the **ratio** tells you: are `a` and `b` occurring together *more than chance* would predict? This is literally **pointwise mutual information (PMI)** in disguise — a concept from information theory measuring statistical association between two events.

### Why this formula matters — practical example
Imagine "t" and "h" are both extremely common individually (both appear in tons of words: "the", "hat", "top", "this"...). Their raw co-occurrence count (as in "th") will be huge just because *both* are everywhere — not necessarily because they're "linguistically glued." BPE would merge them early purely on volume.

WordPiece's denominator `freq(a)×freq(b)` **punishes exactly this case** — dividing by two large numbers shrinks the score back down. Meanwhile a pair like "qu" (where "q" is rare and almost *always* followed by "u" in English) gets a much higher WordPiece score even though its raw count is small, because the *conditional* relationship is nearly deterministic.

### Numerical example (contrast with BPE side-by-side)
Say in a corpus:
- freq("t") = 5000, freq("h") = 4500, freq("t","h" adjacent) = 800
- freq("q") = 25, freq("u") = 3000, freq("q","u" adjacent) = 24 (q is almost always followed by u)

**BPE score (raw count)**: (t,h)=800 vs (q,u)=24 → BPE merges (t,h) first, by a landslide.

**WordPiece score**:
- score(t,h) = 800 / (5000 × 4500) = 800 / 22,500,000 ≈ **0.0000356**
- score(q,u) = 24 / (25 × 3000) = 24 / 75,000 = **0.00032**

WordPiece score for (q,u) is ~9x higher than (t,h), even though its raw count is 33x smaller — because (q,u) is a near-perfect statistical bond, while (t,h) is common mostly because both letters are common on their own. **This is the exact sentence to say in an interview**: "WordPiece normalizes for base-rate frequency, so it favors tightly-bonded pairs over merely co-frequent ones — BPE has no such correction."

### Practical detail: the `##` continuation marker
BERT's WordPiece tokenizer prefixes non-word-initial subword pieces with `##`. Example: "playing" → `play` + `##ing`. This tells the model (and lets you reconstruct text) that `##ing` attaches directly to the previous token with no space, vs a token like `ing` on its own which would mean a new word starting with "ing".

### Where WordPiece is used standalone in practice
- **BERT** (original Google model) and virtually all its direct derivatives: **DistilBERT, ELECTRA, MobileBERT**.
- Google's production NMT system (the original 2016 paper that introduced WordPiece was for Google Translate, before BERT reused it).

---

## 3. Unigram Language Model tokenizer (via SentencePiece)

### Core idea, in plain words
Completely different philosophy from BPE/WordPiece: instead of building the vocab bottom-up by merging, **start with a huge candidate vocabulary and prune it down**, using a probabilistic model of "how likely is this segmentation of the sentence."

Also — and this is the detail people forget — **SentencePiece is not itself an algorithm; it's a language-agnostic *framework/library*** that can run either the BPE algorithm or the Unigram LM algorithm underneath, while treating raw text (including spaces, via a special `▁` symbol) as the input, so it never needs language-specific pre-tokenization (critical for Japanese/Chinese/Thai which have no whitespace between words).

### Algorithm (training time), explained
1. Build a huge candidate vocab (e.g., all frequent substrings up to some length).
2. Assign each candidate token a probability `P(t)`.
3. For any sentence, its probability under one particular way of splitting it into tokens `t1, t2, ..., tn` is:
   ```
   P(sentence) = P(t1) × P(t2) × ... × P(tn)
   ```
   (Plain words: multiply the probabilities of each piece you chose — this is a unigram model, meaning it assumes each token's probability is independent of its neighbors, which is a simplification but works well enough.)
4. Use the EM algorithm to fit `P(t)` values that maximize total corpus likelihood.
5. For each token, measure: "if I deleted this token from the vocab entirely, how much would total corpus likelihood drop?" This is its "loss contribution."
6. Remove the lowest-contributing ~10-20% of tokens, refit probabilities, repeat until you hit the target vocab size.

### Practical example: why multiple segmentations matter
Take the word "unwanted". There might be several valid ways to tokenize it under the trained vocab:
- `un` + `want` + `ed` (probability = P(un)×P(want)×P(ed))
- `un` + `wanted` (probability = P(un)×P(wanted))
- `unwanted` (if it happens to be in vocab) (probability = P(unwanted))

At inference, the **Viterbi algorithm** finds the single highest-probability segmentation — not necessarily the "greedy longest match" a human might guess. This is a real, useful property: **during training**, you can deliberately *sample* a lower-probability (but still valid) segmentation instead of always the best one — this is called **subword regularization**, and it acts like data augmentation, making the model robust to alternate ways the same word could be split, which improves generalization especially in low-resource settings.

### Where Unigram LM / SentencePiece is used standalone in practice
- **T5, ALBERT, XLNet** use the Unigram LM algorithm via SentencePiece.
- **Llama, Llama 2** use SentencePiece as the *framework*, but with the **BPE algorithm** running inside it (not Unigram LM) — this is a common point of confusion, worth stating explicitly in interviews: "SentencePiece is the wrapper/framework; BPE and Unigram LM are two different algorithms you can run inside it."

---

## 4. Side-by-side summary table (memorize this cold)

| | BPE | WordPiece | Unigram LM (via SentencePiece) |
|---|---|---|---|
| Direction | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) |
| Merge/keep criterion | Raw pair frequency | freq(a,b)/(freq(a)×freq(b)) — PMI-like | Likelihood contribution to corpus |
| Math flavor | Counting | Statistical association | Probabilistic (EM + Viterbi) |
| Deterministic segmentation? | Yes, always same result | Yes, always same result | No — supports multiple valid segmentations (regularization) |
| Real models using it standalone | GPT-2/3/4, RoBERTa | BERT, DistilBERT, ELECTRA | T5, ALBERT, XLNet |
| Framework note | Can run inside SentencePiece too | Standalone algorithm | Runs inside SentencePiece; Llama uses SentencePiece-framework+BPE-algorithm, NOT Unigram |

---

## 5. Practical issues (very common interview follow-ups)

**Multilingual tokenization cost**: if a BPE vocab is trained mostly on English text, non-Latin scripts get chopped into more tokens per character (since those characters were rare in training, they don't get merged into big compact tokens). Practical effect: a Hindi or Chinese sentence that means the same thing as an English one can cost 2-4x more tokens → shorter effective context window and higher inference cost for those languages, purely from tokenizer bias, not model capability. Fix: train tokenizer on a balanced multilingual corpus (this is exactly what multilingual models like BLOOM, mT5 do).

**Digit/number handling**: naive BPE tokenizes numbers inconsistently based on frequency — "2024" might be a single token (common year), but "20247" might split unpredictably as "202"+"47" or "2024"+"7" depending on what merges happened to be frequent in training. This inconsistency actively hurts arithmetic ability, since the model can't learn a stable "one digit = one token" pattern. **Fix used by Llama and GPT-4-class tokenizers**: force every digit to be its own separate token, always — this single design choice measurably improves arithmetic performance.

**Tokenizer vocab-size tradeoff, concretely**: 
- Smaller vocab (e.g. 30k, BERT) → longer sequences, smaller embedding matrix.
- Larger vocab (e.g. 128k, Llama-3) → shorter sequences (less attention compute, since attention scales O(n²)), but bigger embedding + output softmax layer (more parameters, more memory).
- Modern large models trend toward bigger vocabs because, at scale, the O(n²) attention savings from shorter sequences outweighs the linear cost of a bigger embedding table.

**OOV handling, resolved**: with byte-level BPE or SentencePiece, there is no true OOV state — worst case, a completely novel string decomposes all the way down to individual bytes/characters, which is *always* in vocab by construction. This is why word-level tokenizers with `<UNK>` tokens are essentially extinct in modern LLMs.

---

## 6. Quick-fire Q&A (self-test — cover the answers and try to recall)

**Q: Why not just use character-level tokenization and avoid this whole problem?**
A: Attention is O(n²) in sequence length, so 4-6x longer sequences means far more compute; also the model has to relearn word-formation patterns from scratch every time instead of treating common words as one concept.

**Q: Exact formula difference between BPE and WordPiece merge criteria?**
A: BPE: `argmax freq(a,b)`. WordPiece: `argmax freq(a,b)/(freq(a)·freq(b))` — a PMI-style normalization that favors statistically-bonded pairs over merely co-frequent ones.

**Q: Why does GPT use byte-level BPE instead of character-level?**
A: Bytes (256 possible values) form a mathematically complete base vocabulary — any string in any language or encoding is representable, so there is never an `<UNK>` token needed.

**Q: What's fundamentally different about Unigram LM vs BPE/WordPiece?**
A: It's top-down (prune from a large candidate set) and probabilistic (models full-sentence segmentation likelihood via EM+Viterbi), vs the bottom-up greedy merges of BPE/WordPiece. It also natively supports multiple valid tokenizations of the same input, enabling subword regularization as a training-time augmentation.

**Q: Is SentencePiece an algorithm or a framework? What's the tricky part people get wrong?**
A: It's a framework/library, not an algorithm itself — it can run either BPE or Unigram LM internally. The trap: Llama uses SentencePiece but with the BPE algorithm inside it, not Unigram LM — a lot of people assume "SentencePiece" automatically means "Unigram," which is wrong.

**Q: A word tokenizes into far more subwords in Language X than in English — what does that cost, concretely?**
A: Fewer "units of meaning" per token for that language → effectively shorter usable context window, higher inference cost per unit of meaning, and often worse downstream performance, unless the tokenizer was explicitly trained on a balanced multilingual corpus.

**Q: Why force digit-level tokenization for numbers?**
A: Without it, the same number can tokenize inconsistently depending on training-data frequency, which prevents the model from learning stable arithmetic patterns; forcing one-digit-per-token gives a consistent representation that measurably improves math performance.

---
*End of Module 1 (expanded). Next: Module 2 — Pretraining Objectives (CLM, MLM, prefix LM, span corruption).*
