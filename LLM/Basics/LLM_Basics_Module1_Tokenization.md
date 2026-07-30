# Module 1 — Tokenization (Master Notes, Expanded)


---

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

### 📌 Added Explanation: what "O(n²)" actually means here, in plain terms

The self-attention mechanism inside a Transformer computes a pairwise interaction score between *every* token and *every other* token in the sequence. If a sequence has `n` tokens, that's `n × n = n²` pairwise comparisons per attention layer.

- **In simple terms**: imagine a group chat. If there are 10 people, everyone can plausibly reply to everyone else — that's roughly 10×10=100 possible "who's talking to whom" pairs. If the group grows to 100 people, it's not 10x more pairs, it's 100×100=10,000 — 100x more. Sequence length behaves the same way inside attention.
- **Why this matters for tokenization**: this is precisely why granularity choice is not just a "nice-to-have" — going from word-level to character-level tokenization might turn a 10-token sentence into a 50-token sentence, which is a 5x increase in `n`, but a **25x** increase in attention compute (5²). This single fact is the core economic argument for subword tokenization.

### 🧮 Numerical Example: quantifying the O(n²) cost difference

Take the sentence: *"The unhappiness was overwhelming."*

| Tokenization | Token count (n) | Relative attention compute (n²) |
|---|---|---|
| Character-level (incl. spaces) | ~34 | 34² = 1,156 |
| Subword (BPE-style) | ~7 (`The`, `un`, `happi`, `ness`, `was`, `overwhelm`, `ing`, `.`) | 7² = 49 |
| Word-level | 5 (`The`, `unhappiness`, `was`, `overwhelming`, `.`) | 5² = 25 |

Going from subword to character-level tokenization here is roughly a **23x increase in attention compute** (1156 / 49 ≈ 23.6) for the *same sentence* — while word-level, despite being cheapest computationally, would likely send "unhappiness" or "overwhelming" to `<UNK>` if either is rare in the training vocabulary, destroying information. This table is the numeric backbone of why subword tokenization is the practical sweet spot.

---

## 1. Byte-Pair Encoding (BPE)

### Core idea, in plain words
BPE is dead simple: **look at the whole training corpus, find the two symbols that sit next to each other most often, glue them into one new symbol, and repeat.** No probability theory — just "what pair shows up together the most, over and over."

### 📌 Added Explanation: BPE in one analogy
Think of BPE like repeatedly stapling together the two most commonly seen sticky-note fragments on a wall until you run out of staples (your merge budget `k`). You never ask *why* two fragments go together statistically — you just staple whichever pair you see glued next to each other most often, over and over, until you've used up your allotted number of staples. This is why BPE is called a **greedy, frequency-driven, bottom-up** algorithm: greedy because it always takes the single best merge available *right now* without looking ahead, bottom-up because it starts from the smallest units (characters/bytes) and builds upward.

### Algorithm (training time)
1. Split corpus into words; represent each word as a sequence of characters + an end-of-word marker (so the model knows where a word ends, e.g. `low` → `l o w </w>`).
2. Count frequency of every adjacent symbol pair across the *entire* corpus.
3. Merge the single most frequent pair everywhere it occurs → this becomes one new symbol/token.
4. Repeat steps 2–3 for `k` iterations. `k` is a hyperparameter you choose upfront — it directly controls final vocab size.

### 📌 Added Explanation: why the end-of-word marker `</w>` matters
Without a boundary marker, the substring "er" inside "lower" (verb-ish ending) and the "er" inside, say, "herbal" (just a coincidental letter sequence inside a word, not at a word edge) would be indistinguishable to the merge-counting step. The `</w>` marker anchors *where in the word* a merge happened, so BPE can learn, for instance, that "er" specifically as a *word-final* suffix (as in "lower", "faster") is a meaningfully recurring pattern, separate from "er" appearing mid-word by coincidence. This is also exactly why the final merged unit in the worked example below is written as `low` and not just `low` floating freely — the end-of-word marker is silently attached to track word boundaries through every merge step.

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

### 🧮 Numerical Example: applying the trained merges to a brand-new, unseen word

Suppose at inference time you see the word **"lowest"** for the first time (it never appeared in the training corpus above). Because BPE tokenization at inference is just "replay the merge list in the same order," here's the step-by-step:

Start: `l o w e s t </w>`

1. Apply merge 1, `(e,s) → es`: is "e" immediately followed by "s" anywhere? Yes → `l o w es t </w>`
2. Apply merge 2, `(es,t) → est`: is "es" immediately followed by "t"? Yes → `l o w est </w>`
3. Apply merge 3, `(l,o) → lo`: is "l" immediately followed by "o"? Yes → `lo w est </w>`
4. Apply merge 4, `(lo,w) → low`: is "lo" immediately followed by "w"? Yes → `low est </w>`

**Final tokenization of "lowest": `low` + `est`** — two tokens, even though "lowest" was never in the training data. This is the generalization payoff of subword tokenization in action: the tokenizer correctly recovers the linguistically sensible split (root + superlative suffix) purely from merges learned on *other* words.

### Byte-level BPE (this is what GPT-2/3/4 actually use)
Instead of starting from Unicode *characters*, start from raw **bytes** (256 possible values, 0-255). Why this matters practically:
- A Unicode character-based vocab can still hit unseen characters (obscure scripts, emoji combos, corrupted text) → still needs an `<UNK>` fallback.
- A byte-level vocab is **mathematically complete** — every possible string, in every language, including binary garbage, is representable, because everything is bytes underneath. Zero `<UNK>` tokens, ever.
- Cost: non-Latin scripts (Chinese, Korean, Hindi) use multi-byte UTF-8 encodings per character, so those languages end up needing *more* tokens per character than English does. This is a real, measured inefficiency in GPT-family models for non-English text.

**Numbers to know**: GPT-2 vocab = 50,257 (256 base bytes + 50,000 merges + 1 special `<|endoftext|>` token).

### 📌 Added Explanation: why "256 possible byte values" gives mathematical completeness

A byte is 8 bits, and 2⁸ = 256. Every file on every computer — text in any language, images, audio, executable code — is, underneath, a sequence of bytes, each one of which is a number from 0 to 255. If your base vocabulary contains all 256 possible byte values as individual tokens, then *any* input, no matter how exotic (an emoji, a corrupted file, Thai script, an untranslated Klingon string), can always be represented as *some* sequence of those 256 known bytes. There is no such thing as a "257th byte value" that could appear and be unrecognized. This is why byte-level BPE achieves a hard guarantee — not a practical approximation — of zero out-of-vocabulary tokens.

**In simple terms**: it's like saying your alphabet has all possible "raw ingredients" (every byte value), so no matter what dish (input string) someone hands you, you can always break it down into ingredients you already have a name for — you just might have to use a lot of very small ingredients (single bytes) instead of one convenient pre-made block (a merged token), which is exactly the non-Latin-script cost described above.

### 🧮 Numerical Example: multi-byte cost for non-Latin scripts

The Chinese character 你 ("you") is encoded in UTF-8 as **3 bytes** (E4 BD A0 in hex), whereas the English letter "y" is encoded as **1 byte**. If a byte-level BPE tokenizer hasn't learned a merge that collapses those 3 specific bytes into a single token (which happens far less often if the training corpus was mostly English), then:

- English "you" → potentially 1 token (if "you" was merged as a whole word)
- Chinese 你 (same meaning) → potentially 3 tokens (one per raw byte, if no merge learned)

That's a **3x token-cost penalty** for expressing the identical concept, purely as an artifact of tokenizer training data imbalance — not because Chinese is inherently "3x harder" for the model to understand. This directly previews the "Multilingual tokenization cost" issue discussed later in Section 5.

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

### 📌 Added Explanation: full derivation, where this formula "comes from"

This formula is a simplified, count-based stand-in for **pointwise mutual information (PMI)**, defined formally as:

```
PMI(a, b) = log( P(a, b) / (P(a) × P(b)) )
```

Where:
- **`P(a, b)`** — the probability of observing `a` and `b` adjacent to each other, estimated from data as `freq(a,b) / total_pairs`.
- **`P(a)`, `P(b)`** — the marginal (individual, standalone) probabilities of symbol `a` and symbol `b` occurring at all, estimated as `freq(a) / total_symbols` and `freq(b) / total_symbols`.
- **`log(...)`** — PMI conventionally takes a log so that: independent events → PMI = log(1) = 0 (a clean, interpretable "no association" baseline); positively associated events → PMI > 0; negatively associated (avoid each other) → PMI < 0.

WordPiece's `score(a,b) = freq(a,b) / (freq(a) × freq(b))` is the **un-logged, raw-count version** of the *ratio inside* the PMI formula (the normalization constants like `total_pairs` and `total_symbols` cancel out or are treated as constant across all candidate pairs within one merge step, so they don't affect *which* pair scores highest — only the absolute score value — which is why WordPiece implementations can skip them and just use raw frequency counts directly).

**Why divide instead of subtract or do something else?** Because the entire question being asked is "is this a *ratio* worth noticing" — i.e., "is the joint frequency large *relative to* what independence would predict," not "is the joint frequency large in absolute terms" (that would just be BPE). Dividing by the product `freq(a)×freq(b)` is precisely the mathematical operation that answers "relative to chance."

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

### 🧮 Numerical Example: verifying the "9x higher, 33x smaller count" claim step by step

Let's double check the arithmetic from the original example so you can reproduce it under exam/interview pressure:

1. Raw count ratio: `800 / 24 = 33.33` → confirms (t,h) has ~33x the raw co-occurrence count of (q,u).
2. WordPiece score ratio: `0.00032 / 0.0000356 = 8.99 ≈ 9` → confirms (q,u) scores ~9x higher under WordPiece despite having a much smaller raw count.
3. **Sanity-check interpretation**: multiply out what "expected under independence" looks like for each pair —
   - Expected co-occurrence of (t,h) if independent: this is proportional to `freq(t) × freq(h) = 22,500,000` — a huge number, meaning "yes, t and h would co-occur a LOT even by pure chance, simply because both are everywhere."
   - Expected co-occurrence of (q,u) if independent: proportional to `freq(q) × freq(u) = 75,000` — much smaller, meaning "q and u co-occurring even a little bit is already notable, since q by itself is rare."
   - The *actual* observed count for (q,u), 24, is almost equal to `freq(q)=25` itself — meaning nearly **every single occurrence of "q" is followed by "u"** (24 out of 25 times, i.e., 96% conditional probability `P(u|q)`). That near-deterministic relationship is exactly what WordPiece's score is designed to surface, and exactly why English spelling conventions ("q is always followed by u") get captured early by WordPiece even though "q" is a rare letter overall.

### Practical detail: the `##` continuation marker
BERT's WordPiece tokenizer prefixes non-word-initial subword pieces with `##`. Example: "playing" → `play` + `##ing`. This tells the model (and lets you reconstruct text) that `##ing` attaches directly to the previous token with no space, vs a token like `ing` on its own which would mean a new word starting with "ing".

### 📌 Added Explanation: why you need *any* marker at all (reconstruction problem)

Once a sentence is split into subword tokens, the model (or any downstream code) needs to convert the token sequence back into normal, correctly-spaced text (this is called **detokenization**). If you just had a flat list of tokens like `["play", "ing", "the", "game"]`, you could not tell whether it should reconstruct to `"playing the game"` or `"play ing the game"` — both are plausible without extra information. The `##` prefix disambiguates this: `##ing` means "glue me directly onto the token before me, no space," while a bare `ing` (no `##`) means "I start a new word, insert a space before me." This is a small notational detail with an outsized practical consequence — get it wrong and every detokenized sentence in your pipeline is subtly broken.

### Where WordPiece is used standalone in practice
- **BERT** (original Google model) and virtually all its direct derivatives: **DistilBERT, ELECTRA, MobileBERT**.
- Google's production NMT system (the original 2016 paper that introduced WordPiece was for Google Translate, before BERT reused it).

---

## 3. Unigram Language Model tokenizer (via SentencePiece)

### Core idea, in plain words
Completely different philosophy from BPE/WordPiece: instead of building the vocab bottom-up by merging, **start with a huge candidate vocabulary and prune it down**, using a probabilistic model of "how likely is this segmentation of the sentence."

### 📌 Added Explanation: the "top-down pruning" analogy
If BPE/WordPiece are like building a Lego tower one brick-fusion at a time from the ground up, Unigram LM is like starting with a giant block of marble (a huge candidate vocabulary of every plausible substring) and **carving away** the pieces that contribute least to an overall sculpture (the corpus likelihood), repeating that carving process until only the essential shape remains at your target vocab size. Bottom-up = additive/constructive. Top-down = subtractive/reductive. Same end goal (a fixed-size subword vocab), fundamentally opposite direction of construction.

Also — and this is the detail people forget — **SentencePiece is not itself an algorithm; it's a language-agnostic *framework/library*** that can run either the BPE algorithm or the Unigram LM algorithm underneath, while treating raw text (including spaces, via a special `▁` symbol) as the input, so it never needs language-specific pre-tokenization (critical for Japanese/Chinese/Thai which have no whitespace between words).

### 📌 Added Explanation: what the `▁` (underscore-like) symbol actually solves

Most tokenizers assume you can first split text into words using whitespace, and *then* tokenize each word into subwords. But languages like Japanese, Chinese, and Thai don't reliably use whitespace to separate words at all — so a "split on spaces first" preprocessing step simply doesn't work for them. SentencePiece's fix: treat the space character itself as just another character to be tokenized, replacing it with a visible placeholder symbol `▁` before training. This means the tokenizer's merge/prune algorithm operates on completely raw text — spaces and all — uniformly across every language, with no special-cased, language-specific word-splitting logic needed anywhere in the pipeline. It also has a nice side benefit: because the space is now *part of* a token (e.g., `▁the` vs. `the` are different tokens, marking "start of a new word" vs. "continuation"), detokenization becomes trivial — just remove the `▁` symbols and you get correctly spaced text back, no separate rule needed (contrast with WordPiece's `##`, which encodes the *opposite* piece of information — "don't add a space here").

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

### 📌 Added Explanation: unpacking `P(sentence) = P(t1) × P(t2) × ... × P(tn)`

- **Where this comes from**: this is the chain rule of probability, `P(t1, t2, ..., tn) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...`, but with a **simplifying independence assumption**: the "unigram" in "Unigram LM" specifically means *no conditioning on previous tokens at all* — every `P(ti | previous tokens)` is simplified down to just `P(ti)`, its standalone probability, ignoring context. This is why the whole chain collapses to a simple product.
- **Why multiply, not add?** Because independent probabilities of a joint event combine multiplicatively (this is the same P(A and B) = P(A)×P(B) rule seen earlier in the WordPiece PMI discussion). Choosing token `t1` AND token `t2` AND token `t3` (in sequence) as your segmentation is a joint event, so their individual probabilities multiply.
- **Why is this a *simplification*, and why is that OK?** Real language obviously has context — "un" is far more likely to precede "happy" than to precede a random noun. A true model would need `P(t2 | t1)`, etc. But tracking every possible conditional dependency between candidate subword pieces is expensive and, empirically, unnecessary just for the *segmentation* decision (as opposed to the actual downstream language model, which absolutely does model context via attention/RNNs). The unigram assumption is a deliberate, practical simplification scoped only to "which way should I cut this text into pieces," not to "how well can I predict the next word" — those are different jobs.
- **What is `EM` (Expectation-Maximization) doing here, intuitively?** You don't know the true `P(t)` values upfront, and you don't know the true best segmentation of every sentence upfront either — these two unknowns depend on each other (better probabilities → better segmentations → which in turn should refine the probability estimates). EM alternates between two steps until convergence: **E-step** — given current `P(t)` estimates, compute (or softly weight) the likely segmentations of the corpus; **M-step** — given those segmentations, re-estimate `P(t)` values that best explain them (maximize likelihood). Repeat until the probabilities stop changing much. This chicken-and-egg alternating refinement is the classic EM pattern used throughout ML (e.g., also in Gaussian Mixture Models).

### 🧮 Numerical Example: comparing candidate segmentations by hand

Suppose after training, a toy Unigram vocabulary has these probabilities (invented numbers for illustration):

| Token | P(t) |
|---|---|
| `un` | 0.02 |
| `want` | 0.015 |
| `wanted` | 0.004 |
| `ed` | 0.06 |
| `unwanted` | 0.0007 |

For the word **"unwanted"**, compare the three candidate segmentations mentioned in the original notes:

1. `un` + `want` + `ed`: `P = 0.02 × 0.015 × 0.06 = 0.000018`
2. `un` + `wanted`: `P = 0.02 × 0.004 = 0.00008`
3. `unwanted` (single token): `P = 0.0007`

**Ranking**: segmentation 3 (`unwanted` as one token) has the highest probability (0.0007), followed by segmentation 2 (0.00008), then segmentation 1 (0.000018) — roughly **39x** lower than segmentation 3. The Viterbi algorithm (mentioned below) would select segmentation 3 as the best split at inference time, *provided* "unwanted" survived as a standalone token in the pruned vocabulary. If it didn't survive pruning (i.e., it's not in the final vocab at all, so it has no `P(t)` entry), then the algorithm would correctly fall back to comparing only segmentations 1 and 2, picking `un + wanted` as the winner. This numeric example shows exactly why Unigram LM is called *probabilistic* rather than *greedy*: it's comparing whole candidate splits against each other by multiplying probabilities, not just repeatedly taking "the biggest merge available right now" the way BPE does.

### Practical example: why multiple segmentations matter
Take the word "unwanted". There might be several valid ways to tokenize it under the trained vocab:
- `un` + `want` + `ed` (probability = P(un)×P(want)×P(ed))
- `un` + `wanted` (probability = P(un)×P(wanted))
- `unwanted` (if it happens to be in vocab) (probability = P(unwanted))

At inference, the **Viterbi algorithm** finds the single highest-probability segmentation — not necessarily the "greedy longest match" a human might guess. This is a real, useful property: **during training**, you can deliberately *sample* a lower-probability (but still valid) segmentation instead of always the best one — this is called **subword regularization**, and it acts like data augmentation, making the model robust to alternate ways the same word could be split, which improves generalization especially in low-resource settings.

### 📌 Added Explanation: what Viterbi is actually doing, and why it's needed (not brute force)

In principle, "find the highest-probability segmentation" could be done by literally listing every possible way to cut up a word/sentence into vocabulary pieces and computing `P(t1)×P(t2)×...` for each, then taking the max. But the number of possible segmentations of a string grows **exponentially** with its length (every character boundary is either "cut here" or "don't cut here," so an n-character string has up to 2^(n-1) possible segmentations). That's computationally infeasible for long sentences.

**Viterbi** is a *dynamic programming* algorithm that avoids this explosion. Intuitively: the best segmentation of the *whole* string up to some position `i` can always be built from the best segmentation up to some earlier position `j`, plus one more token spanning from `j` to `i` — you never need to "re-decide" earlier parts of the string once you know their best-so-far score. This lets Viterbi solve the problem in time roughly proportional to (length of string × max token length), instead of exponential time. This is the exact same algorithmic trick used in Hidden Markov Models for finding the most likely state sequence — same "reuse the best sub-solution" idea, different application.

**Subword regularization, in simple terms**: instead of *always* feeding the model the single "best" (Viterbi-optimal) segmentation of a training word, sometimes deliberately feed it a slightly-less-optimal-but-still-valid alternate segmentation (sampled proportionally to segmentation probability). This is directly analogous to image data augmentation (randomly cropping/rotating an image so a vision model doesn't overfit to one exact pixel arrangement) — except here you're augmenting *how a word is sliced into pieces*, so the model doesn't overfit to assuming a word can only ever be split one specific way, which helps especially when the model later sees noisy text, typos, or rare words at inference time that may not segment the "textbook" way.

### Where Unigram LM / SentencePiece is used standalone in practice
- **T5, ALBERT, XLNet** use the Unigram LM algorithm via SentencePiece.
- **Llama, Llama 2** use SentencePiece as the *framework*, but with the **BPE algorithm** running inside it (not Unigram LM) — this is a common point of confusion, worth stating explicitly in interviews: "SentencePiece is the wrapper/framework; BPE and Unigram LM are two different algorithms you can run inside it."

> **⚠️ Flag (accuracy check, not guessed confidently)**: Llama 3 (as distinct from Llama/Llama 2) switched to a byte-level BPE tokenizer similar in spirit to GPT-4's/`tiktoken`-style tokenizers, with a much larger vocabulary (~128k tokens), rather than continuing with the SentencePiece library used by Llama 1/2. If this distinction matters for your interview prep (e.g., you're asked specifically about Llama 3), please verify against current official documentation/model cards rather than relying solely on this note, since tokenizer details are exactly the kind of implementation fact that changes across model versions and is easy to misremember.

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

### 📌 Added Explanation: one more row worth memorizing — how each handles a brand-new unseen word

| | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| Unseen word handling | Replay merge list in fixed order until no more merges apply; whatever's left (down to bytes/chars) is the tokenization | Same replay-merge idea, but greedy longest-match-first against the vocab is the typical implementation | Run Viterbi over the (already pruned, fixed) vocab to find the single highest-probability valid segmentation |

---

## 5. Practical issues (very common interview follow-ups)

**Multilingual tokenization cost**: if a BPE vocab is trained mostly on English text, non-Latin scripts get chopped into more tokens per character (since those characters were rare in training, they don't get merged into big compact tokens). Practical effect: a Hindi or Chinese sentence that means the same thing as an English one can cost 2-4x more tokens → shorter effective context window and higher inference cost for those languages, purely from tokenizer bias, not model capability. Fix: train tokenizer on a balanced multilingual corpus (this is exactly what multilingual models like BLOOM, mT5 do).

**Digit/number handling**: naive BPE tokenizes numbers inconsistently based on frequency — "2024" might be a single token (common year), but "20247" might split unpredictably as "202"+"47" or "2024"+"7" depending on what merges happened to be frequent in training. This inconsistency actively hurts arithmetic ability, since the model can't learn a stable "one digit = one token" pattern. **Fix used by Llama and GPT-4-class tokenizers**: force every digit to be its own separate token, always — this single design choice measurably improves arithmetic performance.

### 🧮 Numerical Example: the digit-tokenization inconsistency problem, concretely

Imagine a BPE vocab where, purely due to training-data frequency, these merges happened to form:
- "20" is a learned merge (common as a century prefix: "2019", "2020", "2021"...)
- "24" is a learned merge (common: "2024", "1924"...)
- But "202" and "247" are *not* learned merges (rare as standalone chunks)

Then:
- `"2024"` → tokenizes as `20` + `24` (2 tokens) — clean and consistent
- `"2047"` → tokenizes as `20` + `47`, assuming "47" was learned — still 2 tokens, but a *different pairing pattern*
- `"20247"` (a 5-digit number, e.g., a product SKU) → might tokenize as `20` + `247` or `202` + `47` or `20` + `24` + `7`, depending entirely on which merges exist — the model sees a *different, unpredictable* digit grouping every time a number doesn't happen to align with previously-merged chunks.

**Why this actually breaks arithmetic**: if the model is asked to add `202 + 47`, but at training time it only ever saw the *digit sequence* "20247" chunked as `20`+`247` in unrelated contexts (like a serial number, never as "202 plus 47"), it has no clean, consistent internal representation to hook place-value reasoning onto — the token boundaries don't align with the actual mathematical structure (hundreds/tens/ones) of the number. Forcing one-digit-per-token means `"20247"` always tokenizes as `2`+`0`+`2`+`4`+`7` — five tokens, every time, in every context — giving the model a stable, position-consistent representation to learn place-value arithmetic patterns on top of, regardless of what specific number it's seeing.

**Tokenizer vocab-size tradeoff, concretely**: 
- Smaller vocab (e.g. 30k, BERT) → longer sequences, smaller embedding matrix.
- Larger vocab (e.g. 128k, Llama-3) → shorter sequences (less attention compute, since attention scales O(n²)), but bigger embedding + output softmax layer (more parameters, more memory).
- Modern large models trend toward bigger vocabs because, at scale, the O(n²) attention savings from shorter sequences outweighs the linear cost of a bigger embedding table.

### 🧮 Numerical Example: quantifying the vocab-size tradeoff

Say a model has hidden dimension `d = 4096` (a Llama-family-scale number).

- **Embedding table size** = `vocab_size × d`. 
  - 30k vocab: `30,000 × 4096 ≈ 123M` parameters.
  - 128k vocab: `128,000 × 4096 ≈ 524M` parameters.
  - Difference: roughly **400M more parameters** just from a bigger vocab (this cost is duplicated again for the output/softmax layer if input and output embeddings aren't tied, so potentially ~800M extra parameters total).
- **Attention compute savings**: if the larger vocab reduces average sequence length by, say, 30% (because more text gets compressed into fewer, longer tokens), and attention cost scales with `n²`, then attention compute drops to `(0.7)² ≈ 0.49`, i.e., **roughly half the attention compute** for the same text, for every single attention layer, at every single training/inference step, across the model's entire lifetime of usage.
- **Why big models still choose the bigger vocab**: the ~400-800M extra parameters is a **one-time, fixed cost** (paid once, stored on disk/in memory). The ~2x attention compute savings is a **recurring cost** paid on every forward pass, for every token, for the life of the model. At the scale of billions of inference calls, the recurring compute savings dwarfs the one-time parameter cost — this is precisely the reasoning stated in the original bullet point above, now with concrete numbers behind it.

**OOV handling, resolved**: with byte-level BPE or SentencePiece, there is no true OOV state — worst case, a completely novel string decomposes all the way down to individual bytes/characters, which is *always* in vocab by construction. This is why word-level tokenizers with `<UNK>` tokens are essentially extinct in modern LLMs.

---

## 6. Quick-fire Q&A (self-test — cover the answers and try to recall)

*(Original questions and answers below, kept fully intact. Each answer has been additionally expanded with fuller reasoning per your request — expansions marked 📌.)*

**Q: Why not just use character-level tokenization and avoid this whole problem?**
A: Attention is O(n²) in sequence length, so 4-6x longer sequences means far more compute; also the model has to relearn word-formation patterns from scratch every time instead of treating common words as one concept.

📌 **Expanded reasoning**: There are really two independent costs being described here, and it's worth separating them explicitly, since interviewers often probe which one you mean: (1) a **pure compute/systems cost** — quadratic attention scaling means character-level sequences (4-6x longer than word-level) cost roughly 16-36x more attention FLOPs for the same piece of text (per the O(n²) numeric example in Section 0); and (2) a **statistical/learning-efficiency cost** — even ignoring compute entirely, a character-level model must, in every single training example containing the word "cat," re-derive from raw co-occurring characters `c-a-t` that this specific 3-letter sequence maps to a stable, meaningful concept, rather than being handed "cat" as an atomic unit whose embedding can directly accumulate semantic information across training. Both costs point the same direction (against character-level), but they're conceptually distinct — one is about FLOPs, the other is about sample efficiency/representation learning.

**Q: Exact formula difference between BPE and WordPiece merge criteria?**
A: BPE: `argmax freq(a,b)`. WordPiece: `argmax freq(a,b)/(freq(a)·freq(b))` — a PMI-style normalization that favors statistically-bonded pairs over merely co-frequent ones.

📌 **Expanded reasoning**: The `argmax` notation means "the pair (a,b), out of all candidate adjacent pairs currently in the corpus, that maximizes this expression" — i.e., at every merge step, you're not just computing one score, you're computing this score for *every* currently-adjacent pair across the whole corpus and picking the single winner. The practical consequence, worked numerically above (the t,h vs q,u example), is that BPE's merge order tends to front-load "common because the letters themselves are common" pairs, while WordPiece's merge order front-loads "reliably-glued-together" pairs regardless of how common the individual letters are — this is the single sentence that captures the entire philosophical difference between the two algorithms.

**Q: Why does GPT use byte-level BPE instead of character-level?**
A: Bytes (256 possible values) form a mathematically complete base vocabulary — any string in any language or encoding is representable, so there is never an `<UNK>` token needed.

📌 **Expanded reasoning**: It's worth being precise about *why* "character-level" doesn't already give you this guarantee, since this is a common point of confusion — Unicode currently defines over 149,000 characters, and new ones are added over time (new emoji, new scripts); a character-level vocabulary built at training time is necessarily a *finite snapshot* of characters seen so far, so it can still be blindsided by a genuinely novel character introduced after training (or simply rare enough to not appear in the training corpus at all). Bytes, by contrast, are a fixed, closed set defined by computer architecture itself (0-255) — this is not a matter of "how big is the training corpus," it's a hard mathematical ceiling that can never be exceeded, which is the actual distinction between "very unlikely to hit `<UNK>`" (large character vocab) and "provably impossible to hit `<UNK>`" (byte vocab).

**Q: What's fundamentally different about Unigram LM vs BPE/WordPiece?**
A: It's top-down (prune from a large candidate set) and probabilistic (models full-sentence segmentation likelihood via EM+Viterbi), vs the bottom-up greedy merges of BPE/WordPiece. It also natively supports multiple valid tokenizations of the same input, enabling subword regularization as a training-time augmentation.

📌 **Expanded reasoning**: The "greedy" label on BPE/WordPiece is doing a lot of work here and is worth spelling out: at each merge step, both algorithms commit permanently to the single best-scoring merge *at that step*, and never revisit or undo that decision later, even if a different order of merges might have produced a globally better vocabulary. Unigram LM's EM-based refitting, by contrast, repeatedly re-evaluates *every* token's contribution to overall corpus likelihood and can effectively "undo" earlier choices by pruning a token that looked good in isolation but turns out to be a poor global fit once other tokens are accounted for. This greedy-vs-globally-refitted distinction is the deep reason Unigram LM is described as "probabilistic" while BPE/WordPiece are described as "counting"/"statistical association" respectively — only Unigram LM has a mechanism to revise past decisions in light of new information.

**Q: Is SentencePiece an algorithm or a framework? What's the tricky part people get wrong?**
A: It's a framework/library, not an algorithm itself — it can run either BPE or Unigram LM internally. The trap: Llama uses SentencePiece but with the BPE algorithm inside it, not Unigram LM — a lot of people assume "SentencePiece" automatically means "Unigram," which is wrong.

📌 **Expanded reasoning**: A useful mental model for keeping this straight: "SentencePiece" answers the question *"how is raw text turned into a stream of symbols before any algorithm even runs?"* (its answer: treat everything, including spaces via `▁`, as raw text with no language-specific pre-splitting). "BPE" and "Unigram LM" answer a completely different question: *"given that stream of symbols, what rule decides the actual vocabulary and segmentation?"* These are two independent axes/decisions, and any combination is technically possible — SentencePiece just happens to be the library that implements both algorithm choices under one roof, which is exactly why people conflate the library name with one specific algorithm.

**Q: A word tokenizes into far more subwords in Language X than in English — what does that cost, concretely?**
A: Fewer "units of meaning" per token for that language → effectively shorter usable context window, higher inference cost per unit of meaning, and often worse downstream performance, unless the tokenizer was explicitly trained on a balanced multilingual corpus.

📌 **Expanded reasoning**: To make "shorter usable context window" concrete: if a model's maximum context length is fixed at, say, 8,000 tokens, and English text averages 1.3 tokens per word while Language X averages 3 tokens per word (due to tokenizer bias, as in the 你/"you" byte example above), then the *same document* that would fit in ~6,150 English words only leaves room for ~2,667 words of Language X before hitting the same 8,000-token ceiling — the model can "see" less than half as much actual content in Language X, purely as a tokenizer artifact, even though the underlying attention mechanism and parameter count are identical. This is a real, measured, and actively-researched fairness/equity issue in multilingual LLM deployment, not a hypothetical concern.

**Q: Why force digit-level tokenization for numbers?**
A: Without it, the same number can tokenize inconsistently depending on training-data frequency, which prevents the model from learning stable arithmetic patterns; forcing one-digit-per-token gives a consistent representation that measurably improves math performance.

📌 **Expanded reasoning**: see the fully worked "20247" numeric example in Section 5 above for the mechanics of exactly *how* this inconsistency arises and *why* it specifically breaks place-value arithmetic reasoning, not just "makes things messier in general."

---

## ❓ Interview Q&A (Apple / Google-style ML Engineer questions — newly added section)

*(These are additional interview-style questions in the spirit of what's typically asked in FAANG/Apple ML Engineer interviews on tokenization, going beyond the quick-fire set above. Answers are given in full below each question — scroll past the question to self-test first if you'd like.)*

**Q1. You're given a fixed compute and memory budget and asked to choose a tokenizer for a new multilingual LLM covering English, Mandarin, and Arabic. Walk through your decision process.**

*Model answer*: I'd start from the requirement that OOV must be impossible at deployment scale across three scripts with very different structures (Latin, Chinese logographic, Arabic abjad/cursive) — so I'd rule out plain word-level tokenization immediately. Between byte-level BPE and SentencePiece-Unigram, both are viable; I'd lean toward SentencePiece (either algorithm) specifically because it treats raw text uniformly without language-specific pre-tokenization, which matters a lot for Chinese (no whitespace word boundaries) and Arabic (complex morphology, contextual letter shaping). The single highest-leverage decision, though, isn't BPE-vs-Unigram — it's making sure the **training corpus for the tokenizer itself is balanced** across the three languages (not just English-dominant), since, as covered above, an English-majority training corpus systematically under-serves the other two languages with more tokens-per-concept, directly costing effective context window and inference budget for exactly the users I'm trying to support well. I'd also force digit-level tokenization regardless of the base algorithm chosen, since arithmetic consistency is orthogonal to the multilingual concern and is a near-free fix.

**Q2. Prove, using the two formulas from these notes, that BPE and WordPiece can select a different "winning" merge from the exact same underlying corpus statistics.**

*Model answer*: This is exactly the t/h vs q/u example worked out above — I'd reproduce it: given freq(t)=5000, freq(h)=4500, freq(t,h)=800, freq(q)=25, freq(u)=3000, freq(q,u)=24, BPE's raw-count criterion picks (t,h) since 800 > 24. WordPiece's normalized criterion, `freq(a,b)/(freq(a)×freq(b))`, evaluates to ≈0.0000356 for (t,h) and ≈0.00032 for (q,u) — WordPiece picks (q,u) instead, roughly 9x higher score, precisely *because* it divides out the base-rate frequency of each individual symbol, which BPE never does. This is a clean, fully-reproducible numeric proof that the two algorithms are not just "similar with minor tweaks" — they can disagree on their very first merge decision given identical input statistics.

**Q3. Why is the Unigram LM approach described as supporting "subword regularization," and how would you implement this as a training-time data augmentation in a PyTorch data pipeline?**

*Model answer*: Because the trained vocabulary has explicit per-token probabilities `P(t)`, you can enumerate (or sample from) multiple valid segmentations of the same input string, each with its own overall likelihood — rather than deterministically always taking the Viterbi-optimal single best segmentation. In a PyTorch pipeline, I'd implement this in the `Dataset.__getitem__` (or a custom collate function) by calling the SentencePiece tokenizer's sampling API (e.g., `enable_sampling=True`, with an `alpha` temperature parameter controlling how close to "always pick the best" vs. "sample proportional to probability" the sampling behaves) on every call, rather than the deterministic `.encode()` call — so that across training epochs, the exact same raw sentence gets tokenized slightly differently each time it's seen, functioning analogously to random crop/flip augmentation in vision pipelines.

**Q4. A teammate says "SentencePiece is just BPE with extra steps." What's wrong with this claim, and how would you correct them precisely?**

*Model answer*: The claim conflates a framework with an algorithm. SentencePiece is a text-preprocessing and training framework (handling raw-text-with-spaces input via the `▁` marker, with no language-specific pre-tokenization); it is not itself a merge/prune rule. It can run the BPE algorithm internally (as Llama/Llama2 does) *or* the Unigram LM algorithm internally (as T5/ALBERT/XLNet do) — these produce meaningfully different vocabularies and different segmentation behaviors (deterministic greedy merges vs. probabilistic EM-pruned + Viterbi segmentation with regularization support) even though both can be wrapped inside the same SentencePiece library. I'd correct the claim by pointing out that "SentencePiece" and "algorithm choice" are two independent axes, as covered in the quick-fire Q&A above.

**Q5. Suppose your production model's tokenizer was trained with a small (30k) vocab, but you now want to reduce average sequence length by 25% without retraining the entire model from scratch. What are your options, and what are the risks of each?**

*Model answer*: The core tension is that the embedding matrix and (if untied) output projection are shaped directly by vocab size, so simply swapping in a bigger-vocab tokenizer post-hoc breaks weight compatibility — the model's embedding table literally doesn't have rows for the new tokens. Options, roughly in order of increasing invasiveness: (a) **do nothing at the tokenizer level and instead prune/optimize elsewhere** (e.g., quantization, KV-cache compression) if the actual goal is inference cost rather than sequence length specifically; (b) **train a new larger-vocab tokenizer and do embedding-matrix surgery** — initialize new tokens' embeddings (e.g., as an average of the sub-piece embeddings that would have made up that token under the old tokenizer) and continue pretraining/fine-tuning briefly to let the model adapt, which is cheaper than full retraining but still carries real risk of representation mismatch and requires careful evaluation; (c) **full retrain with the new tokenizer**, the safest but most expensive option, appropriate only if the 25% sequence-length reduction is projected to pay for itself at serving scale (tie this back to the vocab-size tradeoff numeric example above — bigger vocab is a fixed one-time cost against a recurring O(n²) attention savings, so at high enough serving volume, full retraining can be worth it).

**Q6. In one sentence each, state the failure mode that WordPiece's PMI-style scoring is specifically designed to prevent, and give a concrete pair of letters from a language other than English where this failure mode would occur.**

*Model answer*: WordPiece's PMI-style scoring is specifically designed to prevent merging pairs that are frequent only because their *individual* symbols are common, not because they're linguistically/statistically bonded (the "t,h" problem). In French, the letters "e" and "s" are both extremely frequent individually (plural endings, common vowel) — a raw-frequency algorithm like BPE might over-eagerly merge "e,s" early purely from base-rate frequency, similar to the English "t,h" case, whereas WordPiece's normalization would correctly discount this unless "es" is genuinely more tightly bonded than base rates predict. *(Flagged: this specific French example is illustrative reasoning by analogy to the documented English case, not a verified empirical finding from a specific WordPiece training run on French corpora — worth independently checking real French WordPiece merge logs if this exact claim needs to hold up under scrutiny.)*

---

*End of Module 1 (expanded). Next: Module 2 — Pretraining Objectives (CLM, MLM, prefix LM, span corruption).*
