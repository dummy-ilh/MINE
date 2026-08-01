# Chapter 2 — Tokenization & Vocabulary (Master Notes, Apple MLE Prep)

> Goal of this doc: explain from memory why word-level and char-level tokenization both fail, how WordPiece actually scores merges (not just "counts pairs" — this is a real interview trap), trace one sentence to token IDs by hand, and defend every special token's existence.

---

## 0. One-sentence version

> "Tokenization is the tradeoff between vocabulary size and sequence length — word-level gives short sequences but an unbounded, OOV-prone vocabulary; character-level gives a tiny vocabulary but blows up sequence length and destroys meaning; WordPiece is the middle ground, building a fixed ~30k vocabulary of subword pieces so common words stay whole and rare words decompose into meaningful, previously-seen fragments."

---

## 1. Why three "obvious" choices all fail

The real constraint underneath all of this: a neural net has a **fixed vocabulary size** (the embedding table) and a **fixed max sequence length** (positional embeddings, Chapter 3). Every tokenization scheme is a different point on the tradeoff between those two fixed budgets.

### 1.1 Word-level — vocabulary side loses control

**Why "just add every new word to the vocab" doesn't work long-term**: the embedding table is a lookup matrix of shape `[vocab_size, 768]`. Every word you add is another *learnable row* that needs enough training examples to get a good vector — rare words (names, typos, new slang) get few or zero updates and stay near-random. And you can't retroactively add rows after deployment: a fixed-vocabulary model literally has no slot for a word it's never seen.

**What if we just used a huge vocab, like 1 million words, to cover everything?** Two costs, not one: (1) the embedding table itself becomes enormous — 1M × 768 floats is ~3GB just for input embeddings, before the rest of the model; (2) it doesn't even solve OOV, because language keeps producing new words (new drugs, new slang, new usernames) faster than any fixed list can capture, and rare words in your million-word vocab still get too few training examples to be useful.

**Why morphological blindness is a real cost, not a nitpick**: "run," "running," "runner," "ran" get four *unrelated* rows in the embedding table. The model has to independently relearn "this is about running" four times from four separate, smaller pools of training data, instead of sharing statistical strength across a common root.

### 1.2 Character-level — sequence length side loses control

**Why tiny vocab isn't a free win**: yes, ~100 characters means no OOV, ever — any string can be spelled out. But you've pushed the entire burden onto sequence length and onto the model's *capacity to compose meaning from scratch*.

**The concrete cost, quantified**: "The cat sat on the mat" is 6 tokens at word-level, ~22 characters at char-level — roughly a 3-4x sequence length increase. Self-attention's compute and memory cost scales **quadratically** with sequence length (every token attends to every other token — see Chapter 1's $QK^T$). A 4x longer sequence isn't 4x more expensive, it's **~16x more expensive** in the attention computation. That's the real reason char-level tokenization is a nonstarter for Transformers at scale, not just "sequences look long."

**What if we accepted the cost for the OOV-robustness benefit?** Some systems do (character-level or byte-level models exist, e.g. ByT5). The tradeoff is real and sometimes worth it for very noisy or multilingual text — but for a general-purpose encoder like BERT, the compute cost at 512-token budgets was judged not worth it versus subwords, which get *most* of the OOV robustness at a fraction of the sequence-length cost.

### 1.3 WordPiece — subword is the actual constraint-satisfying answer

The insight: **most of the vocabulary "explosion" problem comes from rare words, and most rare words are built from common pieces** (prefixes, suffixes, roots). So: keep a fixed budget of ~30k tokens, spend it mostly on whole common words, and let rare words fall back to a *composition* of pieces the model has already seen many times elsewhere.

---

## 2. How the WordPiece vocabulary is actually built — corrected and simplified

**Important correction to flag going in**: the chapter's step-by-step worked example (Section "Step 3: Count All Adjacent Pairs" → "merge the highest count pair") describes **raw frequency-based merging**. That is actually the **BPE (Byte-Pair Encoding)** algorithm — the one GPT-2/GPT-3/RoBERTa use. **WordPiece**, which is what BERT specifically uses, merges based on a different score. This distinction is a common interview trap, so let's get it exactly right.

### 2.1 The two scoring rules, side by side

**BPE merge rule** (raw co-occurrence count):
$$\text{score}_{BPE}(a, b) = \text{count}(a, b)$$
Just merge whichever adjacent pair appears together most often in the corpus.

**WordPiece merge rule** (likelihood gain, simplified):
$$\text{score}_{WordPiece}(a, b) = \frac{\text{count}(a, b)}{\text{count}(a) \times \text{count}(b)}$$

Plain-language read of every term:
- $\text{count}(a, b)$ — how often symbol $a$ is immediately followed by symbol $b$ in the corpus (same as BPE's numerator).
- $\text{count}(a)$, $\text{count}(b)$ — how often $a$ and $b$ each appear *on their own*, anywhere in the corpus (not just next to each other).
- **Dividing by the product of individual frequencies** is the whole trick: it asks "how much does merging $a$+$b$ pay off *relative to* how useful $a$ and $b$ already are as separate, common symbols?" A pair that's very frequent together but where each half is *already* extremely common elsewhere (like "t" and "h" — both used constantly in tons of other words) scores lower than a pair that's less frequent together but whose halves almost *only* ever occur next to each other.

### 2.2 Why this distinction matters — worked toy numbers

Toy corpus with two candidate merges:

**Candidate pair A: ("t", "h")** — very frequent overall, since "t" and "h" both appear in tons of unrelated words.
```
count(t, h) = 4
count(t)    = 10   (appears in many other words too)
count(h)    = 6    (appears in many other words too)
```
- BPE score: **4** (just the raw count)
- WordPiece score: 4 / (10 × 6) = 4/60 ≈ **0.067**

**Candidate pair B: ("z", "q")** — rarer overall, but almost always occurs together (imagine a rare loanword where z and q are joined).
```
count(z, q) = 3
count(z)    = 3    (barely appears anywhere except next to q)
count(q)    = 3    (same)
```
- BPE score: **3** (lower raw count than A)
- WordPiece score: 3 / (3 × 3) = 3/9 ≈ **0.333**

**Result**: BPE merges (t, h) first — it has the higher raw count. WordPiece merges (z, q) first — it has the higher *likelihood-gain* score, because z and q gain almost nothing from staying separate (they're rare on their own and only really "mean something" together), whereas t and h are already both individually useful, common building blocks that don't urgently need merging.

**The intuition to say out loud in an interview**: WordPiece asks "which merge most increases the probability of the training corpus under a unigram language model over the vocabulary" — it prioritizes merges that reduce redundancy for symbols that are otherwise wasteful to keep separate, not just whichever pair happens to co-occur most in absolute terms. Practically, this tends to make WordPiece slightly more conservative about merging very common symbols and quicker to lock in pairs that are tightly bound.

### 2.3 The rest of the algorithm (this part the original chapter got right)

1. Start with every character as a token (initial vocab, ~40 symbols for a small corpus, ~100+ for full Unicode coverage).
2. Score every adjacent pair using the formula above.
3. Merge the highest-scoring pair into a new single token; add it to the vocabulary.
4. Repeat, rescoring after every merge (merging changes what "adjacent" means for the next round), until the vocabulary hits the target size — **30,522 for BERT**.

**What if we picked a smaller target, like 5,000 tokens?** Fewer whole-word tokens survive; more common words get needlessly fragmented, inflating sequence length and diluting the signal per word (similar failure mode to character-level, just less extreme).

**What if we picked a much larger target, like 200,000 tokens?** You approach word-level tokenization's problems again — a bigger embedding table, more rare/undertrained rows, less benefit from subword sharing, since almost everything just gets its own whole-word token. 30,522 is an empirically-tuned middle point for English-heavy corpora, not a theoretically derived optimum.

---

## 3. From text to token IDs — the full pipeline

```
Raw text → lowercase → WordPiece split → add [CLS]/[SEP] → look up integer IDs
```
*(see the pipeline diagram above — each stage's output is the next stage's input, nothing more.)*

**Worked example, "The cat sat":**
```
lowercase:        "the cat sat"
WordPiece split:   ["the", "cat", "sat"]      ← all common, stay whole
add special tok:   ["[CLS]", "the", "cat", "sat", "[SEP]"]
token IDs:         [101, 1996, 4937, 2938, 102]
```

**Why lowercase first, specifically for BERT-base-uncased**: lowercasing before tokenizing means "Cat" and "cat" collapse to the same token, which is more data-efficient — the model doesn't need to separately learn that both mean the same thing. **The tradeoff**: casing carries real signal sometimes (proper nouns, acronyms, sentence starts, sarcasm via "NO"), which is exactly why `bert-base-cased` also exists as a separate checkpoint — it's a deliberate choice per use case, not a universal default.

**What if you use the cased tokenizer's IDs with the uncased model's weights (or vice versa)?** This is a real production bug class: the vocab files differ between cased/uncased checkpoints, so token ID 1996 might mean something completely different in each. Mismatching tokenizer and model silently produces garbage — always load them as a matched pair.

---

## 4. Every special token, with the "what if we didn't have it"

### [CLS] (ID 101) — always position 0

**What it's for**: after 12 layers of self-attention, `[CLS]`'s final vector has attended to every other token in the input, so it ends up holding a learned summary of the whole sequence — used as the input to a classifier head for sentence-level tasks (sentiment, entailment, etc.).

**What if we instead just averaged all the real tokens' final vectors for classification, and skipped [CLS] entirely?** This is a legitimate alternative (mean pooling), and people do use it. The tradeoff: `[CLS]` is trained *specifically* to be a good aggregator (via the pre-training objectives, including Next Sentence Prediction, which directly supervises `[CLS]`'s behavior), whereas raw mean-pooling of token vectors averages in whatever those vectors were optimized for (per-token masked-word prediction, not sentence-level summary) — often works fine in practice but isn't purpose-built the way `[CLS]` is.

### [SEP] (ID 102) — segment boundary marker

**What it's for**: tells the model exactly where sentence/segment A ends and B begins — essential for any two-sequence task (QA: question + passage; NLI: premise + hypothesis).

**What if we just concatenated two sentences with a space, no [SEP]?** The model would have no explicit signal for the boundary — it would have to *infer* segment structure purely from content, which is strictly harder and throws away free, unambiguous structure you could've just told it directly. This is compounded by segment IDs (the 0/0/.../1/1/... array) providing a *second*, redundant boundary signal — belt and suspenders.

### [MASK] (ID 103) — pre-training-only placeholder

**What it's for**: stands in for a hidden token during MLM pre-training (see Chapter 1, section 2.6).

**Why it's a training-only artifact, and why that itself is a designed-around problem**: `[MASK]` never appears in real downstream text — no fine-tuning dataset or production input will ever contain the literal string `[MASK]`. If BERT only ever saw `[MASK]` in that position during training, it would learn a representation specialized for "predict the word under this special placeholder token" — a skill useless at inference time, where there's no placeholder, just real words. This is exactly why the actual MLM procedure (Chapter 1) doesn't mask 100% of the selected 15% with the literal token — it swaps in the correct word 10% of the time and a random wrong word 10% of the time, forcing the model to build robust representations at *every* position, not just masked ones.

### [PAD] (ID 0) — batch-alignment filler

**What it's for**: batches need every sequence at the same length for the tensor math to work (a batch is one big rectangular tensor), so shorter sequences get padded with a token that carries no meaning.

**What if we forgot to also pass the attention mask?** This is a real, common bug: without an attention mask, the model would compute attention scores *involving* the pad tokens as if they were real content, letting real tokens attend to meaningless padding and letting `[PAD]`'s own (meaningless) vector influence the sentence-level `[CLS]` representation. The attention mask (a parallel 1/0 array) is what actually excludes padding positions from the softmax in self-attention — the `[PAD]` token ID alone does nothing to protect you; you must apply the mask.

---

## 5. The 512-token limit

**Where the number literally comes from**: BERT's positional embedding table (Chapter 3) has exactly 512 learned rows — one vector per position, 0 through 511. There is no position-512 vector; it doesn't exist in the trained model. This isn't a soft heuristic, it's a hard architectural ceiling.

**What if you truncate a document to fit?** You lose everything past the cutoff — for a long contract or research paper, that's a real, common failure mode: the answer to a question might live in the truncated tail. (Chapter 12, per the original text, presumably covers sliding-window / hierarchical approaches to work around this — chunk the document, run BERT per chunk, then aggregate.)

**What if you just fed in more than 512 tokens anyway?** You'd either get an index-out-of-range error at the positional embedding lookup, or (if the library silently truncates for you) lose content without necessarily realizing it — a classic silent-failure production bug.

Budget math: `[CLS]` + sentence A + `[SEP]` + sentence B + `[SEP]` = 3 special-token slots, so `len(A) + len(B) ≤ 509` for two-segment inputs.

---

## 6. Domain vocabulary mismatch — why this is a real production problem, not a curiosity

**The mechanism, stated precisely**: WordPiece's vocabulary was built by frequency (or likelihood-gain, per the corrected formula above) on BERT's *general-English pre-training corpus* (Wikipedia + BooksCorpus). A word is "common" or "rare" *relative to that corpus*, not relative to your domain. "Myocardial" is rare in Wikipedia-and-books English, so it gets shredded into `my ##oca ##rd ##ial` — four disconnected embedding-table rows that individually mean almost nothing, even though "myocardial" is an everyday word to a cardiologist.

**Why this hurts the model, specifically**: the model never got to build one strong, well-trained representation for the *concept* "myocardial" — its meaning is scattered across four generic-sounding fragments that also show up in totally unrelated words. Self-attention has to work harder to reassemble something coherent from noisier pieces, and there's simply less concentrated training signal behind any single fragment's contribution to that specific medical meaning.

**What if we just fine-tuned standard BERT on medical text without changing the tokenizer?** You can, and it helps somewhat — the model can adjust how it *combines* the existing fragments. But the tokenizer's vocabulary itself is frozen; you're still stuck describing "myocardial" as four generic pieces, you're just getting a bit better at recombining them. You never get the efficiency and concentrated signal of a single dedicated token.

**The actual fix**: retrain WordPiece from scratch on domain text (what BioBERT/ClinicalBERT do) so that domain-common words earn their own whole-token slot in the fixed 30k budget, at the cost of some general-English words now being slightly less "whole" than they'd otherwise be — you're still spending the same fixed token budget, just reallocating it toward your domain's frequency distribution.

---

## 7. Design-choice summary table (boosted with the "why")

| Design choice | Why | What breaks without it |
|---|---|---|
| Subword, not whole-word | Bounded vocab + graceful OOV handling via composition | Unbounded vocab, OOV → `[UNK]`, no morphological sharing |
| Subword, not character | Tokens carry real meaning; sequence stays short | Sequence length inflates ~3-4x → attention cost inflates ~10-16x (quadratic) |
| 30,522 vocab size | Empirical sweet spot: common English words mostly whole, budget not wasted | Too small → excess fragmentation; too big → bloated table, undertrained rare rows |
| WordPiece likelihood score (not raw BPE frequency) | Prioritizes merges that most reduce corpus "surprise," not just raw co-occurrence | Using BPE's rule instead gives a systematically different, not identical, vocabulary |
| `[CLS]` always first | Gives the model a dedicated, purpose-trained slot to aggregate sentence meaning | You'd fall back to ad hoc pooling (e.g. mean) not specifically optimized for this |
| `[SEP]` between segments | Explicit, unambiguous boundary signal (redundant with segment IDs, by design) | Model must infer boundaries from content alone — strictly harder |
| `[MASK]` only at pre-training, with 80/10/10 mixing | Prevents the model overfitting to a token it will never see again at inference | Train/inference mismatch — degraded representations at non-masked positions |
| 512 token limit | Tied to a fixed, learned positional embedding table (Ch. 3) | Documents longer than 512 tokens get silently truncated or error out |

---

## 8. Diagnostics — misconceptions to pre-empt

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "WordPiece just merges the most frequent pair each round, same as BPE" | WordPiece scores by likelihood gain (count(a,b) / (count(a)·count(b))), which can and does pick a different merge order than raw-frequency BPE | They're related but distinct algorithms; GPT-family models use BPE, BERT uses WordPiece |
| "`[UNK]` never happens with WordPiece, ever" | Individual *characters* outside the vocabulary's character set (rare Unicode, some emoji/scripts) can still fail to decompose and fall back to `[UNK]` | WordPiece drastically reduces `[UNK]` frequency but doesn't mathematically guarantee zero |
| "More subword splitting = worse performance, always" | Splitting itself isn't inherently bad — the cost is *fragmenting a domain-important concept into weakly-trained generic pieces*. A well-trodden root/suffix split (`hospital` + `##ization`) is nearly free | The problem is domain-frequency mismatch, not subword splitting in general |
| "Padding tokens are automatically ignored by the model" | The `[PAD]` token ID by itself does nothing special in the attention computation — it's the separate attention mask array that excludes padded positions | Forgetting to pass/apply the attention mask lets the model attend into padding |
| "The 512 limit is a soft guideline you can usually push past" | It's tied to a fixed-size, trained lookup table of positional embeddings — there is no position 512 to look up | It's a hard architectural ceiling, not a tunable hyperparameter |

---

## 9. Q&A practice set (self-test — answers below the line)

**Q1 (easy).** In one sentence, why can't word-level tokenization ever fully solve the OOV problem, no matter how large you make the vocabulary?

**Q2 (easy).** Why does character-level tokenization make Transformer training meaningfully more expensive, not just "produce longer sequences"?

**Q3 (medium).** What does the `##` prefix in a WordPiece token actually encode, and why does it matter for reconstructing the original text?

**Q4 (medium — calculation).** Pair X has count(X)=8 total co-occurrences, with count(a)=40 and count(b)=50 individually. Pair Y has count(Y)=5 total co-occurrences, with count(a)=6 and count(b)=5 individually. Under WordPiece's scoring rule, which pair merges first? Show the scores.

**Q5 (medium).** Why is `[MASK]` never present at inference time, and how does BERT's training procedure specifically compensate for that gap?

**Q6 (hard).** A teammate loads `bert-base-cased`'s tokenizer but accidentally loads `bert-base-uncased`'s model weights. What specifically goes wrong, mechanically?

**Q7 (hard).** Why does WordPiece's likelihood-based scoring rule tend to delay merging very common character pairs like ("t","h") compared to raw-frequency BPE?

**Q8 (hard — spot the bug).** A production pipeline batches sequences together, pads them to the same length, and passes the padded token IDs into BERT — but the engineer forgot to also construct and pass the attention mask. Describe what goes wrong and why the bug might not be obvious from output quality alone on short sequences.

---
---

### Answers

**A1.** Even an arbitrarily large fixed vocabulary can only ever include words that existed (and were frequent enough to include) at the time the vocabulary was built — language continuously produces new words (names, slang, technical terms, typos) faster than any static list can be updated, so some future input will always fall outside it.

**A2.** Self-attention cost scales quadratically with sequence length, since every token computes a compatibility score against every other token ($QK^T$). Character-level tokenization roughly triples-to-quadruples sequence length versus word-level, which — because of the quadratic scaling — increases attention compute by roughly an order of magnitude, not just proportionally to the length increase.

**A3.** `##` marks that a token is a continuation of the previous token rather than the start of a new word — it's purely a detokenization/boundary marker. Without it, `["un", "believe", "able"]` would be ambiguous about whether these are three separate words or fragments of one word; `["un", "##believe", "##able"]` unambiguously reconstructs to "unbelievable" with no spaces inserted between the pieces.

**A4.** WordPiece score = count(pair) / (count(a) × count(b)).
- Pair X: 8 / (40 × 50) = 8/2000 = **0.004**
- Pair Y: 5 / (6 × 5) = 5/30 ≈ **0.167**

Pair Y merges first — despite having a *lower raw co-occurrence count* (5 vs 8), its much smaller individual symbol frequencies make it score far higher under the likelihood-gain formula. This is precisely the BPE-vs-WordPiece divergence: a pure frequency-based approach (BPE) would have picked X.

**A5.** `[MASK]` is a training device introduced specifically to create prediction targets during pre-training; no real downstream input (a support ticket, a search query, a fine-tuning dataset row) will ever literally contain the string `[MASK]`, so the token has no reason to appear post-training. BERT compensates by not making `[MASK]` the *only* thing that ever occupies a "to-be-predicted" position during training: of the 15% of tokens selected for prediction, only 80% are replaced with the literal `[MASK]` token, 10% are replaced with a random other word, and 10% are left unchanged — forcing the model to build genuinely reliable contextual representations at every position, since it can never be fully certain, at training time, whether the token in front of it is "trustworthy" or not.

**A6.** The two checkpoints have different vocabulary files (cased preserves capitalization as meaningful, so its ID-to-token mapping differs from uncased's), so token ID 1996 under the cased vocabulary may map to a completely different string than ID 1996 under the uncased vocabulary. Feeding cased-tokenizer IDs into uncased-model weights means the model's embedding lookup pulls the *wrong* learned vector for each ID — effectively feeding it a scrambled, meaningless sequence, silently, with no error thrown. Output degrades but doesn't necessarily crash, making this a nasty bug to catch without specifically checking that tokenizer and model come from the same checkpoint.

**A7.** Very common characters like "t" and "h" already appear frequently on their own, in many other pairings throughout the corpus — count(t) and count(h) are both large. Dividing count(t,h) by that large product shrinks the score, because merging them buys comparatively little: both are already well-represented, useful, standalone symbols. A rarer pair whose two halves rarely occur *except* together (small individual counts) scores much higher, because merging them captures nearly all of the co-occurrence's "value" without needing to preserve their (rare, low-value) separate usages.

**A8.** Without an attention mask, self-attention computes compatibility scores between every position — including real tokens attending to `[PAD]` positions — and includes those pad positions' (meaningless) vectors in the weighted average that forms every other token's contextual representation, including `[CLS]`. On short sequences within a batch dominated by long ones, this might be barely noticeable — a handful of padding positions contribute a small, diffuse noise to the softmax-weighted average and may not visibly move metrics much. The bug becomes serious as padding proportion grows (short sequences batched with much longer ones) or at the tails of a distribution, and it's exactly the kind of silent, gradual-degradation bug that's hard to catch from aggregate accuracy numbers alone — you have to specifically test with mixed-length batches and heavy padding to expose it.

---

## 10. Quick recap card (last-minute review)

- **Word-level**: unbounded vocab, OOV fails hard, no morphological sharing.
- **Char-level**: tiny vocab, zero OOV, but sequence length explosion → quadratic attention cost blowup, and tokens carry no inherent meaning.
- **WordPiece**: fixed ~30k budget, common words whole, rare words decompose into meaningful, previously-seen fragments.
- **WordPiece ≠ BPE**: WordPiece scores merges by likelihood gain `count(a,b)/(count(a)·count(b))`, not raw frequency — can pick a different merge order than BPE (used by GPT-family models).
- **Four special tokens, four distinct jobs**: `[CLS]` aggregates sentence meaning (trained for it), `[SEP]` marks segment boundaries, `[MASK]` is pre-training-only (with 80/10/10 mixing to avoid a train/inference mismatch), `[PAD]` is inert filler that *requires* an attention mask to actually be ignored.
- **512-token limit**: a hard architectural ceiling from the fixed-size positional embedding table, not a soft guideline.
- **Domain mismatch**: standard BERT fragments domain-specific vocabulary (medical, legal, code) into weak generic pieces — fix at the tokenizer level (retrain WordPiece on domain text), not just via fine-tuning.

*(Chapter 3 — The Three Embeddings — picks up here: how token IDs, position indices, and segment IDs each become a 768-dim vector, and how they combine before the first Transformer layer.)*
