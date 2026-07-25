# Module 2 — Pretraining Objectives (Master Notes, Expanded)

## 0. The big picture — what "pretraining objective" even means

Before any fine-tuning/RLHF, a model is trained on a huge unlabeled text corpus with a **self-supervised objective** — the "labels" are derived automatically from the text itself (no humans annotate anything). The objective you pick determines:
- What architecture fits naturally (encoder-only, decoder-only, encoder-decoder)
- What the model gets good at downstream (generation vs understanding/classification)
- How you can use it later (can it generate text left-to-right? can it look at full context?)

**The three families you must know cold**: Causal LM (CLM), Masked LM (MLM), and denoising/span-corruption objectives (Prefix LM, T5-style span corruption, UL2).

---

## 1. Causal Language Modeling (CLM) — the GPT family

### Core idea, in plain words
Predict the **next token**, given only the tokens that came *before* it — never look ahead. This is autoregressive: generate one token, feed it back in, predict the next, repeat.

### The formula, explained term by term
The training objective is to maximize the probability the model assigns to the actual next token, for every position in every sequence:
```
L_CLM = - Σ (over all positions t) log P(x_t | x_1, x_2, ..., x_{t-1})
```
- `x_t` = the actual token at position t (ground truth from the corpus).
- `P(x_t | x_1...x_{t-1})` = the probability the model assigns to that correct token, given everything before it.
- `log(...)` then negated = **cross-entropy loss** — if the model assigns high probability (close to 1) to the correct token, `log(P)` is close to 0 (low loss); if it assigns low probability, `log(P)` is a large negative number, so `-log(P)` is a large positive loss. This is the standard classification loss, just applied at every position simultaneously, with "classes" = vocabulary tokens.
- Sum across all positions in the sequence, and average over the whole batch/corpus.

### Numerical example
Say the model sees "The cat sat on the ___" and the true next word is "mat". Suppose the model's softmax output gives:
- P("mat") = 0.4
- P("floor") = 0.3
- P("chair") = 0.1
- (rest split among other tokens)

Loss for this one position = `-log(0.4)` = **0.916** (natural log). If instead the model had been very confident and correct, say P("mat") = 0.9, loss = `-log(0.9)` = **0.105** — much lower. If the model was confidently *wrong*, say it gave "mat" only P=0.01, loss = `-log(0.01)` = **4.6** — a big penalty. This is why cross-entropy punishes confident wrong answers much more harshly than uncertain ones — the loss curve is steep near 0.

### The masking mechanism (architectural side)
CLM is implemented via a **causal attention mask** — a triangular mask where position `t` can only attend to positions `≤ t`, never to future positions. This is what makes a Transformer "decoder-only": every layer respects this mask, so information from the future token literally cannot leak backward during training (which would make the task trivial/cheating).

### Why CLM dominates modern LLMs
- Training signal is "free" and dense — every single token in every sequence is a training example (predict token 2 from token 1, predict token 3 from tokens 1-2, etc.) — one pass over N tokens gives N training signals.
- The objective matches the actual use case: text *generation*, one token at a time, exactly how you'll use the model at inference.
- Scales beautifully — this is the objective behind every scaling law study (GPT-2, GPT-3, Chinchilla) because it's simple, stable, and doesn't need special data preparation (no artificial masking/corruption step).

### Where CLM is used standalone in practice
**GPT-2, GPT-3, GPT-4, Llama family, PaLM, Claude, Mistral** — essentially every modern "chat" LLM you interact with is CLM-pretrained at its core (decoder-only architecture).

---

## 2. Masked Language Modeling (MLM) — BERT

### Core idea, in plain words
Instead of predicting the next token from the past only, **randomly hide (mask) some tokens in the middle of the sequence, and ask the model to predict them using context from *both sides*** (left and right). This is why BERT is "bidirectional."

### The recipe (BERT's exact procedure — this is a common interview detail)
Given a sequence, randomly select **15%** of tokens. For each selected token:
- **80% of the time**: replace it with a special `[MASK]` token.
- **10% of the time**: replace it with a random other token from the vocab.
- **10% of the time**: leave it unchanged.

Then the model must predict the *original* token at each of these 15% selected positions, using full bidirectional context.

### Numerical worked example
Sentence: "The cat sat on the mat" (6 tokens). 15% of 6 ≈ 1 token selected (in practice this is done over huge batches so percentages average out; for a single short sentence, say we select "sat"):
- 80% chance: sentence becomes "The cat [MASK] on the mat" → model must predict "sat" using both "The cat ___" (left context) and "___ on the mat" (right context).
- 10% chance: sentence becomes "The cat zebra on the mat" (random wrong word inserted) → model must still learn to predict "sat" as the correct answer despite the corrupted input — this forces the model not to *blindly trust* the input token is always correct, which helps because at fine-tuning/inference time there's no `[MASK]` token at all, so the model needs to build good representations for every real token, not just for `[MASK]` placeholders.
- 10% chance: sentence stays "The cat sat on the mat" unchanged, but the model is *still asked* to predict "sat" at that position — this keeps the model from getting lazy and assuming "if it's not `[MASK]`, don't bother computing anything useful there."

### Why the 80/10/10 split exists (the actual reasoning, worth memorizing)
If you *always* used `[MASK]`, the model would only ever need to be good at predicting masked positions — but at fine-tuning time, `[MASK]` tokens never appear in real downstream input, creating a **train/inference mismatch**. The 10%/10% "noise" forces the model to build robust contextual representations for *every* token position, not just artificially masked ones — closing that mismatch gap.

### The loss formula
Only compute loss on the masked (selected) positions, not the whole sequence:
```
L_MLM = - Σ (over masked positions i) log P(x_i | x_context)
```
Where `x_context` = the entire surrounding sequence (both directions), unlike CLM's left-only context.

### Practical downside vs CLM (a favorite interview gotcha)
MLM only gets a training signal from ~15% of tokens per pass (vs. CLM's 100%), so MLM pretraining is **less sample-efficient per token seen** — you need more passes over data/more compute to extract the same amount of learning signal. This is one reason later encoder architectures (like ELECTRA, see below) tried to fix this specific inefficiency.

### Where MLM is used standalone in practice
**BERT, RoBERTa** (RoBERTa removes the 10% "leave unchanged"/uses dynamic masking each epoch instead of static masking, but same core idea), **DistilBERT, ALBERT**. Used for encoder-only models — good at *understanding* tasks (classification, NER, embeddings, search/retrieval) but **cannot natively generate text** left-to-right, since there's no autoregressive mechanism — this is the key practical limitation to state in interviews.

---

## 3. Prefix LM and Span Corruption (T5-style) — the encoder-decoder middle ground

### Core idea, in plain words
CLM sees only left context (good for generation, bad for full-context understanding). MLM sees full bidirectional context but can't generate (good for understanding, bad for generation). **Prefix LM and span corruption try to get both**: bidirectional attention over an input chunk, then autoregressive generation for an output chunk.

### Prefix LM
Split each training sequence into two parts: a **prefix** (context) and a **target** (continuation). The prefix gets **full bidirectional attention** (like MLM — every prefix token can see every other prefix token, both directions). The target is generated **autoregressively** (like CLM — each target token only sees prefix + previous target tokens, causal mask applies only within the target region).

**Practical example**: prefix = "Translate English to French: The cat is black." → target = "Le chat est noir." During training, the model can freely look back and forth within "Translate English to French: The cat is black." (full bidirectional understanding of the instruction+input), but must generate "Le chat est noir." one token at a time, causally.

### Span corruption (T5's actual pretraining objective — different from vanilla Prefix LM)
Instead of a natural prefix/target split, **randomly corrupt contiguous spans of the input** (not single tokens like MLM), replace each corrupted span with a single unique sentinel token (e.g. `<extra_id_0>`, `<extra_id_1>`...), and train the model to **generate the missing spans** (not the whole reconstructed sentence — just the corrupted parts, concatenated) as the decoder target.

### Numerical worked example
Original: "The cat sat on the mat and looked happy"

Corrupt two spans: "sat on" and "and looked" → each replaced by one sentinel token:
```
Input (encoder sees):  The cat <extra_id_0> the mat <extra_id_1> happy
Target (decoder must generate):  <extra_id_0> sat on <extra_id_1> and looked <extra_id_2>
```
(The final `<extra_id_2>` marks end-of-target — a convention to know.)

**Why this is more efficient than MLM's single-token masking**: corrupting spans (average span length ~3 tokens in T5's original setup) at a 15% overall corruption rate means fewer, longer gaps rather than many scattered single-token gaps — this shortens the target sequence the decoder must produce (only the corrupted spans, not the whole sentence), making training compute cheaper per example while still giving a rich bidirectional-encoding + autoregressive-decoding signal.

### Where Prefix LM / span corruption is used standalone in practice
**T5** (span corruption is literally T5's pretraining objective — "Text-to-Text Transfer Transformer," everything, including classification, is cast as generating target text). **BART** uses a related but distinct denoising objective (text infilling + sentence permutation, span corruption is one of BART's several noise functions).

---

## 4. UL2 (Unifying Language Learning) — worth a one-paragraph mention

Google's UL2 paper observed that CLM, MLM-style span corruption, and Prefix LM are all special cases of a general "denoising" framework, differing mainly in (a) how much of the input is corrupted, and (b) whether spans are short/scattered or long/contiguous. UL2 trains on a **mixture of denoising objectives simultaneously** (they call these "R-denoising" = regular short-span corruption like T5, "S-denoising" = sequential/prefix-LM-style, "X-denoising" = extreme corruption of long spans, closer to CLM-like generation from little context), tagged with special mode tokens, so a single model learns to be good at multiple objective "modes" and you pick the mode at inference time depending on the task. **Interview-level takeaway**: this is evidence the field converged on "these objectives aren't fundamentally different tasks — they're points on a corruption-rate/span-length spectrum," which is a good higher-level insight to voice if asked "how do these objectives relate to each other."

---

## 5. Perplexity — the standard pretraining metric

### The formula, explained term by term
```
Perplexity = exp( (1/N) × Σ -log P(x_t | context) )  =  exp(average cross-entropy loss)
```
In plain words: perplexity is just **e raised to the average per-token loss**. It converts the abstract "loss" number into something with a more intuitive interpretation: "on average, how many equally-likely choices was the model effectively choosing among, at each position?"

### Numerical example
If average cross-entropy loss = 2.0 (nats), perplexity = `exp(2.0)` ≈ **7.39** — meaning the model's uncertainty at each token is roughly as if it were choosing uniformly among ~7.4 options. Lower perplexity = more confident/accurate model. A perfect model that always assigns probability 1.0 to the correct token has loss=0, so perplexity = `exp(0)` = **1** (no uncertainty at all). A model just guessing uniformly among a 50,000-token vocabulary would have perplexity near 50,000 (extremely bad).

### The key limitation (a very common interview trap)
**Perplexity measures how well the model predicts held-out text from the same distribution as training data — it does NOT directly measure downstream task performance** (reasoning, instruction-following, factual accuracy). Two models can have very similar perplexity but noticeably different capability on actual tasks, especially once you factor in fine-tuning/alignment — this disconnect is exactly why benchmarks like MMLU exist alongside perplexity, and it's the same underlying issue behind the "emergent abilities" debate you'll cover in Module 3.

---

## 6. Pretraining data practicalities (common system-design-flavored follow-ups)

**Deduplication**: near-duplicate documents/passages in the training corpus cause the model to overfit/memorize repeated content and waste compute re-learning the same thing — standard practice (used in GPT-3, Llama papers) is aggressive fuzzy deduplication (e.g. MinHash/LSH-based near-duplicate detection) before training.

**Data mixing / curriculum**: pretraining corpora blend multiple sources (web text, books, code, Wikipedia, academic papers) at deliberately chosen **mixture ratios**, not just "everything available in whatever proportion it naturally occurs" — e.g. code and Wikipedia are typically *upweighted* relative to their natural frequency in raw web crawl, because they're higher-quality/information-dense signal. Some training runs also use a **curriculum** (change the data mixture over the course of training, e.g. more code/reasoning-heavy data toward the end) — this is an active research area, not fully settled.

**Tokens-per-parameter ratio**: directly connects to Module 3's Chinchilla scaling laws — the ratio of training tokens (D) to model parameters (N) that a fixed compute budget should be split into is not "as many params as possible," but a specific compute-optimal ratio (~20 tokens per parameter, per Chinchilla) — flag this connection now, full derivation comes next module.

---

## 7. Side-by-side summary table (memorize this cold)

| | CLM | MLM | Prefix LM / Span Corruption |
|---|---|---|---|
| Context direction | Left-only (causal) | Full bidirectional | Bidirectional on prefix/input, causal on target/output |
| Training signal density | 100% of tokens | ~15% of tokens (masked positions only) | Only corrupted spans are the target (compute-efficient) |
| Architecture | Decoder-only | Encoder-only | Encoder-decoder |
| Natively generates text? | Yes | No | Yes |
| Best at | Open-ended generation | Understanding/classification/embeddings | Text-to-text tasks (translation, summarization) |
| Standalone models | GPT-2/3/4, Llama, PaLM, Claude | BERT, RoBERTa, ALBERT | T5, BART |

---

## 8. Quick-fire Q&A (self-test)

**Q: Write the CLM loss formula and explain each term in one sentence.**
A: `L = -Σ log P(x_t | x_1...x_{t-1})` — sum of negative log-probabilities the model assigns to the true next token at every position, using only left context; this is cross-entropy loss applied autoregressively.

**Q: Why does BERT use an 80/10/10 masking split instead of always using [MASK]?**
A: Always using `[MASK]` would create a train/inference mismatch, since `[MASK]` never appears in real downstream input — the 10% random-token and 10% unchanged cases force the model to build good representations for every real token, not just placeholder positions.

**Q: Why is MLM less sample-efficient than CLM per token processed?**
A: MLM only computes loss on the ~15% masked positions per pass, while CLM gets a training signal from every single token in the sequence — so CLM extracts more learning signal per token seen.

**Q: What's the actual difference between T5's span corruption and BERT's MLM masking?**
A: BERT masks scattered single tokens and predicts them in-place with an encoder-only model; T5 corrupts contiguous spans, replaces each with one sentinel token, and generates only the missing spans (not the whole sequence) autoregressively via a full encoder-decoder — this makes the target sequence shorter and the objective inherently generative.

**Q: What does perplexity of 7.4 actually mean in plain language?**
A: On average, at each token position, the model is about as uncertain as if it were randomly choosing among ~7.4 equally likely options — it's `exp(average cross-entropy loss)`.

**Q: Does lower perplexity guarantee better downstream task performance?**
A: No — perplexity only measures next-token prediction fit to the training-like distribution; it doesn't directly capture reasoning, instruction-following, or factual accuracy, which is why benchmark evaluation exists as a separate, necessary measurement.

**Q: Can a BERT-style model generate open-ended text the way GPT can? Why or why not?**
A: No — BERT's bidirectional MLM training and encoder-only architecture never learn an autoregressive left-to-right generation mechanism; there's no causal mask or decoder to sample tokens one at a time.

---
*End of Module 2 (expanded). Next: Module 3 — Scaling Laws & Emergent Abilities (Kaplan vs Chinchilla, compute-optimal training, the emergent-abilities debate).*
