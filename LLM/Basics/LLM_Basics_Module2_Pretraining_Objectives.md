# Module 2 — Pretraining Objectives (Master Notes, Expanded)

> **Editor's note on this pass**: Every word of your original notes is preserved below, in its original order. Nothing has been cut or shortened. All additions are clearly tagged with **📌 Added Explanation**, **🧮 Numerical Example**, or **❓ Interview Q&A** so you can see at a glance what's new vs. original. New material is placed directly under the section it expands.

---

## 0. The big picture — what "pretraining objective" even means

Before any fine-tuning/RLHF, a model is trained on a huge unlabeled text corpus with a **self-supervised objective** — the "labels" are derived automatically from the text itself (no humans annotate anything). The objective you pick determines:
- What architecture fits naturally (encoder-only, decoder-only, encoder-decoder)
- What the model gets good at downstream (generation vs understanding/classification)
- How you can use it later (can it generate text left-to-right? can it look at full context?)

**The three families you must know cold**: Causal LM (CLM), Masked LM (MLM), and denoising/span-corruption objectives (Prefix LM, T5-style span corruption, UL2).

### 📌 Added Explanation: "self-supervised" vs "supervised" vs "unsupervised" — a common point of confusion

People sometimes call pretraining "unsupervised," but that's technically imprecise, and interviewers will notice if you use the wrong word:

- **Supervised learning**: a human labels each example (e.g., "this image is a cat," "this email is spam"). Expensive to scale, but the label is exactly what you want the model to predict.
- **Unsupervised learning**: no labels at all — you're looking for structure in the data itself (e.g., clustering), with no specific "correct answer" being predicted.
- **Self-supervised learning**: no *human* labels, but the data itself is manipulated to automatically generate a "correct answer" — e.g., hide a word and the "label" is simply the word that was actually there in the raw text, which you already have for free. This is why pretraining can use effectively unlimited raw internet text (no human annotation bottleneck) while still training on a well-defined, correct-answer-having task (predict the next/masked/corrupted token) exactly like supervised learning does architecturally — it's supervised in *mechanism*, but the supervision is manufactured automatically from raw data, hence "self."

**In simple terms**: it's like being handed a book with random words blacked out — you don't need a teacher to tell you the "right answer," because you can just peek at an unredacted copy (the original text) to check yourself. The redaction pattern is invented, but the ground truth is real and free.

### 📌 Added Explanation: why the choice of objective determines the architecture (preview of what's coming)

The three families differ in **what a token is allowed to "see" when predicting another token** — this single design choice (the attention mask pattern) is what forces you into decoder-only, encoder-only, or encoder-decoder architecture:
- If every prediction can only look backward (causal mask) → naturally decoder-only (CLM).
- If every prediction can look both backward and forward, with no notion of generating a sequence left-to-right at all → naturally encoder-only (MLM).
- If some tokens (a "prefix"/"input") get to look both ways, but a separate set of tokens (a "target"/"output") must be generated causally, conditioned on that bidirectional input → naturally encoder-decoder (Prefix LM, span corruption).

Keep this "who can see whom" framing in mind — it's the thread connecting all three sections below.

---

## 1. Causal Language Modeling (CLM) — the GPT family

### Core idea, in plain words
Predict the **next token**, given only the tokens that came *before* it — never look ahead. This is autoregressive: generate one token, feed it back in, predict the next, repeat.

### 📌 Added Explanation: "autoregressive," unpacked
"Auto" = self, "regressive" = regressing/predicting based on prior values of the same series. An autoregressive model predicts the next element of a sequence using the sequence's own previous elements as input — exactly like a weather forecast that predicts tomorrow's temperature partly from today's and yesterday's temperatures, rather than from some entirely separate signal. For language models specifically: once the model generates a token, that token is appended to the input and fed back in for the *next* prediction step — this feedback loop (generate → append → re-feed → generate again) is what "autoregressive generation" refers to at inference time, and it's the same mechanism the model was trained to do at every position during pretraining.

### The formula, explained term by term
The training objective is to maximize the probability the model assigns to the actual next token, for every position in every sequence:
```
L_CLM = - Σ (over all positions t) log P(x_t | x_1, x_2, ..., x_{t-1})
```
- `x_t` = the actual token at position t (ground truth from the corpus).
- `P(x_t | x_1...x_{t-1})` = the probability the model assigns to that correct token, given everything before it.
- `log(...)` then negated = **cross-entropy loss** — if the model assigns high probability (close to 1) to the correct token, `log(P)` is close to 0 (low loss); if it assigns low probability, `log(P)` is a large negative number, so `-log(P)` is a large positive loss. This is the standard classification loss, just applied at every position simultaneously, with "classes" = vocabulary tokens.
- Sum across all positions in the sequence, and average over the whole batch/corpus.

### 📌 Added Explanation: where this formula comes from — the maximum likelihood derivation

This loss is derived directly from **maximum likelihood estimation (MLE)**, one of the most fundamental ideas in statistics. Here's the chain of reasoning, step by step:

1. **Start with the goal**: we want the model's probability distribution over the *entire training corpus* to be as high as possible — i.e., we want the model to consider the real, observed text as "likely," not "surprising."
2. **Chain rule of probability**: the joint probability of an entire sequence `x_1, x_2, ..., x_T` can always be decomposed (exactly, no approximation) as:
   ```
   P(x_1, ..., x_T) = P(x_1) × P(x_2|x_1) × P(x_3|x_1,x_2) × ... × P(x_T|x_1,...,x_{T-1})
   ```
   This is just the general chain rule — no independence assumption here (contrast with Unigram LM's tokenizer, which *did* assume independence; CLM does not).
3. **Take the log** of both sides. Logs turn products into sums (`log(a×b) = log(a) + log(b)`), which is purely a mathematical convenience — sums are far easier to differentiate and numerically stable than products of many small probabilities (multiplying thousands of numbers each less than 1 quickly underflows to zero in floating-point arithmetic; summing their logs does not):
   ```
   log P(x_1,...,x_T) = Σ_t log P(x_t | x_1,...,x_{t-1})
   ```
4. **Flip the sign** to turn "maximize log-likelihood" into "minimize negative log-likelihood" — purely a convention, since optimizers (gradient descent) are conventionally framed as *minimizing* a loss, not maximizing an objective. That negative sign is the entire origin of the "minus" in front of the formula.

**Why this is called "cross-entropy" specifically**: cross-entropy between a true distribution and a predicted distribution is defined as `-Σ P_true(x) log P_predicted(x)`. Since the true distribution here is a one-hot vector (100% probability on the actual observed token, 0% everywhere else), the sum collapses to just `-log P_predicted(true token)` — which is exactly the formula above, applied at every position. This is why "cross-entropy loss" and "negative log-likelihood" are used interchangeably in this context — they are the same quantity here, not just similar.

### Numerical example
Say the model sees "The cat sat on the ___" and the true next word is "mat". Suppose the model's softmax output gives:
- P("mat") = 0.4
- P("floor") = 0.3
- P("chair") = 0.1
- (rest split among other tokens)

Loss for this one position = `-log(0.4)` = **0.916** (natural log). If instead the model had been very confident and correct, say P("mat") = 0.9, loss = `-log(0.9)` = **0.105** — much lower. If the model was confidently *wrong*, say it gave "mat" only P=0.01, loss = `-log(0.01)` = **4.6** — a big penalty. This is why cross-entropy punishes confident wrong answers much more harshly than uncertain ones — the loss curve is steep near 0.

### 🧮 Numerical Example: full-sequence loss computation (extending the original example)

Let's compute the *sequence-level* loss for a toy 4-token continuation, to show how individual per-position losses in the formula above actually get summed in practice (as the formula's Σ specifies). Suppose the model is autoregressively scored on the sentence "the mat was warm" (4 tokens), and its assigned probabilities to the *true* token at each step are:

| Position | True token | Context seen | P(true token) | Loss = -log(P) |
|---|---|---|---|---|
| 1 | "the" | (nothing / start-of-sequence) | 0.25 | -log(0.25) = 1.386 |
| 2 | "mat" | "the" | 0.40 | -log(0.40) = 0.916 |
| 3 | "was" | "the mat" | 0.55 | -log(0.55) = 0.598 |
| 4 | "warm" | "the mat was" | 0.30 | -log(0.30) = 1.204 |

**Total sequence loss** = `Σ = 1.386 + 0.916 + 0.598 + 1.204 = 4.104`
**Average per-token loss** = `4.104 / 4 = 1.026`

This average-per-token number is exactly what later gets exponentiated into **perplexity** in Section 5 (`exp(1.026) ≈ 2.79` — meaning the model was, on average across this short sequence, about as uncertain as choosing among ~2.8 equally likely options per position). This ties the CLM loss formula directly to the perplexity metric covered later, using one consistent worked example.

### The masking mechanism (architectural side)
CLM is implemented via a **causal attention mask** — a triangular mask where position `t` can only attend to positions `≤ t`, never to future positions. This is what makes a Transformer "decoder-only": every layer respects this mask, so information from the future token literally cannot leak backward during training (which would make the task trivial/cheating).

### 📌 Added Explanation: what the causal mask literally looks like, and why "leaking" would make training trivial

Concretely, the causal mask is implemented as a matrix added to the raw attention scores *before* the softmax, with `0` in allowed positions and `-∞` (or a very large negative number) in disallowed (future) positions — because `softmax(-∞) = 0`, this forces the attention weight on any future position to become exactly zero, regardless of what the raw attention score would otherwise have been. For a 4-token sequence, the mask looks like:

```
        pos1  pos2  pos3  pos4
pos1 [   0,   -∞,   -∞,   -∞  ]   ← position 1 can only see itself
pos2 [   0,    0,   -∞,   -∞  ]   ← position 2 can see positions 1-2
pos3 [   0,    0,    0,   -∞  ]   ← position 3 can see positions 1-3
pos4 [   0,    0,    0,    0  ]   ← position 4 can see everything (1-4)
```

This upper-triangular `-∞` pattern is why it's often visualized as a triangle. **Why leakage would make training "trivial/cheating"**: if position 2, tasked with predicting token 3, could directly attend to the actual token 3 sitting right there in the input sequence, the model could achieve zero loss instantly by just copying the answer through the attention mechanism — it would never be forced to learn any actual predictive structure of language, and at inference time (where future tokens genuinely don't exist yet, since you're generating them one at a time) it would have learned a completely useless shortcut that doesn't generalize. The mask enforces that training-time difficulty exactly matches inference-time difficulty — a critical alignment between how the model is trained and how it's actually used.

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

### 📌 Added Explanation: an analogy for why bidirectional context helps "understanding" tasks specifically

Imagine you're handed a sentence with one word blacked out: "The trophy didn't fit in the suitcase because ___ was too big." To correctly fill in the blank ("it" referring to the trophy, not the suitcase — a classic Winograd-schema-style ambiguity), you genuinely need to read the *entire* sentence, including words that come *after* the blank, not just the words before it. A pure left-to-right (causal) model predicting this blank would only have "The trophy didn't fit in the suitcase because" to work with at that position — it hasn't been "shown" the disambiguating "was too big" yet, because in a strict autoregressive framing, you only ever predict forward. MLM's bidirectional context is precisely what lets a model resolve this kind of both-sides-needed ambiguity, which is why encoder-only bidirectional models tend to shine at tasks requiring deep, whole-sentence comprehension (classification, entity recognition, semantic similarity) rather than open-ended generation.

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

### 🧮 Numerical Example: scaling the 15%/80/10/10 split to a realistic batch size

The original example uses a 6-token sentence, where "15% ≈ 1 token" is a rounding approximation, as the notes themselves flag. Let's see the split behave properly at realistic scale — a batch of sequences totaling **10,000 tokens**:

1. **Select 15% for masking**: `10,000 × 0.15 = 1,500` tokens selected as prediction targets.
2. **Of those 1,500, apply the 80/10/10 split**:
   - `1,500 × 0.80 = 1,200` tokens replaced with `[MASK]`.
   - `1,500 × 0.10 = 150` tokens replaced with a random wrong token.
   - `1,500 × 0.10 = 150` tokens left unchanged (but still scored).
3. **Loss is computed on all 1,500 selected positions** (not just the 1,200 `[MASK]`'d ones) — the model has to predict the correct original token whether it saw `[MASK]`, a random substitute, or the correct token itself sitting unchanged in front of it.

This step-by-step breakdown makes concrete exactly how many tokens fall into each bucket at a scale where the percentages aren't rounding artifacts, and it reinforces that **loss is always computed on the full 15%**, not just the `[MASK]` subset — a detail that's easy to misstate under interview pressure.

### Why the 80/10/10 split exists (the actual reasoning, worth memorizing)
If you *always* used `[MASK]`, the model would only ever need to be good at predicting masked positions — but at fine-tuning time, `[MASK]` tokens never appear in real downstream input, creating a **train/inference mismatch**. The 10%/10% "noise" forces the model to build robust contextual representations for *every* token position, not just artificially masked ones — closing that mismatch gap.

### The loss formula
Only compute loss on the masked (selected) positions, not the whole sequence:
```
L_MLM = - Σ (over masked positions i) log P(x_i | x_context)
```
Where `x_context` = the entire surrounding sequence (both directions), unlike CLM's left-only context.

### 📌 Added Explanation: term-by-term comparison of `L_MLM` against `L_CLM`

Both formulas share the identical outer skeleton — a negative sum of log-probabilities of true tokens, i.e., cross-entropy — so it's worth being precise about the *two* differences, since interviewers often ask "aren't these the same formula?":

1. **Which positions are summed over**: `L_CLM` sums over *every* position `t` in the sequence (dense signal). `L_MLM` sums only over the *subset* `i` of positions that were randomly selected for masking (sparse signal, ~15%). This is the "training signal density" difference discussed later in Section 7.
2. **What `x_context` means in the conditioning**: in `L_CLM`, the condition is strictly `x_1, ..., x_{t-1}` (left-only, causal). In `L_MLM`, the condition `x_context` is the *entire* sequence surrounding position `i`, both left and right (with position `i` itself replaced by `[MASK]`/noise/unchanged per the 80/10/10 rule) — this is the bidirectional difference.

Both are still, fundamentally, cross-entropy loss — the *mechanism* of "penalize low probability on the true token" is identical; only *which positions get a loss term* and *what context is visible* differ.

### Practical downside vs CLM (a favorite interview gotcha)
MLM only gets a training signal from ~15% of tokens per pass (vs. CLM's 100%), so MLM pretraining is **less sample-efficient per token seen** — you need more passes over data/more compute to extract the same amount of learning signal. This is one reason later encoder architectures (like ELECTRA, see below) tried to fix this specific inefficiency.

### 📌 Added Explanation: how ELECTRA actually fixes this (since the notes mention it by name without detail)

ELECTRA replaces the "predict the masked word" task with a **"replaced token detection"** task: a small generator network (itself a tiny MLM) proposes plausible replacement tokens for masked positions, and then a separate, larger **discriminator** network must classify — for *every single token in the entire sequence*, not just the masked 15% — whether that token is "original" or "replaced by the generator." Because every position now contributes a loss term (a binary real-vs-replaced classification), ELECTRA restores 100%-of-tokens training signal density while keeping the bidirectional-context benefit of MLM, directly targeting the exact inefficiency called out in this section. **In simple terms**: instead of a fill-in-the-blank quiz where only the blanks are graded (BERT/MLM), ELECTRA is more like a proofreading task where you have to judge *every single word* in the passage as either "genuine" or "subtly swapped by an impostor" — every word gives you a graded answer, not just the ones that were blanked out.

### Where MLM is used standalone in practice
**BERT, RoBERTa** (RoBERTa removes the 10% "leave unchanged"/uses dynamic masking each epoch instead of static masking, but same core idea), **DistilBERT, ALBERT**. Used for encoder-only models — good at *understanding* tasks (classification, NER, embeddings, search/retrieval) but **cannot natively generate text** left-to-right, since there's no autoregressive mechanism — this is the key practical limitation to state in interviews.

> **⚠️ Flag (accuracy check)**: The notes state RoBERTa "removes the 10% leave unchanged." To be precise: RoBERTa's key changes vs. original BERT are (1) **dynamic masking** — the masking pattern is generated fresh each time a sequence is fed to the model (e.g., by duplicating the data multiple times with different random masks) rather than fixed once during preprocessing, and (2) removal of BERT's Next Sentence Prediction (NSP) auxiliary objective, plus larger batches/more data/longer training. Whether RoBERTa specifically altered the 80/10/10 masked-token substitution ratio itself, versus just making the *masking pattern* dynamic across epochs, is a finer implementation detail worth double-checking against the original RoBERTa paper if this exact point comes up in an interview — I don't want to assert the ratio change confidently without being certain.

---

## 3. Prefix LM and Span Corruption (T5-style) — the encoder-decoder middle ground

### Core idea, in plain words
CLM sees only left context (good for generation, bad for full-context understanding). MLM sees full bidirectional context but can't generate (good for understanding, bad for generation). **Prefix LM and span corruption try to get both**: bidirectional attention over an input chunk, then autoregressive generation for an output chunk.

### 📌 Added Explanation: why this is called a "middle ground" architecturally

Recall the "who can see whom" framing from Section 0: CLM = triangular causal mask over the *entire* sequence; MLM = fully open (no mask at all) attention over the *entire* sequence. Prefix LM/span corruption combine both patterns in **one single sequence**, split into two zones: the prefix zone uses the MLM-style fully-open mask (any prefix token attends to any other prefix token, both directions), while the target zone uses the CLM-style triangular mask, but *also* gets to attend fully back into the entire prefix zone (since the prefix is already fully known/fixed by the time generation starts — there's nothing to "leak" by looking at it). This hybrid mask pattern is the literal architectural mechanism that makes "bidirectional-in, causal-out" possible within a single attention mechanism (or, in T5's case, this is explicitly separated into two stacks — an encoder using full attention, and a decoder using causal self-attention plus cross-attention into the encoder's output).

### Prefix LM
Split each training sequence into two parts: a **prefix** (context) and a **target** (continuation). The prefix gets **full bidirectional attention** (like MLM — every prefix token can see every other prefix token, both directions). The target is generated **autoregressively** (like CLM — each target token only sees prefix + previous target tokens, causal mask applies only within the target region).

**Practical example**: prefix = "Translate English to French: The cat is black." → target = "Le chat est noir." During training, the model can freely look back and forth within "Translate English to French: The cat is black." (full bidirectional understanding of the instruction+input), but must generate "Le chat est noir." one token at a time, causally.

### Span corruption (T5's actual pretraining objective — different from vanilla Prefix LM)
Instead of a natural prefix/target split, **randomly corrupt contiguous spans of the input** (not single tokens like MLM), replace each corrupted span with a single unique sentinel token (e.g. `<extra_id_0>`, `<extra_id_1>`...), and train the model to **generate the missing spans** (not the whole reconstructed sentence — just the corrupted parts, concatenated) as the decoder target.

### 📌 Added Explanation: why span corruption ≠ Prefix LM, precisely

It's easy to conflate these two since both are "bidirectional encoder + causal decoder" architecturally, but the *data construction* is fundamentally different: **Prefix LM** takes a naturally contiguous chunk of real text and splits it at one arbitrary cut point into "everything before" (prefix) vs. "everything after" (target) — the prefix is a real, unbroken run of text. **Span corruption** instead *punches holes* at multiple scattered locations throughout the sequence, and the "target" is not a continuation of the text at all — it's a reconstruction of just the missing, scattered pieces, stitched together with sentinel tokens marking which hole each piece fills. In other words: Prefix LM's target answers "what comes next after this point," while span corruption's target answers "what was originally sitting in each of these specific holes I've cut out of the middle" — a meaningfully different task shape, even though both share the bidirectional-in/causal-out attention pattern.

### Numerical worked example
Original: "The cat sat on the mat and looked happy"

Corrupt two spans: "sat on" and "and looked" → each replaced by one sentinel token:
```
Input (encoder sees):  The cat <extra_id_0> the mat <extra_id_1> happy
Target (decoder must generate):  <extra_id_0> sat on <extra_id_1> and looked <extra_id_2>
```
(The final `<extra_id_2>` marks end-of-target — a convention to know.)

**Why this is more efficient than MLM's single-token masking**: corrupting spans (average span length ~3 tokens in T5's original setup) at a 15% overall corruption rate means fewer, longer gaps rather than many scattered single-token gaps — this shortens the target sequence the decoder must produce (only the corrupted spans, not the whole sentence), making training compute cheaper per example while still giving a rich bidirectional-encoding + autoregressive-decoding signal.

### 🧮 Numerical Example: quantifying the "fewer, longer gaps" compute saving

Take a 100-token input sequence with a fixed 15% corruption rate (`100 × 0.15 = 15` tokens to corrupt, matching T5's original recipe).

**Scenario A — BERT-style scattered single-token masking**: 15 separate single-token masks scattered throughout → the decoder-equivalent "target" (if you were to generate all 15 in an autoregressive framing) would need 15 separate generation steps, each with essentially no useful adjacent-token context from other masked positions (they're scattered and mostly independent).

**Scenario B — T5-style span corruption, average span length 3**: 15 tokens corrupted, but grouped into `15 / 3 = 5` contiguous spans of ~3 tokens each. The decoder target sequence is: `<extra_id_0> [3 tokens] <extra_id_1> [3 tokens] <extra_id_2> [3 tokens] <extra_id_3> [3 tokens] <extra_id_4> [3 tokens] <extra_id_5>` — that's `5 sentinels + 15 real tokens = 20 tokens` total target length.

**Compute comparison**: the decoder only ever needs to process/generate a **20-token target**, regardless of the fact that 15 tokens were corrupted somewhere across a 100-token input. Compare this to a hypothetical fully-generative reconstruction of the *entire* 100-token sequence (as a naive "regenerate everything" approach might require) — T5's approach is roughly **5x shorter** target sequence (20 vs. 100), which — recalling the O(n²) attention-cost discussion from Module 1 — translates to a much bigger, non-linear compute saving in the decoder's self-attention specifically, on top of the straightforward saving from simply not having to output tokens that were never corrupted in the first place.

### Where Prefix LM / span corruption is used standalone in practice
**T5** (span corruption is literally T5's pretraining objective — "Text-to-Text Transfer Transformer," everything, including classification, is cast as generating target text). **BART** uses a related but distinct denoising objective (text infilling + sentence permutation, span corruption is one of BART's several noise functions).

---

## 4. UL2 (Unifying Language Learning) — worth a one-paragraph mention

Google's UL2 paper observed that CLM, MLM-style span corruption, and Prefix LM are all special cases of a general "denoising" framework, differing mainly in (a) how much of the input is corrupted, and (b) whether spans are short/scattered or long/contiguous. UL2 trains on a **mixture of denoising objectives simultaneously** (they call these "R-denoising" = regular short-span corruption like T5, "S-denoising" = sequential/prefix-LM-style, "X-denoising" = extreme corruption of long spans, closer to CLM-like generation from little context), tagged with special mode tokens, so a single model learns to be good at multiple objective "modes" and you pick the mode at inference time depending on the task. **Interview-level takeaway**: this is evidence the field converged on "these objectives aren't fundamentally different tasks — they're points on a corruption-rate/span-length spectrum," which is a good higher-level insight to voice if asked "how do these objectives relate to each other."

### 📌 Added Explanation: a unifying "two-dial" mental model for all objectives in this module

Here's a compact way to hold the entire module in your head at once, useful for framing a strong interview answer to "how do CLM, MLM, and span corruption relate?": every objective covered so far can be placed on a 2D grid with two dials —

- **Dial 1 — Corruption rate**: what fraction of the input is hidden/corrupted? (CLM: effectively corrupts "everything after the current position," which -- viewed as a single training example -- is a very high, structured corruption rate applied consistently left-to-right; MLM: ~15%; span corruption: ~15% but grouped into spans.)
- **Dial 2 — Span contiguity/length**: is the corrupted material scattered single tokens, or long contiguous chunks? (MLM: scattered singles; span corruption: short contiguous spans; Prefix LM/CLM-as-extreme-case: one single very long contiguous "span" — literally "everything from some cut point to the end.")

UL2's R/S/X-denoising modes are literally named for *where on these two dials* each mode sits (Regular = T5-like moderate corruption/moderate spans; Sequential = prefix-style single long contiguous target; eXtreme = very high corruption of very long spans, pushing toward "generate a lot from very little context," which behaviorally resembles open-ended CLM generation). This "two-dial" framing is exactly the kind of synthesis that distinguishes a strong senior-level interview answer from a rote listing of three separate objectives.

---

## 5. Perplexity — the standard pretraining metric

### The formula, explained term by term
```
Perplexity = exp( (1/N) × Σ -log P(x_t | context) )  =  exp(average cross-entropy loss)
```
In plain words: perplexity is just **e raised to the average per-token loss**. It converts the abstract "loss" number into something with a more intuitive interpretation: "on average, how many equally-likely choices was the model effectively choosing among, at each position?"

### 📌 Added Explanation: why exponentiating the average loss gives you "effective number of choices"

This is worth deriving carefully, since "why exp specifically, and why does that mean 'number of choices'" is a genuinely common point of confusion.

Consider the simplest possible case: a model that assigns *equal* probability to exactly `k` options and zero to everything else (a uniform distribution over `k` choices) — e.g., a fair k-sided die. Its probability of the correct outcome, every time, is `P = 1/k`. Plug this into the loss formula: `loss = -log(1/k) = log(k)`. Now exponentiate: `perplexity = exp(log(k)) = k`. 

**This is the key result**: for a perfectly uniform guess among `k` options, perplexity comes out to *exactly* `k` — this is not a coincidence or approximation, it's the algebraic identity `exp(log(k)) = k` falling directly out of the definitions. So when a real model (which isn't uniform, but some mix of confident-right and confident-wrong across different tokens) gets an average loss whose `exp(...)` equals, say, 7.4, the *interpretation* "as if choosing uniformly among 7.4 options" is not a loose metaphor — it's the literal mathematical inverse of the uniform-distribution case just derived. Perplexity is, precisely, "the equivalent branching factor" of the model's uncertainty.

**In simple terms**: if a friend says "I was perplexed, there were basically 7 things it could have been," that's exactly the intuition — perplexity numerically answers "how many roughly-equally-plausible options did the model seem to be juggling, on average, at each position," even though in reality the model's actual distribution is almost never perfectly uniform over exactly that many tokens — it's an *effective*, not literal, count.

### Numerical example
If average cross-entropy loss = 2.0 (nats), perplexity = `exp(2.0)` ≈ **7.39** — meaning the model's uncertainty at each token is roughly as if it were choosing uniformly among ~7.4 options. Lower perplexity = more confident/accurate model. A perfect model that always assigns probability 1.0 to the correct token has loss=0, so perplexity = `exp(0)` = **1** (no uncertainty at all). A model just guessing uniformly among a 50,000-token vocabulary would have perplexity near 50,000 (extremely bad).

### 🧮 Numerical Example: computing perplexity from the Section 1 worked example

Recall the 4-token CLM sequence loss worked out earlier ("the mat was warm"): total loss = 4.104, average per-token loss = 1.026.

**Perplexity = `exp(1.026)`**. Computing this: `exp(1.026) ≈ 2.79`.

**Interpretation**: across those 4 tokens, the model behaved, on average, as if it were choosing uniformly among about **2.8 options** per position — a reasonably confident model for this short toy example (compare: a model with zero useful signal, guessing uniformly across, say, a 30,000-token vocabulary, would show perplexity near 30,000 — a difference of four orders of magnitude, illustrating how dramatically perplexity separates a well-trained model from a untrained/random one). This also demonstrates concretely how the Section 1 loss computation and the Section 5 perplexity metric are the *same underlying quantity*, just presented on two different scales (log-scale loss vs. exponentiated "effective choices" scale) — exactly the connection flagged in the numerical example under Section 1.

### The key limitation (a very common interview trap)
**Perplexity measures how well the model predicts held-out text from the same distribution as training data — it does NOT directly measure downstream task performance** (reasoning, instruction-following, factual accuracy). Two models can have very similar perplexity but noticeably different capability on actual tasks, especially once you factor in fine-tuning/alignment — this disconnect is exactly why benchmarks like MMLU exist alongside perplexity, and it's the same underlying issue behind the "emergent abilities" debate you'll cover in Module 3.

### 📌 Added Explanation: a concrete scenario illustrating the perplexity/capability disconnect

Suppose Model A and Model B have nearly identical perplexity (say, 8.1 vs. 8.3) on a held-out slice of web text. Model A was trained purely on raw web crawl; Model B was trained on the same amount of data, but with a higher proportion of code, math, and structured reasoning text mixed in (recall the "data mixing" discussion in Section 6). Both models are, on average, similarly good at "predict the next plausible word in typical web text" — that's what perplexity is measuring, and typical web text is dominated by ordinary conversational/narrative language, not by multi-step logical reasoning chains. But Model B might dramatically outperform Model A on a benchmark like GSM8K (grade-school math word problems) or MMLU (broad academic knowledge), simply because its training distribution better matches the *skills* those benchmarks probe — a skill difference that a generic "next word in web text" perplexity score is not designed to detect at all, since most of any web-text corpus isn't testing multi-step reasoning in the first place. This is precisely why serious model evaluation always reports task-specific benchmark scores *alongside* (never instead of) perplexity.

---

## 6. Pretraining data practicalities (common system-design-flavored follow-ups)

**Deduplication**: near-duplicate documents/passages in the training corpus cause the model to overfit/memorize repeated content and waste compute re-learning the same thing — standard practice (used in GPT-3, Llama papers) is aggressive fuzzy deduplication (e.g. MinHash/LSH-based near-duplicate detection) before training.

### 📌 Added Explanation: how MinHash/LSH deduplication actually works, briefly

You cannot practically compare every document to every other document for near-duplicate detection — with billions of documents, that's a quadratic (again, O(n²)-style) comparison problem, just like the attention-cost issue from Module 1, but now over documents instead of tokens. **MinHash** is a technique for compressing a document into a small, fixed-size "fingerprint" (a set of hash values) such that two documents with high textual overlap will very likely produce *similar* fingerprints, while dissimilar documents produce very different fingerprints — crucially, this fingerprint comparison is vastly cheaper than comparing full documents. **LSH (Locality-Sensitive Hashing)** then buckets documents by fingerprint similarity so you only need to compare documents *within the same bucket* for true near-duplicates, rather than against the entire corpus — turning an infeasible all-pairs comparison into a feasible bucketed one. **In simple terms**: instead of reading every pair of essays in a huge stack side-by-side to spot plagiarism (infeasible at scale), you compute a short "summary fingerprint" for each essay, group essays with similar fingerprints together, and only closely compare essays that already landed in the same group.

**Data mixing / curriculum**: pretraining corpora blend multiple sources (web text, books, code, Wikipedia, academic papers) at deliberately chosen **mixture ratios**, not just "everything available in whatever proportion it naturally occurs" — e.g. code and Wikipedia are typically *upweighted* relative to their natural frequency in raw web crawl, because they're higher-quality/information-dense signal. Some training runs also use a **curriculum** (change the data mixture over the course of training, e.g. more code/reasoning-heavy data toward the end) — this is an active research area, not fully settled.

### 🧮 Numerical Example: what "upweighting" a data source concretely means

Suppose raw web crawl naturally contains code at only **3%** of total bytes (a realistic rough figure for unfiltered web crawl), but a training team decides code should make up **15%** of the actual training mixture, because code teaches structured, logically-consistent reasoning patterns useful even for non-code tasks. To hit that 15% target from a 3%-natural-frequency source without simply "having more code exist," the team **upsamples** it — i.e., the same code documents are seen by the model multiple times per pass over the full corpus (roughly `15% / 3% = 5x` as often as their natural frequency would imply), while other, over-represented low-value sources (e.g., low-quality boilerplate web pages) are correspondingly downweighted/subsampled. This is a direct, deliberate override of "whatever the internet naturally contains in whatever proportion" — a system design choice with measurable downstream effects on model capability, not an emergent accident of scraping.

**Tokens-per-parameter ratio**: directly connects to Module 3's Chinchilla scaling laws — the ratio of training tokens (D) to model parameters (N) that a fixed compute budget should be split into is not "as many params as possible," but a specific compute-optimal ratio (~20 tokens per parameter, per Chinchilla) — flag this connection now, full derivation comes next module.

### 📌 Added Explanation: a quick preview intuition (full math deferred to Module 3, as your notes say)

**In simple terms, ahead of the full derivation next module**: given a fixed compute budget (think of it as a fixed amount of money to spend), you face a genuine tradeoff between buying a *bigger* model (more parameters) versus buying *more training data passes* (more tokens) — spending everything on one extreme (a gigantic model trained on comparatively little data, or a tiny model trained on an enormous amount of data) turns out to be wasteful in both directions. Chinchilla's empirical finding was that, for a fixed compute budget, there's a specific balance point (~20 tokens of training data per parameter) that gets you the lowest achievable loss — spending compute unevenly (as many earlier large models, like GPT-3, arguably did, being "over-parameterized and under-trained" relative to this ratio) leaves performance on the table for the same compute cost. This module flags the connection now precisely because "tokens vs. parameters" is a direct consequence of *which pretraining objective and data mixture you choose* (this module) feeding into *how you then allocate compute against that data* (next module) — the two modules are genuinely sequential dependencies, not independent topics.

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

### 📌 Added Explanation: one more row worth memorizing — inference-time behavior

| | CLM | MLM | Prefix LM / Span Corruption |
|---|---|---|---|
| How you use it at inference | Autoregressive sampling, one token at a time, feeding each output back in | Single forward pass; typically produces embeddings/classification logits, not a generated sequence | Encode the full input once (bidirectional), then autoregressively decode the output, one token at a time (same generation loop as CLM, but conditioned on a bidirectionally-encoded input) |

---

## 8. Quick-fire Q&A (self-test)

*(Original questions and answers below, kept fully intact. Each answer has been additionally expanded with fuller reasoning per your request — expansions marked 📌.)*

**Q: Write the CLM loss formula and explain each term in one sentence.**
A: `L = -Σ log P(x_t | x_1...x_{t-1})` — sum of negative log-probabilities the model assigns to the true next token at every position, using only left context; this is cross-entropy loss applied autoregressively.

📌 **Expanded reasoning**: To fully justify "why this exact formula," you'd want to walk through the maximum-likelihood derivation given above: it starts from wanting to maximize the joint probability of the observed corpus (`P(x_1,...,x_T)`), decomposes that joint probability exactly via the chain rule (no independence assumption, unlike Unigram LM), takes a log to turn the resulting product into a sum for numerical stability and easy differentiation, and flips the sign purely by loss-minimization convention. Every piece of the formula — the sum, the log, the negative sign, the left-only conditioning — traces back to one of those four steps, not an arbitrary design choice.

**Q: Why does BERT use an 80/10/10 masking split instead of always using [MASK]?**
A: Always using `[MASK]` would create a train/inference mismatch, since `[MASK]` never appears in real downstream input — the 10% random-token and 10% unchanged cases force the model to build good representations for every real token, not just placeholder positions.

📌 **Expanded reasoning**: It's worth separating exactly what each of the two 10%-cases individually defends against, since they're not redundant with each other: the **10% random-token** case specifically prevents the model from learning "just copy whatever non-`[MASK]` token is already sitting here" as a lazy shortcut — since the token might be *wrong*, the model must actually compute a real contextual prediction rather than a trivial identity pass-through. The **10% unchanged** case specifically prevents the model from learning "only spend computation on `[MASK]` positions, treat every non-`[MASK]` position as already-solved and skip it" — since even an unchanged, correct-looking token is still silently being graded, the model can never be sure which positions "don't matter," forcing it to build a genuinely useful representation everywhere, all the time — exactly what's needed downstream, since fine-tuning tasks query representations at arbitrary positions, not just artificially masked ones.

**Q: Why is MLM less sample-efficient than CLM per token processed?**
A: MLM only computes loss on the ~15% masked positions per pass, while CLM gets a training signal from every single token in the sequence — so CLM extracts more learning signal per token seen.

📌 **Expanded reasoning**: Concretely, per the numerical example above, a 10,000-token MLM batch yields exactly 1,500 scored positions, while the identical 10,000-token batch under CLM yields 10,000 scored positions — a **6.67x** difference in raw number of loss-contributing predictions per batch, for the exact same amount of raw text and exact same forward/backward pass compute cost. This is precisely why, historically, achieving comparable downstream quality with MLM-style pretraining has generally required either more total training steps, more total data passes, or architectural fixes like ELECTRA's all-positions replaced-token-detection objective (covered above) that restore full-sequence training signal density while keeping bidirectional context.

**Q: What's the actual difference between T5's span corruption and BERT's MLM masking?**
A: BERT masks scattered single tokens and predicts them in-place with an encoder-only model; T5 corrupts contiguous spans, replaces each with one sentinel token, and generates only the missing spans (not the whole sequence) autoregressively via a full encoder-decoder — this makes the target sequence shorter and the objective inherently generative.

📌 **Expanded reasoning**: Beyond "contiguous spans vs. scattered singles," the deeper architectural difference is *how the prediction is made*: BERT predicts each masked token **in-place**, via a classification head applied directly at that token's position in a single encoder forward pass — there's no sequential/autoregressive generation happening at all, even conceptually. T5 predicts missing spans **out-of-place**, via a genuinely separate decoder that autoregressively generates a *new, shorter sequence* (the concatenated missing spans plus sentinel tokens) conditioned on the full encoder output — meaning T5's objective inherently exercises and pretrains the autoregressive generation mechanism itself, which is exactly why T5 (encoder-decoder) can natively generate open-ended text for downstream tasks while BERT (encoder-only) fundamentally cannot, regardless of fine-tuning.

**Q: What does perplexity of 7.4 actually mean in plain language?**
A: On average, at each token position, the model is about as uncertain as if it were randomly choosing among ~7.4 equally likely options — it's `exp(average cross-entropy loss)`.

📌 **Expanded reasoning**: As derived above, this isn't a loose figure of speech — it follows directly from the algebraic identity that a perfectly uniform guess among exactly `k` options produces loss `log(k)` and therefore perplexity exactly `k`. A real model's perplexity of 7.4 means its *actual*, non-uniform probability distribution (confident on some tokens, unsure on others, averaged across the whole evaluation set) produces the *same average loss* as a hypothetical model uniformly guessing among 7.4 options would — hence "effective branching factor" is the precise technical phrase, not just a casual description.

**Q: Does lower perplexity guarantee better downstream task performance?**
A: No — perplexity only measures next-token prediction fit to the training-like distribution; it doesn't directly capture reasoning, instruction-following, or factual accuracy, which is why benchmark evaluation exists as a separate, necessary measurement.

📌 **Expanded reasoning**: see the fully worked Model-A-vs-Model-B scenario above (Section 5) for a concrete illustration of exactly *how* two models can have near-identical perplexity yet diverge sharply on task benchmarks — the mechanism is that perplexity is dominated by the statistics of whatever the bulk of the evaluation corpus looks like (typically ordinary prose), while benchmarks like MMLU/GSM8K probe specific skills (broad knowledge recall, multi-step arithmetic/logical reasoning) that may be a small, non-representative slice of "typical next-word prediction difficulty."

**Q: Can a BERT-style model generate open-ended text the way GPT can? Why or why not?**
A: No — BERT's bidirectional MLM training and encoder-only architecture never learn an autoregressive left-to-right generation mechanism; there's no causal mask or decoder to sample tokens one at a time.

📌 **Expanded reasoning**: To be maximally precise about *why*, not just *that*: generation requires being able to produce token `t` having only seen tokens `1` through `t-1` (since tokens after `t` don't exist yet at generation time) — this is exactly the causal-masking mechanism from Section 1. BERT's training *never once* exposes the model to that constrained, left-only-context scenario; every single training example gives the model full bidirectional access to the whole sequence around any position it's asked about. There is no learned circuitry, no trained weights, corresponding to "predict this token using only what came before it" — the capability simply doesn't exist in the trained parameters, and no amount of clever prompting can retroactively install it without further training under a genuinely autoregressive objective (which is effectively what turning an encoder-only model into a generator would require).

---

## ❓ Interview Q&A (Apple / Google-style ML Engineer questions — newly added section)

*(These are additional interview-style questions in the spirit of what's typically asked in FAANG/Apple ML Engineer interviews on pretraining objectives, going beyond the quick-fire set above. Answers are given in full below each question — scroll past the question to self-test first if you'd like.)*

**Q1. You're designing a new foundation model and need it to both (a) power a chatbot that generates long free-form answers, and (b) power a semantic search/embedding feature over a document corpus. Would you pick one pretraining objective, or use two models? Justify your answer using what you know about CLM vs MLM.**

*Model answer*: These are genuinely different jobs pulling in different directions architecturally — free-form generation fundamentally requires the autoregressive/causal mechanism from CLM (no causal mask, no generation capability, full stop, per the reasoning above), while high-quality embeddings for semantic search traditionally benefit from bidirectional context (MLM-style encoders like BERT/Sentence-BERT have historically been the standard for embeddings, since a bidirectional representation of a whole query/document tends to capture meaning more holistically than a strictly left-to-right one). In practice, modern systems increasingly use a **single decoder-only (CLM) model for both**, extracting embeddings from intermediate hidden states or a final-token representation of an otherwise-generative model — this has become viable as CLM models have scaled up and picked up strong general representations "for free," and it simplifies infrastructure (one model to serve, not two). But if embedding quality is the top priority and infra cost of a second, smaller model is acceptable, a dedicated bidirectional encoder (MLM-family) fine-tuned specifically for retrieval/embeddings still tends to be a defensible, sometimes stronger, choice — I'd frame the decision explicitly around this quality-vs-infrastructure-simplicity tradeoff rather than asserting one answer is universally correct.

**Q2. Derive, from first principles, why cross-entropy loss is the natural choice for language model training rather than, say, mean-squared-error (MSE) on token IDs.**

*Model answer*: I'd start by pointing out that token IDs are categorical labels (e.g., token 4521 vs token 17), not points on a continuous numeric scale — MSE would implicitly assume that token 4522 is "closer" to token 4521 than token 9000 is, purely because of numerically adjacent IDs, which is semantically meaningless (token IDs are typically assigned by vocabulary-building order/frequency, not by any notion of similarity). Cross-entropy, by contrast, treats prediction as choosing a probability distribution over a fixed, unordered set of categories (the vocabulary) — exactly matching the true structure of the problem — and, as derived above, falls directly out of maximum likelihood estimation applied to the actual joint probability of the observed corpus under the chain rule. There's no meaningful "distance" concept between tokens for MSE to exploit correctly, whereas cross-entropy only cares about "how much probability mass did the model correctly place on the one true answer," which is the actual quantity we care about.

**Q3. A colleague claims "T5 is strictly better than BERT because it can do everything BERT can do, plus generation." Is this true? Push back if not.**

*Model answer*: Not quite — "can do everything, plus more" isn't the same as "strictly better in practice." T5's encoder-decoder architecture roughly doubles the parameter count relative to a same-sized encoder-only model for a given hidden dimension (you're paying for both an encoder stack and a decoder stack), so for a fixed parameter/compute budget, a same-size BERT-family model can often dedicate more effective capacity purely to bidirectional representation quality on pure understanding/classification tasks, potentially outperforming a same-total-size T5 model on those specific tasks. There's also the training-signal-density point from Section 7 — span corruption's efficiency gain over pure MLM narrows but doesn't eliminate this compute-allocation tradeoff. So the accurate framing is: T5 is more *flexible* (covers a strictly larger set of task types out of the box), but "flexible" and "compute-optimal for a specific narrow task at a fixed budget" are different axes, and BERT-family models can still win on the latter for pure understanding tasks — I'd push back specifically on the word "strictly."

**Q4. Explain, with a worked numeric example, why forcing loss computation on 100% of masked-selected positions (not just the 80% that got `[MASK]`) matters for the model's learned representations.**

*Model answer*: I'd walk through the exact 10,000-token batch example above: 1,500 positions are selected, and *all 1,500* — the 1,200 `[MASK]`'d, the 150 randomly substituted, and the 150 left unchanged — contribute a loss term. If, hypothetically, loss were computed *only* on the 1,200 `[MASK]`'d positions (i.e., the 10%/10% noise was applied to the input but not scored), the model would have zero training pressure to compute anything useful at the 150 randomly-substituted or 150 unchanged positions — it could, in principle, learn a shortcut of "only pay attention to genuinely-`[MASK]`ed positions, treat everything else as a free pass." Because loss is in fact computed on all 1,500 (per the original notes' loss formula, `Σ over masked positions i`, where `i` is defined as the full 15%-selected set, not the 80% `[MASK]` subset), the model is forced to maintain a genuinely useful, discriminative representation across the noised and unchanged positions too — which is exactly the mechanism that closes the train/inference gap discussed in the 80/10/10 rationale.

**Q5. You observe that your model's perplexity keeps dropping steadily during pretraining, but a downstream reasoning benchmark score is flat for a long stretch before suddenly jumping. How would you explain this to a non-technical stakeholder, and what does it suggest about how you should be evaluating training progress?**

*Model answer*: I'd explain that perplexity is a smooth, continuous, always-available signal (it's directly the training objective, so it improves incrementally with every gradient step, by construction), while a discrete downstream benchmark (like a reasoning task with a right/wrong-per-question scoring) can require a *combination* of several underlying sub-skills all to be "good enough" simultaneously before the visible aggregate score moves at all — this "flat then sudden jump" pattern is central to the emergent-abilities debate covered in Module 3, where one perspective is that the underlying capability is actually improving smoothly and continuously (mirroring the smooth perplexity curve) but the specific way the benchmark is scored (e.g., exact-match on a final multi-step answer) hides that gradual improvement until it crosses a threshold. Practically, this means I would never rely on perplexity alone to decide "is the model getting better at reasoning" — I'd track task-specific benchmarks throughout training (not just perplexity) and, where possible, use more granular/partial-credit scoring on those benchmarks (rather than strict exact-match) specifically to reduce the risk of missing real, gradual underlying progress that a coarse pass/fail metric might mask until a late, sudden-looking jump.

**Q6. In one sentence, state which of CLM, MLM, or span corruption you'd choose as the pretraining objective for a code-completion assistant that needs to both understand a codebase's existing structure and generate new code inline, and justify it in two more sentences.**

*Model answer*: I'd choose CLM (decoder-only), the same family as Codex/Copilot-style models. Code completion is fundamentally a left-to-right generation task at the point of insertion (predict what comes next given everything typed so far), so the causal, autoregressive mechanism is a direct match for the actual use case, exactly the "objective matches the actual use case" argument made for CLM's dominance in Section 1. While a Prefix-LM/encoder-decoder setup (bidirectional over the surrounding file, causal over the inserted completion) is architecturally appealing for "fill-in-the-middle" code completion specifically and is in fact used by some code models via a fill-in-the-middle training trick layered on top of a CLM base — the dominant, simplest, and most scalable choice remains a CLM decoder-only model, often with that fill-in-the-middle objective added as an auxiliary training-time trick rather than switching the base architecture entirely.

---

*End of Module 2 (expanded). Next: Module 3 — Scaling Laws & Emergent Abilities (Kaplan vs Chinchilla, compute-optimal training, the emergent-abilities debate).*
