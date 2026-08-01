# Chapter 1 — The Problem BERT Solves (Master Notes, Apple MLE Prep)

> Goal of this doc: you should be able to close this file and explain, from memory, *why BERT exists*, defend it against "just use GPT" pushback, do the one calculation an interviewer might ask for, and answer the Q&A set cold.

---

## 0. One-sentence version (say this first in an interview)

> "BERT solves the problem that every NLP model before it had to pick one of two flaws: either it had no real context (Word2Vec), or it only had context from one direction (RNNs and GPT) — BERT is a Transformer encoder pre-trained to read the whole sentence at once, in both directions, before it decides what any word means."

Everything below is the evidence for that sentence.

---

## 1. The core challenge, restated precisely

"Bank" isn't ambiguous to *you* because you don't process "I went to the bank to fish" one word at a time and commit to a meaning at word 4. You hold the sentence, then resolve. Every architecture in this chapter is a different answer to: **how much of the sentence can the model see before it has to decide what a word means, and in which direction?**

That single axis — *how much context, from which directions, before commitment* — is the entire chapter. Keep coming back to it.

---

## 2. The four eras, boosted

### 2.1 Bag of Words — zero context
`{the: 2, cat: 1, sat: 1, on: 1, mat: 1}`

**Why it's not just "bad," it's *structurally* incapable**: there is no position information anywhere in the representation, so *any* permutation of the same multiset of words is identical to the model. "Dog bites man" = "Man bites dog." This isn't a training problem you can fix with more data — the representation itself has thrown away the information you'd need.

**What if we just added position as a feature?** You'd get closer to n-grams (bigrams, trigrams), which do capture *local* order. That was in fact the next real step historically. But n-grams explode combinatorially (vocabulary of 30k words → 30k² = 900M possible bigrams) and still can't capture "the movie, which I expected to hate, was actually great" — a dependency spanning 8+ words.

### 2.2 Word2Vec — meaning without context

**The core idea, simplified**: words that show up in similar surrounding words get pushed to similar vectors. That's it. No labels, no grammar rules — just "words with similar neighbors end up close together in vector space" (the *distributional hypothesis*, older than Word2Vec itself, going back to linguist J.R. Firth: "you shall know a word by the company it keeps").

**The equation, simplified**: Word2Vec (skip-gram variant) trains by trying to predict a word's neighbors:

$$P(\text{context word} \mid \text{center word}) = \frac{\exp(v_{context} \cdot v_{center})}{\sum_{w \in V} \exp(v_w \cdot v_{center})}$$

Plain-language read of every term:
- $v_{center}$, $v_{context}$ — the vectors for the current word and a nearby word. These are literally what the model is learning; everything else is fixed math.
- $v_{context} \cdot v_{center}$ (dot product) — a similarity score. Bigger dot product = model thinks these two words appear near each other more often.
- $\exp(\cdot)$ and the sum over the whole vocabulary $V$ — this is just a **softmax**: turn raw similarity scores into a probability distribution over "which word is likely nearby," so training can push probability mass toward the words that actually *were* nearby and away from the rest.

**One fully worked toy example** (2-dimensional vectors, tiny vocabulary of 3 words, so you can see the mechanics):

Vocabulary: `bank`, `river`, `money`. Suppose after some training steps we have:
```
v(bank)  = [0.9, 0.1]
v(river) = [0.8, 0.2]
v(money) = [0.1, 0.9]
```
Dot products from `bank`:
- `bank · river` = 0.9×0.8 + 0.1×0.2 = 0.74
- `bank · money` = 0.9×0.1 + 0.1×0.9 = 0.18

Softmax over just these two candidates:
- exp(0.74) ≈ 2.10, exp(0.18) ≈ 1.20, sum ≈ 3.30
- P(river | bank) ≈ 2.10 / 3.30 ≈ **0.64**
- P(money | bank) ≈ 1.20 / 3.30 ≈ **0.36**

That's the whole mechanism: gradient descent nudges `v(bank)` and `v(river)` closer together every time they co-occur in training data, and nudges unrelated pairs apart. Do this over billions of sentences and you get vectors where `king − man + woman ≈ queen` falls out as a side effect of the geometry — nobody programmed that arithmetic in.

**Why this is a dead end for understanding, not just "not perfect"**: `v(bank)` above is a *single point*, fixed after training. It had to average over every sense of "bank" the model ever saw — river banks, savings banks, "bank on it," "bank a plane." The vector literally cannot represent more than one meaning at once, because it's one fixed point in space. This isn't a data or scale problem — throwing more text at Word2Vec makes the single vector a *better average*, never a *context-sensitive* one.

**What if we trained a separate vector per sense of each word (multi-sense embeddings)?** People tried this. Problem: you need to know in advance how many senses a word has, and you still pick one *statically* per occurrence rather than computing it from the live sentence — so "bank" in a never-before-seen sentence still can't be freshly disambiguated.

**Other Word2Vec failure modes, with the "why":**
- **OOV words get nothing.** The vector table is a fixed lookup built at training time — a word not in that table has no row to look up. (Subword methods like FastText partially fix this by building word vectors from character n-grams, but that's a different chapter.)
- **Negation is invisible.** "good" and "not good" are two separate tokens with two separate, unrelated vectors, and nothing in the architecture composes them — "not" doesn't have a mechanism to *flip* or *modify* the meaning of the word next to it.

### 2.3 RNNs / LSTMs — order and memory, but one-directional and slow

**Simplified recurrence equation:**

$$h_t = f(h_{t-1}, x_t)$$

Read this as: "the model's understanding *after* word $t$ ($h_t$) is a function of its understanding *before* word $t$ ($h_{t-1}$) plus the new word ($x_t$)." That's the entire idea of a recurrent network — an evolving summary, updated one word at a time. LSTMs are the same equation with extra gates (learned "how much do I keep vs. forget vs. write") to fight the next problem.

**Why long-range dependencies fail (the actual mechanism, not just "it forgets"):** $h_t$ is a fixed-size vector (say, 512 numbers) that has to represent *everything relevant so far*, no matter whether "so far" is 3 words or 300. Every new word gets blended into that same fixed-size vector. Information from word 1 doesn't get *deleted*, it gets *diluted* — mathematically, gradients from a loss at step 300 back to a parameter update at step 1 have to flow through ~300 multiplications, and repeated multiplication by numbers under 1 shrinks toward zero (**vanishing gradients**). That's why the network effectively stops "hearing" distant words. LSTMs' gates soften this but don't remove the fixed-size-bottleneck problem itself.

**What if we just made $h_t$ bigger?** Diminishing returns — you're still forcing an unbounded amount of information through a bottleneck of fixed width, and now every step is more expensive too.

**Why sequential processing kills training speed, concretely**: to compute $h_{50}$ you need $h_{49}$, which needs $h_{48}$... all the way back to $h_1$. On a GPU built for doing thousands of independent multiplications *simultaneously*, this is close to worst-case: you can't start step 50's matrix multiply until step 49's is fully done. Training time scales roughly linearly with sequence length with no parallel speedup, versus a Transformer where all positions in a sequence can be processed in one batched matrix multiply.

**Bidirectional RNNs — the "obvious fix" and why it's not actually BERT:** Yes, you can run one LSTM left-to-right and a second LSTM right-to-left and concatenate their hidden states (a **BiLSTM**). This *does* give you both-direction context. So why isn't a BiLSTM "BERT before BERT"? Two reasons interviewers like to probe:
1. It's **two separate, independently-processed passes glued together at the end**, not one unified representation where every word attends to every other word directly at every layer. The left-to-right pass at word 5 still hasn't "talked to" word 8 during its own processing — it only meets that information at concatenation.
2. It **still inherits the sequential-speed and long-range-decay problems** in each direction — you've doubled the compute, not removed the bottleneck.

### 2.4 The Transformer — parallel, direct connections, but built for translation

The mechanism that replaces recurrence is **self-attention**. Simplified formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Term by term, in plain language:
- **$Q$ (query), $K$ (key), $V$ (value)** — three different learned projections of the *same* input tokens. Intuition: for every word, $Q$ asks "what am I looking for?", $K$ answers "what do I contain?", and $V$ is "what information do I actually hand over if picked." (Library metaphor: $Q$ is your search query, $K$ is the index card on every book, $V$ is the book's actual content.)
- **$QK^T$** — every query dotted with every key → a raw compatibility score between every pair of words. This is the direct "every word talks to every other word" step — the thing RNNs could never do in one step.
- **$\sqrt{d_k}$** — just a scaling constant (square root of the key dimension) to keep the dot products from growing huge and making softmax too peaked/unstable as dimensions grow. Pure numerical-stability trick, not a conceptual one.
- **softmax(...)** — turns the raw scores into attention *weights* that sum to 1 per word — "how much should I weight each other word's information."
- **$\times V$** — use those weights to take a weighted average of every word's value vectors. The output for "bank" becomes a *blend* of "river," "money," etc., weighted by relevance. This is exactly the fan-diagram above, expressed as one matrix equation.

**What if we removed positional information entirely?** Self-attention on its own is *permutation-invariant* — "dog bites man" and "man bites dog" would get identical attention outputs, same failure as Bag of Words! This is why Transformers add **positional encodings** (a vector added to each word embedding that encodes its position) — without them, you've accidentally rebuilt Bag of Words with extra steps.

**Why the original Transformer isn't "just use this for everything":** it's an encoder-decoder built for translation. The **decoder** generates the output sentence one word at a time and is deliberately blocked ("masked") from seeing future words it hasn't generated yet — otherwise it could "cheat" during training by peeking at the answer. So half of the original Transformer is unidirectional by construction, for a very good reason (you can't peek at words you haven't generated).

### 2.5 GPT — decoder-only, unidirectional by necessity

GPT keeps only the decoder half and drops translation — it's pre-trained to predict the next word given everything before it:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

**Why GPT can't be bidirectional even if you wanted it to be**: this isn't an oversight, it's forced by the task. If GPT could see word $t+1$ while predicting word $t$ during training, the "prediction" would be trivial — the model would just be copying the answer. Autoregressive generation *requires* the future to be hidden. This is the key interview point: **GPT's unidirectionality is not a limitation someone forgot to fix — it's a structural requirement of the objective it's trained on.**

**Why that's fine for generation and costly for understanding**: generation genuinely only has past tokens available (you haven't written the next word yet!). But classification/NER/QA tasks aren't generating anything — the full sentence already exists when you're deciding "is this positive or negative sentiment," so throwing away right-side context is a self-imposed handicap, not a task requirement.

### 2.6 BERT — bidirectional encoder via masking

**The trick that makes bidirectional pre-training even possible:** you can't train a "predict the next word" objective bidirectionally (see above — it'd be trivial). BERT sidesteps this with the **Masked Language Model (MLM)** objective: randomly hide ~15% of tokens, and force the model to predict *only those hidden tokens* using everything else — both directions — as context. Since the masked words are removed from the input the model sees, there's no "peeking at the answer" problem, and both directions are legitimately available.

**Simplified MLM loss** (cross-entropy over just the masked positions):

$$\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\setminus \text{masked}})$$

Plain language: for each masked position $i$, look at the probability the model assigned to the *correct* word, take the negative log (so being confidently right = low loss, being confidently wrong = huge loss), and sum this over all masked positions. Standard classification loss — the only thing special is *which positions* it's computed over (masked ones only) and *what context is used* to predict them (the entire sentence, both directions, at once).

**One tiny worked numeric example** (vocab of 4 words, 1 masked position):

Sentence: `"The [MASK] sat on the mat"`, true answer = `cat`. Suppose the model's softmax output over a toy 4-word vocabulary `{cat, dog, mat, sat}` is:
```
P(cat) = 0.60   P(dog) = 0.25   P(mat) = 0.10   P(sat) = 0.05
```
Loss for this one masked token = $-\log(0.60) \approx 0.51$ nats.

If instead the model had been badly wrong and said `P(cat) = 0.05`, loss = $-\log(0.05) \approx 3.00$ — six times higher. That's the whole incentive structure: BERT is punished in proportion to how *confidently wrong* it is, on masked-word prediction only, using both-direction context to make that prediction.

**What if we masked 100% of tokens, or 0%?** Google actually ablated this. Too few masked tokens (near 0%) → training signal per sentence is too sparse, painfully slow to learn. Too many masked (near 100%) → not enough surrounding, un-masked context left for the model to actually use bidirectionally — you'd be back to something closer to unconditional generation. ~15% was an empirically-tuned sweet spot balancing "enough signal per sentence" against "enough remaining context per prediction." (Interview-relevant nuance: of that 15%, BERT doesn't just replace all of them with `[MASK]` — 80% become `[MASK]`, 10% become a random word, 10% stay unchanged. This exists to fix a train/inference mismatch: the literal `[MASK]` token never appears at fine-tuning or inference time, so training the model to *only* ever see `[MASK]` in that position would teach it a token it will never encounter again. Mixing in random/unchanged tokens forces it to build genuinely robust contextual representations for every position, not just a special "fill-in-the-blank" trick for the mask token specifically.)

---

## 3. Timeline, boosted with the "why" at each arrow

```
Bag of Words   → no order, no meaning              [fix: track word identity + counts]
     ↓  problem: "dog bites man" = "man bites dog"
Word2Vec       → meaning, but ONE vector per word   [fix: learn from co-occurrence]
     ↓  problem: same vector for every sense of a word — context-blind
RNN/LSTM       → order + memory, one direction      [fix: sequential hidden state]
     ↓  problem: left-to-right only; vanishing gradients over long ranges; no parallelism
Transformer    → all positions at once, in parallel [fix: self-attention replaces recurrence]
     ↓  problem: built encoder+decoder for translation; decoder half is unidirectional by design
GPT            → decoder-only, great at generation  [fix: next-word prediction, huge scale]
     ↓  problem: unidirectional by necessity (can't peek at ungenerated future) → weak for understanding
BERT           → encoder-only, bidirectional        [fix: mask tokens, predict from both sides]
```

---

## 4. What BERT is / isn't (expanded)

| | BERT |
|---|---|
| Architecture | Transformer **encoder only** (no decoder) |
| Directionality | Bidirectional — every layer, every token sees the whole sequence |
| Pre-training objective | Masked Language Model (+ Next Sentence Prediction, a secondary objective for sentence-pair tasks) |
| Output | One contextual vector per token, informed by full sentence |
| Great for | Classification, NER, extractive QA, sentence-pair tasks — anything that *understands* fixed input |
| Bad for | Free-form text generation, translation — it has no decoder and no mechanism to generate token-by-token |
| Usage pattern | Pre-train once on massive unlabeled text → fine-tune (or now, often just use frozen embeddings) on your specific task |

**Interview trap to watch for**: "Why not just use BERT for text generation?" — the honest answer isn't "it's worse," it's "it structurally can't do it the way GPT does." BERT was never trained to predict the *next* token given only the past; every position was trained seeing the whole sentence, including tokens after it. There's no clean way to generate left-to-right one token at a time from that training regime without leaking future information the model was trained to rely on.

---

## 5. Diagnostics — common misconceptions to pre-empt in an interview

| Misconception | Why it's wrong | Correct framing |
|---|---|---|
| "BERT is just Word2Vec but bigger" | Word2Vec is one static vector per word type; BERT computes a fresh vector per token *occurrence*, conditioned on that specific sentence | BERT embeddings are contextual; Word2Vec's are static/type-level |
| "BiLSTM already solved bidirectionality, so BERT wasn't a big leap" | BiLSTM concatenates two independently-processed passes; it doesn't let every word attend directly to every other word at every layer, and still has the vanishing-gradient / no-parallelism problems of recurrence | BERT's bidirectionality comes from unrestricted self-attention, not from stitching two one-directional passes |
| "GPT could just remove its causal mask to become bidirectional" | The training *objective itself* (predict next word from the past) becomes trivial/undefined if future tokens are visible — this isn't a switch you can flip without changing what's being predicted | You'd need a different pre-training objective (like MLM) to legitimately train bidirectionally |
| "More masking = more learning" | Masking too much removes the context the model needs to actually use bidirectionally; masking too little gives too sparse a training signal | 15% (with the 80/10/10 split) is a tuned tradeoff, not "more is better" |
| "Attention replaces the need for word order info" | Self-attention alone is permutation-invariant — swap word order and raw attention output is identical unless you add positional encodings | Position must be injected explicitly; it's not free from the attention mechanism |

---

## 6. Q&A practice set (self-test — answers below the line)

**Q1 (easy).** Why does "river bank" and "savings bank" get the *same* vector in Word2Vec but *different* vectors in BERT?

**Q2 (easy).** What's the single-sentence reason a Bag-of-Words model can't tell "dog bites man" from "man bites dog"?

**Q3 (medium).** Explain, mechanically (not just "it forgets"), why an LSTM struggles with dependencies across 50+ words.

**Q4 (medium).** A BiLSTM processes a sentence left-to-right and right-to-left and concatenates the results. Why isn't this considered equivalent to what BERT does?

**Q5 (medium — calculation).** In self-attention, why do we divide $QK^T$ by $\sqrt{d_k}$? What would go wrong if we skipped this?

**Q6 (hard).** Why can't GPT simply remove its causal (look-ahead) mask during pre-training to get bidirectional context, the way BERT does?

**Q7 (hard).** In BERT's masking scheme, only 80% of the selected 15% of tokens are actually replaced with the literal `[MASK]` token. Why not mask all of them?

**Q8 (hard — spot the bug).** A colleague says: "I removed positional encodings from our Transformer and training got faster with barely any accuracy drop on a bag-of-words-style topic classification task — so positional encodings mostly don't matter." Is this reasoning sound? What's the confound?

---
---

### Answers

**A1.** Word2Vec assigns exactly one fixed vector per word *type*, learned once from an average over all the contexts that word appeared in during training — it cannot change per sentence. BERT computes a fresh vector per word *occurrence*, built by self-attention over that specific sentence's other words, so "bank" near "river" and "bank" near "savings" get pulled toward different neighboring tokens and end up as different vectors.

**A2.** Bag-of-Words discards word order entirely and represents a sentence only as counts, so any permutation of the same multiset of words produces the identical representation — there's no position information to distinguish them.

**A3.** The LSTM's hidden state is a fixed-size vector that has to summarize everything seen so far; each new word blends into it, gradually diluting older information rather than deleting it outright. During backpropagation, the gradient signal from a late time step has to flow back through many repeated multiplications to reach an early time step's parameters, and repeated multiplication by values under 1 causes that gradient to shrink toward zero (vanishing gradients) — so the model effectively stops learning from distant context.

**A4.** A BiLSTM is two separately-run, one-directional passes whose outputs are concatenated only at the end — at word 5's own processing step, the left-to-right pass still hasn't "talked to" word 8 directly. BERT's self-attention lets every word directly attend to every other word, in both directions, at every layer of processing, not just at a final concatenation. BiLSTMs also still inherit the sequential-processing speed bottleneck and long-range-decay issues of RNNs, which self-attention removes entirely.

**A5.** As the dimensionality $d_k$ of queries/keys grows, dot products $QK^T$ tend to grow large in magnitude just from having more terms summed. Large inputs to softmax push it into a very "peaked"/saturated regime (near one-hot), which makes gradients extremely small and training unstable. Dividing by $\sqrt{d_k}$ rescales the dot products back down to a stable range regardless of dimension — it's a numerical-stability fix, not a conceptual part of "what attention means."

**A6.** GPT's objective is to predict the next token *given only the past*. If future tokens were visible during that prediction, the model wouldn't need to learn anything — it could just copy the answer sitting right there, making the loss meaningless as a training signal. Removing the mask doesn't just "add more context," it destroys the task the model is being trained on. Bidirectional training instead requires an objective like BERT's MLM, where the target tokens are actually hidden from the input (not just from an attention mask) so there's nothing to trivially copy.

**A7.** If 100% of masked positions always literally contained `[MASK]`, the model could learn a narrow trick: "build a good representation only for filling in `[MASK]` tokens." But `[MASK]` never appears in real downstream data — no fine-tuning task or real sentence ever contains that token. Occasionally substituting a random word or leaving the original word unchanged forces the model to produce robust contextual representations for *every* token position generally, since it can never be sure whether a given (non-mask) token is the "true" one or a corrupted one it should still be modeling correctly — closing the gap between pre-training and real usage.

**A8.** The reasoning is confounded by the task choice. Topic classification from a bag-of-words-style signal is largely driven by *which* words are present, not their order — so it's one of the few tasks where discarding position genuinely costs little. This result doesn't generalize; on tasks that depend on word order (sentiment with negation, syntax-sensitive tasks, translation, anything needing "who did what to whom"), removing positional encodings collapses the model toward Bag-of-Words-level blindness to order, since self-attention alone is permutation-invariant. The colleague measured on a task specifically insensitive to the thing they removed.

---

## 7. Quick recap card (for last-minute review before the interview)

- **Bag of Words**: no order, no meaning.
- **Word2Vec**: meaning, but one frozen vector per word — context-blind.
- **RNN/LSTM**: order + some memory, but one-directional and can't parallelize; long-range info decays via vanishing gradients.
- **Transformer**: self-attention lets every word see every word directly, in parallel — but built for translation (encoder + causally-masked decoder).
- **GPT**: decoder-only, unidirectional *by necessity* of its next-word-prediction objective — great for generation, structurally limited for understanding.
- **BERT**: encoder-only, trained with Masked Language Modeling so bidirectional context is actually trainable without "cheating" — the right tool for classification/NER/QA, not for generation.
- **The unifying question behind the whole chapter**: how much context, from which directions, before the model commits to a meaning?

*(Chapter 2 — Tokenization — picks up from here: how raw text like "unbelievable" becomes the integer IDs BERT actually consumes.)*
