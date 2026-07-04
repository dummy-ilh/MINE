**Roadmap:**
1. Why BERT exists (vs. bag-of-words/TF-IDF/RNNs you've used)
2. Transformer basics — attention, encoder architecture
3. Tokenization (WordPiece) & input representation
4. Pre-training objectives (Masked LM, Next Sentence Prediction)
5. Fine-tuning BERT for your own text classification task
6. Practical code walkthrough (HuggingFace)
7. Common pitfalls / tips

Let's start with **Part 1: Why BERT?**

In your previous ML pipeline, text probably went: clean text → TF-IDF/word2vec vectors → classifier (SVM, logistic regression, maybe an LSTM). The big limitation: each word gets a **fixed representation** regardless of context. "Bank" in "river bank" and "bank account" looks the same to TF-IDF.

BERT's key idea: **contextual embeddings**. The vector for "bank" changes depending on the sentence around it, because BERT reads the *entire* sentence at once (not left-to-right like RNNs) and uses **self-attention** to let every word "look at" every other word before deciding its representation.

Second key idea: **pre-train once, fine-tune anywhere**. Instead of training a classifier from scratch on your labeled data, you start from a BERT model already trained on huge amounts of text (it already "knows" grammar, facts, word relationships), then just adapt it to your task with much less labeled data.

---
## Part 2: Transformer & Self-Attention Mechanics

BERT is just a **stack of Transformer encoder layers**. Since it's an "encoder-only" model (no decoder), it's built for *understanding* text, not generating it.

**Transformer Architecture**
[image_search result above]

Each encoder layer has two sub-blocks:
1. **Multi-Head Self-Attention**
2. **Feed-Forward Network**

...with residual connections + layer normalization after each.

---

### Step 1: Self-Attention — the core idea

For each word, we create three vectors by multiplying its embedding with learned weight matrices:

$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V$$

Where:
- **Q (Query)**: "what am I looking for?"
- **K (Key)**: "what do I contain?"
- **V (Value)**: "what info do I actually pass along?"

Then attention scores are computed as:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$d_k$ = dimension of key vectors (used to scale scores so gradients don't explode).### Step 2: Worked Numeric Example

Let's take a tiny example — sentence: **"I love NLP"** (3 tokens), with embedding dimension = 4 (real BERT uses 768, but 4 keeps math visible).

Suppose after embedding + positional encoding, we get:

$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 2 \\ 1 & 1 & 1 & 1 \end{bmatrix} \quad \text{(rows = "I", "love", "NLP")}$$

Assume simple weight matrices (normally learned):

$$W^Q = W^K = W^V = \begin{bmatrix} 1&0\\0&1\\1&0\\0&1 \end{bmatrix} \quad (4\times2)$$

**Compute Q, K, V** (each becomes 3×2):

$$Q = K = V = X W^Q = \begin{bmatrix} 2 & 0 \\ 0 & 4 \\ 2 & 2 \end{bmatrix}$$

**Compute scores** $QK^T$ (3×3 matrix — how much each word attends to every other word):

$$QK^T = \begin{bmatrix} 2&0\\0&4\\2&2\end{bmatrix}\begin{bmatrix} 2&0&2\\0&4&2\end{bmatrix} = \begin{bmatrix} 4 & 0 & 4 \\ 0 & 16 & 8 \\ 4 & 8 & 8 \end{bmatrix}$$

**Scale** by $\sqrt{d_k} = \sqrt{2} \approx 1.41$:

$$\approx \begin{bmatrix} 2.83 & 0 & 2.83 \\ 0 & 11.3 & 5.66 \\ 2.83 & 5.66 & 5.66 \end{bmatrix}$$

**Softmax each row** → gives attention weights (they sum to 1 per row). For row 1 ("I"):

$$\text{softmax}([2.83, 0, 2.83]) \approx [0.475,\ 0.05,\ 0.475]$$

→ meaning: word "I" attends **47.5% to itself, 5% to "love", 47.5% to "NLP"**.

**Final step** — multiply weights by V to get the new contextual representation of "I":

$$\text{output}_{I} = 0.475 \times [2,0] + 0.05 \times [0,4] + 0.475 \times [2,2] \approx [1.9,\ 1.15]$$

That's it — this new vector for "I" now contains *context* from the other words, weighted by relevance. This is what makes BERT "contextual."### Step 3: Multi-Head Attention

Instead of doing this once, BERT does it **12 times in parallel** (12 "heads" in BERT-base), each with different learned $W^Q, W^K, W^V$. Each head can learn a different relationship (one head might track subject-verb agreement, another might track coreference like "it" → "the dog").

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

The outputs are concatenated and passed through one more weight matrix $W^O$ to mix them back into a single vector per word.

**BERT-base config:**
| Component | Value |
|---|---|
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Head dimension | 768/12 = 64 |
| Feed-forward size | 3072 |
| Parameters | ~110M |

---

### Step 4: Feed-Forward + Residual/LayerNorm

After attention, each position passes through a small MLP:

$$\text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$$

And each sub-layer wraps with a residual connection + normalization:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

This is stacked **12 times** — each layer refines the contextual representation further using the output of the previous layer as input.

---

How do I know what each head leaens? How is it different xani control it?
Good question — this trips up a lot of people because it feels like heads should be assigned roles, but they're **not**. Nobody designs "head 3 = syntax head." It emerges from training.

**How we *find out* what a head learned (after the fact):**

1. **Attention visualization** — tools like BertViz let you pick a layer/head and see attention weights as arcs between words for a given sentence. You literally watch which words light up.
2. **Probing studies** — researchers ran BERT on thousands of sentences and found patterns like:
   - Some heads mostly attend to the **next/previous token** (positional).
   - Some heads attend to the **syntactic head** of a word (dependency parsing-like behavior).
   - Some heads track **coreference** (pronoun → noun).
   - Many heads in later layers do a mix of things, less interpretable.
3. **Ablation/pruning experiments** — famous paper *"Are Sixteen Heads Really Better than One?"* found you can prune most heads with little performance loss, meaning many heads learn redundant or overlapping things.**Can you control what a head learns? Short answer: not directly, but indirectly, yes.**

You can't say "head 5, learn subject-verb agreement." Heads are just $W^Q, W^K, W^V$ matrices initialized randomly, and gradient descent shapes them based on whatever helps minimize the pre-training loss (Masked LM + Next Sentence Prediction). Whatever attention pattern reduces loss the most is what survives.

Ways to *influence* it indirectly:

| Method | What it does |
|---|---|
| **Change pre-training objective** | Different task → different attention patterns emerge (e.g. models trained for translation develop different heads than MLM models) |
| **Attention supervision / auxiliary loss** | Add a loss term that penalizes a head if it doesn't attend the way you want (used in some syntax-aware BERT variants) |
| **Architecture constraints** | Restrict a head's attention window (e.g. local attention) to force it to specialize |
| **Fine-tuning** | On your downstream task, heads reshape slightly toward what's useful for *your* labels — this is the main lever available to you practically |
| **Head pruning/masking** | You can *remove* heads that aren't contributing (post-hoc control, not "teaching" but "curating") |

For your practical purposes (fine-tuning BERT for text classification), you generally won't hand-engineer heads — you let fine-tuning nudge the existing 144 heads (12 layers × 12 heads) toward your task, and inspect afterward with BertViz if you're curious what changed.

---

## Part 3: Tokenization (WordPiece) & Input Representation

BERT doesn't split text into whole words like your old TF-IDF pipeline did. It uses **WordPiece tokenization** — splitting rare/unknown words into smaller known subword pieces.

### Why subwords?

Your classical pipeline probably had a fixed vocabulary and treated out-of-vocabulary words as `<UNK>` or dropped them. BERT instead breaks unfamiliar words into pieces it *has* seen, so it almost never hits a true unknown word.

**Example:**
```
Word: "unhappiness"
WordPiece tokens: ["un", "##happiness"]
```
The `##` prefix means "this piece connects to the previous token" (not a new word).

Another example:
```
Word: "playing"       → ["playing"]        (common word, stays whole)
Word: "playfulness"   → ["play", "##ful", "##ness"]
Word: "ChatGPT"        → ["Chat", "##G", "##PT"]  (rare/unseen word gets split finer)
```

BERT-base vocab size: **30,522 tokens** (covers common words + subword pieces + individual characters as fallback).### Special Tokens

Every input to BERT gets wrapped with two special tokens:

- **`[CLS]`** — placed at the very start. Its final hidden vector is used as the **aggregate representation of the whole sentence** — this is what you'll use for classification.
- **`[SEP]`** — marks the end of a sentence, or separates two sentences (used for sentence-pair tasks like question-answering or next-sentence prediction).

**Example — single sentence classification input:**

```
Input:  "I love NLP"
Tokens: [CLS] I love NLP [SEP]
IDs:    101   1045 2293 17953 2361 102
```

**Example — sentence-pair input** (e.g. for entailment tasks):

```
Sentence A: "The movie was great"
Sentence B: "I really enjoyed it"

Tokens: [CLS] the movie was great [SEP] i really enjoyed it [SEP]
```

### Step 2: Building the Final Input Embedding

Each token's actual input into BERT is the **sum of three embeddings**:

$$E_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$$

| Embedding | Purpose | Example |
|---|---|---|
| **Token Embedding** | Meaning of the subword itself | lookup vector for "love" |
| **Segment Embedding** | Which sentence (A or B) this token belongs to | all Sentence-A tokens get `E_A`, Sentence-B tokens get `E_B` |
| **Position Embedding** | Word order (since attention has no built-in notion of order) | position 0, 1, 2... each has a learned vector |

**Concrete shape example (BERT-base, hidden size 768):**

For the sentence `[CLS] I love NLP [SEP]` (5 tokens):

- Token embedding matrix lookup → shape `(5, 768)`
- Segment embedding (all segment A here) → shape `(5, 768)`, all rows identical (= `E_A`)
- Position embedding (positions 0–4) → shape `(5, 768)`

$$E_{\text{input}} = \text{(5, 768)} + \text{(5, 768)} + \text{(5, 768)} = \text{(5, 768)}$$

This final `(5, 768)` matrix is what actually enters Layer 1 of the Transformer encoder stack you learned in Part 2.

### Quick sanity-check numbers

For a max sequence length of 512 (BERT's default limit):
- Token embedding table: `30522 × 768` params
- Position embedding table: `512 × 768` params
- Segment embedding table: `2 × 768` params (just A/B)

This is why BERT truncates or splits long documents — anything past 512 tokens gets cut off unless you use a long-document variant (Longformer, BigBird, etc.).

---

## Part 4: Pre-training Objectives

BERT learns language understanding *before* ever seeing your labeled data, using two self-supervised tasks on massive unlabeled text (Wikipedia + BookCorpus, ~3.3 billion words). No human labels needed — the text itself provides the labels.

### Task 1: Masked Language Modeling (MLM)

Randomly mask 15% of tokens, and train BERT to predict the original word using **context from both directions** (this is BERT's key innovation over left-to-right models like GPT).

**Example:**

```
Original: "I love natural language processing"
Masked:   "I love [MASK] language processing"
Target:   predict "[MASK]" = "natural"
```

Of the 15% chosen tokens:
- **80%** replaced with `[MASK]`
- **10%** replaced with a random word (forces BERT to not over-rely on seeing `[MASK]` literally)
- **10%** left unchanged (forces BERT to still build good representations even for un-tampered tokens)

**Why this mix?** At fine-tuning/inference time, there's no `[MASK]` token in real sentences. If BERT only ever saw `[MASK]`, it would learn "just predict something plausible whenever you see this weird token" and might not build strong representations for normal tokens.

**Loss function** — standard cross-entropy over the vocabulary for masked positions only:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\setminus \text{masked}})$$

Where $P(x_i \mid \cdot)$ comes from a softmax over all 30,522 vocab tokens, computed from the final hidden vector at that position:

$$P(x_i) = \text{softmax}(h_i W_{\text{vocab}}^T + b)$$

$h_i$ = final 768-dim hidden vector at position $i$, $W_{\text{vocab}}$ = `(30522, 768)` matrix (often tied to the input token embedding matrix).### Task 2: Next Sentence Prediction (NSP)

Teaches BERT to understand **relationships between sentences** (useful for QA, entailment, etc.), not just word-level meaning.

**How training examples are built:**

```
50% of the time — real consecutive pairs (label = IsNext):
Sentence A: "The man went to the store."
Sentence B: "He bought a gallon of milk."

50% of the time — random pairs (label = NotNext):
Sentence A: "The man went to the store."
Sentence B: "Penguins live in Antarctica."
```

Input format: `[CLS] Sentence A [SEP] Sentence B [SEP]`

The final hidden vector at the `[CLS]` position is fed into a simple binary classifier:

$$P(\text{IsNext}) = \text{softmax}(W_{\text{NSP}} \cdot h_{[CLS]} + b)$$

$W_{\text{NSP}}$ is just a `(2, 768)` matrix — tiny compared to the rest of the model.

**Note:** Later research (RoBERTa paper) found NSP contributes little and sometimes even hurts performance — RoBERTa dropped it entirely and just trained longer on MLM with better data, and got better results. Worth knowing since you'll see "RoBERTa" recommended over "BERT" often in practice.

### Combined Pre-training Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

Both are trained **jointly**, in a single forward/backward pass, over ~40 epochs on the full corpus (in the original paper).

---

### Recap of what's "baked in" after pre-training:

| Capability | Learned from |
|---|---|
| Word meaning in context | MLM |
| Grammar/syntax patterns | MLM (emergent) |
| Sentence-relationship understanding | NSP |
| General world knowledge | Both, from massive corpus exposure |

This is the checkpoint you download from HuggingFace (`bert-base-uncased`) — all of this training already done. You never redo this part.

---
## Part 5: Fine-tuning BERT for Text Classification

This is where everything connects to what you already know. Instead of TF-IDF vectors → SVM, you now do: BERT's `[CLS]` vector → a small classifier head, and train the **whole thing end-to-end** on your labeled data.

### Step 1: Add a Classification Head

Take the pre-trained BERT, and bolt a tiny new layer on top of the `[CLS]` token's final hidden vector:

$$\text{logits} = W_{\text{cls}} \cdot h_{[CLS]} + b$$

- $h_{[CLS]}$ → shape `(768,)` (BERT-base hidden size)
- $W_{\text{cls}}$ → shape `(\text{num\_classes}, 768)` — **randomly initialized**, this is the only brand-new part
- Output → raw scores (logits), one per class

Then softmax to get probabilities:

$$P(y=k) = \frac{e^{\text{logit}_k}}{\sum_j e^{\text{logit}_j}}$$

**Example — sentiment classification (3 classes: negative, neutral, positive):**

```
Input: "This movie was absolutely fantastic!"
→ [CLS] this movie was absolutely fantastic ! [SEP]
→ BERT encoder (12 layers) 
→ h_[CLS] = 768-dim vector
→ W_cls (3×768) · h_[CLS] + b = [-2.1, 0.3, 4.8]  (logits)
→ softmax = [0.001, 0.007, 0.992]
→ Predicted: "positive" (99.2%)
```### Step 2: Loss Function

Standard **cross-entropy loss** against your true labels — nothing exotic, same as your old classifiers:

$$\mathcal{L} = -\sum_{k} y_k \log P(y=k)$$

where $y_k$ is 1 for the true class, 0 otherwise.

### Step 3: What Actually Gets Updated

This is the key difference from your old pipeline — **everything is trainable**, not just the head:

| Component | Trainable? | Notes |
|---|---|---|
| Token/position/segment embeddings | ✅ Yes | Adjust slightly to your domain |
| All 12 encoder layers (attention + FFN) | ✅ Yes | Whole network fine-tunes, small learning rate |
| New classification head $W_{cls}$ | ✅ Yes | Trained from scratch, larger effective updates |

Because BERT already "knows" language, fine-tuning needs **far fewer labeled examples** than training from scratch — often a few hundred to a few thousand examples work well, versus needing much more data for a classical model trained from zero.

### Step 4: Practical Training Recipe (matches the original paper's recommendations)

```
Learning rate:     2e-5 to 5e-5   (small! full training from scratch uses much higher)
Batch size:        16 or 32
Epochs:            2 to 4          (BERT overfits fast if you go longer)
Optimizer:         AdamW
Warmup steps:      ~10% of total steps, then linear decay
Max seq length:    128 or 256 (unless you need full 512)
```

**Why so few epochs and such a tiny learning rate?** BERT already has strong representations. A large learning rate or many epochs will "forget" that pre-trained knowledge (catastrophic forgetting) rather than gently adapt it. This is the opposite of training an SVM/LSTM from scratch where you'd train much longer.

### Step 5: Two Ways to Use BERT for Your Task

| Approach | What happens | When to use |
|---|---|---|
| **Feature extraction** | Freeze BERT, only train classifier head on top of extracted embeddings | Fast, less data, slightly lower accuracy |
| **Full fine-tuning** | Update all BERT weights + head together | Standard approach, best accuracy, what most people do today |

---

## Part 6: Practical Code Walkthrough (HuggingFace)

Let's fine-tune BERT for text classification end-to-end, using a binary sentiment example (positive/negative) so you can adapt it directly to your own dataset.

### Step 1: Install & Import

```python
pip install transformers datasets torch scikit-learn

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import pandas as pd
```

### Step 2: Load Tokenizer & Prepare Data

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Your labeled data — same format as you'd use for any classifier
data = {
    "text": ["I loved this movie!", "Absolutely terrible, waste of time.", 
              "Best film I've seen all year", "Boring and predictable."],
    "label": [1, 0, 1, 0]   # 1 = positive, 0 = negative
}
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
```

This runs the WordPiece tokenization from Part 3 automatically — turning raw text into `input_ids`, `attention_mask`, and `token_type_ids`.

**Peek at what tokenization actually produced:**

```python
print(tokenizer.tokenize("I loved this movie!"))
# ['i', 'loved', 'this', 'movie', '!']

print(tokenizer("I loved this movie!"))
# {'input_ids': [101, 1045, 3866, 2023, 3185, 999, 102], 
#  'token_type_ids': [0,0,0,0,0,0,0], 
#  'attention_mask': [1,1,1,1,1,1,1]}
```
Notice `101` = `[CLS]` and `102` = `[SEP]` — matches what you learned in Part 3.

### Step 3: Load Pre-trained Model with Classification Head

```python
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2   # binary classification; set to your number of classes
)
```

This is exactly the "add a classification head" step from Part 5 — HuggingFace attaches a fresh `nn.Linear(768, 2)` on top of `[CLS]` automatically, randomly initialized, while the 12 encoder layers load pre-trained weights.

### Step 4: Set Training Arguments (matches the recipe from Part 5)

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)
```

### Step 5: Train

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # use a real held-out split in practice
)

trainer.train()
```

Internally, each step does exactly what you learned: tokenize → embed (token+segment+position) → 12 encoder layers with self-attention → `[CLS]` vector → classification head → cross-entropy loss → backprop through the *entire* network.

### Step 6: Inference on New Text

```python
model.eval()
text = "This was an incredible experience"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)

print(probs)  # e.g. tensor([[0.02, 0.98]])
print(pred)   # tensor([1]) -> positive
```

### Quick comparison to your old pipeline

| Step | Old (TF-IDF + SVM) | BERT |
|---|---|---|
| Feature extraction | Manual (TF-IDF vectors) | Learned automatically (contextual embeddings) |
| Vocabulary handling | Fixed vocab, OOV → unknown | WordPiece subwords, rarely truly unknown |
| Training data needed | More, since starting from scratch | Less, since pre-trained knowledge transfers |
| Training time | Fast (seconds/minutes) | Slower (needs GPU, minutes/hours) |
| Context sensitivity | None (bag-of-words) | Full sentence context per word |

---
## Part 7: Common Pitfalls & Practical Tips

Wrapping up with the practical lessons that separate "it runs" from "it actually works well" in production.

### 1. Overfitting — BERT overfits fast

Because BERT has ~110M parameters and your dataset might only have a few thousand examples, it's easy to overfit within 2-3 epochs.

**Signs:** training loss keeps dropping, validation loss starts rising.

**Fixes:**
- Stick to 2-4 epochs max (rarely more)
- Use `weight_decay=0.01` (already in the recipe above)
- Try `dropout` (already built into BERT, default 0.1 — don't disable it)
- Early stopping based on validation loss, not training loss

### 2. Class Imbalance

Same problem you've hit with classical classifiers, but BERT doesn't automatically fix it.

```python
from torch.nn import CrossEntropyLoss

# Weight the loss by inverse class frequency
class_weights = torch.tensor([1.0, 3.0])  # if class 1 is rarer
loss_fct = CrossEntropyLoss(weight=class_weights)
```
Or oversample the minority class before tokenizing — same tricks you already know from imbalanced text classification.

### 3. Sequence Length Tradeoffs

- `max_length=128` is usually enough for sentences/short paragraphs
- `max_length=512` (BERT's absolute max) — needed for full documents, but **4x slower and more memory** since attention is $O(n^2)$ in sequence length
- If your documents are long (legal, medical, research), consider:
  - Truncating smartly (keep the first + last N tokens, not just first 512)
  - Using **Longformer** or **BigBird** (sparse attention, handles thousands of tokens)

### 4. Don't Always Reach for Full BERT

| Situation | Better choice |
|---|---|
| Very simple task, huge labeled dataset, need low latency | TF-IDF + Logistic Regression may perform comparably and run 100x faster |
| Small labeled data (<500 examples), nuanced task | BERT fine-tuning — this is where BERT shines most |
| Need small model for mobile/edge deployment | **DistilBERT** (40% smaller, 60% faster, ~97% of BERT's performance) |
| Need best possible accuracy, don't care about speed | **RoBERTa** or **DeBERTa** — trained longer, better data, usually beats vanilla BERT |
| Multiple sentence-pair tasks (similarity, entailment) | Sentence-BERT (SBERT) — much faster for computing sentence embeddings/similarity |

### 5. Reproducibility Gotchas

- Set seeds (`torch.manual_seed`, `transformers.set_seed`) — BERT fine-tuning has notable run-to-run variance, especially on small datasets
- Different random seeds can shift accuracy by 1-3%, don't over-interpret single-run results — run 3-5 seeds if comparing approaches

### 6. Debugging Checklist If Accuracy Is Bad

1. Check tokenization actually looks sane (`tokenizer.tokenize(sample_text)`)
2. Confirm `[CLS]`/`[SEP]` and `attention_mask` padding are correct (padding tokens shouldn't leak into attention)
3. Verify labels are correctly aligned with encoded batches (easy to shuffle mismatch)
4. Check learning rate isn't too high (2e-5 to 5e-5 is the safe zone — 1e-3 will destroy pre-trained weights)
5. Try a linear-probe baseline (freeze BERT, only train head) — if this underperforms your old TF-IDF+SVM, something's likely broken upstream

---

### Full Picture — What You Now Know

```
Part 1: Why BERT (contextual embeddings vs. static)
Part 2: Self-attention math (Q,K,V, softmax, multi-head)
Part 3: WordPiece tokenization + input embeddings (token+segment+position)
Part 4: Pre-training (MLM + NSP)
Part 5: Fine-tuning mechanics (CLS head, loss, learning rate)
Part 6: Actual working HuggingFace code
Part 7: Pitfalls, tradeoffs, when to use what
```

That's BERT end-to-end. From here, the natural next steps if you want to go deeper: try fine-tuning on a real dataset (IMDB or your own), then explore **RoBERTa/DeBERTa** as drop-in upgrades, or **Sentence-BERT** if you ever need embeddings for similarity/search rather than classification.

