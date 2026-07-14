# Chapter 10: Fine-Tuning BERT

Pre-trained BERT knows language deeply. But it has one job: predict masked tokens. To make it do anything useful — classify sentiment, answer questions, tag entities — you need fine-tuning.

Fine-tuning is the process of taking pre-trained weights and continuing training on your labeled task data, with a small task-specific head added on top.

---

## 10.1 The Core Idea

```
Pre-trained BERT (knows language)
        +
Small task-specific head (randomly initialized)
        +
Your labeled data (hundreds to millions of examples)
        +
Low learning rate + few epochs
        =
Task-specific BERT
```

The key insight: **BERT already knows language**. You're not teaching it what words mean or how grammar works — that's done. You're teaching it to apply that knowledge to your specific task.

This is why fine-tuning works with surprisingly little data. A task with 5,000 labeled examples can produce a state-of-the-art model because BERT brings 110M parameters of pre-learned language understanding to the table.

---

## 10.2 What Changes During Fine-Tuning

This is a common interview question. Be precise:

```
UPDATED during fine-tuning:
  All 12 Transformer blocks (attention + FFN weights)
  All 3 embedding tables (token, position, segment)
  The new task-specific head weights

NOT changed:
  The architecture itself
  The vocabulary (still 30,522 tokens)
  The 512 token limit
```

The pre-trained weights are not frozen by default — they all update. But because the learning rate is very small (2e-5 to 5e-5), they shift only slightly from their pre-trained values.

Think of it as **gentle nudging**, not rewriting.

---

## 10.3 The Learning Rate — Why So Small?

Pre-training used learning rate ~1e-4. Fine-tuning uses 2e-5 to 5e-5 — roughly 5-50× smaller.

**Why?**

```
Pre-trained weights encode valuable language knowledge
Too large a learning rate → weights move far from pre-trained values
→ BERT "forgets" what it learned → called catastrophic forgetting

Small learning rate → weights stay close to pre-trained values
→ Task learning happens on top of language knowledge
→ Pre-training knowledge is preserved
```

A useful mental model:

```
Pre-trained BERT = expert linguist
Fine-tuning = teaching that linguist your specific domain

You don't need to re-teach them English.
You just need to show them your task.
Small adjustments, not a full retraining.
```

---

## 10.4 Task 1: Text Classification

**Examples:** Sentiment analysis, topic classification, spam detection, intent classification.

### Architecture

```
[CLS] I loved this movie [SEP]
  ↓
12 Transformer blocks
  ↓
h_CLS: [768-d vector]   ← take only the [CLS] token output
  ↓
Dropout(0.1)
  ↓
Linear(768 → num_classes)
  ↓
Softmax
  ↓
Class probabilities
```

The task head is just **one linear layer** on top of the [CLS] vector.

### Numerical Example — Sentiment Analysis (2 classes)

**Input:** "I loved this movie"

```
After tokenization:
[CLS] i loved this movie [SEP]
  101  1045  3866  2023  3185   102

After 12 blocks:
h_CLS = [0.83, -0.21, 0.44, 0.67, ..., -0.12]   shape: [768]

Task head (Linear 768→2):
W = [[0.3, -0.2, 0.8, ...],    ← positive class weights
     [-0.4, 0.5, -0.3, ...]]   ← negative class weights

logits = h_CLS · Wᵀ
       = [2.14, -1.83]

Softmax:
  e^2.14 = 8.50
  e^-1.83 = 0.16
  Sum = 8.66

P(positive) = 8.50/8.66 = 0.982
P(negative) = 0.16/8.66  = 0.018

Prediction: POSITIVE (98.2% confidence)
Label:      POSITIVE
Loss:       -log(0.982) = 0.018   ← very low, model was right
```

### Why [CLS] and Not Other Tokens?

Common question. Three reasons:

```
1. [CLS] was specifically designed for this role
   Its position 0 means it has no "own meaning" to preserve
   It's free to become a pure sentence-level aggregator

2. Through 12 layers of attention, [CLS] has attended to
   every other token repeatedly. It's seen everything.

3. The NSP task during pre-training specifically trained
   [CLS] to produce sentence-level representations
   for binary classification (IsNext/NotNext)
   → Already pre-trained for classification tasks
```

**Alternative: mean pooling**

Some practitioners average all token vectors instead of using [CLS]:

```
h_mean = mean(h_1, h_2, ..., h_n)   across all non-[PAD] tokens
```

Empirically, mean pooling sometimes outperforms [CLS] — especially when fine-tuning data is small and [CLS] hasn't had enough examples to calibrate properly.

---

## 10.5 Task 2: Named Entity Recognition (NER)

**Examples:** Finding people, organizations, locations, dates in text.

### The Key Difference from Classification

Classification: one label for the whole sequence.
NER: one label **per token**.

```
Input:   "Barack Obama visited Paris yesterday"
Labels:   PER    PER    O       LOC   O
```

### Architecture

```
[CLS] Barack Obama visited Paris yesterday [SEP]
  ↓      ↓      ↓       ↓      ↓      ↓      ↓
12 Transformer blocks
  ↓      ↓      ↓       ↓      ↓      ↓      ↓
h_0    h_1    h_2     h_3    h_4    h_5    h_6    ← all 768-d vectors

IGNORE h_0 ([CLS]) and h_6 ([SEP])

For each token position:
  Linear(768 → num_entity_labels)
  Softmax
  → Per-token class label
```

### Entity Labels (BIO Scheme)

```
B-PER:  Beginning of a person entity
I-PER:  Inside a person entity (continuation)
B-LOC:  Beginning of a location entity
I-LOC:  Inside a location entity
B-ORG:  Beginning of an organization
I-ORG:  Inside an organization
O:      Outside any entity
```

### Numerical Example

**Input:** "Barack Obama visited Paris"

```
Tokens: [CLS] Barack Obama visited Paris [SEP]

After 12 blocks:
  h_[CLS]   = [...]    → ignored
  h_Barack  = [0.71, -0.32, 0.88, ...]
  h_Obama   = [0.68, -0.29, 0.91, ...]
  h_visited = [-0.12, 0.44, -0.21, ...]
  h_Paris   = [0.55, 0.61, -0.33, ...]
  h_[SEP]   = [...]    → ignored

Linear(768→7) applied to each:

  h_Barack  → logits [2.1, 1.8, -0.3, -1.2, 0.1, -0.8, -1.5]
            → softmax → P(B-PER)=0.71, P(I-PER)=0.19, ...
            → Prediction: B-PER ✓

  h_Obama   → logits [-0.2, 2.8, 0.1, -0.9, 0.3, -0.5, -1.1]
            → softmax → P(I-PER)=0.82, P(B-PER)=0.09, ...
            → Prediction: I-PER ✓   (continuation of Barack)

  h_visited → logits [-1.2, -0.8, -0.3, -0.7, -0.2, -0.9, 2.9]
            → softmax → P(O)=0.88
            → Prediction: O ✓

  h_Paris   → logits [0.1, -0.3, 2.6, 1.1, -0.4, -0.7, -0.8]
            → softmax → P(B-LOC)=0.73
            → Prediction: B-LOC ✓
```

### The Subword Problem in NER

Remember WordPiece splits rare words:

```
"Schwarzenegger" → ["Sch", "##war", "##zen", "##egg", "##er"]
```

Five tokens, but one entity span. Which token gets the B-PER label?

**Standard approach:** Label only the **first subword** token, ignore the rest:

```
Sch     → B-PER
##war   → ignored in loss computation
##zen   → ignored
##egg   → ignored
##er    → ignored
```

Or propagate the label to all subwords:

```
Sch     → B-PER
##war   → I-PER
##zen   → I-PER
##egg   → I-PER
##er    → I-PER
```

Then at inference, merge subword predictions back to the original word.

---

## 10.6 Task 3: Question Answering (SQuAD)

This is the most elegant fine-tuning design. Understanding it deeply impresses interviewers.

### The Task

Given a question and a passage, find the **span** in the passage that answers the question.

```
Question: "Where did the cat sit?"
Passage:  "The fluffy cat sat on the old wooden mat near the window."
Answer:   "the old wooden mat"   (a span within the passage)
```

The model doesn't generate an answer — it just points to where the answer starts and ends in the passage.

### Input Format

```
[CLS] where did the cat sit ? [SEP] the fluffy cat sat on the old wooden mat [SEP]
  ↑                             ↑                                              ↑
sentence start           sentence boundary                              sequence end

Segment IDs:
  0  ← question tokens → 0    1 ← passage tokens → 1
```

### Architecture — Two Vectors

The task head is beautifully minimal: just **two learned vectors**:

```
start_vector: [768-d]   learned during fine-tuning
end_vector:   [768-d]   learned during fine-tuning
```

For every token in the passage, compute a start score and end score:

```
start_score_i = dot(start_vector, h_i)
end_score_i   = dot(end_vector, h_i)
```

Softmax over all passage positions:

```
P(token i is answer start) = softmax(start_scores)[i]
P(token i is answer end)   = softmax(end_scores)[i]
```

The answer span = [argmax(start_scores), argmax(end_scores)]

### Numerical Example

**Simplified:** 6 passage tokens after [SEP]: "the cat sat on the mat"

```
After 12 blocks, passage token hidden states:
  h_the  = [0.31, -0.12, 0.55, ...]
  h_cat  = [0.72,  0.44, -0.21, ...]
  h_sat  = [0.18,  0.63, 0.39, ...]
  h_on   = [-0.22, 0.11, 0.28, ...]
  h_the2 = [0.29, -0.08, 0.51, ...]
  h_mat  = [0.61,  0.52, -0.18, ...]

Learned start_vector = [0.8, 0.3, -0.1, ...]
Learned end_vector   = [-0.2, 0.6, 0.4, ...]

Start scores (dot product with start_vector):
  the:  0.31×0.8 + (-0.12)×0.3 + 0.55×(-0.1) = 0.248-0.036-0.055 = 0.157
  cat:  0.72×0.8 +   0.44×0.3 + (-0.21)×(-0.1) = 0.576+0.132+0.021 = 0.729
  sat:  0.18×0.8 +   0.63×0.3 + 0.39×(-0.1)  = 0.144+0.189-0.039 = 0.294
  on:  -0.22×0.8 +   0.11×0.3 + 0.28×(-0.1)  =-0.176+0.033-0.028 =-0.171
  the2: 0.29×0.8 + (-0.08)×0.3 + 0.51×(-0.1) = 0.232-0.024-0.051 = 0.157
  mat:  0.61×0.8 +   0.52×0.3 +(-0.18)×(-0.1) = 0.488+0.156+0.018 = 0.662

Softmax(start_scores):
  the:   0.157  → e^0.157=1.170
  cat:   0.729  → e^0.729=2.073   ← highest
  sat:   0.294  → e^0.294=1.342
  on:   -0.171  → e^-0.171=0.843
  the2:  0.157  → e^0.157=1.170
  mat:   0.662  → e^0.662=1.939
                   Sum=8.537

  P(start):
  the:  0.137
  cat:  0.243   ← argmax → answer starts at "cat"
  sat:  0.157
  on:   0.099
  the2: 0.137
  mat:  0.227

End scores (dot product with end_vector):
  the:  0.31×(-0.2)+(-0.12)×0.6+0.55×0.4 = -0.062-0.072+0.220 = 0.086
  cat:  0.72×(-0.2)+0.44×0.6+(-0.21)×0.4 = -0.144+0.264-0.084 = 0.036
  sat:  0.18×(-0.2)+0.63×0.6+0.39×0.4   = -0.036+0.378+0.156 = 0.498
  on:  -0.22×(-0.2)+0.11×0.6+0.28×0.4   =  0.044+0.066+0.112 = 0.222
  the2: 0.29×(-0.2)+(-0.08)×0.6+0.51×0.4 =-0.058-0.048+0.204 = 0.098
  mat:  0.61×(-0.2)+0.52×0.6+(-0.18)×0.4 =-0.122+0.312-0.072 = 0.118

  P(end): argmax → "sat"

Answer span: "cat sat"
```

Not perfect (real answer is just "mat") — but remember this uses toy numbers. In a trained model, start/end vectors are calibrated across thousands of examples to point precisely to answer boundaries.

### Why This Design Is Clever

```
No generation needed       → much simpler than seq2seq
No external memory         → answer must exist in passage
Two vectors, 2×768 params  → tiny task head, almost nothing to learn
All the work done by BERT  → passage-question interaction in 12 layers
```

The question tokens and passage tokens attend to each other across all 12 layers. By layer 12, passage tokens know exactly what the question is asking, and the start/end vectors simply need to find which passage token best matches that query.

---

## 10.7 Fine-Tuning Hyperparameters

These are battle-tested defaults. Memorize them for interviews.

```
Learning rate:    2e-5 to 5e-5       (much smaller than pre-training)
Batch size:       16 or 32
Epochs:           2 to 4             (more → catastrophic forgetting)
Warmup steps:     10% of total steps
Dropout:          0.1
Max seq length:   128 (short tasks) or 512 (QA, long doc tasks)
Optimizer:        Adam with weight decay (AdamW)
```

**Why only 2-4 epochs?**

```
Epoch 1-2: Model learns task-specific patterns
           Classification head calibrates
           Upper BERT layers shift toward task

Epoch 3-4: Diminishing returns
           Risk of overfitting on small datasets
           Risk of catastrophic forgetting increasing

Epoch 5+:  Often hurts performance
           Model starts memorizing training examples
           Pre-trained knowledge gets overwritten
```

---

## 10.8 Freezing Layers — When and Why

Sometimes you don't update all of BERT. You **freeze** some layers.

```
Frozen layers:  weights don't update during fine-tuning
Unfrozen layers: weights update normally
```

### When to Freeze

```
Large fine-tuning dataset (100k+ examples):
  → Fine-tune all layers
  → Enough data to adjust everything safely

Small fine-tuning dataset (< 5,000 examples):
  → Freeze layers 1-8, fine-tune layers 9-12 + head
  → Prevents overfitting
  → Lower layers encode general language (don't need changing)
  → Upper layers encode task-specific patterns (do need changing)

Very small dataset (< 1,000 examples):
  → Freeze all BERT layers
  → Only train the task head
  → BERT acts as a pure feature extractor
```

### Gradual Unfreezing

A more sophisticated approach (ULMFiT-style):

```
Epoch 1: Unfreeze only task head
Epoch 2: Unfreeze layers 10-12 + head
Epoch 3: Unfreeze layers 7-12 + head
Epoch 4: Unfreeze all layers
```

Lower layers get fewer gradient updates — they change less, preserving general language knowledge while upper layers adapt to the task.

---

## 10.9 The Complete Fine-Tuning Picture

```
                    PRE-TRAINED BERT
                   (knows language)
                         ↓
              ┌──────────────────────┐
              │   Your labeled data  │
              │   Your task head     │
              │   Small LR (2e-5)    │
              │   2-4 epochs         │
              └──────────────────────┘
                         ↓
              ┌──────────────────────────────────────────┐
              │          Task-Specific BERT              │
              │                                          │
              │  Classification: [CLS] → Linear → label │
              │  NER:    each token → Linear → tag       │
              │  QA:     passage tokens → start/end span │
              └──────────────────────────────────────────┘
```

---

## 10.10 What Fine-Tuning Actually Does to the Weights

Researchers have studied how much each layer changes during fine-tuning:

```
Layer    Average weight change (sentiment task)
──────────────────────────────────────────────
1-3      0.003%    ← almost unchanged
4-6      0.008%    ← minimal change
7-9      0.021%    ← modest change
10-12    0.089%    ← meaningful change
Head     100%      ← initialized randomly, fully trained
```

Lower layers barely move. Upper layers shift meaningfully. The task head is fully trained from scratch.

This pattern makes intuitive sense:

```
Layers 1-3:   basic syntax → same for all tasks → don't need to change
Layers 10-12: high-level semantics → task-specific → do need to change
```

---

## Chapter 10 Summary

| Task | Input | Head | Output |
|---|---|---|---|
| Classification | [CLS] + sentence | Linear(768→classes) | Class label |
| NER | [CLS] + sentence | Linear(768→tags) per token | Tag per token |
| QA | [CLS] + question + [SEP] + passage | Two vectors (start, end) | Span in passage |

### Key Numbers to Remember
```
Learning rate:   2e-5 to 5e-5
Epochs:          2 to 4
Warmup:          10% of steps
Batch size:      16 or 32
Head parameters: tiny compared to BERT's 110M
```

### The Power of Fine-Tuning
```
Pre-training:  3.3B words, 4 days, 16 TPUs, ~$50,000
Fine-tuning:   thousands of examples, minutes to hours, 1 GPU, ~$10
Performance:   often state-of-the-art on the target task
```

This asymmetry — expensive pre-training, cheap fine-tuning — is what made BERT transformative. One model, pre-trained once, fine-tuned for hundreds of different tasks.

---

You now understand BERT end to end:

```
Chapter 2:  Text → tokens
Chapter 3:  Tokens → embeddings
Chapter 4:  Self-attention
Chapter 5:  Multi-head attention
Chapter 6:  Feed-forward network
Chapter 7:  Residuals + LayerNorm
Chapter 8:  12 blocks stacked
Chapter 9:  Pre-training (MLM + NSP)
Chapter 10: Fine-tuning for tasks
```

Two chapters left. **Chapter 11** covers BERT variants — RoBERTa, DistilBERT, ALBERT — when to use which and why. **Chapter 12** is your Google interview prep — the exact questions, how to answer them, and the tradeoffs you need to articulate.

Ready for 11?
