# Chapter 2: Tokenization & Vocabulary

Before BERT can do anything, raw text must become numbers. This sounds mechanical, but the *how* matters enormously — bad tokenization breaks the model. Let's build the intuition from scratch.

---

## 2.1 The Fundamental Problem

A neural network needs integers as input. So we need a mapping:

```
"The cat sat" → [some sequence of integers]
```

The question is: **what are the units we assign integers to?**

You have three obvious choices. All three are wrong in different ways.

---

## 2.2 Option 1: Word-Level Tokenization

Split on spaces. Each unique word gets an ID.

```
"The cat sat on the mat"
→ ["The", "cat", "sat", "on", "the", "mat"]
→ [1423,   892,  1847,  312,  1423,  2091]
```

**Seems fine. Here's what breaks:**

**Problem 1 — Vocabulary explodes.**
English has 500,000+ words. Add names, technical terms, misspellings, "running/runner/ran/runs" as separate entries... your embedding table becomes enormous.

**Problem 2 — Out-of-vocabulary (OOV) words.**
User types "GPT-4o" or "COVID-19" or a name the model never saw during training. You have no vector for it. You're forced to use a generic `[UNK]` token, and all information is lost.

**Problem 3 — Morphological blindness.**
"run", "running", "runner", "ran" are clearly related. Word-level tokenization treats them as completely unrelated entries.

---

## 2.3 Option 2: Character-Level Tokenization

Go the other direction. Each character gets an ID.

```
"cat" → ["c", "a", "t"] → [23, 11, 44]
```

**Advantages:**
- Tiny vocabulary (~100 characters)
- No OOV problem ever
- Handles any new word automatically

**What breaks:**
"The cat sat on the mat" → 22 tokens (counting spaces).
"The cat sat on the mat and the dog watched from the windowsill" → 62 tokens.

For BERT's max sequence length of 512, you can fit maybe 400 characters of actual content. Paragraphs become unwieldy. And "c", "a", "t" as separate tokens carry no inherent meaning — the model must learn to group them into concepts from scratch, requiring far more data and computation.

---

## 2.4 Option 3: WordPiece — The Sweet Spot

BERT uses **WordPiece tokenization**. The key idea: **break words into frequent subword units.**

Common words stay whole. Rare or complex words get split into meaningful pieces.

```
"cat"           → ["cat"]               ← common, stays whole
"running"       → ["running"]           ← common enough
"unbelievable"  → ["un", "##believe", "##able"]
"embeddings"    → ["em", "##bed", "##ding", "##s"]
"ChatGPT"       → ["Chat", "##GP", "##T"]
"COVID"         → ["CO", "##VI", "##D"]
```

The `##` prefix means *"I am a continuation of the previous token, not a word start."*

---

## 2.5 How WordPiece Vocabulary Is Built

It's built **bottom-up** from a large corpus using a greedy algorithm:

**Step 1:** Start with every individual character as a token.
```
vocabulary = {a, b, c, ..., z, A, ..., Z, 0, ..., 9, ...}
```

**Step 2:** Count all adjacent pairs across the corpus.
```
"low", "lower", "lowest" → "l"+"o" appears frequently → merge into "lo"
```

**Step 3:** Merge the pair that **most increases likelihood of the training data** (not just raw frequency — this is the WordPiece distinction from BPE).

**Step 4:** Repeat until vocabulary reaches target size.

BERT's vocabulary: **30,522 tokens.** This covers:
- Common whole words
- Frequent subwords
- Individual characters (fallback)
- Special tokens

**The result:** Any word in any language can be tokenized. The model never sees a truly unknown token.

---

## 2.6 Numerical Walkthrough: Text to Token IDs

Let's trace exactly what happens to the sentence:

```
"The cat sat"
```

**Step 1: Lowercase** (BERT-base-uncased lowercases everything)
```
"the cat sat"
```

**Step 2: WordPiece tokenize**
```
["the", "cat", "sat"]
```
All common words → stay whole.

**Step 3: Add special tokens**
```
["[CLS]", "the", "cat", "sat", "[SEP]"]
```

**Step 4: Convert to IDs** (lookup in vocabulary table)
```
[CLS]  → 101
the    → 1996
cat    → 4937
sat    → 2938
[SEP]  → 102
```

**Final input to BERT:**
```
Token IDs: [101, 1996, 4937, 2938, 102]
```

---

## 2.7 The Special Tokens — Each One Has a Job

BERT uses four special tokens. Each is critical.

### [CLS] — ID 101

Prepended to **every single input**, always at position 0.

It has no linguistic meaning at the start. But after passing through 12 Transformer layers, it has attended to every other token. Its final vector becomes the **sentence-level representation** used for classification tasks.

Think of it as an empty vessel that fills up with global meaning as it passes through the layers.

### [SEP] — ID 102

Marks the **end of a sentence** (or segment). Used to separate two sentences in two-sentence tasks.

```
Single sentence:   [CLS] the cat sat [SEP]
Two sentences:     [CLS] the cat sat [SEP] the dog watched [SEP]
```

Without [SEP], BERT can't tell where sentence A ends and sentence B begins — critical for tasks like question answering (question = sentence A, passage = sentence B).

### [MASK] — ID 103

Used **only during pre-training**. Replaces masked tokens that the model must predict.

```
[CLS] the [MASK] sat [SEP]  → model predicts "cat" at position 2
```

You will **never see [MASK] during fine-tuning or inference.** It's purely a training device.

### [PAD] — ID 0

Neural networks process batches, and all sequences in a batch must be the same length. Shorter sequences get padded.

```
Sequence 1: [101, 1996, 4937, 2938, 102,   0,   0,   0]  ← padded
Sequence 2: [101, 2054, 2003, 1996, 2749, 1029,  102,  0]  ← padded
```

An **attention mask** tells BERT which positions are real (1) vs padding (0), so padding tokens are never attended to.

---

## 2.8 A Harder Example: Rare and Compound Words

```
Input: "unaffable"   (an obscure word meaning 'unfriendly')
```

WordPiece tokenization:
```
"unaffable" → ["un", "##aff", "##able"]
Token IDs  → [4895, 14546, 3085]
```

Now even if BERT never saw "unaffable" in pre-training, it has seen:
- "un" (prefix meaning negation) — in "unhappy", "unclear", etc.
- "##able" (suffix meaning capacity) — in "comfortable", "capable", etc.

It can make a reasonable inference about the word's meaning from its parts. This is WordPiece's superpower.

---

## 2.9 Sequence Length and the 512 Limit

BERT has a hard limit of **512 tokens** per input. This comes from positional embeddings (Chapter 3) — BERT only has learned positions for slots 0 through 511.

**In practice:** After WordPiece, most sentences fit well within 512. But long documents (articles, contracts, research papers) get **truncated**.

```
Max tokens:         512
Minus [CLS] + [SEP]: -2
Usable tokens:      510
```

For two-sentence inputs:
```
[CLS] sentence_A [SEP] sentence_B [SEP]
              ↑                  ↑
         counts as 1        counts as 1
```

Total: len(A) + len(B) + 3 special tokens ≤ 512.

This 512 limit is a real-world constraint you'll hit constantly in production. (Chapter 12 covers how to handle it.)

---

## 2.10 Full Worked Example: Two Sentences

Let's tokenize a QA-style input completely:

```
Question: "Where did the cat sit?"
Context:  "The cat sat on the mat."
```

**After WordPiece + special tokens:**
```
[CLS] where did the cat sit ? [SEP] the cat sat on the mat . [SEP]
```

**Token IDs:**
```
101  2073  2106  1996  4937  2303  1029  102  1996  4937  2938  2006  1996  13523  1012  102
```

**Positions:**
```
0     1     2     3     4     5     6    7    8     9     10    11    12    13     14    15
```

**Segment IDs:**
```
0     0     0     0     0     0     0    0    1     1     1     1     1     1      1     1
```
(0 = sentence A / question, 1 = sentence B / context)

These three arrays — token IDs, positions, segment IDs — are the three inputs to BERT's embedding layer, which is exactly what Chapter 3 covers.

---

## Chapter 2 Summary

| Design Choice | Why |
|---|---|
| Subword (not word) | Handles OOV, manageable vocab size |
| Subword (not char) | Tokens carry meaning, shorter sequences |
| 30,522 vocab size | Empirically good tradeoff for English |
| [CLS] always first | Aggregates sentence-level meaning through layers |
| [SEP] between sentences | Lets model know segment boundaries |
| 512 token limit | Tied to learned positional embeddings |

---

The key takeaway: by the end of tokenization, you have **three integer arrays** for every input — token IDs, position indices, and segment IDs. Chapter 3 shows exactly how each one becomes a 768-dimensional vector, and how those three vectors are combined before the first Transformer layer ever sees them.

Ready for **Chapter 3**?
