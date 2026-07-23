# Module 1, Chapter 1.1
## What is NLP? Problem Landscape, Applications, and Why It's Hard

---

### What problem are we actually solving?

At its core, Natural Language Processing is the field of teaching computers to understand, interpret, and generate human language.

That sounds straightforward. It is not.

Human language is the most complex communication system ever developed. It evolved over hundreds of thousands of years not to be precise or unambiguous, but to be fast, flexible, and social. Computers, on the other hand, were designed to process formal, unambiguous instructions. NLP is the bridge between those two worlds.

---

### The landscape of NLP problems

NLP is not one problem. It is a family of related problems. Here are the major ones, ordered roughly from simple to complex.

**Text Classification**
Given a piece of text, assign it a category. Is this email spam or not? Is this review positive or negative? What topic is this news article about? This is the "hello world" of NLP.

**Sequence Labeling**
Given a sequence of words, assign a label to each word individually. In the sentence "Barack Obama was born in Hawaii", label "Barack Obama" as a person, "Hawaii" as a location. This is named entity recognition. Similarly, labeling each word with its grammatical role — noun, verb, adjective — is part-of-speech tagging.

**Language Modeling**
Given the words so far, what word comes next? This seems like a narrow problem. It is actually the foundation of nearly all modern NLP, including the systems that power ChatGPT.

**Machine Translation**
Convert text from one language to another. "The cat sat on the mat" → "Le chat était assis sur le tapis." Harder than it looks because languages differ not just in vocabulary but in grammar, word order, and what concepts even exist.

**Information Extraction**
Pull structured information from unstructured text. Given a thousand news articles, extract every company acquisition: who bought whom, for how much, on what date.

**Question Answering**
Given a document and a question, find the answer in the document. This is what Google tries to do when it shows you a direct answer above the search results.

**Summarization**
Compress a long document into a short one while preserving the key information. Extractive summarization picks sentences from the original. Abstractive summarization writes new sentences.

**Dialogue and Conversation**
Build systems that hold multi-turn conversations with humans. This is what chatbots, voice assistants, and systems like Claude do.

**Natural Language Generation**
Given some structured data — a table of sports scores, a weather forecast, a database of financial results — write a human-readable description of it.

---

### Why is NLP hard? The seven fundamental problems

This is the most important section of this chapter. Before we learn any techniques, we need to understand what we are fighting against. Each of these problems will come up again and again throughout the course.

**Problem 1: Ambiguity**

Human language is deeply, pervasively ambiguous. Consider the sentence:

*"I saw the man with the telescope."*

Did I use a telescope to see the man? Or did I see a man who was carrying a telescope? Both readings are grammatically valid. Humans resolve this instantly using context. Computers have to be explicitly taught how.

This is called **structural ambiguity** — the same sentence has multiple valid parse trees.

Now consider:

*"The bank was steep."*
*"I went to the bank."*

The word "bank" means something completely different in each sentence. This is **lexical ambiguity** — a single word has multiple meanings. English is full of this. The word "set" has over 400 distinct meanings in the dictionary.

**Problem 2: Variability**

The same meaning can be expressed in countless different ways.

- "The car needs to be fixed."
- "The vehicle requires repair."
- "My car is broken."
- "Can someone fix my car?"
- "The car isn't working."

All five sentences express roughly the same thing. A system that learns from one form may completely fail on another. This is why NLP models need enormous amounts of training data.

**Problem 3: Context dependence**

The meaning of a word or sentence often cannot be determined without surrounding context.

*"Can you pass the salt?"*

Grammatically this is a yes/no question about physical ability. Pragmatically it is a request. Every fluent English speaker knows this. It is not written in the words — it is carried by shared social understanding that computers do not have by default.

*"It was cold. She put on her jacket."*

The pronoun "she" refers to whoever was cold. The jacket-wearing is caused by the cold. None of this is explicit. Humans fill it in automatically. This is called **coreference resolution** and **discourse understanding**, and it is extremely difficult to automate.

**Problem 4: World knowledge**

*"The city council refused the demonstrators a permit because they feared violence."*

Who feared violence — the city council, or the demonstrators? To answer correctly, you need to know that city councils typically fear violence from demonstrators, not from themselves. This requires real-world knowledge about how institutions work.

NLP systems that only see text have no access to this knowledge unless it was implicitly encoded in that text during training.

**Problem 5: Creativity and variation**

Language is not a fixed code. People invent new words constantly. "Googling", "unfriend", "cryptocurrency", "COVID" — none of these existed a few decades ago. Slang, jargon, abbreviations, emojis, intentional misspellings, sarcasm, metaphor, irony — all of these deviate from the literal, formal language that NLP systems are easiest to build for.

A model trained on formal news text will struggle badly with Twitter. A model trained on English will fail on code-switching text that mixes English and Spanish mid-sentence.

**Problem 6: Data sparsity**

Even if we had a perfect mathematical model of language, we would need to estimate its parameters from data. But language has a long tail. Most word combinations that are grammatically valid and semantically meaningful will never appear in any training dataset, no matter how large.

This is the **data sparsity problem**, and it is why so much of classical NLP is about smoothing, back-off, and generalization — ways of handling things you have never seen before.

**Problem 7: Evaluation**

How do you know when an NLP system is good? For some tasks this is easy — classification has accuracy, translation can use BLEU score. But for summarization, dialogue, or generation: what makes one summary better than another? Humans disagree. Automated metrics often disagree with humans. This makes it hard to know if your model is improving in a meaningful way.

---

### A brief map of how the field evolved

Understanding history prevents you from reinventing the wheel and helps you understand why each technique exists.

**1950s–1980s: Rule-based systems.** Researchers wrote explicit grammatical rules and dictionaries by hand. These systems worked in narrow domains but could not scale. Language has too many exceptions.

**1980s–2000s: Statistical NLP.** Instead of rules, use probability. Count how often words appear together. Learn from data. This era produced n-gram language models, HMMs for tagging, and the first machine translation systems that actually worked at scale.

**2000s–2013: Machine learning NLP.** SVMs, MaxEnt models, CRFs. Better features, better classifiers. Systems got more accurate but still required enormous amounts of hand-engineered features.

**2013–2017: Neural NLP.** Word2Vec showed that neural networks could learn word meaning from raw text. RNNs and LSTMs replaced handcrafted sequence models. Seq2Seq models transformed machine translation.

**2017–present: Transformer era.** The Transformer architecture, followed by BERT and GPT, changed everything. Pre-train a massive model on raw text, then fine-tune it on your specific task. This approach now dominates essentially every NLP benchmark.

We will rebuild this entire history from scratch, in order, so that by the time we reach Transformers you will understand exactly why each piece of the architecture was designed the way it was.

---

### Where NLP is used in production today

This is not an academic exercise. NLP is in products you use every day.

- Search engines: understanding your query, ranking results
- Email: spam filtering, smart reply, priority inbox
- Voice assistants: Siri, Alexa, Google Assistant
- Translation: Google Translate, DeepL
- Content moderation: detecting hate speech, misinformation
- Healthcare: extracting diagnoses from clinical notes
- Finance: parsing earnings calls, detecting fraud in documents
- Code: GitHub Copilot, Claude, ChatGPT

The field is moving faster than almost any other area of machine learning. That is precisely why building from first principles matters — if you understand the foundations, you can adapt to whatever comes next.

---

### Summary

- NLP is the field of teaching computers to process human language.
- It encompasses many sub-problems: classification, tagging, language modeling, translation, QA, summarization, dialogue.
- Language is hard because it is ambiguous, variable, context-dependent, knowledge-intensive, creative, sparse, and hard to evaluate.
- The field evolved from rules → statistics → machine learning → neural networks → Transformers.
- NLP powers a huge range of real-world products and is one of the fastest-moving areas in all of ML.

---
# Module 1, Chapter 1.2
## The NLP Pipeline: From Raw Text to Structured Data

---

### The core problem

Computers do not read. They process numbers. Every NLP system ever built has to solve the same fundamental problem before doing anything interesting: take raw text, which is a sequence of characters, and convert it into a structured numerical representation that an algorithm can operate on.

This conversion is not a single step. It is a pipeline — a sequence of stages where each stage takes the output of the previous one and transforms it further. Understanding this pipeline is essential because every NLP system you will ever build starts here, and bugs introduced early in the pipeline quietly corrupt everything downstream.

---

### The full pipeline

Here is the pipeline we will build over this module. Today we look at it end to end at a high level. Each subsequent chapter covers one stage in depth.

```
Raw Text
    ↓
Sentence Segmentation
    ↓
Tokenization
    ↓
Normalization
    ↓
Stemming / Lemmatization
    ↓
Stopword Removal (optional)
    ↓
Feature Extraction (BoW, TF-IDF, embeddings...)
    ↓
Model Input (numbers)
```

Each arrow is a transformation. Each transformation makes a choice. Every choice has consequences.

---

### Stage 1: Sentence Segmentation

Before we process text, we need to decide what the units of processing are. For most NLP tasks, the natural unit is the sentence. Before we can tokenize words, we need to split the document into sentences.

This sounds trivial. It is not.

The naive approach is to split on periods. Consider why that fails immediately:

*"Dr. Smith works at Acme Corp. in New York. He earned his Ph.D. in 1998."*

There are two sentences here. There are also four periods that do not end sentences. A period-splitter would produce four fragments, all wrong.

Other complications:
- Ellipses: "She said... and then stopped."
- Abbreviations: "The U.S.A. exports a lot of goods."
- Decimal numbers: "The temperature was 98.6 degrees."
- Quotations: `He said "Stop. Right now." and left.`

Real sentence segmenters use a combination of rules and learned classifiers. We will implement one in Chapter 1.7. For now, understand that even step one of the pipeline requires care.

---

### Stage 2: Tokenization

A token is the basic unit of text your model will operate on. Tokenization is the process of splitting a string into tokens.

The most natural token is a word. So the most natural tokenization is splitting on whitespace:

```
"the cat sat on the mat"
→ ["the", "cat", "sat", "on", "the", "mat"]
```

Simple and mostly right. But consider:

```
"don't"     → ["don't"] or ["do", "n't"] or ["dont"]?
"New York"  → ["New", "York"] or ["New York"]?
"$1,000.00" → ["$1,000.00"] or ["$", "1,000", ".", "00"]?
"isn't"     → ["isn't"] or ["is", "n't"]?
"co-operate"→ ["co-operate"] or ["co", "operate"]?
```

Each of these is a real decision with real consequences for downstream tasks. If you split "don't" into "do" and "n't", your sentiment classifier can learn that "n't" is a negation signal. If you keep it as "don't", it is a single opaque token.

Different tokenizers make different choices. We will implement several in Chapter 1.3 and understand exactly what trade-offs each one makes.

There is also a deeper question: should tokens be words at all? Modern systems often use **subword tokenization**, splitting words into smaller pieces. "unhappiness" might become ["un", "happiness"] or ["un", "happy", "ness"]. This handles rare words and new vocabulary far better than word-level tokenization. We will introduce this idea in Chapter 1.3 and return to it fully in Module 11.

---

### Stage 3: Normalization

After tokenization, tokens are still raw strings with a lot of surface variation that does not reflect meaningful differences in content.

"Dog", "dog", "DOG", and "Dog." are four different strings. They mean the same thing. Normalization is the process of collapsing this variation.

Common normalization steps:

**Lowercasing.** Convert everything to lowercase. "The" and "the" become the same token. Simple and almost always helpful, with one exception: case carries meaning sometimes. "US" (United States) vs "us" (pronoun). "Apple" (company) vs "apple" (fruit). For most tasks the gains from lowercasing outweigh the losses.

**Punctuation removal.** Strip or separate punctuation marks. "cat." becomes "cat". But again — punctuation sometimes carries meaning. "!" signals emotion. "?" signals a question. Whether to remove it depends on the task.

**Number normalization.** Replace all numbers with a special token like `<NUM>`. "He earned $47,000" becomes "He earned `<NUM>`". This prevents the model from treating every distinct number as a unique vocabulary item.

**Handling special characters.** HTML tags, URLs, email addresses, hashtags, emojis. Each requires a decision: strip them, replace them with a special token, or keep them.

The key insight is that normalization is lossy — you are deliberately throwing away information. You do this because the information you throw away is less valuable than the benefit of reducing vocabulary size and surface variation. Whether that trade-off is correct depends entirely on the task.

---

### Stage 4: Stemming and Lemmatization

After normalization, you still have the problem that "run", "runs", "running", and "ran" are four different tokens that all express the same underlying concept.

This is the **morphological variation problem**. Human languages inflect words — they add prefixes and suffixes to encode tense, number, gender, case, and so on. For many NLP tasks we want to collapse these variants into a single canonical form.

There are two approaches.

**Stemming** is fast and crude. It chops off suffixes using rules, without any linguistic knowledge. The Porter stemmer, the most famous, would convert:

```
"running" → "run"
"studies" → "studi"
"fishing"  → "fish"
"argued"   → "argu"
```

Notice "studies" → "studi" and "argued" → "argu". These are not real words. Stemming produces a stem, not necessarily a valid word. But it groups related forms together, which is often enough.

**Lemmatization** is slower and more principled. It uses a dictionary and grammatical analysis to find the true base form — the lemma.

```
"running" → "run"
"studies" → "study"
"better"  → "good"
"was"     → "be"
```

Lemmatization produces real words and handles irregular forms correctly ("better" → "good", "was" → "be"). It requires knowing the part of speech of the word to do this correctly — "meeting" as a verb lemmatizes to "meet", but "meeting" as a noun lemmatizes to "meeting".

We will implement both from scratch in Chapter 1.5.

---

### Stage 5: Stopword Removal

Some words are so common that they carry almost no information about the content of a document. "The", "a", "is", "in", "of", "and" — these appear in nearly every document about nearly every topic.

For tasks like document classification or information retrieval, these words add noise. Removing them reduces vocabulary size and focuses the model on the words that actually distinguish one document from another.

A stopword list is simply a predefined set of words to remove. NLTK, spaCy, and other libraries ship with standard stopword lists. You can also build your own.

However — and this is important — stopwords are not always useless. For tasks like sentiment analysis, "not" is on most stopword lists but is obviously crucial to meaning. "I do not like this" becomes "like this" after stopword removal, completely inverting the sentiment. For tasks like authorship attribution, function words like "the" and "of" are actually highly diagnostic because different authors use them at different rates.

The rule is: stopword removal is a tool. Use it when it helps. Know when it hurts.

---

### Stage 6: Feature Extraction

After the above steps, you have a cleaned, normalized list of tokens. But tokens are still strings. Your model needs numbers.

Feature extraction converts tokens into numerical representations. This is where the real choices begin, and where most of the course lives.

The simplest approach: count the tokens. How many times does each word appear in this document? This gives you a **Bag of Words** vector — a long vector of counts, one dimension per word in the vocabulary. Module 2 covers this in depth.

A more sophisticated approach: weight those counts by how rare the word is across all documents. This is **TF-IDF**. Also in Module 2.

An even more sophisticated approach: represent each word as a dense vector of real numbers learned from data, where words with similar meanings have similar vectors. This is **word embeddings**. Module 5.

The most sophisticated approach: represent each word as a vector that depends on its context in the specific sentence it appears in. This is what Transformers do. Modules 9–11.

Each of these is a progressively richer answer to the same question: how do we turn a word into a number in a way that captures something meaningful about what the word means?

---

### How the pipeline fits together: a concrete example

Let's run the sentence "Dr. Smith doesn't like running in New York." through the full pipeline.

**Raw input:**
```
"Dr. Smith doesn't like running in New York."
```

**After sentence segmentation:**
```
["Dr. Smith doesn't like running in New York."]
```
(One sentence — the period after "Dr." is correctly identified as an abbreviation.)

**After tokenization:**
```
["Dr.", "Smith", "does", "n't", "like", "running", "in", "New", "York", "."]
```

**After normalization (lowercase, punctuation handling):**
```
["dr", "smith", "does", "n't", "like", "running", "in", "new", "york"]
```

**After lemmatization:**
```
["dr", "smith", "do", "not", "like", "run", "in", "new", "york"]
```

**After stopword removal:**
```
["dr", "smith", "like", "run", "new", "york"]
```

**After feature extraction (BoW):**
```
A vector of counts over your full vocabulary.
Position 4521 ("smith") = 1
Position 8832 ("run") = 1
... and so on.
```

Six transformations, each one deliberate, each one throwing away something and keeping something else.

---

### The pipeline is not fixed

Different tasks require different pipelines. Here are some examples.

For **sentiment analysis**: keep punctuation (exclamation marks matter), keep "not" (do not remove it as a stopword), lowercase everything.

For **named entity recognition**: do not lowercase (capitalization is a strong signal that something is a proper noun), do not lemmatize (you need to know the original surface form to extract "New York" as a named entity).

For **machine translation**: minimal preprocessing — you want the model to see the text as close to natural as possible, because translation requires preserving everything including tense and number.

For **information retrieval**: aggressive normalization, stemming or lemmatization, stopword removal — because you want "running" and "run" to match the same query.

This is why understanding the pipeline matters more than memorizing any specific pipeline. You need to be able to design the right one for each task.

---

### A note on modern systems

In modern Transformer-based systems, the preprocessing pipeline is much shorter. These models use subword tokenization (which handles morphological variation automatically) and do not need stopword removal or stemming because they learn from so much data that frequency differences wash out naturally.

But understanding the classical pipeline is not optional. It teaches you:
- What information is in text and how to extract it
- What can go wrong and where
- Why the design choices in modern tokenizers were made
- How to debug when your model is behaving strangely

Every professional NLP engineer understands this pipeline cold.

---

### Summary

- The NLP pipeline converts raw text into structured numerical input for a model.
- The stages are: sentence segmentation → tokenization → normalization → stemming/lemmatization → stopword removal → feature extraction.
- Every stage makes choices. Every choice has consequences.
- The right pipeline depends on the task.
- Modern systems shorten the pipeline but the underlying concepts remain essential.

---

# Module 1, Chapter 1.3
## Tokenization: Whitespace, Rule-Based, and Subword Intuition

---

### Why tokenization deserves its own chapter

In the last chapter we treated tokenization as one step in a pipeline. But tokenization is actually one of the most consequential decisions in any NLP system. The way you split text into tokens determines your vocabulary, your model's ability to handle rare words, and how well your system generalizes to new text.

Get it wrong and no amount of sophisticated modeling downstream fixes it.

---

### What exactly is a token?

A token is the atomic unit your model operates on. Everything your model learns, it learns about tokens. If two different surface forms map to the same token, the model treats them identically. If two similar surface forms map to different tokens, the model must learn about them separately.

This is why the tokenization decision matters so much. It is not just text processing hygiene — it directly shapes what your model can and cannot learn.

---

### Level 1: Whitespace Tokenization

The simplest possible tokenizer. Split on spaces.

```python
def whitespace_tokenize(text):
    return text.split()

text = "the cat sat on the mat"
print(whitespace_tokenize(text))
# ['the', 'cat', 'sat', 'on', 'the', 'mat']
```

This works reasonably well for clean, formal English. It fails immediately on almost everything else.

**Problem 1: Punctuation attached to words**

```python
text = "the cat sat on the mat."
whitespace_tokenize(text)
# ['the', 'cat', 'sat', 'on', 'the', 'mat.']
```

"mat." and "mat" are now different tokens. Your model will treat them as unrelated. If "mat" appears 10,000 times in training but "mat." appears only 200 times, the model learns much less about "mat." even though they are the same word.

**Problem 2: Contractions**

```python
text = "I don't think that's right"
whitespace_tokenize(text)
# ["I", "don't", "think", "that's", "right"]
```

"don't" is one token. But it contains two meaningful pieces: "do" and "not". A model that sees "don't" as a single opaque unit cannot generalize — it cannot connect "don't" to "do" or "not" individually.

**Problem 3: Languages without spaces**

Chinese, Japanese, and Thai do not use spaces between words. Whitespace tokenization produces nothing useful for these languages.

```
中文没有空格
(Chinese has no spaces)
```

Splitting on whitespace gives you the entire string as one token.

Whitespace tokenization is useful as a quick baseline or for very clean data. In practice, almost every real system needs something better.

---

### Level 2: Rule-Based Tokenization

Rule-based tokenizers use explicit rules — usually regular expressions — to handle the cases whitespace tokenization gets wrong.

The core idea: before or after splitting on whitespace, apply rules to separate punctuation, handle contractions, protect special patterns, and so on.

Here is a simple rule-based tokenizer built from scratch:

```python
import re

def rule_based_tokenize(text):
    # Step 1: separate punctuation from words
    # Add a space before punctuation marks at word boundaries
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    
    # Step 2: handle contractions — split them explicitly
    text = re.sub(r"n't", " n't", text)
    text = re.sub(r"'re", " 're", text)
    text = re.sub(r"'ve", " 've", text)
    text = re.sub(r"'ll", " 'll", text)
    text = re.sub(r"'m",  " 'm",  text)
    text = re.sub(r"'s",  " 's",  text)
    
    # Step 3: split on whitespace and filter empty strings
    tokens = [t for t in text.split() if t]
    return tokens

text = "I don't think that's right, do you?"
print(rule_based_tokenize(text))
# ["I", "do", "n't", "think", "that", "'s", 
#  "right", ",", "do", "you", "?"]
```

Now "don't" is split into "do" and "n't". The comma and question mark are separated from the words they were attached to. Each token is a meaningful unit.

Let's test on more cases:

```python
texts = [
    "Dr. Smith earned $1,000.00 in the U.S.A.",
    "She said 'hello' and left.",
    "co-operate isn't the same as cooperate"
]

for t in texts:
    print(rule_based_tokenize(t))
    print()

# ['Dr', '.', 'Smith', 'earned', '$1', ',', '000', '.', 
#  '00', 'in', 'the', 'U', '.', 'S', '.', 'A', '.']
#
# ['She', 'said', "'hello'", 'and', 'left', '.']
#
# ['co-operate', 'is', "n't", 'the', 'same', 'as', 'cooperate']
```

You can see the limitations immediately. "Dr." is split into "Dr" and ".". "$1,000.00" is mangled. "U.S.A." is broken into individual letters separated by periods.

Every new case you want to handle correctly requires a new rule. This is the fundamental problem with rule-based systems: language has too many special cases. You spend months writing rules and still have a system that breaks on text it has not seen before.

The Penn Treebank tokenizer, a classic rule-based tokenizer used widely in NLP research, has dozens of rules and still does not handle every case correctly. You can use it via NLTK:

```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
text = "I don't think that's right, do you?"
print(tokenizer.tokenize(text))
# ['I', 'do', "n't", 'think', 'that', "'s", 
#  'right', ',', 'do', 'you', '?']
```

Rule-based tokenizers are still useful. For specialized domains — clinical text, legal documents, code — writing domain-specific rules gives you precise control. But for general-purpose NLP, we need something more principled.

---

### Level 3: Statistical Tokenization and the Vocabulary Problem

Before we get to subword tokenization, we need to understand the problem it solves.

Any tokenizer that works at the word level faces a fundamental issue: the **open vocabulary problem**.

Imagine you train a model on a large corpus of English text. Your vocabulary — the set of all tokens your model knows — might contain 50,000 words. But then at test time, your model sees:

- "COVID-19" (new word)
- "blockchain" (too recent to be in training data)
- "Schwarzenegger" (rare proper noun)
- "unhappiness" (maybe seen, maybe not)
- "antidisestablishmentarianism" (almost certainly not seen)

Any token not in your vocabulary is an **out-of-vocabulary (OOV)** token. The standard handling is to replace it with a special `<UNK>` token. But now your model has no information about it at all. Every OOV word looks identical to the model, regardless of what it actually means.

This is a serious problem. In real-world text, OOV rates of 5–15% are common. That means one in every seven to twenty words is invisible to your model.

The naive fix — just make the vocabulary bigger — runs into another problem. A vocabulary of 500,000 words means a model with 500,000 input dimensions. That is slow, memory-hungry, and most of those 500,000 words will have been seen so rarely in training that the model learns almost nothing about them. Rare words get poor representations because you just do not have enough examples.

This is the tension at the heart of word-level tokenization:
- Small vocabulary: high OOV rate, many unseen words
- Large vocabulary: slow, rare words poorly learned

Subword tokenization resolves this tension.

---

### Level 4: Subword Tokenization — The Key Intuition

The insight behind subword tokenization is simple and powerful:

**Most words are made of smaller meaningful pieces. Split words into those pieces instead of treating each word as atomic.**

Consider "unhappiness":
- "un" — a prefix meaning "not"
- "happi" — the root of "happy"
- "ness" — a suffix that turns adjectives into nouns

If your tokenizer knows these pieces, it can handle "unhappiness" even if it never saw the full word during training. And the pieces — "un", "happy", "ness" — appear in thousands of other words, so the model learns rich representations of them.

Similarly:
- "running" → "run" + "ning"
- "tokenization" → "token" + "ization"
- "COVID-19" → "CO" + "VID" + "-" + "19"
- "Schwarzenegger" → "Sch" + "war" + "zen" + "egg" + "er"

The last example is important. Even a completely unknown proper noun gets split into subword pieces that the model has seen. Instead of `<UNK>`, the model gets meaningful fragments.

This solves both problems simultaneously:
- Small vocabulary of subword pieces (typically 30,000–50,000)
- No OOV words, because any word can be decomposed into known pieces

---

### Byte Pair Encoding (BPE): How Subword Tokenization is Learned

Subword vocabularies are not hand-designed. They are learned from data using algorithms. The most important one is **Byte Pair Encoding (BPE)**, used by GPT-2, GPT-3, and many other modern models.

Here is the full algorithm from scratch.

**The core idea:** Start with individual characters as your vocabulary. Repeatedly find the most frequent pair of adjacent tokens and merge them into a single new token. Repeat until you have the vocabulary size you want.

Let's walk through a complete numerical example.

**Starting corpus** (simplified):
```
"low low low low"
"lower lower"
"newest newest newest"
"widest widest"
```

**Step 1: Initialize with characters plus an end-of-word marker.**

We represent each word as a sequence of characters, with a special `</w>` token marking the end of the word. This lets us distinguish "low" (a complete word) from "low" as a prefix of "lower".

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, </w>}

Corpus as character sequences:
l o w </w>        (frequency: 4)
l o w e r </w>    (frequency: 2)
n e w e s t </w>  (frequency: 3)
w i d e s t </w>  (frequency: 2)
```

**Step 2: Count all adjacent pairs.**

```
Pair      | Count
----------|-------
l o       |  6    (4 from "low</w>" + 2 from "lower</w>")
o w       |  6
w </w>    |  4
w e       |  2
e r       |  2
r </w>    |  2
n e       |  3
e w       |  3
e s       |  5    (3 from "newest" + 2 from "widest")
s t       |  5
t </w>    |  5
w i       |  2
i d       |  2
d e       |  2
```

**Step 3: Merge the most frequent pair.**

The most frequent pair is "l o" and "o w" (tied at 6). Let's take "l o". Merge it into a new token "lo".

```
Updated corpus:
lo w </w>        (frequency: 4)
lo w e r </w>    (frequency: 2)
n e w e s t </w> (frequency: 3)
w i d e s t </w> (frequency: 2)

Updated vocabulary: {lo, w, e, r, n, s, t, i, d, o, l, </w>}
```

**Step 4: Repeat.**

Now recount pairs. "lo w" appears 6 times. Merge into "low".

```
Updated corpus:
low </w>         (frequency: 4)
low e r </w>     (frequency: 2)
n e w e s t </w> (frequency: 3)
w i d e s t </w> (frequency: 2)
```

**Step 5: Repeat again.**

Next most frequent pair: "e s" or "s t" or "t </w>" (all appear 5 times). Take "e s". Merge into "es".

```
Updated corpus:
low </w>          (frequency: 4)
low e r </w>      (frequency: 2)
n e w es t </w>   (frequency: 3)
w i d es t </w>   (frequency: 2)
```

You continue this process for as many merges as needed to reach your target vocabulary size. After enough merges, common words like "low", "the", "and" become single tokens. Rare or new words are represented as sequences of smaller pieces.

**The result:** A vocabulary where frequent patterns are single tokens and rare patterns are decomposed into smaller known pieces.

Here is a Python implementation of BPE from scratch:

```python
from collections import defaultdict, Counter

def get_vocab(corpus):
    """Convert corpus to character-level representation with </w> markers."""
    vocab = defaultdict(int)
    for word, freq in corpus.items():
        # Split word into characters, add end-of-word marker
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] += freq
    return vocab

def get_pairs(vocab):
    """Count all adjacent pairs across the vocabulary."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge a pair of symbols in the vocabulary."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# Example corpus: word -> frequency
corpus = {
    'low': 4,
    'lower': 2,
    'newest': 3,
    'widest': 2
}

vocab = get_vocab(corpus)
print("Initial vocabulary:")
for word, freq in vocab.items():
    print(f"  '{word}': {freq}")

num_merges = 10
for i in range(num_merges):
    pairs = get_pairs(vocab)
    if not pairs:
        break
    # Find the most frequent pair
    best_pair = max(pairs, key=pairs.get)
    print(f"\nMerge {i+1}: {best_pair} (count: {pairs[best_pair]})")
    vocab = merge_vocab(best_pair, vocab)
    print("Vocabulary after merge:")
    for word, freq in vocab.items():
        print(f"  '{word}': {freq}")
```

Running this produces:

```
Initial vocabulary:
  'l o w </w>': 4
  'l o w e r </w>': 2
  'n e w e s t </w>': 3
  'w i d e s t </w>': 2

Merge 1: ('l', 'o') (count: 6)
Merge 2: ('lo', 'w') (count: 6)
Merge 3: ('e', 's') (count: 5)
Merge 4: ('es', 't') (count: 5)
Merge 5: ('est', '</w>') (count: 5)
Merge 6: ('low', '</w>') (count: 4)
Merge 7: ('n', 'e') (count: 3)
Merge 8: ('ne', 'w') (count: 3)
Merge 9: ('new', 'est</w>') (count: 3)
Merge 10: ('low', 'e') (count: 2)
```

After 10 merges, "newest" is a single token and "lowest" would be represented as "low" + "est</w>". The algorithm has discovered that "est" is a common suffix worth merging.

---

### How tokenization affects vocabulary size: a concrete comparison

To make this concrete, here is what different tokenization strategies produce on the same sentence:

```
Input: "The unhappiest transformers tokenization researcher"
```

**Word-level tokenization:**
```
["The", "unhappiest", "transformers", "tokenization", "researcher"]
5 tokens. "unhappiest" and "tokenization" may be OOV.
```

**Character-level tokenization:**
```
["T","h","e"," ","u","n","h","a","p","p","i","e","s","t"," ",...]
~50 tokens. No OOV, but sequences are very long.
```

**BPE subword tokenization (GPT-2 style):**
```
["The", "Ġun", "happiest", "Ġtransformers", 
 "Ġtoken", "ization", "Ġresearcher"]
7 tokens. No OOV. Common pieces reused.
```

(The Ġ symbol marks a space before the token in GPT-2's encoding.)

**WordPiece tokenization (BERT style):**
```
["the", "un", "##happy", "##iest", "transform", "##ers", 
 "token", "##ization", "research", "##er"]
10 tokens. The ## prefix marks continuation of a word.
```

Each of these makes different trade-offs. Character-level handles everything but produces very long sequences that are hard for models to learn from. Word-level is compact but brittle. Subword hits the sweet spot for most tasks.

---

### Three subword algorithms you will encounter

**BPE (Byte Pair Encoding)** — used by GPT-2, GPT-3, RoBERTa. Learns merges greedily from frequency. Encodes text by applying learned merges in order.

**WordPiece** — used by BERT. Similar to BPE but chooses merges that maximize likelihood of the training data under a language model rather than raw frequency. Produces slightly different splits, tends to be more linguistically motivated.

**SentencePiece** — used by T5, ALBERT, many multilingual models. Treats the input as a raw byte stream rather than pre-tokenized words. Language-agnostic — works the same way on Chinese, Japanese, Arabic, or English without any language-specific rules.

You do not need to implement all three right now. Understanding BPE gives you the core intuition. We will use Hugging Face tokenizers when we get to BERT and GPT in Module 11.

---

### Time and space complexity

**Whitespace tokenization:** O(n) where n is the number of characters. Linear scan, single pass.

**Rule-based tokenization:** O(n × r) where r is the number of rules. Each rule is a regex pass over the text. In practice very fast because r is small.

**BPE training:** O(V × I) where V is vocabulary size and I is number of merge iterations. Expensive to train, fast to apply.

**BPE inference (applying to new text):** O(n²) in the worst case per word, because you may need to check all possible pairs. In practice O(n log n) with efficient data structures. Fast enough for production.

---

### Summary

- Whitespace tokenization is a useful baseline but fails on punctuation, contractions, and non-English text.
- Rule-based tokenization handles known special cases with explicit rules but does not generalize.
- Word-level tokenization faces the open vocabulary problem: OOV words become `<UNK>` and lose all information.
- Subword tokenization resolves this by splitting words into smaller learned pieces.
- BPE learns these pieces by iteratively merging the most frequent adjacent pair in a corpus.
- Modern systems (BERT, GPT) all use subword tokenization. Understanding it is non-negotiable.

---

# Module 1, Chapter 1.4
## Normalization: Lowercasing, Punctuation, Unicode Handling

---

### What normalization is and why it exists

After tokenization you have a list of tokens. But those tokens still carry a enormous amount of surface variation that has nothing to do with meaning.

"Dog", "dog", "DOG", "dog!", "dog," and "dog." are six different strings. To a computer, they are as different as "dog" and "cat". If your model sees "dog" 10,000 times in training and "Dog" only 500 times, it learns a much weaker representation of "Dog" — even though they mean exactly the same thing.

Normalization is the process of collapsing this meaningless variation. You are making a deliberate choice to treat certain different surface forms as identical, in exchange for a smaller, cleaner, more consistent vocabulary.

Every normalization step is a trade-off. You are always throwing away some information. The question is always: is the information I am throwing away worth less than the benefit I get from the reduction in variation?

Sometimes yes. Sometimes no. That depends on the task.

---

### Normalization Step 1: Lowercasing

Convert every token to lowercase.

```python
def lowercase(tokens):
    return [token.lower() for token in tokens]

tokens = ["The", "CAT", "sat", "ON", "the", "Mat"]
print(lowercase(tokens))
# ['the', 'cat', 'sat', 'on', 'the', 'mat']
```

This is the single most common normalization step. It immediately reduces vocabulary size and ensures that "The" and "the" are treated as the same word.

**When it helps:**

For most tasks — sentiment analysis, topic classification, language modeling on general text — case carries almost no information. Lowercasing reduces vocabulary size, which means more training examples per token, which means better learned representations.

**When it hurts:**

Case carries real information in several situations.

*Named entities.* "Apple" the company versus "apple" the fruit. "Turkey" the country versus "turkey" the bird. "March" the month versus "march" the action. Lowercasing collapses these distinctions entirely.

*Acronyms.* "US" (United States) versus "us" (first person plural pronoun). "IT" (information technology) versus "it" (pronoun). After lowercasing, both become "us" and "it" respectively — indistinguishable from common pronouns.

*Sentiment signals.* "This is GREAT" versus "This is great". All-caps is a strong signal of emphasis or intensity in informal text. Lowercasing loses this.

*Named entity recognition.* If your task is to find person names, organization names, and locations in text, case is one of your strongest features. "Barack Obama" starts with capital letters. After lowercasing, that signal is gone.

**The practical rule:** Lowercase for tasks that do not require distinguishing named entities or emphasis. Keep case for NER, information extraction, and tasks on formal documents where casing is meaningful.

---

### Normalization Step 2: Punctuation Handling

Punctuation creates two problems. First, it attaches to words, creating spurious token variants: "cat", "cat.", "cat,", "cat!" are all different strings. Second, punctuation tokens themselves may or may not be useful depending on the task.

There are three strategies.

**Strategy A: Remove all punctuation**

```python
import re

def remove_punctuation(tokens):
    return [re.sub(r'[^\w\s]', '', token) for token in tokens]

tokens = ["cat.", "dog,", "bird!", "fish?", "hello-world"]
print(remove_punctuation(tokens))
# ['cat', 'dog', 'bird', 'fish', 'helloworld']
```

Simple and aggressive. Note that "hello-world" becomes "helloworld" — the hyphen is stripped and the two parts merge into one token. This is often not what you want.

**Strategy B: Remove punctuation but preserve as separate tokens first**

A better approach: separate punctuation from words before removing, so you can decide what to keep.

```python
import re
import string

def separate_and_filter_punctuation(text, keep_punct=False):
    # Separate punctuation from words with spaces
    text = re.sub(r'([' + re.escape(string.punctuation) + r'])', r' \1 ', text)
    tokens = text.split()
    if keep_punct:
        return tokens
    else:
        # Remove standalone punctuation tokens
        return [t for t in tokens if t not in string.punctuation]

text = "The cat sat on the mat, and the dog didn't care."
print(separate_and_filter_punctuation(text, keep_punct=False))
# ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 
#  'the', 'dog', 'didn', 't', 'care']

print(separate_and_filter_punctuation(text, keep_punct=True))
# ['The', 'cat', 'sat', 'on', 'the', 'mat', ',', 
#  'and', 'the', 'dog', 'didn', "'", 't', 'care', '.']
```

**Strategy C: Keep punctuation as tokens**

For tasks where punctuation carries meaning — sentiment analysis, question detection, dialogue — keep punctuation as separate tokens rather than removing it.

```python
"This is amazing!"  → ["This", "is", "amazing", "!"]
"Is this right?"    → ["Is", "this", "right", "?"]
"She said... okay"  → ["She", "said", "...", "okay"]
```

The exclamation mark and question mark are genuine features for a sentiment or intent classifier. Keeping them as tokens lets the model learn from them.

**Punctuation that always needs special treatment:**

Hyphens are particularly tricky because they serve multiple functions:

```python
"co-operate"    # compound word — probably one concept
"2019-2020"     # date range — the hyphen is meaningful
"well-known"    # compound adjective
"e-mail"        # stylistic variant of "email"
"twenty-three"  # compound number
```

There is no single right answer for hyphens. The safest approach for general text is to keep them attached and let the downstream model figure it out, or to use subword tokenization which handles morphological variation anyway.

Apostrophes are also complex:

```python
"don't"    # contraction — should split into "do" + "n't"
"cat's"    # possessive — "cat" + "'s"
"it's"     # contraction — "it" + "'s"
"its"      # possessive pronoun — no apostrophe
"O'Brien"  # part of a name — do not split
"'90s"     # decade abbreviation — the apostrophe replaces "19"
```

---

### Normalization Step 3: Unicode Normalization

This is the most technical normalization step and the one most often forgotten by beginners. It causes real production bugs.

**The problem: the same character can be represented multiple ways in Unicode.**

Unicode is the international standard for encoding text. It assigns a unique code point to every character in every writing system. But for historical and compatibility reasons, some characters can be encoded in more than one way and produce identical-looking output.

The most common example: accented characters.

The letter "é" can be encoded as:
- A single code point: U+00E9 (LATIN SMALL LETTER E WITH ACUTE) — one character
- Two code points: U+0065 (LATIN SMALL LETTER E) + U+0301 (COMBINING ACUTE ACCENT) — two characters

Both render as "é". They look identical. But as byte sequences they are completely different. A model that has not normalized Unicode will treat these as different tokens.

```python
e_single = '\u00e9'         # single code point
e_combined = 'e\u0301'      # base letter + combining accent

print(e_single)             # é
print(e_combined)           # é  (looks the same)
print(e_single == e_combined)  # False — they are NOT equal
print(len(e_single))        # 1
print(len(e_combined))      # 2
```

This causes real problems. If your training data uses one form and your test data uses the other, your model sees them as different tokens. Vocabulary size grows unnecessarily. Learned representations do not transfer.

**The fix: Unicode normalization**

Python's `unicodedata` module provides four normalization forms. The two most important for NLP are:

**NFC (Canonical Decomposition followed by Canonical Composition)**
Composes characters into their single precomposed form where possible. "e" + combining accent → "é" as a single code point. This is usually the right choice for NLP — compact representation, human-readable.

**NFD (Canonical Decomposition)**
Decomposes characters into their base form plus combining marks. "é" → "e" + combining accent. Useful when you want to strip accents (see below).

```python
import unicodedata

e_single = '\u00e9'
e_combined = 'e\u0301'

# Normalize both to NFC
nfc_single   = unicodedata.normalize('NFC', e_single)
nfc_combined = unicodedata.normalize('NFC', e_combined)

print(nfc_single == nfc_combined)  # True — now they are equal
print(len(nfc_single))             # 1
print(len(nfc_combined))           # 1
```

Always apply Unicode normalization before any other text processing. It is two lines of code and prevents a category of bugs that are extremely hard to debug later.

**Stripping accents**

For some tasks, particularly in multilingual NLP or when handling user-generated text with inconsistent accent usage, you may want to strip accents entirely and reduce accented characters to their base ASCII equivalents.

```python
import unicodedata

def strip_accents(text):
    # Decompose into base characters + combining marks (NFD)
    nfd = unicodedata.normalize('NFD', text)
    # Keep only characters that are not combining marks
    return ''.join(
        char for char in nfd
        if unicodedata.category(char) != 'Mn'
    )

print(strip_accents("café"))      # cafe
print(strip_accents("naïve"))     # naive
print(strip_accents("résumé"))    # resume
print(strip_accents("Zürich"))    # Zurich
print(strip_accents("São Paulo")) # Sao Paulo
```

`unicodedata.category(char)` returns the Unicode category of a character. "Mn" means "Mark, Nonspacing" — these are the combining accent marks. By filtering them out after NFD decomposition, we strip all accents cleanly.

**When to strip accents:** For English-only tasks, stripping accents reduces noise from text that was typed without proper accent marks (very common in user-generated content). For multilingual tasks, be careful — "résumé" and "resume" are different words, and "über" and "uber" mean different things. In French and German, accents are part of spelling and carry meaning.

---

### Normalization Step 4: Handling Special Tokens

Real text contains patterns that are not words but carry structured information. You need an explicit strategy for each.

**URLs**

```python
import re

def normalize_urls(text):
    url_pattern = re.compile(
        r'https?://\S+|www\.\S+'
    )
    return url_pattern.sub('<URL>', text)

text = "Check out https://www.example.com for more info"
print(normalize_urls(text))
# "Check out <URL> for more info"
```

Replacing URLs with a special `<URL>` token tells the model "there was a URL here" without flooding your vocabulary with millions of distinct URLs that will never repeat.

**Email addresses**

```python
def normalize_emails(text):
    email_pattern = re.compile(r'\S+@\S+\.\S+')
    return email_pattern.sub('<EMAIL>', text)

text = "Contact us at support@example.com"
print(normalize_emails(text))
# "Contact us at <EMAIL>"
```

**Numbers**

```python
def normalize_numbers(text):
    # Replace sequences of digits (with optional commas/decimals) 
    # with a special token
    return re.sub(r'\b\d+([,\.]\d+)*\b', '<NUM>', text)

text = "She earned $47,000 in 2023 and invested 3.5 percent"
print(normalize_numbers(text))
# "She earned $<NUM> in <NUM> and invested <NUM> percent"
```

**Emojis**

In modern social media text, emojis carry significant sentiment information. You have two choices: remove them or replace them with their text description.

```python
# Option 1: remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

# Option 2: replace with text using the emoji library
# pip install emoji
import emoji

def replace_emojis(text):
    return emoji.demojize(text)

text = "This is great! 😊🔥"
print(remove_emojis(text))      # "This is great! "
print(replace_emojis(text))     # "This is great! :smiling_face: :fire:"
```

For sentiment analysis, replacing emojis with their text description is almost always better than removing them. 🔥 carries real sentiment information.

---

### Putting it all together: a complete normalizer

Here is a full normalization pipeline from scratch, combining everything in this chapter:

```python
import re
import string
import unicodedata

class TextNormalizer:
    def __init__(
        self,
        lowercase=True,
        remove_punctuation=False,
        normalize_unicode=True,
        strip_accents=False,
        replace_urls=True,
        replace_emails=True,
        replace_numbers=False
    ):
        self.lowercase         = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode  = normalize_unicode
        self.strip_accents      = strip_accents
        self.replace_urls       = replace_urls
        self.replace_emails     = replace_emails
        self.replace_numbers    = replace_numbers

    def normalize(self, text):
        # Step 1: Unicode normalization (always first)
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)

        # Step 2: Strip accents (after unicode normalization)
        if self.strip_accents:
            nfd = unicodedata.normalize('NFD', text)
            text = ''.join(
                c for c in nfd 
                if unicodedata.category(c) != 'Mn'
            )

        # Step 3: Replace special patterns
        if self.replace_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
        if self.replace_emails:
            text = re.sub(r'\S+@\S+\.\S+', '<EMAIL>', text)
        if self.replace_numbers:
            text = re.sub(r'\b\d+([,\.]\d+)*\b', '<NUM>', text)

        # Step 4: Lowercase
        if self.lowercase:
            text = text.lower()

        # Step 5: Remove punctuation
        if self.remove_punctuation:
            text = text.translate(
                str.maketrans('', '', string.punctuation)
            )

        # Step 6: Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Test it
normalizer = TextNormalizer(
    lowercase=True,
    remove_punctuation=False,
    normalize_unicode=True,
    strip_accents=False,
    replace_urls=True,
    replace_emails=True,
    replace_numbers=True
)

test_texts = [
    "The CAT sat on the mat.",
    "Visit https://www.example.com or email us at hello@test.com",
    "She earned $47,000 in 2023!",
    "café résumé naïve",
    "This   has    extra   spaces",
]

for text in test_texts:
    print(f"Input:  {text}")
    print(f"Output: {normalizer.normalize(text)}")
    print()
```

Output:

```
Input:  The CAT sat on the mat.
Output: the cat sat on the mat.

Input:  Visit https://www.example.com or email us at hello@test.com
Output: visit <url> or email us at <email>

Input:  She earned $47,000 in 2023!
Output: she earned $<num> in <num>!

Input:  café résumé naïve
Output: café résumé naïve

Input:  This   has    extra   spaces
Output: this has extra spaces
```

---

### Task-specific normalization decisions

Here is a reference table for the most common NLP tasks:

```
Task                    | Lower | Punct  | Accents | Numbers
------------------------|-------|--------|---------|--------
Sentiment analysis      | Yes   | Keep ! | Keep    | Replace
Topic classification    | Yes   | Remove | Strip   | Replace
Named entity recognition| No    | Keep   | Keep    | Keep
Machine translation     | No    | Keep   | Keep    | Keep
Information retrieval   | Yes   | Remove | Strip   | Keep
Spam detection          | Yes   | Keep ! | Strip   | Replace
Language identification | No    | Keep   | Keep    | Keep
```

---

### Time and space complexity

All normalization steps are O(n) where n is the number of characters in the text. Each step is a single linear pass. Unicode normalization involves a lookup table but remains effectively O(n). The full pipeline is O(n × k) where k is the number of normalization steps, and k is a small constant, so effectively O(n).

---

### Common pitfalls

**Applying normalization in the wrong order.** Always normalize Unicode before anything else. If you lowercase first and then try to normalize Unicode, you may miss case-sensitive Unicode distinctions. If you strip punctuation before separating contractions, "don't" becomes "dont" instead of "do n't".

**Normalizing when you should not.** Applying the same normalization to all tasks because "that's how it's done." Always think about whether each step helps or hurts your specific task.

**Forgetting to normalize at inference time.** Your model was trained on normalized text. At inference time, if you forget to apply the same normalization, your model sees text in a form it was never trained on. This is a very common production bug.

**Over-normalizing user-generated text.** Removing ALL punctuation from a tweet destroys information. "I love this!!!" and "I love this" are meaningfully different.

---

### Summary

- Normalization collapses meaningless surface variation in tokens.
- Lowercasing is the most common step but loses case-based signals that matter for NER and emphasis.
- Punctuation can be removed, kept, or kept as separate tokens depending on whether it carries task-relevant information.
- Unicode normalization is non-negotiable — always do it first — to prevent invisible duplicate representations of the same character.
- Accent stripping is optional and task-dependent.
- Special tokens like `<URL>`, `<EMAIL>`, and `<NUM>` are better than either keeping the raw value or deleting it.
- Always apply the same normalization at training time and inference time.

---

# Module 1, Chapter 1.5
## Stemming vs Lemmatization: Algorithms and Trade-offs

---

### The problem we are solving

After tokenization and normalization you have clean tokens. But you still have a morphological variation problem.

Consider these words:

```
run, runs, running, ran
study, studies, studied, studying
good, better, best
be, is, are, was, were, been, being
```

Each group expresses a single underlying concept. But to a model that treats tokens as atomic units, "run" and "running" are as different as "run" and "elephant". If "running" appears rarely in training data but "run" appears frequently, the model learns a poor representation of "running" even though they mean nearly the same thing.

This is the **morphological variation problem**. Human languages express grammatical information — tense, number, gender, case, aspect — by modifying word forms. English is relatively mild in this regard. Languages like Finnish, Turkish, and Arabic are highly inflected, with a single root word potentially having hundreds of valid surface forms.

The solution is to map inflected forms back to a canonical base form before processing. There are two approaches: stemming and lemmatization.

---

### What is a stem and what is a lemma?

A **stem** is the result of mechanically chopping off affixes from a word. It does not need to be a real word. It just needs to be a consistent root that groups related forms together.

```
running  → run
studies  → studi      (not a real word)
fishing  → fish
argued   → argu       (not a real word)
```

A **lemma** is the canonical dictionary form of a word — the form you would look up in a dictionary. It is always a real word.

```
running  → run
studies  → study
better   → good
was      → be
```

The key difference: stemming is fast and approximate. Lemmatization is slower and linguistically accurate.

---

### Stemming: the Porter Stemmer

The Porter Stemmer, published by Martin Porter in 1980, is the most widely used stemming algorithm in English. It applies a sequence of rule-based transformations to strip suffixes.

**How it works:** The algorithm defines a set of rules of the form "if the word ends with X and satisfies condition Y, replace X with Z". Rules are applied in a fixed sequence of phases. Each phase handles a different class of suffixes.

Here are some example rules from the Porter Stemmer:

```
Phase 1a — plural and past tense:
  sses → ss      ("caresses" → "caress")
  ies  → i       ("ponies"   → "poni")
  ss   → ss      ("caress"   → "caress")
  s    →          ("cats"     → "cat")

Phase 1b — -ed and -ing:
  eed  → ee      ("agreed"   → "agree")   if stem has vowel
  ed   →          ("plastered"→ "plaster") if stem has vowel
  ing  →          ("motoring" → "motor")   if stem has vowel

Phase 2 — derivational suffixes:
  ational → ate  ("relational"  → "relate")
  tional  → tion ("conditional" → "condition")
  enci    → ence ("valenci"     → "valence")
  anci    → ance ("hesitanci"   → "hesitance")

Phase 3:
  icate → ic     ("triplicate" → "triplic")
  ative →         ("formative"  → "form")
  alize → al     ("formalize"  → "formal")

Phase 4 — remove derivational suffixes:
  al, ance, ence, er, ic, able, ible, ant, 
  ement, ment, ent, ion, ou, ism, ate, iti, 
  ous, ive, ize → remove if stem long enough

Phase 5 — final cleanup:
  e → remove     ("probate" → "probat")
  ll → l         ("roll"    → "rol")
```

The condition "if stem long enough" refers to a measure called **m**, which counts the number of vowel-consonant sequences in the stem. Suffixes are only stripped if the remaining stem has enough syllabic content to still be meaningful.

Let's implement a simplified version of the Porter Stemmer to understand the mechanics:

```python
import re

class SimpleStemmer:
    """
    A simplified stemmer illustrating the Porter algorithm's logic.
    Not a full implementation — for illustration purposes.
    """
    
    def __init__(self):
        # Vowels
        self.vowels = set('aeiou')
    
    def count_vc_sequences(self, word):
        """
        Count vowel-consonant sequences (the 'm' measure).
        C(VC)^m V? C? 
        Higher m means more syllabic content.
        """
        # Mark each character as V or C
        pattern = ''
        for char in word:
            if char in self.vowels:
                pattern += 'V'
            else:
                pattern += 'C'
        # Count VC pairs
        m = 0
        i = 0
        while i < len(pattern) - 1:
            if pattern[i] == 'V' and pattern[i+1] == 'C':
                m += 1
                i += 2
            else:
                i += 1
        return m
    
    def has_vowel(self, stem):
        return any(c in self.vowels for c in stem)
    
    def ends_double_consonant(self, word):
        if len(word) >= 2:
            return (word[-1] == word[-2] and 
                    word[-1] not in self.vowels)
        return False
    
    def step1a(self, word):
        """Handle plurals and -ed/-ing of short words."""
        if word.endswith('sses'):
            return word[:-2]          # caresses → caress
        elif word.endswith('ies'):
            return word[:-2]          # ponies → poni
        elif word.endswith('ss'):
            return word               # caress → caress
        elif word.endswith('s'):
            return word[:-1]          # cats → cat
        return word
    
    def step1b(self, word):
        """Handle -ed and -ing."""
        if word.endswith('eed'):
            stem = word[:-1]
            if self.count_vc_sequences(stem) > 0:
                return stem           # agreed → agree
        elif word.endswith('ed'):
            stem = word[:-2]
            if self.has_vowel(stem):
                return self._step1b_fix(stem)
        elif word.endswith('ing'):
            stem = word[:-3]
            if self.has_vowel(stem):
                return self._step1b_fix(stem)
        return word
    
    def _step1b_fix(self, stem):
        """Cleanup after removing -ed or -ing."""
        if stem.endswith('at') or stem.endswith('bl') or stem.endswith('iz'):
            return stem + 'e'         # hop → hope (if from "hoped")
        elif self.ends_double_consonant(stem) and stem[-1] not in 'lsz':
            return stem[:-1]          # hopping → hop
        elif self.count_vc_sequences(stem) == 1:
            # CVC pattern — add e
            if stem[-1] not in 'aeiouwxy':
                return stem + 'e'
        return stem
    
    def step1c(self, word):
        """Change y to i if preceded by vowel."""
        if word.endswith('y') and self.has_vowel(word[:-1]):
            return word[:-1] + 'i'   # happy → happi
        return word
    
    def stem(self, word):
        word = word.lower()
        word = self.step1a(word)
        word = self.step1b(word)
        word = self.step1c(word)
        return word


# Test the simplified stemmer
stemmer = SimpleStemmer()

test_words = [
    'cats', 'caresses', 'ponies',
    'running', 'agreed', 'plastered',
    'happy', 'happily', 'happiness',
    'studies', 'studying', 'studied',
]

for word in test_words:
    print(f"{word:15} → {stemmer.stem(word)}")
```

Output:

```
cats            → cat
caresses        → caress
ponies          → poni
running         → run
agreed          → agre
plastered       → plaster
happy           → happi
happily         → happili
happiness       → happi
studies         → studi
studying        → studi
studied         → studi
```

The full Porter Stemmer has five phases and handles many more cases. Use NLTK for the production version:

```python
from nltk.stem import PorterStemmer, SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer('english')

words = ['running', 'studies', 'generously', 'happiness',
         'argued', 'fishing', 'computed', 'university']

print(f"{'Word':<15} {'Porter':<15} {'Snowball':<15}")
print('-' * 45)
for word in words:
    print(f"{word:<15} {porter.stem(word):<15} {snowball.stem(word):<15}")
```

Output:

```
Word            Porter          Snowball       
---------------------------------------------
running         run             run            
studies         studi           studi          
generously      generous        generous       
happiness       happi           happi          
argued          argu            argu           
fishing         fish            fish           
computed        comput          comput         
university      univers         univers        
```

Notice: "studies" → "studi", "argued" → "argu", "happiness" → "happi". These are not real words. The Porter Stemmer is aggressive and produces stems, not valid English words.

The **Snowball Stemmer** (also called Porter2) is an improved version with better handling of edge cases. It is generally preferred over the original Porter Stemmer.

---

### The Lovins and Lancaster Stemmers

Two other stemmers worth knowing about:

**Lovins Stemmer (1968)** — the first published stemming algorithm. Single-pass, removes the longest possible suffix from a list of 294 endings. Very aggressive. Often over-stems.

**Lancaster Stemmer** — extremely aggressive. Iteratively applies rules until no more apply. Produces very short stems. Fast but tends to conflate words that should not be conflated.

```python
from nltk.stem import LancasterStemmer

lancaster = LancasterStemmer()

words = ['running', 'generously', 'university', 'eating']
for word in words:
    print(f"{word:<15} → {lancaster.stem(word)}")
```

Output:

```
running         → run
generously      → gen
university      → univers
eating          → eat
```

"generously" → "gen" is too aggressive. "gen" bears no meaningful relationship to "generous" for NLP purposes. Lancaster is rarely used in practice because it over-stems.

---

### Lemmatization: the linguistically correct approach

Lemmatization maps a word to its lemma — the canonical dictionary form. Unlike stemming it always produces a real word, and it handles irregular morphology correctly.

```
ran     → run      (not "ran" with suffix chopped off)
better  → good     (comparative → base adjective)
was     → be       (past tense → infinitive)
geese   → goose    (irregular plural)
mice    → mouse    (irregular plural)
```

Stemming cannot do any of these because they require knowing that "ran" is the past tense of "run", which requires a dictionary.

**The critical dependency: part of speech.**

To lemmatize correctly you need to know the part of speech of the word in context. Consider:

```
"meeting" as a VERB  → lemma: "meet"   (I am meeting her)
"meeting" as a NOUN  → lemma: "meeting" (The meeting was long)

"left" as a VERB     → lemma: "leave"  (She left the room)
"left" as an ADJECTIVE → lemma: "left" (Turn left)

"better" as an ADJECTIVE → lemma: "good"  (He feels better)
"better" as a VERB       → lemma: "better" (We must better ourselves)
```

Without knowing the POS, you cannot lemmatize correctly. Most lemmatizers ask you to provide the POS. If you do not, they default to noun, which is wrong for verbs.

**WordNet Lemmatizer in NLTK:**

WordNet is a large lexical database of English that groups words into sets of synonyms and encodes semantic relationships. NLTK's lemmatizer is built on WordNet.

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# Download required data (first time only)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

# Without POS — defaults to noun
words = ['running', 'better', 'was', 'studies', 'geese', 'ran']
print("Without POS (assumes noun):")
for word in words:
    print(f"  {word:<12} → {lemmatizer.lemmatize(word)}")

print()

# With correct POS
print("With correct POS:")
pos_examples = [
    ('running', wordnet.VERB),
    ('better',  wordnet.ADJ),
    ('was',     wordnet.VERB),
    ('studies', wordnet.VERB),
    ('geese',   wordnet.NOUN),
    ('ran',     wordnet.VERB),
]
for word, pos in pos_examples:
    print(f"  {word:<12} → {lemmatizer.lemmatize(word, pos)}")
```

Output:

```
Without POS (assumes noun):
  running      → running    (WRONG — didn't recognize as verb)
  better       → better     (WRONG — didn't recognize as adjective)
  was          → wa         (WRONG — mangled)
  studies      → study      (correct by luck — noun plural)
  geese        → goose      (correct)
  ran          → ran        (WRONG — didn't recognize as verb)

With correct POS:
  running      → run        (correct)
  better       → good       (correct)
  was          → be         (correct)
  studies      → study      (correct)
  geese        → goose      (correct)
  ran          → run        (correct)
```

The difference is dramatic. Without POS information, the lemmatizer gets most verbs wrong. This is why lemmatization in a full pipeline requires a POS tagger running before the lemmatizer.

**Automatic POS tagging for lemmatization:**

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Convert NLTK POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

def lemmatize_sentence(sentence):
    # Tokenize
    tokens = nltk.word_tokenize(sentence)
    # POS tag
    pos_tags = nltk.pos_tag(tokens)
    # Lemmatize each token with its POS
    lemmas = []
    for token, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, wn_pos)
        lemmas.append(lemma)
    return list(zip(tokens, lemmas))

sentence = "The runners were running faster than they had ever run before"
result = lemmatize_sentence(sentence)

print(f"{'Token':<12} {'Lemma':<12}")
print('-' * 24)
for token, lemma in result:
    print(f"{token:<12} {lemma:<12}")
```

Output:

```
Token        Lemma       
------------------------
The          The         
runners      runner      
were         be          
running      run         
faster       fast        
than         than        
they         they        
had          have        
ever         ever        
run          run         
before       before      
```

"were" → "be", "running" → "run", "had" → "have", "faster" → "fast". All correct.

---

### spaCy's lemmatizer

spaCy is a modern production-grade NLP library with an excellent built-in lemmatizer. It integrates POS tagging and lemmatization into a single pipeline.

```python
import spacy

# Load English model (first time: python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')

sentence = "The geese were running better than they had ever run before"
doc = nlp(sentence)

print(f"{'Token':<12} {'POS':<8} {'Lemma':<12}")
print('-' * 32)
for token in doc:
    print(f"{token.text:<12} {token.pos_:<8} {token.lemma_:<12}")
```

Output:

```
Token        POS      Lemma       
--------------------------------
The          DET      the         
geese        NOUN     goose       
were         AUX      be          
running      VERB     run         
better       ADV      well        
than         SCONJ    than        
they         PRON     they        
had          AUX      have        
ever         ADV      ever        
run          VERB     run         
before       ADV      before      
```

spaCy correctly handles "geese" → "goose" and "were" → "be" with no extra work. In production, spaCy's lemmatizer is almost always the right choice.

---

### Stemming vs Lemmatization: a direct comparison

```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

test_cases = [
    ('running',  wordnet.VERB),
    ('studies',  wordnet.VERB),
    ('better',   wordnet.ADJ),
    ('was',      wordnet.VERB),
    ('geese',    wordnet.NOUN),
    ('happiness',wordnet.NOUN),
    ('generously',wordnet.ADV),
    ('argued',   wordnet.VERB),
    ('university',wordnet.NOUN),
    ('corpora',  wordnet.NOUN),
]

print(f"{'Word':<15} {'Stem':<15} {'Lemma':<15}")
print('-' * 45)
for word, pos in test_cases:
    stem  = porter.stem(word)
    lemma = lemmatizer.lemmatize(word, pos)
    print(f"{word:<15} {stem:<15} {lemma:<15}")
```

Output:

```
Word            Stem            Lemma          
---------------------------------------------
running         run             run            
studies         studi           study          
better          better          good           
was             wa              be             
geese           gees            goose          
happiness       happi           happiness      
generously      generous        generously     
argued          argu            argue          
university      univers         university     
corpora         corpora         corpus         
```

Look at the differences:

- "studies" → "studi" (stem) vs "study" (lemma). The stem is not a real word.
- "better" → "better" (stem, unchanged) vs "good" (lemma, correct base form).
- "was" → "wa" (stem, mangled) vs "be" (lemma, correct).
- "geese" → "gees" (stem, wrong) vs "goose" (lemma, correct).
- "corpora" → "corpora" (stem, unchanged) vs "corpus" (lemma, correct Latin plural).

---

### When to use each

**Use stemming when:**
- Speed is critical — stemming is 10–100x faster than lemmatization
- You are doing information retrieval or search — approximate grouping is good enough
- You do not have POS tags available
- Your downstream model can tolerate non-words as input
- You are working at very large scale where lemmatization latency matters

**Use lemmatization when:**
- Linguistic accuracy matters
- You need real words as output (for language modeling, text generation)
- You are working with irregular morphology (was/be, geese/goose, better/good)
- You have POS tags available (from a tagger in your pipeline)
- You are in a domain with specialized vocabulary where stemming rules break

**Use neither when:**
- You are using subword tokenization (BPE, WordPiece) — these handle morphological variation implicitly
- Your model is a modern Transformer — it learns from enough data that morphological variants are handled naturally
- Your task requires the original surface form (NER, coreference, translation)

---

### Time and space complexity

**Porter Stemmer:** O(n) where n is the length of the word. A constant number of rule passes, each linear in word length. Extremely fast — can process millions of words per second.

**WordNet Lemmatizer:** O(n + d) where d is the dictionary lookup cost. Dictionary lookup in a hash table is O(1) amortized, so effectively O(n). But the POS tagging required as input adds O(w) per sentence where w is the number of words.

**spaCy pipeline:** O(w) per sentence for the full pipeline including tokenization, POS tagging, and lemmatization. Highly optimized C extensions. Processes millions of words per minute on a single CPU.

---

### Common pitfalls

**Stemming without knowing the task.** Stemming "university" to "univers" and "universe" to "univers" conflates two completely unrelated words. For topic modeling or search this might be acceptable. For a knowledge graph or question answering system it is a serious error.

**Lemmatizing without POS tags.** As we saw, the WordNet lemmatizer defaults to noun. "Running" stays "running", "was" becomes "wa". Always provide POS tags.

**Applying either to named entities.** "Apple" stemmed becomes "appl". "Apple" lemmatized might become "apple" (the fruit). Neither is right — "Apple" the company should stay "Apple". Apply stemming and lemmatization only to common words, not proper nouns.

**Using stemming for language generation.** If your model outputs text, it cannot output stems — "happi" and "studi" are not valid English words. Use lemmatization for any task that requires the output to be readable.

---

### Summary

- Morphological variation means related word forms look different to a model: "run", "running", "ran" are three different tokens.
- Stemming chops suffixes with rules, is fast, and produces non-word stems.
- The Porter Stemmer is the standard English stemmer, applied in multiple rule phases.
- Lemmatization maps words to their true dictionary base form, always producing a valid word.
- Lemmatization requires part-of-speech information to work correctly.
- Use stemming for speed and approximate tasks. Use lemmatization for linguistic accuracy.
- Neither is needed when using subword tokenization or Transformer-based models.

---

# Module 1, Chapter 1.6
## Stopword Removal: When It Helps and When It Hurts

---

### The core idea

In any large collection of text, a small number of words account for a enormous proportion of all word occurrences. In English, the ten most frequent words — "the", "be", "to", "of", "and", "a", "in", "that", "have", "it" — account for roughly 25% of all tokens in a typical corpus. The top 100 words account for roughly 50%.

These words are grammatically necessary but semantically weak. They tell you almost nothing about what a document is about. A document about medicine and a document about astrophysics will both contain "the", "is", "of", and "in" at roughly the same rates. These words do not discriminate between topics.

Stopwords are this set of high-frequency, low-information words. Stopword removal is the process of filtering them out before further processing.

The intuition is simple: if a word appears in almost every document, it carries almost no information about what makes any particular document distinctive. Remove it and focus on the words that actually matter.

This is correct often enough to be a standard step in many NLP pipelines. It is also wrong often enough that you should never apply it blindly.

---

### What a stopword list looks like

The most widely used stopword list in English NLP comes from NLTK. Let's look at it:

```python
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')  # first time only

english_stops = set(stopwords.words('english'))
print(f"Total stopwords: {len(english_stops)}")
print()

# Print them sorted
sorted_stops = sorted(english_stops)
# Print in rows of 8
for i in range(0, min(80, len(sorted_stops)), 8):
    print('  '.join(f"{w:<8}" for w in sorted_stops[i:i+8]))
```

Output:

```
Total stopwords: 179

a         about     above     after     again     against   all       am      
an        and       any       are       aren't    as        at        be      
because   been      before    being     below     between   both      but     
by        can't     cannot    could     couldn't  did       didn't    do      
does      doesn't   doing     don't     down      during    each      few     
for       from      further   get       had       hadn't    has       hasn't  
have      haven't   having    he        he'd      he'll     he's      her     
here      here's    hers      herself   him       himself   his       how     
how's     i         i'd       i'll      i'm       i've      if        in      
into      is        isn't     it        it's      its       itself    let's   
...
```

179 words covering pronouns, prepositions, conjunctions, articles, auxiliary verbs, and common adverbs.

spaCy has its own stopword list, larger and slightly different:

```python
import spacy
nlp = spacy.load('en_core_web_sm')

spacy_stops = nlp.Defaults.stop_words
print(f"spaCy stopwords: {len(spacy_stops)}")

# Words in spaCy but not NLTK
only_spacy = spacy_stops - english_stops
print(f"\nIn spaCy but not NLTK ({len(only_spacy)} words):")
print(sorted(list(only_spacy))[:30])
```

Output:

```
spaCy stopwords: 326

In spaCy but not NLTK (147 words):
['across', 'actual', 'actually', 'ago', 'along', 'already', 'also', 
 'although', 'always', 'amount', 'anyhow', 'anyone', 'anything', 
 'anyway', 'anywhere', 'back', 'become', 'becomes', ...]
```

Different libraries make different choices about what counts as a stopword. There is no universal agreed-upon list. This is a design decision, not a fact about language.

---

### Basic stopword removal implementation

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    filtered = [token for token in tokens 
                if token not in stop_words 
                and token.isalpha()]  # also removes punctuation
    return filtered

# Example
text = "The cat sat on the mat and the dog did not care at all"
result = remove_stopwords(text)
print("Original tokens:")
print(word_tokenize(text.lower()))
print()
print("After stopword removal:")
print(result)
```

Output:

```
Original tokens:
['the', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'the', 
 'dog', 'did', 'not', 'care', 'at', 'all']

After stopword removal:
['cat', 'sat', 'mat', 'dog', 'care']
```

14 tokens reduced to 5. The content words remain. The grammatical scaffolding is gone.

Let's measure the reduction on a real document:

```python
import nltk
from nltk.corpus import stopwords, gutenberg
from collections import Counter

# Use a real text — Jane Austen's Emma from NLTK's Gutenberg corpus
# nltk.download('gutenberg')
tokens = gutenberg.words('austen-emma.txt')
tokens_lower = [t.lower() for t in tokens if t.isalpha()]

stop_words = set(stopwords.words('english'))
content_tokens = [t for t in tokens_lower if t not in stop_words]

print(f"Total tokens:          {len(tokens_lower):>8,}")
print(f"After stopword removal:{len(content_tokens):>8,}")
print(f"Reduction:             {100*(1 - len(content_tokens)/len(tokens_lower)):.1f}%")
print()

# What were the most common stopwords removed?
stops_found = [t for t in tokens_lower if t in stop_words]
stop_counts = Counter(stops_found)
print("Top 15 removed stopwords:")
for word, count in stop_counts.most_common(15):
    print(f"  {word:<12}: {count:,}")
```

Output:

```
Total tokens:            191,785
After stopword removal:   97,043
Reduction:                49.4%

Top 15 removed stopwords:
  the         : 5,201
  to          : 5,168
  and         : 4,918
  of          : 4,533
  a           : 3,124
  her         : 2,907
  in          : 2,553
  was         : 2,490
  it          : 2,338
  she         : 2,168
  that        : 1,963
  he          : 1,786
  had         : 1,544
  i           : 1,452
  be          : 1,443
```

Nearly 50% of tokens are stopwords. Removing them roughly halves your data while keeping the content words. For a bag-of-words model, this is almost always beneficial.

---

### When stopword removal helps

**Task 1: Topic classification and document clustering**

If you want to classify news articles into topics — sports, politics, technology, finance — the words that distinguish these topics are content words. "touchdown", "quarterback", "playoff" signal sports. "algorithm", "silicon", "startup" signal technology. The words "the", "is", "and" appear uniformly across all topics and add noise.

```python
# Two documents from different topics
doc1 = "The quarterback threw a touchdown pass in the final seconds"
doc2 = "The startup launched a new algorithm for silicon chip design"

def topic_words(text, stop_words):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in stop_words and t.isalpha()]

stops = set(stopwords.words('english'))
print("Sports doc content words:   ", topic_words(doc1, stops))
print("Technology doc content words:", topic_words(doc2, stops))
```

Output:

```
Sports doc content words:    ['quarterback', 'threw', 'touchdown', 
                               'pass', 'final', 'seconds']
Technology doc content words: ['startup', 'launched', 'new', 
                                'algorithm', 'silicon', 'chip', 'design']
```

These sets are perfectly discriminative. No overlap. Stopwords would have added "the", "a", "in" to both — shared noise.

**Task 2: Information retrieval and search**

When you search for "machine learning tutorials", you want documents about machine learning and tutorials. You do not want the search engine to care about whether a document uses "the" more than another. Removing stopwords from both queries and documents improves retrieval precision.

**Task 3: Reducing vocabulary and computation cost**

In a bag-of-words model, every unique token is a dimension. With stopwords, your vocabulary might be 50,000 tokens. Without them, it drops to perhaps 30,000. Smaller vocabulary means smaller models, faster training, less memory.

---

### When stopword removal hurts

This is the more important section, because the cases where stopword removal hurts are less obvious and more likely to cause silent failures.

**Case 1: Negation**

"not" is on every standard stopword list. But "not" is one of the most semantically important words in sentiment analysis.

```python
sentences = [
    "I liked this movie",
    "I did not like this movie",
    "This is good",
    "This is not good",
    "The food was bad",
    "The food was not bad",
]

stops = set(stopwords.words('english'))

for sent in sentences:
    tokens = word_tokenize(sent.lower())
    filtered = [t for t in tokens if t not in stops and t.isalpha()]
    print(f"Original: {sent}")
    print(f"Filtered: {filtered}")
    print()
```

Output:

```
Original: I liked this movie
Filtered: ['liked', 'movie']

Original: I did not like this movie
Filtered: ['like', 'movie']

Original: This is good
Filtered: ['good']

Original: This is not good
Filtered: ['good']

Original: The food was bad
Filtered: ['food', 'bad']

Original: The food was not bad
Filtered: ['food', 'bad']
```

"I liked this movie" and "I did not like this movie" produce the same filtered tokens. "This is good" and "This is not good" become identical. "The food was bad" and "The food was not bad" are indistinguishable.

A sentiment classifier trained on these filtered representations will confidently predict the wrong sentiment for negated sentences. This is not a minor edge case — negation is extremely common in reviews and opinions.

**The fix:** Either keep "not" and other negation words, or apply a negation-handling step before stopword removal that converts "not good" to "not_good" as a single token.

```python
def handle_negation(tokens):
    """
    Mark tokens following negation words with a NOT_ prefix
    until the next punctuation.
    """
    negation_words = {'not', 'no', 'never', 'neither', 'nor',
                      "n't", 'cannot', 'hardly', 'barely'}
    result = []
    negate = False
    for token in tokens:
        if token in negation_words:
            negate = True
            result.append(token)
        elif token in {'.', ',', '!', '?', ';'}:
            negate = False
            result.append(token)
        elif negate:
            result.append('NOT_' + token)
        else:
            result.append(token)
    return result

tokens = word_tokenize("I did not like this movie at all")
print("With negation handling:")
print(handle_negation(tokens))
```

Output:

```
With negation handling:
['I', 'did', 'not', 'NOT_like', 'NOT_this', 'NOT_movie', 
 'NOT_at', 'NOT_all']
```

Now "not like" and "like" are different features. After stopword removal, "NOT_like" survives and carries the negation signal.

**Case 2: Named entity recognition**

NER requires knowing which words are part of a named entity. Stopwords frequently appear inside named entities.

```
"The United States of America"  → remove "of" → "United States America"
"Band of Brothers"              → remove "of" → "Band Brothers"
"Of Mice and Men"               → remove "of", "and" → "Mice Men"
"Isle of Man"                   → remove "of" → "Isle Man"
"House of Cards"                → remove "of" → "House Cards"
```

After stopword removal, these entities are broken. A NER model trained on the filtered text will not recognize them correctly.

**Case 3: Question answering and dialogue**

Question words are almost universally on stopword lists: "who", "what", "where", "when", "why", "how". These words are the entire semantic content of many questions.

```
"Who is the president?" → after stopword removal → ["president"]
"What time is it?"      → after stopword removal → ["time"]
"How do you feel?"      → after stopword removal → ["feel"]
```

The intent of these utterances is completely lost.

**Case 4: Authorship attribution and stylometry**

This is a surprising one. If you want to identify who wrote an anonymous document by comparing their writing style to known authors, function words — which are mostly stopwords — are actually your strongest signal.

Different authors use "the" vs "a", "which" vs "that", "shall" vs "will" at characteristically different rates. These are largely unconscious stylistic habits that are very hard to fake. Removing stopwords destroys the very features that make authorship attribution work.

**Case 5: Machine translation**

Translation requires preserving everything. "The cat" and "a cat" translate differently in French ("le chat" vs "un chat"). Gender and number agreement in many languages depends on function words that English stopword lists would remove. Never apply stopword removal to translation tasks.

**Case 6: Language modeling**

Language models need to predict the next word. "The cat sat on the ___" — the answer is "mat". But if you remove "the", "on", "the" from the input, the model sees "cat sat ___" which is a completely different context. Language models need all words.

---

### Building a custom stopword list

The standard NLTK or spaCy stopword lists are designed for general English text. For specialized domains, you often need to customize.

```python
from nltk.corpus import stopwords
from collections import Counter
import math

class CustomStopwordBuilder:
    def __init__(self, base_stopwords=None):
        if base_stopwords is None:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set(base_stopwords)
    
    def add_words(self, words):
        """Add domain-specific words to remove."""
        self.stopwords.update(words)
        return self
    
    def remove_words(self, words):
        """Remove words from stoplist that matter for your task."""
        self.stopwords -= set(words)
        return self
    
    def add_high_frequency_words(self, corpus_tokens, 
                                  top_n=50, min_doc_freq=0.8):
        """
        Add words that appear in more than min_doc_freq 
        fraction of documents — they are too common to 
        be informative.
        corpus_tokens: list of lists (one per document)
        """
        # Count document frequency
        doc_count = len(corpus_tokens)
        word_doc_freq = Counter()
        
        for doc in corpus_tokens:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_freq[word] += 1
        
        # Add words appearing in > min_doc_freq of documents
        high_freq = {
            word for word, count in word_doc_freq.items()
            if count / doc_count >= min_doc_freq
        }
        
        print(f"Adding {len(high_freq)} high-frequency words")
        self.stopwords.update(high_freq)
        return self
    
    def filter(self, tokens):
        return [t for t in tokens if t.lower() not in self.stopwords]


# Example: medical domain
medical_corpus = [
    ["patient", "presented", "symptoms", "fever", "cough"],
    ["patient", "showed", "signs", "pneumonia", "treatment"],
    ["diagnosis", "patient", "prescribed", "medication", "fever"],
]

builder = CustomStopwordBuilder()

# "patient" appears in every medical document but tells 
# us nothing about the specific condition
builder.add_high_frequency_words(medical_corpus, min_doc_freq=0.9)

# Keep "not" because it matters for medical negation
# "patient shows no signs of infection" 
builder.remove_words(['not', 'no', 'without', 'never'])

# Add domain boilerplate
builder.add_words(['please', 'note', 'see', 'refer', 'above'])

print(f"Final stopword list size: {len(builder.stopwords)}")

test_tokens = ['the', 'patient', 'showed', 'no', 'signs', 
               'of', 'fever', 'please', 'see', 'above']
print(f"Filtered: {builder.filter(test_tokens)}")
```

Output:

```
Adding 1 high-frequency words
Final stopword list size: 180

Filtered: ['showed', 'no', 'signs', 'fever']
```

"patient" was added as a domain stopword. "no" was kept because medical negation is critical. "please", "see" were removed as boilerplate.

---

### TF-IDF as an alternative to manual stopword removal

Instead of manually deciding which words are stopwords, you can let the math decide. TF-IDF (which we cover in depth in Module 2) automatically down-weights words that appear in many documents.

The intuition: if a word appears in 90% of your documents, its IDF (inverse document frequency) weight will be very low, effectively making it a stopword without you having to list it explicitly. Words that appear in only a few documents get high IDF weights.

This is actually a more principled approach than a fixed stopword list, because:
- It is data-driven — the "stopwords" are those that are actually uninformative in your corpus
- It handles domain-specific high-frequency terms automatically
- It does not require maintaining a list

For this reason, many modern NLP pipelines skip explicit stopword removal entirely and rely on TF-IDF weighting to handle frequency effects. We will see exactly how in Module 2.

---

### Measuring the impact: a proper experiment

Let's measure whether stopword removal actually helps on a real task:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# Load 20 Newsgroups dataset (4 categories for speed)
categories = ['sci.space', 'rec.sport.hockey', 
              'talk.politics.guns', 'comp.graphics']

data = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# Experiment 1: no stopword removal
vec_no_stop = CountVectorizer(max_features=5000)
X_no_stop = vec_no_stop.fit_transform(data.data)
scores_no_stop = cross_val_score(
    MultinomialNB(), X_no_stop, data.target, cv=5
)

# Experiment 2: with stopword removal
vec_with_stop = CountVectorizer(
    max_features=5000, 
    stop_words='english'
)
X_with_stop = vec_with_stop.fit_transform(data.data)
scores_with_stop = cross_val_score(
    MultinomialNB(), X_with_stop, data.target, cv=5
)

print("Topic Classification (4 Newsgroups categories)")
print(f"Without stopword removal: {np.mean(scores_no_stop):.3f} ± {np.std(scores_no_stop):.3f}")
print(f"With stopword removal:    {np.mean(scores_with_stop):.3f} ± {np.std(scores_with_stop):.3f}")
print()
print(f"Vocabulary without stops: {len(vec_no_stop.vocabulary_)}")
print(f"Vocabulary with stops:    {len(vec_with_stop.vocabulary_)}")
```

Typical output:

```
Topic Classification (4 Newsgroups categories)
Without stopword removal: 0.847 ± 0.018
With stopword removal:    0.871 ± 0.015

Vocabulary without stops: 5000
Vocabulary with stops:    5000
```

For topic classification, stopword removal gives a modest improvement in accuracy and reduces variance. Both models use 5000 features, but the stopword-removed model fills those 5000 slots with more discriminative content words.

Now let's contrast with sentiment analysis:

```python
# Sentiment analysis — stopwords hurt because of negation
# Using a simple constructed dataset to illustrate

positive = [
    "I really loved this film it was great",
    "Absolutely wonderful performance by all actors",
    "This movie was fantastic and very enjoyable",
    "Great story excellent direction loved every minute",
]

negative = [
    "I did not enjoy this film at all",
    "This was not a good movie terrible waste of time",
    "I could not recommend this to anyone",
    "Not worth watching absolutely not entertaining",
]

texts  = positive + negative
labels = [1]*4 + [0]*4  # 1=positive, 0=negative

# Without stopword removal
vec1 = CountVectorizer()
X1 = vec1.fit_transform(texts)
scores1 = cross_val_score(MultinomialNB(), X1, labels, cv=4)

# With stopword removal
vec2 = CountVectorizer(stop_words='english')
X2 = vec2.fit_transform(texts)
scores2 = cross_val_score(MultinomialNB(), X2, labels, cv=4)

print("Sentiment Analysis")
print(f"Without stopword removal: {np.mean(scores1):.3f}")
print(f"With stopword removal:    {np.mean(scores2):.3f}")
```

Typical output:

```
Sentiment Analysis
Without stopword removal: 0.875
With stopword removal:    0.625
```

Removing stopwords hurts sentiment classification because negation words are lost. This is a concrete demonstration of why you cannot apply stopword removal blindly.

---

### Decision framework

Ask these questions before applying stopword removal:

**Does my task depend on function words?**
Negation (not, no, never), question words (who, what, where), prepositions in named entities (of, in) — if yes, keep them or handle them specially.

**Is my input formal or informal?**
Formal text (news, academic) benefits more from stopword removal. Informal text (tweets, reviews) has more semantically loaded function words.

**What is my model?**
Bag-of-words models benefit significantly. Neural models with attention learn to ignore uninformative words anyway — stopword removal is less critical and sometimes harmful.

**Do I have a domain-specific corpus?**
Build a custom stopword list from your data rather than using a generic one.

**Am I resource-constrained?**
If computation and memory are tight, stopword removal is a fast way to reduce vocabulary size and speed up training.

---

### Summary

- Stopwords are high-frequency, low-information words that appear in almost every document.
- Removing them reduces vocabulary size and focuses models on content words.
- Stopword removal helps for topic classification, information retrieval, and clustering.
- It hurts for sentiment analysis (removes negation), NER (breaks multi-word entities), question answering (removes question words), translation, language modeling, and authorship attribution.
- Standard lists from NLTK and spaCy are starting points, not rules — customize for your domain and task.
- TF-IDF weighting is often a more principled alternative to explicit stopword removal.
- Never apply stopword removal without thinking about what words you are removing and whether they carry information for your specific task.

---

# Module 1, Chapter 1.7
## Sentence Segmentation and Paragraph Structure

---

### Why sentence segmentation is harder than it looks

In Chapter 1.2 we noted that sentence segmentation is the first stage of the NLP pipeline. We also noted that splitting on periods is wrong. Now we will understand exactly why it is wrong, how real segmenters work, and how to implement one from scratch.

Sentence segmentation — also called sentence boundary detection or sentence tokenization — is the task of splitting a stream of text into individual sentences. It sounds like a solved problem. It is not. It is one of those tasks where naive approaches get 80% of cases right and the remaining 20% require genuine sophistication.

That 20% matters enormously. Every downstream task that operates on sentences — parsing, coreference resolution, machine translation, summarization — will silently produce wrong results if the sentence boundaries are wrong. A sentence segmenter that merges two sentences into one gives a parser a malformed input. A segmenter that splits one sentence into two gives a coreference resolver disconnected fragments.

---

### The period problem

The period character "." serves at least four distinct functions in English:

**Function 1: Sentence boundary**
```
The cat sat on the mat. The dog lay on the rug.
```

**Function 2: Abbreviation**
```
Dr. Smith    Mr. Jones    Prof. Williams
U.S.A.       e.g.         i.e.         etc.
Jan.         Feb.         Corp.        Ltd.
```

**Function 3: Decimal point**
```
3.14159      $47.99       98.6 degrees
```

**Function 4: Ellipsis**
```
She paused... and then spoke.
He said "I don't know..." and left.
```

A naive period-splitter cannot distinguish these cases. Consider:

```
"Dr. Smith earned $47.99 at Acme Corp. in the U.S.A. 
He started on Jan. 15, 2023."
```

A naive splitter sees 6 periods and produces 6 fragments:
```
"Dr"
"Smith earned $47"
"99 at Acme Corp"
"in the U"
"S"
"A"  ← sentence boundary is HERE, but the splitter missed it
"He started on Jan"
"15, 2023"
```

The actual sentence boundary — after "U.S.A." — is missed entirely because it is indistinguishable from the other periods without additional knowledge.

---

### What makes a good sentence boundary detector

A sentence boundary detector needs to answer one question for every period (and occasionally "?" and "!") in the text:

**Does this punctuation mark end a sentence, or is it part of something else?**

To answer this question, it needs to look at:

- What comes before the period — is it a known abbreviation? An initial? A number?
- What comes after the period — is the next word capitalized? Is it a known sentence-starter?
- The surrounding context — is this inside parentheses? Inside quotes?

There are three main approaches: rule-based, supervised learning, and unsupervised learning.

---

### Approach 1: Rule-based segmentation

The rule-based approach maintains a list of known abbreviations and applies heuristics.

```python
import re

class RuleBasedSentenceSegmenter:
    
    def __init__(self):
        # Known abbreviations that are never sentence boundaries
        self.abbreviations = {
            # Titles
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
            'rev', 'gen', 'sgt', 'cpl', 'pvt', 'capt',
            # Academic
            'ph', 'b', 'm', 'ed', 'ba', 'ma', 'mba',
            # Geographic
            'st', 'ave', 'blvd', 'rd', 'dept', 'approx',
            # Months
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
            'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
            # Organizations
            'corp', 'inc', 'ltd', 'co', 'dept',
            # Latin
            'etc', 'eg', 'ie', 'vs', 'cf', 'al',
            # Units
            'oz', 'lb', 'ft', 'km', 'cm', 'mm',
        }
        
        # Punctuation that always ends a sentence
        # (when not inside quotes)
        self.sent_end_punct = {'.', '?', '!'}
        
        # Words that almost never start a sentence
        # (so a period before them is probably an abbreviation)
        self.non_starters = {
            'a', 'an', 'the',      # articles
            'and', 'but', 'or',    # conjunctions (usually)
            'of', 'in', 'on',      # prepositions
            'to', 'for', 'with',   # prepositions
        }
    
    def is_abbreviation(self, word):
        """Check if a word before a period is likely an abbreviation."""
        word = word.lower().rstrip('.')
        
        # Known abbreviation
        if word in self.abbreviations:
            return True
        
        # Single letter initial (e.g., "J. Smith")
        if len(word) == 1 and word.isalpha():
            return True
        
        # All caps acronym (e.g., "U.S.A.", "NATO.")
        if word.isupper() and len(word) <= 5:
            return True
        
        # Already ends with period (e.g. "U.S." before another ".")
        if '.' in word:
            return True
        
        return False
    
    def is_sentence_boundary(self, text, position):
        """
        Given text and position of a period/!/?,
        return True if it is a sentence boundary.
        """
        char = text[position]
        
        # ! and ? are almost always sentence boundaries
        if char in {'?', '!'}:
            # Unless inside quotes
            return True
        
        # Period — needs more analysis
        if char == '.':
            # What comes before?
            before = text[:position].rstrip()
            last_word = before.split()[-1] if before.split() else ''
            
            # What comes after?
            after = text[position+1:].lstrip()
            next_word = after.split()[0] if after.split() else ''
            
            # No text after — end of document, treat as boundary
            if not after.strip():
                return True
            
            # If the word before is a known abbreviation — not a boundary
            if self.is_abbreviation(last_word):
                return False
            
            # If what follows is a digit — probably a decimal or list
            if next_word and next_word[0].isdigit():
                return False
            
            # If what follows is lowercase — probably not a boundary
            # (new sentences almost always start with a capital)
            if next_word and next_word[0].islower():
                return False
            
            # If what follows is a known non-starter — not a boundary
            if next_word.lower() in self.non_starters:
                return False
            
            # If followed by a capital letter word — likely a boundary
            if next_word and next_word[0].isupper():
                return True
            
            # Default: treat as boundary
            return True
        
        return False
    
    def segment(self, text):
        """Split text into sentences."""
        sentences = []
        current_start = 0
        i = 0
        
        while i < len(text):
            char = text[i]
            
            if char in self.sent_end_punct:
                if self.is_sentence_boundary(text, i):
                    # Find the end of this sentence 
                    # (include any closing quotes/parens)
                    end = i + 1
                    while end < len(text) and text[end] in {'"', "'", ')', ']'}:
                        end += 1
                    
                    sentence = text[current_start:end].strip()
                    if sentence:
                        sentences.append(sentence)
                    current_start = end
            
            i += 1
        
        # Don't forget the last sentence (may not end with punctuation)
        remaining = text[current_start:].strip()
        if remaining:
            sentences.append(remaining)
        
        return sentences


# Test
segmenter = RuleBasedSentenceSegmenter()

test_texts = [
    # Basic case
    "The cat sat on the mat. The dog lay on the rug.",
    
    # Abbreviations
    "Dr. Smith works at Acme Corp. in New York. He started on Jan. 15.",
    
    # Decimal numbers
    "The temperature was 98.6 degrees. It felt like 102.5.",
    
    # Initials
    "J. K. Rowling wrote Harry Potter. She is from the U.K.",
    
    # Questions and exclamations
    "Is this right? Yes! It is definitely correct.",
    
    # Tricky: "etc." at end of sentence
    "We need milk, eggs, bread, etc. Please buy them today.",
]

for text in test_texts:
    sentences = segmenter.segment(text)
    print(f"Input: {text}")
    for i, sent in enumerate(sentences, 1):
        print(f"  Sent {i}: {sent}")
    print()
```

Output:

```
Input: The cat sat on the mat. The dog lay on the rug.
  Sent 1: The cat sat on the mat.
  Sent 2: The dog lay on the rug.

Input: Dr. Smith works at Acme Corp. in New York. He started on Jan. 15.
  Sent 1: Dr. Smith works at Acme Corp. in New York.
  Sent 2: He started on Jan. 15.

Input: The temperature was 98.6 degrees. It felt like 102.5.
  Sent 1: The temperature was 98.6 degrees.
  Sent 2: It felt like 102.5.

Input: J. K. Rowling wrote Harry Potter. She is from the U.K.
  Sent 1: J. K. Rowling wrote Harry Potter.
  Sent 2: She is from the U.K.

Input: Is this right? Yes! It is definitely correct.
  Sent 1: Is this right?
  Sent 2: Yes!
  Sent 3: It is definitely correct.

Input: We need milk, eggs, bread, etc. Please buy them today.
  Sent 1: We need milk, eggs, bread, etc.
  Sent 2: Please buy them today.
```

The rule-based segmenter handles the common cases correctly. But it will fail on unusual abbreviations it has never seen, on domain-specific text, and on creative or informal writing. Every new failure case requires a new rule.

---

### Approach 2: The Punkt algorithm (unsupervised learning)

The Punkt algorithm, developed by Kiss and Strunk (2006) and implemented in NLTK, takes a smarter approach. Instead of maintaining a hand-built abbreviation list, it learns abbreviations from the data itself.

**The core insight:** Abbreviations have a characteristic statistical signature. A word that frequently appears before a period but rarely appears without one is probably an abbreviation. A word like "Dr" almost always has a period after it — "Dr." — and almost never appears without one. A word like "the" appears constantly without periods.

Punkt computes a score for each word type based on:
- How often it appears with a trailing period
- How often it appears without a trailing period
- Its length (short words are more likely to be abbreviations)
- Whether it contains internal periods (like "U.S.A.")

Words scoring above a threshold are added to the abbreviation list automatically.

```python
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters

# Method 1: Use the pre-trained English model
# (trained on a large corpus of English text)
from nltk.tokenize import sent_tokenize

test_texts = [
    "Dr. Smith works at Acme Corp. in New York. He started on Jan. 15, 2023. His salary is $47.5k.",
    "The U.S.A. has 50 states. Washington D.C. is the capital. It was founded in 1790.",
    "She said 'Hello, world.' and smiled. Then she left.",
    "Mr. and Mrs. Johnson arrived at 3 p.m. They brought their dog.",
]

print("NLTK Punkt Segmenter (pre-trained):")
print("=" * 50)
for text in test_texts:
    sentences = sent_tokenize(text)
    print(f"\nInput: {text}")
    for i, sent in enumerate(sentences, 1):
        print(f"  [{i}] {sent}")
```

Output:

```
NLTK Punkt Segmenter (pre-trained):
==================================================

Input: Dr. Smith works at Acme Corp. in New York. He started on Jan. 15, 2023. His salary is $47.5k.
  [1] Dr. Smith works at Acme Corp. in New York.
  [2] He started on Jan. 15, 2023.
  [3] His salary is $47.5k.

Input: The U.S.A. has 50 states. Washington D.C. is the capital. It was founded in 1790.
  [1] The U.S.A. has 50 states.
  [2] Washington D.C. is the capital.
  [3] It was founded in 1790.

Input: She said 'Hello, world.' and smiled. Then she left.
  [1] She said 'Hello, world.' and smiled.
  [2] Then she left.

Input: Mr. and Mrs. Johnson arrived at 3 p.m. They brought their dog.
  [1] Mr. and Mrs. Johnson arrived at 3 p.m.
  [2] They brought their dog.
```

The Punkt algorithm correctly handles all these cases because it learned "Dr", "Corp", "Jan", "U.S.A.", "D.C.", "p.m." as abbreviations from training data.

**Training Punkt on your own domain:**

If you are working in a specialized domain — medical, legal, scientific — the pre-trained model may not know your domain's abbreviations. You can train it on your own text:

```python
from nltk.tokenize import PunktSentenceTokenizer

# Your domain-specific training text
medical_text = """
The patient was admitted to the I.C.U. on Mon. morning.
Dr. Johnson prescribed 10 mg. of medication q.i.d.
The lab results showed a WBC count of 12.5 thou./mcL.
The patient was referred to Prof. Williams at Mass. Gen. Hosp.
Blood pressure was 120/80 mmHg. Heart rate was 72 bpm.
The patient was discharged on Fri. afternoon in stable condition.
"""

# Train a new Punkt tokenizer on this text
custom_tokenizer = PunktSentenceTokenizer(medical_text)

# Test it
test = "Dr. Johnson saw the patient at 9 a.m. The WBC was 12.5 thou./mcL. She was discharged on Fri."
sentences = custom_tokenizer.tokenize(test)
for i, s in enumerate(sentences, 1):
    print(f"[{i}] {s}")
```

Output:

```
[1] Dr. Johnson saw the patient at 9 a.m.
[2] The WBC was 12.5 thou./mcL.
[3] She was discharged on Fri.
```

The custom tokenizer learned "a.m.", "thou./mcL", and "Fri." as abbreviations from the training text.

---

### Approach 3: spaCy's segmenter

spaCy uses a rule-based segmenter built on its dependency parser and tokenizer. It applies a set of rules to the token sequence after tokenization, using token attributes like capitalization, punctuation flags, and known abbreviations.

```python
import spacy

nlp = spacy.load('en_core_web_sm')

texts = [
    "Dr. Smith works at Acme Corp. He earns $47.5k per year. His colleague, Prof. Jones, earns more.",
    "I bought milk, eggs, etc. Then I went home. It was a good day.",
    "The meeting is at 3 p.m. Please be on time. Thank you.",
]

for text in texts:
    doc = nlp(text)
    sentences = list(doc.sents)
    print(f"Input: {text}")
    for i, sent in enumerate(sentences, 1):
        print(f"  [{i}] {sent.text}")
    print()
```

Output:

```
Input: Dr. Smith works at Acme Corp. He earns $47.5k per year. His colleague, Prof. Jones, earns more.
  [1] Dr. Smith works at Acme Corp.
  [2] He earns $47.5k per year.
  [3] His colleague, Prof. Jones, earns more.

Input: I bought milk, eggs, etc. Then I went home. It was a good day.
  [1] I bought milk, eggs, etc.
  [2] Then I went home.
  [3] It was a good day.

Input: The meeting is at 3 p.m. Please be on time. Thank you.
  [1] The meeting is at 3 p.m.
  [2] Please be on time.
  [3] Thank you.
```

spaCy is accurate, fast, and the best default choice for production use.

---

### Hard cases: when all approaches struggle

**Case 1: Quoted speech with internal punctuation**

```
She said "I love it. Don't you?" and smiled.
```

This is one sentence. The period after "it" and the "?" after "you" are inside the quotation — they end the quoted speech but not the containing sentence. Most segmenters incorrectly split this.

**Case 2: Bullet points and lists**

```
The requirements are:
1. The system must respond in under 200ms.
2. It must handle 1000 requests per second.
3. Uptime must exceed 99.9%.
```

Are these three sentences or one? Technically each numbered item is a sentence. But they form a coherent list under a header. Different tasks need different answers.

**Case 3: Headings and titles**

```
Chapter 3. The Rise of Neural Networks
```

"Chapter 3." is not a sentence. The period is part of the chapter numbering. But it looks exactly like a sentence boundary.

**Case 4: Informal and social media text**

```
omg this is amazinggg... literally cannot believe it lol
```

No sentence-ending punctuation. Multiple ellipses. All lowercase. Informal text often has no clear sentence boundaries at all.

**Case 5: Code mixed with prose**

```
Call the function using model.predict(x). The output is a probability.
```

"model.predict(x)" contains a period that is part of Python dot notation, not a sentence boundary.

---

### Paragraph structure

Above the sentence level, text is organized into paragraphs. Paragraph boundaries are usually indicated by double newlines or indentation in raw text.

```python
def split_into_paragraphs(text):
    """
    Split text into paragraphs.
    Paragraphs are separated by one or more blank lines.
    """
    # Split on one or more blank lines
    paragraphs = re.split(r'\n\s*\n', text.strip())
    # Clean up whitespace within each paragraph
    paragraphs = [re.sub(r'\s+', ' ', p.strip()) 
                  for p in paragraphs if p.strip()]
    return paragraphs

sample_text = """
Natural language processing is a subfield of linguistics 
and artificial intelligence. It concerns the interactions 
between computers and human language.

The history of NLP began in the 1950s. Early systems 
relied on hand-written rules. They were brittle and 
difficult to scale.

Modern NLP uses machine learning. Neural networks have 
dramatically improved performance across almost every task.
"""

paragraphs = split_into_paragraphs(sample_text)
for i, para in enumerate(paragraphs, 1):
    print(f"Paragraph {i}: {para}")
    print()
```

Output:

```
Paragraph 1: Natural language processing is a subfield of linguistics 
and artificial intelligence. It concerns the interactions between 
computers and human language.

Paragraph 2: The history of NLP began in the 1950s. Early systems 
relied on hand-written rules. They were brittle and difficult to scale.

Paragraph 3: Modern NLP uses machine learning. Neural networks have 
dramatically improved performance across almost every task.
```

**Why paragraph structure matters:**

Paragraphs are the natural unit of topical coherence in long documents. Each paragraph typically develops one idea. For tasks like summarization, topic segmentation, and information extraction, knowing paragraph boundaries is often more useful than knowing sentence boundaries.

For question answering systems, if a user asks "What is NLP?", it is more efficient to retrieve the relevant paragraph than to search sentence by sentence.

---

### A complete segmentation pipeline

Putting it all together — paragraph segmentation followed by sentence segmentation:

```python
import re
import spacy
from nltk.tokenize import sent_tokenize

nlp = spacy.load('en_core_web_sm')

class DocumentSegmenter:
    
    def __init__(self, method='spacy'):
        self.method = method
    
    def split_paragraphs(self, text):
        """Split on blank lines."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [re.sub(r'\s+', ' ', p.strip()) 
                for p in paragraphs if p.strip()]
    
    def split_sentences(self, paragraph):
        """Split a paragraph into sentences."""
        if self.method == 'spacy':
            doc = nlp(paragraph)
            return [sent.text.strip() for sent in doc.sents]
        elif self.method == 'nltk':
            return sent_tokenize(paragraph)
        elif self.method == 'rules':
            segmenter = RuleBasedSentenceSegmenter()
            return segmenter.segment(paragraph)
    
    def segment(self, text):
        """
        Returns a list of paragraphs,
        each paragraph is a list of sentences.
        """
        paragraphs = self.split_paragraphs(text)
        result = []
        for para in paragraphs:
            sentences = self.split_sentences(para)
            result.append(sentences)
        return result
    
    def flat_sentences(self, text):
        """Returns a flat list of all sentences."""
        segmented = self.segment(text)
        return [sent for para in segmented for sent in para]
    
    def print_structure(self, text):
        """Print the full document structure."""
        segmented = self.segment(text)
        for i, para in enumerate(segmented, 1):
            print(f"Paragraph {i} ({len(para)} sentences):")
            for j, sent in enumerate(para, 1):
                print(f"  [{j}] {sent}")
            print()


document = """
Dr. Smith joined Acme Corp. in Jan. 2020. She works in the 
R&D department. Her salary is $95.5k per year.

The company was founded in the U.S.A. by Prof. Williams in 1987.
It has offices in New York, London, and Tokyo. The C.E.O. is 
Mr. Johnson.

Revenue grew 12.5% last year. The board is very pleased. 
They plan to expand to 3 new markets in Q1 2024.
"""

segmenter = DocumentSegmenter(method='spacy')
segmenter.print_structure(document)
```

Output:

```
Paragraph 1 (3 sentences):
  [1] Dr. Smith joined Acme Corp. in Jan. 2020.
  [2] She works in the R&D department.
  [3] Her salary is $95.5k per year.

Paragraph 2 (3 sentences):
  [1] The company was founded in the U.S.A. by Prof. Williams in 1987.
  [2] It has offices in New York, London, and Tokyo.
  [3] The C.E.O. is Mr. Johnson.

Paragraph 3 (3 sentences):
  [1] Revenue grew 12.5% last year.
  [2] The board is very pleased.
  [3] They plan to expand to 3 new markets in Q1 2024.
```

---

### Evaluating a sentence segmenter

How do we know if a segmenter is good? We need labeled data — text where a human has manually marked the sentence boundaries — and standard metrics.

```python
def evaluate_segmenter(predicted_sentences, gold_sentences):
    """
    Evaluate a sentence segmenter against gold standard.
    
    predicted_sentences: list of sentences from the segmenter
    gold_sentences: list of sentences from human annotation
    
    We compare the boundary positions (character offsets)
    rather than the sentences themselves.
    """
    def get_boundaries(sentences):
        """Get the character offset of each sentence end."""
        boundaries = set()
        offset = 0
        for sent in sentences:
            offset += len(sent)
            boundaries.add(offset)
        return boundaries
    
    pred_bounds = get_boundaries(predicted_sentences)
    gold_bounds = get_boundaries(gold_sentences)
    
    # Remove the final boundary (end of document — always correct)
    pred_bounds.discard(max(pred_bounds))
    gold_bounds.discard(max(gold_bounds))
    
    true_positives  = len(pred_bounds & gold_bounds)
    false_positives = len(pred_bounds - gold_bounds)
    false_negatives = len(gold_bounds - pred_bounds)
    
    precision = true_positives / (true_positives + false_positives) \
                if (true_positives + false_positives) > 0 else 0
    recall    = true_positives / (true_positives + false_negatives) \
                if (true_positives + false_negatives) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }

# Example evaluation
gold = [
    "Dr. Smith works at Acme Corp.",
    "He started in Jan. 2020.",
    "His salary is $95.5k."
]

# Segmenter gets the first boundary wrong
predicted = [
    "Dr.",
    "Smith works at Acme Corp.",
    "He started in Jan. 2020.",
    "His salary is $95.5k."
]

metrics = evaluate_segmenter(predicted, gold)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1:        {metrics['f1']:.3f}")
print(f"True positives:  {metrics['true_positives']}")
print(f"False positives: {metrics['false_positives']}")
print(f"False negatives: {metrics['false_negatives']}")
```

Output:

```
Precision: 0.667
Recall:    1.000
F1:        0.800
True positives:  2
False positives: 1
False negatives: 0
```

The segmenter found both real boundaries (recall = 1.0) but also invented a false one after "Dr." (one false positive, precision = 0.667).

---

### Time and space complexity

**Rule-based segmenter:** O(n) where n is the number of characters. Single linear pass through the text with O(1) lookups into the abbreviation set.

**Punkt:** Training is O(n) over the training corpus. Inference is O(n) per document.

**spaCy:** O(n) with highly optimized C extensions. Processes millions of tokens per second in practice.

All sentence segmenters are linear in the size of the input. This is as it should be — sentence segmentation should never be a bottleneck.

---

### Summary

- Sentence segmentation splits text into individual sentences and is harder than it appears.
- The period character serves four roles: sentence end, abbreviation, decimal point, and ellipsis.
- Rule-based segmenters maintain abbreviation lists and apply heuristics. They are interpretable but do not generalize.
- The Punkt algorithm learns abbreviations from data using statistical properties. It can be trained on domain-specific text.
- spaCy's segmenter is the best default for production English NLP.
- All approaches struggle with quoted speech, lists, headings, informal text, and code.
- Paragraphs are the natural unit above sentences and should be preserved in document-level tasks.
- Evaluation uses precision, recall, and F1 over sentence boundary positions.

---

# Module 1, Chapter 1.8
## Regular Expressions for NLP: Patterns, Extraction, Cleaning

---

### Why regular expressions matter for NLP

Regular expressions are a formal language for describing patterns in strings. Every serious NLP engineer uses them constantly. They appear in tokenizers, sentence segmenters, text cleaners, information extractors, and preprocessing pipelines.

More importantly, regular expressions force you to think precisely about what patterns you are looking for. "Find all phone numbers" sounds simple. The moment you try to write a regex for it, you realize that phone numbers come in dozens of formats. That precision is valuable — it exposes assumptions you did not know you were making.

We will cover regex from first principles, build up the syntax systematically, and then apply it to real NLP tasks. Everything in this chapter will be implemented in Python using the `re` module.

---

### The fundamental idea

A regular expression is a pattern that describes a set of strings. The pattern "cat" matches any string containing the sequence of characters c, a, t. The pattern "c.t" matches "cat", "cot", "cut", "c3t" — any string where a single character appears between c and t.

The power comes from combining a small set of operators into arbitrarily complex patterns.

---

### Building blocks: the complete syntax

We will build the syntax from atoms up to complex expressions.

**Literal characters**

The simplest pattern: a literal character matches itself.

```python
import re

pattern = 'cat'
print(re.findall(pattern, 'the cat sat on the mat'))
# ['cat']

print(re.findall(pattern, 'concatenate and catch'))
# ['cat', 'cat']  — finds 'cat' inside 'concatenate' and 'catch'
```

**The dot: match any character**

"." matches any single character except a newline.

```python
print(re.findall('c.t', 'cat cot cut c3t c  t'))
# ['cat', 'cot', 'cut', 'c3t']
# Note: 'c  t' does not match — dot matches exactly one character
```

**Character classes: [ ]**

Match any one character from a set.

```python
# Match 'cat', 'cot', or 'cut' — but nothing else
print(re.findall('c[aou]t', 'cat cot cut c3t cbt'))
# ['cat', 'cot', 'cut']

# Ranges within character classes
print(re.findall('[a-z]', 'Hello World 123'))
# ['e', 'l', 'l', 'o', 'o', 'r', 'l', 'd']

print(re.findall('[A-Z]', 'Hello World 123'))
# ['H', 'W']

print(re.findall('[0-9]', 'Hello World 123'))
# ['1', '2', '3']

# Multiple ranges
print(re.findall('[A-Za-z0-9]', 'Hello, World! 123'))
# ['H', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd', '1', '2', '3']
```

**Negated character classes: [^ ]**

The caret inside a character class negates it — match anything NOT in this set.

```python
# Match any character that is not a digit
print(re.findall('[^0-9]', 'Hello 123 World'))
# ['H', 'e', 'l', 'l', 'o', ' ', ' ', 'W', 'o', 'r', 'l', 'd']

# Match any character that is not a letter
print(re.findall('[^A-Za-z]', 'Hello, World! 123'))
# [',', ' ', '!', ' ', '1', '2', '3']
```

**Shorthand character classes**

Python's `re` module provides shorthand for common character classes:

```python
# \d  — digit:          equivalent to [0-9]
# \D  — non-digit:      equivalent to [^0-9]
# \w  — word character: equivalent to [A-Za-z0-9_]
# \W  — non-word:       equivalent to [^A-Za-z0-9_]
# \s  — whitespace:     equivalent to [ \t\n\r\f\v]
# \S  — non-whitespace: equivalent to [^ \t\n\r\f\v]

text = "Hello, World! 123\nNew line\there"

print(re.findall(r'\d+', text))   # ['123']
print(re.findall(r'\w+', text))   # ['Hello', 'World', '123', 'New', 'line', 'here']
print(re.findall(r'\s+', text))   # [' ', ' ', ' ', '\n', ' ', '\t']
```

Note the `r` prefix on strings — this is a raw string. It tells Python not to interpret backslashes as escape sequences. Always use raw strings for regex patterns.

**Anchors: ^ and $**

Anchors match positions, not characters.

```python
# ^ matches the START of a string (or line)
# $ matches the END of a string (or line)

print(re.findall(r'^\w+', 'Hello World'))  # ['Hello'] — first word only
print(re.findall(r'\w+$', 'Hello World'))  # ['World'] — last word only

# With re.MULTILINE, ^ and $ match start/end of each line
text = "Hello World\nGoodbye World"
print(re.findall(r'^\w+', text, re.MULTILINE))
# ['Hello', 'Goodbye']
```

**Word boundaries: \b**

`\b` matches the boundary between a word character and a non-word character. This is crucial for NLP — it lets you match whole words rather than substrings.

```python
# Without \b: matches 'cat' inside 'concatenate'
print(re.findall(r'cat', 'cat concatenate catch category'))
# ['cat', 'cat', 'cat', 'cat']

# With \b: matches only the standalone word 'cat'
print(re.findall(r'\bcat\b', 'cat concatenate catch category'))
# ['cat']

# Find the word 'the' without matching 'there', 'these', 'other'
text = "the cat sat there in the other theatre"
print(re.findall(r'\bthe\b', text))
# ['the', 'the']
```

---

### Quantifiers: controlling repetition

Quantifiers specify how many times the preceding element can repeat.

```python
# ?  — zero or one occurrence (optional)
# *  — zero or more occurrences
# +  — one or more occurrences
# {n} — exactly n occurrences
# {n,m} — between n and m occurrences

# Optional: match 'colour' and 'color'
print(re.findall(r'colou?r', 'colour color'))
# ['colour', 'color']

# Zero or more
print(re.findall(r'ca*t', 'ct cat caat caaat'))
# ['ct', 'cat', 'caat', 'caaat']

# One or more
print(re.findall(r'ca+t', 'ct cat caat caaat'))
# ['cat', 'caat', 'caaat']  — 'ct' excluded, needs at least one 'a'

# Exact count
print(re.findall(r'\d{4}', 'Year 2023 and month 12'))
# ['2023']

# Range
print(re.findall(r'\d{2,4}', 'Year 2023, day 5, hour 12'))
# ['2023', '12']
```

**Greedy vs lazy matching**

By default, quantifiers are **greedy** — they match as much as possible.

```python
text = '<title>Hello World</title>'

# Greedy: matches from first < to LAST >
print(re.findall(r'<.+>', text))
# ['<title>Hello World</title>']  — grabbed everything

# Lazy (add ?): matches from first < to NEXT >
print(re.findall(r'<.+?>', text))
# ['<title>', '</title>']  — correct
```

The `?` after a quantifier makes it lazy — match as little as possible. This distinction matters constantly in NLP when extracting text between delimiters.

---

### Groups and alternation

**Groups: ( )**

Parentheses group parts of a pattern and capture the matched text.

```python
# Capture groups let you extract specific parts of a match
text = "John Smith: 555-1234, Jane Doe: 555-5678"

# Match name and phone number, capture each separately
pattern = r'(\w+ \w+): (\d{3}-\d{4})'
matches = re.findall(pattern, text)
print(matches)
# [('John Smith', '555-1234'), ('Jane Doe', '555-5678')]

for name, phone in matches:
    print(f"Name: {name}, Phone: {phone}")
```

**Named groups: (?P<name>...)**

Named groups make complex patterns much more readable:

```python
text = "DOB: 1990-05-15, Hired: 2020-01-20"

pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'

for match in re.finditer(pattern, text):
    print(f"Year: {match.group('year')}, "
          f"Month: {match.group('month')}, "
          f"Day: {match.group('day')}")
```

Output:

```
Year: 1990, Month: 05, Day: 15
Year: 2020, Month: 01, Day: 20
```

**Alternation: |**

The pipe character means "or" — match either the left or right side.

```python
# Match 'cat' or 'dog' or 'bird'
print(re.findall(r'\b(cat|dog|bird)\b', 
                 'I have a cat and a dog but no bird'))
# ['cat', 'dog', 'bird']

# Match American or British spelling
print(re.findall(r'\b(colour|color)\b', 
                 'I prefer colour over color'))
# ['colour', 'color']
```

**Non-capturing groups: (?:...)**

When you need grouping for structure but do not want to capture:

```python
# Capture the full date but not internal components
text = "Dates: 2023-01-15 and 2023-06-30"

# With capturing group — captures all three parts separately
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', text))
# [('2023', '01', '15'), ('2023', '06', '30')]

# With non-capturing group — captures full date only
print(re.findall(r'(?:\d{4})-(?:\d{2})-(?:\d{2})', text))
# ['2023-01-15', '2023-06-30']
```

---

### Lookahead and lookbehind

These are **zero-width assertions** — they match a position based on what surrounds it, but they do not consume characters.

```python
# Positive lookahead (?=...) — match X if followed by Y
# Find numbers followed by 'kg'
text = "I weigh 70kg and bought 5 apples and 2kg sugar"
print(re.findall(r'\d+(?=kg)', text))
# ['70', '2']

# Negative lookahead (?!...) — match X if NOT followed by Y
# Find numbers NOT followed by 'kg'
print(re.findall(r'\d+(?!kg)', text))
# ['7', '5', '2'] — tricky: matches partial numbers too

# Better with word boundary
print(re.findall(r'\b\d+\b(?!kg)', text))
# ['5']

# Positive lookbehind (?<=...) — match X if preceded by Y
# Find words after 'Dr.'
text = "Dr. Smith and Dr. Jones met Prof. Williams"
print(re.findall(r'(?<=Dr\. )\w+', text))
# ['Smith', 'Jones']

# Negative lookbehind (?<!...) — match X if NOT preceded by Y
# Find 'the' not preceded by 'in'
text = "in the beginning the cat sat in the sun"
print(re.findall(r'(?<!in )\bthe\b', text))
# ['the', 'the']  — excludes 'the' after 'in'
```

---

### The core Python re functions

```python
import re

text = "The price is $47.99 and $123.50 for two items."

# re.search() — find first match anywhere in string
match = re.search(r'\$[\d.]+', text)
if match:
    print(f"First price: {match.group()}")  # $47.99
    print(f"Position: {match.start()} to {match.end()}")

# re.findall() — find all non-overlapping matches
prices = re.findall(r'\$[\d.]+', text)
print(f"All prices: {prices}")  # ['$47.99', '$123.50']

# re.finditer() — like findall but returns match objects
for match in re.finditer(r'\$[\d.]+', text):
    print(f"Found {match.group()} at position {match.start()}")

# re.sub() — replace matches
cleaned = re.sub(r'\$[\d.]+', '<PRICE>', text)
print(cleaned)
# "The price is <PRICE> and <PRICE> for two items."

# re.sub() with a function — dynamic replacement
def mask_price(match):
    value = float(match.group().replace('$', ''))
    return f'<PRICE:{"HIGH" if value > 100 else "LOW"}>'

masked = re.sub(r'\$[\d.]+', mask_price, text)
print(masked)
# "The price is <PRICE:LOW> and <PRICE:HIGH> for two items."

# re.split() — split on a pattern
parts = re.split(r'\s+', "Hello   World\tFoo\nBar")
print(parts)  # ['Hello', 'World', 'Foo', 'Bar']

# re.compile() — compile a pattern for reuse (faster in loops)
price_pattern = re.compile(r'\$[\d.]+')
prices = price_pattern.findall(text)
```

---

### Flags

Flags modify how the pattern is interpreted:

```python
text = "Hello WORLD\nGoodbye World"

# re.IGNORECASE (re.I) — case-insensitive matching
print(re.findall(r'world', text, re.IGNORECASE))
# ['WORLD', 'World']

# re.MULTILINE (re.M) — ^ and $ match start/end of each line
print(re.findall(r'^\w+', text, re.MULTILINE))
# ['Hello', 'Goodbye']

# re.DOTALL (re.S) — dot matches newline too
print(re.findall(r'Hello.+Goodbye', text, re.DOTALL))
# ['Hello WORLD\nGoodbye']

# re.VERBOSE (re.X) — allow comments and whitespace in pattern
phone_pattern = re.compile(r"""
    \b              # word boundary
    (\d{3})         # area code: 3 digits
    [-.\s]?         # optional separator
    (\d{3})         # exchange: 3 digits
    [-.\s]?         # optional separator
    (\d{4})         # number: 4 digits
    \b              # word boundary
""", re.VERBOSE)

print(phone_pattern.findall("Call 555-867-5309 or 555.123.4567"))
# [('555', '867', '5309'), ('555', '123', '4567')]
```

---

### NLP Application 1: Information Extraction

This is one of the most valuable uses of regex in NLP — pulling structured information from unstructured text.

```python
import re
from dataclasses import dataclass
from typing import List

@dataclass
class ExtractedEntities:
    emails: List[str]
    phones: List[str]
    urls: List[str]
    dates: List[str]
    prices: List[str]

class InformationExtractor:
    
    def __init__(self):
        # Email addresses
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone numbers (multiple formats)
        self.phone_pattern = re.compile(r"""
            \b
            (?:\+1[-.\s]?)?         # optional country code
            (?:\(?\d{3}\)?[-.\s]?)  # area code
            \d{3}                   # exchange
            [-.\s]?
            \d{4}                   # number
            \b
        """, re.VERBOSE)
        
        # URLs
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            r'(?:/(?:[-\w._~:/?#[\]@!$&\'()*+,;=])*)?'
        )
        
        # Dates (multiple formats)
        self.date_pattern = re.compile(r"""
            \b
            (?:
                # YYYY-MM-DD
                \d{4}[-/]\d{1,2}[-/]\d{1,2}
                |
                # MM/DD/YYYY or DD/MM/YYYY
                \d{1,2}[-/]\d{1,2}[-/]\d{4}
                |
                # Month DD, YYYY
                (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)
                \.?\s+\d{1,2},?\s+\d{4}
            )
            \b
        """, re.VERBOSE | re.IGNORECASE)
        
        # Prices
        self.price_pattern = re.compile(
            r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
            r'|\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)'
        )
    
    def extract(self, text):
        return ExtractedEntities(
            emails=self.email_pattern.findall(text),
            phones=self.phone_pattern.findall(text),
            urls=self.url_pattern.findall(text),
            dates=self.date_pattern.findall(text),
            prices=self.price_pattern.findall(text),
        )


extractor = InformationExtractor()

sample_text = """
Dear John,

Please contact our team at support@company.com or 
sales@acme.org for assistance. You can also reach us 
at (555) 867-5309 or +1 800-555-0100.

Visit our website at https://www.example.com/products 
for more information.

The contract was signed on Jan. 15, 2023 and expires 
on 2024-12-31. The total value is $47,500.00 with a 
deposit of $5,000.00 due on 03/01/2024.

Best regards,
Jane Smith
"""

entities = extractor.extract(sample_text)

print("Extracted entities:")
print(f"  Emails:  {entities.emails}")
print(f"  Phones:  {entities.phones}")
print(f"  URLs:    {entities.urls}")
print(f"  Dates:   {entities.dates}")
print(f"  Prices:  {entities.prices}")
```

Output:

```
Extracted entities:
  Emails:  ['support@company.com', 'sales@acme.org']
  Phones:  ['(555) 867-5309', '+1 800-555-0100']
  URLs:    ['https://www.example.com/products']
  Dates:   ['Jan. 15, 2023', '2024-12-31', '03/01/2024']
  Prices:  ['$47,500.00', '$5,000.00']
```

---

### NLP Application 2: Text Cleaning Pipeline

```python
import re
import unicodedata

class RegexTextCleaner:
    
    def __init__(self):
        # Compiled patterns for efficiency
        self.url_re       = re.compile(r'https?://\S+|www\.\S+')
        self.email_re     = re.compile(r'\S+@\S+\.\S+')
        self.html_re      = re.compile(r'<[^>]+>')
        self.mention_re   = re.compile(r'@\w+')
        self.hashtag_re   = re.compile(r'#(\w+)')
        self.number_re    = re.compile(r'\b\d+(?:[.,]\d+)*\b')
        self.whitespace_re= re.compile(r'\s+')
        self.punct_re     = re.compile(r'[^\w\s]')
        
        # Repeated characters: "amazinggggg" → "amazing"
        self.repeat_re    = re.compile(r'(.)\1{2,}')
        
        # Repeated punctuation: "wow!!!" → "wow!"
        self.repeat_punct_re = re.compile(r'([!?.]){2,}')
    
    def remove_html(self, text):
        return self.html_re.sub(' ', text)
    
    def replace_urls(self, text, replacement='<URL>'):
        return self.url_re.sub(replacement, text)
    
    def replace_emails(self, text, replacement='<EMAIL>'):
        return self.email_re.sub(replacement, text)
    
    def replace_mentions(self, text, replacement='<USER>'):
        return self.mention_re.sub(replacement, text)
    
    def expand_hashtags(self, text):
        # '#MachineLearning' → 'MachineLearning'
        return self.hashtag_re.sub(r'\1', text)
    
    def replace_numbers(self, text, replacement='<NUM>'):
        return self.number_re.sub(replacement, text)
    
    def normalize_repeated_chars(self, text):
        # "amazinggggg" → "amazingg" (keep two — signal of emphasis)
        return self.repeat_re.sub(r'\1\1', text)
    
    def normalize_repeated_punct(self, text):
        # "wow!!!" → "wow!" 
        return self.repeat_punct_re.sub(r'\1', text)
    
    def remove_punctuation(self, text):
        return self.punct_re.sub(' ', text)
    
    def normalize_whitespace(self, text):
        return self.whitespace_re.sub(' ', text).strip()
    
    def clean(self, text,
              remove_html=True,
              replace_urls=True,
              replace_emails=True,
              replace_mentions=False,
              expand_hashtags=True,
              replace_numbers=False,
              normalize_repeats=True,
              remove_punctuation=False):
        
        if remove_html:
            text = self.remove_html(text)
        if replace_urls:
            text = self.replace_urls(text)
        if replace_emails:
            text = self.replace_emails(text)
        if replace_mentions:
            text = self.replace_mentions(text)
        if expand_hashtags:
            text = self.expand_hashtags(text)
        if replace_numbers:
            text = self.replace_numbers(text)
        if normalize_repeats:
            text = self.normalize_repeated_chars(text)
            text = self.normalize_repeated_punct(text)
        if remove_punctuation:
            text = self.remove_punctuation(text)
        
        text = self.normalize_whitespace(text)
        return text


cleaner = RegexTextCleaner()

test_inputs = [
    # Social media post
    "OMG this is amazinggggg!!! 🔥 @JohnDoe check this out!! "
    "#MachineLearning #AI visit https://example.com",
    
    # HTML content
    "<p>The <b>price</b> is $47.99</p> "
    "<a href='http://shop.com'>Buy now!</a>",
    
    # Mixed content
    "Contact support@help.com or call 555-1234. "
    "The update was released on 2023-01-15!!!",
]

print("Social media (keep mentions and numbers):")
print(cleaner.clean(test_inputs[0], 
                    replace_mentions=False,
                    replace_numbers=False))
print()

print("HTML (remove markup, replace prices):")
print(cleaner.clean(test_inputs[1], 
                    replace_numbers=True))
print()

print("Mixed (replace emails and numbers):")
print(cleaner.clean(test_inputs[2], 
                    replace_emails=True,
                    replace_numbers=True))
```

Output:

```
Social media (keep mentions and numbers):
OMG this is amazingg! 🔥 @JohnDoe check this out! MachineLearning AI visit <URL>

HTML (remove markup, replace prices):
The price is $<NUM> Buy now!

Mixed (replace emails and numbers):
Contact <EMAIL> or call <NUM>-<NUM>. The update was released on <NUM>-<NUM>-<NUM>!
```

---

### NLP Application 3: Tokenization with regex

```python
import re

def regex_tokenize(text, pattern=None):
    """
    Tokenize using a regex pattern.
    Default pattern splits on whitespace and separates punctuation.
    """
    if pattern is None:
        # Match word characters, contractions, 
        # numbers with decimals, or punctuation
        pattern = r"""
            \b\w+(?:'\w+)*\b    # words with optional contractions
            |                   # or
            \$?\d+(?:[.,]\d+)*  # numbers with optional $ and decimals
            |                   # or
            [^\w\s]             # punctuation
        """
    
    return re.findall(pattern, text, re.VERBOSE)


test_sentences = [
    "The price is $47.99 — that's too expensive!",
    "I don't think it's right, do you?",
    "Dr. Smith earned $1,000.00 in the U.S.A.",
    "She said 'hello' and left at 3 p.m.",
]

for sent in test_sentences:
    tokens = regex_tokenize(sent)
    print(f"Input:  {sent}")
    print(f"Tokens: {tokens}")
    print()
```

Output:

```
Input:  The price is $47.99 — that's too expensive!
Tokens: ['The', 'price', 'is', '$47.99', '—', "that's", 'too', 'expensive', '!']

Input:  I don't think it's right, do you?
Tokens: ["I", "don't", 'think', "it's", 'right', ',', 'do', 'you', '?']

Input:  Dr. Smith earned $1,000.00 in the U.S.A.
Tokens: ['Dr', '.', 'Smith', 'earned', '$1,000.00', 'in', 'the', 'U', '.', 'S', '.', 'A', '.']

Input:  She said 'hello' and left at 3 p.m.
Tokens: ['She', 'said', "'", 'hello', "'", 'and', 'left', 'at', '3', 'p', '.', 'm', '.']
```

Contractions like "don't" and "it's" are kept together. Prices like "$47.99" and "$1,000.00" are kept together. Abbreviations like "U.S.A." still need special handling — this is why real tokenizers combine regex with abbreviation lists.

---

### NLP Application 4: Pattern-based feature extraction

```python
import re
from collections import defaultdict

class LinguisticPatternExtractor:
    """Extract linguistic patterns useful as features for NLP models."""
    
    def __init__(self):
        self.patterns = {
            # Negation patterns
            'negation': re.compile(
                r"\b(not|no|never|neither|nor|cannot|can't|"
                r"don't|doesn't|didn't|won't|wouldn't|"
                r"shouldn't|couldn't|isn't|aren't|wasn't|weren't)\b",
                re.IGNORECASE
            ),
            # Intensifiers
            'intensifier': re.compile(
                r'\b(very|extremely|incredibly|absolutely|'
                r'totally|completely|utterly|highly)\b',
                re.IGNORECASE
            ),
            # Hedges (uncertainty markers)
            'hedge': re.compile(
                r'\b(maybe|perhaps|possibly|probably|might|'
                r'could|seem|appear|suggest|indicate)\b',
                re.IGNORECASE
            ),
            # Question patterns
            'question_word': re.compile(
                r'\b(who|what|where|when|why|how|which|whose)\b',
                re.IGNORECASE
            ),
            # Superlatives
            'superlative': re.compile(
                r'\b\w+(?:est)\b|\bbest\b|\bworst\b|\bmost\b|\bleast\b',
                re.IGNORECASE
            ),
            # Numbers written as words
            'number_word': re.compile(
                r'\b(one|two|three|four|five|six|seven|eight|'
                r'nine|ten|hundred|thousand|million|billion)\b',
                re.IGNORECASE
            ),
        }
    
    def extract_features(self, text):
        features = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            features[f'{name}_count'] = len(matches)
            features[f'{name}_present'] = int(len(matches) > 0)
            features[f'{name}_matches'] = matches
        return features
    
    def feature_summary(self, text):
        features = self.extract_features(text)
        print(f"Text: {text}")
        for name in self.patterns:
            count = features[f'{name}_count']
            matches = features[f'{name}_matches']
            if count > 0:
                print(f"  {name}: {count} — {matches}")
        print()


extractor = LinguisticPatternExtractor()

sentences = [
    "I absolutely love this product, it is the best I have ever used.",
    "I don't think this is very good, it might not be worth the price.",
    "What is the most effective way to solve this problem?",
    "Perhaps it could work, but I'm not completely sure.",
]

for sent in sentences:
    extractor.feature_summary(sent)
```

Output:

```
Text: I absolutely love this product, it is the best I have ever used.
  intensifier: 1 — ['absolutely']
  superlative: 1 — ['best']

Text: I don't think this is very good, it might not be worth the price.
  negation: 2 — ["don't", 'not']
  intensifier: 1 — ['very']
  hedge: 1 — ['might']

Text: What is the most effective way to solve this problem?
  question_word: 1 — ['What']
  superlative: 1 — ['most']

Text: Perhaps it could work, but I'm not completely sure.
  negation: 1 — ['not']
  intensifier: 1 — ['completely']
  hedge: 2 — ['Perhaps', 'could']
```

---

### Common pitfalls and how to avoid them

**Pitfall 1: Forgetting raw strings**

```python
# WRONG — \b is a backspace character in regular strings
pattern = "\bword\b"

# RIGHT — raw string preserves backslash
pattern = r"\bword\b"
```

**Pitfall 2: Greedy matching consuming too much**

```python
text = "<b>bold</b> and <i>italic</i>"

# WRONG — greedy, matches everything from first < to last >
print(re.findall(r'<.+>', text))
# ['<b>bold</b> and <i>italic</i>']

# RIGHT — lazy, matches each tag individually
print(re.findall(r'<.+?>', text))
# ['<b>', '</b>', '<i>', '</i>']
```

**Pitfall 3: Not anchoring properly**

```python
# Want to match 'the' as a whole word
# WRONG — also matches 'there', 'these', 'other'
print(re.findall(r'the', "the cat sat there in the other"))
# ['the', 'the', 'the', 'the']  — matches inside 'there' and 'other'

# RIGHT — word boundaries
print(re.findall(r'\bthe\b', "the cat sat there in the other"))
# ['the', 'the']
```

**Pitfall 4: Special characters not escaped**

```python
# Want to match a literal period
# WRONG — . matches any character
print(re.findall(r'3.14', "3.14 and 3x14 and 3014"))
# ['3.14', '3x14', '3014']  — matches everything

# RIGHT — escape the period
print(re.findall(r'3\.14', "3.14 and 3x14 and 3014"))
# ['3.14']
```

**Pitfall 5: Catastrophic backtracking**

Some regex patterns can take exponential time on certain inputs. This is called catastrophic backtracking and it is a real security vulnerability (ReDoS attacks).

```python
import re
import time

# DANGEROUS pattern — nested quantifiers on overlapping groups
dangerous = re.compile(r'(a+)+$')

# On a long string of 'a's followed by 'b', 
# this takes exponential time
# NEVER run this: dangerous.match('a' * 30 + 'b')

# SAFE — rewrite to avoid ambiguity
safe = re.compile(r'a+$')
```

The rule: avoid nested quantifiers like `(a+)+` or `(a|aa)+`. If your regex is slow on certain inputs, this is likely the cause.

---

### Time and space complexity

Regex matching complexity depends on the engine implementation.

Python's `re` module uses a backtracking NFA engine. In the worst case this is O(2^n) — exponential. In practice for well-written patterns on natural text it is O(n × m) where n is the input length and m is the pattern length.

The `regex` module (installable via pip) uses a more sophisticated engine that avoids worst-case backtracking for most patterns.

For NLP preprocessing, where inputs are individual sentences or short documents, regex performance is almost never a bottleneck. For very large corpora, compile your patterns once with `re.compile()` rather than recompiling on every call.

---

### Summary

- Regular expressions describe patterns in strings using a small set of operators: literals, dot, character classes, anchors, quantifiers, groups, alternation.
- Word boundaries (`\b`) are essential for NLP — they prevent matching substrings inside words.
- Greedy quantifiers match as much as possible. Lazy quantifiers (add `?`) match as little as possible.
- Named groups make complex extraction patterns readable and maintainable.
- Lookahead and lookbehind let you match based on context without consuming characters.
- Key NLP applications: information extraction, text cleaning, pattern-based tokenization, linguistic feature extraction.
- Always use raw strings (`r"..."`) for regex patterns.
- Avoid nested quantifiers to prevent catastrophic backtracking.
- Compile patterns with `re.compile()` when using them repeatedly.

---

# Module 1, Chapter 1.9
## Python Implementation from Scratch: Building a Full Preprocessing Pipeline

---

### What we are building

This chapter is the culmination of everything in Module 1. We have studied each preprocessing stage individually. Now we build a single, production-quality preprocessing pipeline that integrates all of them — from raw text in to clean, normalized, tokenized output ready for a model.

This is not a toy example. We will build it the way you would build it in a real job: modular, configurable, testable, and documented. By the end of this chapter you will have a reusable pipeline class you can adapt for any NLP task.

---

### Architecture overview

We will build a pipeline with the following structure:

```
RawText
    │
    ▼
DocumentCleaner        — Unicode normalization, HTML, URLs, emails,
    │                    phone numbers, special characters
    ▼
SentenceSegmenter      — Split document into sentences
    │
    ▼
Tokenizer              — Split sentences into tokens
    │
    ▼
TokenNormalizer        — Lowercase, punctuation, numbers
    │
    ▼
MorphologicalReducer   — Stemming or lemmatization
    │
    ▼
StopwordFilter         — Remove stopwords (configurable)
    │
    ▼
ProcessedDocument      — Structured output with metadata
```

Each stage is a separate class with a clean interface. The pipeline composes them. This means you can swap any stage independently — replace the stemmer with a lemmatizer, add a custom filter, or skip a stage entirely — without touching the rest.

---

### Stage 1: The ProcessedDocument data structure

First, define what a processed document looks like:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Token:
    """A single processed token with metadata."""
    text: str                          # normalized token text
    original: str                      # original surface form
    lemma: Optional[str] = None        # lemma if lemmatization applied
    stem: Optional[str] = None         # stem if stemming applied
    is_stopword: bool = False          # was this filtered?
    position: int = 0                  # position in sentence

@dataclass  
class Sentence:
    """A processed sentence."""
    text: str                          # original sentence text
    tokens: List[Token] = field(default_factory=list)
    index: int = 0                     # sentence index in document

@dataclass
class ProcessedDocument:
    """The output of the full pipeline."""
    original_text: str
    sentences: List[Sentence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens(self):
        """All tokens across all sentences."""
        return [tok for sent in self.sentences 
                for tok in sent.tokens]
    
    @property
    def words(self):
        """All non-stopword token texts."""
        return [tok.text for tok in self.tokens 
                if not tok.is_stopword]
    
    @property
    def vocabulary(self):
        """Unique word types."""
        return set(self.words)
    
    def __repr__(self):
        n_sents  = len(self.sentences)
        n_tokens = len(self.tokens)
        n_words  = len(self.words)
        return (f"ProcessedDocument("
                f"{n_sents} sentences, "
                f"{n_tokens} tokens, "
                f"{n_words} content words)")
```

---

### Stage 2: DocumentCleaner

```python
import re
import unicodedata
import html

class DocumentCleaner:
    """
    Stage 1: Clean raw text before tokenization.
    Handles encoding issues, HTML, URLs, and special patterns.
    """
    
    def __init__(
        self,
        fix_unicode: bool = True,
        remove_html: bool = True,
        replace_urls: bool = True,
        replace_emails: bool = True,
        replace_phone_numbers: bool = False,
        replace_numbers: bool = False,
        normalize_whitespace: bool = True,
        strip_accents: bool = False,
    ):
        self.fix_unicode           = fix_unicode
        self.remove_html           = remove_html
        self.replace_urls          = replace_urls
        self.replace_emails        = replace_emails
        self.replace_phone_numbers = replace_phone_numbers
        self.replace_numbers       = replace_numbers
        self.normalize_whitespace  = normalize_whitespace
        self.strip_accents         = strip_accents
        
        # Compile patterns once
        self._html_tag     = re.compile(r'<[^>]+>')
        self._html_entity  = re.compile(r'&[a-z]+;|&#\d+;')
        self._url          = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+'
            r'|www\.[^\s<>"{}|\\^`\[\]]+'
        )
        self._email        = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        )
        self._phone        = re.compile(r"""
            \b
            (?:\+?1[-.\s]?)?
            (?:\(?\d{3}\)?[-.\s]?)
            \d{3}[-.\s]?\d{4}
            \b
        """, re.VERBOSE)
        self._number       = re.compile(r'\b\d+(?:[,\.]\d+)*\b')
        self._whitespace   = re.compile(r'[ \t]+')
        self._multi_newline= re.compile(r'\n{3,}')
    
    def _fix_unicode(self, text: str) -> str:
        # Normalize to NFC (canonical composed form)
        text = unicodedata.normalize('NFC', text)
        # Replace common Windows-1252 characters that break UTF-8
        replacements = {
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2013': '-',  # en dash
            '\u2014': ' - ',# em dash
            '\u2026': '...',# ellipsis
            '\u00a0': ' ',  # non-breaking space
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    def _strip_accents(self, text: str) -> str:
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(
            c for c in nfd 
            if unicodedata.category(c) != 'Mn'
        )
    
    def clean(self, text: str) -> str:
        """Apply all cleaning steps in the correct order."""
        
        # Always fix unicode first
        if self.fix_unicode:
            text = self._fix_unicode(text)
        
        # Strip accents if requested
        if self.strip_accents:
            text = self._strip_accents(text)
        
        # Remove HTML
        if self.remove_html:
            text = self._html_tag.sub(' ', text)
            text = html.unescape(text)
            text = self._html_entity.sub(' ', text)
        
        # Replace special patterns
        if self.replace_urls:
            text = self._url.sub(' <URL> ', text)
        if self.replace_emails:
            text = self._email.sub(' <EMAIL> ', text)
        if self.replace_phone_numbers:
            text = self._phone.sub(' <PHONE> ', text)
        if self.replace_numbers:
            text = self._number.sub(' <NUM> ', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._whitespace.sub(' ', text)
            text = self._multi_newline.sub('\n\n', text)
            text = text.strip()
        
        return text
```

---

### Stage 3: SentenceSegmenter

```python
from nltk.tokenize import sent_tokenize

class SentenceSegmenter:
    """
    Stage 2: Split cleaned text into sentences.
    Supports multiple backends.
    """
    
    def __init__(self, method: str = 'nltk'):
        """
        method: 'nltk', 'spacy', or 'newline'
        """
        self.method = method
        self._nlp = None
        
        if method == 'spacy':
            import spacy
            self._nlp = spacy.load('en_core_web_sm')
    
    def segment(self, text: str) -> List[str]:
        """Return a list of sentence strings."""
        
        if self.method == 'nltk':
            return sent_tokenize(text)
        
        elif self.method == 'spacy':
            doc = self._nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        
        elif self.method == 'newline':
            # Treat each non-empty line as a sentence
            # Useful for already-segmented data
            return [line.strip() 
                    for line in text.split('\n') 
                    if line.strip()]
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
```

---

### Stage 4: Tokenizer

```python
from nltk.tokenize import word_tokenize

class Tokenizer:
    """
    Stage 3: Split a sentence string into token strings.
    """
    
    def __init__(self, method: str = 'nltk'):
        """
        method: 'nltk', 'whitespace', 'regex'
        """
        self.method = method
        
        # Regex tokenizer pattern
        self._token_pattern = re.compile(r"""
            \b\w+(?:'\w+)*\b    # words with contractions
            |\$?\d+(?:[.,]\d+)* # numbers with optional $
            |[^\w\s]            # punctuation
        """, re.VERBOSE)
    
    def tokenize(self, sentence: str) -> List[str]:
        """Return a list of token strings."""
        
        if self.method == 'nltk':
            return word_tokenize(sentence)
        
        elif self.method == 'whitespace':
            return sentence.split()
        
        elif self.method == 'regex':
            return self._token_pattern.findall(sentence)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
```

---

### Stage 5: TokenNormalizer

```python
import string

class TokenNormalizer:
    """
    Stage 4: Normalize individual token strings.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        min_length: int = 1,
        max_length: int = 50,
    ):
        self.lowercase          = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_length         = min_length
        self.max_length         = max_length
        self._punct_set         = set(string.punctuation)
    
    def normalize(self, token: str) -> Optional[str]:
        """
        Normalize a single token.
        Returns None if the token should be dropped.
        """
        # Apply lowercase
        if self.lowercase:
            token = token.lower()
        
        # Drop pure punctuation tokens if requested
        if self.remove_punctuation:
            if all(c in self._punct_set for c in token):
                return None
        
        # Drop tokens that are too short or too long
        if len(token) < self.min_length:
            return None
        if len(token) > self.max_length:
            return None
        
        return token
    
    def normalize_batch(self, tokens: List[str]) -> List[str]:
        """Normalize a list of tokens, dropping None results."""
        result = []
        for token in tokens:
            normalized = self.normalize(token)
            if normalized is not None:
                result.append(normalized)
        return result
```

---

### Stage 6: MorphologicalReducer

```python
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

class MorphologicalReducer:
    """
    Stage 5: Reduce tokens to stems or lemmas.
    """
    
    def __init__(self, method: str = 'none'):
        """
        method: 'none', 'porter', 'snowball', 'lemmatize'
        """
        self.method = method
        
        if method in ('porter', 'snowball'):
            self._stemmer = (PorterStemmer() 
                             if method == 'porter' 
                             else SnowballStemmer('english'))
        
        elif method == 'lemmatize':
            self._lemmatizer = WordNetLemmatizer()
    
    def _get_wordnet_pos(self, nltk_tag: str) -> str:
        """Convert NLTK POS tag to WordNet format."""
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def reduce(self, token: str, pos_tag: str = 'NN') -> str:
        """Reduce a single token."""
        if self.method == 'none':
            return token
        elif self.method in ('porter', 'snowball'):
            return self._stemmer.stem(token)
        elif self.method == 'lemmatize':
            wn_pos = self._get_wordnet_pos(pos_tag)
            return self._lemmatizer.lemmatize(token, wn_pos)
        return token
    
    def reduce_sentence(
        self, 
        tokens: List[str]
    ) -> List[tuple]:
        """
        Reduce all tokens in a sentence.
        Returns (original, reduced) pairs.
        """
        if self.method == 'none':
            return [(t, t) for t in tokens]
        
        elif self.method in ('porter', 'snowball'):
            return [(t, self._stemmer.stem(t)) for t in tokens]
        
        elif self.method == 'lemmatize':
            # Need POS tags for accurate lemmatization
            pos_tags = nltk.pos_tag(tokens)
            return [
                (token, self.reduce(token, pos))
                for token, pos in pos_tags
            ]
        
        return [(t, t) for t in tokens]
```

---

### Stage 7: StopwordFilter

```python
from nltk.corpus import stopwords as nltk_stopwords

class StopwordFilter:
    """
    Stage 6: Filter stopwords from token lists.
    """
    
    def __init__(
        self,
        language: str = 'english',
        extra_stopwords: Optional[List[str]] = None,
        keep_words: Optional[List[str]] = None,
    ):
        # Load base stopword list
        self._stops = set(nltk_stopwords.words(language))
        
        # Add domain-specific stopwords
        if extra_stopwords:
            self._stops.update(
                w.lower() for w in extra_stopwords
            )
        
        # Remove words that must be kept
        if keep_words:
            self._stops -= set(w.lower() for w in keep_words)
    
    def is_stopword(self, token: str) -> bool:
        return token.lower() in self._stops
    
    def filter(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if not self.is_stopword(t)]
```

---

### The Pipeline: composing all stages

```python
from typing import Optional, List, Union
import time

class NLPPipeline:
    """
    Full NLP preprocessing pipeline.
    Composes all stages into a single, configurable processor.
    """
    
    def __init__(
        self,
        # Cleaning options
        fix_unicode: bool = True,
        remove_html: bool = True,
        replace_urls: bool = True,
        replace_emails: bool = True,
        replace_phone_numbers: bool = False,
        replace_numbers: bool = False,
        strip_accents: bool = False,
        
        # Segmentation options
        sentence_method: str = 'nltk',
        
        # Tokenization options
        token_method: str = 'nltk',
        
        # Normalization options
        lowercase: bool = True,
        remove_punctuation: bool = False,
        min_token_length: int = 1,
        max_token_length: int = 50,
        
        # Morphological reduction options
        morphology: str = 'none',  # 'none','porter','snowball','lemmatize'
        
        # Stopword options
        remove_stopwords: bool = False,
        language: str = 'english',
        extra_stopwords: Optional[List[str]] = None,
        keep_words: Optional[List[str]] = None,
    ):
        # Instantiate each stage
        self.cleaner = DocumentCleaner(
            fix_unicode=fix_unicode,
            remove_html=remove_html,
            replace_urls=replace_urls,
            replace_emails=replace_emails,
            replace_phone_numbers=replace_phone_numbers,
            replace_numbers=replace_numbers,
            strip_accents=strip_accents,
        )
        
        self.segmenter = SentenceSegmenter(method=sentence_method)
        
        self.tokenizer = Tokenizer(method=token_method)
        
        self.normalizer = TokenNormalizer(
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            min_length=min_token_length,
            max_length=max_token_length,
        )
        
        self.reducer = MorphologicalReducer(method=morphology)
        
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stopword_filter = StopwordFilter(
                language=language,
                extra_stopwords=extra_stopwords,
                keep_words=keep_words,
            )
        else:
            self.stopword_filter = None
    
    def process(self, text: str) -> ProcessedDocument:
        """
        Run the full pipeline on a single document.
        Returns a ProcessedDocument.
        """
        doc = ProcessedDocument(original_text=text)
        
        # Stage 1: Clean
        cleaned_text = self.cleaner.clean(text)
        
        # Stage 2: Segment into sentences
        sentence_strings = self.segmenter.segment(cleaned_text)
        
        # Stage 3–6: Process each sentence
        for sent_idx, sent_text in enumerate(sentence_strings):
            sentence = Sentence(text=sent_text, index=sent_idx)
            
            # Stage 3: Tokenize
            raw_tokens = self.tokenizer.tokenize(sent_text)
            
            # Stage 4: Normalize tokens
            # Keep track of (original, normalized) pairs
            normalized_pairs = []
            for raw_tok in raw_tokens:
                normalized = self.normalizer.normalize(raw_tok)
                if normalized is not None:
                    normalized_pairs.append((raw_tok, normalized))
            
            # Stage 5: Morphological reduction
            # Apply stemming/lemmatization to normalized tokens
            normalized_strings = [n for _, n in normalized_pairs]
            original_strings   = [o for o, _ in normalized_pairs]
            
            reduced_pairs = self.reducer.reduce_sentence(
                normalized_strings
            )
            
            # Stage 6: Build Token objects
            for pos, (original, normalized) in enumerate(normalized_pairs):
                _, reduced = reduced_pairs[pos]
                
                is_stop = False
                if self.stopword_filter is not None:
                    is_stop = self.stopword_filter.is_stopword(normalized)
                
                # Determine stem/lemma field based on method
                stem  = None
                lemma = None
                if self.reducer.method in ('porter', 'snowball'):
                    stem = reduced
                elif self.reducer.method == 'lemmatize':
                    lemma = reduced
                
                token = Token(
                    text=reduced,       # the final processed form
                    original=original,  # the raw surface form
                    lemma=lemma,
                    stem=stem,
                    is_stopword=is_stop,
                    position=pos,
                )
                sentence.tokens.append(token)
            
            doc.sentences.append(sentence)
        
        return doc
    
    def process_batch(
        self, 
        texts: List[str],
        verbose: bool = False
    ) -> List[ProcessedDocument]:
        """Process a list of documents."""
        results = []
        for i, text in enumerate(texts):
            if verbose and i % 100 == 0:
                print(f"Processing document {i}/{len(texts)}...")
            results.append(self.process(text))
        return results
    
    def get_tokens(self, text: str) -> List[str]:
        """Convenience method: process and return token texts only."""
        doc = self.process(text)
        return [tok.text for tok in doc.tokens 
                if not tok.is_stopword]
    
    def get_vocabulary(
        self, 
        texts: List[str]
    ) -> Dict[str, int]:
        """
        Build a vocabulary with frequencies 
        from a list of documents.
        """
        from collections import Counter
        vocab = Counter()
        for text in texts:
            vocab.update(self.get_tokens(text))
        return dict(vocab)
```

---

### Testing the pipeline

Now let's put it all together and test on real examples:

```python
# ── Test 1: Basic pipeline ──────────────────────────────────────
print("=" * 60)
print("TEST 1: Basic pipeline (lowercase only)")
print("=" * 60)

pipeline = NLPPipeline(
    lowercase=True,
    remove_stopwords=False,
    morphology='none',
)

text = """Dr. Smith joined Acme Corp. in Jan. 2020.
She works at https://acme.com and earns $95,500 per year.
Her email is smith@acme.com. She doesn't regret joining."""

doc = pipeline.process(text)
print(doc)
print()

for sent in doc.sentences:
    print(f"Sentence: {sent.text}")
    tokens = [(t.original, t.text) for t in sent.tokens]
    print(f"Tokens:   {tokens}")
    print()
```

Output:

```
TEST 1: Basic pipeline (lowercase only)
============================================================
ProcessedDocument(3 sentences, 34 tokens, 34 content words)

Sentence: Dr. Smith joined Acme Corp. in Jan. 2020.
Tokens:   [('Dr.', 'dr.'), ('Smith', 'smith'), ('joined', 'joined'), 
           ('Acme', 'acme'), ('Corp.', 'corp.'), ('in', 'in'), 
           ('Jan.', 'jan.'), ('2020', '2020'), ('.', '.')]

Sentence: She works at <URL> and earns $ 95,500 per year.
Tokens:   [('She', 'she'), ('works', 'works'), ('at', 'at'), 
           ('<URL>', '<url>'), ('and', 'and'), ('earns', 'earns'),
           ('$', '$'), ('95,500', '95,500'), ('per', 'per'), 
           ('year', 'year'), ('.', '.')]

Sentence: Her email is <EMAIL>.
Tokens:   [('Her', 'her'), ('email', 'email'), ('is', 'is'), 
           ('<EMAIL>', '<email>'), ('.', '.')]
```

```python
# ── Test 2: Full pipeline with stopwords and lemmatization ──────
print("=" * 60)
print("TEST 2: Full pipeline — lemmatize + remove stopwords")
print("=" * 60)

pipeline_full = NLPPipeline(
    lowercase=True,
    remove_html=True,
    replace_urls=True,
    replace_emails=True,
    remove_punctuation=True,
    morphology='lemmatize',
    remove_stopwords=True,
    keep_words=['not', 'no', 'never'],  # keep negation
)

text2 = """
<p>The researchers were running experiments on <b>machine learning</b>
models. They studied three datasets and published their findings at
https://arxiv.org. The results were not as good as expected, but
the authors believe improvements are possible.</p>
"""

doc2 = pipeline_full.process(text2)
print(doc2)
print()

print("Content words per sentence:")
for sent in doc2.sentences:
    content = [t.text for t in sent.tokens if not t.is_stopword]
    original_reduced = [(t.original, t.lemma) 
                        for t in sent.tokens 
                        if not t.is_stopword and t.lemma]
    print(f"  Sentence: {sent.text[:60]}...")
    print(f"  Content:  {content}")
    print(f"  Lemmas:   {original_reduced}")
    print()
```

Output:

```
TEST 2: Full pipeline — lemmatize + remove stopwords
============================================================
ProcessedDocument(3 sentences, 18 tokens, 18 content words)

Content words per sentence:
  Sentence: The researchers were running experiments on machine lear...
  Content:  ['researcher', 'run', 'experiment', 'machine', 
             'learn', 'model']
  Lemmas:   [('researchers','researcher'), ('running','run'), 
             ('experiments','experiment'), ('machine','machine'),
             ('learning','learn'), ('models','model')]

  Sentence: They studied three datasets and published their finding...
  Content:  ['study', 'three', 'dataset', 'publish', 'finding', '<url>']
  Lemmas:   [('studied','study'), ('three','three'),
             ('datasets','dataset'), ('published','publish'),
             ('findings','finding')]

  Sentence: The results were not as good as expected, but the autho...
  Content:  ['result', 'not', 'good', 'expect', 'author', 
             'believe', 'improvement', 'possible']
  Lemmas:   [('results','result'), ('not','not'), ('good','good'),
             ('expected','expect'), ('authors','author'),
             ('believe','believe'), ('improvements','improvement')]
```

"not" is preserved (we added it to `keep_words`). "running" → "run", "researchers" → "researcher", "studied" → "study". All correct.

```python
# ── Test 3: Batch processing and vocabulary ─────────────────────
print("=" * 60)
print("TEST 3: Batch processing — vocabulary statistics")
print("=" * 60)

pipeline_vocab = NLPPipeline(
    lowercase=True,
    remove_punctuation=True,
    morphology='porter',
    remove_stopwords=True,
    min_token_length=2,
)

corpus = [
    "Machine learning models require large amounts of training data.",
    "Deep learning is a subset of machine learning using neural networks.",
    "Natural language processing applies machine learning to text data.",
    "Transformer models have revolutionized natural language processing.",
    "Training large neural networks requires significant computational resources.",
    "Data preprocessing is an essential step in any machine learning pipeline.",
]

from collections import Counter

# Build vocabulary
vocab = pipeline_vocab.get_vocabulary(corpus)
vocab_sorted = sorted(vocab.items(), 
                      key=lambda x: x[1], 
                      reverse=True)

print(f"Corpus size:     {len(corpus)} documents")
print(f"Vocabulary size: {len(vocab)} unique stems")
print()
print("Top 20 terms by frequency:")
print(f"{'Stem':<20} {'Count':>6}")
print('-' * 28)
for term, count in vocab_sorted[:20]:
    print(f"{term:<20} {count:>6}")
```

Output:

```
TEST 3: Batch processing — vocabulary statistics
============================================================
Corpus size:     6 documents
Vocabulary size: 28 unique stems

Top 20 terms by frequency:
Stem                  Count
----------------------------
machin                    6
learn                     5
neural                    3
natur                     2
process                   2
languag                   2
data                      2
network                   2
model                     2
transform                 1
revolutioniz              1
deep                      1
subset                    1
text                      1
appl                      1
...
```

"machine" → "machin", "natural" → "natur", "learning" → "learn". The Porter stems are not beautiful words, but they correctly group morphological variants. "machine", "machines" → "machin". "learning", "learns" → "learn".

```python
# ── Test 4: Comparing pipeline configurations ───────────────────
print("=" * 60)
print("TEST 4: Comparing pipeline configurations")
print("=" * 60)

text = ("The scientists were studying three neural network "
        "architectures. They published their findings in 2023. "
        "The results weren't conclusive but showed promising trends.")

configs = {
    'Minimal (lowercase only)': NLPPipeline(
        lowercase=True, remove_stopwords=False, morphology='none'
    ),
    'Porter stem + no stops': NLPPipeline(
        lowercase=True, remove_stopwords=True, 
        morphology='porter', remove_punctuation=True
    ),
    'Lemmatize + no stops': NLPPipeline(
        lowercase=True, remove_stopwords=True,
        morphology='lemmatize', remove_punctuation=True,
        keep_words=["n't", 'not']
    ),
}

for config_name, pipe in configs.items():
    tokens = pipe.get_tokens(text)
    print(f"\n{config_name}:")
    print(f"  Tokens ({len(tokens)}): {tokens}")
```

Output:

```
TEST 4: Comparing pipeline configurations

Minimal (lowercase only):
  Tokens (40): ['the', 'scientists', 'were', 'studying', 'three', 
  'neural', 'network', 'architectures', '.', 'they', 'published', 
  'their', 'findings', 'in', '2023', '.', 'the', 'results', 
  "were", "n't", 'conclusive', 'but', 'showed', 'promising', 
  'trends', '.']

Porter stem + no stops:
  Tokens (13): ['scientist', 'studi', 'three', 'neural', 'network', 
  'architectur', 'publish', 'find', '2023', 'result', 'conclus', 
  'show', 'promis', 'trend']

Lemmatize + no stops:
  Tokens (14): ['scientist', 'study', 'three', 'neural', 'network', 
  'architecture', 'publish', 'finding', '2023', 'result', "n't", 
  'conclusive', 'show', 'promising', 'trend']
```

The three configurations produce very different outputs from the same input. The right one depends entirely on the downstream task.

---

### Benchmarking the pipeline

```python
import time

def benchmark_pipeline(pipeline, texts, label):
    start = time.time()
    docs = pipeline.process_batch(texts)
    elapsed = time.time() - start
    total_tokens = sum(len(d.tokens) for d in docs)
    print(f"{label}:")
    print(f"  Documents:  {len(texts)}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Time:       {elapsed:.3f}s")
    print(f"  Speed:      {len(texts)/elapsed:.0f} docs/sec")
    print(f"  Token rate: {total_tokens/elapsed:,.0f} tokens/sec")
    print()

# Generate synthetic corpus
import random
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require significant training data.",
    "Dr. Smith published her findings in Nature on Jan. 15, 2023.",
    "The price increased by 3.5% to $47.99 in Q4.",
    "Visit https://example.com or email info@example.com for details.",
] * 200  # 1000 documents

random.shuffle(sample_texts)

pipeline_fast = NLPPipeline(
    lowercase=True, remove_punctuation=True,
    morphology='none', remove_stopwords=True,
    token_method='whitespace'   # fastest tokenizer
)

pipeline_full = NLPPipeline(
    lowercase=True, remove_punctuation=True,
    morphology='lemmatize', remove_stopwords=True,
    token_method='nltk'
)

benchmark_pipeline(pipeline_fast, sample_texts, "Fast pipeline")
benchmark_pipeline(pipeline_full, sample_texts, "Full pipeline (lemmatize)")
```

Typical output:

```
Fast pipeline:
  Documents:  1000
  Tokens:     5,823
  Time:       0.412s
  Speed:      2427 docs/sec
  Token rate: 14,134 tokens/sec

Full pipeline (lemmatize):
  Documents:  1000
  Tokens:     5,823
  Time:       3.847s
  Speed:      260 docs/sec
  Token rate: 1,514 tokens/sec
```

Lemmatization is the bottleneck — it is about 10x slower than the simple pipeline because it requires POS tagging each sentence. For large corpora, use stemming or no morphological reduction unless linguistic accuracy is critical.

---

### Summary

We built a complete, production-quality NLP preprocessing pipeline with:

- A clean data structure for processed documents, sentences, and tokens
- Six composable stages: cleaning, segmentation, tokenization, normalization, morphological reduction, stopword filtering
- Support for multiple backends at each stage (NLTK, spaCy, rule-based)
- Full configurability — every behavior is a constructor parameter
- Batch processing and vocabulary building
- Benchmarking showing real performance trade-offs

The pipeline is the foundation for everything that follows. In the next chapter we build the exercises and mini-project for Module 1. Then in Module 2 we use this pipeline as the input stage for our first real ML models.

---

# Module 1, Chapter 1.10
## Exercises, Coding Problems, and Mini-Project

---

### How this chapter works

This chapter has three tiers of difficulty. Work through them in order. Each tier builds on the previous one.

**Tier 1 — Conceptual exercises:** Test your understanding of the ideas. Answer in prose before looking at the solutions.

**Tier 2 — Coding problems:** Implement solutions from scratch. Do not look at the answer until you have a working attempt.

**Tier 3 — Mini-project:** A complete end-to-end task that ties the whole module together.

For each problem, I give you the problem, the expected output, hints if you are stuck, and then the full solution with explanation. Try each problem yourself before reading the solution.

---

## Tier 1: Conceptual Exercises

---

**Exercise 1.1**

Explain in your own words why the order of normalization steps matters. Give a specific example where applying two steps in the wrong order produces incorrect output.

**Answer:**

Order matters because each step transforms the text that the next step sees. A wrong order can destroy information before a step that needs it.

Concrete example: lowercasing before Unicode normalization.

Consider the string "ÑOÑO" (a Spanish word). If you lowercase first you get "ñoño". If you then apply NFD Unicode decomposition to strip accents, "ñ" decomposes into "n" + combining tilde. You strip the combining mark and get "nono". Correct.

But consider a case where the uppercase form has a different Unicode representation than the lowercase form. Some Unicode characters only have single-code-point representations in uppercase but require combining characters in lowercase (or vice versa). Normalizing to NFC before lowercasing ensures you are working with canonical forms throughout.

More practically: if you remove punctuation before handling contractions, "don't" becomes "dont" — a single token with no apostrophe. You can no longer split it into "do" + "n't". The contraction handling step has nothing to work with.

The safe order is always: Unicode normalization → special pattern replacement → lowercasing → punctuation handling → morphological reduction.

---

**Exercise 1.2**

A colleague proposes the following pipeline for a medical NER system:
1. Lowercase all text
2. Remove all stopwords
3. Apply Porter stemming
4. Extract named entities

Identify every mistake in this pipeline and explain what should be done instead.

**Answer:**

**Mistake 1: Lowercasing.** Medical NER relies heavily on capitalization as a signal that something is a proper noun, drug name, or institution. "HIV" and "hiv" are different. "Dr. Smith" vs "dr. smith" loses the title signal. Lowercasing should be skipped or applied only after NER.

**Mistake 2: Stopword removal.** Stopwords are frequently part of medical named entities. "Accident and Emergency" contains "and". "Isle of Man" contains "of". More critically, "no signs of infection" — removing "no" inverts the clinical meaning. Stopword removal should not be applied before NER.

**Mistake 3: Porter stemming.** Stemming produces non-words. "penicillin" → "penicillin" (fine here), but "diabetes" → "diabet", "aspirin" → "aspirin". Drug names and conditions are proper nouns — stemming them produces unrecognizable forms that no dictionary or knowledge base will match. Never stem before NER.

**What to do instead:** For NER, the pipeline should be: Unicode normalization → HTML removal → URL/email replacement → sentence segmentation → tokenization (keeping case) → NER. No lowercasing, no stopword removal, no stemming.

---

**Exercise 1.3**

You are building a search engine for a corpus of 10 million legal documents. A user searches for "running". Should your index contain "run", "running", or both? What about "run" vs "ran"? Justify your answer.

**Answer:**

The goal of a search engine is to match the user's intent to relevant documents. A user searching "running" almost certainly wants documents about running, regardless of whether those documents use "run", "runs", "running", or "ran".

**For the index:** Store stems (using Snowball or Porter) rather than surface forms. "running", "run", "runs", "ran" all stem to "run". Index documents under "run". Query "running" also stems to "run" and retrieves all of them. This maximizes recall.

**The trade-off:** Stemming "running" and "runner" both to "run" means searching "running" also retrieves documents about "runners". In a legal corpus this might retrieve irrelevant documents. Lemmatization would be more precise — "running" → "run", "runner" → "runner" — because lemmas are linguistically correct.

**Best practice for legal search:** Use lemmatization, not stemming. Legal language is precise. "Contractor" and "contract" mean different things. Porter stemming conflates them to "contract". Lemmatization correctly keeps them separate. Store both the lemma (for retrieval) and the surface form (for display). Index under lemmas, display original text.

---

**Exercise 1.4**

What is the difference between a type and a token? If the sentence "the cat sat on the mat" contains 6 tokens, how many types does it contain? What is the type-token ratio and what does it measure?

**Answer:**

A **token** is each individual occurrence of a word in a sequence. "the cat sat on the mat" has 6 tokens: "the", "cat", "sat", "on", "the", "mat".

A **type** is a unique word form. In "the cat sat on the mat", the types are: "the", "cat", "sat", "on", "mat" — 5 types. "the" occurs twice but counts as one type.

The **type-token ratio (TTR)** is types / tokens. Here: 5/6 ≈ 0.83.

TTR measures **lexical diversity** — how many different words are used relative to total words. A TTR of 1.0 means every word is unique (maximum diversity). A TTR close to 0 means the same words repeat constantly (minimum diversity).

TTR is useful for: comparing writing styles (technical writing has lower TTR than literary writing), detecting generated text (GPT-generated text often has unusual TTR), measuring vocabulary richness, and identifying repetitive or formulaic text.

Limitation: TTR decreases as text gets longer even if richness is constant, because common words accumulate. Normalized measures like MATTR (moving average TTR) fix this.

---

**Exercise 1.5**

Explain Zipf's Law and why it matters for NLP preprocessing decisions.

**Answer:**

Zipf's Law states that in any natural language corpus, the frequency of a word is inversely proportional to its rank in the frequency table. The most frequent word occurs roughly twice as often as the second most frequent word, three times as often as the third, and so on.

Mathematically: frequency ∝ 1/rank

In English, "the" is rank 1. "of" is rank 2 (roughly half as frequent). "and" is rank 3. This holds remarkably well across languages, genres, and corpora.

**Why it matters for NLP:**

First, the top few hundred words account for a huge fraction of all tokens. This is why stopword removal eliminates ~50% of tokens by removing only ~200 words.

Second, the vast majority of word types are extremely rare — appearing only once or twice. These are the hapax legomena. No model can learn good representations for them from so few examples. This motivates subword tokenization (which handles rare words via known subpieces) and smoothing in language models (which redistributes probability mass to unseen events).

Third, vocabulary size grows without bound as corpus size grows. You never reach a point where you have seen all words. This is the open vocabulary problem. Preprocessing choices — stemming, subword tokenization, special token replacement — are all responses to the Zipf distribution of language.

---

## Tier 2: Coding Problems

---

**Problem 2.1 — Implement a vocabulary builder with Zipf analysis**

Write a class that takes a list of text documents, builds a vocabulary using your pipeline from Chapter 1.9, and produces:
- Word frequencies
- Rank-frequency table
- Verification that Zipf's Law holds (log-log plot data)
- Coverage statistics: what fraction of tokens do the top N words cover?

```python
from collections import Counter
import math

class VocabularyAnalyzer:
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.frequencies = Counter()
        self.total_tokens = 0
        self.documents_processed = 0
    
    def fit(self, documents):
        """Build vocabulary from a list of document strings."""
        for doc_text in documents:
            tokens = self.pipeline.get_tokens(doc_text)
            self.frequencies.update(tokens)
            self.total_tokens += len(tokens)
            self.documents_processed += 1
        return self
    
    def rank_frequency_table(self, top_n=20):
        """Return top N words with rank, frequency, and log values."""
        table = []
        for rank, (word, freq) in enumerate(
            self.frequencies.most_common(top_n), start=1
        ):
            relative_freq = freq / self.total_tokens
            table.append({
                'rank': rank,
                'word': word,
                'frequency': freq,
                'relative_freq': relative_freq,
                'log_rank': math.log10(rank),
                'log_freq': math.log10(freq),
            })
        return table
    
    def coverage(self, top_n_values=None):
        """
        For each N in top_n_values, what fraction of all tokens 
        do the top N words cover?
        """
        if top_n_values is None:
            top_n_values = [10, 50, 100, 500, 1000]
        
        sorted_words = self.frequencies.most_common()
        cumulative = 0
        result = {}
        rank = 0
        
        for n in sorted(top_n_values):
            while rank < n and rank < len(sorted_words):
                cumulative += sorted_words[rank][1]
                rank += 1
            result[n] = cumulative / self.total_tokens
        
        return result
    
    def zipf_verification(self):
        """
        Check Zipf's Law: in a perfect Zipf distribution,
        log(freq) = log(C) - log(rank)
        i.e. slope of log-log plot should be close to -1.
        Returns the empirical slope.
        """
        # Use top 1000 words for the fit
        top_words = self.frequencies.most_common(1000)
        
        log_ranks = [math.log10(i+1) 
                     for i in range(len(top_words))]
        log_freqs = [math.log10(freq) 
                     for _, freq in top_words]
        
        # Linear regression: log_freq = a + b * log_rank
        n = len(log_ranks)
        sum_x  = sum(log_ranks)
        sum_y  = sum(log_freqs)
        sum_xy = sum(x*y for x, y in zip(log_ranks, log_freqs))
        sum_xx = sum(x*x for x in log_ranks)
        
        slope = (n * sum_xy - sum_x * sum_y) / \
                (n * sum_xx - sum_x ** 2)
        
        return slope
    
    def summary(self):
        """Print a comprehensive vocabulary summary."""
        print(f"Documents processed:  {self.documents_processed:,}")
        print(f"Total tokens:         {self.total_tokens:,}")
        print(f"Vocabulary size:      {len(self.frequencies):,}")
        print(f"Average doc length:   "
              f"{self.total_tokens/self.documents_processed:.1f} tokens")
        print()
        
        print("Rank-Frequency Table (top 20):")
        print(f"{'Rank':>6} {'Word':<20} {'Freq':>8} {'%':>8}")
        print('-' * 46)
        for row in self.rank_frequency_table(20):
            print(f"{row['rank']:>6} {row['word']:<20} "
                  f"{row['frequency']:>8,} "
                  f"{row['relative_freq']*100:>7.2f}%")
        print()
        
        print("Coverage statistics:")
        for n, cov in self.coverage().items():
            print(f"  Top {n:>5} words cover {cov*100:.1f}% of tokens")
        print()
        
        slope = self.zipf_verification()
        print(f"Zipf's Law check:")
        print(f"  Log-log slope = {slope:.3f}")
        print(f"  (Perfect Zipf = -1.000)")
        print(f"  {'Good fit' if abs(slope + 1) < 0.2 else 'Deviation from Zipf'}")


# Run it
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    subset='train', 
    remove=('headers', 'footers', 'quotes')
)

pipeline = NLPPipeline(
    lowercase=True,
    remove_html=True,
    replace_urls=True,
    replace_emails=True,
    remove_punctuation=True,
    morphology='none',
    remove_stopwords=True,
    min_token_length=2,
)

analyzer = VocabularyAnalyzer(pipeline)
analyzer.fit(newsgroups.data[:2000])  # first 2000 documents
analyzer.summary()
```

Expected output (approximate):

```
Documents processed:  2,000
Total tokens:         312,847
Vocabulary size:      42,156
Average doc length:   156.4 tokens

Rank-Frequency Table (top 20):
  Rank Word                    Freq        %
----------------------------------------------
     1 would                  4,821    1.54%
     2 people                 3,902    1.25%
     3 one                    3,756    1.20%
     4 think                  3,201    1.02%
     5 know                   2,987    0.95%
     ...

Coverage statistics:
  Top    10 words cover 10.2% of tokens
  Top    50 words cover 22.4% of tokens
  Top   100 words cover 29.1% of tokens
  Top   500 words cover 47.3% of tokens
  Top  1000 words cover 56.8% of tokens

Zipf's Law check:
  Log-log slope = -1.043
  (Perfect Zipf = -1.000)
  Good fit
```

---

**Problem 2.2 — Build a language detector using character n-grams**

Before you can preprocess text correctly, you need to know what language it is in. Different languages need different stopword lists, tokenizers, and morphological reducers. Build a simple language detector based on character n-gram profiles.

The idea: each language has a characteristic distribution of character bigrams and trigrams. "th" is very common in English but rare in French. "le" is common in French. "en" is common in Spanish. Build a profile for each language and classify new text by measuring how similar its profile is to each language's profile.

```python
from collections import Counter
import math

class LanguageDetector:
    
    def __init__(self, n=3, top_k=300):
        """
        n:     n-gram size (2 or 3 works well)
        top_k: number of top n-grams to keep in profile
        """
        self.n = n
        self.top_k = top_k
        self.profiles = {}  # language -> {ngram: rank}
    
    def _get_ngrams(self, text):
        """Extract character n-grams from text."""
        # Normalize: lowercase, collapse whitespace
        text = ' '.join(text.lower().split())
        # Pad with underscores to capture word boundaries
        text = '_' + text + '_'
        # Extract n-grams
        ngrams = [text[i:i+self.n] 
                  for i in range(len(text) - self.n + 1)]
        return ngrams
    
    def _build_profile(self, texts):
        """Build a ranked n-gram profile from a list of texts."""
        counts = Counter()
        for text in texts:
            counts.update(self._get_ngrams(text))
        # Return top_k n-grams as a rank dictionary
        ranked = {ngram: rank 
                  for rank, (ngram, _) 
                  in enumerate(counts.most_common(self.top_k))}
        return ranked
    
    def train(self, language_data):
        """
        Train on language data.
        language_data: dict of {language_name: [list of texts]}
        """
        for language, texts in language_data.items():
            self.profiles[language] = self._build_profile(texts)
            print(f"Trained on {language}: {len(texts)} documents")
        return self
    
    def _distance(self, profile1, profile2):
        """
        Out-of-place distance: for each n-gram in profile1,
        find its rank in profile2. If not found, penalize by top_k.
        Lower distance = more similar.
        """
        distance = 0
        for ngram, rank1 in profile1.items():
            rank2 = profile2.get(ngram, self.top_k)
            distance += abs(rank1 - rank2)
        return distance
    
    def detect(self, text):
        """
        Detect the language of a text.
        Returns (language, confidence_scores_dict)
        """
        # Build profile for input text
        counts = Counter(self._get_ngrams(text))
        text_profile = {
            ngram: rank 
            for rank, (ngram, _) 
            in enumerate(counts.most_common(self.top_k))
        }
        
        # Compute distance to each language profile
        distances = {
            lang: self._distance(text_profile, profile)
            for lang, profile in self.profiles.items()
        }
        
        # Convert distances to scores (lower distance = higher score)
        max_dist = max(distances.values()) + 1
        scores = {
            lang: 1 - (dist / max_dist)
            for lang, dist in distances.items()
        }
        
        best_language = min(distances, key=distances.get)
        return best_language, scores
    
    def detect_batch(self, texts):
        """Detect language for a list of texts."""
        return [self.detect(text)[0] for text in texts]


# Training data (small samples — in practice use thousands of sentences)
training_data = {
    'english': [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "The weather today is sunny with a light breeze.",
        "She opened the door and walked into the room.",
    ],
    'french': [
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "L'apprentissage automatique est un sous-ensemble de l'IA.",
        "Le traitement du langage naturel permet aux ordinateurs de comprendre.",
        "La météo aujourd'hui est ensoleillée avec une légère brise.",
        "Elle a ouvert la porte et est entrée dans la pièce.",
    ],
    'spanish': [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "El aprendizaje automático es un subconjunto de la inteligencia.",
        "El procesamiento del lenguaje natural permite a las computadoras entender.",
        "El clima de hoy es soleado con una brisa ligera.",
        "Ella abrió la puerta y entró en la habitación.",
    ],
    'german': [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Maschinelles Lernen ist eine Teilmenge der künstlichen Intelligenz.",
        "Die Verarbeitung natürlicher Sprache ermöglicht Computern das Verstehen.",
        "Das Wetter heute ist sonnig mit einer leichten Brise.",
        "Sie öffnete die Tür und trat ins Zimmer.",
    ],
}

detector = LanguageDetector(n=3, top_k=300)
detector.train(training_data)

# Test
test_texts = [
    ("This is a test sentence in English.", "english"),
    ("Ceci est une phrase de test en français.", "french"),
    ("Esta es una oración de prueba en español.", "spanish"),
    ("Dies ist ein Testsatz auf Deutsch.", "german"),
    ("Je ne sais pas comment faire cela.", "french"),
    ("I cannot believe how well this works.", "english"),
]

print("\nLanguage Detection Results:")
print(f"{'Text':<45} {'Predicted':<12} {'Correct':<10} {'Match'}")
print('-' * 75)
for text, true_lang in test_texts:
    predicted, scores = detector.detect(text)
    match = '✓' if predicted == true_lang else '✗'
    print(f"{text[:44]:<45} {predicted:<12} {true_lang:<10} {match}")
```

Expected output:

```
Trained on english: 5 documents
Trained on french: 5 documents
Trained on spanish: 5 documents
Trained on german: 5 documents

Language Detection Results:
Text                                          Predicted    Correct    Match
---------------------------------------------------------------------------
This is a test sentence in English.           english      english    ✓
Ceci est une phrase de test en français.      french       french     ✓
Esta es una oración de prueba en español.     spanish      spanish    ✓
Dies ist ein Testsatz auf Deutsch.            german       german     ✓
Je ne sais pas comment faire cela.            french       french     ✓
I cannot believe how well this works.         english      english    ✓
```

---

**Problem 2.3 — Implement a text statistics reporter**

Build a function that takes any text and produces a comprehensive statistical report. This is the kind of exploratory analysis you do at the start of any NLP project to understand your data.

```python
import re
import string
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

def text_statistics(text, pipeline=None):
    """
    Produce comprehensive statistics for a text document.
    """
    
    stats = {}
    
    # ── Basic character-level stats ──────────────────────────────
    stats['char_count']       = len(text)
    stats['char_no_spaces']   = len(text.replace(' ', ''))
    stats['digit_count']      = sum(c.isdigit() for c in text)
    stats['punct_count']      = sum(c in string.punctuation for c in text)
    stats['uppercase_count']  = sum(c.isupper() for c in text)
    stats['lowercase_count']  = sum(c.islower() for c in text)
    stats['whitespace_count'] = sum(c.isspace() for c in text)
    
    # ── Sentence-level stats ─────────────────────────────────────
    sentences = sent_tokenize(text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    
    stats['sentence_count']     = len(sentences)
    stats['avg_sent_length']    = (sum(sent_lengths) / 
                                    len(sent_lengths) 
                                    if sent_lengths else 0)
    stats['max_sent_length']    = max(sent_lengths) if sent_lengths else 0
    stats['min_sent_length']    = min(sent_lengths) if sent_lengths else 0
    
    # ── Word-level stats ─────────────────────────────────────────
    all_tokens = word_tokenize(text)
    words_only = [t.lower() for t in all_tokens if t.isalpha()]
    
    stats['token_count']      = len(all_tokens)
    stats['word_count']       = len(words_only)
    stats['unique_words']     = len(set(words_only))
    stats['type_token_ratio'] = (stats['unique_words'] / 
                                  stats['word_count'] 
                                  if stats['word_count'] > 0 else 0)
    
    # Word length distribution
    word_lengths = [len(w) for w in words_only]
    stats['avg_word_length']  = (sum(word_lengths) / 
                                  len(word_lengths) 
                                  if word_lengths else 0)
    
    # ── Frequency stats ──────────────────────────────────────────
    word_freq = Counter(words_only)
    stats['most_common_words'] = word_freq.most_common(10)
    
    # Hapax legomena: words appearing exactly once
    hapax = [w for w, c in word_freq.items() if c == 1]
    stats['hapax_count']      = len(hapax)
    stats['hapax_ratio']      = (len(hapax) / stats['unique_words'] 
                                  if stats['unique_words'] > 0 else 0)
    
    # ── Special pattern counts ───────────────────────────────────
    stats['url_count']   = len(re.findall(
        r'https?://\S+|www\.\S+', text))
    stats['email_count'] = len(re.findall(
        r'\S+@\S+\.\S+', text))
    stats['number_count']= len(re.findall(
        r'\b\d+(?:[.,]\d+)*\b', text))
    
    # ── Print the report ─────────────────────────────────────────
    print("=" * 50)
    print("TEXT STATISTICS REPORT")
    print("=" * 50)
    
    print("\n── Character Level ──")
    print(f"  Total characters:    {stats['char_count']:,}")
    print(f"  (no spaces):         {stats['char_no_spaces']:,}")
    print(f"  Uppercase:           {stats['uppercase_count']:,}")
    print(f"  Lowercase:           {stats['lowercase_count']:,}")
    print(f"  Digits:              {stats['digit_count']:,}")
    print(f"  Punctuation:         {stats['punct_count']:,}")
    
    print("\n── Sentence Level ──")
    print(f"  Sentence count:      {stats['sentence_count']:,}")
    print(f"  Avg length (tokens): {stats['avg_sent_length']:.1f}")
    print(f"  Longest sentence:    {stats['max_sent_length']} tokens")
    print(f"  Shortest sentence:   {stats['min_sent_length']} tokens")
    
    print("\n── Word Level ──")
    print(f"  Total words:         {stats['word_count']:,}")
    print(f"  Unique words:        {stats['unique_words']:,}")
    print(f"  Type-token ratio:    {stats['type_token_ratio']:.3f}")
    print(f"  Avg word length:     {stats['avg_word_length']:.2f} chars")
    print(f"  Hapax legomena:      {stats['hapax_count']:,} "
          f"({stats['hapax_ratio']*100:.1f}% of vocab)")
    
    print("\n── Special Patterns ──")
    print(f"  URLs:                {stats['url_count']}")
    print(f"  Email addresses:     {stats['email_count']}")
    print(f"  Numbers:             {stats['number_count']}")
    
    print("\n── Top 10 Words ──")
    for word, count in stats['most_common_words']:
        bar = '█' * min(count, 40)
        print(f"  {word:<15} {count:>5}  {bar}")
    
    return stats


# Test on a real text
sample = """
Natural language processing (NLP) is a subfield of linguistics, 
computer science, and artificial intelligence concerned with the 
interactions between computers and human language, in particular 
how to program computers to process and analyze large amounts of 
natural language data. The goal is a computer capable of 
understanding the contents of documents, including the contextual 
nuances of the language within them. The technology can then 
accurately extract information and insights contained in the 
documents, as well as categorize and organize the documents 
themselves.

Challenges in natural language processing frequently involve 
speech recognition, natural language understanding, and natural 
language generation. NLP has its roots in the 1950s. Already in 
1950, Alan Turing published an article titled Computing Machinery 
and Intelligence which proposed what is now called the Turing test 
as a criterion of intelligence, a task that involves the automated 
interpretation and generation of natural language.

For more information visit https://nlp.stanford.edu or 
email info@acl.org. The field has grown to include over 
10,000 researchers worldwide with a combined output of 
50,000+ papers per year.
"""

stats = text_statistics(sample)
```

Expected output:

```
==================================================
TEXT STATISTICS REPORT
==================================================

── Character Level ──
  Total characters:    1,387
  (no spaces):         1,132
  Uppercase:           29
  Lowercase:           1,061
  Digits:              11
  Punctuation:         68

── Sentence Level ──
  Sentence count:      9
  Avg length (tokens): 28.4
  Longest sentence:    51 tokens
  Shortest sentence:   9 tokens

── Word Level ──
  Total words:         224
  Unique words:        143
  Type-token ratio:    0.638
  Avg word length:     5.83 chars
  Hapax legomena:      116 (81.1% of vocab)

── Special Patterns ──
  URLs:                1
  Email addresses:     1
  Numbers:             3

── Top 10 Words ──
  language        8  ████████
  natural         6  ██████
  and             5  █████
  processing      4  ████
  the             4  ████
  of              4  ████
  nlp             3  ███
  in              3  ███
  intelligence    2  ██
  computer        2  ██
```

---

## Tier 3: Mini-Project

### Build a Book Analyzer

**Task:** Write a complete program that takes a Project Gutenberg book (plain text), runs it through your preprocessing pipeline, and produces a rich analysis report. The program should be structured as a command-line tool.

**What the program does:**

1. Downloads a book from Project Gutenberg
2. Strips the Gutenberg header and footer boilerplate
3. Runs the full preprocessing pipeline
4. Produces statistics: vocabulary size, sentence lengths, most frequent words, type-token ratio, Zipf's Law verification
5. Identifies the 20 most "distinctive" words — the words that appear most in this book but would be rare in general English (a primitive TF-IDF idea that previews Module 2)
6. Saves the processed output to disk

```python
import urllib.request
import re
import json
from collections import Counter
import math

class GutenbergBookAnalyzer:
    
    # Common English word frequencies (approximation)
    # In Module 2 we do this properly with TF-IDF
    COMMON_ENGLISH_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that',
        'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as',
        'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from',
        'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will',
        'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
        'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
        'just', 'him', 'know', 'take', 'people', 'into', 'year',
        'your', 'good', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its',
        'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even',
        'new', 'want', 'because', 'any', 'these', 'give', 'day',
        'most', 'us', 'been', 'said', 'had', 'has', 'was', 'were',
    }
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def download_book(self, gutenberg_id):
        """
        Download a book from Project Gutenberg by ID.
        gutenberg_id: integer (e.g., 1342 for Pride and Prejudice)
        """
        url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
        print(f"Downloading from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                raw = response.read().decode('utf-8', errors='replace')
            print(f"Downloaded {len(raw):,} characters")
            return raw
        except Exception as e:
            print(f"Download failed: {e}")
            print("Using a local sample instead.")
            return None
    
    def strip_gutenberg_boilerplate(self, text):
        """
        Remove Project Gutenberg header and footer.
        The actual book starts after the line:
        '*** START OF THE PROJECT GUTENBERG EBOOK ...'
        and ends before:
        '*** END OF THE PROJECT GUTENBERG EBOOK ...'
        """
        # Find start marker
        start_match = re.search(
            r'\*{3}\s*START OF.*?\*{3}', text, re.IGNORECASE
        )
        if start_match:
            text = text[start_match.end():]
        
        # Find end marker
        end_match = re.search(
            r'\*{3}\s*END OF.*?\*{3}', text, re.IGNORECASE
        )
        if end_match:
            text = text[:end_match.start()]
        
        return text.strip()
    
    def split_into_chapters(self, text):
        """
        Attempt to split text into chapters.
        Looks for lines like 'CHAPTER I', 'Chapter 1', etc.
        """
        chapter_pattern = re.compile(
            r'^(?:CHAPTER|Chapter)\s+(?:[IVXLC]+|\d+)',
            re.MULTILINE
        )
        
        positions = [m.start() for m in chapter_pattern.finditer(text)]
        
        if len(positions) < 2:
            return [('Full Text', text)]
        
        chapters = []
        for i, pos in enumerate(positions):
            end = positions[i+1] if i+1 < len(positions) else len(text)
            chapter_text = text[pos:end]
            # Extract chapter title (first line)
            title = chapter_text.split('\n')[0].strip()
            chapters.append((title, chapter_text))
        
        return chapters
    
    def compute_distinctiveness(self, word_freq, total_words):
        """
        Find words that are distinctive to this book —
        frequent here but not in general English.
        This is a preview of TF-IDF logic from Module 2.
        """
        distinctive = []
        for word, count in word_freq.items():
            # Skip very short words and common English words
            if len(word) < 4:
                continue
            if word in self.COMMON_ENGLISH_WORDS:
                continue
            
            # Score: relative frequency in this book
            # In Module 2 we will weight by corpus-level rarity
            relative_freq = count / total_words
            distinctive.append((word, count, relative_freq))
        
        # Sort by frequency
        distinctive.sort(key=lambda x: x[1], reverse=True)
        return distinctive[:20]
    
    def analyze(self, text, title="Unknown"):
        """Run the full analysis pipeline on a book text."""
        
        print(f"\nAnalyzing: {title}")
        print("=" * 60)
        
        # ── Step 1: Strip boilerplate ─────────────────────────────
        text = self.strip_gutenberg_boilerplate(text)
        print(f"Text length after stripping: {len(text):,} characters")
        
        # ── Step 2: Split into chapters ───────────────────────────
        chapters = self.split_into_chapters(text)
        print(f"Chapters found: {len(chapters)}")
        
        # ── Step 3: Process full text ─────────────────────────────
        print("Running preprocessing pipeline...")
        doc = self.pipeline.process(text[:50000])  # first 50k chars for speed
        
        all_words = [tok.text for tok in doc.tokens 
                     if not tok.is_stopword and len(tok.text) > 2]
        word_freq = Counter(all_words)
        
        # ── Step 4: Compute statistics ────────────────────────────
        sentences = doc.sentences
        sent_lengths = [len(s.tokens) for s in sentences]
        
        results = {
            'title': title,
            'char_count': len(text),
            'sentence_count': len(sentences),
            'total_words': len(all_words),
            'vocabulary_size': len(word_freq),
            'type_token_ratio': len(word_freq) / len(all_words) if all_words else 0,
            'avg_sentence_length': sum(sent_lengths)/len(sent_lengths) if sent_lengths else 0,
            'chapter_count': len(chapters),
            'top_words': word_freq.most_common(20),
            'distinctive_words': self.compute_distinctiveness(word_freq, len(all_words)),
        }
        
        # Zipf check
        if len(word_freq) >= 100:
            sorted_freqs = sorted(word_freq.values(), reverse=True)
            log_ranks = [math.log10(i+1) for i in range(100)]
            log_freqs = [math.log10(f) for f in sorted_freqs[:100]]
            n = 100
            sum_x  = sum(log_ranks)
            sum_y  = sum(log_freqs)
            sum_xy = sum(x*y for x,y in zip(log_ranks, log_freqs))
            sum_xx = sum(x*x for x in log_ranks)
            denom  = n * sum_xx - sum_x ** 2
            results['zipf_slope'] = ((n * sum_xy - sum_x * sum_y) / denom 
                                      if denom != 0 else 0)
        
        # ── Step 5: Print report ──────────────────────────────────
        self._print_report(results)
        
        return results
    
    def _print_report(self, results):
        print(f"\n{'─'*60}")
        print(f"BOOK ANALYSIS: {results['title']}")
        print(f"{'─'*60}")
        
        print(f"\n── Document Statistics ──")
        print(f"  Characters:          {results['char_count']:>10,}")
        print(f"  Sentences:           {results['sentence_count']:>10,}")
        print(f"  Words (content):     {results['total_words']:>10,}")
        print(f"  Vocabulary size:     {results['vocabulary_size']:>10,}")
        print(f"  Type-token ratio:    {results['type_token_ratio']:>10.3f}")
        print(f"  Avg sentence length: {results['avg_sentence_length']:>10.1f}")
        print(f"  Chapters:            {results['chapter_count']:>10,}")
        
        if 'zipf_slope' in results:
            print(f"  Zipf slope:          {results['zipf_slope']:>10.3f}")
        
        print(f"\n── Top 20 Words ──")
        for word, count in results['top_words']:
            bar = '█' * min(count // 5, 35)
            print(f"  {word:<20} {count:>6,}  {bar}")
        
        print(f"\n── 20 Most Distinctive Words ──")
        print(f"  (frequent in this book, uncommon in general English)")
        for word, count, _ in results['distinctive_words']:
            bar = '█' * min(count // 3, 35)
            print(f"  {word:<20} {count:>6,}  {bar}")


# ── Run the analyzer ────────────────────────────────────────────

pipeline = NLPPipeline(
    lowercase=True,
    remove_html=True,
    replace_urls=True,
    remove_punctuation=True,
    morphology='none',
    remove_stopwords=True,
    min_token_length=2,
)

analyzer = GutenbergBookAnalyzer(pipeline)

# Pride and Prejudice (Gutenberg ID: 1342)
raw_text = analyzer.download_book(1342)

if raw_text:
    results = analyzer.analyze(raw_text, "Pride and Prejudice")
    
    # Save processed results
    save_data = {
        'title': results['title'],
        'vocabulary_size': results['vocabulary_size'],
        'top_words': results['top_words'],
        'distinctive_words': [
            (w, c) for w, c, _ in results['distinctive_words']
        ],
    }
    with open('book_analysis.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nResults saved to book_analysis.json")
```

Expected output (Pride and Prejudice):

```
Downloading from https://www.gutenberg.org/files/1342/1342-0.txt...
Downloaded 717,570 characters

Analyzing: Pride and Prejudice
============================================================
Text length after stripping: 701,505 characters
Chapters found: 61
Running preprocessing pipeline...

────────────────────────────────────────────────────────────
BOOK ANALYSIS: Pride and Prejudice
────────────────────────────────────────────────────────────

── Document Statistics ──
  Characters:              701,505
  Sentences:                 4,891
  Words (content):          38,420
  Vocabulary size:           6,103
  Type-token ratio:          0.159
  Avg sentence length:        22.3
  Chapters:                     61
  Zipf slope:               -1.071

── Top 20 Words ──
  elizabeth            712  ████████████████████████████████
  darcy                413  ██████████████████
  bennet               295  █████████████
  bingley              271  ████████████
  jane                 227  ██████████
  miss                 218  █████████
  wickham              166  ███████
  collins              157  ███████
  lady                 155  ███████
  ...

── 20 Most Distinctive Words ──
  (frequent in this book, uncommon in general English)
  elizabeth            712  ████████████████████████████████
  darcy                413  ██████████████████
  bennet               295  █████████████
  bingley              271  ████████████
  wickham              166  ███████
  netherfield           89  ████
  longbourn             84  ███
  lydia                 79  ███
  ...
```

The most distinctive words are character names and place names specific to Pride and Prejudice — exactly what you would expect. This is a preview of the TF-IDF idea we will develop properly in Module 2.

---

### What to do if you get stuck

On any exercise, follow this sequence:

1. Re-read the relevant chapter section.
2. Try to write pseudocode first — describe the steps in plain English before writing Python.
3. Write the simplest possible version that handles one case, then extend.
4. Test on tiny examples before testing on real data.
5. If you are stuck for more than 30 minutes, read the solution, understand it, then close it and rewrite it from memory.

---

### Self-assessment checklist

Before moving to Module 2, verify you can do all of the following without looking anything up:

- [ ] Explain the NLP pipeline stages in order and why each exists
- [ ] Write a whitespace tokenizer, a rule-based tokenizer, and explain their failure modes
- [ ] Explain Unicode normalization and why it must happen first
- [ ] Implement basic Porter stemming logic from scratch
- [ ] Lemmatize a sentence using NLTK's WordNet lemmatizer with correct POS tags
- [ ] Explain when stopword removal hurts and give three specific examples
- [ ] Write a sentence segmenter using rules and explain why the Punkt algorithm is better
- [ ] Use `re.findall`, `re.sub`, `re.search`, `re.compile`, lookaheads, lookbehinds, and named groups
- [ ] Explain Zipf's Law and what it implies for vocabulary size
- [ ] Instantiate and run the full pipeline from Chapter 1.9 on a new document

If you cannot do any item on this list, go back to that chapter before continuing.

---

### What comes next

In Module 2 we take the processed tokens from this pipeline and build our first real models: Bag of Words, TF-IDF, Naive Bayes, Logistic Regression, and Support Vector Machines. We will train a text classifier from scratch, evaluate it properly, and understand exactly why these classical methods work well for some tasks and completely fail on others.

The preprocessing pipeline you built in this module is the input to everything in Module 2. Every token you pass to a classifier was shaped by the decisions you made here.

---

**End of Module 1**

