
## NLP Mastery Curriculum — Chapter-Level Plan

---

### MODULE 1 · NLP Foundations and Text Preprocessing

- Ch 1.1 — What is NLP? Problem landscape, applications, and why it's hard
- Ch 1.2 — The NLP pipeline: from raw text to structured data
- Ch 1.3 — Tokenization: whitespace, rule-based, subword intuition (preview)
- Ch 1.4 — Normalization: lowercasing, punctuation, unicode handling
- Ch 1.5 — Stemming vs lemmatization: algorithms and trade-offs
- Ch 1.6 — Stopword removal: when it helps and when it hurts
- Ch 1.7 — Sentence segmentation and paragraph structure
- Ch 1.8 — Regular expressions for NLP: patterns, extraction, cleaning
- Ch 1.9 — Python implementation from scratch: building a full preprocessing pipeline
- Ch 1.10 — Exercises, coding problems, and mini-project

---

### MODULE 2 · Classical NLP and Feature Engineering

- Ch 2.1 — The representation problem: how do we feed text to a model?
- Ch 2.2 — Bag of Words: intuition, construction, limitations
- Ch 2.3 — TF-IDF: term frequency, inverse document frequency, full derivation
- Ch 2.4 — N-gram features: unigrams, bigrams, trigrams
- Ch 2.5 — Naive Bayes classifier: probability review, derivation, implementation
- Ch 2.6 — Logistic Regression for text classification
- Ch 2.7 — Support Vector Machines: intuition and text applications
- Ch 2.8 — Evaluation metrics: accuracy, precision, recall, F1, confusion matrix
- Ch 2.9 — Sentiment analysis project from scratch
- Ch 2.10 — Limitations of classical methods: the sparsity and semantics problem
- Ch 2.11 — Exercises, coding problems, and mini-project

---

### MODULE 3 · Probabilistic Language Modeling

- Ch 3.1 — What is a language model? Probability over sequences
- Ch 3.2 — The chain rule of probability
- Ch 3.3 — Unigram, bigram, trigram models: derivation and intuition
- Ch 3.4 — Maximum likelihood estimation
- Ch 3.5 — The sparsity problem: why most n-grams are never seen
- Ch 3.6 — Laplace (add-one) smoothing
- Ch 3.7 — Backoff and interpolation
- Ch 3.8 — Kneser-Ney smoothing: derivation and intuition
- Ch 3.9 — Perplexity: what it measures and how to compute it
- Ch 3.10 — Python implementation: building an n-gram language model
- Ch 3.11 — Exercises, coding problems, and mini-project

---

### MODULE 4 · Hidden Markov Models and Conditional Random Fields

- Ch 4.1 — Sequence labeling: the problem and why it's different from classification
- Ch 4.2 — POS tagging and NER as motivating tasks
- Ch 4.3 — Markov assumption and Markov chains
- Ch 4.4 — Hidden Markov Models: states, observations, parameters
- Ch 4.5 — The three HMM problems: likelihood, decoding, learning
- Ch 4.6 — Forward algorithm: full derivation and implementation
- Ch 4.7 — Viterbi algorithm: full derivation, numerical example, implementation
- Ch 4.8 — Baum-Welch algorithm: intuition and overview
- Ch 4.9 — Limitations of HMMs: independence assumptions, feature constraints
- Ch 4.10 — Conditional Random Fields: discriminative vs generative models
- Ch 4.11 — CRF objective, features, and inference
- Ch 4.12 — HMM vs CRF: when to use which
- Ch 4.13 — Python implementation: POS tagger with HMM, NER with CRF
- Ch 4.14 — Exercises, coding problems, and mini-project

---

### MODULE 5 · Word Embeddings

- Ch 5.1 — The semantic gap: why BoW and n-grams miss meaning
- Ch 5.2 — Distributional hypothesis: you shall know a word by its company
- Ch 5.3 — Co-occurrence matrices and SVD-based embeddings
- Ch 5.4 — Word2Vec: the core idea and two architectures
- Ch 5.5 — Skip-gram: objective, derivation, intuition
- Ch 5.6 — CBOW: objective and comparison to skip-gram
- Ch 5.7 — Negative sampling: why and how
- Ch 5.8 — Training Word2Vec from scratch in Python
- Ch 5.9 — GloVe: global co-occurrence, objective derivation, intuition
- Ch 5.10 — FastText: subword embeddings and OOV handling
- Ch 5.11 — Evaluating embeddings: analogy tasks, similarity benchmarks
- Ch 5.12 — Limitations of static embeddings: the polysemy problem
- Ch 5.13 — Exercises, coding problems, and mini-project

---

### MODULE 6 · Neural Language Models

- Ch 6.1 — Why neural networks for language? Generalizing beyond n-grams
- Ch 6.2 — Neural network review: forward pass, loss, backprop
- Ch 6.3 — Bengio's feedforward language model: architecture and intuition
- Ch 6.4 — Embeddings as a learned lookup table
- Ch 6.5 — Training neural LMs: cross-entropy loss, SGD, mini-batches
- Ch 6.6 — Vanishing and exploding gradients: the core problem
- Ch 6.7 — Dropout, weight initialization, and training tricks
- Ch 6.8 — Implementing a feedforward LM from scratch in Python
- Ch 6.9 — Comparing n-gram LMs vs neural LMs on perplexity
- Ch 6.10 — Exercises, coding problems, and mini-project

---

### MODULE 7 · Recurrent Neural Networks, LSTMs, and GRUs

- Ch 7.1 — The sequence problem: why feedforward networks fall short
- Ch 7.2 — Recurrent Neural Networks: architecture, hidden state, intuition
- Ch 7.3 — RNN forward pass: full derivation and numerical example
- Ch 7.4 — Backpropagation through time (BPTT): full derivation
- Ch 7.5 — Vanishing gradient in RNNs: why it happens, why it matters
- Ch 7.6 — Long Short-Term Memory: motivation and architecture
- Ch 7.7 — LSTM gates: forget, input, output — full derivation
- Ch 7.8 — LSTM numerical walkthrough: step by step
- Ch 7.9 — Gated Recurrent Unit: architecture, comparison to LSTM
- Ch 7.10 — Bidirectional RNNs and stacked RNNs
- Ch 7.11 — Implementing RNN and LSTM from scratch in Python
- Ch 7.12 — Character-level language model project
- Ch 7.13 — Exercises, coding problems, and mini-project

---

### MODULE 8 · Sequence-to-Sequence Models and Decoding

- Ch 8.1 — Variable-length input to variable-length output: the new problem
- Ch 8.2 — Encoder-decoder architecture: intuition and design
- Ch 8.3 — The context vector: what it encodes and its bottleneck
- Ch 8.4 — Teacher forcing: what it is and why it's used
- Ch 8.5 — Greedy decoding: fast but suboptimal
- Ch 8.6 — Beam search: algorithm, full derivation, numerical example
- Ch 8.7 — Length normalization and decoding hyperparameters
- Ch 8.8 — BLEU score: derivation and interpretation
- Ch 8.9 — Implementing a Seq2Seq model from scratch
- Ch 8.10 — The fundamental limitations of RNN-based Seq2Seq
- Ch 8.11 — The bottleneck problem, long-range dependency failure, serial computation
- Ch 8.12 — Setting the stage: what a better architecture needs to solve
- Ch 8.13 — Exercises, coding problems, and mini-project

---

### *** PHASE 1 ENDS HERE ***

*By the end of Module 8 you will be able to clearly articulate why RNNs and Seq2Seq models were insufficient — the bottleneck, the inability to parallelize, the collapse of long-range information — and exactly what problem Transformers were designed to solve.*

---

### MODULE 9 · Attention Mechanism

- Ch 9.1 — The core idea: letting the decoder look at all encoder states
- Ch 9.2 — Bahdanau (additive) attention: full derivation and numerical example
- Ch 9.3 — Luong (multiplicative) attention: derivation and comparison
- Ch 9.4 — Attention weights as alignment: visualizing what the model learns
- Ch 9.5 — Dot-product attention and scaled dot-product attention
- Ch 9.6 — Self-attention: attending to your own sequence
- Ch 9.7 — Multi-head attention: intuition, derivation, implementation
- Ch 9.8 — Implementing attention from scratch in Python
- Ch 9.9 — Exercises, coding problems, and mini-project

---

### MODULE 10 · The Transformer Architecture

- Ch 10.1 — "Attention is All You Need": what the paper proposed and why it mattered
- Ch 10.2 — Positional encoding: why and how
- Ch 10.3 — The encoder block: self-attention + feedforward + residuals + layer norm
- Ch 10.4 — The decoder block: masked self-attention, cross-attention, feedforward
- Ch 10.5 — Full encoder-decoder Transformer: end-to-end walkthrough
- Ch 10.6 — Complexity analysis: why Transformers parallelize where RNNs cannot
- Ch 10.7 — Implementing the Transformer from scratch in Python
- Ch 10.8 — Training a Transformer on a translation task
- Ch 10.9 — Exercises, coding problems, and mini-project

---

### MODULE 11 · BERT, GPT, and Pre-trained Language Models

- Ch 11.1 — The pre-training paradigm: learn language first, fine-tune second
- Ch 11.2 — GPT: autoregressive language modeling with Transformers
- Ch 11.3 — BERT: masked language modeling and next sentence prediction
- Ch 11.4 — BERT vs GPT: encoder-only vs decoder-only, when to use which
- Ch 11.5 — Fine-tuning BERT for classification, NER, and QA
- Ch 11.6 — The Hugging Face ecosystem: models, tokenizers, pipelines
- Ch 11.7 — Tokenization revisited: WordPiece, BPE, SentencePiece
- Ch 11.8 — Practical fine-tuning project with real data
- Ch 11.9 — Exercises, coding problems, and mini-project

---


---

That's the full chapter map — roughly 100+ chapters across 11 modules. Where would you like to start, or do you want to adjust anything before we begin?
