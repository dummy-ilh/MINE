**BEIR** (Benchmarking-IR) is a heterogeneous benchmark for evaluating **zero-shot information retrieval** models — i.e., how well a retriever generalizes to new domains and tasks without being fine-tuned on them. It was introduced by Thakur et al. in 2021 and has become a standard reference point for comparing dense retrievers, sparse retrievers (like BM25), and hybrid methods.

**Why it exists**
Most retrieval models used to be trained and evaluated on MS MARCO, which doesn't tell you much about how they'll perform on legal documents, scientific papers, or fact-checking tasks. BEIR tests robustness across very different domains and query types.

**Structure**
It bundles **18 datasets** across 9 task types, including:
- **Fact-checking**: FEVER, Climate-FEVER, SciFact
- **Question answering**: Natural Questions, HotpotQA, FiQA
- **Bio-medical IR**: TREC-COVID, NFCorpus, BioASQ
- **Duplicate question retrieval**: Quora
- **Entity retrieval**: DBPedia
- **News retrieval**: TREC-NEWS, Robust04
- **Citation prediction**: SCIDOCS
- **Argument retrieval**: ArguAna, Touché-2020
- **Tweet retrieval**: Signal-1M

**Metric**
Primarily **NDCG@10**, averaged across datasets, since it captures ranking quality well for retrieval.

**Key findings from the original paper**
- BM25 (a decades-old sparse lexical method) turned out to be a surprisingly strong zero-shot baseline — many neural dense retrievers fine-tuned only on MS MARCO underperformed it out-of-domain.
- This spurred a lot of follow-up work on making dense retrievers more robust (e.g., contrastive pretraining, hybrid sparse+dense approaches, instruction-tuned embedding models).

If you want, I can search for the current leaderboard to see which models top BEIR today (this space moves fast and rankings shift often with new embedding models) — want me to pull that up?
