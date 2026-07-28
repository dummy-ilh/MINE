# Chapter 5 — Attention as Explanation, and Its Limits

Chapter 4 covered gradient-based explanation methods. This chapter covers a different, transformer-specific candidate for explanation — attention weights — and the genuine, ongoing debate about whether they deserve to be treated as an explanation at all. It closes with Concept Activation Vectors (TCAV), which sidesteps the whole debate by asking a different kind of question entirely.

## 5.1 The intuitive appeal of attention weights

**What attention weights are, briefly.** In a transformer, at every layer, each token computes an attention weight over every other token — a number representing "how much should this token's representation be influenced by that other token, at this layer." These weights are a completely natural byproduct of how the model computes its output — no extra computation is required to obtain them, unlike every method in Chapter 4, which required computing a gradient or a specific downstream quantity specifically for explanation purposes.

**Why it's tempting to treat attention weights directly as an explanation:** if a translation model, translating the word "bank" in a sentence about rivers, shows a high attention weight from "bank" to the word "river" elsewhere in the sentence, this seems to directly and transparently show "the model looked at 'river' to correctly disambiguate 'bank' as a riverbank rather than a financial institution" — a satisfyingly legible, human-readable story about the model's reasoning, apparently available "for free" just by reading off numbers the model already computed.

## 5.2 The "attention is not explanation" debate

This is a genuine, actively-debated methodological question in NLP interpretability research, and it's worth knowing the actual evidence on both sides, not just the headline claim.

**The core empirical challenge (Jain & Wallace, 2019, among others):** researchers found that you can often **substantially alter a model's attention weights while barely changing its output prediction at all** — meaning the attention pattern isn't uniquely determined by what the model needs to produce its answer; multiple, quite different attention patterns can produce essentially the same output. If attention weights genuinely reflected "what the model is using to make its decision" in a strong, load-bearing sense, you wouldn't expect the output to survive such a large change to the attention pattern with so little effect. This finding is the crux of the "attention is not [a reliable, unique] explanation" position.

**A partial rebuttal (Wiegreffe & Pinter, 2019, among others):** this line of work argued that some of the adversarial attention-manipulation experiments used **implausible** alternative attention distributions — ones that could never actually arise from the model's own training process — and that when you restrict the comparison to attention distributions the model could plausibly have learned, attention weights show more consistency with other importance measures than the strongest version of the "attention is not explanation" critique suggests. The debate, in other words, partly comes down to **what counts as a fair comparison** — comparing to genuinely implausible alternative attention patterns overstates the fragility; comparing only to plausible alternatives shows a more nuanced, partially-reassuring picture.

**The synthesis, worth having as your own defensible position:** attention weights are **not a rigorously grounded explanation method** in the way SHAP is (no equivalent to the Shapley axioms exists for attention), and shouldn't be presented as a definitive, unique account of "what the model used" — but they're also not *worthless* as a rough, exploratory signal, especially when cross-checked against a more principled method (Integrated Gradients, or SHAP-style attribution adapted for transformers) rather than trusted alone. **The interview-ready framing:** *"Attention weights are a convenient, free-to-compute signal, but they lack the uniqueness/faithfulness guarantees a method like SHAP or Integrated Gradients has — I'd treat a raw attention map as a hypothesis to cross-check with a more principled method, not as a standalone explanation, especially for anything high-stakes."*

## 5.3 What attention weights can and can't legitimately be used to claim

**What they can reasonably support:** a rough, exploratory hypothesis about which tokens a specific layer/head is relating to which other tokens — useful for debugging (e.g., "is this attention head specializing in syntactic relationships, or something else entirely"), and useful as one input among several when trying to understand model behavior, provided it's not the only input.

**What they cannot legitimately support, given §5.2's evidence:** a claim that attention weights uniquely and faithfully identify "the reason" for a specific prediction, in the strong sense that changing them would necessarily change the output, or that they satisfy any of the fairness/uniqueness properties SHAP's axioms guarantee (Chapter 5 of your Feature Importance syllabus). Treating a single attention map as "the explanation" for a high-stakes NLP model's decision — the way you might treat a well-validated SHAP explanation for a tabular model — overstates what the evidence actually supports.

## 5.4 Concept Activation Vectors (TCAV) — a fundamentally different question

**Why this section belongs in the same chapter as the attention debate:** TCAV sidesteps the entire "does this raw internal quantity (attention weight, activation, gradient) mean what we hope it means" problem by asking a cleanly different, more directly testable question: **"is a specific, human-defined concept represented in this network's internal representations at all, and if so, how much does it influence a given prediction?"** — rather than trying to interpret a raw internal quantity's meaning directly, TCAV tests a specific hypothesis about a named concept.

**The procedure, conceptually:**
1. **Define a concept** you want to test for (e.g., "stripes," for a network that classifies zebra images) by collecting a set of example images that clearly represent that concept (photos of striped objects — could be anything striped, not just zebras) and a contrasting set of random images that don't represent it.
2. **Train a simple linear classifier** on the network's internal activations (at some chosen layer) to distinguish the concept examples from the random/contrasting examples. The direction this linear classifier learns — a vector in the network's internal activation space — is the **Concept Activation Vector (CAV)** for "stripes" at that layer.
3. **Measure how much this concept direction influences a specific prediction** by computing the directional derivative of the network's output with respect to movement along the CAV direction, for a specific input — this tells you "if this input's internal representation moved further in the 'stripes' direction, would the network's predicted probability for 'zebra' increase?"
4. Aggregate this sensitivity measure across many examples of the target class (e.g., many zebra images) to get an overall **TCAV score** — the fraction of examples for which moving toward the "stripes" concept direction would have increased the "zebra" prediction, giving you a global, human-interpretable statement like "the concept of stripes positively influences 87% of this network's zebra predictions at this layer."

**Why this is a meaningfully different and, in some ways, more trustworthy approach than raw saliency or attention:** you're testing a **specific, human-articulated hypothesis** ("does this network use the concept of stripes") rather than trying to reverse-engineer meaning from a raw internal quantity (a gradient map or an attention weight) that might or might not correspond to anything semantically coherent at all. This sidesteps the noise problem from Chapter 4 (§4.2) and the ambiguity problem from this chapter's attention debate (§5.2), at the cost of requiring you to have a specific concept in mind to test in the first place — TCAV can confirm or refute a hypothesis about a named concept, but (unlike a saliency map) it can't spontaneously surface a concept you hadn't thought to test for.

## 5.5 Quick self-check before Chapter 6

- Can you state the core empirical finding behind the "attention is not explanation" critique, and the partial rebuttal to it, in your own words?
- Can you explain why TCAV sidesteps the interpretation-ambiguity problem that raw saliency and attention both face?
- Given a network you suspect relies on a specific, nameable visual concept, could you sketch how you'd test that hypothesis using TCAV?

---

**Next: Chapter 6 — Explainability for Specific Modalities**, covering NLP-specific attribution challenges, the sanity-check literature showing some popular saliency methods don't actually depend on the model's learned weights, and the extra difficulty of attributing importance to temporal patterns in time series.
