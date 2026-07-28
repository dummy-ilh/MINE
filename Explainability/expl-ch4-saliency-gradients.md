# Chapter 4 — Explaining Deep Learning Models: Saliency & Gradient-Based Methods

Chapter 3 covered counterfactuals, which work for any model type. Starting here, Chapters 4–6 are specifically about explaining models whose input isn't naturally tabular — images, text, and other high-dimensional data where a plain feature-attribution table (per-column SHAP values, say) doesn't map cleanly onto "which pixels" or "which words" mattered. This chapter starts with the simplest and most historically important family: gradient-based saliency methods.

## 4.1 The basic idea: use the gradient

**Setup:** you have a deep network f, an input x (say, an image), and an output f(x) (say, the predicted probability of "cat"). You want to know which parts of x mattered most for that prediction.

**The simplest possible approach — vanilla gradient saliency:** compute the gradient of the output with respect to the input, ∂f(x)/∂x. This gives you one number per input pixel (or per input feature, more generally), representing **how much a tiny change in that pixel's value would change the output** — a large gradient magnitude at a pixel means the model's prediction is highly sensitive to that pixel right now; a near-zero gradient means the prediction barely depends on it. Visualizing this gradient as a heatmap over the image — the "saliency map" — highlights the regions the model appears to be most sensitive to.

**Why this is a natural starting point, connecting back to material you already know:** this is exactly the same object as the gradient used in backpropagation during training (your optimization prep) — you're not learning a new mathematical operation, just applying the gradient computation you already understand to the *input* instead of to the model's *weights*, and interpreting the result as an explanation rather than as a training signal.

## 4.2 Known problems with raw saliency

**Problem 1 — Noise.** Raw gradients tend to be visually noisy — the resulting saliency map often highlights scattered, seemingly-random pixels rather than a clean, semantically coherent region (e.g., you might expect a "cat" saliency map to cleanly highlight the cat's outline, but instead get a speckled, hard-to-interpret pattern spread across much of the image). This happens because a deep network's loss landscape (with respect to the input) can be quite jagged/non-smooth at the local, pixel level, even though the network's overall predictions are stable — the gradient captures this local jaggedness, which doesn't correspond to anything semantically meaningful.

**Problem 2 — Vulnerability to adversarial manipulation of the explanation itself, not just the prediction.** This is a subtler and more concerning issue: it's been shown that you can make small, visually imperceptible changes to an input that leave the model's **prediction** completely unchanged, while drastically changing the resulting **saliency map** — meaning an adversary (or even just unlucky circumstances) could make a model's explanation say almost anything, without the underlying prediction changing at all. This directly echoes Chapter 1's concern about post-hoc explanations of black boxes being potentially misleading — raw saliency is one of the more fragile methods on that front, precisely because it depends on a raw, unregularized gradient computation with no smoothing or principled averaging built in.

## 4.3 Integrated Gradients — a more principled fix

**The core idea:** rather than looking at the gradient at a single point (the actual input x), **integrate the gradient along a straight-line path from a baseline input (typically an all-black image, or an all-zero input) to the actual input x**, accumulating the gradient's contribution at every point along that path rather than just the endpoint.

**The formula, in words:** Integrated Gradients for feature i is (x_i − baseline_i) times the average gradient of the output with respect to feature i, computed at many points along the straight-line path from the baseline to x, then averaged (in the limit, this average becomes a genuine integral, hence the name).

**Why integrating along a path fixes the noise problem:** averaging the gradient over many points along the path smooths out the local jaggedness that makes a single-point gradient noisy — a pixel that only *appears* important due to a sharp, local kink in the loss landscape at the exact input x will have that spurious sensitivity washed out once you average over the whole path from baseline to x, while a pixel that's genuinely, consistently important across the path retains a strong signal.

**Why this is a Shapley-adjacent idea, worth connecting explicitly to material you already know:** Integrated Gradients satisfies an axiom directly analogous to SHAP's efficiency axiom (Chapter 5 of your Feature Importance syllabus) — called **completeness** here — the sum of all features' Integrated Gradients attributions exactly equals f(x) − f(baseline), the total difference between the actual prediction and the baseline prediction. This isn't a coincidence: Integrated Gradients can be understood as a continuous, gradient-based analogue of the same "attribute the total difference from a baseline, fairly, across all features" idea that Shapley values formalize discretely for tabular features — the same underlying fairness intuition, adapted to a setting where you can take derivatives directly (a deep network) rather than needing to enumerate discrete feature subsets.

## 4.4 Grad-CAM — a coarser, class-specific approach for convolutional networks

**The idea, specific to convolutional neural networks (CNNs):** rather than computing pixel-level gradients directly on the raw input, Grad-CAM ("Gradient-weighted Class Activation Mapping") looks at the **activation maps of a late convolutional layer** — the feature maps a CNN produces just before its final classification layers, which still retain spatial structure (roughly corresponding to regions of the original image) but represent higher-level, more semantically meaningful features than raw pixels do.

**The procedure, conceptually:**
1. Pick the convolutional layer to explain from (typically the last one before the network flattens spatial structure into a final prediction).
2. Compute the gradient of the target class's output score with respect to each activation map (each "channel") in that layer — this tells you how much each channel matters for predicting this specific class.
3. Use these gradients to compute a weighted average of the activation maps, weighting each channel by how important the gradient computation says it is for the target class.
4. The result is a coarse, low-resolution heatmap (since it's derived from a late, spatially-downsampled layer) that's then upsampled back to the original image's resolution for visualization, highlighting the broad regions the network relied on for this specific class prediction.

**Why "class-specific" matters, and how it differs from vanilla saliency:** because Grad-CAM starts from the gradient of one *particular* class's score, you get a different heatmap depending on which class you're explaining — e.g., an image containing both a cat and a dog would produce a Grad-CAM heatmap highlighting the cat region when explaining the "cat" prediction, and a heatmap highlighting the dog region when explaining the "dog" prediction, even though it's the exact same image and exact same trained network. This class-conditional property is a genuinely useful feature Grad-CAM has that a plain, single vanilla-gradient saliency map doesn't naturally provide.

**Why Grad-CAM tends to look visually cleaner than raw saliency, despite being coarser:** because it operates on a later convolutional layer's activations (which already represent higher-level, more spatially-coherent features than raw pixels — a late layer's activations correspond to whole object parts or textures, not individual pixel intensities), the resulting heatmap is naturally smoother and more semantically interpretable, at the direct cost of spatial resolution (it can point to "this general region" but can't point to individual pixels the way a saliency map technically can, even if that pixel-level precision was mostly noise anyway per §4.2).

## 4.5 Quick self-check before Chapter 5

- Can you explain, mechanistically, why integrating the gradient along a path from a baseline to the actual input reduces the noise problem that vanilla gradient saliency has?
- Can you state the completeness axiom for Integrated Gradients and explain its resemblance to SHAP's efficiency axiom?
- Given an image with two objects, can you explain why Grad-CAM would produce two different heatmaps depending on which class you ask it to explain, while a plain saliency map computed against the whole output vector would not naturally do this?

---

**Next: Chapter 5 — Attention as Explanation, and Its Limits**, covering the intuitive appeal of transformer attention weights as an explanation, the "attention is not explanation" debate and its supporting evidence, and Concept Activation Vectors (TCAV) as a fundamentally different, concept-driven alternative.
