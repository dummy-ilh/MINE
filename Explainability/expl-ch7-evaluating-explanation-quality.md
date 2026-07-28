# Chapter 7 — Evaluating Explanation Quality

Chapters 2–6 covered a wide range of methods for producing explanations. This chapter asks the question that ties directly back to Chapter 6's sanity-check result: **how do you actually know whether an explanation is good**, rather than just plausible-looking? This applies equally to interpretable-by-design models and post-hoc methods, and to every modality covered so far.

## 7.1 Why "it looks reasonable to me" is not evaluation

**The core problem with visual/subjective assessment alone:** a human looking at a saliency map, a SHAP summary plot, or a counterfactual explanation will very often find it *plausible* — it tells a coherent-sounding story, it highlights things a person's own intuition would also flag as relevant. But Chapter 6's sanity-check finding (§6.2) demonstrated directly that a method can produce visually plausible output that has **nothing to do with what the model actually learned** — plausibility to a human reviewer and genuine correspondence to the model's actual computation are simply different properties, and an explanation can score well on the first while failing badly on the second. This is precisely why this topic needs **quantitative, falsifiable evaluation criteria**, not just "does this look right."

## 7.2 Faithfulness: does the explanation reflect what the model is actually doing?

**The core question faithfulness evaluation asks:** if an explanation method says "these features/pixels/tokens are the important ones," does the model's behavior actually change in the way that claim predicts when you manipulate those specific elements?

**Deletion metrics.** Take the features/pixels the explanation ranked as most important, and progressively **remove or mask them** (replace with a baseline value, a blur, a zero), then track how the model's prediction changes. **A faithful explanation should show the prediction degrading quickly** as the truly important elements are removed first — if removing the "most important" elements (per the explanation) barely changes the prediction at all, while removing supposedly "unimportant" elements causes a large change, that's direct, quantitative evidence the explanation is not faithful to the model's actual behavior.

**Insertion metrics.** The complementary test: start from a fully-masked/baseline input (no information at all) and progressively **add back** the features/pixels in the order the explanation ranked them, most important first, tracking how quickly the prediction recovers toward its original value. **A faithful explanation should show the prediction recovering quickly** as the truly important elements are added back first — slow recovery suggests the explanation's ranking doesn't actually correspond to what the model needs to make its prediction.

**Why both directions matter, not just one:** deletion and insertion test subtly different things — deletion asks "does removing what's flagged as important hurt performance the way it should," while insertion asks "is what's flagged as important actually *sufficient* to recover the prediction on its own." A method could pass one test while failing the other (e.g., an explanation might correctly identify elements whose *removal* hurts performance, while still missing that a much smaller different set would be *sufficient* to reconstruct the prediction) — running both gives a more complete faithfulness picture than either alone.

**Connecting this back to the sanity-check result (Chapter 6, §6.2):** the weight-randomization sanity check is really a special, extreme case of faithfulness testing — instead of manipulating the input, it manipulates the model itself (randomizing weights) and checks whether the explanation changes appropriately. Deletion/insertion metrics and the randomization sanity check are two different tools answering the same underlying faithfulness question from different angles.

## 7.3 Robustness/stability: does a meaningless perturbation wreck the explanation?

**The core question:** if you make a tiny, semantically meaningless change to the input (adding a small amount of imperceptible noise to an image, say), does the explanation change drastically, even though the model's actual prediction barely changes at all?

**Why instability here is a red flag:** a genuinely faithful, well-grounded explanation method should be reasonably robust to small, meaningless input perturbations — if the model's prediction is essentially unchanged, a trustworthy explanation of "why" should also be essentially unchanged, since nothing meaningful about the input actually changed. An explanation method whose output swings wildly under an imperceptible perturbation is telling you that its output depends heavily on some fragile, high-frequency detail of the exact input rather than on the model's genuine, stable reasoning — this is closely related to the noise problem flagged for vanilla gradient saliency in Chapter 4 (§4.2), and Integrated Gradients' path-averaging was specifically motivated as a partial fix for exactly this kind of instability.

**How this is measured in practice:** generate several small, random perturbations of the same input (small enough that the model's prediction is essentially unchanged), compute the explanation for each perturbed version, and measure how much the explanations vary across these near-identical inputs — high variance here is a direct, quantitative stability failure, independent of whether any single explanation "looks reasonable" in isolation.

## 7.4 Human-grounded evaluation: does the explanation actually help a person?

**Why faithfulness and robustness alone aren't the whole story:** an explanation can be perfectly faithful (it genuinely reflects the model's computation, verified via deletion/insertion and stability testing) and still be **useless to the actual human** who needs to use it — too technical, too cluttered, or simply not aligned with the kind of information that person actually needs to make their decision. Evaluating whether an explanation genuinely helps a real person requires a different kind of test entirely: a **user study**, not just a computational metric.

**Common human-grounded evaluation designs:**
- **Simulatability / forward prediction:** show a person the model's explanation for several examples, then ask them to **predict what the model will output** on a new, unseen example — if people who see the explanation predict the model's behavior more accurately than people who don't, that's direct evidence the explanation genuinely conveys something true and useful about the model's behavior, rather than just feeling satisfying.
- **Decision-support studies:** in a task where a person makes a decision assisted by the model (e.g., a doctor reviewing a diagnostic model's output), compare decision quality (accuracy, appropriate trust/distrust in the model's suggestion) with vs. without the explanation provided — a good explanation should measurably improve decision quality, not just make the person feel more confident regardless of whether that confidence is warranted.
- **Trust calibration:** measure whether the explanation helps people trust the model *appropriately* — increasing trust when the model is actually correct, and appropriately decreasing trust when the model is wrong or the input is out-of-distribution — rather than uniformly increasing trust regardless of whether the underlying prediction is actually reliable. An explanation that makes people trust the model *more* on average, without improving their ability to distinguish reliable from unreliable predictions, has arguably made things worse, not better, since misplaced confidence is often more costly than appropriate skepticism.

**The interview-ready synthesis, pulling §7.2–7.4 together:** *"I'd evaluate an explanation on three distinct axes: faithfulness (deletion/insertion metrics, and ideally a randomization sanity check), robustness (does a meaningless perturbation wreck it), and human utility (does it actually improve a real person's ability to predict or appropriately trust the model, tested via a user study) — a method can pass some of these and fail others, and 'looks reasonable to me' isn't a substitute for any of them."*

## 7.5 Quick self-check before Chapter 8

- Can you explain the difference between a deletion metric and an insertion metric, and why running both gives a more complete picture than either alone?
- Can you connect the weight-randomization sanity check from Chapter 6 to the general faithfulness-testing framework in this chapter — how are they the same kind of test, applied differently?
- Can you describe a human-grounded evaluation design (simulatability, decision-support, or trust calibration) and explain what specific failure mode it's designed to catch that a purely computational faithfulness metric wouldn't catch?

---

**Next: Chapter 8 — Explainability in Practice: Stakeholders and Regulation**, covering the different explanation needs of different audiences, GDPR's "right to explanation" at a high level, the connection back to Model Cards and Datasheets from your Fairness & Responsible AI prep, and explainability as an internal debugging tool distinct from external-facing explanation.
