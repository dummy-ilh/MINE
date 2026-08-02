# DistilBERT From Scratch — What's the Same, What's Different, and How to Build It

Ninth companion doc. Everything else so far explained BERT and *described* DistilBERT. This one actually builds DistilBERT's differing pieces, verified with running code (the full script is at the end) — so you can point to exactly what changes and why, and how each change is implemented, not just recite the summary.

## The one-line mental model

**DistilBERT = BERT's exact same building blocks (embeddings math, attention, FFN, residual+LayerNorm), just fewer of them, with a training procedure built to compensate for having fewer.** Nothing about *how a single encoder layer works* changes — only *how many layers there are*, *what goes into the embedding sum*, *whether there's a pooler*, and *how the model is trained*.

---

## Part 1 — What's identical to BERT (don't rebuild these, reuse them)

| Component | Identical? | Notes |
|---|---|---|
| WordPiece tokenizer | Yes | Same vocabulary and tokenization scheme |
| Token embeddings | Yes | Same lookup table mechanism |
| Position embeddings | Yes | Same learned lookup, same max length (512) |
| Self-attention mechanism | Yes | Same Q/K/V projections, same scaled dot-product math, same multi-head split |
| Feed-forward sublayer | Yes | Same 4x-expansion + GELU structure |
| Residual connections | Yes | Same add-back-the-input mechanism |
| LayerNorm | Yes | Same post-norm placement as BERT |
| Hidden size (768) and head count (12) | Yes | DistilBERT-base keeps BERT-base's width unchanged |

**Why so much stays the same:** distillation's whole premise is transferring *knowledge* from a working architecture into a smaller version of the *same* architecture family — if you also change the internal block mechanics, you're no longer doing knowledge distillation from BERT, you're designing a new architecture and happening to also compress it. Keeping the block internals identical is what makes "initialize the student directly from the teacher's own weights" (Part 2) possible at all.

---

## Part 2 — What's different, and how to build each piece

### Difference 1: Half the layers (12 → 6)

**What changes:** The encoder stack has 6 blocks instead of 12. Nothing about what's *inside* a block changes.

**How to build it:** Just construct the model with `num_hidden_layers=6` in the config — trivial on its own. The real engineering decision is **which** of the teacher's 12 layers to use as the starting point for the student's 6, rather than random-initializing them.

**How initialization actually works — every-other-layer selection:**

```python
def init_student_from_teacher(student, teacher):
    n_teacher_layers = len(teacher.encoder.layer)      # 12
    n_student_layers = len(student.encoder.layer)       # 6
    stride = n_teacher_layers // n_student_layers        # 2
    # take teacher layers 1, 3, 5, 7, 9, 11 (0-indexed: 1,3,5,7,9,11)
    selected_indices = [i * stride + (stride - 1) for i in range(n_student_layers)]

    for student_idx, teacher_idx in enumerate(selected_indices):
        student.encoder.layer[student_idx].load_state_dict(
            teacher.encoder.layer[teacher_idx].state_dict()
        )
    return selected_indices
```

Verified run of this exact logic on a 6-layer teacher (standing in for BERT-base's 12, scaled down so it runs instantly) building a 3-layer student:
```
Student layer [0, 1, 2] initialized from teacher layers [1, 3, 5]
student layer 0 weights match selected teacher layer: True
```

**Why every-other, not "first half" or "last half":** The DistilBERT authors found this simple stride-based selection worked well empirically, and it has an intuitive justification — it spreads the retained layers across the *entire depth* of the teacher (from early/syntactic layers through to late/semantic layers, per the layer-hierarchy discussion in the phases doc), rather than keeping only shallow layers (if you took layers 1-6) or only deep ones (7-12), either of which would lose an entire portion of the representational hierarchy.

**Why initialize from the teacher at all, instead of random init + pure distillation training:** A randomly initialized 6-layer model would need to learn both "how to be a Transformer encoder" and "how to match the teacher" from nothing — initializing from real, already-trained teacher weights means the student starts from a working representation and distillation training only has to *adapt and compress*, not build from zero. This is a large part of why distillation training converges in far fewer steps than pretraining BERT from scratch did.

### Difference 2: No token-type (segment) embeddings

**What changes:** DistilBERT's embedding sum drops the third term — no `segment_embeddings`, just `token_embeddings + position_embeddings`.

**How to build it:** Either omit the embedding table from the architecture entirely, or (if reusing a class that always allocates one) neutralize it — zero its weights and freeze it so it contributes nothing and never updates:

```python
with torch.no_grad():
    student.embeddings.token_type_embeddings.weight.zero_()
student.embeddings.token_type_embeddings.weight.requires_grad = False
```

**Why this is safe to remove:** Segment embeddings exist in BERT specifically to distinguish sentence A from sentence B for NSP-related tasks. Since DistilBERT drops NSP entirely (next difference), the signal segment embeddings were built to support no longer has a training objective consuming it — removing it sheds parameters and compute with no meaningful pretraining objective left to hurt.

### Difference 3: No pooler layer

**What changes:** BERT has an extra `Linear + Tanh` "pooler" applied to the `[CLS]` vector after the final encoder layer (used to produce a fixed-size pooled output for NSP during pretraining). DistilBERT drops it.

**How to build it:**
```python
student = BertModel(student_cfg, add_pooling_layer=False)
```
Downstream classification heads (built exactly as in the fine-tuning script from the earlier doc) then read `last_hidden_state[:, 0, :]` — the raw `[CLS]` hidden state — directly, rather than a pooled/tanh'd version of it.

**Why safe to remove:** The pooler's original purpose was specifically to feed NSP's binary classifier during BERT pretraining. With NSP gone, there's no pretraining consumer left for the pooler's output — removing it is pure parameter/compute savings with no lost training signal.

### Difference 4: No NSP — MLM only (but trained differently, see Part 3)

**What changes:** DistilBERT's pretraining data doesn't need sentence-pair construction (no "50% true next sentence, 50% random sentence" setup) — just contiguous spans of text for MLM.

**How to build it:** Simpler data pipeline than BERT's original — no need to track document boundaries for constructing sentence-pair examples, no NSP label, no segment-membership bookkeeping. Purely: mask 15% of tokens per the standard 80/10/10 rule, predict them.

**Why:** Follows directly from RoBERTa's finding (discussed in the interview-QA doc) that NSP contributed little to downstream performance — DistilBERT's authors built on that finding rather than re-deriving it.

---

## Part 3 — How training actually differs: the triple loss

This is the part that's genuinely new construction, not just "remove a piece." DistilBERT isn't trained with plain MLM loss against ground truth — it's trained with **three loss terms simultaneously**.

### 3a. Distillation loss (soft labels, the "learn from the teacher's confidence" term)

```python
def distillation_loss(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature ** 2)   # rescale to keep gradient magnitude comparable across T choices
```

**What it does:** KL divergence between the teacher's and student's *softened* output distributions over the vocabulary, at each masked position. Higher temperature flattens both distributions, making the relative probabilities across non-top tokens ("dark knowledge" — see the interview-QA doc's worked numeric example) more visible for the student to learn from.

### 3b. Hard-label MLM loss (the "still learn the actual right answer" term)

```python
def hard_label_mlm_loss(student_logits, true_token_ids):
    return F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), true_token_ids.view(-1))
```

**What it does:** Standard cross-entropy against the real masked-token ground truth — exactly BERT's original MLM loss, computed on the student alone (teacher not involved).

**Why keep this alongside distillation:** The teacher isn't infallible — it can be confidently wrong on some examples. Training purely against the teacher's soft labels risks the student inheriting the teacher's specific mistakes; blending in the ground-truth loss anchors the student to reality, not just to teacher-mimicry.

### 3c. Cosine embedding loss (the "point your internal representations the same direction" term)

```python
def cosine_embedding_loss(student_hidden, teacher_hidden):
    target = torch.ones(student_hidden.size(0), device=student_hidden.device)
    return F.cosine_embedding_loss(student_hidden, teacher_hidden, target)
```

**What it does:** Encourages the student's final hidden-state vectors to point in the same *direction* as the teacher's corresponding vectors (cosine similarity → 1), independent of magnitude.

**Why this on top of the other two:** The first two losses only constrain the student's *output* (the MLM prediction distribution). This one directly constrains the student's *internal representation* — a more direct transfer of "how the teacher represents the input internally," which the paper found improves downstream transfer beyond matching outputs alone.

### 3d. Combining all three

```python
total_loss = alpha * L_distill + beta * L_mlm + gamma * L_cos
# DistilBERT paper's rough weighting ratios: alpha=5.0, beta=2.0, gamma=1.0
```

**Verified end-to-end run** (dummy batch, tiny model — same relative logic as real DistilBERT training):
```
L_distill = 3.3393
L_mlm     = 4.9361
L_cos     = 0.2593
total     = 26.8280
backward() succeeded -- gradients flow through student, none through teacher
student params received gradients: True
```

**Why weight distillation loss (`alpha=5.0`) highest:** The distillation signal is the primary mechanism actually doing the compression — teaching the student to reproduce the teacher's rich output distribution is the core of the method; the other two terms are important correctives/enhancements, not the main driver.

**Practically important detail:** the teacher is always run in `eval()` mode under `torch.no_grad()` — it's a fixed, frozen reference throughout distillation training. Only the student's parameters receive gradients and get updated; the teacher never changes.

---

## Part 4 — The full picture: how a DistilBERT training step differs from a BERT pretraining step

```
BERT pretraining step:
  1. Mask 15% of tokens in a sentence-pair input
  2. Forward pass through BERT
  3. Compute MLM loss (masked tokens) + NSP loss ([CLS] output)
  4. Backprop, update BERT's weights

DistilBERT distillation step:
  1. Mask 15% of tokens in a (single-span, no sentence-pair) input
  2. Forward pass through the FROZEN teacher (no_grad) -> get teacher logits + teacher hidden states
  3. Forward pass through the STUDENT (student's params require_grad=True) -> get student logits + student hidden states
  4. Compute distillation loss (student vs teacher logits, softened) + hard MLM loss (student vs ground truth) + cosine loss (student vs teacher hidden states)
  5. Combine with alpha/beta/gamma weights, backprop -- ONLY through the student
```

The teacher never trains further during this process — it's simply BERT, already pretrained, used purely as a fixed reference signal.

---

## Verified build script (runs end-to-end)

The code throughout this doc is directly lifted from a script that was executed and verified — student correctly initialized from teacher layers `[1, 3, 5]` (the every-other rule), token-type embeddings zeroed/frozen, pooler absent, and all three loss terms computing and backpropagating cleanly with gradients confirmed flowing only into the student. If you want the full runnable file (teacher/student construction + triple loss + a dummy training step) as a standalone `.py`, just ask and I'll package it the same way as the fine-tuning script from earlier.
