# Copilot Prompt — ML Basics Notes (filled for your checklist)

Paste this whole block into Copilot Chat with **Opus 4.8** selected.

---

## PROMPT TO PASTE

You are helping me build interview-prep notes covering the **"ML Basics"** area below. The list
I'm giving you is a **starting point, not exhaustive** — use your judgment to also include other
standard ML-basics topics that naturally belong alongside these (e.g. if it's normally taught or
asked about in the same breath as these, include it, and briefly say what you added and why at
the very top of the doc).

**Subtopics to cover at minimum, in this order:**
1. Bias-variance tradeoff
2. Overfitting vs. underfitting
3. Cross-validation (k-fold, stratified k-fold, LOOCV — compare all three directly)
4. Evaluation metrics: precision, recall, F1, AUC-ROC, AUC-PR (compare AUC-ROC vs AUC-PR directly,
   especially under class imbalance)
5. Class imbalance handling (resampling, class weights, threshold tuning, SMOTE, etc.)
6. Train/test/production skew ("data skew" / train-serving skew — covariate shift, label shift,
   concept drift, and how this differs from the classical bias-variance framing)
7. **[Your call]** — Add any other core ML-basics topics that round this list out (for example,
   regularization, the confusion matrix itself as a standalone building block, calibration,
   or anything else that's conventionally grouped with the above). Don't force it — only add
   what genuinely belongs.

For **each subtopic**, pull from this menu of possible sections. **Not every subtopic needs every
section** — use only the ones that are actually meaningful for that specific subtopic, and skip
the rest rather than padding. For example: a topic like "precision vs. recall" may not have a
"Diagnosis"/"Solutions" pair in the bias-variance sense at all — it may instead need a section on
"how to choose a threshold" or "how to pick which metric matters for the business problem." Use
judgment on which sections fit, don't force the full menu onto everything.

### Menu of possible sections (pick what fits each subtopic)

- **Core Definitions** — Formal setup (notation, what's being estimated/measured). Bold key
  terms, precise math in LaTeX (`$...$` inline, `$$...$$` display) plus one plain-English sentence
  on what each term *measures*. Close with a compressed "cheat line" recap.
- **Core Decomposition / Mechanics** — If there's a formula, derivation, or algorithmic mechanic,
  show it explicitly with the key trick or step spelled out, plus a **"common interview trap"**
  callout where the textbook version breaks down.
- **Intuition** — A comparison table laying out the extremes/options side by side, plus a
  physical/visual analogy where one genuinely exists (don't force an analogy that doesn't fit).
- **Pitfalls** — Numbered interview traps/misconceptions specific to this subtopic (only include
  as many as are real and specific — don't stretch to hit a count). Each: naive belief →
  correction → concrete failure case.
- **Diagnosis** — Only where "which failure mode am I in" is actually a meaningful empirical
  question for this subtopic. How to tell empirically, plus a caveat about an over-trusted
  diagnostic.
- **Solutions / How to Choose** — Only where there's a real decision or fix to make. Could be
  "fix list by failure mode" (bias-variance) or "how to choose between options" (which CV
  strategy, which metric) — whichever framing actually fits the subtopic.
- **Worked Numerical Example** — Include wherever a hand-computable example adds real value (a
  confusion matrix and the metrics computed from it, a toy bias-variance decomposition, a k-fold
  vs LOOCV comparison on a tiny dataset). Flag real/derivable numbers vs. illustrative ones.
- **Conceptual Q&A (clever/trick questions)** — Include for every subtopic, but scale the count to
  how much genuine trick-question territory the topic has (a dense topic like bias-variance
  might get 10-12; a narrower one might only have 4-5 real ones — don't pad with filler
  questions).
- **Extended Reference Table** — Only where there's a natural set of named methods/models/metrics
  to map against this subtopic's core axis. Add a **"Correction flag"** for imprecise but commonly
  repeated claims.
- **Framing / Debate** — Only where there's a genuinely contested "it depends" question — not
  every subtopic has one. Skip rather than manufacture a fake debate.

Whatever sections you use, close every subtopic with one quotable "biggest misconception about
X" closing line — that one's not optional.

### Cross-cutting tone rules (apply to every subtopic)

- Dense, compressed interview-notes tone — no filler, no "in conclusion."
- All math in LaTeX, never left as prose only.
- Be self-critical: flag imprecise or debated claims explicitly rather than stating them as
  settled fact. Never invent a specific statistic and present it as verified — mark illustrative
  numbers as illustrative.
- Prefer concrete named examples (specific metrics, papers, phenomena) over vague generalities.
- Bold key terms on first use; use tables for every comparison instead of prose.
- End each subtopic with one quotable "biggest misconception about X" closing line.
- Explicitly cross-link subtopics where they interact — e.g. note where class imbalance changes
  which cross-validation strategy you should use (stratified), or where AUC-PR vs AUC-ROC choice
  connects back to the imbalance section, or where "production skew" is often misdiagnosed as a
  bias-variance problem when it's actually a distribution-shift problem.

### Output format

Return the whole thing as a single Markdown document. Use `#` for the overall title, `##` for
each subtopic name, and `###` for whichever sections you chose to include for that subtopic
(unnumbered is fine — just use the section names from the menu above). Ready to paste directly
into a `.md` notes file.

---

## Notes on using this in VS Code

- Open Copilot Chat, select **Opus 4.8**, paste the block above as one message.
- This is a long ask (6 subtopics × 10 sections) — if the response gets truncated or thins out
  toward the end, follow up with: "Continue from [subtopic name], same depth as before, don't
  skip sections 7 or 8."
- Save as `notes/ml-basics.md`, or ask Copilot in the same chat to "save this as a new file at
  notes/ml-basics.md."
- Want a subtopic split into its own file later (e.g. just cross-validation)? Reuse the single-
  topic template from `ml-notes-copilot-prompt.md` with `{TOPIC}` = that subtopic — same
  structure, one file per concept instead of one big file.
