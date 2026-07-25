# Chapter 1: Problem Framing in Recommendation Systems

## 1. Intuition

Every recsys problem starts with one question: **what signal do you actually have, and what are you trying to predict from it?**

Two axes define the problem space:

- **Feedback type**: did the user *tell you* they liked something (explicit), or did you *infer* it from behavior (implicit)?
- **Task type**: are you predicting a *score* (rating prediction) or a *ranked list* (ranking)?

Get this framing wrong and everything downstream — model choice, loss function, evaluation metric — is wrong too. This is why interviewers open recsys questions here: it filters out candidates who jump straight to "I'd use matrix factorization" without asking what data actually exists.

## 2. Explicit vs Implicit Feedback

**Explicit feedback**: user directly states preference.
- Examples: 5-star ratings, thumbs up/down, explicit "not interested"
- Pros: unambiguous signal, direct measure of preference
- Cons: sparse (users rarely rate), biased (people rate things they feel strongly about — the "J-shaped distribution" of ratings), and expensive to collect

**Implicit feedback**: preference inferred from behavior.
- Examples: clicks, watch time, purchases, dwell time, skips
- Pros: abundant, free, reflects real behavior not stated preference
- Cons: ambiguous (a click doesn't mean "liked," a non-click doesn't mean "disliked" — could just mean "never saw it"). No negative signal, only absence of positive signal.

**The core technical consequence**: with explicit feedback you have real 1-5 star labels and can do regression. With implicit feedback you only have positive observations — this is the **one-class problem**. You cannot treat unclicked items as negatives outright, because "unclicked" conflates "shown and rejected" with "never shown" and "shown but not noticed." This single fact is why implicit-feedback models (Ch. 6, Hu-Koren-Volinsky) need a fundamentally different loss than explicit rating prediction (plain SVD, Ch. 5).

Most production systems today (YouTube, Google Play, ads) are implicit-feedback systems, because explicit ratings don't scale and are gameable.

## 3. Rating Prediction vs Ranking

**Rating prediction**: predict $\hat{r}_{ui}$, a real-valued score for user $u$, item $i$. Evaluated with RMSE/MAE. This was the Netflix Prize framing (2006-2009) and it's largely a **historical relic** in production — nobody at Google ships a system whose end goal is "predict the star rating."

**Ranking**: produce an ordered list of items for a user, optimizing for the *order* being correct at the top, not the absolute score being calibrated. Evaluated with NDCG, MAP, Precision@K (Ch. 2).

**Why this distinction matters for interviews**: a model can have excellent RMSE on rating prediction and terrible ranking quality. If a model predicts 4.9, 4.8, 4.7 for items whose true ratings are 5, 4, 3 — RMSE is tiny, and the ranking is perfect. But if it predicts 3.0, 3.1, 3.2 for true ratings 5, 4, 3, RMSE might be similar in magnitude yet the ranking is **completely inverted**. Since production systems show a ranked list (10 videos, 20 search results), ranking quality is what the business cares about, not calibration.

This is why almost the entire industry moved from "rating prediction" (SVD/RMSE-style) to "learning-to-rank" (Module 3) as implicit feedback became the norm — you don't need calibrated scores, you need correct order.

## 4. The Cold-Start Taxonomy

Cold-start = you have a user, item, or system with **no or minimal interaction history**. There are three distinct sub-problems, and interviewers expect you to name all three, not just say "cold start is hard":

| Type | Problem | Typical mitigation |
|---|---|---|
| **User cold-start** | New user, no interaction history | Onboarding surveys, demographic/contextual features, popularity-based defaults, fast exploration (bandits) |
| **Item cold-start** | New item, no interactions yet | Content-based features (text/image/metadata embeddings) to place item in embedding space without behavioral data |
| **System cold-start** | Brand new product, no data at all | Rules-based or content-based system until enough interaction data accumulates to train collaborative models |

Note the pattern: collaborative filtering (Module 2) is powerless in all three cases because it relies on interaction history. The universal fix is falling back to **content-based signals** (item metadata, user profile features) until enough behavioral data exists — this is also why most production systems are *hybrid* (CF + content), not pure CF.

## 5. Worked Example — Why Loss Function Choice Follows Directly From Framing

Say you have this raw data for one user:

| Item | Rating given | Clicked? | Watch time (sec) |
|---|---|---|---|
| A | 5 | Yes | 600 |
| B | — | Yes | 30 |
| C | — | No | 0 |
| D | 2 | Yes | 45 |

- If you're doing **explicit rating prediction**: you only have 2 usable labels (A=5, D=2). Items B, C are unusable for this framing — massive data loss. This is why explicit-only systems starve.
- If you're doing **implicit ranking**: you can use all 4 rows. A and B are positives (clicked), weighted by confidence (watch time: A is a much stronger positive than B). C is an unobserved item, not a confirmed negative — in BPR-style setups (Ch. 9) you'd sample it as an implicit negative *relative to A*, not as an absolute "user hates C" label. D is interesting: explicit signal says "disliked" (rating=2) but implicit signal shows a click — real systems often trust implicit behavioral signal more than explicit self-report here, since users rate inconsistently but watch time doesn't lie as much.

This single example is why L5 candidates are expected to say "it depends what feedback we have" before naming any model.

## 6. Production Considerations

- Google/YouTube-scale systems are **implicit, ranking-first**. Explicit signals (thumbs up/down) exist but are used as auxiliary features, not the primary training label.
- Position bias is baked in from day one: an item ranked #1 gets clicked more *because* it's #1, not necessarily because it's more relevant. Naively training on implicit clicks without correcting for this creates a feedback loop (rich-get-richer). This foreshadows Module 6 (counterfactual evaluation).
- Real systems blend multiple implicit signals (click, dwell time, share, save) into a single composite label — this is itself a modeling decision, often done via multi-task learning (foreshadows Ch. 25).

## 7. Interview Traps

- Jumping to "I'll use matrix factorization" before asking whether feedback is explicit or implicit — this is the single most common way candidates lose points in the first 2 minutes.
- Treating unclicked items as confirmed negatives in an implicit setting without acknowledging the exposure bias problem.
- Conflating rating prediction accuracy (RMSE) with ranking quality — assuming low RMSE implies good ranking.
- Failing to distinguish the three cold-start types and giving one generic "add more features" answer for all of them.

## 8. L5-Differentiating Talking Points

- Explicitly state that **most modern industrial recsys are implicit + ranking-based**, and explain *why* (scale, no rating friction, better reflects true behavior) — this signals you know where the field actually is, not just textbook Netflix-Prize framing.
- Bring up **position bias** unprompted when discussing implicit feedback — this is a classic signal of production experience vs. academic-only knowledge.
- Note that cold-start mitigation is fundamentally about **falling back to content-based signals** when collaborative signals don't exist yet — ties Module 1 directly into why hybrid systems are the industry default.
- Mention that the choice of implicit signal itself (click vs. dwell time vs. completion rate) is a modeling decision with real trade-offs (e.g., clickbait thumbnails inflate clicks but tank watch time) — this is the kind of "you've actually shipped this" detail that separates L5 from L4 answers.

## 9. Comprehension Check

1. Why can't you simply treat every non-interacted item as a negative label in an implicit-feedback dataset?
2. Give an example where RMSE is low but ranking quality (NDCG) is poor.
3. Name the three types of cold-start and one mitigation for each.
4. Why do most large-scale production recsys (YouTube, Google Play) use implicit feedback rather than explicit ratings as the primary training signal?
5. In the worked example (Section 5), why might you trust the implicit signal (click) over the explicit signal (low rating) for item D?
