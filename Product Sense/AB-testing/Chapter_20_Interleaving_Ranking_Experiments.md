# Chapter 20 (continued): Interleaving Methods — the Why, the Variants, and the Numbers

> This extends the chapter's core idea — same user, blended rankings, within-subject comparison — with the part that usually gets compressed to one sentence: *which* interleaving method, why it beats a plain A/B test on the actual numbers, and where each variant quietly breaks.

---

## 1. Why now — what problem is actually being solved

A standard A/B test on a ranking algorithm asks: *"do users shown ranking A click more than users shown ranking B?"* That's a **between-subjects** question, and it's a statistically expensive one to ask, because the biggest source of variance in the data has nothing to do with ranking quality — it's that some people click on five things a session and some people click on nothing, regardless of what's on the page. The real signal (A vs. B) is a small wobble on top of that huge person-to-person noise floor.

Interleaving reframes the question as: *"within this one person's one session, seeing both rankings blended together, did they click more of A's picks or more of B's picks?"* That's a **within-subject** question — every source of "this particular user just isn't a clicker" cancels out automatically, because the same user is the baseline for both algorithms simultaneously. This is the *same* underlying move as a paired t-test or CUPED: stop comparing across people, start comparing within them.

## 2. Interleaving variants — comparison table

The chapter names team-draft interleaving as the default answer, but there are meaningfully different variants with different failure modes:

| Method | How it blends | What it protects against | Why it won't work / limitations |
|---|---|---|---|
| **Balanced interleaving** | Alternate strictly A, B, A, B (or B, A, B, A on 50% of queries) at fixed positions | Simple, easy to implement and explain | Biased toward whichever algorithm ranks a shared item slightly higher when both lists agree — ties get resolved in a way that isn't neutral, and it handles duplicate top-picks between A and B poorly |
| **Team-draft interleaving** | A and B "take turns" drafting their next-favorite unpicked item; draft order randomized per query | Removes position bias (whoever picks first gets top slots) by randomizing which algorithm goes first | Still assumes a click on an item is a vote for the algorithm that contributed it — doesn't model *how* a user's attention decays down a list beyond simple turn-taking |
| **Probabilistic interleaving** | Each algorithm's items are drawn into the blended list with probability proportional to their rank (softmax-like), not deterministic turn-taking | More statistically robust bias correction than team-draft; better handles algorithms with very similar top results | Meaningfully more complex to implement and explain — the added robustness usually isn't worth it unless team-draft is showing clear bias in your specific setting |

**Practical takeaway:** team-draft is the right default to name in an interview and in most production systems — it's the simplest method that correctly handles position bias. Reach for probabilistic interleaving only when you have evidence team-draft's simpler correction isn't enough (e.g., algorithms whose top results overlap heavily).

## 3. Extending the worked example — how sensitive is this, really?

Original result: 3,800 vs. 3,200 wins out of 7,000 non-tied comparisons → 54.3% win rate for B, z ≈ 7.19, overwhelmingly significant.

**What if the effect were smaller — a 52% win rate on the same 7,000 comparisons?**

$$SE = \sqrt{\frac{0.5 \times 0.5}{7000}} \approx 0.00598, \quad z = \frac{0.52 - 0.50}{0.00598} \approx 3.35$$

z ≈ 3.35 corresponds to p ≈ 0.0004 — **still clearly significant**, even at a win rate barely above a coin flip. This is the number worth internalizing: interleaving's within-subject variance reduction is powerful enough that even a 2-percentage-point preference edge is detectable at moderate query volume, something a between-subjects business-metric A/B test at the same sample size would very likely miss entirely.

| Win rate (of 7,000 non-tied) | z-score | Significant at α=0.05? |
|---|---|---|
| 51% | 1.67 | No (borderline, p ≈ 0.095) |
| 52% | 3.35 | Yes (p ≈ 0.0004) |
| 54.3% | 7.19 | Yes, overwhelmingly |

Note the sharp jump between 51% and 52% — this is just the mechanics of a binomial test at this sample size, not a special property of interleaving, but it's a good intuition check: interleaving doesn't make *tiny* differences detectable for free, it just makes moderate differences detectable at far smaller sample sizes than a between-subjects design would need.

## 4. Why interleaving still isn't the final word

| Question interleaving answers | Question interleaving does NOT answer |
|---|---|
| Which ranking do users click more of, right now, in this session? | Which ranking makes users more satisfied, more retained, more valuable long-term? |
| Is there a genuine relative preference between A and B? | Is that preference driven by real quality, or by clickbait-y surface appeal? |
| Fast, cheap screen: is B even in the running? | Final call: should B actually ship, measured against the true OEC? |

This is why the standard production pattern is **interleaving as a fast filter → standard A/B test as the final gate** — not either/or, but a funnel: interleaving cheaply eliminates clearly-worse candidates using far fewer queries, and only survivors go through the slower, more expensive, OEC-validated A/B test.

## 5. Interview-ready summary

- **Name the technique unprompted** the moment a ranking/search comparison shows up — don't default to "just A/B test it."
- **Explain the mechanism**, not just the name: within-subject comparison removes between-user variance, same principle as paired tests/CUPED.
- **Name team-draft specifically**, and say *why* the randomized draft order matters (position bias).
- **Position it correctly**: fast relative-preference filter, not a replacement for OEC-based validation — clickbait-gaming risk is exactly why the final A/B test stays mandatory.
- **Be ready to run the numbers live** — a binomial/sign test on win-rate-vs-50% is simple enough to compute on a whiteboard, and doing so is a strong differentiator over candidates who only describe the idea qualitatively.

## 6. Comprehension check

1. A colleague proposes balanced (strict alternating) interleaving instead of team-draft to save engineering time. What specific bias are they reintroducing, and when would it matter most?
2. At what approximate win rate does the 7,000-comparison example stop being statistically significant at α = 0.05? (Work from the table in Section 3.)
3. Why does probabilistic interleaving exist at all if team-draft already fixes position bias? What extra failure mode is it addressing?
4. An algorithm wins its interleaving comparison decisively but *loses* on the follow-up business-metric A/B test. What's the most likely explanation, and which chapter concept does it connect back to?
5. Explain, without using the word "variance," why the same user seeing both rankings makes the comparison so much more sensitive than showing different users different rankings.
