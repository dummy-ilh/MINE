
1. **Funnel conversion** — signup → activation → purchase, step-by-step drop-off, usually via conditional aggregation or self-joins on a session/user timeline.
2. **Gaps and islands** — find consecutive streaks (active days, login streaks) or find gaps (missing dates, churn windows).
3. **Cohort retention** — group users by signup week/month, track what % return in week N.
4. **Top-N per group** — top 3 products per category, most recent record per user.
5. **Running/rolling metrics** — cumulative sum, moving average, rolling 7-day active users.
6. **Period-over-period comparison** — day-over-day, week-over-week deltas using LAG/LEAD.
9. **Self-join for pairwise comparisons** — e.g., users who bought A and B, friend recommendations, A/B pairing.
10. **Histogram/bucketing** — binning continuous values (age, revenue) into ranges then aggregating.
11. **Pivoting / unpivoting** — long-to-wide or wide-to-long transforms, often with conditional `SUM(CASE WHEN...)`.
12. **Set logic on user segments** — users in A but not B, overlap between two behaviors.
13. **Percentile / distribution questions** — median, p90 latency, using `PERCENTILE_CONT` or NTILE.
14. **Hierarchical/recursive queries** — org charts, category trees, recursive CTEs.
Got it — here's the pattern taxonomy, the shapes interviewers reuse over and over:

1. **Funnel conversion** — signup → activation → purchase, step-by-step drop-off, usually via conditional aggregation or self-joins on a session/user timeline.
2. **Gaps and islands** — find consecutive streaks (active days, login streaks) or find gaps (missing dates, churn windows).
3. **Cohort retention** — group users by signup week/month, track what % return in week N.
4. **Top-N per group** — top 3 products per category, most recent record per user.
5. **Running/rolling metrics** — cumulative sum, moving average, rolling 7-day active users.
6. **Period-over-period comparison** — day-over-day, week-over-week deltas using LAG/LEAD.
7. **Deduplication** — keep first/last/most-complete record per key.
8. **Sessionization** — group events into sessions based on time gaps between them.
9. **Self-join for pairwise comparisons** — e.g., users who bought A and B, friend recommendations, A/B pairing.
10. **Histogram/bucketing** — binning continuous values (age, revenue) into ranges then aggregating.
11. **Pivoting / unpivoting** — long-to-wide or wide-to-long transforms, often with conditional `SUM(CASE WHEN...)`.
12. **Set logic on user segments** — users in A but not B, overlap between two behaviors.
13. **Percentile / distribution questions** — median, p90 latency, using `PERCENTILE_CONT` or NTILE.
14. **Hierarchical/recursive queries** — org charts, category trees, recursive CTEs.

Funnel, cohort retention, and gaps-and-islands are the three DS-specific ones Google/Apple lean on hardest since they map directly to product metrics. Want me to build out funnel and cohort retention as dedicated deep-dive files the way you did for the others, or do you already have those covered in your existing 46-topic set?

