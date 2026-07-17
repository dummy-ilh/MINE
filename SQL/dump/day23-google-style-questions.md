# Day 23 — Google-Style SQL Questions
**FAANG SQL 30-Day Prep**

---

## What Google Tests

```
1. Search & ads metrics (CTR, RPM, Quality Score, nDCG)
2. DAU / MAU / WAU stickiness ratios
3. L1 / L7 / L28 active user windows
4. Large-scale aggregations with ROLLUP/partitioning
5. Ranking & scoring systems
6. Edge cases: NULLs, division by zero, sparse data
```

---

## 1. CTR & Quality Score

```sql
-- CTR by position
SELECT position,
  COUNT(*) AS impressions, SUM(was_clicked) AS clicks,
  ROUND(SUM(was_clicked)*100.0/COUNT(*), 4) AS ctr,
  AVG(CASE WHEN was_clicked=1 THEN dwell_time_seconds END) AS avg_dwell
FROM search_impressions GROUP BY position ORDER BY position;

-- Composite quality score per query
WITH qm AS (
  SELECT query_id, COUNT(*) AS impressions,
    SUM(was_clicked)*1.0/COUNT(*) AS ctr,
    AVG(CASE WHEN was_clicked=1 THEN dwell_time_seconds ELSE 0 END) AS avg_dwell,
    SUM(CASE WHEN was_clicked=1 AND dwell_time_seconds>30 THEN 1 ELSE 0 END)*1.0/
      NULLIF(SUM(was_clicked),0) AS long_click_rate
  FROM search_impressions GROUP BY query_id HAVING COUNT(*) >= 100
)
SELECT query_id, impressions,
  ROUND(ctr*40 + LEAST(avg_dwell/120,1)*35 + long_click_rate*25, 2) AS quality_score
FROM qm ORDER BY quality_score DESC;
```

---

## 2. DAU / MAU / WAU Stickiness

```sql
WITH daily   AS (SELECT event_date, COUNT(DISTINCT user_id) AS dau FROM user_events GROUP BY event_date),
     weekly  AS (SELECT DATE_SUB(event_date, INTERVAL DAYOFWEEK(event_date)-1 DAY) AS week_start,
                   COUNT(DISTINCT user_id) AS wau FROM user_events GROUP BY 1),
     monthly AS (SELECT DATE_FORMAT(event_date,'%Y-%m-01') AS month,
                   COUNT(DISTINCT user_id) AS mau FROM user_events GROUP BY 1)
SELECT d.event_date, d.dau, w.wau, m.mau,
  ROUND(d.dau*100.0/NULLIF(m.mau,0),2) AS stickiness_pct,
  ROUND(d.dau*100.0/NULLIF(w.wau,0),2) AS dau_wau_ratio
FROM daily d
JOIN weekly  w ON DATE_SUB(d.event_date, INTERVAL DAYOFWEEK(d.event_date)-1 DAY) = w.week_start
JOIN monthly m ON DATE_FORMAT(d.event_date,'%Y-%m-01') = m.month
ORDER BY d.event_date;
-- Stickiness > 20% = healthy, WhatsApp ~70%
```

---

## 3. L1 / L7 / L28 Windows

```sql
SELECT CURRENT_DATE AS report_date,
  COUNT(DISTINCT CASE WHEN event_date >= CURRENT_DATE - INTERVAL  1 DAY THEN user_id END) AS l1,
  COUNT(DISTINCT CASE WHEN event_date >= CURRENT_DATE - INTERVAL  7 DAY THEN user_id END) AS l7,
  COUNT(DISTINCT CASE WHEN event_date >= CURRENT_DATE - INTERVAL 28 DAY THEN user_id END) AS l28,
  ROUND(COUNT(DISTINCT CASE WHEN event_date >= CURRENT_DATE - INTERVAL 7 DAY
    THEN user_id END) * 100.0 /
    NULLIF(COUNT(DISTINCT CASE WHEN event_date >= CURRENT_DATE - INTERVAL 28 DAY
    THEN user_id END), 0), 2) AS l7_l28_ratio
FROM user_events WHERE event_date >= CURRENT_DATE - INTERVAL 28 DAY;
```

---

## 4. nDCG — Ranking Quality

```sql
-- DCG = Σ relevance / log2(position + 1)
WITH dcg AS (
  SELECT query_id,
    SUM(relevance_score / LOG(2, position + 1)) AS dcg_score
  FROM search_results GROUP BY query_id
),
idcg AS (
  SELECT query_id,
    SUM(relevance_score / LOG(2, ideal_pos + 1)) AS idcg_score
  FROM (
    SELECT query_id, relevance_score,
      ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY relevance_score DESC) AS ideal_pos
    FROM search_results
  ) t GROUP BY query_id
)
SELECT d.query_id, d.dcg_score, i.idcg_score,
  ROUND(d.dcg_score / NULLIF(i.idcg_score, 0), 4) AS ndcg
FROM dcg d JOIN idcg i ON d.query_id = i.query_id
ORDER BY ndcg DESC;
```

---

## 5. Ad Revenue: RPM, CPC, MoM

```sql
WITH stats AS (
  SELECT campaign_id,
    DATE_FORMAT(impression_time,'%Y-%m') AS month,
    COUNT(*) AS impressions, SUM(was_clicked) AS clicks,
    SUM(revenue) AS revenue,
    ROUND(SUM(revenue)*1000.0/COUNT(*), 4)             AS rpm,
    ROUND(SUM(revenue)/NULLIF(SUM(was_clicked),0), 4)  AS cpc
  FROM ad_impressions
  GROUP BY campaign_id, DATE_FORMAT(impression_time,'%Y-%m')
)
SELECT *,
  RANK() OVER (PARTITION BY month ORDER BY rpm DESC) AS rpm_rank,
  ROUND((rpm - LAG(rpm) OVER (PARTITION BY campaign_id ORDER BY month))*100.0 /
    NULLIF(LAG(rpm) OVER (PARTITION BY campaign_id ORDER BY month),0),2) AS rpm_mom_pct
FROM stats ORDER BY month, rpm_rank;
```

---

## 6. User Engagement Segmentation

```sql
WITH stats AS (
  SELECT user_id,
    COUNT(DISTINCT event_date) AS active_days_l28,
    COUNT(*) AS total_events,
    COUNT(DISTINCT event_type) AS feature_diversity
  FROM user_events
  WHERE event_date >= CURRENT_DATE - INTERVAL 28 DAY
  GROUP BY user_id
)
SELECT *,
  CASE
    WHEN active_days_l28 >= 21 THEN 'Power User'
    WHEN active_days_l28 >= 14 THEN 'Regular'
    WHEN active_days_l28 >=  7 THEN 'Casual'
    WHEN active_days_l28 >=  1 THEN 'At Risk'
    ELSE 'Dormant'
  END AS tier,
  ROUND(active_days_l28*3 + feature_diversity*5 +
        LEAST(total_events/10,20), 2) AS engagement_score
FROM stats ORDER BY engagement_score DESC;
```

---

## 7. Funnel Drop-Off

```sql
SELECT DATE(impression_time) AS dt,
  COUNT(DISTINCT user_id) AS searchers,
  COUNT(DISTINCT CASE WHEN was_clicked=1 THEN user_id END) AS clickers,
  COUNT(DISTINCT CASE WHEN was_clicked=1 AND dwell_time_seconds>30 THEN user_id END) AS long_clickers,
  COUNT(DISTINCT CASE WHEN was_clicked=1 AND dwell_time_seconds>30 AND converted=1 THEN user_id END) AS converters,
  ROUND(SUM(was_clicked)*100.0/NULLIF(COUNT(*),0),2) AS search_to_click_pct,
  ROUND(COUNT(DISTINCT CASE WHEN was_clicked=1 AND dwell_time_seconds>30 THEN user_id END)*100.0/
    NULLIF(COUNT(DISTINCT CASE WHEN was_clicked=1 THEN user_id END),0),2) AS click_to_longclick_pct
FROM search_impressions GROUP BY DATE(impression_time) ORDER BY dt;
```

---

## 8. Metric Anomaly Detection

```sql
WITH daily AS (
  SELECT event_date,
    COUNT(DISTINCT user_id) AS dau,
    SUM(revenue) AS revenue
  FROM user_events GROUP BY event_date
),
rolling AS (
  SELECT event_date, dau, revenue,
    AVG(dau) OVER (ORDER BY event_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS avg_dau_7d
  FROM daily
)
SELECT event_date, dau, revenue,
  ROUND((dau - avg_dau_7d)*100.0/NULLIF(avg_dau_7d,0),2) AS vs_rolling_pct,
  CASE WHEN (dau - avg_dau_7d)*100.0/NULLIF(avg_dau_7d,0) < -20
       THEN '🔴 DAU DROP' ELSE '✅ NORMAL' END AS alert
FROM rolling ORDER BY event_date;
```

---

## Practice Questions

### Q1 — Medium ✅
DAU / WAU / MAU + stickiness + rolling 7d avg + alert flag.

```sql
WITH daily AS (
  SELECT event_date, COUNT(DISTINCT user_id) AS dau
  FROM user_events WHERE YEAR(event_date) = 2025 GROUP BY event_date
),
weekly AS (
  SELECT DATE_SUB(event_date, INTERVAL DAYOFWEEK(event_date)-1 DAY) AS week_start,
    COUNT(DISTINCT user_id) AS wau
  FROM user_events WHERE YEAR(event_date) = 2025 GROUP BY 1
),
monthly AS (
  SELECT DATE_FORMAT(event_date,'%Y-%m-01') AS month,
    COUNT(DISTINCT user_id) AS mau
  FROM user_events WHERE YEAR(event_date) = 2025 GROUP BY 1
),
combined AS (
  SELECT d.event_date, d.dau, w.wau, m.mau,
    AVG(d.dau) OVER (ORDER BY d.event_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7d
  FROM daily d
  JOIN weekly  w ON DATE_SUB(d.event_date, INTERVAL DAYOFWEEK(d.event_date)-1 DAY) = w.week_start
  JOIN monthly m ON DATE_FORMAT(d.event_date,'%Y-%m-01') = m.month
)
SELECT event_date, dau, wau, mau,
  ROUND(dau*100.0/NULLIF(mau,0),2) AS stickiness_pct,
  ROUND(dau*100.0/NULLIF(wau,0),2) AS dau_wau_ratio,
  ROUND(rolling_7d, 0) AS rolling_7d_avg,
  ROUND((dau-rolling_7d)*100.0/NULLIF(rolling_7d,0),2) AS vs_rolling_pct,
  CASE WHEN (dau-rolling_7d)*100.0/NULLIF(rolling_7d,0) < -20
       THEN '🔴 DAU DROP' ELSE '✅ NORMAL' END AS alert
FROM combined ORDER BY event_date;
```

### Q2 — Hard ✅
Query quality score with CTR + dwell + long-click rate (50+ impressions).

```sql
WITH qm AS (
  SELECT query_id, COUNT(*) AS impressions,
    ROUND(SUM(was_clicked)*100.0/COUNT(*),4) AS ctr,
    ROUND(AVG(CASE WHEN was_clicked=1 THEN dwell_time_seconds END),2) AS avg_dwell,
    ROUND(SUM(CASE WHEN was_clicked=1 AND dwell_time_seconds>30 THEN 1 ELSE 0 END)*100.0/
      NULLIF(SUM(was_clicked),0),4) AS long_click_rate
  FROM search_impressions GROUP BY query_id HAVING COUNT(*) >= 50
)
SELECT query_id, impressions, ctr, avg_dwell, long_click_rate,
  ROUND((ctr/100)*40 + LEAST(avg_dwell/120.0,1)*35 + (long_click_rate/100)*25, 4) AS quality_score,
  RANK() OVER (ORDER BY ROUND((ctr/100)*40 + LEAST(avg_dwell/120.0,1)*35 +
    (long_click_rate/100)*25,4) DESC) AS quality_rank
FROM qm ORDER BY quality_score DESC;
```

### Q3 — Very Hard ✅
Weekly search health report by platform with WoW alerts.

```sql
WITH weekly_base AS (
  SELECT DATE_SUB(DATE(si.impression_time),
      INTERVAL DAYOFWEEK(DATE(si.impression_time))-1 DAY) AS week_start,
    ue.platform,
    COUNT(*) AS total_impressions,
    COUNT(DISTINCT si.user_id) AS unique_searchers,
    ROUND(SUM(si.was_clicked)*100.0/COUNT(*),4) AS ctr,
    ROUND(AVG(CASE WHEN si.was_clicked=1 THEN si.position END),2) AS avg_clicked_position,
    ROUND(SUM(CASE WHEN si.was_clicked=1 AND si.dwell_time_seconds>30 THEN 1 ELSE 0 END)*100.0/
      NULLIF(SUM(si.was_clicked),0),4) AS long_click_rate
  FROM search_impressions si
  JOIN user_events ue ON si.user_id=ue.user_id AND DATE(si.impression_time)=ue.event_date
  GROUP BY week_start, ue.platform
),
with_trends AS (
  SELECT *,
    AVG(ctr) OVER (PARTITION BY platform ORDER BY week_start
      ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS rolling_4wk_ctr,
    LAG(ctr)            OVER (PARTITION BY platform ORDER BY week_start) AS prev_ctr,
    LAG(long_click_rate) OVER (PARTITION BY platform ORDER BY week_start) AS prev_lcr
  FROM weekly_base
)
SELECT week_start, platform, total_impressions, unique_searchers,
  ctr, avg_clicked_position, long_click_rate,
  ROUND(rolling_4wk_ctr,4) AS rolling_4wk_ctr,
  ROUND((ctr-prev_ctr)*100.0/NULLIF(prev_ctr,0),2) AS ctr_wow_pct,
  ROUND((long_click_rate-prev_lcr)*100.0/NULLIF(prev_lcr,0),2) AS lcr_wow_pct,
  CASE
    WHEN (ctr-prev_ctr)*100.0/NULLIF(prev_ctr,0) < -10 THEN '🔴 CTR DROP'
    WHEN (long_click_rate-prev_lcr)*100.0/NULLIF(prev_lcr,0) < -15 THEN '🔴 LCR DROP'
    ELSE '✅ HEALTHY'
  END AS health_flag
FROM with_trends ORDER BY week_start, platform;
```

---

## Key Takeaways

- **Stickiness** = DAU/MAU — benchmark: >20% healthy, >50% exceptional
- **L7/L28 ratio** → what % of monthly base is weekly active
- **Long-click rate** → proxy for user satisfaction (dwell > 30s)
- **Quality score** → weighted blend of CTR + dwell + satisfaction
- **nDCG** → DCG/IDCG, measures ranking quality (1.0 = perfect)
- **RPM** → revenue × 1000 / impressions — ads efficiency metric
- **Rolling baseline** → always compare vs prior 7-day avg not just yesterday
- **Always NULLIF denominators** → Google expects zero-division safety
- **Platform split** → always break down by mobile vs desktop

---

*Day 23 complete — 7 days to go 🚀*
