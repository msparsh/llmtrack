-- llmtrack analytics queries

-- NAME: cost_per_day
-- Summary of LLM usage and total cost grouped by day and model.
SELECT date(ts_utc) AS day,
       COALESCE(model, 'unknown') AS model,
       COUNT(*) AS calls,
       SUM(input_tokens) AS in_tok,
       SUM(output_tokens) AS out_tok,
       ROUND(SUM(cost_usd), 4) AS cost_usd
FROM llm_calls
WHERE (? IS NULL OR ts_utc >= ?)
GROUP BY day, model
ORDER BY day DESC, cost_usd DESC;

-- NAME: p95_latency
-- Nearest-rank 95th percentile latency (ms) per model for successful calls.
WITH ranked AS (
  SELECT model, latency_ms,
         ROW_NUMBER() OVER (PARTITION BY model ORDER BY latency_ms) AS rn,
         COUNT(*) OVER (PARTITION BY model) AS n
  FROM llm_calls
  WHERE status = 'ok'
    AND (? IS NULL OR ts_utc >= ?)
)
SELECT model, n AS calls, ROUND(MIN(latency_ms), 2) AS p95_ms
FROM ranked
WHERE rn >= CAST(0.95 * n + 0.999999 AS INTEGER)  -- nearest-rank p95
GROUP BY model
ORDER BY p95_ms DESC;

-- NAME: error_and_retry_rates
-- Error rate (%) and retry rate (%) grouped by day and model.
SELECT date(ts_utc) AS day,
       COALESCE(model, 'unknown') AS model,
       COUNT(*) AS total_calls,
       SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_calls,
       ROUND(100.0 * SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) / COUNT(*), 2) AS error_rate_pct,
       SUM(CASE WHEN retries > 0 THEN 1 ELSE 0 END) AS retried_calls,
       ROUND(100.0 * SUM(CASE WHEN retries > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS retry_rate_pct,
       SUM(retries) AS total_retries
FROM llm_calls
WHERE (? IS NULL OR ts_utc >= ?)
GROUP BY day, model
ORDER BY day DESC, total_calls DESC;

-- NAME: top_tags
-- Top 10 most expensive tags.
SELECT COALESCE(tag, '(none)') AS tag,
       COUNT(*) AS calls,
       SUM(input_tokens) AS in_tok,
       SUM(output_tokens) AS out_tok,
       ROUND(SUM(cost_usd), 4) AS total_cost_usd,
       ROUND(AVG(cost_usd), 6) AS avg_cost_usd
FROM llm_calls
WHERE (? IS NULL OR ts_utc >= ?)
GROUP BY tag
ORDER BY total_cost_usd DESC
LIMIT 10;

-- NAME: slowest_calls
-- Slowest 20 calls logged.
SELECT id,
       ts_utc,
       provider,
       model,
       ROUND(latency_ms, 1) AS latency_ms,
       attempts,
       status,
       COALESCE(tag, '') AS tag
FROM llm_calls
WHERE (? IS NULL OR ts_utc >= ?)
ORDER BY latency_ms DESC
LIMIT 20;

-- NAME: rate_limit_incidents
-- Rate limit incidents per hour (either retried calls or rate limit error types).
SELECT strftime('%Y-%m-%d %H:00', ts_utc) AS hour_utc,
       COUNT(*) AS incidents,
       SUM(retries) AS total_retries,
       SUM(CASE WHEN error_type LIKE '%RateLimit%' OR error_message LIKE '%429%' THEN 1 ELSE 0 END) AS rate_limit_errors
FROM llm_calls
WHERE (retries > 0 OR error_type LIKE '%RateLimit%' OR error_message LIKE '%429%')
  AND (? IS NULL OR ts_utc >= ?)
GROUP BY hour_utc
ORDER BY hour_utc DESC
LIMIT 24;
