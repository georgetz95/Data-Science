CREATE VIEW IF NOT EXISTS service_family_summary AS
SELECT
  year,
  operations_group,
  service_family,
  COUNT(*) AS records,
  COUNT(DISTINCT rendering_npi) AS providers,
  COUNT(DISTINCT market) AS markets,
  SUM(total_services) AS services,
  SUM(total_beneficiaries) AS beneficiaries,
  SUM(estimated_medicare_payment) AS estimated_payment,
  AVG(payment_per_service) AS avg_payment_per_service,
  AVG(charge_to_allowed_ratio) AS avg_charge_to_allowed_ratio
FROM raw_claims
GROUP BY year, operations_group, service_family;

CREATE VIEW IF NOT EXISTS market_summary_2024 AS
SELECT
  provider_state,
  provider_city,
  market,
  COUNT(*) AS records,
  COUNT(DISTINCT rendering_npi) AS providers,
  COUNT(DISTINCT service_family) AS service_families,
  SUM(total_services) AS services,
  SUM(estimated_medicare_payment) AS estimated_payment,
  AVG(charge_to_allowed_ratio) AS avg_charge_to_allowed_ratio,
  SUM(CASE WHEN operational_priority_decile = 10 THEN 1 ELSE 0 END) AS top_priority_records
FROM raw_claims
WHERE year = 2024
GROUP BY provider_state, provider_city, market;

CREATE VIEW IF NOT EXISTS operational_priority_decile_summary_2024 AS
WITH deciles AS (
  SELECT
    operational_priority_decile,
    COUNT(*) AS records,
    COUNT(DISTINCT rendering_npi) AS providers,
    SUM(total_services) AS services,
    SUM(estimated_medicare_payment) AS estimated_payment,
    SUM(high_volume_flag) AS high_volume_flags,
    SUM(high_intensity_flag) AS high_intensity_flags,
    SUM(rapid_growth_flag) AS rapid_growth_flags
  FROM raw_claims
  WHERE year = 2024
  GROUP BY operational_priority_decile
)
SELECT
  operational_priority_decile,
  records,
  providers,
  services,
  estimated_payment,
  high_volume_flags,
  high_intensity_flags,
  rapid_growth_flags,
  1.0 * records / SUM(records) OVER () AS record_share,
  1.0 * services / SUM(services) OVER () AS service_share,
  1.0 * estimated_payment / SUM(estimated_payment) OVER () AS payment_share
FROM deciles;

CREATE VIEW IF NOT EXISTS provider_type_summary_2024 AS
SELECT
  provider_type,
  COUNT(*) AS records,
  COUNT(DISTINCT rendering_npi) AS providers,
  SUM(total_services) AS services,
  SUM(estimated_medicare_payment) AS estimated_payment,
  SUM(high_volume_flag) AS high_volume_flags,
  SUM(rapid_growth_flag) AS rapid_growth_flags
FROM raw_claims
WHERE year = 2024
GROUP BY provider_type;
