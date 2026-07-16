DROP VIEW IF EXISTS claims_line_features;
CREATE VIEW claims_line_features AS
SELECT
    year,
    rendering_npi,
    provider_name,
    provider_entity_type,
    provider_city,
    provider_state,
    provider_zip5,
    provider_type,
    medicare_participating,
    hcpcs_code,
    hcpcs_description,
    hcpcs_drug_indicator,
    place_of_service,
    total_beneficiaries,
    total_services,
    total_beneficiary_day_services,
    average_submitted_charge,
    average_medicare_allowed,
    average_medicare_payment,
    average_medicare_standardized,
    total_services * average_submitted_charge AS estimated_submitted_charge,
    total_services * average_medicare_allowed AS estimated_allowed_amount,
    total_services * average_medicare_payment AS estimated_medicare_payment,
    total_services * average_medicare_standardized AS estimated_standardized_payment,
    total_services / NULLIF(total_beneficiaries, 0) AS services_per_beneficiary,
    total_beneficiary_day_services / NULLIF(total_beneficiaries, 0) AS beneficiary_days_per_beneficiary,
    average_submitted_charge / NULLIF(average_medicare_allowed, 0) AS charge_to_allowed_ratio,
    average_medicare_payment / NULLIF(average_medicare_allowed, 0) AS payment_to_allowed_ratio,
    average_medicare_standardized / NULLIF(average_medicare_payment, 0) AS standardized_to_payment_ratio
FROM raw_claims;

DROP VIEW IF EXISTS hcpcs_specialty_summary;
CREATE VIEW hcpcs_specialty_summary AS
SELECT
    year,
    provider_state,
    provider_type,
    hcpcs_code,
    hcpcs_description,
    place_of_service,
    COUNT(*) AS line_count,
    COUNT(DISTINCT rendering_npi) AS provider_count,
    SUM(total_beneficiaries) AS beneficiaries,
    SUM(total_services) AS services,
    SUM(estimated_medicare_payment) AS estimated_medicare_payment,
    AVG(services_per_beneficiary) AS average_services_per_beneficiary,
    AVG(charge_to_allowed_ratio) AS average_charge_to_allowed_ratio
FROM claims_line_features
GROUP BY year, provider_state, provider_type, hcpcs_code, hcpcs_description, place_of_service;

DROP VIEW IF EXISTS provider_payment_profile;
CREATE VIEW provider_payment_profile AS
SELECT
    year,
    rendering_npi,
    provider_name,
    provider_entity_type,
    provider_city,
    provider_state,
    provider_zip5,
    provider_type,
    COUNT(*) AS billed_hcpcs_lines,
    COUNT(DISTINCT hcpcs_code) AS distinct_hcpcs_codes,
    SUM(total_beneficiaries) AS summed_beneficiary_counts,
    SUM(total_services) AS total_services,
    SUM(estimated_medicare_payment) AS estimated_medicare_payment,
    SUM(estimated_submitted_charge) AS estimated_submitted_charge,
    SUM(estimated_allowed_amount) AS estimated_allowed_amount,
    SUM(estimated_submitted_charge) / NULLIF(SUM(estimated_allowed_amount), 0) AS submitted_to_allowed_ratio,
    SUM(total_services) / NULLIF(SUM(total_beneficiaries), 0) AS services_per_beneficiary
FROM claims_line_features
GROUP BY
    year,
    rendering_npi,
    provider_name,
    provider_entity_type,
    provider_city,
    provider_state,
    provider_zip5,
    provider_type;

DROP VIEW IF EXISTS yearly_state_summary;
CREATE VIEW yearly_state_summary AS
SELECT
    year,
    provider_state,
    COUNT(*) AS line_count,
    COUNT(DISTINCT rendering_npi) AS provider_count,
    COUNT(DISTINCT hcpcs_code) AS hcpcs_code_count,
    SUM(total_services) AS total_services,
    SUM(estimated_medicare_payment) AS estimated_medicare_payment,
    SUM(estimated_submitted_charge) AS estimated_submitted_charge,
    SUM(estimated_allowed_amount) AS estimated_allowed_amount,
    SUM(estimated_submitted_charge) / NULLIF(SUM(estimated_allowed_amount), 0) AS submitted_to_allowed_ratio
FROM claims_line_features
GROUP BY year, provider_state;

DROP VIEW IF EXISTS high_volume_code_summary;
CREATE VIEW high_volume_code_summary AS
SELECT
    year,
    hcpcs_code,
    hcpcs_description,
    place_of_service,
    COUNT(*) AS line_count,
    COUNT(DISTINCT rendering_npi) AS provider_count,
    SUM(total_services) AS total_services,
    SUM(estimated_medicare_payment) AS estimated_medicare_payment,
    SUM(estimated_submitted_charge) AS estimated_submitted_charge
FROM claims_line_features
GROUP BY year, hcpcs_code, hcpcs_description, place_of_service
HAVING COUNT(*) >= 20
ORDER BY estimated_medicare_payment DESC;
