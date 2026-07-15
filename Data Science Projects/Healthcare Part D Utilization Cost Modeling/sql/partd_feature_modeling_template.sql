/*
Medicare Part D geography/drug feature template.

The R workflow uses CSV files directly, but this SQL shows the same feature
logic as it would usually be staged in a warehouse. The raw table is assumed
to contain one CMS Part D geography/drug release per year with the original
CMS column names converted to snake_case.
*/

create table analytics.partd_state_drug_features as
with state_rows as (
    select
        cast(year as integer) as year,
        prscrbr_geo_cd as state_fips,
        prscrbr_geo_desc as state_name,
        brnd_name as brand_name,
        gnrc_name as generic_name,
        cast(tot_prscrbrs as numeric) as total_prescribers,
        cast(tot_clms as numeric) as total_claims,
        cast(tot_30day_fills as numeric) as total_30day_fills,
        cast(tot_drug_cst as numeric) as total_drug_cost,
        cast(tot_benes as numeric) as total_beneficiaries,
        cast(ge65_tot_clms as numeric) as age65_claims,
        cast(ge65_tot_30day_fills as numeric) as age65_30day_fills,
        cast(ge65_tot_drug_cst as numeric) as age65_drug_cost,
        cast(ge65_tot_benes as numeric) as age65_beneficiaries,
        cast(lis_bene_cst_shr as numeric) as lis_beneficiary_cost_share,
        cast(non_lis_bene_cst_shr as numeric) as non_lis_beneficiary_cost_share,
        case when opioid_drug_flag = 'Y' then 1 else 0 end as opioid_flag,
        case when opioid_la_drug_flag = 'Y' then 1 else 0 end as long_acting_opioid_flag,
        case when antbtc_drug_flag = 'Y' then 1 else 0 end as antibiotic_flag,
        case when antpsyct_drug_flag = 'Y' then 1 else 0 end as antipsychotic_flag
    from raw.cms_partd_geography_drug
    where prscrbr_geo_lvl = 'State'
      and prscrbr_geo_desc is not null
      and brnd_name is not null
      and gnrc_name is not null
),
analytic_rows as (
    select
        *,
        concat(state_name, '||', brand_name, '||', generic_name) as state_drug_key,
        total_drug_cost / nullif(total_beneficiaries, 0) as cost_per_beneficiary,
        total_drug_cost / nullif(total_claims, 0) as cost_per_claim,
        total_drug_cost / nullif(total_30day_fills, 0) as cost_per_30day_fill,
        total_claims / nullif(total_beneficiaries, 0) as claims_per_beneficiary,
        total_30day_fills / nullif(total_beneficiaries, 0) as fills_per_beneficiary,
        age65_beneficiaries / nullif(total_beneficiaries, 0) as age65_beneficiary_share,
        age65_drug_cost / nullif(total_drug_cost, 0) as age65_cost_share,
        (lis_beneficiary_cost_share + non_lis_beneficiary_cost_share) /
            nullif(total_drug_cost, 0) as beneficiary_cost_share
    from state_rows
    where total_drug_cost > 0
      and total_beneficiaries > 0
      and total_claims > 0
      and total_30day_fills > 0
)
select *
from analytic_rows;

create table analytics.partd_high_cost_training_panel as
with ranked as (
    select
        *,
        case
            when cost_per_beneficiary >= percentile_cont(0.90)
                within group (order by cost_per_beneficiary)
                over (partition by year)
            then 1 else 0
        end as high_cost_decile
    from analytics.partd_state_drug_features
),
paired as (
    select
        year,
        year + 1 as target_year,
        state_drug_key,
        state_name,
        brand_name,
        generic_name,
        ln(1 + total_beneficiaries) as log_total_beneficiaries,
        ln(1 + total_30day_fills) as log_total_30day_fills,
        ln(1 + total_drug_cost) as log_total_drug_cost,
        ln(1 + cost_per_beneficiary) as log_cost_per_beneficiary,
        claims_per_beneficiary,
        fills_per_beneficiary,
        age65_beneficiary_share,
        age65_cost_share,
        beneficiary_cost_share,
        opioid_flag,
        long_acting_opioid_flag,
        antibiotic_flag,
        antipsychotic_flag,
        lead(high_cost_decile) over (
            partition by state_drug_key
            order by year
        ) as next_year_high_cost_decile,
        lead(total_drug_cost) over (
            partition by state_drug_key
            order by year
        ) as next_year_total_drug_cost
    from ranked
)
select *
from paired
where next_year_high_cost_decile is not null;
