-- PostgreSQL-style feature/scoring template for a fraud-risk model.
-- The local project uses SQLite for portability, but this shows the shape
-- of a production feature table that could feed batch or API scoring.

with base_transactions as (
    select
        transaction_id,
        transaction_time_seconds as time,
        amount,
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
        v21, v22, v23, v24, v25, v26, v27, v28,
        fraud_label
    from raw.credit_card_transactions
    where transaction_time_seconds is not null
),

feature_table as (
    select
        transaction_id,
        time,
        amount,
        ln(1 + amount) as amount_log,
        time / 3600.0 as time_hours,
        mod(time / 3600.0, 24) as hour_of_day,
        case when amount = 0 then 1 else 0 end as amount_is_zero,
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
        v21, v22, v23, v24, v25, v26, v27, v28,
        fraud_label
    from base_transactions
)

select *
from feature_table;
