library(tidyverse)
library(janitor)
library(broom)
library(scales)
library(pROC)
library(PRROC)
library(ranger)
library(officer)
library(flextable)

set.seed(20260714)

find_project_dir <- function() {
  cwd <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

  if (basename(cwd) == "scripts" && dir.exists(file.path(dirname(cwd), "data", "raw"))) {
    return(dirname(cwd))
  }

  if (dir.exists(file.path(cwd, "data", "raw"))) {
    return(cwd)
  }

  candidate <- file.path(cwd, "Healthcare_PartD_Utilization_Cost_Modeling")
  if (dir.exists(file.path(candidate, "data", "raw"))) {
    return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
  }

  stop("Run this script from the project folder, its scripts folder, or the jobs workspace.")
}

project_dir <- find_project_dir()
raw_dir <- file.path(project_dir, "data", "raw")
processed_dir <- file.path(project_dir, "data", "processed")
figures_dir <- file.path(project_dir, "figures")
tables_dir <- file.path(project_dir, "outputs", "tables")
artifacts_dir <- file.path(project_dir, "outputs", "model_artifacts")
outputs_dir <- file.path(project_dir, "outputs")

walk(
  c(processed_dir, figures_dir, tables_dir, artifacts_dir, outputs_dir),
  ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE)
)

numeric_cols <- c(
  "tot_prscrbrs", "tot_clms", "tot_30day_fills", "tot_drug_cst", "tot_benes",
  "ge65_tot_clms", "ge65_tot_30day_fills", "ge65_tot_drug_cst",
  "ge65_tot_benes", "lis_bene_cst_shr", "non_lis_bene_cst_shr"
)

read_partd_file <- function(path) {
  year <- as.integer(str_extract(basename(path), "\\d{4}"))

  read_csv(
    path,
    col_types = cols(.default = col_character()),
    na = c("", "NA", "*", "#"),
    show_col_types = FALSE,
    progress = FALSE
  ) |>
    clean_names() |>
    mutate(year = year) |>
    mutate(across(any_of(numeric_cols), parse_number)) |>
    mutate(
      opioid_drug_flag = coalesce(opioid_drug_flag, "N"),
      opioid_la_drug_flag = coalesce(opioid_la_drug_flag, "N"),
      antbtc_drug_flag = coalesce(antbtc_drug_flag, "N"),
      antpsyct_drug_flag = coalesce(antpsyct_drug_flag, "N")
    )
}

raw_files <- list.files(raw_dir, pattern = "partd_geography_drug_\\d{4}\\.csv$", full.names = TRUE)
if (length(raw_files) == 0) {
  stop("No raw CMS Part D files were found in data/raw.")
}

partd_raw <- map_dfr(raw_files, read_partd_file)

state_lookup <- tibble(
  state_name = c(state.name, "District of Columbia"),
  state_code = c(state.abb, "DC")
)

partd <- partd_raw |>
  mutate(
    state_fips = prscrbr_geo_cd,
    state_name = prscrbr_geo_desc,
    brand_name = str_squish(brnd_name),
    generic_name = str_squish(gnrc_name),
    opioid_flag = opioid_drug_flag == "Y",
    long_acting_opioid_flag = opioid_la_drug_flag == "Y",
    antibiotic_flag = antbtc_drug_flag == "Y",
    antipsychotic_flag = antpsyct_drug_flag == "Y",
    drug_flag_group = case_when(
      antipsychotic_flag ~ "Antipsychotic",
      long_acting_opioid_flag ~ "Long-acting opioid",
      opioid_flag ~ "Opioid",
      antibiotic_flag ~ "Antibiotic",
      TRUE ~ "Other"
    )
  ) |>
  left_join(state_lookup, by = "state_name")

analytic <- partd |>
  filter(
    prscrbr_geo_lvl == "State",
    !is.na(state_code),
    !is.na(generic_name),
    !is.na(brand_name),
    tot_drug_cst > 0,
    tot_benes > 0,
    tot_30day_fills > 0,
    tot_clms > 0
  ) |>
  mutate(
    state_drug_key = str_c(state_code, brand_name, generic_name, sep = "||"),
    cost_per_bene = tot_drug_cst / tot_benes,
    cost_per_claim = tot_drug_cst / tot_clms,
    cost_per_30day_fill = tot_drug_cst / tot_30day_fills,
    claims_per_bene = tot_clms / tot_benes,
    fills_per_bene = tot_30day_fills / tot_benes,
    prescribers_per_100_benes = 100 * tot_prscrbrs / tot_benes,
    ge65_bene_share = ge65_tot_benes / tot_benes,
    ge65_cost_share = ge65_tot_drug_cst / tot_drug_cst,
    bene_cost_share = (lis_bene_cst_shr + non_lis_bene_cst_shr) / tot_drug_cst
  ) |>
  mutate(
    across(
      c(cost_per_bene, cost_per_claim, cost_per_30day_fill, claims_per_bene,
        fills_per_bene, prescribers_per_100_benes, ge65_bene_share,
        ge65_cost_share, bene_cost_share),
      ~ if_else(is.finite(.x), .x, NA_real_)
    ),
    ge65_bene_share = pmin(pmax(ge65_bene_share, 0), 1),
    ge65_cost_share = pmin(pmax(ge65_cost_share, 0), 1),
    bene_cost_share = pmin(pmax(bene_cost_share, 0), 1)
  )

write_csv(analytic, file.path(processed_dir, "partd_state_drug_year_features.csv"))

data_quality <- tibble(
  measure = c(
    "Raw rows",
    "State rows for 50 states plus DC",
    "Analytic rows with positive cost, beneficiaries, claims, and fills",
    "Years",
    "Distinct state codes",
    "Distinct generic names",
    "Distinct brand names",
    "Suppressed or missing total beneficiary values",
    "Suppressed or missing age 65+ beneficiary values",
    "Suppressed or missing age 65+ cost values"
  ),
  value = c(
    nrow(partd_raw),
    nrow(partd |> filter(prscrbr_geo_lvl == "State", !is.na(state_code))),
    nrow(analytic),
    n_distinct(analytic$year),
    n_distinct(analytic$state_code),
    n_distinct(analytic$generic_name),
    n_distinct(analytic$brand_name),
    sum(is.na(partd$tot_benes)),
    sum(is.na(partd$ge65_tot_benes)),
    sum(is.na(partd$ge65_tot_drug_cst))
  )
)

year_summary <- analytic |>
  group_by(year) |>
  summarise(
    rows = n(),
    total_drug_cost = sum(tot_drug_cst, na.rm = TRUE),
    total_claims = sum(tot_clms, na.rm = TRUE),
    total_30day_fills = sum(tot_30day_fills, na.rm = TRUE),
    total_beneficiary_drug_records = sum(tot_benes, na.rm = TRUE),
    median_cost_per_bene = median(cost_per_bene, na.rm = TRUE),
    median_cost_per_30day_fill = median(cost_per_30day_fill, na.rm = TRUE),
    .groups = "drop"
  )

flag_summary <- analytic |>
  group_by(year, drug_flag_group) |>
  summarise(
    rows = n(),
    total_drug_cost = sum(tot_drug_cst, na.rm = TRUE),
    total_30day_fills = sum(tot_30day_fills, na.rm = TRUE),
    median_cost_per_bene = median(cost_per_bene, na.rm = TRUE),
    median_cost_per_30day_fill = median(cost_per_30day_fill, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(cost_share = total_drug_cost / sum(total_drug_cost), .by = year)

state_summary_2024 <- analytic |>
  filter(year == 2024) |>
  group_by(state_code, state_name) |>
  summarise(
    rows = n(),
    total_drug_cost = sum(tot_drug_cst, na.rm = TRUE),
    total_claims = sum(tot_clms, na.rm = TRUE),
    total_30day_fills = sum(tot_30day_fills, na.rm = TRUE),
    median_cost_per_bene = median(cost_per_bene, na.rm = TRUE),
    median_cost_per_30day_fill = median(cost_per_30day_fill, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(total_drug_cost))

top_drugs_2024 <- analytic |>
  filter(year == 2024) |>
  group_by(generic_name, brand_name, drug_flag_group) |>
  summarise(
    state_count = n_distinct(state_code),
    total_drug_cost = sum(tot_drug_cst, na.rm = TRUE),
    total_30day_fills = sum(tot_30day_fills, na.rm = TRUE),
    total_beneficiary_drug_records = sum(tot_benes, na.rm = TRUE),
    median_cost_per_bene = median(cost_per_bene, na.rm = TRUE),
    median_cost_per_30day_fill = median(cost_per_30day_fill, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(total_drug_cost)) |>
  slice_head(n = 25)

concentration_2024 <- analytic |>
  filter(year == 2024) |>
  arrange(desc(tot_drug_cst)) |>
  mutate(
    row_rank = row_number(),
    row_share = row_rank / n(),
    cumulative_cost_share = cumsum(tot_drug_cst) / sum(tot_drug_cst)
  ) |>
  select(row_rank, row_share, cumulative_cost_share, state_code, brand_name, generic_name, tot_drug_cst)

top_5pct_cost_share <- concentration_2024 |>
  filter(row_share <= 0.05) |>
  summarise(cost_share = max(cumulative_cost_share)) |>
  pull(cost_share)

h1_data <- analytic |>
  filter(!is.na(cost_per_bene), !is.na(fills_per_bene)) |>
  mutate(
    year_centered = year - min(year),
    log_cost_per_bene = log1p(cost_per_bene),
    log_total_benes = log1p(tot_benes),
    log_total_30day_fills = log1p(tot_30day_fills),
    ge65_bene_share = replace_na(ge65_bene_share, median(ge65_bene_share, na.rm = TRUE)),
    bene_cost_share = replace_na(bene_cost_share, median(bene_cost_share, na.rm = TRUE))
  )

cost_model <- lm(
  log_cost_per_bene ~ year_centered + log_total_benes + log_total_30day_fills +
    fills_per_bene + ge65_bene_share + bene_cost_share +
    opioid_flag + long_acting_opioid_flag + antibiotic_flag + antipsychotic_flag,
  data = h1_data
)

cost_model_results <- tidy(cost_model, conf.int = TRUE) |>
  mutate(
    estimate_pct = exp(estimate) - 1,
    conf_low_pct = exp(conf.low) - 1,
    conf_high_pct = exp(conf.high) - 1
  )

high_cost_panel <- analytic |>
  group_by(year) |>
  mutate(
    high_cost_decile = cost_per_bene >= quantile(cost_per_bene, 0.90, na.rm = TRUE),
    cost_decile = ntile(cost_per_bene, 10)
  ) |>
  ungroup() |>
  arrange(state_drug_key, year) |>
  group_by(state_drug_key) |>
  mutate(
    target_year = lead(year),
    high_cost_next = lead(high_cost_decile),
    next_cost_per_bene = lead(cost_per_bene),
    next_total_drug_cost = lead(tot_drug_cst)
  ) |>
  ungroup() |>
  filter(target_year == year + 1, !is.na(high_cost_next)) |>
  mutate(
    log_total_benes = log1p(tot_benes),
    log_total_30day_fills = log1p(tot_30day_fills),
    log_total_drug_cost = log1p(tot_drug_cst),
    log_cost_per_bene = log1p(cost_per_bene),
    opioid_flag_num = as.integer(opioid_flag),
    antibiotic_flag_num = as.integer(antibiotic_flag),
    antipsychotic_flag_num = as.integer(antipsychotic_flag),
    long_acting_opioid_flag_num = as.integer(long_acting_opioid_flag)
  )

feature_cols <- c(
  "log_total_benes", "log_total_30day_fills", "log_total_drug_cost",
  "log_cost_per_bene", "claims_per_bene", "fills_per_bene",
  "ge65_bene_share", "ge65_cost_share", "bene_cost_share",
  "opioid_flag_num", "long_acting_opioid_flag_num",
  "antibiotic_flag_num", "antipsychotic_flag_num"
)

model_df <- high_cost_panel |>
  select(
    year, target_year, state_code, state_name, brand_name, generic_name,
    high_cost_next, next_cost_per_bene, next_total_drug_cost,
    all_of(feature_cols)
  )

train_df <- model_df |> filter(year <= 2022)
test_df <- model_df |> filter(year == 2023)

for (col in feature_cols) {
  fill_value <- median(train_df[[col]], na.rm = TRUE)
  train_df[[col]][is.na(train_df[[col]])] <- fill_value
  test_df[[col]][is.na(test_df[[col]])] <- fill_value
}

logit_formula <- as.formula(str_c("high_cost_next ~ ", str_c(feature_cols, collapse = " + ")))
logit_model <- glm(logit_formula, data = train_df, family = binomial())

rf_train <- train_df |>
  mutate(high_cost_next = factor(if_else(high_cost_next, "Yes", "No"), levels = c("No", "Yes")))

rf_test <- test_df |>
  mutate(high_cost_next = factor(if_else(high_cost_next, "Yes", "No"), levels = c("No", "Yes")))

rf_formula <- as.formula(str_c("high_cost_next ~ ", str_c(feature_cols, collapse = " + ")))
rf_model <- ranger(
  rf_formula,
  data = rf_train |> select(high_cost_next, all_of(feature_cols)),
  probability = TRUE,
  num.trees = 400,
  mtry = 4,
  min.node.size = 25,
  importance = "permutation",
  seed = 20260714
)

test_scored <- test_df |>
  mutate(
    pred_logit = as.numeric(predict(logit_model, newdata = test_df, type = "response")),
    pred_rf = predict(rf_model, data = rf_test |> select(all_of(feature_cols)))$predictions[, "Yes"],
    pred_current_cost = percent_rank(log_cost_per_bene)
  )

roc_auc <- function(truth, score) {
  as.numeric(auc(roc(response = truth, predictor = score, quiet = TRUE)))
}

average_precision <- function(truth, score) {
  PRROC::pr.curve(
    scores.class0 = score[truth],
    scores.class1 = score[!truth],
    curve = FALSE
  )$auc.integral
}

review_metrics <- function(data, score_col, label, review_share = 0.10) {
  n_review <- ceiling(nrow(data) * review_share)
  selected <- data |>
    arrange(desc(.data[[score_col]])) |>
    mutate(selected = row_number() <= n_review)

  tibble(
    model = label,
    review_share = review_share,
    reviewed_state_drug_pairs = n_review,
    precision = mean(selected$high_cost_next[selected$selected]),
    recall = sum(selected$high_cost_next[selected$selected]) / sum(selected$high_cost_next),
    captured_2024_drug_cost_share = sum(selected$next_total_drug_cost[selected$selected], na.rm = TRUE) /
      sum(selected$next_total_drug_cost, na.rm = TRUE),
    high_cost_pairs_captured = sum(selected$high_cost_next[selected$selected]),
    total_high_cost_pairs = sum(selected$high_cost_next)
  )
}

model_metrics <- tibble(
  model = c("Logistic regression", "Random forest", "Current cost per beneficiary baseline"),
  target_year = 2024,
  roc_auc = c(
    roc_auc(test_scored$high_cost_next, test_scored$pred_logit),
    roc_auc(test_scored$high_cost_next, test_scored$pred_rf),
    roc_auc(test_scored$high_cost_next, test_scored$pred_current_cost)
  ),
  average_precision = c(
    average_precision(test_scored$high_cost_next, test_scored$pred_logit),
    average_precision(test_scored$high_cost_next, test_scored$pred_rf),
    average_precision(test_scored$high_cost_next, test_scored$pred_current_cost)
  )
) |>
  bind_cols(
    bind_rows(
      review_metrics(test_scored, "pred_logit", "Logistic regression"),
      review_metrics(test_scored, "pred_rf", "Random forest"),
      review_metrics(test_scored, "pred_current_cost", "Current cost per beneficiary baseline")
    ) |>
      select(review_share, reviewed_state_drug_pairs, precision, recall, captured_2024_drug_cost_share)
  )

risk_deciles <- test_scored |>
  arrange(desc(pred_rf)) |>
  mutate(risk_decile = ceiling(row_number() / n() * 10)) |>
  group_by(risk_decile) |>
  summarise(
    rows = n(),
    high_cost_rate = mean(high_cost_next),
    captured_2024_drug_cost = sum(next_total_drug_cost, na.rm = TRUE),
    captured_2024_drug_cost_share = captured_2024_drug_cost / sum(test_scored$next_total_drug_cost, na.rm = TRUE),
    .groups = "drop"
  )

selected_for_review <- test_scored |>
  arrange(desc(pred_rf)) |>
  slice_head(prop = 0.10)

program_cost_per_state_drug_pair <- 2000
targeted_cost <- sum(selected_for_review$next_total_drug_cost, na.rm = TRUE)
targeted_pairs <- nrow(selected_for_review)
break_even_reduction <- (targeted_pairs * program_cost_per_state_drug_pair) / targeted_cost

cost_benefit_scenario <- tibble(
  avoidable_cost_reduction = c(0.005, 0.01, 0.02, 0.03, 0.05),
  reviewed_state_drug_pairs = targeted_pairs,
  program_cost_per_state_drug_pair = program_cost_per_state_drug_pair,
  review_program_cost = targeted_pairs * program_cost_per_state_drug_pair,
  targeted_2024_drug_cost = targeted_cost,
  estimated_avoidable_cost = targeted_2024_drug_cost * avoidable_cost_reduction,
  net_savings = estimated_avoidable_cost - review_program_cost,
  break_even_reduction = break_even_reduction
)

rf_importance <- enframe(rf_model$variable.importance, name = "feature", value = "importance") |>
  arrange(desc(importance))

write_csv(data_quality, file.path(tables_dir, "data_quality_checks.csv"))
write_csv(year_summary, file.path(tables_dir, "annual_partd_summary.csv"))
write_csv(flag_summary, file.path(tables_dir, "drug_flag_summary.csv"))
write_csv(state_summary_2024, file.path(tables_dir, "state_summary_2024.csv"))
write_csv(top_drugs_2024, file.path(tables_dir, "top_drugs_2024.csv"))
write_csv(concentration_2024, file.path(tables_dir, "cost_concentration_2024.csv"))
write_csv(cost_model_results, file.path(tables_dir, "log_cost_model_results.csv"))
write_csv(model_metrics, file.path(tables_dir, "prediction_model_metrics.csv"))
write_csv(risk_deciles, file.path(tables_dir, "risk_decile_capture.csv"))
write_csv(cost_benefit_scenario, file.path(tables_dir, "cost_benefit_scenario.csv"))
write_csv(rf_importance, file.path(tables_dir, "random_forest_feature_importance.csv"))
write_csv(test_scored, file.path(processed_dir, "partd_2024_scored_state_drug_pairs.csv"))

saveRDS(logit_model, file.path(artifacts_dir, "logistic_high_cost_model.rds"))
saveRDS(rf_model, file.path(artifacts_dir, "ranger_high_cost_model.rds"))

theme_set(theme_minimal(base_size = 11))

ggplot(year_summary, aes(year, total_drug_cost)) +
  geom_line(linewidth = 0.9, color = "#24693d") +
  geom_point(size = 2.4, color = "#24693d") +
  scale_y_continuous(labels = label_dollar(scale = 1e-9, suffix = "B")) +
  scale_x_continuous(breaks = year_summary$year) +
  labs(
    title = "Medicare Part D Drug Cost in State-Level Records",
    x = NULL,
    y = "Total drug cost"
  )
ggsave(file.path(figures_dir, "annual_partd_cost_trend.png"), width = 7, height = 4.5, dpi = 300, bg = "white")

ggplot(analytic |> filter(year == 2024), aes(cost_per_bene)) +
  geom_histogram(bins = 45, fill = "#4e79a7", color = "white", linewidth = 0.2) +
  scale_x_log10(labels = label_dollar()) +
  labs(
    title = "Distribution of 2024 Cost per Beneficiary Record",
    x = "Cost per beneficiary record, log scale",
    y = "State-drug records"
  )
ggsave(file.path(figures_dir, "cost_distribution_2024.png"), width = 7, height = 4.5, dpi = 300, bg = "white")

ggplot(concentration_2024, aes(row_share, cumulative_cost_share)) +
  geom_line(color = "#8f5d2b", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  labs(
    title = "2024 Part D Cost Concentration",
    x = "Share of state-drug records, sorted by cost",
    y = "Cumulative share of drug cost"
  )
ggsave(file.path(figures_dir, "cost_concentration_2024.png"), width = 7, height = 4.5, dpi = 300, bg = "white")

state_summary_2024 |>
  slice_max(total_drug_cost, n = 15) |>
  ggplot(aes(total_drug_cost, fct_reorder(state_name, total_drug_cost))) +
  geom_col(fill = "#3f6f88") +
  scale_x_continuous(labels = label_dollar(scale = 1e-9, suffix = "B")) +
  labs(
    title = "Highest 2024 State-Level Part D Drug Cost Totals",
    x = "Total drug cost",
    y = NULL
  )
ggsave(file.path(figures_dir, "top_states_2024.png"), width = 7, height = 4.8, dpi = 300, bg = "white")

flag_terms <- c(
  opioid_flagTRUE = "Opioid",
  long_acting_opioid_flagTRUE = "Long-acting opioid",
  antibiotic_flagTRUE = "Antibiotic",
  antipsychotic_flagTRUE = "Antipsychotic"
)

cost_model_results |>
  filter(term %in% names(flag_terms)) |>
  mutate(term = recode(term, !!!flag_terms)) |>
  ggplot(aes(estimate_pct, fct_reorder(term, estimate_pct))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = conf_low_pct, xmax = conf_high_pct), height = 0.18, color = "#5d5d5d") +
  geom_point(size = 2.5, color = "#9b5c7a") +
  scale_x_continuous(labels = percent) +
  labs(
    title = "Adjusted Difference in Cost per Beneficiary",
    x = "Percent difference in cost per beneficiary",
    y = NULL
  )
ggsave(file.path(figures_dir, "flag_cost_model_effects.png"), width = 7, height = 4.3, dpi = 300, bg = "white")

roc_df <- bind_rows(
  as_tibble(coords(roc(test_scored$high_cost_next, test_scored$pred_logit, quiet = TRUE), "all")) |>
    mutate(model = "Logistic regression"),
  as_tibble(coords(roc(test_scored$high_cost_next, test_scored$pred_rf, quiet = TRUE), "all")) |>
    mutate(model = "Random forest"),
  as_tibble(coords(roc(test_scored$high_cost_next, test_scored$pred_current_cost, quiet = TRUE), "all")) |>
    mutate(model = "Current-cost baseline")
) |>
  mutate(false_positive_rate = 1 - specificity)

ggplot(roc_df, aes(false_positive_rate, sensitivity, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  coord_equal() +
  labs(
    title = "Prediction of 2024 High-Cost State-Drug Records",
    x = "False positive rate",
    y = "True positive rate",
    color = NULL
  ) +
  theme(legend.position = "bottom")
ggsave(file.path(figures_dir, "model_roc_curve.png"), width = 6.5, height = 5, dpi = 300, bg = "white")

ggplot(risk_deciles, aes(factor(risk_decile), captured_2024_drug_cost_share)) +
  geom_col(fill = "#756bb1") +
  scale_y_continuous(labels = percent) +
  labs(
    title = "2024 Drug Cost Captured by Predicted Risk Decile",
    x = "Predicted risk decile, 1 = highest risk",
    y = "Share of 2024 drug cost"
  )
ggsave(file.path(figures_dir, "risk_decile_cost_capture.png"), width = 7, height = 4.5, dpi = 300, bg = "white")

summary_lines <- c(
  "Medicare Part D Utilization and Cost Modeling",
  "",
  str_glue("CMS Part D geography/drug files used: {min(analytic$year)}-{max(analytic$year)}."),
  str_glue("Analytic state-drug-year records: {comma(nrow(analytic))}."),
  str_glue("2024 state-level drug cost represented in analytic records: {dollar(sum((analytic |> filter(year == 2024))$tot_drug_cst, na.rm = TRUE))}."),
  str_glue("Top 5% of 2024 state-drug records accounted for {percent(top_5pct_cost_share, accuracy = 0.1)} of represented drug cost."),
  str_glue("Random forest ROC AUC for predicting 2024 high-cost records: {round((model_metrics |> filter(model == 'Random forest'))$roc_auc, 3)}."),
  str_glue("Random forest average precision: {round((model_metrics |> filter(model == 'Random forest'))$average_precision, 3)}."),
  str_glue("Top 10% review list captured {percent((model_metrics |> filter(model == 'Random forest'))$captured_2024_drug_cost_share, accuracy = 0.1)} of represented 2024 drug cost."),
  str_glue("Break-even avoidable-cost reduction in the review scenario: {percent(break_even_reduction, accuracy = 0.01)}.")
)
writeLines(summary_lines, file.path(outputs_dir, "analysis_summary.txt"))

deck_path <- file.path(outputs_dir, "healthcare_partd_summary_deck.pptx")

try({
  ppt <- read_pptx()

  ppt <- add_slide(ppt, layout = "Title Slide", master = "Office Theme")
  ppt <- ph_with(ppt, "Medicare Part D Utilization and Cost Modeling", location = ph_location_type(type = "ctrTitle"))
  ppt <- ph_with(ppt, "State-drug cost concentration, high-cost prediction, and review scenario", location = ph_location_type(type = "subTitle"))

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Key Findings", location = ph_location_type(type = "title"))
  ppt <- ph_with(
    ppt,
    unordered_list(
      level_list = c(1, 1, 1),
      str_list = c(
        str_glue("Analytic file includes {comma(nrow(analytic))} state-drug-year records across {n_distinct(analytic$year)} CMS releases."),
        str_glue("The top 5% of 2024 state-drug records represented {percent(top_5pct_cost_share, accuracy = 0.1)} of drug cost."),
        str_glue("The highest-risk decile from the random forest captured {percent((model_metrics |> filter(model == 'Random forest'))$captured_2024_drug_cost_share, accuracy = 0.1)} of represented 2024 drug cost.")
      )
    ),
    location = ph_location_type(type = "body")
  )

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Model Validation", location = ph_location_type(type = "title"))
  metrics_ft <- model_metrics |>
    mutate(
      roc_auc = round(roc_auc, 3),
      average_precision = round(average_precision, 3),
      precision = percent(precision, accuracy = 0.1),
      recall = percent(recall, accuracy = 0.1),
      captured_2024_drug_cost_share = percent(captured_2024_drug_cost_share, accuracy = 0.1)
    ) |>
    select(model, roc_auc, average_precision, precision, recall, captured_2024_drug_cost_share) |>
    flextable() |>
    autofit()
  ppt <- ph_with(ppt, metrics_ft, location = ph_location_type(type = "body"))

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Review Scenario", location = ph_location_type(type = "title"))
  scenario_ft <- cost_benefit_scenario |>
    mutate(
      avoidable_cost_reduction = percent(avoidable_cost_reduction),
      review_program_cost = dollar(review_program_cost),
      estimated_avoidable_cost = dollar(estimated_avoidable_cost),
      net_savings = dollar(net_savings),
      break_even_reduction = percent(break_even_reduction, accuracy = 0.01)
    ) |>
    select(avoidable_cost_reduction, review_program_cost, estimated_avoidable_cost, net_savings, break_even_reduction) |>
    flextable() |>
    autofit()
  ppt <- ph_with(ppt, scenario_ft, location = ph_location_type(type = "body"))

  print(ppt, target = deck_path)
}, silent = TRUE)

message("Analysis complete. Outputs written to: ", project_dir)
