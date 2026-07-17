#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(haven)
  library(survey)
  library(broom)
  library(mice)
  library(janitor)
})

options(survey.lonely.psu = "adjust")
set.seed(20260716)

project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (basename(project_dir) == "scripts") {
  project_dir <- dirname(project_dir)
}

raw_dir <- file.path(project_dir, "data", "raw")
processed_dir <- file.path(project_dir, "data", "processed")
output_dir <- file.path(project_dir, "outputs")
figure_dir <- file.path(project_dir, "figures")

dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

files <- tribble(
  ~source_file, ~url, ~description,
  "DEMO_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt", "Demographics and survey design variables",
  "HUQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HUQ_J.xpt", "Hospital utilization and access to care",
  "HIQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HIQ_J.xpt", "Health insurance"
)

download_if_missing <- function(source_file, url) {
  target <- file.path(raw_dir, source_file)
  if (!file.exists(target) || file.size(target) < 100000) {
    message("Downloading ", source_file)
    download.file(url, target, mode = "wb", quiet = TRUE)
  }
  target
}

walk2(files$source_file, files$url, download_if_missing)

demo <- read_xpt(file.path(raw_dir, "DEMO_J.XPT"))
huq <- read_xpt(file.path(raw_dir, "HUQ_J.XPT"))
hiq <- read_xpt(file.path(raw_dir, "HIQ_J.XPT"))

raw_inventory <- tibble(
  source_file = c("DEMO_J", "HUQ_J", "HIQ_J"),
  rows = c(nrow(demo), nrow(huq), nrow(hiq)),
  columns = c(ncol(demo), ncol(huq), ncol(hiq))
)

merged <- demo %>%
  left_join(huq, by = "SEQN") %>%
  left_join(hiq, by = "SEQN")

adult <- merged %>%
  filter(RIDAGEYR >= 18)

analytic <- adult %>%
  transmute(
    seqn = SEQN,
    age_years = RIDAGEYR,
    age_group = cut(
      RIDAGEYR,
      breaks = c(17, 34, 49, 64, Inf),
      labels = c("18-34", "35-49", "50-64", "65+")
    ),
    gender = case_when(
      RIAGENDR == 1 ~ "Male",
      RIAGENDR == 2 ~ "Female",
      TRUE ~ NA_character_
    ),
    race_ethnicity = case_when(
      RIDRETH3 == 1 ~ "Mexican American",
      RIDRETH3 == 2 ~ "Other Hispanic",
      RIDRETH3 == 3 ~ "Non-Hispanic White",
      RIDRETH3 == 4 ~ "Non-Hispanic Black",
      RIDRETH3 == 6 ~ "Non-Hispanic Asian",
      RIDRETH3 == 7 ~ "Other/multiracial",
      TRUE ~ NA_character_
    ),
    education = case_when(
      DMDEDUC2 %in% c(1, 2) ~ "Less than high school",
      DMDEDUC2 == 3 ~ "High school/GED",
      DMDEDUC2 == 4 ~ "Some college/AA",
      DMDEDUC2 == 5 ~ "College graduate+",
      TRUE ~ NA_character_
    ),
    poverty_ratio = if_else(INDFMPIR >= 0 & INDFMPIR <= 5, INDFMPIR, NA_real_),
    health_status = case_when(
      HUQ010 == 1 ~ "Excellent",
      HUQ010 == 2 ~ "Very good",
      HUQ010 == 3 ~ "Good",
      HUQ010 == 4 ~ "Fair",
      HUQ010 == 5 ~ "Poor",
      TRUE ~ NA_character_
    ),
    fair_poor = case_when(
      HUQ010 %in% c(4, 5) ~ 1,
      HUQ010 %in% c(1, 2, 3) ~ 0,
      TRUE ~ NA_real_
    ),
    insurance_status = case_when(
      HIQ011 == 1 ~ "Insured",
      HIQ011 == 2 ~ "Uninsured",
      TRUE ~ NA_character_
    ),
    usual_care = case_when(
      HUQ030 %in% c(1, 3) ~ "Has a usual place",
      HUQ030 == 2 ~ "No usual place",
      TRUE ~ NA_character_
    ),
    care_visits = case_when(
      HUQ051 == 0 ~ "0",
      HUQ051 == 1 ~ "1",
      HUQ051 == 2 ~ "2-3",
      HUQ051 == 3 ~ "4-5",
      HUQ051 %in% 4:8 ~ "6+",
      TRUE ~ NA_character_
    ),
    wtint2yr = WTINT2YR,
    strata = SDMVSTRA,
    psu = SDMVPSU
  )

variable_dictionary <- tribble(
  ~analysis_variable, ~source_file, ~source_variable, ~role, ~notes,
  "seqn", "DEMO_J", "SEQN", "Linkage key", "Public respondent sequence number",
  "age_years", "DEMO_J", "RIDAGEYR", "Covariate", "Age in years at screening, top-coded at 80",
  "age_group", "DEMO_J", "RIDAGEYR", "Derived covariate", "18-34, 35-49, 50-64, 65+",
  "gender", "DEMO_J", "RIAGENDR", "Covariate", "Male, Female",
  "race_ethnicity", "DEMO_J", "RIDRETH3", "Covariate", "Race and Hispanic origin with Non-Hispanic Asian category",
  "education", "DEMO_J", "DMDEDUC2", "Covariate", "Adult education grouped into four categories",
  "poverty_ratio", "DEMO_J", "INDFMPIR", "Covariate", "Family income to poverty ratio, 0 to 5",
  "health_status", "HUQ_J", "HUQ010", "Outcome source", "Self-rated general health",
  "fair_poor", "HUQ_J", "HUQ010", "Primary outcome", "Fair or poor versus excellent, very good, or good",
  "insurance_status", "HIQ_J", "HIQ011", "Primary exposure", "Currently insured versus uninsured",
  "usual_care", "HUQ_J", "HUQ030", "Primary exposure", "Has one or more routine places for care versus no usual place",
  "care_visits", "HUQ_J", "HUQ051", "Descriptive variable", "Health care professional visits in the past 12 months",
  "wtint2yr", "DEMO_J", "WTINT2YR", "Survey design", "Full sample two-year interview weight",
  "strata", "DEMO_J", "SDMVSTRA", "Survey design", "Masked variance stratum",
  "psu", "DEMO_J", "SDMVPSU", "Survey design", "Masked variance PSU"
)

missing_summary <- analytic %>%
  summarise(across(
    c(fair_poor, insurance_status, usual_care, education, poverty_ratio, wtint2yr, strata, psu),
    ~ sum(is.na(.x))
  )) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_n") %>%
  mutate(
    denominator = nrow(analytic),
    missing_pct = round(100 * missing_n / denominator, 2)
  )

quality_summary <- bind_rows(
  raw_inventory %>%
    transmute(check = paste0("Rows in ", source_file), value = as.character(rows)),
  tibble(
    check = c(
      "Merged person-level rows",
      "Adult rows age 18+",
      "Adult rows with modeled outcome",
      "Adult rows with insurance status",
      "Adult rows with usual care status",
      "Adult rows missing family income to poverty ratio"
    ),
    value = as.character(c(
      nrow(merged),
      nrow(analytic),
      sum(!is.na(analytic$fair_poor)),
      sum(!is.na(analytic$insurance_status)),
      sum(!is.na(analytic$usual_care)),
      sum(is.na(analytic$poverty_ratio))
    ))
  )
)

model_base <- analytic %>%
  filter(
    !is.na(fair_poor),
    !is.na(insurance_status),
    !is.na(usual_care),
    !is.na(age_group),
    !is.na(gender),
    !is.na(race_ethnicity),
    !is.na(education),
    !is.na(wtint2yr),
    !is.na(strata),
    !is.na(psu)
  )

impute_frame <- model_base %>%
  transmute(
    poverty_ratio,
    age_years,
    fair_poor,
    insurance_status = factor(insurance_status),
    usual_care = factor(usual_care),
    gender = factor(gender),
    race_ethnicity = factor(race_ethnicity),
    education = factor(education)
  )

method <- make.method(impute_frame)
method[] <- ""
method["poverty_ratio"] <- "pmm"

predictor_matrix <- make.predictorMatrix(impute_frame)
predictor_matrix[,] <- 0
predictor_matrix["poverty_ratio", setdiff(names(impute_frame), "poverty_ratio")] <- 1

imputed <- mice(
  impute_frame,
  m = 1,
  maxit = 5,
  method = method,
  predictorMatrix = predictor_matrix,
  printFlag = FALSE,
  seed = 20260716
)

completed_imputation <- complete(imputed, 1)

model_data <- model_base %>%
  mutate(
    poverty_ratio_missing = if_else(is.na(poverty_ratio), "Missing income ratio", "Observed income ratio"),
    poverty_ratio_imputed = completed_imputation$poverty_ratio,
    fair_poor = as.numeric(fair_poor),
    insurance_status = relevel(factor(insurance_status), ref = "Insured"),
    usual_care = relevel(factor(usual_care), ref = "Has a usual place"),
    age_group = relevel(factor(age_group), ref = "18-34"),
    gender = relevel(factor(gender), ref = "Female"),
    race_ethnicity = relevel(factor(race_ethnicity), ref = "Non-Hispanic White"),
    education = relevel(factor(education), ref = "College graduate+"),
    poverty_ratio_missing = relevel(factor(poverty_ratio_missing), ref = "Observed income ratio")
  )

survey_design <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~wtint2yr,
  nest = TRUE,
  data = model_data
)

weighted_overall <- svymean(~fair_poor, survey_design, na.rm = TRUE)
overall_estimate <- tibble(
  group_type = "Overall",
  group = "Adults 18+",
  estimate = as.numeric(coef(weighted_overall)),
  se = as.numeric(SE(weighted_overall)),
  ci_low = estimate - 1.96 * se,
  ci_high = estimate + 1.96 * se
)

survey_mean_by <- function(group_var, label) {
  out <- svyby(
    ~fair_poor,
    as.formula(paste0("~", group_var)),
    survey_design,
    svymean,
    na.rm = TRUE,
    vartype = c("se", "ci")
  )
  as_tibble(out) %>%
    rename(group = all_of(group_var), estimate = fair_poor, se = se, ci_low = ci_l, ci_high = ci_u) %>%
    mutate(group_type = label, .before = 1)
}

weighted_estimates <- bind_rows(
  overall_estimate,
  survey_mean_by("insurance_status", "Insurance status"),
  survey_mean_by("usual_care", "Usual source of care"),
  survey_mean_by("age_group", "Age group")
) %>%
  mutate(across(c(estimate, se, ci_low, ci_high), ~ round(100 * .x, 2)))

fit <- svyglm(
  fair_poor ~ insurance_status + usual_care + age_group + gender + race_ethnicity +
    education + poverty_ratio_imputed + poverty_ratio_missing,
  design = survey_design,
  family = quasibinomial()
)

tidy_svyglm <- function(fit, design) {
  df <- degf(design)
  crit <- qt(0.975, df = df)
  tidy(fit, conf.int = FALSE) %>%
    mutate(
      statistic = estimate / std.error,
      p.value = 2 * pt(abs(statistic), df = df, lower.tail = FALSE),
      conf.low = estimate - crit * std.error,
      conf.high = estimate + crit * std.error
    )
}

model_results <- tidy_svyglm(fit, survey_design) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    odds_ratio = exp(estimate),
    conf_low_or = exp(conf.low),
    conf_high_or = exp(conf.high),
    across(c(estimate, std.error, statistic, p.value, conf.low, conf.high, odds_ratio, conf_low_or, conf_high_or), ~ round(.x, 4))
  )

complete_case_data <- model_data %>%
  filter(!is.na(poverty_ratio))

complete_case_design <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~wtint2yr,
  nest = TRUE,
  data = complete_case_data
)

complete_case_fit <- svyglm(
  fair_poor ~ insurance_status + usual_care + age_group + gender + race_ethnicity +
    education + poverty_ratio,
  design = complete_case_design,
  family = quasibinomial()
)

complete_case_results <- tidy_svyglm(complete_case_fit, complete_case_design) %>%
  filter(term %in% c("insurance_statusUninsured", "usual_careNo usual place")) %>%
  mutate(
    odds_ratio = exp(estimate),
    conf_low_or = exp(conf.low),
    conf_high_or = exp(conf.high),
    across(c(estimate, std.error, statistic, p.value, conf.low, conf.high, odds_ratio, conf_low_or, conf_high_or), ~ round(.x, 4))
  )

imputation_summary <- tibble(
  measure = c(
    "Model rows",
    "Rows with observed poverty ratio",
    "Rows with imputed poverty ratio",
    "Observed poverty ratio mean",
    "Imputed poverty ratio mean"
  ),
  value = c(
    nrow(model_data),
    sum(!is.na(model_base$poverty_ratio)),
    sum(is.na(model_base$poverty_ratio)),
    round(mean(model_base$poverty_ratio, na.rm = TRUE), 3),
    round(mean(model_data$poverty_ratio_imputed, na.rm = TRUE), 3)
  )
)

readr::write_csv(analytic, file.path(processed_dir, "nhanes_adult_access_analysis.csv"))
readr::write_csv(variable_dictionary, file.path(output_dir, "variable_dictionary.csv"))
readr::write_csv(quality_summary, file.path(output_dir, "data_quality_summary.csv"))
readr::write_csv(missing_summary, file.path(output_dir, "missingness_summary.csv"))
readr::write_csv(weighted_estimates, file.path(output_dir, "weighted_fair_poor_estimates.csv"))
readr::write_csv(model_results, file.path(output_dir, "survey_weighted_model_results.csv"))
readr::write_csv(complete_case_results, file.path(output_dir, "complete_case_sensitivity_results.csv"))
readr::write_csv(imputation_summary, file.path(output_dir, "imputation_summary.csv"))

fig_prev <- weighted_estimates %>%
  filter(group_type %in% c("Insurance status", "Usual source of care")) %>%
  ggplot(aes(x = group, y = estimate, ymin = ci_low, ymax = ci_high, fill = group_type)) +
  geom_col(width = 0.65, color = "grey30") +
  geom_errorbar(width = 0.16) +
  facet_wrap(~group_type, scales = "free_x") +
  labs(
    x = NULL,
    y = "Weighted percent fair or poor health",
    title = "Fair or poor self-rated health by access measure"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

ggsave(file.path(figure_dir, "weighted_prevalence_by_access.png"), fig_prev, width = 7, height = 4.5, dpi = 300)

fig_missing <- missing_summary %>%
  mutate(variable = fct_reorder(variable, missing_pct)) %>%
  ggplot(aes(x = missing_pct, y = variable)) +
  geom_col(fill = "#5B8C85") +
  labs(
    x = "Missing percent",
    y = NULL,
    title = "Missingness in selected analytic fields"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(figure_dir, "analytic_missingness.png"), fig_missing, width = 7, height = 4, dpi = 300)

fig_model <- model_results %>%
  filter(term %in% c("insurance_statusUninsured", "usual_careNo usual place", "poverty_ratio_imputed")) %>%
  mutate(
    label = recode(
      term,
      "insurance_statusUninsured" = "Uninsured",
      "usual_careNo usual place" = "No usual place for care",
      "poverty_ratio_imputed" = "Family income to poverty ratio"
    )
  ) %>%
  ggplot(aes(x = odds_ratio, y = fct_rev(label))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = conf_low_or, xmax = conf_high_or), height = 0.18, color = "grey30") +
  geom_point(size = 2.5, color = "#2C5871") +
  scale_x_log10() +
  labs(
    x = "Adjusted odds ratio, log scale",
    y = NULL,
    title = "Survey-weighted model estimates"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(figure_dir, "model_odds_ratios.png"), fig_model, width = 7, height = 3.8, dpi = 300)

message("Analysis complete.")
message("Analytic rows: ", nrow(analytic))
message("Model rows: ", nrow(model_data))
