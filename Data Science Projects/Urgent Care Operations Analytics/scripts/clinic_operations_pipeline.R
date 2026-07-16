suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(scales)
})

options(timeout = 300)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", file_arg), winslash = "/", mustWork = TRUE)
  project_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
} else {
  project_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
}

raw_dir <- file.path(project_dir, "data", "raw")
processed_dir <- file.path(project_dir, "data", "processed")
output_dir <- file.path(project_dir, "outputs")
table_dir <- file.path(output_dir, "tables")
figure_dir <- file.path(project_dir, "figures")
sql_dir <- file.path(project_dir, "sql")

dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

cms_datasets <- c(
  `2023` = "0e9f2f2b-7bf9-451a-912c-e02e654dd725",
  `2024` = "335e5f35-eca6-482d-87b3-f99883e213e3"
)

states <- c("IL", "IN")
hcpcs_codes <- c(
  "99203", "99204", "99213", "99214", "99215",
  "99232", "99285", "93000", "97110"
)

code_family_lookup <- tibble::tribble(
  ~hcpcs_code, ~service_family, ~operations_group,
  "99203", "New patient office visit", "Core visits",
  "99204", "New patient office visit", "Core visits",
  "99213", "Established patient office visit", "Core visits",
  "99214", "Established patient office visit", "Core visits",
  "99215", "Established patient office visit", "Core visits",
  "99232", "Subsequent hospital or follow-up care", "Follow-up and acuity",
  "99285", "Emergency or high-acuity evaluation", "Follow-up and acuity",
  "93000", "ECG and diagnostic testing", "Diagnostics and procedures",
  "97110", "Therapeutic procedure", "Diagnostics and procedures"
)

column_map <- c(
  Rndrng_NPI = "rendering_npi",
  Rndrng_Prvdr_Last_Org_Name = "provider_last_or_org_name",
  Rndrng_Prvdr_First_Name = "provider_first_name",
  Rndrng_Prvdr_MI = "provider_middle_initial",
  Rndrng_Prvdr_Crdntls = "provider_credentials",
  Rndrng_Prvdr_Ent_Cd = "provider_entity_type",
  Rndrng_Prvdr_City = "provider_city",
  Rndrng_Prvdr_State_Abrvtn = "provider_state",
  Rndrng_Prvdr_Zip5 = "provider_zip5",
  Rndrng_Prvdr_RUCA = "provider_ruca",
  Rndrng_Prvdr_RUCA_Desc = "provider_ruca_description",
  Rndrng_Prvdr_Cntry = "provider_country",
  Rndrng_Prvdr_Type = "provider_type",
  Rndrng_Prvdr_Mdcr_Prtcptg_Ind = "medicare_participating",
  HCPCS_Cd = "hcpcs_code",
  HCPCS_Desc = "hcpcs_description",
  HCPCS_Drug_Ind = "hcpcs_drug_indicator",
  Place_Of_Srvc = "place_of_service",
  Tot_Benes = "total_beneficiaries",
  Tot_Srvcs = "total_services",
  Tot_Bene_Day_Srvcs = "total_beneficiary_day_services",
  Avg_Sbmtd_Chrg = "average_submitted_charge",
  Avg_Mdcr_Alowd_Amt = "average_medicare_allowed",
  Avg_Mdcr_Pymt_Amt = "average_medicare_payment",
  Avg_Mdcr_Stdzd_Amt = "average_medicare_standardized"
)

numeric_cols <- c(
  "total_beneficiaries", "total_services", "total_beneficiary_day_services",
  "average_submitted_charge", "average_medicare_allowed",
  "average_medicare_payment", "average_medicare_standardized"
)

cms_url <- function(dataset_id, params) {
  query <- paste0(
    utils::URLencode(names(params), reserved = TRUE),
    "=",
    utils::URLencode(as.character(params), reserved = TRUE),
    collapse = "&"
  )
  paste0("https://data.cms.gov/data-api/v1/dataset/", dataset_id, "/data?", query)
}

fetch_cms_slice <- function(year, state, hcpcs_code, page_size = 5000) {
  raw_file <- file.path(raw_dir, paste0("cms_provider_service_", year, "_", state, "_", hcpcs_code, ".csv"))
  if (file.exists(raw_file)) {
    return(read_csv(raw_file, show_col_types = FALSE, col_types = cols(.default = col_character())))
  }

  all_rows <- list()
  offset <- 0
  page_index <- 1

  repeat {
    params <- list(
      size = page_size,
      offset = offset,
      `filter[Rndrng_Prvdr_State_Abrvtn]` = state,
      `filter[HCPCS_Cd]` = hcpcs_code
    )
    dataset_id <- unname(cms_datasets[as.character(year)])
    payload <- jsonlite::fromJSON(cms_url(dataset_id, params), flatten = TRUE)
    page <- if (is.data.frame(payload)) payload else if ("value" %in% names(payload)) payload$value else data.frame()
    if (nrow(page) == 0) break

    all_rows[[page_index]] <- page
    if (nrow(page) < page_size) break

    offset <- offset + page_size
    page_index <- page_index + 1
    Sys.sleep(0.15)
  }

  if (length(all_rows) == 0) {
    result <- tibble::tibble()
  } else {
    result <- bind_rows(all_rows)
  }
  write_csv(result, raw_file)
  result
}

download_data <- function() {
  local_cms_extract <- file.path(dirname(project_dir), "Healthcare_Claims_Audit_Targeting", "data", "processed", "provider_service_claims_clean.csv")
  if (file.exists(local_cms_extract)) {
    message("Using local CMS provider-service extract: ", local_cms_extract)
    raw <- read_csv(local_cms_extract, show_col_types = FALSE)
    quality_df <- raw %>%
      count(year, provider_state, hcpcs_code, name = "rows_downloaded") %>%
      rename(state = provider_state) %>%
      mutate(source_dataset_id = unname(cms_datasets[as.character(year)]))
    write_csv(quality_df, file.path(table_dir, "download_quality_summary.csv"))
    return(list(raw = raw, quality = quality_df))
  }

  frames <- list()
  quality <- list()
  row_id <- 1

  for (year in names(cms_datasets)) {
    for (state in states) {
      for (hcpcs_code in hcpcs_codes) {
        message("Downloading ", year, " ", state, " ", hcpcs_code)
        page <- fetch_cms_slice(as.integer(year), state, hcpcs_code)
        quality[[row_id]] <- tibble::tibble(
          year = as.integer(year),
          state = state,
          hcpcs_code = hcpcs_code,
          rows_downloaded = nrow(page),
          source_dataset_id = unname(cms_datasets[year])
        )
        if (nrow(page) > 0) {
          page$year <- as.integer(year)
          page$extract_state <- state
          page$extract_hcpcs_code <- hcpcs_code
          frames[[length(frames) + 1]] <- page
        }
        row_id <- row_id + 1
      }
    }
  }

  quality_df <- bind_rows(quality)
  write_csv(quality_df, file.path(table_dir, "download_quality_summary.csv"))
  list(raw = bind_rows(frames), quality = quality_df)
}

clean_data <- function(raw) {
  claims <- raw
  shared <- intersect(names(column_map), names(claims))
  names(claims)[match(shared, names(claims))] <- unname(column_map[shared])

  keep_cols <- c(unname(column_map), "year", "extract_state", "extract_hcpcs_code")
  claims <- claims[, intersect(keep_cols, names(claims))]

  for (col in c("rendering_npi", "provider_zip5", "hcpcs_code", "extract_hcpcs_code")) {
    if (col %in% names(claims)) claims[[col]] <- as.character(claims[[col]])
  }

  for (col in numeric_cols) {
    claims[[col]] <- suppressWarnings(as.numeric(claims[[col]]))
  }

  claims <- claims %>%
    left_join(code_family_lookup, by = "hcpcs_code") %>%
    mutate(
      provider_city = toupper(trimws(provider_city)),
      provider_name = if_else(
        provider_entity_type == "O",
        coalesce(provider_last_or_org_name, ""),
        trimws(paste0(coalesce(provider_last_or_org_name, ""), ", ", coalesce(provider_first_name, "")))
      ),
      place_of_service_label = recode(place_of_service, F = "Facility", O = "Office", .default = "Other"),
      estimated_submitted_charge = total_services * average_submitted_charge,
      estimated_allowed_amount = total_services * average_medicare_allowed,
      estimated_medicare_payment = total_services * average_medicare_payment,
      estimated_standardized_payment = total_services * average_medicare_standardized,
      services_per_beneficiary = total_services / if_else(total_beneficiaries == 0, NA_real_, total_beneficiaries),
      payment_per_beneficiary = estimated_medicare_payment / if_else(total_beneficiaries == 0, NA_real_, total_beneficiaries),
      charge_to_allowed_ratio = average_submitted_charge / if_else(average_medicare_allowed == 0, NA_real_, average_medicare_allowed),
      payment_per_service = estimated_medicare_payment / if_else(total_services == 0, NA_real_, total_services),
      market = paste(provider_city, provider_state, sep = ", ")
    ) %>%
    filter(
      !is.na(rendering_npi),
      !is.na(provider_state),
      !is.na(hcpcs_code),
      !is.na(total_services),
      !is.na(estimated_medicare_payment)
    ) %>%
    distinct(year, rendering_npi, provider_state, provider_city, hcpcs_code, place_of_service, .keep_all = TRUE)

  write_csv(claims, file.path(processed_dir, "clinic_operations_clean.csv"))
  claims
}

rank_pct <- function(x) {
  if (all(is.na(x)) || length(x) == 1) return(rep(0.5, length(x)))
  percent_rank(x)
}

build_scores <- function(claims) {
  provider_code <- claims %>%
    group_by(year, provider_state, rendering_npi, hcpcs_code) %>%
    summarise(
      provider_code_services = sum(total_services, na.rm = TRUE),
      provider_code_payment = sum(estimated_medicare_payment, na.rm = TRUE),
      .groups = "drop"
    )

  prior <- provider_code %>%
    mutate(year = year + 1) %>%
    rename(
      prior_provider_code_services = provider_code_services,
      prior_provider_code_payment = provider_code_payment
    )

  scored <- claims %>%
    left_join(prior, by = c("year", "provider_state", "rendering_npi", "hcpcs_code")) %>%
    mutate(
      service_growth_rate = (total_services - prior_provider_code_services) / if_else(prior_provider_code_services == 0, NA_real_, prior_provider_code_services),
      payment_growth_rate = (estimated_medicare_payment - prior_provider_code_payment) / if_else(prior_provider_code_payment == 0, NA_real_, prior_provider_code_payment)
    ) %>%
    group_by(year, provider_state) %>%
    mutate(
      volume_percentile = rank_pct(total_services),
      payment_percentile = rank_pct(estimated_medicare_payment),
      payment_per_service_percentile = rank_pct(payment_per_service)
    ) %>%
    ungroup() %>%
    group_by(year, provider_state, hcpcs_code, place_of_service_label) %>%
    mutate(
      service_intensity_percentile = rank_pct(services_per_beneficiary),
      charge_ratio_percentile = rank_pct(charge_to_allowed_ratio)
    ) %>%
    ungroup() %>%
    group_by(year, provider_state, operations_group) %>%
    mutate(
      growth_percentile = rank_pct(coalesce(service_growth_rate, 0))
    ) %>%
    ungroup() %>%
    mutate(
      high_volume_flag = as.integer(coalesce(volume_percentile >= 0.90, FALSE)),
      high_payment_flag = as.integer(coalesce(payment_percentile >= 0.90, FALSE)),
      high_intensity_flag = as.integer(coalesce(service_intensity_percentile >= 0.90 & services_per_beneficiary > 1.25, FALSE)),
      high_charge_ratio_flag = as.integer(coalesce(charge_ratio_percentile >= 0.90 & charge_to_allowed_ratio > 2, FALSE)),
      rapid_growth_flag = as.integer(coalesce(year == 2024 & service_growth_rate > 0.30 & payment_growth_rate > 0.30, FALSE)),
      operational_priority_score =
        30 * coalesce(volume_percentile, 0) +
        25 * coalesce(payment_percentile, 0) +
        15 * coalesce(service_intensity_percentile, 0) +
        10 * coalesce(charge_ratio_percentile, 0) +
        10 * coalesce(payment_per_service_percentile, 0) +
        10 * coalesce(growth_percentile, 0) +
        3 * high_volume_flag +
        3 * high_intensity_flag +
        3 * rapid_growth_flag
    ) %>%
    group_by(year) %>%
    mutate(operational_priority_decile = pmin(10L, pmax(1L, ceiling(rank(operational_priority_score, ties.method = "first") / n() * 10)))) %>%
    ungroup()

  write_csv(scored, file.path(processed_dir, "clinic_operations_scored_records.csv"))
  scored
}

write_tables <- function(scored, quality) {
  latest <- scored %>% filter(year == 2024)

  service_family_summary <- scored %>%
    group_by(year, operations_group, service_family) %>%
    summarise(
      records = n(),
      providers = n_distinct(rendering_npi),
      markets = n_distinct(market),
      services = sum(total_services, na.rm = TRUE),
      beneficiaries = sum(total_beneficiaries, na.rm = TRUE),
      estimated_payment = sum(estimated_medicare_payment, na.rm = TRUE),
      estimated_allowed = sum(estimated_allowed_amount, na.rm = TRUE),
      median_payment_per_service = median(payment_per_service, na.rm = TRUE),
      median_charge_to_allowed_ratio = median(charge_to_allowed_ratio, na.rm = TRUE),
      .groups = "drop"
    )

  code_summary <- scored %>%
    group_by(year, hcpcs_code, hcpcs_description, operations_group, service_family) %>%
    summarise(
      records = n(),
      providers = n_distinct(rendering_npi),
      services = sum(total_services, na.rm = TRUE),
      estimated_payment = sum(estimated_medicare_payment, na.rm = TRUE),
      median_services_per_beneficiary = median(services_per_beneficiary, na.rm = TRUE),
      median_payment_per_service = median(payment_per_service, na.rm = TRUE),
      high_volume_flags = sum(high_volume_flag, na.rm = TRUE),
      rapid_growth_flags = sum(rapid_growth_flag, na.rm = TRUE),
      .groups = "drop"
    )

  city_market_summary <- latest %>%
    group_by(provider_state, provider_city, market) %>%
    summarise(
      records = n(),
      providers = n_distinct(rendering_npi),
      service_families = n_distinct(service_family),
      services = sum(total_services, na.rm = TRUE),
      estimated_payment = sum(estimated_medicare_payment, na.rm = TRUE),
      median_charge_to_allowed_ratio = median(charge_to_allowed_ratio, na.rm = TRUE),
      top_priority_records = sum(operational_priority_decile == 10, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(services))

  provider_type_summary <- latest %>%
    group_by(provider_type) %>%
    summarise(
      records = n(),
      providers = n_distinct(rendering_npi),
      services = sum(total_services, na.rm = TRUE),
      estimated_payment = sum(estimated_medicare_payment, na.rm = TRUE),
      high_volume_flags = sum(high_volume_flag, na.rm = TRUE),
      rapid_growth_flags = sum(rapid_growth_flag, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(services))

  priority_deciles <- latest %>%
    group_by(operational_priority_decile) %>%
    summarise(
      records = n(),
      providers = n_distinct(rendering_npi),
      services = sum(total_services, na.rm = TRUE),
      estimated_payment = sum(estimated_medicare_payment, na.rm = TRUE),
      high_volume_flags = sum(high_volume_flag, na.rm = TRUE),
      high_intensity_flags = sum(high_intensity_flag, na.rm = TRUE),
      rapid_growth_flags = sum(rapid_growth_flag, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      record_share = records / sum(records),
      service_share = services / sum(services),
      payment_share = estimated_payment / sum(estimated_payment)
    )

  top_records <- latest %>%
    arrange(desc(operational_priority_score)) %>%
    select(
      year, rendering_npi, provider_name, provider_state, provider_city, provider_type,
      hcpcs_code, hcpcs_description, operations_group, service_family, place_of_service_label,
      total_services, total_beneficiaries, estimated_medicare_payment,
      services_per_beneficiary, payment_per_service, charge_to_allowed_ratio,
      service_growth_rate, payment_growth_rate, operational_priority_score, operational_priority_decile
    ) %>%
    slice_head(n = 250)

  write_csv(service_family_summary, file.path(table_dir, "service_family_summary.csv"))
  write_csv(code_summary, file.path(table_dir, "code_summary.csv"))
  write_csv(city_market_summary, file.path(table_dir, "city_market_summary_2024.csv"))
  write_csv(provider_type_summary, file.path(table_dir, "provider_type_summary_2024.csv"))
  write_csv(priority_deciles, file.path(table_dir, "operational_priority_decile_summary_2024.csv"))
  write_csv(top_records, file.path(table_dir, "top_operational_attention_records_2024.csv"))

  list(
    service_family_summary = service_family_summary,
    code_summary = code_summary,
    city_market_summary = city_market_summary,
    provider_type_summary = provider_type_summary,
    priority_deciles = priority_deciles,
    top_records = top_records,
    quality = quality
  )
}

create_figures <- function(scored, tables) {
  latest_family <- tables$service_family_summary %>%
    filter(year == 2024) %>%
    arrange(desc(services))

  family_long <- latest_family %>%
    transmute(service_family, services, estimated_payment_m = estimated_payment / 1e6) %>%
    tidyr::pivot_longer(cols = c(services, estimated_payment_m), names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric, services = "Services", estimated_payment_m = "Estimated payment, $M"))

  p1 <- ggplot(family_long, aes(x = reorder(service_family, value), y = value, fill = metric)) +
    geom_col(show.legend = FALSE) +
    coord_flip() +
    facet_wrap(~metric, scales = "free_x") +
    labs(title = "2024 Volume and Payment by Service Family", x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(strip.text = element_text(face = "bold"))
  ggsave(file.path(figure_dir, "service_family_volume_payment_2024.png"), p1, width = 7.4, height = 4.5, dpi = 300)

  deciles <- tables$priority_deciles
  p2 <- ggplot(deciles, aes(x = operational_priority_decile, y = payment_share)) +
    geom_col(fill = "#4f6f52") +
    geom_hline(yintercept = 0.10, linetype = "dashed", color = "#444444") +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_continuous(breaks = 1:10) +
    labs(title = "2024 Payment Share by Operational Priority Decile", x = "Priority decile, low to high", y = "Estimated payment share") +
    theme_minimal(base_size = 10)
  ggsave(file.path(figure_dir, "priority_decile_payment_capture_2024.png"), p2, width = 7.2, height = 4.2, dpi = 300)

  p3 <- scored %>%
    filter(year == 2024, !is.na(service_growth_rate), service_growth_rate > -1, service_growth_rate < 5) %>%
    ggplot(aes(x = service_growth_rate, y = payment_growth_rate, size = estimated_medicare_payment, color = operations_group)) +
    geom_point(alpha = 0.35) +
    geom_vline(xintercept = 0.30, linetype = "dashed", color = "#444444") +
    geom_hline(yintercept = 0.30, linetype = "dashed", color = "#444444") +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_size_continuous(range = c(1, 8), labels = dollar_format()) +
    labs(title = "Provider-Service Growth From 2023 to 2024", x = "Service growth", y = "Payment growth", color = NULL, size = "Payment") +
    theme_minimal(base_size = 10)
  ggsave(file.path(figure_dir, "provider_service_growth_2024.png"), p3, width = 7.4, height = 4.6, dpi = 300)

  top_markets <- tables$city_market_summary %>%
    slice_head(n = 12) %>%
    arrange(services)
  p4 <- ggplot(top_markets, aes(x = reorder(market, services), y = services, fill = provider_state)) +
    geom_col(show.legend = FALSE) +
    coord_flip() +
    scale_y_continuous(labels = comma_format()) +
    labs(title = "Top 2024 Markets by Selected Service Volume", x = NULL, y = "Services") +
    theme_minimal(base_size = 10)
  ggsave(file.path(figure_dir, "top_markets_by_service_volume_2024.png"), p4, width = 7.2, height = 4.5, dpi = 300)

  p5 <- scored %>%
    filter(year == 2024) %>%
    ggplot(aes(x = operational_priority_score)) +
    geom_histogram(bins = 30, fill = "#3d6f8e", color = "white") +
    labs(title = "Distribution of 2024 Operational Priority Scores", x = "Operational priority score", y = "Provider-service records") +
    theme_minimal(base_size = 10)
  ggsave(file.path(figure_dir, "operational_score_distribution_2024.png"), p5, width = 7.2, height = 4.2, dpi = 300)
}

write_sql_files <- function() {
  csv_path <- normalizePath(file.path(processed_dir, "clinic_operations_scored_records.csv"), winslash = "/", mustWork = TRUE)
  db_path <- file.path(processed_dir, "clinic_operations.sqlite")
  if (file.exists(db_path)) file.remove(db_path)

  import_sql <- c(
    ".mode csv",
    paste0(".import \"", csv_path, "\" raw_claims"),
    paste0(".read \"", normalizePath(file.path(sql_dir, "clinic_operations_views.sql"), winslash = "/", mustWork = TRUE), "\"")
  )
  system2("sqlite3", args = shQuote(normalizePath(db_path, winslash = "/", mustWork = FALSE)), input = import_sql)
}

write_summary <- function(scored, tables) {
  latest <- scored %>% filter(year == 2024)
  prior <- scored %>% filter(year == 2023)
  top_decile <- tables$priority_deciles %>% filter(operational_priority_decile == 10)
  diagnostic <- tables$service_family_summary %>%
    filter(year == 2024, operations_group == "Diagnostics and procedures") %>%
    summarise(services = sum(services), payment = sum(estimated_payment), .groups = "drop")
  all_2024 <- tables$service_family_summary %>%
    filter(year == 2024) %>%
    summarise(services = sum(services), payment = sum(estimated_payment), .groups = "drop")
  top_market <- tables$city_market_summary %>% slice_head(n = 1)

  summary <- list(
    analytic_records = nrow(scored),
    records_2024 = nrow(latest),
    states = paste(states, collapse = ", "),
    hcpcs_codes = paste(hcpcs_codes, collapse = ", "),
    providers_2024 = n_distinct(latest$rendering_npi),
    markets_2024 = n_distinct(latest$market),
    services_2024 = sum(latest$total_services, na.rm = TRUE),
    estimated_payment_2024 = sum(latest$estimated_medicare_payment, na.rm = TRUE),
    estimated_payment_growth_2023_2024 = sum(latest$estimated_medicare_payment, na.rm = TRUE) / sum(prior$estimated_medicare_payment, na.rm = TRUE) - 1,
    top_decile_record_share = top_decile$record_share,
    top_decile_service_share = top_decile$service_share,
    top_decile_payment_share = top_decile$payment_share,
    top_decile_high_volume_flags = top_decile$high_volume_flags,
    top_decile_high_intensity_flags = top_decile$high_intensity_flags,
    top_decile_rapid_growth_flags = top_decile$rapid_growth_flags,
    diagnostic_service_share = diagnostic$services / all_2024$services,
    diagnostic_payment_share = diagnostic$payment / all_2024$payment,
    top_market = top_market$market,
    top_market_services = top_market$services,
    top_market_estimated_payment = top_market$estimated_payment,
    downloaded_rows = sum(tables$quality$rows_downloaded)
  )

  write_json(summary, file.path(output_dir, "analysis_summary.json"), pretty = TRUE, auto_unbox = TRUE)

  lines <- c(
    paste0("Analytic records: ", comma(summary$analytic_records)),
    paste0("2024 provider-service records: ", comma(summary$records_2024)),
    paste0("2024 providers: ", comma(summary$providers_2024)),
    paste0("2024 markets: ", comma(summary$markets_2024)),
    paste0("2024 services: ", comma(round(summary$services_2024))),
    paste0("2024 estimated Medicare payment: $", comma(round(summary$estimated_payment_2024 / 1e6, 2)), "M"),
    paste0("Payment growth in selected cohort, 2023-2024: ", percent(summary$estimated_payment_growth_2023_2024, accuracy = 0.1)),
    paste0("Top operational-priority decile record share: ", percent(summary$top_decile_record_share, accuracy = 0.1)),
    paste0("Top operational-priority decile service share: ", percent(summary$top_decile_service_share, accuracy = 0.1)),
    paste0("Top operational-priority decile payment share: ", percent(summary$top_decile_payment_share, accuracy = 0.1)),
    paste0("Diagnostics and procedures service share: ", percent(summary$diagnostic_service_share, accuracy = 0.1)),
    paste0("Top market: ", summary$top_market, " with ", comma(round(summary$top_market_services)), " services")
  )
  writeLines(lines, file.path(output_dir, "analysis_summary.txt"))
}

main <- function() {
  downloaded <- download_data()
  claims <- clean_data(downloaded$raw)
  scored <- build_scores(claims)
  tables <- write_tables(scored, downloaded$quality)
  create_figures(scored, tables)
  write_sql_files()
  write_summary(scored, tables)
  cat(readLines(file.path(output_dir, "analysis_summary.txt")), sep = "\n")
}

main()
