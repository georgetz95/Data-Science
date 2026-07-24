library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(stringr)
library(scales)

project_dir <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(project_dir, "data", "raw")
processed_dir <- file.path(project_dir, "data", "processed")
output_dir <- file.path(project_dir, "outputs")
figure_dir <- file.path(project_dir, "figures")
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

clean_count <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  ifelse(is.na(x_num) | x_num < 0, NA_real_, x_num)
}

race_labels <- c(
  `1` = "White",
  `2` = "Black",
  `3` = "Hispanic",
  `4` = "Asian",
  `5` = "American Indian/Alaska Native",
  `6` = "Native Hawaiian/Pacific Islander",
  `7` = "Two or more races",
  `99` = "Total"
)

zscore <- function(x) {
  if (all(is.na(x)) || sd(x, na.rm = TRUE) == 0) {
    rep(0, length(x))
  } else {
    as.numeric(scale(x))
  }
}

directory <- read_csv(file.path(raw_dir, "ccd_directory_2018_2024_maine207.csv"), show_col_types = FALSE) %>%
  mutate(
    enrollment = clean_count(enrollment),
    teachers_fte = clean_count(teachers_fte),
    free_or_reduced_price_lunch = clean_count(free_or_reduced_price_lunch),
    frpl_share = if_else(enrollment > 0, free_or_reduced_price_lunch / enrollment, NA_real_),
    school_label = str_replace_all(school_name, " High School", " HS")
  )

school_lookup <- directory %>%
  filter(year == max(year)) %>%
  select(ncessch, school_name, school_label, leaid, lea_name) %>%
  distinct()

directory_latest <- directory %>%
  filter(year == max(year), enrollment > 0) %>%
  select(year, ncessch, school_name, school_label, enrollment, teachers_fte, frpl_share)

write_csv(directory, file.path(processed_dir, "school_directory_2018_2024.csv"))
write_csv(directory_latest, file.path(processed_dir, "school_directory_latest.csv"))

chronic <- read_csv(file.path(raw_dir, "crdc_chronic_absenteeism_2020_2021_maine207.csv"), show_col_types = FALSE) %>%
  mutate(
    students_chronically_absent = clean_count(students_chronically_absent),
    race_label = recode(as.character(race), !!!race_labels, .default = "Unknown race")
  )

enrollment_2021 <- read_csv(file.path(raw_dir, "crdc_enrollment_2021_maine207.csv"), show_col_types = FALSE) %>%
  mutate(enrollment_crdc = clean_count(enrollment_crdc))

chronic_totals <- chronic %>%
  filter(race == 99, sex == 99, disability == 99, lep == 99, homeless == 99) %>%
  left_join(
    directory %>% select(year, ncessch, school_name, school_label, enrollment, frpl_share),
    by = c("year", "ncessch")
  ) %>%
  mutate(
    chronic_absence_rate = if_else(enrollment > 0, students_chronically_absent / enrollment, NA_real_)
  ) %>%
  arrange(year, school_name)

write_csv(chronic_totals, file.path(processed_dir, "chronic_absenteeism_school_trend.csv"))

subgroup_absence <- chronic %>%
  filter(year == 2021, race != 99, sex == 99, disability == 99, lep == 99, homeless == 99) %>%
  select(ncessch, race, race_label, students_chronically_absent) %>%
  left_join(
    enrollment_2021 %>%
      filter(race != 99, sex == 99, disability == 99, lep == 99) %>%
      select(ncessch, race, enrollment_crdc),
    by = c("ncessch", "race")
  ) %>%
  left_join(school_lookup %>% select(ncessch, school_name, school_label), by = "ncessch") %>%
  mutate(
    chronic_absence_rate = if_else(enrollment_crdc > 0, students_chronically_absent / enrollment_crdc, NA_real_),
    stable_reporting_group = enrollment_crdc >= 20
  ) %>%
  arrange(school_name, desc(chronic_absence_rate))

write_csv(subgroup_absence, file.path(processed_dir, "subgroup_chronic_absenteeism_2021.csv"))

discipline <- read_csv(file.path(raw_dir, "crdc_discipline_instances_2020_2021_maine207.csv"), show_col_types = FALSE) %>%
  mutate(suspensions_instances = clean_count(suspensions_instances)) %>%
  filter(disability == 99) %>%
  left_join(
    directory %>% select(year, ncessch, school_name, school_label, enrollment),
    by = c("year", "ncessch")
  ) %>%
  mutate(suspension_instances_per_100 = if_else(enrollment > 0, 100 * suspensions_instances / enrollment, NA_real_)) %>%
  arrange(year, school_name)

write_csv(discipline, file.path(processed_dir, "discipline_instances_school_trend.csv"))

mtss_priority <- chronic_totals %>%
  filter(year == 2021) %>%
  select(ncessch, school_name, school_label, enrollment, frpl_share, chronic_absence_rate) %>%
  left_join(
    discipline %>% filter(year == 2021) %>% select(ncessch, suspension_instances_per_100),
    by = "ncessch"
  ) %>%
  mutate(
    chronic_component = zscore(chronic_absence_rate),
    behavior_component = zscore(suspension_instances_per_100),
    context_component = zscore(frpl_share),
    mtss_priority_score = 0.50 * chronic_component + 0.30 * behavior_component + 0.20 * context_component,
    priority_rank = dense_rank(desc(mtss_priority_score)),
    dashboard_action = case_when(
      priority_rank == 1 ~ "Highest priority for attendance and behavior support review",
      priority_rank <= 3 ~ "Review subgroup attendance patterns and tiered support capacity",
      TRUE ~ "Monitor trend and maintain routine dashboard review"
    )
  ) %>%
  arrange(priority_rank)

write_csv(mtss_priority, file.path(processed_dir, "mtss_priority_school_table.csv"))

dashboard_kpis <- tibble(
  metric = c(
    "Schools with positive enrollment in latest CCD year",
    "Latest CCD enrollment across active District 207 schools",
    "District 2021 chronic absenteeism rate",
    "District 2021 suspension instances per 100 students",
    "Stable race subgroup rows in 2021 review",
    "Dashboard-ready processed tables"
  ),
  value = c(
    sum(directory_latest$enrollment > 0, na.rm = TRUE),
    sum(directory_latest$enrollment, na.rm = TRUE),
    sum(chronic_totals$students_chronically_absent[chronic_totals$year == 2021], na.rm = TRUE) /
      sum(chronic_totals$enrollment[chronic_totals$year == 2021], na.rm = TRUE),
    100 * sum(discipline$suspensions_instances[discipline$year == 2021], na.rm = TRUE) /
      sum(discipline$enrollment[discipline$year == 2021], na.rm = TRUE),
    sum(subgroup_absence$stable_reporting_group, na.rm = TRUE),
    5
  ),
  display_value = c(
    as.character(sum(directory_latest$enrollment > 0, na.rm = TRUE)),
    comma(sum(directory_latest$enrollment, na.rm = TRUE)),
    percent(sum(chronic_totals$students_chronically_absent[chronic_totals$year == 2021], na.rm = TRUE) /
              sum(chronic_totals$enrollment[chronic_totals$year == 2021], na.rm = TRUE), accuracy = 0.1),
    number(100 * sum(discipline$suspensions_instances[discipline$year == 2021], na.rm = TRUE) /
             sum(discipline$enrollment[discipline$year == 2021], na.rm = TRUE), accuracy = 0.1),
    as.character(sum(subgroup_absence$stable_reporting_group, na.rm = TRUE)),
    "5"
  )
)
write_csv(dashboard_kpis, file.path(processed_dir, "dashboard_kpis.csv"))

data_quality_checks <- tibble(
  check = c(
    "ccd_directory_rows",
    "crdc_chronic_rows",
    "crdc_enrollment_2021_rows",
    "discipline_total_rows",
    "active_latest_schools",
    "missing_school_names_in_chronic",
    "missing_chronic_rate_total_rows",
    "small_subgroup_rows_suppressed_from_main_equity_chart"
  ),
  value = c(
    nrow(directory),
    nrow(chronic),
    nrow(enrollment_2021),
    nrow(discipline),
    nrow(directory_latest),
    sum(is.na(chronic_totals$school_name)),
    sum(is.na(chronic_totals$chronic_absence_rate)),
    sum(!subgroup_absence$stable_reporting_group, na.rm = TRUE)
  )
)
write_csv(data_quality_checks, file.path(processed_dir, "data_quality_checks.csv"))
write_csv(data_quality_checks, file.path(output_dir, "data_quality_checks.csv"))

ggplot(directory %>% filter(enrollment > 0), aes(year, enrollment, color = school_label)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.8) +
  scale_y_continuous(labels = comma) +
  labs(x = NULL, y = "Enrollment", color = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
ggsave(file.path(figure_dir, "enrollment_trend_by_school.png"), width = 7.5, height = 4.8, dpi = 300)

ggplot(chronic_totals %>% filter(enrollment > 0), aes(factor(year), chronic_absence_rate, fill = school_label)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(x = NULL, y = "Chronically absent students / enrollment", fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
ggsave(file.path(figure_dir, "chronic_absence_school_trend.png"), width = 7.5, height = 4.8, dpi = 300)

subgroup_plot <- subgroup_absence %>%
  filter(stable_reporting_group, !is.na(chronic_absence_rate)) %>%
  mutate(race_label = factor(race_label, levels = unique(race_label[order(-chronic_absence_rate)])))

ggplot(subgroup_plot, aes(chronic_absence_rate, race_label, fill = school_label)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(x = "2021 chronic absenteeism rate", y = NULL, fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
ggsave(file.path(figure_dir, "subgroup_chronic_absence_2021.png"), width = 7.5, height = 4.8, dpi = 300)

ggplot(discipline %>% filter(enrollment > 0), aes(factor(year), suspension_instances_per_100, fill = school_label)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  labs(x = NULL, y = "Suspension instances per 100 students", fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
ggsave(file.path(figure_dir, "suspension_instances_per_100.png"), width = 7.5, height = 4.8, dpi = 300)

ggplot(mtss_priority, aes(reorder(school_label, mtss_priority_score), mtss_priority_score, fill = chronic_absence_rate)) +
  geom_col(width = 0.65) +
  coord_flip() +
  scale_fill_gradient(low = "#6aa6a4", high = "#c46a4a", labels = percent_format(accuracy = 1)) +
  labs(x = NULL, y = "Composite support-priority score", fill = "Chronic absence") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
ggsave(file.path(figure_dir, "mtss_priority_score.png"), width = 7.5, height = 4.8, dpi = 300)

summary_lines <- c(
  paste0("Active latest-year schools: ", nrow(directory_latest)),
  paste0("Latest active enrollment: ", comma(sum(directory_latest$enrollment, na.rm = TRUE))),
  paste0("District 2021 chronic absenteeism rate: ", dashboard_kpis$display_value[dashboard_kpis$metric == "District 2021 chronic absenteeism rate"]),
  paste0("District 2021 suspension instances per 100 students: ", dashboard_kpis$display_value[dashboard_kpis$metric == "District 2021 suspension instances per 100 students"]),
  paste0("Highest priority school in 2021 composite table: ", mtss_priority$school_name[1])
)
writeLines(summary_lines, file.path(output_dir, "analysis_summary.txt"))

message("Saved processed tables, figures, and summary outputs.")
