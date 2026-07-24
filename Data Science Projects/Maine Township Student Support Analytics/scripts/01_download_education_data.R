library(jsonlite)
library(dplyr)
library(readr)
library(purrr)

project_dir <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(project_dir, "data", "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

base_url <- "https://educationdata.urban.org/api/v1"
target_leaid <- "1724090"

fetch_endpoint <- function(url) {
  pages <- list()
  next_url <- url
  while (!is.null(next_url) && !is.na(next_url) && nzchar(next_url)) {
    message("Fetching: ", next_url)
    page <- fromJSON(next_url, flatten = TRUE)
    pages[[length(pages) + 1]] <- page$results
    next_url <- page$`next`
  }
  bind_rows(pages)
}

write_endpoint <- function(url, csv_name) {
  dat <- fetch_endpoint(url)
  write_csv(dat, file.path(raw_dir, csv_name))
  dat
}

directory_years <- 2018:2024
directory <- map_dfr(directory_years, function(yr) {
  url <- paste0(base_url, "/schools/ccd/directory/", yr, "/?fips=17")
  dat <- fetch_endpoint(url) %>%
    filter(leaid == target_leaid)
  write_csv(dat, file.path(raw_dir, paste0("ccd_directory_", yr, "_maine207.csv")))
  dat
})
write_csv(directory, file.path(raw_dir, "ccd_directory_2018_2024_maine207.csv"))

crdc_years <- c(2020, 2021)

chronic_absenteeism <- map_dfr(crdc_years, function(yr) {
  url <- paste0(base_url, "/schools/crdc/chronic-absenteeism/", yr, "/race/sex/?leaid=", target_leaid)
  dat <- fetch_endpoint(url)
  write_csv(dat, file.path(raw_dir, paste0("crdc_chronic_absenteeism_", yr, "_maine207.csv")))
  dat
})
write_csv(chronic_absenteeism, file.path(raw_dir, "crdc_chronic_absenteeism_2020_2021_maine207.csv"))

enrollment_2021 <- write_endpoint(
  paste0(base_url, "/schools/crdc/enrollment/2021/race/sex/?leaid=", target_leaid),
  "crdc_enrollment_2021_maine207.csv"
)

discipline_instances <- map_dfr(crdc_years, function(yr) {
  url <- paste0(base_url, "/schools/crdc/discipline-instances/", yr, "/?leaid=", target_leaid)
  dat <- fetch_endpoint(url)
  write_csv(dat, file.path(raw_dir, paste0("crdc_discipline_instances_", yr, "_maine207.csv")))
  dat
})
write_csv(discipline_instances, file.path(raw_dir, "crdc_discipline_instances_2020_2021_maine207.csv"))

download_summary <- tibble(
  source = c("CCD directory", "CRDC chronic absenteeism", "CRDC enrollment", "CRDC discipline instances"),
  rows = c(nrow(directory), nrow(chronic_absenteeism), nrow(enrollment_2021), nrow(discipline_instances)),
  years = c("2018-2024", "2020-2021", "2021", "2020-2021")
)
write_csv(download_summary, file.path(raw_dir, "download_summary.csv"))

message("Saved raw education data to: ", raw_dir)
