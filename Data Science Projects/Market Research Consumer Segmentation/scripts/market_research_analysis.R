suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(cluster)
  library(broom)
  library(scales)
  library(forcats)
})

# Setup -----------------------------------------------------------------------

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)

if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg[1]), mustWork = TRUE)
  project_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
} else {
  project_dir <- normalizePath(getwd(), mustWork = TRUE)
}

raw_dir <- file.path(project_dir, "data", "raw")
processed_dir <- file.path(project_dir, "data", "processed")
figure_dir <- file.path(project_dir, "figures")
output_dir <- file.path(project_dir, "outputs")
table_dir <- file.path(output_dir, "client_tables")

dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

theme_set(
  theme_minimal(base_size = 12) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      strip.background = element_rect(fill = "white", color = NA)
    )
)

project_colors <- c(
  blue = "#2B6CB0",
  gold = "#C48A1A",
  olive = "#6B8E23",
  pink = "#B83280",
  charcoal = "#2D3748",
  slate = "#718096"
)

price_rank <- c(low = 1, medium = 2, high = 3)

read_csv_clean <- function(file_name) {
  read_csv(
    file.path(raw_dir, file_name),
    na = c("?", "", "NA"),
    locale = locale(encoding = "Latin1"),
    show_col_types = FALSE
  ) |>
    clean_names()
}

most_common <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(NA_character_)
  }
  names(sort(table(x), decreasing = TRUE))[1]
}

pretty_text <- function(x) {
  x |>
    str_replace_all("[_-]", " ") |>
    str_squish() |>
    str_to_title()
}

clean_payment <- function(x) {
  x |>
    str_to_lower() |>
    str_replace_all("[^a-z0-9]", "")
}


# Read data -------------------------------------------------------------------

ratings <- read_csv_clean("rating_final.csv")
profiles <- read_csv_clean("userprofile.csv")
places <- read_csv_clean("geoplaces2.csv")
user_cuisine <- read_csv_clean("usercuisine.csv")
place_cuisine <- read_csv_clean("chefmozcuisine.csv")
user_payment <- read_csv_clean("userpayment.csv")
place_payment <- read_csv_clean("chefmozaccepts.csv")


# Clean and prepare variables -------------------------------------------------

profiles <- profiles |>
  mutate(
    age = 2012 - birth_year,
    age_group = cut(
      age,
      breaks = c(0, 24, 34, 44, Inf),
      labels = c("18-24", "25-34", "35-44", "45+")
    ),
    bmi = weight / (height ^ 2),
    bmi_group = cut(
      bmi,
      breaks = c(0, 18.5, 25, 30, Inf),
      labels = c("underweight", "healthy", "overweight", "obese"),
      right = FALSE
    )
  )

places <- places |>
  mutate(
    price_value = unname(price_rank[price]),
    alcohol_clean = case_when(
      alcohol == "No_Alcohol_Served" ~ "No alcohol",
      alcohol == "Wine-Beer" ~ "Wine or beer",
      alcohol == "Full_Bar" ~ "Full bar",
      TRUE ~ pretty_text(alcohol)
    ),
    concept_family = paste(
      pretty_text(price),
      "price",
      alcohol_clean,
      pretty_text(rambience),
      sep = " | "
    )
  )

cuisine_matches <- user_cuisine |>
  distinct(user_id, rcuisine) |>
  inner_join(
    place_cuisine |> distinct(place_id, rcuisine),
    by = "rcuisine",
    relationship = "many-to-many"
  ) |>
  distinct(user_id, place_id) |>
  mutate(cuisine_match = 1L)

payment_matches <- user_payment |>
  mutate(payment_key = clean_payment(upayment)) |>
  distinct(user_id, payment_key) |>
  inner_join(
    place_payment |>
      mutate(payment_key = clean_payment(rpayment)) |>
      distinct(place_id, payment_key),
    by = "payment_key",
    relationship = "many-to-many"
  ) |>
  distinct(user_id, place_id) |>
  mutate(payment_match = 1L)


# Segment consumers -----------------------------------------------------------

segment_input <- profiles |>
  select(
    user_id,
    smoker,
    drink_level,
    dress_preference,
    ambience,
    transport,
    marital_status,
    hijos,
    age_group,
    interest,
    personality,
    activity,
    budget,
    bmi_group
  )

gower_input <- segment_input |>
  select(-user_id) |>
  mutate(across(everything(), function(x) factor(replace_na(as.character(x), "missing"))))

gower_distance <- daisy(gower_input, metric = "gower")

candidate_rows <- list()

for (k in 3:5) {
  pam_candidate <- pam(gower_distance, k = k, diss = TRUE)

  candidate_rows[[as.character(k)]] <- tibble(
    k = k,
    avg_silhouette = unname(pam_candidate$silinfo$avg.width),
    min_cluster_size = min(pam_candidate$clusinfo[, "size"])
  )
}

cluster_candidates <- bind_rows(candidate_rows)

selected_k <- cluster_candidates |>
  filter(min_cluster_size >= 10) |>
  arrange(desc(avg_silhouette), k) |>
  slice(1) |>
  pull(k)

if (length(selected_k) == 0) {
  selected_k <- cluster_candidates |>
    arrange(desc(avg_silhouette), k) |>
    slice(1) |>
    pull(k)
}

final_cluster <- pam(gower_distance, k = selected_k, diss = TRUE)

customer_segments <- profiles |>
  mutate(segment = factor(final_cluster$clustering))

user_rating_summary <- ratings |>
  mutate(high_rating = as.integer(rating == 2)) |>
  group_by(user_id) |>
  summarise(
    n_ratings = n(),
    avg_rating = mean(rating, na.rm = TRUE),
    high_ratings = sum(high_rating, na.rm = TRUE),
    high_rating_share = mean(high_rating, na.rm = TRUE),
    .groups = "drop"
  )

segment_profiles <- customer_segments |>
  left_join(user_rating_summary, by = "user_id") |>
  group_by(segment) |>
  summarise(
    customers = n(),
    ratings = sum(replace_na(n_ratings, 0)),
    median_age = median(age, na.rm = TRUE),
    avg_rating = weighted.mean(avg_rating, w = pmax(n_ratings, 1), na.rm = TRUE),
    high_rating_share = sum(replace_na(high_ratings, 0)) / sum(replace_na(n_ratings, 0)),
    low_budget_share = mean(budget == "low", na.rm = TRUE),
    medium_or_high_budget_share = mean(budget %in% c("medium", "high"), na.rm = TRUE),
    student_share = mean(activity == "student", na.rm = TRUE),
    professional_share = mean(activity == "professional", na.rm = TRUE),
    car_owner_share = mean(transport == "car owner", na.rm = TRUE),
    social_drinker_share = mean(drink_level %in% c("social drinker", "casual drinker"), na.rm = TRUE),
    top_budget = most_common(budget),
    top_activity = most_common(activity),
    top_interest = most_common(interest),
    top_personality = most_common(personality),
    .groups = "drop"
  ) |>
  mutate(
    segment_label = case_when(
      car_owner_share >= 0.60 & social_drinker_share >= 0.70 ~
        paste0("S", segment, ": Mobile social students"),
      student_share >= 0.90 & top_interest == "technology" ~
        paste0("S", segment, ": Tech-focused students"),
      social_drinker_share < 0.50 ~
        paste0("S", segment, ": Quieter mixed students"),
      TRUE ~ paste0("S", segment, ": ", pretty_text(top_interest), " students")
    )
  )

customer_segments <- customer_segments |>
  left_join(segment_profiles |> select(segment, segment_label), by = "segment")


# Build the rating-level analysis file ---------------------------------------

ratings_modeling <- ratings |>
  mutate(high_rating = as.integer(rating == 2)) |>
  left_join(customer_segments, by = "user_id") |>
  left_join(
    places |>
      select(
        place_id,
        name,
        city,
        state,
        alcohol,
        alcohol_clean,
        smoking_area,
        dress_code,
        accessibility,
        price,
        price_value,
        rambience,
        franchise,
        area,
        other_services,
        concept_family
      ),
    by = "place_id"
  ) |>
  left_join(cuisine_matches, by = c("user_id", "place_id")) |>
  left_join(payment_matches, by = c("user_id", "place_id")) |>
  mutate(
    cuisine_match = replace_na(cuisine_match, 0L),
    payment_match = replace_na(payment_match, 0L),
    budget_value = unname(price_rank[budget]),
    price_at_or_below_budget = as.integer(price_value <= budget_value),
    budget_price_fit = case_when(
      is.na(price_value) | is.na(budget_value) ~ NA_character_,
      price_value < budget_value ~ "below stated budget",
      price_value == budget_value ~ "matches stated budget",
      price_value > budget_value ~ "above stated budget"
    ),
    segment_label = replace_na(segment_label, "Unsegmented")
  )

model_data <- ratings_modeling |>
  mutate(
    price = fct_na_value_to_level(factor(price), level = "missing"),
    alcohol_clean = fct_lump_min(factor(alcohol_clean), min = 40, other_level = "Other"),
    rambience = fct_na_value_to_level(factor(rambience), level = "missing"),
    activity = fct_lump_min(factor(activity), min = 40, other_level = "Other"),
    budget = fct_na_value_to_level(factor(budget), level = "missing"),
    price_at_or_below_budget = factor(
      price_at_or_below_budget,
      levels = c(0, 1),
      labels = c("No", "Yes")
    ),
    cuisine_match = factor(cuisine_match, levels = c(0, 1), labels = c("No", "Yes")),
    payment_match = factor(payment_match, levels = c(0, 1), labels = c("No", "Yes"))
  ) |>
  filter(
    !is.na(high_rating),
    !is.na(price_at_or_below_budget),
    !is.na(cuisine_match),
    !is.na(payment_match),
    !is.na(price),
    !is.na(alcohol_clean),
    !is.na(rambience),
    !is.na(activity),
    !is.na(budget)
  )


# Hypothesis tests and models -------------------------------------------------

segment_test_data <- ratings_modeling |>
  filter(!is.na(segment_label), !is.na(high_rating))

segment_chisq <- chisq.test(table(segment_test_data$segment_label, segment_test_data$high_rating))

service_cue_data <- ratings_modeling |>
  filter(!is.na(high_rating), !is.na(alcohol_clean), !is.na(rambience)) |>
  mutate(
    service_cue = case_when(
      alcohol_clean %in% c("Wine or beer", "Full bar") & rambience == "familiar" ~
        "Familiar concept with alcohol service",
      price == "low" & alcohol_clean == "No alcohol" ~
        "Low-price no-alcohol concept",
      TRUE ~ "Other concept"
    )
  ) |>
  filter(service_cue %in% c(
    "Familiar concept with alcohol service",
    "Low-price no-alcohol concept"
  ))

service_cue_chisq <- chisq.test(table(service_cue_data$service_cue, service_cue_data$high_rating))

base_driver_model <- glm(
  high_rating ~ activity + budget,
  data = model_data,
  family = binomial()
)

fit_driver_model <- glm(
  high_rating ~ cuisine_match + price_at_or_below_budget + payment_match +
    activity + budget,
  data = model_data,
  family = binomial()
)

driver_model <- glm(
  high_rating ~ cuisine_match + price_at_or_below_budget + payment_match +
    price + alcohol_clean + rambience + activity + budget,
  data = model_data,
  family = binomial()
)

venue_increment_test <- anova(fit_driver_model, driver_model, test = "Chisq")

model_odds_ratios <- suppressMessages(
  tidy(driver_model, conf.int = TRUE, exponentiate = TRUE)
) |>
  filter(term != "(Intercept)") |>
  mutate(
    term_label = term |>
      str_replace("^cuisine_match", "Cuisine match: ") |>
      str_replace("^price_at_or_below_budget", "Price within budget: ") |>
      str_replace("^payment_match", "Payment match: ") |>
      str_replace("^price", "Venue price: ") |>
      str_replace("^alcohol_clean", "Alcohol: ") |>
      str_replace("^rambience", "Ambience: ") |>
      str_replace("^activity", "Customer activity: ") |>
      str_replace("^budget", "Customer budget: ") |>
      pretty_text(),
    log_effect = abs(log(estimate))
  ) |>
  arrange(desc(log_effect))

hypothesis_tests <- tibble(
  hypothesis = c(
    "H1: High satisfaction differs across profile-derived consumer segments",
    "H2: Familiar concepts with alcohol service have higher high-satisfaction rates than low-price no-alcohol concepts",
    "H3: Venue attributes add explanatory value after consumer profile and fit controls"
  ),
  statistical_test = c(
    "Pearson chi-square test of high rating by segment",
    "Pearson chi-square test comparing selected concept families",
    "Likelihood-ratio test comparing nested logistic models"
  ),
  statistic = c(
    unname(segment_chisq$statistic),
    unname(service_cue_chisq$statistic),
    venue_increment_test$Deviance[2]
  ),
  df = c(
    unname(segment_chisq$parameter),
    unname(service_cue_chisq$parameter),
    venue_increment_test$Df[2]
  ),
  p_value = c(
    segment_chisq$p.value,
    service_cue_chisq$p.value,
    venue_increment_test$`Pr(>Chi)`[2]
  )
)


# Concept scoring -------------------------------------------------------------

concept_scores <- ratings_modeling |>
  filter(!is.na(concept_family), !is.na(segment_label)) |>
  group_by(segment_label, concept_family) |>
  summarise(
    n_ratings = n(),
    avg_rating = mean(rating, na.rm = TRUE),
    high_rating_share = mean(high_rating, na.rm = TRUE),
    cuisine_match_share = mean(cuisine_match, na.rm = TRUE),
    price_within_budget_share = mean(price_at_or_below_budget, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(n_ratings >= 8) |>
  group_by(segment_label) |>
  arrange(desc(high_rating_share), desc(n_ratings), .by_group = TRUE) |>
  mutate(segment_rank = row_number()) |>
  ungroup()

overall_concept_scores <- ratings_modeling |>
  filter(!is.na(concept_family)) |>
  group_by(concept_family) |>
  summarise(
    n_ratings = n(),
    avg_rating = mean(rating, na.rm = TRUE),
    high_rating_share = mean(high_rating, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(n_ratings >= 15) |>
  arrange(desc(high_rating_share), desc(n_ratings))

service_cue_summary <- service_cue_data |>
  group_by(service_cue) |>
  summarise(
    n_ratings = n(),
    high_rating_share = mean(high_rating, na.rm = TRUE),
    avg_rating = mean(rating, na.rm = TRUE),
    .groups = "drop"
  )


# Figures ---------------------------------------------------------------------

segment_summary_plot <- segment_profiles |>
  mutate(segment_label = fct_reorder(segment_label, high_rating_share)) |>
  ggplot(aes(x = segment_label, y = high_rating_share, fill = customers)) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = percent(high_rating_share, accuracy = 1)),
    hjust = -0.12,
    color = project_colors[["charcoal"]],
    size = 3.5
  ) +
  coord_flip(clip = "off") +
  scale_y_continuous(labels = percent, limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  scale_fill_gradient(low = "#D6E4F0", high = project_colors[["blue"]]) +
  labs(
    title = "High satisfaction rate by consumer segment",
    subtitle = "Share of ratings equal to 2 on the UCI 0 to 2 rating scale",
    x = NULL,
    y = "High satisfaction share",
    fill = "Customers"
  ) +
  theme(
    legend.position = "bottom",
    plot.title.position = "plot",
    panel.grid.major.y = element_blank()
  )

ggsave(
  file.path(figure_dir, "segment_satisfaction.png"),
  segment_summary_plot,
  width = 8.5,
  height = 5.2,
  dpi = 300
)

profile_heatmap_data <- segment_profiles |>
  select(
    segment_label,
    low_budget_share,
    medium_or_high_budget_share,
    student_share,
    professional_share,
    car_owner_share,
    social_drinker_share
  ) |>
  pivot_longer(-segment_label, names_to = "profile_metric", values_to = "share") |>
  mutate(
    profile_metric = recode(
      profile_metric,
      low_budget_share = "Low budget",
      medium_or_high_budget_share = "Medium/high budget",
      student_share = "Student",
      professional_share = "Professional",
      car_owner_share = "Car owner",
      social_drinker_share = "Social/casual drinker"
    )
  )

profile_heatmap <- profile_heatmap_data |>
  ggplot(aes(x = segment_label, y = profile_metric, fill = share)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = percent(share, accuracy = 1)), size = 3.2, color = project_colors[["charcoal"]]) +
  scale_fill_gradient(low = "#F7FAFC", high = project_colors[["gold"]], labels = percent) +
  labs(
    title = "Segment profile markers",
    subtitle = "Percent of consumers in each segment with selected survey characteristics",
    x = NULL,
    y = NULL,
    fill = "Share"
  ) +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.title.position = "plot"
  )

ggsave(
  file.path(figure_dir, "segment_profile_heatmap.png"),
  profile_heatmap,
  width = 8.5,
  height = 5.4,
  dpi = 300
)

top_concepts <- concept_scores |>
  filter(segment_rank <= 3) |>
  mutate(
    concept_short = concept_family |>
      str_replace("No alcohol", "No alc.") |>
      str_replace("Wine or beer", "Wine/beer")
  )

concept_plot <- top_concepts |>
  mutate(concept_short = fct_reorder(concept_short, high_rating_share)) |>
  ggplot(aes(x = concept_short, y = high_rating_share, fill = segment_label)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(
    aes(label = paste0(percent(high_rating_share, accuracy = 1), "\n", "n=", n_ratings)),
    hjust = -0.12,
    size = 3.0,
    color = project_colors[["charcoal"]]
  ) +
  facet_wrap(~ segment_label, scales = "free_y") +
  coord_flip(clip = "off") +
  scale_y_continuous(labels = percent, limits = c(0, 1), expand = expansion(mult = c(0, 0.16))) +
  scale_fill_manual(values = rep(
    c(project_colors[["blue"]], project_colors[["olive"]], project_colors[["pink"]], project_colors[["gold"]]),
    3
  )) +
  labs(
    title = "Top observed restaurant concepts by segment",
    subtitle = "Concepts combine price, alcohol service, and ambience with at least 8 ratings",
    x = NULL,
    y = "High satisfaction share"
  ) +
  theme(
    plot.title.position = "plot",
    panel.grid.major.y = element_blank(),
    strip.text = element_text(size = 9.5),
    plot.margin = margin(8, 35, 8, 8)
  )

ggsave(
  file.path(figure_dir, "concept_score_by_segment.png"),
  concept_plot,
  width = 11.5,
  height = 7,
  dpi = 300
)

model_plot_data <- model_odds_ratios |>
  filter(is.finite(conf.low), is.finite(conf.high)) |>
  slice_max(order_by = log_effect, n = 12) |>
  mutate(term_label = fct_reorder(term_label, estimate))

driver_plot <- model_plot_data |>
  ggplot(aes(x = term_label, y = estimate)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = project_colors[["slate"]]) +
  geom_pointrange(
    aes(ymin = conf.low, ymax = conf.high),
    color = project_colors[["blue"]],
    linewidth = 0.7
  ) +
  coord_flip() +
  scale_y_log10(labels = label_number(accuracy = 0.01)) +
  labs(
    title = "Model-estimated drivers of a high restaurant rating",
    subtitle = "Odds ratios from a logistic model; intervals are profile-likelihood confidence intervals",
    x = NULL,
    y = "Odds ratio, log scale"
  ) +
  theme(
    plot.title.position = "plot",
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(figure_dir, "preference_driver_odds_ratios.png"),
  driver_plot,
  width = 8.5,
  height = 6.2,
  dpi = 300
)

cluster_plot <- cluster_candidates |>
  mutate(k = factor(k), selected = k == as.character(selected_k)) |>
  ggplot(aes(x = k, y = avg_silhouette, group = 1)) +
  geom_line(color = project_colors[["slate"]], linewidth = 0.8) +
  geom_point(aes(color = selected), size = 3) +
  scale_color_manual(values = c(`FALSE` = project_colors[["slate"]], `TRUE` = project_colors[["pink"]])) +
  labs(
    title = "Segmentation solution check",
    subtitle = "Average silhouette width for PAM solutions using Gower distance",
    x = "Number of segments",
    y = "Average silhouette width"
  ) +
  theme(
    legend.position = "none",
    plot.title.position = "plot"
  )

ggsave(
  file.path(figure_dir, "cluster_selection.png"),
  cluster_plot,
  width = 7,
  height = 4.5,
  dpi = 300
)


# Save analysis tables --------------------------------------------------------

write_csv(customer_segments, file.path(processed_dir, "customer_segments.csv"))
write_csv(ratings_modeling, file.path(processed_dir, "rating_modeling_dataset.csv"))
write_csv(segment_profiles, file.path(table_dir, "segment_profiles.csv"))
write_csv(cluster_candidates, file.path(table_dir, "cluster_candidate_diagnostics.csv"))
write_csv(model_odds_ratios, file.path(table_dir, "model_odds_ratios.csv"))
write_csv(concept_scores, file.path(table_dir, "segment_concept_scores.csv"))
write_csv(overall_concept_scores, file.path(table_dir, "overall_concept_scores.csv"))
write_csv(hypothesis_tests, file.path(table_dir, "hypothesis_tests.csv"))
write_csv(service_cue_summary, file.path(table_dir, "service_cue_summary.csv"))


# Optional summary deck -------------------------------------------------------

if (requireNamespace("officer", quietly = TRUE) && requireNamespace("flextable", quietly = TRUE)) {
  library(officer)
  library(flextable)

  deck_path <- file.path(output_dir, "ipsos_style_summary_deck.pptx")

  segment_table <- segment_profiles |>
    transmute(
      Segment = segment_label,
      Customers = customers,
      Ratings = ratings,
      `High satisfaction` = percent(high_rating_share, accuracy = 1),
      `Top interest` = pretty_text(top_interest),
      `Top personality` = pretty_text(top_personality)
    )

  top_overall <- overall_concept_scores |>
    slice_head(n = 5) |>
    transmute(
      Concept = concept_family,
      Ratings = n_ratings,
      `High satisfaction` = percent(high_rating_share, accuracy = 1),
      `Average rating` = round(avg_rating, 2)
    )

  ppt <- read_pptx()

  ppt <- add_slide(ppt, layout = "Title Slide", master = "Office Theme")
  ppt <- ph_with(ppt, "Restaurant Consumer Segmentation", location = ph_location_type(type = "ctrTitle"))
  ppt <- ph_with(
    ppt,
    "UCI Restaurant and Consumer Data | R segmentation, driver modeling, and concept scoring",
    location = ph_location_type(type = "subTitle")
  )

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "What the analysis is designed to answer", location = ph_location_type(type = "title"))
  ppt <- ph_with(
    ppt,
    "Which consumer segments are present, what restaurant concepts rate best within them, and which measurable features are most associated with high satisfaction?",
    location = ph_location_type(type = "body")
  )

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Segment profile summary", location = ph_location_type(type = "title"))
  ppt <- ph_with(ppt, flextable(segment_table) |> autofit(), location = ph_location_type(type = "body"))

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Top observed concepts", location = ph_location_type(type = "title"))
  ppt <- ph_with(ppt, flextable(top_overall) |> autofit(), location = ph_location_type(type = "body"))

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Segment satisfaction", location = ph_location_type(type = "title"))
  ppt <- ph_with(
    ppt,
    external_img(file.path(figure_dir, "segment_satisfaction.png"), width = 8.6, height = 4.8),
    location = ph_location(left = 0.6, top = 1.3, width = 8.8, height = 5.0)
  )

  ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
  ppt <- ph_with(ppt, "Multivariate drivers", location = ph_location_type(type = "title"))
  ppt <- ph_with(
    ppt,
    external_img(file.path(figure_dir, "preference_driver_odds_ratios.png"), width = 8.5, height = 5.0),
    location = ph_location(left = 0.7, top = 1.1, width = 8.5, height = 5.2)
  )

  print(ppt, target = deck_path)
}


# Short text summary ----------------------------------------------------------

summary_lines <- c(
  "Ipsos-style market research project summary",
  paste0(
    "Source: UCI Restaurant and Consumer Data, ",
    nrow(ratings), " ratings, ",
    n_distinct(ratings$user_id), " consumers, ",
    n_distinct(ratings$place_id), " restaurants."
  ),
  paste0("Selected segmentation solution: ", selected_k, " segments, based on PAM clustering over Gower distance."),
  paste0("Model rows used for driver analysis: ", nrow(model_data), "."),
  paste0(
    "Top overall observed concept: ",
    overall_concept_scores$concept_family[[1]],
    " (",
    percent(overall_concept_scores$high_rating_share[[1]], accuracy = 1),
    " high satisfaction, n=",
    overall_concept_scores$n_ratings[[1]],
    ")."
  )
)

writeLines(summary_lines, file.path(output_dir, "analysis_summary.txt"))

message("Analysis complete.")
message("Project root: ", project_dir)
message("Selected k: ", selected_k)
message("Figures written to: ", figure_dir)
message("Tables written to: ", table_dir)
