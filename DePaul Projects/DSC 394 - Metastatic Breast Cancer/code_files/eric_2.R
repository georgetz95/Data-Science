# Directory
setwd("C:/Users/erice/Documents/wids")

# Libraries
library(car)
library(fastDummies)
library(MASS)
library(tidyverse)

# Data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
train_i <- read.csv("train_imputed.csv")
test_i <- read.csv("test_imputed.csv")

# Outlier removal
removal_outlier <- function(df, column, threshold) {
  removed_outliers <- df[df[column] <= threshold, ]
  return(removed_outliers)
}

threshold_value <- 50
train_i <- removal_outlier(train_i, 'bmi', threshold_value)
test_i <- removal_outlier(test_i, 'bmi', threshold_value)

# Create dummy variables
train_i <- dummy_cols(train_i, select_columns = c("patient_race", "payer_type", "Region"))
test_i <- dummy_cols(test_i, select_columns = c("patient_race", "payer_type", "Region"))

# Change dummy variable names
colnames(train_i) <- gsub("patient_race_", "race_", colnames(train_i))
colnames(train_i) <- gsub("payer_type_", "payer_", colnames(train_i))
colnames(train_i) <- gsub("Region_", "region_", colnames(train_i))

colnames(test_i) <- gsub("patient_race_", "race_", colnames(test_i))
colnames(test_i) <- gsub("payer_type_", "payer_", colnames(test_i))
colnames(test_i) <- gsub("Region_", "region_", colnames(test_i))

# Change variable
train_i$patient_gender <- ifelse(train_i$patient_gender == "FALSE", "Female", train_i$patient_gender)
test_i$patient_gender <- ifelse(test_i$patient_gender == "FALSE", "Female", test_i$patient_gender)

# Remove variable (commented out for PCA)
#train_i <- train_i[, !names(train_i) %in% c("Division")]
#test_i <- test_i[, !names(test_i) %in% c("Division")]

# # Correlation analysis
# # Select numeric variables
# numeric_vars <- train_i %>% select_if(is.numeric)
# 
# # Calculate correlation matrix
# correlation_matrix <- cor(numeric_vars)
# 
# # Extract correlation of predictors with the outcome variable
# outcome_correlation <- correlation_matrix["metastatic_diagnosis_period", ]
# 
# # Sort variables by correlation strength
# sorted_correlation <- sort(abs(outcome_correlation), decreasing = TRUE)
# 
# # Save as data frame
# sorted_correlation_df <- data.frame(variable = names(sorted_correlation), correlation = sorted_correlation)
# 
# # ANOVA
# # Perform ANOVA for State
# state_anova <- aov(metastatic_diagnosis_period ~ patient_state, data = train_i)
# summary(state_anova)
# 
# # Perform ANOVA for breast_cancer_diagnosis_code
# breast_cancer_anova <- aov(metastatic_diagnosis_period ~ breast_cancer_diagnosis_code, data = train_i)
# summary(breast_cancer_anova)
# 
# # Perform ANOVA for metastatic_cancer_diagnosis_code
# metastatic_cancer_anova <- aov(metastatic_diagnosis_period ~ metastatic_cancer_diagnosis_code, data = train_i)
# summary(metastatic_cancer_anova)
# 
# # Tukey testing
# # Post-hoc pairwise comparisons for patient_state
# state_posthoc <- TukeyHSD(state_anova)
# 
# # Post-hoc pairwise comparisons for breast_cancer_diagnosis_code
# breast_cancer_posthoc <- TukeyHSD(breast_cancer_anova)
# 
# # Post-hoc pairwise comparisons for metastatic_cancer_diagnosis_code
# metastatic_cancer_posthoc <- TukeyHSD(metastatic_cancer_anova)
# 
# # Save post-hoc pairwise comparisons for patient_state to CSV
# state_posthoc_df <- as.data.frame(state_posthoc$`patient_state`)
# write.csv(state_posthoc_df, "state_posthoc.csv", row.names = TRUE)
# 
# # Save post-hoc pairwise comparisons for breast_cancer_diagnosis_code to CSV
# breast_cancer_posthoc_df <- as.data.frame(breast_cancer_posthoc$`breast_cancer_diagnosis_code`)
# write.csv(breast_cancer_posthoc_df, "breast_cancer_posthoc.csv", row.names = TRUE)
# 
# # Save post-hoc pairwise comparisons for metastatic_cancer_diagnosis_code to CSV
# metastatic_cancer_posthoc_df <- as.data.frame(metastatic_cancer_posthoc$`metastatic_cancer_diagnosis_code`)
# write.csv(metastatic_cancer_posthoc_df, "metastatic_cancer_posthoc.csv", row.names = TRUE)
# 
# # Calculate means of Tukey test results
# state_posthoc_mean <- mean(state_posthoc_df$diff)
# breast_cancer_posthoc_mean <- mean(breast_cancer_posthoc_df$diff)
# metastatic_cancer_posthoc_mean <- mean(metastatic_cancer_posthoc_df$diff)
# 
# # Check for outliers using z-scores
# state_posthoc_z <- (state_posthoc_df$diff - state_posthoc_mean) / sd(state_posthoc_df$diff)
# state_outliers <- state_posthoc_df[abs(state_posthoc_z) > 2, ] # CO and ND
# 
# breast_cancer_posthoc_z <- (breast_cancer_posthoc_df$diff - breast_cancer_posthoc_mean) / sd(breast_cancer_posthoc_df$diff)
# breast_cancer_outliers <- breast_cancer_posthoc_df[abs(breast_cancer_posthoc_z) > 2, ]
# 
# metastatic_cancer_posthoc_z <- (metastatic_cancer_posthoc_df$diff - metastatic_cancer_posthoc_mean) / sd(metastatic_cancer_posthoc_df$diff)
# metastatic_cancer_outliers <- metastatic_cancer_posthoc_df[abs(metastatic_cancer_posthoc_z) > 2, ] # C7880 and C7900
# 
# # Testing means across groups to see if assumptions hold
# # State outliers
# # Step 1: Calculate Mean Metastatic Diagnosis Periods for all states except CO and ND
# code_counts_state <- table(train_i$patient_state)
# 
# # Filter codes with more than 100 observations
# codes_over_100_obs_state <- names(code_counts_state[code_counts_state > 100])
# 
# # Subset Tukey outliers dataframe for codes with more than 100 observations
# tukey_outliers_over_100_obs_state <- state_posthoc_df[breast_cancer_posthoc_df$`rowname` %in% codes_over_100_obs_state, ]
# 
# # Filtered
# filtered_tukey_outliers_state <- tukey_outliers_over_100_obs_state[abs(tukey_outliers_over_100_obs_state$diff) > 2, ]
# 
# mean_diagnosis_period_other_states <- mean(train_i$metastatic_diagnosis_period[!train_i$patient_state %in% c("CO")])
# 
# # Step 2: Calculate Mean for CO
# mean_diagnosis_period_CO <- mean(train_i$metastatic_diagnosis_period[train_i$patient_state %in% c("CO")])
# 
# # Breast cancer code outliers
# code_counts_bcc <- table(train_i$breast_cancer_diagnosis_code)
# 
# # Calculate Mean for C7880 and C7900
# mean_diagnosis_period_C7880_C7900 <- mean(train_i$metastatic_diagnosis_period[train_i$metastatic_cancer_diagnosis_code %in% c("C7880", "C7900")])
# 
# # Checking observations of C7880 and C7900
# code_counts_mcc <- table(train_i$metastatic_cancer_diagnosis_code)
# code_counts_c7880 <- code_counts["C7880"]
# code_counts_c7900 <- code_counts["C7900"]
# 
# # Get counts for each code
# code_counts_mcc <- table(train_i$metastatic_cancer_diagnosis_code)
# 
# # Filter codes with more than 100 observations
# codes_over_100_obs_mcc <- names(code_counts[code_counts > 100])
# 
# # Subset Tukey outliers dataframe for codes with more than 100 observations
# tukey_outliers_over_100_obs_mcc <- metastatic_cancer_posthoc_df[metastatic_cancer_posthoc_df$`rowname` %in% codes_over_100_obs, ]
# 
# # Filtered
# filtered_tukey_outliers_mcc <- tukey_outliers_over_100_obs[abs(tukey_outliers_over_100_obs$diff) > 2, ]
# 
# # Stepwise regression
# # Exclude categorical variables
# predictor_vars <- setdiff(names(train_i), c("patient_race", "payer_type", "Region", "patient_gender", "patient_state", "breast_cancer_diagnosis_code", "metastatic_cancer_diagnosis_code"))
# 
# # Escape variable names with backticks
# predictor_vars_escaped <- paste0("`", predictor_vars, "`")
# 
# # Construct the formula dynamically
# formula <- as.formula(paste("metastatic_diagnosis_period ~", paste(predictor_vars_escaped, collapse = "+")))
# 
# # Perform stepwise regression (forward) using the constructed formula
# stepwise_forward <- step(lm(formula, data = train_i), direction = "forward")
# 
# # Summary of the model
# summary(stepwise_forward)
# 
# # Variables for Bayesian model:
#   # patient_age
#   # family_dual_income
#   # income_houshold_10_to_15
#   # income_household_100_to_150
#   # income_houshold_150_over
#   # income_household_six_figure
#   # education_less_highschool
#   # education_highschool
#   # education_some_college
#   # education_bachelors
#   # labor_force_participation
#   # Average.of.Jul.18
#   # bmi_null
#   # patient_state (categorical)
#   # race_Asian
#   # race_Hispanic
#   # payer_COMMERCIAL
#   # payer_MEDICAID
# 
# # Variable list
# variables_of_interest <- c("patient_age", "family_dual_income", "income_houshold_10_to_15",
#                            "income_household_100_to_150", "income_houshold_150_over", 
#                            "income_household_six_figure", "education_less_highschool", 
#                            "education_highschool", "education_some_college", 
#                            "education_bachelors", "labor_force_participation", 
#                            "Average.of.Jul.18", "bmi_null", "patient_state", 
#                            "race_Asian", "race_Hispanic", "payer_COMMERCIAL", 
#                            "payer_MEDICAID")

# PCA Data - Use this instead of previous since less variables and variance explained
train_pca <- read.csv("train_pca.csv")
test_pca <- read.csv("test_pca.csv")

# Mean encoding for categorical variables
char_vars_to_encode <- c("patient_race", "payer_type", "patient_state", "Region", "Division", "breast_cancer_diagnosis_code", "metastatic_cancer_diagnosis_code")

# Loop through each character variable and apply target encoding
for (var in char_vars_to_encode) {
  # Calculate mean of metastatic_diagnosis_period for each category
  mean_target <- aggregate(metastatic_diagnosis_period ~ ., data = train_i[, c(var, "metastatic_diagnosis_period")], FUN = mean)
  
  # Create mapping of category to mean
  target_map <- setNames(mean_target$metastatic_diagnosis_period, mean_target[, var])
  
  # Replace categorical variable with target encoded variable in train_pca and test_pca
  train_pca[[paste0(var, "_mean")]] <- target_map[train_pca[[var]]]
  test_pca[[paste0(var, "_mean")]] <- target_map[test_pca[[var]]]
  
  # Remove original categorical variable
  train_pca <- train_pca[, !names(train_pca) %in% var]
  test_pca <- test_pca[, !names(test_pca) %in% var]
}

# Perform stepwise regression
stepwise_model <- step(lm(metastatic_diagnosis_period ~ ., data = train_pca), direction = "both")

# Summary of the stepwise model
summary(stepwise_model)

# Variables for model, final:
# PC1
# patient_age
# patient_state_mean
# breast_cancer_diagnosis_code_mean
# metastatic_cancer_diagnosis_code_mean

# Checking distribution of chosen variables
# Histograms
# Histogram for PC1
hist(train_pca$PC1, main = "Histogram of PC1", xlab = "PC1", col = "blue")

# Histogram for patient_age
hist(train_pca$patient_age, main = "Histogram of Patient Age", xlab = "Patient Age", col = "green")

# Histogram for patient_state_mean
hist(train_pca$patient_state_mean, main = "Histogram of Patient State (Mean)", xlab = "Patient State (Mean)", col = "red")

# Histogram for breast_cancer_diagnosis_code_mean
hist(train_pca$breast_cancer_diagnosis_code_mean, main = "Histogram of Breast Cancer Diagnosis Code (Mean)", xlab = "Breast Cancer Diagnosis Code (Mean)", col = "orange")

# Histogram for metastatic_cancer_diagnosis_code_mean
hist(train_pca$metastatic_cancer_diagnosis_code_mean, main = "Histogram of Metastatic Cancer Diagnosis Code (Mean)", xlab = "Metastatic Cancer Diagnosis Code (Mean)", col = "purple")

# Transformations
# Take absolute value of PC1 and then apply log transformation
train_pca$log_PC1 <- log(abs(train_pca$PC1))

# Checking the distribution of log-transformed PC1
hist(train_pca$log_PC1, main = "Histogram of Log-transformed (Absolute) PC1", xlab = "Log-transformed (Absolute) PC1")

# Apply square root transformation PC1
train_pca$sqrt_PC1 <- sqrt(abs(train_pca$PC1))

# Checking the distribution of square root transformed PC1
hist(train_pca$sqrt_PC1, main = "Histogram of Square Root Transformed (Absolute) PC1", xlab = "Square Root Transformed (Absolute) PC1")

# Log transform for patient_state_mean
train_pca$log_patient_state_mean <- log(train_pca$patient_state_mean)

# Plot the histogram of log-transformed patient_state_mean
hist(train_pca$patient_state_mean, main = "Histogram of Log-transformed patient_state_mean")

# Apply square root transformation to patient_state_mean
train_pca$sqrt_patient_state_mean <- sqrt(train_pca$patient_state_mean)

# Checking the distribution of square root transformed patient_state_mean
hist(train_pca$sqrt_PC1, main = "Histogram of Square Root Transformed patient_state_mean", xlab = "Square Root Transformed (Absolute) PC1")

# Counts for breast_cancer_diagnosis_code_mean
value_counts_bcdcm <- table(train_pca$breast_cancer_diagnosis_code_mean)
print(value_counts_bcdcm)

# Counts for metastatic_cancer_diagnosis_code_mean
value_counts_mcdcm <- table(train_pca$metastatic_cancer_diagnosis_code_mean)
print(value_counts_bcdcm)

# Boxplot for breast_cancer_diagnosis_code_mean
boxplot(train_pca$breast_cancer_diagnosis_code_mean, main = "Boxplot of Breast Cancer Diagnosis Code (Mean)")

# Summary statistics for breast_cancer_diagnosis_code_mean
summary(train_pca$breast_cancer_diagnosis_code_mean)

# Boxplot for metastatic_cancer_diagnosis_code_mean
boxplot(train_pca$metastatic_cancer_diagnosis_code_mean, main = "Boxplot of Metastatic Cancer Diagnosis Code (Mean)")

# Summary statistics for metastatic_cancer_diagnosis_code_mean
summary(train_pca$metastatic_cancer_diagnosis_code_mean)

# Categorize breast_cancer_diagnosis_code_mean
train_pca$breast_cancer_diagnosis_code_mean_category <- cut(train_pca$breast_cancer_diagnosis_code_mean, 
                                                            breaks = c(-Inf, 56.44, 69.03, Inf), 
                                                            labels = c(1, 2, 3))

# Categorize metastatic_cancer_diagnosis_code_mean
train_pca$metastatic_cancer_diagnosis_code_mean_category <- cut(train_pca$metastatic_cancer_diagnosis_code_mean, 
                                                                breaks = c(-Inf, 88.84, 108.09, Inf), 
                                                                labels = c(1, 2, 3))

# Summary statistics for breast_cancer_diagnosis_code_mean_category
summary(train_pca$breast_cancer_diagnosis_code_mean_category)

# Summary statistics for metastatic_cancer_diagnosis_code_mean_category
summary(train_pca$metastatic_cancer_diagnosis_code_mean_category)

# Calculate means for each category of breast_cancer_diagnosis_code_mean_category
breast_cancer_means <- tapply(train_pca$metastatic_diagnosis_period, train_pca$breast_cancer_diagnosis_code_mean_category, mean)

# Calculate means for each category of metastatic_cancer_diagnosis_code_mean_category
metastatic_cancer_means <- tapply(train_pca$metastatic_diagnosis_period, train_pca$metastatic_cancer_diagnosis_code_mean_category, mean)

# Calculate stan dev for each category of breast_cancer_diagnosis_code_mean_category
breast_cancer_sd <- tapply(train_pca$metastatic_diagnosis_period, train_pca$breast_cancer_diagnosis_code_mean_category, sd)

# Calculate stan dev each category of metastatic_cancer_diagnosis_code_mean_category
metastatic_cancer_sd <- tapply(train_pca$metastatic_diagnosis_period, train_pca$metastatic_cancer_diagnosis_code_mean_category, sd)

# Display the means
print("Breast Cancer Diagnosis Code Mean Category Means:")
print(breast_cancer_means)

print("Metastatic Cancer Diagnosis Code Mean Category Means:")
print(metastatic_cancer_means)

# Display the sd
print("Breast Cancer Diagnosis Code Mean Category SD:")
print(breast_cancer_sd)

print("Metastatic Cancer Diagnosis Code Mean Category SD:")
print(metastatic_cancer_sd)

# JAGS Modeling
library(coda)
library(pROC)
library(rjags)

# Number of rows
N <- nrow(train_pca)

# Predictor variables
predictors <- c("PC1","patient_age", "patient_state_mean", "breast_cancer_diagnosis_code_mean", "metastatic_cancer_diagnosis_code_mean")

# Number of predictors
K <- length(predictors)

# Data list
data_list <- list(
  N = N,
  y = train_pca$metastatic_diagnosis_period,
  PC1 = train_pca$PC1,
  patient_age = train_pca$patient_age,
  patient_state_mean = train_pca$patient_state_mean,
  breast_cancer_diagnosis_code_mean_category = train_pca$breast_cancer_diagnosis_code_mean_category,
  metastatic_cancer_diagnosis_code_mean_category = train_pca$metastatic_cancer_diagnosis_code_mean_category
)

model_string <- textConnection("
model {
  # Likelihood for metastatic diagnosis period
  for (i in 1:N) {
    # Linear predictor for metastatic diagnosis period
    mu[i] <- beta0 + beta_PC1 * PC1[i] + beta_age * patient_age[i] + beta_state * patient_state_mean[i] + beta_bcd[breast_cancer_diagnosis_code_mean_category[i]] + beta_mcd[metastatic_cancer_diagnosis_code_mean_category[i]]

    # Normal likelihood for metastatic diagnosis period
    y[i] ~ dnorm(mu[i], tau)
  }

  # Priors
  beta0 ~ dnorm(0, 0.001)
  beta_PC1 ~ dt(0, 1, 1)
  beta_age ~ dnorm(0, 0.001)
  beta_state ~ dnorm(0, 0.001)
  for (j in 1:3) {
    beta_bcd[j] ~ dnorm(0, 0.001)
    beta_mcd[j] ~ dnorm(0, 0.001)
  }
  tau ~ dgamma(0.001, 0.001) # Precision parameter
}")

# Compile the model
model <- jags.model(model_string, data = data_list, n.chains = 4)

# Model specifications
burn_in <- 10000
iterations <- 20000

# Update the model
update(model, burn_in)

# Sampling for all beta parameters to check convergence
samples_all_betas <- coda.samples(model, variable.names = c("beta0", "beta_PC1", "beta_age",
"beta_state", paste0("beta_bcd[", 1:3, "]"), paste0("beta_mcd[", 1:3, "]")), n.iter = iterations)

# Trace plots
traceplot(samples_all_betas)

# Geweke diagnostics
geweke <- geweke.diag(samples_all_betas)

# Effective sample size
ess <- effectiveSize(samples_all_betas)

# Number of posterior predictive samples
n_samples <- 5000

# Sample predictions directly from the model
predicted_values_samples <- coda.samples(model, variable.names = "mu", n.iter = n_samples)

# Extract the predicted values
predicted_values <- as.matrix(predicted_values_samples[[1]])

# Calculate mean predictions across all posterior predictive samples
mean_predictions <- colMeans(predicted_values)

# Compare the mean predicted values with the actual metastatic_diagnosis_period
comparison_df <- data.frame(
  Actual = train_pca$metastatic_diagnosis_period,
  Predicted = mean_predictions
)

# View the comparison dataframe
view(comparison_df)

# Calculate RMSE, MAE, and MSE for the Bayesian model
rmse_bayes <- sqrt(mean((train_pca$metastatic_diagnosis_period - mean_predictions)^2))
mae_bayes <- mean(abs(train_pca$metastatic_diagnosis_period - mean_predictions))
mse_bayes <- mean((train_pca$metastatic_diagnosis_period - mean_predictions)^2)

# Display the results for the Bayesian model
cat("RMSE Bayesian:", rmse_bayes, "\n")
cat("MAE Bayesian:", mae_bayes, "\n")
cat("MSE Bayesian:", mse_bayes, "\n")