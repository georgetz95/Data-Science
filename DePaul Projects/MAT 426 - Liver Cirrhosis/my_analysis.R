## ----setup, include=FALSE-----------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(ggplot2)
library(tableone)
library(kableExtra)
library(clipr)
library(missForest)
library(ggcorrplot)
library(DescTools) 
library(survival)
library(survminer)
library(caret)
library(MASS)
library(rsample)
library(pROC)
library(ResourceSelection)
library(car)
# library(timeROC)

source('mixed_corr_matrix.R')


## -----------------------------------------------------------------------------------------------------
show_colors <- function() {
  scales::show_col(scales::hue_pal()(20))
  print(scales::hue_pal()(20))
}


## -----------------------------------------------------------------------------------------------------
df <- read.csv('cirrhosis.csv')
df <- df %>% filter(Status %in% c('C', 'D'))
df <- df %>% 
  mutate(
    Status = factor(
      case_when(
        Status == 'C' ~ 0,
        Status == 'D' ~ 1
      ), levels=c(0, 1)
    ),
    Drug = factor(Drug, levels=c('Placebo', 'D-penicillamine')),
    Sex = factor(
      case_when(
        Sex == 'F' ~ 'Female',
        Sex == 'M' ~ 'Male'
      ), levels=c('Female', 'Male'),
    ),
    Ascites = factor(
      case_when(
        Ascites == 'N' ~ 'No',
        Ascites == 'Y' ~ 'Yes',
      ), levels=c('No', 'Yes')
    ),
    Hepatomegaly = factor(
      case_when(
        Hepatomegaly == 'N' ~ 'No',
        Hepatomegaly == 'Y' ~ 'Yes',
      ), levels=c('No', 'Yes')
    ),
    Spiders = factor(
      case_when(
        Spiders == 'N' ~ 'No',
        Spiders == 'Y' ~ 'Yes',
      ), levels=c('No', 'Yes')
    ),
    Edema = factor(Edema, levels=c('N', 'S', 'Y')),
    Stage = factor(Stage, levels=c(1, 2, 3, 4), ordered=T),
    
    Age_years = Age / 365.25,
    N_Years = N_Days / 365.25
  ) %>%
  filter(!is.na(Drug)) %>%
  dplyr::select((-c(Age, N_Days)))
head(df)


## -----------------------------------------------------------------------------------------------------
categorical_vars <- names(df)[sapply(df, is.factor)][-c(1,2)] # Remove Drug stratification variable
numerical_vars <-names(df)[sapply(df, is.numeric)][-1] # Remove ID variable
# stopifnot(length(categorical_vars) + length(numerical_vars) == ncol(df)-2)

table1 <- CreateTableOne(
  vars = c(numerical_vars, categorical_vars, c('Status')),
  strata = 'Drug',
  data=df,
  factorVars=categorical_vars
)


## -----------------------------------------------------------------------------------------------------
p <- print(table1, printToggle = F, noSpaces = T, showAllLevels = T, missing=F)
# Run in console mode
kbl(p, booktabs = T, longtable=T, format = "latex", caption="Summary of patient baseline characteristics, including demographic, clinical, and lifestyle factors, stratified by treatment.", label="tableone") %>% 
  kable_styling(latex_options = c('striped', 'repeat_header')) %>% write_clip()


## -----------------------------------------------------------------------------------------------------
missing_counts <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(cols=everything(), names_to='Feature', values_to='Missing_Count') %>%
  mutate(Missing_Perc = (Missing_Count / nrow(df)) * 100) %>%
  mutate(Missing_Info = paste0(Missing_Count, ' (', round(Missing_Perc, 1), '%)')) %>%
  arrange(desc(Missing_Count)) %>%
  filter(Missing_Count > 0) %>%
  dplyr::select(Feature, Missing_Info)

missing_counts


## -----------------------------------------------------------------------------------------------------
miss_forest <- missForest(df)
print(miss_forest$OOBerror)


## -----------------------------------------------------------------------------------------------------
df_imputed <- miss_forest$ximp
head(df_imputed)


## -----------------------------------------------------------------------------------------------------
p <- df_imputed %>%
  mutate(Status = dplyr::recode(Status, `0` = 'Censored', `1` = 'Died')) %>%
  ggplot(aes(x=Sex, fill=Status)) +
  geom_bar(position='dodge') +
  labs(
    title = 'Patient Survival by Gender',
    x = 'Gender',
    y = 'Count',
    fill = 'Status'
  ) +
  theme_classic()
ggsave('status_count_by_gender.png', plot=p, width=6, height=4, dpi=300)
print(p)


## -----------------------------------------------------------------------------------------------------
p <- df_imputed %>%
  mutate(Status = dplyr::recode(Status, `0` = 'Censored', `1` = 'Died')) %>%
  ggplot(aes(x=Drug, fill=Status)) +
  geom_bar(position='dodge') +
  labs(
    title = 'Patient Survival by Treatment Arm',
    x = 'Treatment',
    y = 'Count',
    fill = 'Status'
  ) +
  theme_classic()
ggsave('status_count_by_treatment.png', plot=p, width=6, height=4, dpi=300)
print(p)

## -----------------------------------------------------------------------------------------------------
p <- df_imputed %>%
  mutate(Status = dplyr::recode(Status, `0` = 'Censored', `1` = 'Died')) %>%
  ggplot(aes(x=Stage, fill=Status)) +
  geom_bar(position='dodge') +
  labs(
    title = 'Patient Survival by Stage of Cirrhosis',
    x = 'Stage',
    y = 'Count',
    fill = 'Status'
  ) +
  theme_classic()
ggsave('status_count_by_stage.png', plot=p, width=6, height=4, dpi=300)
print(p)


## -----------------------------------------------------------------------------------------------------
p <- df_imputed %>%
  # mutate(Status = recode(Status, `0` = 'Censored', `1` = 'Died')) %>%
  ggplot(aes(x=Sex, y=Age_years, fill=Sex)) +
  geom_boxplot() +
  labs(
    title = 'Age Distribution by Gender',
    x = 'Gender',
    y = 'Age (in years)',
    fill = 'Gender'
  ) +
  # coord_flip() +
  theme_classic() +
  theme(legend.position='none')
ggsave('age_boxplot_by_gender.png', plot=p, width=6, height=4, dpi=300)
print(p)


## -----------------------------------------------------------------------------------------------------
corr_matrix <- compute_mixed_corr(df_imputed %>% dplyr::select(-c(ID, Age_years, N_Years)))
corr_matrix


## -----------------------------------------------------------------------------------------------------
p <- ggcorrplot(corr_matrix, 
           method = "square",    # Boxed shape for grid
           type = "lower",       # Show only lower triangle
           lab = TRUE,           # Show correlation values inside the boxes
           lab_size = 2,         # Adjust text size
           digits = 2,           # Round correlation values
           tl.cex = 10,          # Increase axis label size
           tl.srt = 45,          # Rotate axis labels for better readability
           colors = c("red", "white", "blue")) +  # Red (-1), White (0), Blue (+1)
  
  ggtitle("Correlation Matrix Heatmap") +  # Add title
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center title
    panel.background = element_rect(fill = "white", color = NA),  # Fix transparent issue
    plot.background = element_rect(fill = "white", color = NA)  # Ensure full white background
  )

ggsave('corr_matrix.png', plot=p, width=6, height=5, dpi=300)
print(p)

## -----------------------------------------------------------------------------------------------------
surv_object <- Surv(time=tst$N_Years, event=as.numeric(tst$Status))
km_fit <- survfit(surv_object ~ Drug, data=df_imputed)
logrank_test <- survdiff(surv_object ~ Drug, data=df_imputed)
print(logrank_test)


## -----------------------------------------------------------------------------------------------------
grid.draw.ggsurvplot <- function(x){
  survminer:::print.ggsurvplot(x, newpage = FALSE)
}

p <- ggsurvplot(
  km_fit,
  data = df_imputed,
  pval = TRUE,        # Show Log-Rank p-value
  conf.int = F,    # Show confidence intervals
  risk.table = TRUE,  # Show risk table
  risk.table.y.text.col = TRUE,   # Color risk table text
  risk.table.y.text = FALSE,      # Remove vertical risk table labels
  risk.table.height = 0.25,       # Adjust risk table size
  risk.table.col = "strata",      # Color risk table by group
  break.time.by = 2,            # Change time intervals (e.g., every 500 days)
  xlab = "Time in years",          # Customize x-axis label
  title = "Kaplan-Meier Survival Curve",  # Add title
  risk.table.title = "Number at Risk",    # Risk table title
  ggtheme = theme_classic()
)

ggsave('km_plot.png', plot=p, width=6, height=5, dpi=300)
print(p)

## -----------------------------------------------------------------------------------------------------
set.seed(123)
split <- initial_split(df_imputed, prop=0.5, strata=Status)
train_data <- training(split)
test_data <- testing(split)

cat('dim(train_data):', dim(train_data), '\n')
cat('dim(test_data):', dim(test_data), '\n')


## -----------------------------------------------------------------------------------------------------
response_var <- 'Status'
time_var <- 'N_Years'
predictors <- df_imputed %>% dplyr::select(-c(ID, N_Years, Drug, Status)) %>% colnames()
reg_formula <- as.formula(paste(response_var, '~', paste(predictors, collapse='+')))
print(logreg_formula)


## -----------------------------------------------------------------------------------------------------
logit_model <- glm(reg_formula, data=train_data, family=binomial(link='logit'))
summary(logit_model)

## -----------------------------------------------------------------------------------------------------
logit_train_preds <- predict(logit_model, train_data, type='response')
logit_test_preds <- predict(logit_model, test_data, type='response')

cat('Train AUC:', auc(roc(train_data$Status, logit_train_preds, levels=c(0,1), direction='<')), '\n')
cat('Test AUC:', auc(roc(test_data$Status, logit_test_preds, levels=c(0,1), direction='<')), '\n')


## -----------------------------------------------------------------------------------------------------
stepwise_model <- step(glm(reg_formula, data=train_data, family=binomial(link='logit')), direction='both')
summary(stepwise_model)


## -----------------------------------------------------------------------------------------------------
stepwise_train_preds <- predict(stepwise_model, train_data, type='response')
stepwise_test_preds <- predict(stepwise_model, test_data, type='response')

cat('Train AUC:', auc(roc(train_data$Status, stepwise_train_preds, levels=c(0,1), direction='<')), '\n')
cat('Test AUC:', auc(roc(test_data$Status, stepwise_test_preds, levels=c(0,1), direction='<')), '\n')


## -----------------------------------------------------------------------------------------------------
calibration_data <- data.frame(Status=as.numeric(as.character(test_data$Status)), RiskScore=stepwise_test_preds) %>%
  mutate(calib_bin = ntile(RiskScore, 10)) %>%
  group_by(calib_bin) %>%
  summarise(
    mean_predicted = mean(RiskScore),
    observed_rate = mean(Status),
    abs_deviation = abs(mean_predicted - observed_rate)
  )

calibration_data


## -----------------------------------------------------------------------------------------------------
p <- ggplot(calibration_data, aes(x = mean_predicted, y = observed_rate, label = calib_bin)) +
  geom_point(size = 3, color = 'black', fill = 'blue', shape = 21, stroke = 2) +  # Blue dots
  geom_text(vjust = -0.8, size = 4, color = 'black') +  # Label points with bin numbers
  xlim(0, max(max(calibration_data$mean_predicted), max(calibration_data$observed_rate))) +
  ylim(0, max(max(calibration_data$mean_predicted), max(calibration_data$observed_rate))) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed', color = 'red', linewidth = 1) +  # Reference line
  labs(
    title = 'Model Calibration Plot',
    x = 'Predicted Probability',
    y = 'Observed Event Rate'
  ) +
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )

ggsave('calibration_plot.png', plot=p, width=6, height=5, dpi=300)
print(p)

## -----------------------------------------------------------------------------------------------------
hl_test <- hoslem.test(as.numeric(as.character(test_data$Status)), logit_test_preds, g=10)
print(hl_test)


## -----------------------------------------------------------------------------------------------------
cox_formula <- as.formula(paste0('Surv(', time_var, ', ', response_var, ') ~ ', paste(predictors, collapse='+')))
print(cox_formula)


## -----------------------------------------------------------------------------------------------------
train_data_cox <- train_data %>% mutate(Status = as.numeric(as.character(Status)))
test_data_cox <- test_data %>% mutate(Status = as.numeric(as.character(Status)))
cox_model <- coxph(cox_formula, data=train_data_cox)
summary(cox_model)


## -----------------------------------------------------------------------------------------------------
cox_train_preds <- predict(cox_model, train_data_cox, type='lp')
cox_test_preds <- predict(cox_model, test_data_cox, type='lp')

cox_train_preds <- predict(cox_model, train_data_cox, type='risk')
cox_test_preds <- predict(cox_model, test_data_cox, type='risk')


roc_train <- timeROC(
  T = train_data_cox$N_Years,
  delta = train_data_cox$Status,
  marker = cox_train_preds,
  cause = 1,
  times = c()
  
)
print(roc_train$AUC)
hist(predict(cox_model, train_data_cox, type='risk'))


## -----------------------------------------------------------------------------------------------------
calibration_data <- data.frame(Status=as.numeric(as.character(test_data$Status)), RiskScore=cox_test_preds) %>%
  mutate(calib_bin = ntile(RiskScore, 10)) %>%
  group_by(calib_bin) %>%
  summarise(
    mean_predicted = mean(RiskScore),
    observed_rate = mean(Status),
    abs_deviation = abs(mean_predicted - observed_rate)
  )

calibration_data


## -----------------------------------------------------------------------------------------------------
p <- ggplot(calibration_data, aes(x = mean_predicted, y = observed_rate, label = calib_bin)) +
  geom_point(size = 3, color = 'black', fill = 'blue', shape = 21, stroke = 2) +  # Blue dots
  geom_text(vjust = -0.8, size = 4, color = 'black') +  # Label points with bin numbers
  xlim(0, max(max(calibration_data$mean_predicted), max(calibration_data$observed_rate))) +
  ylim(0, max(max(calibration_data$mean_predicted), max(calibration_data$observed_rate))) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed', color = 'red', linewidth = 1) +  # Reference line
  labs(
    title = 'Model Calibration Plot',
    x = 'Predicted Probability',
    y = 'Observed Event Rate'
  ) +
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )

ggsave('calibration_plot_probit.png', plot=p, width=6, height=5, dpi=300)
print(p)


## -----------------------------------------------------------------------------------------------------
hl_test <- hoslem.test(as.numeric(as.character(test_data$Status)), probit_test_preds, g=10)
print(hl_test)


## -----------------------------------------------------------------------------------------------------
time_var <- 'N_Years'
event_var <- 'Status'
treatment_var <- 'Drug'
risk_group_var <- 'RiskGroup'
n_groups <- 3

results_data <- data.frame(ID=test_data$ID, Drug=test_data$Drug, N_Years=test_data$N_Years, Status=as.numeric(as.character(test_data$Status)), RiskScore=cox_test_preds) %>%
  mutate(RiskGroup = as.factor(ntile(RiskScore, n_groups)))
head(results_data)


## -----------------------------------------------------------------------------------------------------
fit_treatment_riskgroup_interacted <- survfit(Surv(N_Years, Status) ~ Drug + RiskGroup, data=results_data)
print(fit_treatment_riskgroup_interacted)


## -----------------------------------------------------------------------------------------------------
# Define survival formulas
model_wo_covariates <- coxph(Surv(N_Years, Status) ~ 1, data = results_data, ties = 'breslow')
model_w_covariates <- coxph(Surv(N_Years, Status) ~ Drug * RiskGroup, data = results_data, ties = 'breslow')


# Model comparison using log-likelihood, AIC, and BIC
tibble(
  Criterion = c('-2 Log L', 'AIC', 'SBC'),
  `Without Covariates` = c(
    -2 * as.numeric(logLik(model_wo_covariates)),
    AIC(model_wo_covariates),
    BIC(model_wo_covariates)
  ),
  `With Covariates` = c(
    -2 * as.numeric(logLik(model_w_covariates)),
    AIC(model_w_covariates),
    BIC(model_w_covariates)
  )
)


## -----------------------------------------------------------------------------------------------------
model_summary <- summary(model_w_covariates)
likelihood_ratio <- model_summary$logtest
score_test <- model_summary$sctest
wald_test <- model_summary$waldtest

tibble(
  Test = c('Likelihood Ratio', 'Score', 'Wald'),
  `Chi Square` = c(likelihood_ratio['test'], score_test['test'], wald_test['test']),
  DF = c(likelihood_ratio['df'], score_test['df'], wald_test['df']),
  `Pr > ChiSq` = c(likelihood_ratio['pvalue'], score_test['pvalue'], wald_test['pvalue'])
)



## -----------------------------------------------------------------------------------------------------
anova_results <- Anova(model_w_covariates, test.statistic = 'Wald', type = '3')

joint_test <- tibble(
  Effect = rownames(anova_results),
  DF = anova_results$DF,
  `Wald Chi-Square` = round(anova_results$Chisq, 4),
  `Pr > ChiSq` = round(anova_results$`Pr(>Chisq)`, 6)
)

joint_test


## -----------------------------------------------------------------------------------------------------
tibble(
  Parameter = rownames(model_summary$coefficients),
  DF = rep(1, length(Parameter)),
  `Parameter Estimate` = round(model_summary$coefficients[, 'coef'], 5),
  `Standard Error` = round(model_summary$coefficients[, 'se(coef)'], 5),
  `Chi-Square` = round((model_summary$coefficients[, 'coef'] / model_summary$coefficients[, 'se(coef)'])^2, 4),
  `Pr > ChiSq` = round(model_summary$coefficients[, 'Pr(>|z|)'], 4),
  `Hazard Ratio` = round(model_summary$coefficients[, 'exp(coef)'], 3),
  `95% CI` = paste0('(', round(model_summary$conf.int[, 'lower .95'], 3), ', ', round(model_summary$conf.int[, 'upper .95'], 3), ')')
)



## -----------------------------------------------------------------------------------------------------
hr_results <- tibble(
  RiskGroup = character(),
  HazardRatio = numeric(),
  LowerCI = numeric(),
  UpperCI = numeric(),
  stringAsFactors = FALSE
)

formula <- as.formula(paste0('Surv(', time_var, ', ', event_var, ') ~ ', treatment_var))

for (groupi in levels(results_data[[risk_group_var]])) {
  subset_data <- results_data %>% filter(!!sym(risk_group_var) == groupi)
  group_model <- coxph(formula, data = subset_data)

  hr <- exp(coef(group_model))
  ci <- exp(confint(group_model))

  hr_results <- rbind(hr_results, tibble(
    RiskGroup = groupi,
    HazardRatio = round(hr, 4),
    LowerCI = round(ci[1], 4),
    UpperCI = round(ci[2], 4)
  ))
}

print(hr_results)


## -----------------------------------------------------------------------------------------------------
p <- ggplot(hr_results, aes(x = RiskGroup, y = HazardRatio)) +
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.2, size = 2, color = 'orange', alpha = 1) +
  geom_point(color = 'black', size = 4, fill = 'blue', shape = 21, stroke = 1) +
  geom_hline(yintercept = 1, linetype = 'solid', color = 'black') +
  scale_y_log10() +
  labs(
    # title = "HTE on the Relative Scale (Hazard Ratio)",
    y = "Hazard Ratio (95% CI)",
    x = "Risk Group"
  ) +
  theme_classic()

# ggsave(file.path(IMG_BASE_DIR, IMG_SUBDIR, 'hr_by_risk_group.png'), plot = p, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)

print(p)


