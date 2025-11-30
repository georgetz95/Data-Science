# Load necessary libraries
library(ltm)          # For Point-Biserial correlation
library(lsr)          # For Eta Squared
library(dplyr)        # For data manipulation
library(tidyr)        # For handling missing values
library(DescTools)    # For Cramér's V & Phi coefficient
library(reshape2)

# Function to compute correlation matrix for mixed data types
compute_mixed_corr <- function(df, filter_p = FALSE) {
  df <- df %>% drop_na()  # Remove rows with missing values
  cols <- colnames(df)
  n <- length(cols)
  
  # Initialize correlation matrix
  cor_matrix <- matrix(NA, nrow = n, ncol = n, dimnames = list(cols, cols))
  
  # Define correlation functions
  calc_pearson <- function(x, y) cor(x, y, use = "complete.obs", method = "pearson")
  calc_point_biserial <- function(num, bin) biserial.cor(num, as.numeric(bin), level = 2)  # Ensure binary is numeric
  calc_phi <- function(bin1, bin2) Phi(table(bin1, bin2))  # Fix: Use Phi() from DescTools
  calc_cramers_v <- function(cat1, cat2) CramerV(table(cat1, cat2))
  calc_eta_squared <- function(num, cat) {
    if (length(unique(cat)) > 1) {
      eta_sq <- etaSquared(aov(num ~ cat), type = 2)
      return(as.numeric(eta_sq[1]))  # Convert to numeric
    } else {
      return(NA)
    }
  }
  
  # Identify feature types
  numeric_features <- names(df)[sapply(df, is.numeric)]
  binary_features <- names(df)[sapply(df, function(x) is.factor(x) && nlevels(x) == 2)]
  categorical_features <- names(df)[sapply(df, function(x) is.factor(x) && nlevels(x) > 2)]
  
  # Compute correlation values
  for (i in 1:n) {
    for (j in i:n) {
      var1 <- df[[cols[i]]]
      var2 <- df[[cols[j]]]
      
      # Self-correlation
      if (i == j) {
        cor_matrix[i, j] <- 1
        next
      }
      
      # Numeric-Numeric: Pearson
      if (cols[i] %in% numeric_features && cols[j] %in% numeric_features) {
        cor_matrix[i, j] <- cor_matrix[j, i] <- calc_pearson(var1, var2)
        
        # Binary-Numeric: Point-Biserial
      } else if ((cols[i] %in% binary_features && cols[j] %in% numeric_features) ||
                 (cols[j] %in% binary_features && cols[i] %in% numeric_features)) {
        if (cols[i] %in% binary_features) {
          cor_matrix[i, j] <- cor_matrix[j, i] <- calc_point_biserial(var2, var1)
        } else {
          cor_matrix[i, j] <- cor_matrix[j, i] <- calc_point_biserial(var1, var2)
        }
        
        # Binary-Binary: Phi
      } else if (cols[i] %in% binary_features && cols[j] %in% binary_features) {
        cor_matrix[i, j] <- cor_matrix[j, i] <- calc_phi(var1, var2)
        
        # Categorical-Categorical: Cramér’s V
      } else if (cols[i] %in% categorical_features && cols[j] %in% categorical_features) {
        cor_matrix[i, j] <- cor_matrix[j, i] <- calc_cramers_v(var1, var2)
        
        # Binary-Categorical: Cramér’s V
      } else if ((cols[i] %in% binary_features && cols[j] %in% categorical_features) ||
                 (cols[j] %in% binary_features && cols[i] %in% categorical_features)) {
        cor_matrix[i, j] <- cor_matrix[j, i] <- calc_cramers_v(var1, var2)
        
        # Numeric-Categorical: Eta-Squared
      } else if ((cols[i] %in% numeric_features && cols[j] %in% categorical_features) ||
                 (cols[j] %in% numeric_features && cols[i] %in% categorical_features)) {
        if (cols[i] %in% numeric_features) {
          cor_matrix[i, j] <- cor_matrix[j, i] <- calc_eta_squared(var1, var2)
        } else {
          cor_matrix[i, j] <- cor_matrix[j, i] <- calc_eta_squared(var2, var1)
        }
      }
    }
  }
  
  # Convert correlation matrix to data frame
  cor_df <- as.data.frame(cor_matrix)
  cor_df <- round(cor_df, 2)
  
  # Optional: Filter correlations below p < 0.05
  if (filter_p) {
    cor_df[abs(cor_df) < 0.05] <- NA
  }
  
  return(cor_df)
}
