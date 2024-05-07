library(dplyr)
library(Hmisc) #Describe Function
library(psych) #Multiple Functions for Statistics and Multivariate Analysis
library(foreign)
library(CCA)
library(yacca)
library(MASS)
library(corrplot) #Plot Correlations
library(REdaS) #Bartlett's Test of Sphericity
library(psych) #PCA/FA functions
library(factoextra) #PCA Visualizations
library("FactoMineR") #PCA functions
library(ade4) #PCA Visualizations
options("scipen"=100, "digits"=5)

setwd("/home/georgetz/Desktop/Classes/Summer 2023/DSC 324 - Advanced Data Analysis/Final Project")
df <- read.csv('Final_Life ExpectancyData.csv')
df$Status <- factor(df$Status, levels = c('Developing', 'Developed'), labels = c(0, 1))
numeric <- df %>%
  dplyr::select(-Year, -Country, -Status, -Life.expectancy)

head(numeric)


# Testing for significant correlations

MCorrTest = corr.test(numeric, adjust="none")
M <-MCorrTest$p
MTest = ifelse(M < .01, T, F)
print(colSums(MTest) - 1 )

corrplot(cor(numeric), diag=F, title='Correlation Matrix')


#  Kaiser-Meyer-Olkin (KMO) test to assess the adequacy of the sample for conducting FA
# The KMO test evaluates the level of common variance among variables and whether it is suitable for extracting meaningful factors.
# The KMO test addresses the question of whether the observed variables are likely to be influenced by underlying latent factors.
# If the variables are indeed influenced by common factors, factor analysis is more likely to yield meaningful results.
# KMO Value close to 1: Indicates that the observed variables are highly correlated and suitable for factor analysis.
# A high KMO value suggests that the data is appropriate for extracting underlying factors.
KMO(numeric)

# Bartlett's Test of Sphericity is a statistical test used in the context of factor analysis to determine whether 
# the correlation matrix of observed variables is significantly different from an identity matrix.
# It assesses whether the variables in your dataset are related enough to warrant conducting factor analysis.
# If the p-value associated with the test statistic is below the chosen significance level (e.g., p-value < 0.05), 
# you reject the null hypothesis. This indicates that the correlation matrix is significantly different from an identity matrix,
# implying that the variables are correlated and factor analysis might be appropriate.
bart_spher(numeric)
# Scaling the data for Cronbach's Alpha Test
scaled_data <- scale(numeric)
cron_alpha <-psych::alpha(scaled_data, check.keys=TRUE)
cron_alpha$total


p = prcomp(numeric, center=T, scale=T)
summary(p)

# The first 6 components have an eigenvalue greater than 1.
# The first 3 principal components have a cumulative variance proportion of 0.5428 (54.28%). 

plot(p, main='Principal Component Explained Variance')
abline(1, 0)
# Using the elbow method, we will run Factor Analysis with 3 principal components.

# Since the principal components are uncorrelated linear combinations of the original variables,
# we will use VARIMAX rotation to simplify the feature contributions for each component.

p2 = psych::principal(numeric, rotate="varimax", nfactors=3, scores=TRUE)
print(p2$loadings, cutoff=.4, sort=T)

# The first factor seems to be loaded with features regarding the availability of resources in the country (**resources**).  
# The largest factor loadings include GDP (0.724), schooling (0.733) and Income.composition.of.resources (0.7). 
# As economic resources become more widely available, nutritional sustenance goes up for the population.  

# The second factor is loaded with features regarding how fast the population is increasing or decreasing (**population growth**). 
# It is interesting to note that population size and measurements of death (such as infant deaths, measles deaths, etc) have a
# positive relationship with the second factor. This could be due to the death measurements increasing proportionally as the 
# population size increases.  

# The third factor is loaded with features regarding vaccination rates of infants (**infant immunization**).   
# Although the factor loadings seem to have a degree of intuitive sense, they factors cannot be relied upon due to 
# the very low Cronbach's Alpha score of 0.00011332.
p2$loadings



