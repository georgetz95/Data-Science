* Importing the dataset;
proc import datafile="TestScores.csv" out=test_scores replace;
delimiter=",";
getnames=yes;
run;

* Create dummy variables for the categorical features;
data test_scores replace;
set test_scores;
n_gender = (gender = "male");
n_ethnicity_hispanic = (ethnicity = "hispa");
n_ethnicity_afam = (ethnicity = "afam");
n_fcollege = (fcollege = "yes");
n_mcollege = (mcollege = "yes");
n_home = (home = "yes");
n_urban = (urban = "yes");
n_income = (income = "high");
n_region = (region = "west");
run;

*Print first 10 observations;
proc print data=test_scores (obs=10);
title "First 10 Observations";
run;

* Distribution of score;
proc univariate data=test_scores;
title "score Histogram";
var score;
histogram /normal (mu=est sigma=est);
run;

*Distribution of unemp;
proc univariate data=test_scores;
title "unemp Histogram";
var unemp;
histogram /normal (mu=est sigma=est);
run;

*Log-Transformation of unemp;
data test_scores replace;
set test_scores;
ln_unemp = log(unemp);
run;

*Distribution of ln_unemp;
proc univariate data=test_scores;
title "ln_unemp Histogram";
var ln_unemp;
histogram /normal (mu=est sigma=est);
run;

*Distribution of wage;
proc univariate data=test_scores;
title "wage Histogram";
var wage;
histogram /normal (mu=est sigma=est);
run;

*Distribution of distance;
proc univariate data=test_scores;
title "distance Histogram";
var distance;
histogram /normal (mu=est sigma=est);
run;

*Log-Transformation of distance;
data test_scores replace;
set test_scores;
ln_distance = log(distance);
run;

*Distribution of ln_distance;
proc univariate data=test_scores;
title "ln_distance Histogram";
var ln_distance;
histogram /normal (mu=est sigma=est);
run;

*Distribution of tuition;
proc univariate data=test_scores;
title "tuition Histogram";
var tuition;
histogram /normal (mu=est sigma=est);
run;

* Categorical Variable Bar Plots;

* home Bar Plot;
proc sgplot data=test_scores;
title "home Bar Plot";
vbar home;
run;

* urban Bar Plot;
proc sgplot data=test_scores;
title "urban Bar Plot";
vbar urban;
run;

* gender Bar Plot;
proc sgplot data=test_scores;
title "gender Bar Plot";
vbar gender;
run;

* gender Bar Plot;
proc sgplot data=test_scores;
title "ethnicity Bar Plot";
vbar ethnicity;
run;

* mcollege Bar Plot;
proc sgplot data=test_scores;
title "mcollege Bar Plot";
vbar mcollege;
run;

* fcollege Bar Plot;
proc sgplot data=test_scores;
title "fcollege Bar Plot";
vbar fcollege;
run;

* income Bar Plot;
proc sgplot data=test_scores;
title "income Bar Plot";
vbar income;
run;

* region Bar Plot;
proc sgplot data=test_scores;
title "region Bar Plot";
vbar region;
run;

* education Bar Plot;
proc sgplot data=test_scores;
title "education Bar Plot";
vbar education;
run;

* Numeric Variable Scatter Plots;

proc sgplot;
title "ln_unemp/score Scatter Plot";
scatter x=ln_unemp y=score;
run;

proc sgplot;
title "wage/score Scatter Plot";
scatter x=wage y=score;
run;

proc sgplot;
title "ln_distance/score Scatter Plot";
scatter x=ln_distance y=score;
run;

proc sgplot;
title "tuition/score Scatter Plot";
scatter x=tuition y=score;
run;

*Categorical Variable Box Plots;

*Gender Box Plot;
proc sort;
by gender;
run;

proc boxplot;
title "Scores by Gender";
ods graphics off;
plot score * gender;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Gender';
run;

* Ethnicity Box Plot;
proc sort;
by ethnicity;
run;

proc boxplot;
title "Scores by Ethnicity";
ods graphics off;
plot score * ethnicity;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Ethnicity';
run;

* fcollege Box Plot;
proc sort;
by fcollege;
run;

proc boxplot;
title "Scores by Father Graduation Status";
ods graphics off;
plot score * fcollege;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Father Graduation Status';
run;

* mcollege Box Plot;
proc sort;
by mcollege;
run;

proc boxplot;
title "Scores by Mother Graduation Status";
ods graphics off;
plot score * mcollege;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Mother Graduation Status';
run;

* home Box Plot;
proc sort;
by home;
run;

proc boxplot;
title "Scores by Home Ownership";
ods graphics off;
plot score * home;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Home Ownership';
run;

* urban Box Plot;
proc sort;
by urban;
run;

proc boxplot;
title "Scores by Urban School Area";
ods graphics off;
plot score * urban;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Urban School Area';
run;

* income Box Plot;
proc sort;
by income;
run;

proc boxplot;
title "Scores by Income Category";
ods graphics off;
plot score * income;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Income Category';
run;

* region Box Plot;
proc sort;
by region;
run;

proc boxplot;
title "Scores by Region";
ods graphics off;
plot score * region;
inset min mean max stddev/
header = 'Overall Statistics'
pos = tm;
insetgroup min mean Q1 Q2 Q3 max range stddev/
header = 'Statistics by Region';
run;

*Print first 10 observations;
proc print data=test_scores (obs=10);
title "First 10 Observations";
run;

* Full Linear Regression Model;
proc reg data=test_scores;
title "Full Linear Regression Model";
model score= n_gender n_ethnicity_hispanic n_ethnicity_afam n_fcollege n_mcollege n_home n_urban n_income n_region ln_unemp ln_distance wage tuition education /vif tol;
run;           



