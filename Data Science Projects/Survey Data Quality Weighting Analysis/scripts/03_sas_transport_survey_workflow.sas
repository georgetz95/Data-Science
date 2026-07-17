/* NHANES 2017-2018 survey workflow in SAS.
   Update ROOT before running in a local SAS session. */

%let root = C:\Users\georg\Desktop\jobs\Survey_Data_Quality_Weighting_Analysis;

libname demo_xpt xport "&root.\data\raw\DEMO_J.XPT";
libname huq_xpt  xport "&root.\data\raw\HUQ_J.XPT";
libname hiq_xpt  xport "&root.\data\raw\HIQ_J.XPT";
libname outlib "&root.\data\processed";

proc copy in=demo_xpt out=work;
run;

proc copy in=huq_xpt out=work;
run;

proc copy in=hiq_xpt out=work;
run;

proc sort data=work.demo_j; by seqn; run;
proc sort data=work.huq_j; by seqn; run;
proc sort data=work.hiq_j; by seqn; run;

data outlib.nhanes_adult_access_analysis_sas;
  merge work.demo_j(in=in_demo)
        work.huq_j
        work.hiq_j;
  by seqn;
  if in_demo and ridageyr >= 18;

  length gender $6 race_ethnicity $22 education $23
         insurance_status $9 usual_care $17;

  if riagendr = 1 then gender = "Male";
  else if riagendr = 2 then gender = "Female";

  select (ridreth3);
    when (1) race_ethnicity = "Mexican American";
    when (2) race_ethnicity = "Other Hispanic";
    when (3) race_ethnicity = "Non-Hispanic White";
    when (4) race_ethnicity = "Non-Hispanic Black";
    when (6) race_ethnicity = "Non-Hispanic Asian";
    when (7) race_ethnicity = "Other/multiracial";
    otherwise race_ethnicity = "";
  end;

  if dmdeduc2 in (1, 2) then education = "Less than high school";
  else if dmdeduc2 = 3 then education = "High school/GED";
  else if dmdeduc2 = 4 then education = "Some college/AA";
  else if dmdeduc2 = 5 then education = "College graduate+";

  if 0 <= indfmpir <= 5 then poverty_ratio = indfmpir;

  if huq010 in (4, 5) then fair_poor = 1;
  else if huq010 in (1, 2, 3) then fair_poor = 0;

  if hiq011 = 1 then insurance_status = "Insured";
  else if hiq011 = 2 then insurance_status = "Uninsured";

  if huq030 in (1, 3) then usual_care = "Has usual place";
  else if huq030 = 2 then usual_care = "No usual place";
run;

proc means data=outlib.nhanes_adult_access_analysis_sas n nmiss;
  var fair_poor poverty_ratio wtint2yr sdmvstra sdmvpsu;
run;

proc surveyfreq data=outlib.nhanes_adult_access_analysis_sas;
  strata sdmvstra;
  cluster sdmvpsu;
  weight wtint2yr;
  tables insurance_status*fair_poor usual_care*fair_poor / row cl;
run;

proc surveylogistic data=outlib.nhanes_adult_access_analysis_sas;
  strata sdmvstra;
  cluster sdmvpsu;
  weight wtint2yr;
  class insurance_status(ref="Insured")
        usual_care(ref="Has usual place")
        gender(ref="Female")
        race_ethnicity(ref="Non-Hispanic White")
        education(ref="College graduate+") / param=ref;
  model fair_poor(event="1") =
        insurance_status usual_care ridageyr gender race_ethnicity
        education poverty_ratio;
run;
