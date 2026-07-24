DROP VIEW IF EXISTS vw_dashboard_kpis;
CREATE VIEW vw_dashboard_kpis AS
SELECT metric, display_value
FROM dashboard_kpis;

DROP VIEW IF EXISTS vw_chronic_absence_school_trend;
CREATE VIEW vw_chronic_absence_school_trend AS
SELECT
  year,
  school_name,
  enrollment,
  students_chronically_absent,
  chronic_absence_rate
FROM chronic_absenteeism_school_trend
WHERE enrollment > 0
ORDER BY year, chronic_absence_rate DESC;

DROP VIEW IF EXISTS vw_subgroup_equity_review;
CREATE VIEW vw_subgroup_equity_review AS
SELECT
  school_name,
  race_label,
  enrollment_crdc,
  students_chronically_absent,
  chronic_absence_rate,
  stable_reporting_group
FROM subgroup_chronic_absenteeism_2021
WHERE stable_reporting_group = 1
ORDER BY school_name, chronic_absence_rate DESC;

DROP VIEW IF EXISTS vw_behavior_support_monitoring;
CREATE VIEW vw_behavior_support_monitoring AS
SELECT
  year,
  school_name,
  enrollment,
  suspensions_instances,
  suspension_instances_per_100
FROM discipline_instances_school_trend
WHERE enrollment > 0
ORDER BY year, suspension_instances_per_100 DESC;

DROP VIEW IF EXISTS vw_mtss_priority_queue;
CREATE VIEW vw_mtss_priority_queue AS
SELECT
  priority_rank,
  school_name,
  enrollment,
  chronic_absence_rate,
  suspension_instances_per_100,
  frpl_share,
  mtss_priority_score,
  dashboard_action
FROM mtss_priority_school_table
ORDER BY priority_rank;

DROP VIEW IF EXISTS vw_data_quality_checks;
CREATE VIEW vw_data_quality_checks AS
SELECT "check", value
FROM data_quality_checks;
