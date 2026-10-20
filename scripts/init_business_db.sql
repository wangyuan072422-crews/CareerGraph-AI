-- Initialize the read-only business database used by the Database Agent.
-- Run this with a MySQL administrator account, not with career_app.

CREATE DATABASE IF NOT EXISTS career_assistant
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE career_assistant;

CREATE TABLE IF NOT EXISTS job_applications (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  company VARCHAR(255) NOT NULL,
  role_title VARCHAR(255) NOT NULL,
  status VARCHAR(32) NOT NULL,
  applied_at DATETIME NOT NULL,
  PRIMARY KEY (id),
  KEY idx_job_applications_status (status),
  KEY idx_job_applications_applied_at (applied_at)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;

-- The existing schema has no unique business key, so use INSERT ... SELECT
-- checks to make the demo seed safe to run repeatedly.
INSERT INTO job_applications (company, role_title, status, applied_at)
SELECT 'Acme Cloud', 'AI Application Intern', 'interviewing', '2026-09-18 10:00:00'
WHERE NOT EXISTS (
  SELECT 1
  FROM job_applications
  WHERE company = 'Acme Cloud'
    AND role_title = 'AI Application Intern'
    AND applied_at = '2026-09-18 10:00:00'
);

INSERT INTO job_applications (company, role_title, status, applied_at)
SELECT 'Northstar Analytics', 'Backend Engineering Intern', 'applied', '2026-09-25 14:30:00'
WHERE NOT EXISTS (
  SELECT 1
  FROM job_applications
  WHERE company = 'Northstar Analytics'
    AND role_title = 'Backend Engineering Intern'
    AND applied_at = '2026-09-25 14:30:00'
);

INSERT INTO job_applications (company, role_title, status, applied_at)
SELECT 'Orbit Labs', 'Data Platform Intern', 'offer', '2026-08-30 09:15:00'
WHERE NOT EXISTS (
  SELECT 1
  FROM job_applications
  WHERE company = 'Orbit Labs'
    AND role_title = 'Data Platform Intern'
    AND applied_at = '2026-08-30 09:15:00'
);
