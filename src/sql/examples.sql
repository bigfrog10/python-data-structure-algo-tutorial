-- over_budget

WITH expense AS (
    SELECT p.id, SUM(e.salary) AS spending
    FROM employees e, employees_projects ep, projects p
    WHERE e.id = ep.employee_id AND p.project_id = p.id
)
SELECT p.id, p.budget, ex.spending * DATEDIFF(p.end_date - p.start_date) / 365.0
From projects p, expense ex
WHERE p.id = ex.id AND ex.spending * DATEDIFF(p.end_date - p.start_date) / 365.0 > p.budget

