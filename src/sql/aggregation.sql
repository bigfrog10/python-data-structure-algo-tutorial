-- https://learnsql.com/blog/what-is-a-running-total-and-how-to-compute-it-in-sql/
SELECT country, registration_date,registred_users,
  SUM(registred_users)
  OVER (PARTITION BY country ORDER BY registration_date)
  AS total_users
FROM registration;

-- LC579. Find Cumulative Salary of an Employee
SELECT e.id, e.month, SUM(e2.Salary) as Salary
FROM Employee e
JOIN Employee e2 ON (e.Id = e2.Id AND e.Month >= e2.Month AND (e.Month - e2.Month <= 2))
WHERE e.Month < (SELECT MAX(Month) from Employee where Id = e.Id)
GROUP BY e.Id, e.Month
ORDER BY e.Id, e.Month DESC

-- moving average
SELECT date, close, AVG (close) OVER (ORDER BY date DESC RANGE INTERVAL 9 DAY PRECEDING)
FROM tbl
WHERE date <= DATE '2002-07-05'
AND name_id = 2
ORDER BY date DESC
