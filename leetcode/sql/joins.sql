-- https://www.guru99.com/joins.html

-- 1581. Customer Who Visited but Did Not Make Any Transactions
SELECT
    customer_id,
    COUNT(*) AS count_no_trans
FROM
    Visits AS v
        LEFT JOIN Transactions AS t ON v.visit_id = t.visit_id
WHERE
    t.visit_id IS NULL
GROUP BY
    customer_id

-- LC175. Combine Two Tables
SELECT a.FirstName, a.LastName, b.City, b.State
FROM Person a LEFT OUTER JOIN Address b ON a.PersonId = b.PersonId

-- LC181. Employees Earning More Than Their Managers
SELECT a.Name AS Employee
FROM Employee a LEFT JOIN Employee b ON a.ManagerId = b.Id
WHERE a.Salary > b.Salary

-- LC184. Department Highest Salary
With max_tbl as
(
    Select DepartmentId, max(Salary) as Salary
    from Employee
    group by DepartmentId
)
select b.Name as Department, a.Name as Employee, c.Salary as Salary
from Employee as a, Department as b, max_tbl as c
where a.DepartmentId = b.Id and b.Id = c.DepartmentId and a.Salary = c.Salary

-- LC196. Delete Duplicate Emails
WITH deletes AS (SELECT min(Id) as id FROM Person GROUP BY Email)
DELETE FROM Person WHERE Id NOT IN (SELECT id FROM deletes);

-- LC180. Consecutive Numbers
SELECT DISTINCT l1.Num AS ConsecutiveNums
FROM
    Logs l1,
    Logs l2,
    Logs l3
WHERE
    l1.Id = l2.Id - 1
    AND l2.Id = l3.Id - 1
    AND l1.Num = l2.Num
    AND l2.Num = l3.Num

-- LC197. Rising Temperature
SELECT weather.id AS 'Id'
FROM weather JOIN weather w ON DATEDIFF(weather.recordDate, w.recordDate) = 1
    AND weather.Temperature > w.Temperature

-- LC183. Customers Who Never Order
SELECT Name as Customers
FROM Customers as a LEFT JOIN Orders b on a.id = b.CustomerId
WHERE b.id is NULL

-- LC1398. Customers Who Bought Products A and B but Not C
select a.customer_id, a.customer_name
from customers a , orders b
where a.customer_id  = b.customer_id
group by a.customer_id
having sum(b.product_name="A") >0 and sum(b.product_name="B") > 0 and sum(b.product_name="C")=0

