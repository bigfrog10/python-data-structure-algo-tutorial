-- noinspection SqlNoDataSourceInspectionForFile

-- noinspection SqlDialectInspectionForFile

-- LC1164. Product Price at a Given Date
select distinct product_id, 10 as price
from Products
group by product_id
having (min(change_date) > "2019-08-16")
union
select p2.product_id, new_price
from Products p2
where (p2.product_id, p2.change_date) in
(
    select product_id, max(change_date) as recent_date
    from Products
    where change_date <= "2019-08-16"
    group by product_id
)

-- LC182. Duplicate Emails
SELECT Email
from Person
group by Email having count(*) > 1

-- LC569. Median Employee Salary
SELECT MIN(A.Id) AS Id, A.Company, A.Salary
FROM Employee A, Employee B
WHERE A.Company = B.Company
GROUP BY A.Company, A.Salary
HAVING SUM(CASE WHEN B.Salary >= A.Salary then 1 ELSE 0 END) >= COUNT(*)/2
AND SUM(CASE WHEN B.Salary <= A.Salary then 1 ELSE 0 END) >= COUNT(*)/2
-- Another way
with t1 as(
    select *, row_number() over(partition by Company order by Salary) as row1,
    count(Id) over(partition by Company) as cnt
    from Employee
)
select Id, Company, Salary
from t1
where row1 between cnt/2.0 and cnt/2.0+1;

-- LC1112. Highest Grade For Each Student
SELECT DISTINCT student_id, MIN(course_id) AS course_id, grade
FROM Enrollments
WHERE (student_id,grade) IN (
    SELECT DISTINCT student_id, max(grade) FROM Enrollments GROUP BY student_id
)
GROUP BY student_id ORDER BY student_id;

-- LC614. Second Degree Follower
Select f1.follower, count(distinct f2.follower) as num
from follow f1 inner join follow f2 on f1.follower = f2.followee
Group by f1.follower

-- LC1050. Actors and Directors Who Cooperated At Least Three Times
SELECT actor_id, director_id
FROM ActorDirector
GROUP BY actor_id, director_id
HAVING COUNT(1) >= 3

-- LC570. Managers with at Least 5 Direct Reports
WITH managers AS (
    SELECT ManagerId
    FROM Employee
    GROUP BY ManagerId
    HAVING COUNT(ManagerId) >= 5
)
SELECT Name
FROM Employee e JOIN managers m ON e.id = m.ManagerId

-- LC574. Winning Candidate
SELECT name AS 'Name'
FROM Candidate JOIN
    (SELECT Candidateid
    FROM Vote
    GROUP BY Candidateid
    ORDER BY COUNT(*) DESC
    LIMIT 1) AS winner
WHERE Candidate.id = winner.Candidateid

-- LC1077. Project Employees III
WITH exp AS (
    SELECT project_id, max(experience_years) AS experience_years
    FROM Project p, Employee e
    WHERE p.employee_id = e.employee_id
    GROUP BY project_id
)
SELECT p.project_id, p.employee_id
FROM Project p, Employee e, exp
WHERE p.employee_id = e.employee_id AND p.project_id = exp.project_id
    AND e.experience_years = exp.experience_years

-- LC1076. Project Employees II
SELECT project_id
FROM project
GROUP BY project_id
HAVING COUNT(employee_id) =
(
    SELECT count(employee_id)
    FROM project
    GROUP BY project_id
    ORDER BY count(employee_id) desc
    LIMIT 1
)

-- LC1303. Find the Team Size
SELECT employee_id, team_size
FROM Employee AS e
 JOIN (
      SELECT team_id, COUNT(employee_id) AS team_size
      FROM Employee
      GROUP BY team_id
) AS teams
ON e.team_id = teams.team_id

-- LC619. Biggest Single Number
SELECT MAX(num) AS num
FROM (SELECT num FROM my_numbers
        GROUP BY num HAVING COUNT(num) = 1) AS t

-- LC1045. Customers Who Bought All Products
select customer_id
from customer c
group by customer_id
having count(distinct product_key)=(select count(distinct product_key) from product)
