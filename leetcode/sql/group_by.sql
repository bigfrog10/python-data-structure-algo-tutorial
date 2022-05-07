-- noinspection SqlNoDataSourceInspectionForFile

-- noinspection SqlDialectInspectionForFile

-- LC1699. Number of Calls Between Two Persons
SELECT
    CASE WHEN from_id > to_id THEN to_id ELSE from_id END AS person1,
    CASE WHEN from_id > to_id THEN from_id ELSE to_id END AS person2,
    COUNT(duration) AS call_count,
    SUM(duration) AS total_duration
FROM Calls
GROUP BY person2,person1

-- 1511. Customer Order Frequency
SELECT customer_id, name
FROM Customers JOIN Orders USING(customer_id)
               JOIN Product USING(product_id)
GROUP BY customer_id
HAVING SUM(IF(LEFT(order_date, 7) = '2020-06', quantity, 0) * price) >= 100
   AND SUM(IF(LEFT(order_date, 7) = '2020-07', quantity, 0) * price) >= 100

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

-- LC597. Friend Requests I: Overall Acceptance Rate
-- https://leetcode.com/problems/friend-requests-i-overall-acceptance-rate/discuss/358575/Detailed-Explaination-for-Question-and-2-follow-up
-- group by doesn't work
select round(
    ifnull(
    (select count(distinct requester_id, accepter_id) from RequestAccepted)
    /
    (select count(distinct sender_id, send_to_id) from FriendRequest),
    0)
, 2) as accept_rate;
-- follow-up 1: Can you write a query to return the accept rate but for every month?
select if(d.req =0, 0.00, round(c.acp/d.req,2)) as accept_rate, c.month from
(select count(distinct requester_id, accepter_id) as acp, YEAR(accept_date) + Month(accept_date) as month from request_accepted) c,
(select count(distinct sender_id, send_to_id) as req, YEAR(request_date) + Month(request_date) as month from friend_request) d
where c.month = d.month
group by c.month

-- follow-up 2: How about the cumulative accept rate for every day?
-- sum up the case when ind is 'a', which means it belongs to accept table, divided by sum of ind is 'r', which means it belong to request table
select s.date1, ifnull(round(sum(case when t.ind = 'a' then t.cnt else 0 end)/sum(case when t.ind = 'r' then t.cnt else 0 end),2), 0)
from
-- get a table of all unique dates
(select distinct x.request_date as date1 from friend_request x
-- The reason here use union sicne we don't want duplicate date
union
 select distinct y.accept_date as date1 from request_accepted y
) s
-- left join to make sure all dates are in the final output
left join
-- get a table of all dates, count of each days, ind to indicate which table it comes from
(select v.request_date as date1, count(*) as cnt,'r' as ind from friend_request v group by v.request_date
-- The reason here use union all sicne union all will be faster
union all
select w.accept_date as date1, count(*) as cnt,'a' as ind from request_accepted w group by w.accept_date) t
-- s.date1 >= t.date1, which for each reacord in s, it will join with all records earlier than it in t
on s.date1 >= t.date1
-- group by s.date1 then we can get a cumulative result to that day
group by s.date1
order by s.date1
