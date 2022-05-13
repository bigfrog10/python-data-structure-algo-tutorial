-- LC1322. Ads Performance
SELECT ad_id, ROUND(CASE
    WHEN Clicked + Viewed = 0 THEN 0
    ELSE 100 * Clicked / (Clicked + Viewed) END, 2) as ctr
FROM
( SELECT ad_id,
  SUM(CASE WHEN action = 'Clicked' THEN 1 ELSE 0 END) as Clicked,
  SUM(CASE WHEN action = 'Viewed' THEN 1 ELSE 0 END) as Viewed
  FROM Ads
  GROUP BY ad_id
) as agg
ORDER BY ctr DESC, ad_id ASC

-- LC626. Exchange Seats
select case
    when id % 2 = 0 then id - 1
    when id % 2 = 1 and id < (select count(*) from seat) then id + 1
    else id
end as id, student
from seat
order by id;

-- LC1179. Reformat Department Table
SELECT
    id,
    sum( if( month = 'Jan', revenue, null ) ) AS Jan_Revenue,
    sum( if( month = 'Feb', revenue, null ) ) AS Feb_Revenue,
    sum( if( month = 'Mar', revenue, null ) ) AS Mar_Revenue,
    sum( if( month = 'Apr', revenue, null ) ) AS Apr_Revenue,
    sum( if( month = 'May', revenue, null ) ) AS May_Revenue,
    sum( if( month = 'Jun', revenue, null ) ) AS Jun_Revenue,
    sum( if( month = 'Jul', revenue, null ) ) AS Jul_Revenue,
    sum( if( month = 'Aug', revenue, null ) ) AS Aug_Revenue,
    sum( if( month = 'Sep', revenue, null ) ) AS Sep_Revenue,
    sum( if( month = 'Oct', revenue, null ) ) AS Oct_Revenue,
    sum( if( month = 'Nov', revenue, null ) ) AS Nov_Revenue,
    sum( if( month = 'Dec', revenue, null ) ) AS Dec_Revenue
FROM Department
GROUP BY id;

-- LC1777. Product's Price for Each Store
select product_id,
    sum(case when store = 'store1' then price else null end) store1,
    sum(case when store = 'store2' then price else null end) store2,
    sum(case when store = 'store3' then price else null end) store3
from products
group by product_id

-- LC608. Tree Node
# Write your MySQL query statement below
SELECT DISTINCT a.id, CASE
    WHEN a.p_id IS NULL THEN 'Root'
    WHEN b.id IS NULL THEN 'Leaf'
    ELSE 'Inner'
    END AS Type
FROM tree a
LEFT JOIN tree b ON a.id = b.p_id
ORDER BY a.id
