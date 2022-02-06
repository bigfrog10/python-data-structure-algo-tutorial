-- LC1667. Fix Names in a Table
select user_id, concat(upper(substr(name,1,1)), lower(substr(name,2))) as name
from Users
order by 1

-- LC620. Not Boring Movies
SELECT id, movie, description, rating
FROM cinema
WHERE id % 2 = 1 AND description != 'boring'
ORDER BY rating DESC

-- LC595. Big Countries
SELECT name, population, area
FROM World
WHERE area > 3000000 OR population > 25000000
