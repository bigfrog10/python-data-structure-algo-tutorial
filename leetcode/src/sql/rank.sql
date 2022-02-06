-- LC178. Rank Scores
SELECT Score, DENSE_RANK() OVER(ORDER BY Score DESC) AS `Rank`
FROM Scores
ORDER BY `Rank`
# Rank is reserved word. Rank() has gaps and Dense_rank() does not.

-- LC1285. Find the Start and End Number of Continuous Ranges
SELECT min(log_id) as start_id, max(log_id) as end_id
FROM ( SELECT log_id, RANK() OVER(ORDER BY log_id) as num FROM Logs ) a
GROUP BY log_id - num
