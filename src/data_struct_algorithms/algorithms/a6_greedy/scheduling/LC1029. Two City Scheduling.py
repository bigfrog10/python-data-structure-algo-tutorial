# LC1029. Two City Scheduling
from typing import List
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # Sort by a gain which company has by sending a person to city A and not to city B
        costs.sort(key = lambda x : x[0] - x[1]) # greedy
        total = 0
        n = len(costs) // 2
        # To optimize the company expenses, send the first n persons to the city A
        # and the others to the city B
        for i in range(n): total += costs[i][0] + costs[i + n][1]
        return total
