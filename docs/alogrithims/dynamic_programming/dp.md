
## Optimal substructure:
The optimal solution can be easily constructured efficiently from optimal 
solutions of its sub-problems.

In other word, the sub-solution of an optimal solution is the optimal solution 
of the corresponding sub-problem.

## Overlapping sub-problems
This leads to caching, and thus save time.
This leads to result selection, such as min, max, etc.

backpack problem

## Graphs
Travelling Salesman Problem
Hamiltonian Cycle (backtrack)
In a complete graph, there are (n-1)!/2 different cycles (different path ).
First, we fix one starting point to prevent duplicates (a cycle can start at any
node). So there are only (n-1) nodes left. Since a cycle inverse is the same
cycle, so we divide it by 2.

https://www.baeldung.com/cs/tsp-dynamic-programming
https://www.tutorialspoint.com/design_and_analysis_of_algorithms/design_and_analysis_of_algorithms_travelling_salesman_problem.htm

The hard part of DP is to generalize the problem in the right dimensions.
