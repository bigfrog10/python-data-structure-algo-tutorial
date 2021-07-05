# Graphs

Directed or undirected.

adjacent matrix representation and dict representation.

References
- [Top 10 Graph Algorithms](https://www.youtube.com/watch?v=RqQBh_Wbcu4)
- Scope: https://web.stanford.edu/class/cs97si/ + shortest path
- Many problems here:  
  https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/#maximumFlow
- https://en.wikipedia.org/wiki/List_of_algorithms#Graph_algorithms
- https://algs4.cs.princeton.edu/41graph/

## Connectivity and Traversal
___
### DFS Traversal
To find numbers of connected components in a graph, the recursion is
```
def traverse_dfs(node):
    if not node.visit:
        node.visit = True
        for n in node.neghbours:
            traverse_dfs(n)
            
def find_num_comps(graph):
    num = 0
    for node in graph:
        if not node.visit:
            traverse_dfs(node)
            num += 1
    return num
```
Runtime is Θ(|V| + |E|) because we visit all vertices and there is a neighbour loop coming from edges.

LC 323

### Find-Union
To find numbers of connected components in a graph. Here we assume nodes
are labeled as 0 ... n and edges are labeled as (i, j).  
Assume we have a disjoint set implementation (see source)
```
def find_num_comps(graph):
    djs = djs(graph.nodes)
    for edge in graph:
        djs.union(edge[0], edge[1])
        
    roots = set()
    for n in graph.nodes:
        roots.add(djs.find(n))
    return len(roots)
```
DJS runtime is (N + M) lg* N for M operations, so we have (N + N) lg* N ≈ O(N)

#### Disjoint Sets or Union Find

disjoint sets and union find
https://www.cs.princeton.edu/~rs/AlgsDS07/
https://opendsa-server.cs.vt.edu/ODSA/Books/Everything/html/UnionFind.html
https://en.wikipedia.org/wiki/Disjoint-set_data_structure

### Strongly Connected Components(SCCs) in directed graphs 
A component is [strongly connected](https://en.wikipedia.org/wiki/Strongly_connected_component) 
if every vertex in the component is reachable from all other vertices.

Every node itself is a SCC, but we want the largest possibles. Every directed
graph can be decomposed to SCCs.

For undirected graphs, every connected component is also strong connected.
Since there is nothing special in this case, we concentrate on directed graphs.

https://www.youtube.com/watch?v=HOOmetF56BI

- Kosaraju's Algorithm  
  
  start from a vertex, use DFS. save to another stack after DFS call, so that
  end nodes are at the bottom of the stack.
  Then reverse directions. Pop top node in stack and traverse, once it reaches
  the current top node in stack, the traversed nodes form a strongly connected
  components.
  
https://www.youtube.com/watch?v=Rs6DXyWpWrI
```
def kosaraju(graph):
    def dfs1(node, stack):
        if not node.visit:
            node.visit = True
            for n in node.neighbours:
                dfs1(node, stack)
                
            stack.push(node)
    
    stack = []
    for node in graph:
        dfs1(node, stack)
    
    # reverse directions
    for node in graph:
        for n in node.neighbours:
            n.neighbours.append(node)
            node.neighbours.remove(n)
            
    for n in graph:
        n.visit = False  # reset
        
    def dfs2(node, scc):
        if not node.visit:
            node.visit = True
            scc.append(node)
            for n in node.neighbours:
                dfs2(node.scc)
    
        
    while stack:
        node = stack.pop()
        if not node.visit:
            scc = []
            dfs2(node, scc)
            print(scc)  # this is one scc
```

The runtime is O(|V| + |E|)

- [Tarjan's algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)

  This is one pass(though stack usage does not help on speed in worst case) 
  illustrated on [YouTube](https://www.youtube.com/watch?v=wUgWX0nc4NY).
  It's tedious to keep track states, we need ids, low link values, stack, and 
  on_stack. We update low_links on the way back(i.e., after dfs() is called).  
```
def tarjan(graph): 
    def dfs(node, id, stack):
        if node.visit:
            return
        node.visit = True
        
        stack.push(node) # state management
        node.on_stack = True
        node.id = id       
        node.low_link = id
        id += 1
        
        for n in node.neighbours:  # visit neighbour
            dfs(n, id, stack)  # recursion
            if n.on_stack: # backtrack low_links
                n.low_link = min(n.low_link, node.low_link)
        
        if node.id == node.low_link: # found SCC root, first element
            while stack: # bump out all nodes in current SCC
                n = stack.pop()
                n.on_stack = False
                n.low_link = node.id  # we may choose to save n to the component list, which is defined outside while loop
                if n == node:
                    break
                    
    stack = []
    id = 0
    for node in graph:
        dfs(node, id, stack)
        
    # now node.low_link has SCC id
```

### Bridge and Articulation Points
A bridge in a graph is an edge whose removal cause its component disconnected,
and thus the number of components in the graph increases.

Similarly, an articulation point is a node whose removal cause its component 
disconnected.

Bridges exist between SCCs and we can use similar algorithms to find them.
The trivial solution is to remove each edge and see whether numer of components
increased. This runtime is O(E*(V+E)). Tarjan's method will run O(V+E).
https://algs4.cs.princeton.edu/41graph/Bridge.java.html
https://www.youtube.com/watch?v=erlX-1MJlv8
```
def detect_bridges(graph):



    stack = []
    id = 0
    for node in graph:
        
```

https://cp-algorithms.com/graph/bridge-searching.html


LC 1192

### Cycle Detection
To find whether there is a cycle in a graph, use DFS:
```
def dfs(node, stack):
    if not node.visit:
        node.visit = True
        stack.push(node)
        for n in node.neighbours:
            res = dfs(n, stack)
            if res:
                return True
        
        return False    
    else:
         return True
                
def detect_cycles(grpah):
    stack = []
    for node in graph:
        res = dfs(node, stack)
            if res:
                return True
        
    return False   
```

To find all cycles in a graph, use DFS to traverse graph and backtrack to collect cycles  
https://stackoverflow.com/questions/12367801/finding-all-cycles-in-undirected-graphs
https://stackoverflow.com/questions/20586163/find-all-cycles-in-undirected-graph
https://stackoverflow.com/questions/546655/finding-all-cycles-in-a-directed-graph
```
def dfs(node, stack):
    if node.visited:
        

def detect_cycles(grpah):
    stack = []
    for node in graph:
        res = dfs(node, stack)
```

convert cyclic graphs to acyclic graphs.

## Shortest Distances
BFS

For weighted graphs with positive weights, use Dijkstra
For weighted graphs with negative weights, use Bellman Ford's
### Dijkstra’s Algorithm for shortest paths for undirected graphs 
This algorithm computes the shortest path from a given vertex to all other vertices.
It starts from the source node, cumulate costs for all adjacent nodes. All costs are
positive.
Then for these neighbour nodes, cumulate costs for them and set to smaller accumulation.
So this is a greedy approach.
```
def dijkstra(graph, v):
    
```

  https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/

### Bellman–Ford Algorithm
___
Find shortest paths from source to all vertices in weighted directed graphs.

The diff between this and Dijkstra is that this can handle negative weights while
Dijkstra could break.

start from source, use DFS update cumulations along the way until all edges are visited.
repeat this process len(V) - 1 times. Then the cumulations are the results.

### Floyd Warshall Algorithm
___
Find shortest distances between every pair of vertices in weighted directed graphs. So the result is a 2D
matrix that map each node to another node with shortest distance.

Initialize matrix with input adjacent matrix. Then set all edge weights in matrix.

Then for each node, we scan entire matrix to fix ```dist[i][j] > dist[i][k] + dist[k][j]``` for any k.

https://www.freecodecamp.org/news/i-dont-understand-graph-theory-1c96572a1401/
78

## Topological Sorting
___
One way is to use DFS on all unvisited nodes. If there is no neighbour, push it to stack and move on to next node.

Another way is to start with nodes that have no parent. output these nodes and then remove these nodes' edges.
This converts some nodes to no-parent nodes.

LC 210

## Minimum Spanning Tree
___
A subset of edges in a weighted graph such that:
- include all vertices
- minimal weights
- no cycles

Algorithms to generate MST:
- Prim's algorithm  
  start with a vertex and keep add smallest edges until all vertices included.
- Kruskal's algorithm  
  start with smallest edges, keep add smallest edges except those making cycles,
  until all vertices included.

LC 1135

## Cycles
Travelling Salesman Problem
Hamiltonian Cycle (backtrack)
In a complete graph, there are (n-1)!/2 different cycles (different path ).
First, we fix one starting point to prevent duplicates (a cycle can start at any
node). So there are only (n-1) nodes left. Since a cycle inverse is the same
cycle, so we divide it by 2.


Eulerian path and Hierholzer's Algorithm


graph coloring:
https://www.includehelp.com/algorithms/graph-coloring-problem-solution-using-backtracking-algorithm.aspx

A* search

# https://slaystudy.com/hierholzers-algorithm/
# https://www.geeksforgeeks.org/eulerian-path-and-circuit/
# Eulerian Path is a path in graph that visits every edge exactly once.
# Eulerian Circuit is an Eulerian Path which starts and ends on the same vertex
# A graph is called Eulerian if it has an Eulerian Cycle.
# A graph is called Semi-Eulerian if it has an Eulerian Path.

# Undirect Graphs:
# Eulerian Cycle <==> All vertices have even degree and graph is connected.
# Eulerian Path <==> If zero or two vertices have odd degree and all other vertices have even degree

# https://www.geeksforgeeks.org/euler-circuit-directed-graph/

# 753. Cracking the Safe
# 847. Shortest Path Visiting All Nodes

# # https://www.geeksforgeeks.org/eulerian-path-and-circuit/
