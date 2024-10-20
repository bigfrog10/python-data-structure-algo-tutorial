
from typing import List

# LC802. Find Eventual Safe States safe node terminal node
def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
    n = len(graph)  # O(V + E) time and space
    outdegree = [0] * n
    adj = [[] for _ in range(n)]
    for i in range(n):
        for node in graph[i]:
            adj[node].append(i)  # opposite, from node to i
            outdegree[i] += 1
    q = deque([i for i in range(n) if outdegree[i] == 0])  # terminal nodes
    safe = [False] * n
    while q:
        node = q.popleft()
        safe[node] = True
        for neighbor in adj[node]:
            outdegree[neighbor] -= 1
            if outdegree[neighbor] == 0:
                q.append(neighbor)
    return [i for i in range(n) if safe[i]]

# LC207. Course Schedule - true if can finish
def canFinish(self, n, prerequisites):  # O(V + E)
    G = [[] for i in range(n)] # g[i] needs i finished earlier
    degree = [0] * n
    for j, i in prerequisites:
        G[i].append(j)
        degree[j] += 1
    bfs = [i for i in range(n) if degree[i] == 0]
    for i in bfs:
        for j in G[i]:
            degree[j] -= 1
            if degree[j] == 0: bfs.append(j)
    return len(bfs) == n

# LC210. Course Schedule II - return course order to take
def findOrder(self, numCourses, prerequisites): # O(V + E)
    n = numCourses
    G = [[] for i in range(n)] # g[i] needs i finished earlier
    degree = [0] * n
    for j, i in prerequisites:
        G[i].append(j)  # take i first then take j
        degree[j] += 1
    bfs = [i for i in range(n) if degree[i] == 0]
    for i in bfs:
        for j in G[i]:
            degree[j] -= 1
            if degree[j] == 0: bfs.append(j)
    return bfs if len(bfs) == n else []

# LC1462. Course Schedule IV
def checkIfPrerequisite(self, n: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
    graph = collections.defaultdict(list)
    in_degree = [0] * n
    pres = [set() for _ in range(n)]
    for pre, course in prerequisites:
        graph[pre].append(course)
        in_degree[course] += 1
        pres[course].add(pre)
    queue = collections.deque(course for course, degree in enumerate(in_degree)
                              if degree == 0)
    while queue:
        pre = queue.popleft()
        for course in graph[pre]:
            pres[course] |= pres[pre]
            in_degree[course] -= 1
            if in_degree[course] == 0:
                queue.append(course)
    return [pre in pres[course] for pre, course in queries]
