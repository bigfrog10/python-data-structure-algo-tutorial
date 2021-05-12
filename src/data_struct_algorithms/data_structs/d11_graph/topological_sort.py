from collections import Counter, defaultdict, deque
from typing import List
# LC207. Course Schedule, top100
def canFinish(self, n, prerequisites):
    G = [[] for i in range(n)]
    degree = [0] * n
    for j, i in prerequisites:
        G[i].append(j)
        degree[j] += 1

    bfs = [i for i in range(n) if degree[i] == 0]
    for i in bfs:
        for j in G[i]:
            degree[j] -= 1
            if degree[j] == 0:
                bfs.append(j)

    return len(bfs) == n

# LC210. Course Schedule II
def findOrder(self, numCourses, prerequisites):
    indegree = [set() for _ in range(numCourses)]
    outdegree = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        indegree[a].add(b) # take b first, then a
        outdegree[b].append(a)
    ret, zero_in = [], [i for i in range(numCourses) if not indegree[i]]
    while zero_in: # start contains courses without prerequisites
        newStart = []
        for i in zero_in:
            ret.append(i)
            for j in outdegree[i]: # remove i in neighbour
                indegree[j].remove(i)
                if not indegree[j]: newStart.append(j)
        zero_in = newStart # newStart contains new courses with no prerequisites
    return ret if len(ret) == numCourses else [] # can finish if ret contains all courses

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
