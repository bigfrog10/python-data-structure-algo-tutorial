from collections import Counter, defaultdict, deque
import heapq

# LC332. Reconstruct Itinerary O(E^d), d max flights from any node.
def findItinerary(self, tickets: List[List[str]]) -> List[str]:  # O(ElogE)
    if not tickets: return []
    dgraph = defaultdict(list)
    for origin, dest in tickets: dgraph[origin].append(dest)
    for _, dests in dgraph.items(): dests.sort(reverse=True) # lexical order
    ret = []
    def dfs(origin):
        dests = dgraph[origin]
        while dests: dfs(dests.pop()) # remove edge by Hierholzer's Algorithm
        ret.append(origin)  # dfs adds end nodes first, lastly add start.  - postorder traverse
    dfs('JFK')
    return ret[::-1]

# LC787. Cheapest Flights Within K Stops - weighted graph
def findCheapestPrice(self, n, flights, src, dst, K):  # O(E) runtime, O(E + V) space
    if src == dst: return 0
    d, seen = collections.defaultdict(list), collections.defaultdict(lambda: float('inf'))
    for u, v, p in flights: d[u] += [(v, p)]  # O(E) space
    dque = deque([(src, -1, 0)])
    while dque:  # O(E)
        pos, k, cost = dque.popleft()
        if pos == dst or k == K: continue
        for nei, p in d[pos]:
            if cost + p >= seen[nei]: continue
            else:
                seen[nei] = cost+p  # O(V) space
                dque.append((nei, k+1, cost+p))
    return seen[dst] if seen[dst] < float('inf') else -1

# LC2065. Maximum Path Quality of a Graph
def maximalPathQuality(self, values: List[int], edges: List[List[int]], maxTime: int) -> int:
    G = collections.defaultdict(dict)
    for i, j, t in edges: G[i][j] = G[j][i] = t
    def dfs(i, seen, time):
        res = sum(values[j] for j in seen) if i == 0 else 0
        for j in G[i]:
            if time >= G[i][j]: res = max(res, dfs(j, seen | {j}, time - G[i][j]))
        return res
    return dfs(0, {0}, maxTime)



# LC743. Network Delay Time
def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
    graph = collections.defaultdict(list) # dijkstra's with heap
    for u, v, w in times: graph[u].append((v, w))
    pq, dist = [(0, k)], {} # cost 0 at node k; distance or cost
    while pq:
        d, node = heapq.heappop(pq)
        if node in dist: continue
        dist[node] = d
        for nei, d2 in graph[node]:
            if nei not in dist: heapq.heappush(pq, (d+d2, nei))
    return max(dist.values()) if len(dist) == n else -1





# LC1135. Connecting Cities With Minimum Cost
def minimumCost(self, N: int, connections: List[List[int]]) -> int:
    G = collections.defaultdict(list)
    for city1, city2, cost in connections:  # O(E)
        G[city1].append((cost, city2))
        G[city2].append((cost, city1))
    queue, visited = [(0, N)], set()  # [1] Arbitrary starting point N costs 0.
    total = 0
    while queue: # [3] Exit if all cities are visited. O(VlogV)
        cost, city = heapq.heappop(queue)
        if city not in visited:
            visited.add(city)
            total += cost # [2] Grow tree by one edge.
            for edge in G[city]:
                heapq.heappush(queue, edge)
    return total if len(visited) == N else -1


