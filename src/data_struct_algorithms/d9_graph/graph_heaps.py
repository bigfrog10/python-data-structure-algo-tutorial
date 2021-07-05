from collections import Counter, defaultdict, deque
import heapq

# LC332. Reconstruct Itinerary O(E^d), d max flights from any node.
def findItinerary(self, tickets: List[List[str]]) -> List[str]:
    if not tickets: return []
    dgraph = defaultdict(list)
    for origin, dest in tickets: dgraph[origin].append(dest)
    for _, dests in dgraph.items(): dests.sort(reverse=True) # lexical order
    ret = []
    def dfs(origin):
        dests = dgraph[origin]
        while dests: dfs(dests.pop()) # remove edge by Hierholzer's Algorithm
        ret.append(origin)  # dfs adds end nodes first, lastly add start.
    dfs('JFK')
    return ret[::-1]

# LC787. Cheapest Flights Within K Stops - weighted graph + heap
def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
    graph = defaultdict(list)  # Dijkstra
    for source,dest,cost in flights: graph[source].append((cost,dest))
    heap = []
    heapq.heappush(heap, (0, src, -1))  # add src to kick things off
    while heap:
        nextCost, nextDest, currSteps = heapq.heappop(heap)
        if currSteps > K: continue
        if nextDest == dst: return nextCost
        for nc,nd in graph[nextDest]:  # this is not optimal as dijkstra
            heapq.heappush(heap, (nc + nextCost, nd, currSteps + 1))
    return -1

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


