# Find all paths from source to destiniation in DAG
# no cycle included, i.e., no dupe node in each path.

from collections import defaultdict
def all_paths(graph, src, dest):
    res, visited = [], defaultdict(bool)
    def dfs(node, path):
        if node == dest:
            res.append(path + [node])
        else:
            path.append(node)
            visited[node] = True
            for ch in graph[node]:
                if not visited[ch]:
                    dfs(ch, path)
            path.pop() # backout path and visit
            visited[node] = False
    dfs(src, [])
    return res

graph = {}
graph[0] = [1, 2, 3]  # 0 -> 3 is a path
graph[1] = [0]  # 0 <--> 1 is a cycle
graph[2] = [3]  # 0 -> 2 -> 3 is another path

res = all_paths(graph, 0, 3)
print(res)
