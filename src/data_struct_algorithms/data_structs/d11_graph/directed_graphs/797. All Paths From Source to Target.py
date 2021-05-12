# LC797. All Paths From Source to Target
def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    results, target = [], len(graph) - 1
    def backtrack(currNode, path): # we need all paths, so backtrack on path.
        if currNode == target:
            results.append(list(path)) # new path
            return
        for nextNode in graph[currNode]: # neighbours
            path.append(nextNode)
            backtrack(nextNode, path)
            path.pop() # backout
    backtrack(0, [0]) # 0 is starting point, [0] is current path
    return results
