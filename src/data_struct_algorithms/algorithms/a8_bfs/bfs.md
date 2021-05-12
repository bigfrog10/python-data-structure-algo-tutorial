ask for minimal, or all minimal solutions

LC 547 - number of provinces, adjacent matrix dfs

## BFS template

## DFS template

### Non recursive

```
visited = set()
stack = [initial elements]
while stack:
    e = stack.pop()
    if e is solution:
        either return result
        or collect result and continue
        
    if e in visited:
        return
    visited.add(e)
    
    for c in e.children :
        stack.append(c)
```

### recursive

```
def dfs(e, visited):
    if e is solution:
        either return result
        or collect result and return
    
    if e in visited:
        return
    visited.add(e)
    
    for c in e.children :
        dfs(c, visited)
```
## DFS with backtrack

```

```

