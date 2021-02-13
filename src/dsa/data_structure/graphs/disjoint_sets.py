from collections import defaultdict


class DisjointSetsGraph:
    def __init__(self, directed=False):
        self._directed = directed
        self._edges = defaultdict(list)

    def add_edge(self, n1, n2):
        self._edges[n1].append(n2)
        if not self._directed:
            self._edges[n2].append(n1)

    def find(self, n):
        pass


# https://www.cs.princeton.edu/~rs/AlgsDS07/
# https://opendsa-server.cs.vt.edu/ODSA/Books/Everything/html/UnionFind.html
# Starting from an empty data structure, any sequence of M union and
# find operations on N objects takes O(N + M lg* N) time
class DisjointSetsArray:
    def __init__(self, array):
        self._values = array
        self._ids = [i for i in range(len(array))]  # parents
        self._weights = defaultdict(lambda x: 1)  # union is faster with this

    def find(self, p, q):
        return self._root(p) == self._root(q)

    def union(self, p, q):  # O(logN), N is array size
        i = self._root(p)
        j = self._root(q)
        if i != j:  # path compression, this reduces the tree height to logN
            if self._weights[i] <= self._weights[j]:
                self._ids[i] = j
                self._weights[j] += self._weights[i]
            else:
                self._ids[j] = i
                self._weights[i] += self._weights[j]

    def _root(self, n):
        while n != self._ids[n]:
            # a simple acceleration, set to grandparent
            # we could set every node along the path to final root.
            self._ids[n] = self._ids[self._ids[n]]
            n = self._ids[n]

        return n
