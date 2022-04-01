# https://cp-algorithms.com/data_structures/disjoint_set_union.html
# 2076. Process Restricted Friend Requests
class UnionFind:
    def __init__(self, alphabet):
        self.parent = {c: c for c in alphabet}
        self.value = {c: 1 for c in alphabet}
        self.rank = {c: 1 for c in alphabet}
    def find(self, u):
        if u != self.parent[u]:
            self.parent[u], val = self.find(self.parent[u])
            self.value[u] *= val
        return self.parent[u], self.value[u]
    def union(self, u, v, w):
        pu, vu = self.find(u)
        pv, vv = self.find(v)
        if pu == pv: return
        if self.rank[pu] > self.rank[pv]: self.union(v, u, 1/w)
        else:
            self.parent[pu] = self.parent[pv]
            self.value[pu] = w * vv / vu
            self.rank[pv] += self.rank[pu]
