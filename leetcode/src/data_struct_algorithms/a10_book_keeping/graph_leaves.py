# https://leetcode.com/discuss/interview-question/1197906/DAG-Graph-Leave-nodes
# https://cp-algorithms.com/graph/lca_tarjan.html
# https://cp-algorithms.com/data_structures/disjoint_set_union.html
# make get leaves as fast as possible
# binary index tree in graph context, union find?
from collections import defaultdict

class DAG:
    def __init__(self):
        self.graph = defaultdict(list)

    def AddChild(self, str1, str2):
        self.graph[str1].append(str2)
        # self.graph[str2] = []  # ?

    def GetLeaves(self, node):
        ret = set()
        if node not in self.graph:
            ret.add(node)
            return ret
        children = self.graph[node]
        for child in children:
            if len(self.graph[node]) == 0:
                ret.add(node)
            else:
                ret.update(self.GetLeaves(child))
        return ret

d = DAG()
d.AddChild("A", "B")
d.AddChild("B", "C")
d.AddChild("B", "D")
d.AddChild("A", "C")
d.AddChild("C", "E")
d.AddChild("E", "F")
print(d.GetLeaves("C"))
