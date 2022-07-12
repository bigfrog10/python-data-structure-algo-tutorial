import heapq
class GentNextMin:
    def __init__(self, l1, l2, l3):
        self.heap = []

        if l1:
            l1.sort(key=lambda x : -x)
            heapq.heappush(self.heap, (l1.pop(), l1))
        if l2:
            l2.sort(key=lambda x : -x)
            heapq.heappush(self.heap, (l2.pop(), l2))
        if l3:
            l3.sort(key=lambda x : -x)
            heapq.heappush(self.heap, (l3.pop(), l3))

    def has_next(self):
        if self.heap:
            return True
        return False

    def next(self):
        if not self.has_next():
            raise Exception()

        e, l = heapq.heappop(self.heap)
        if l:
            heapq.heappush(self.heap, (l.pop(), l))
        return e


i = GentNextMin([5,1,2,4], [4,6,3], [9,0,7])
while i.has_next():
    print(i.next())
