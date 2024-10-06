
# LC295. Find Median from Data Stream, top100  median stream  median data stream
class MedianFinder:
    def __init__(self):
        self.heaps = [], []  # max heap from smalls, min heap from larges
    def addNum(self, num: int) -> None:
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small): heappush(large, -heappop(small))
    def findMedian(self) -> float:
        small, large = self.heaps
        if len(large) > len(small): return float(large[0])
        return (large[0] - small[0]) / 2.0

# LC1865. Finding Pairs With a Certain Sum
class FindSumPairs:
    def __init__(self, nums1: List[int], nums2: List[int]):
        self.freq1 = Counter(nums1)
        self.freq2 = Counter(nums2)
        self.nums2 = nums2  # for index purpose
    def add(self, index: int, val: int) -> None:
        self.freq2[self.nums2[index]] -= 1  # Remove old one
        self.nums2[index] += val
        self.freq2[self.nums2[index]] += 1  # Count new one
    def count(self, tot: int) -> int:
        return sum(val * self.freq2[tot - key] for key, val in self.freq1.items())

# LC1570. Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.pairs = []
        for i, v in enumerate(nums):  # pairs are sorted by i
            if v != 0: self.pairs.append([i, v])
    def dotProduct(self, vec: 'SparseVector') -> int:
        p = q = result = 0
        while p < len(self.pairs) and q < len(vec.pairs):
            if self.pairs[p][0] == vec.pairs[q][0]:
                result += self.pairs[p][1] * vec.pairs[q][1]
                p, q = p+1, q+1
            elif self.pairs[p][0] < vec.pairs[q][0]: p += 1
            else: q += 1
        return result

# LC346. Moving Average from Data Stream   mv data stream
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.mv = 0
    def next(self, val: int) -> float:
        size, queue = self.size, self.queue
        queue.append(val)
        if len(queue) > size:
            v = queue.popleft()
            self.mv = self.mv + (val - v) / size
        else: self.mv = sum(queue) / len(queue)
        return self.mv

# LC703. Kth Largest Element in a Stream  kth highest test score
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = heapq.nlargest(k, nums)  # O(klogk)
        heapq.heapify(self.nums)  # O(klogk)
    def add(self, val: int) -> int:  # logk
        if len(self.nums) < self.k: heapq.heappush(self.nums, val)
        elif val > self.nums[0]: heapq.heapreplace(self.nums, val)
        return self.nums[0]  # min, which is the kth largest

# LC1206. Design Skiplist - operations in O(logn)
# Or use this: https://leetcode.com/problems/design-skiplist/discuss/500036/python-nodes-with-two-pointers
class Node(object):
    def __init__(self,val):
        self.val,self.next, self.down = val, None, None

class Skiplist(object):
    def __init__(self, levels = 30):
        self.heads = [Node(-float('inf')) for _ in range(levels)]
        for c,n in zip(self.heads, self.heads[1:]): c.down = n

    def search(self, target):
        cur = self.heads[0]
        while(cur):
            if cur.next is None or cur.val < target <= cur.next.val:
                if cur.next and target == cur.next.val: return True
                cur = cur.down
            else: cur = cur.next
        return False

    def add(self, num):
        stack,cur,prev = collections.deque([]),self.heads[0], None
        while(cur):
            if cur.next is None or cur.val < num  <= cur.next.val:
                stack.append(cur)
                cur = cur.down
            else: cur = cur.next
        while(stack):
            cur = stack.pop()
            node = Node(num)
            node.next,cur.next = cur.next, node
            if prev: node.down = prev
            prev = node
            if random.randint(0,len(self.heads)-1) < len(self.heads) -1 : break

    def erase(self, num):
        b,cur = False,self.heads[0]
        while(cur):
            if cur.next is None or cur.val < num <= cur.next.val:
                if cur.next and cur.next.val == num:
                    b,cur.next = True,cur.next.next
                cur = cur.down
            else: cur = cur.next
        return b
class ListNode:
    def __init__(self, val, cnt=1, next=None, down=None):
        self.val = val
        self.cnt = cnt
        self.next = next
        self.down = down
class Skiplist:
    def __init__(self):
        self.head = ListNode(-inf)
        self.p = 1/4
    def search(self, target: int) -> bool:
        node = self.head
        while node and node.val < target:
            if node.next and node.next.val <= target: node = node.next
            else: node = node.down
        return node
    def add(self, num: int) -> None:
        node = self.head
        stack = []
        while node and node.val < num:
            if node.next and node.next.val <= num: node = node.next
            else:
                stack.append(node)
                node = node.down
        if node:
            while node:
                node.cnt += 1
                node = node.down
        else:
            prev = None
            while True:
                if stack:
                    node = stack.pop()
                    node.next = prev = ListNode(num, down=prev, next=node.next)
                else:
                    self.head = ListNode(-inf, down=self.head)
                    self.head.next = prev = ListNode(num, down=prev)
                if random.random() >= self.p: break
    def erase(self, num: int) -> bool:
        node = self.head
        stack = []
        ans = False
        while node:
            if node.next and node.next.val < num: node = node.next
            else:
                stack.append(node)
                node = node.down
        while stack:
            node = stack.pop()
            if node.next and node.next.val == num:
                ans = True
                if node.next.cnt > 1: node.next.cnt -= 1
                else: node.next = node.next.next
            else: break
        return ans
