
# LC953. Verifying an Alien Dictionary
from typing import List
def isAlienSorted(self, words: List[str], order: str) -> bool:
    orderd = {x: idx for idx, x in enumerate(order)}  # O(m*n)
    for i in range(len(words) - 1):  # O(n)
        for j in range(len(words[i])):  # O(m)
            if j >= len(words[i + 1]): return False  # apple > app, wrong - space is ahead of letters
            if words[i][j] != words[i + 1][j]:
                if orderd[words[i][j]] > orderd[words[i + 1][j]]: return False
                break  # j is sorted correctly, so check j+1
    return True

# LC269. Alien Dictionary, top100
def alienOrder(self, words: List[str]) -> str:
    adj_list = defaultdict(set)  #  the in_degree of each unique letter to 0.
    in_degree = Counter({c: 0 for word in words for c in word})
    for first_word, second_word in zip(words[:-1], words[1:]):
        first_word = first_word.strip()
        second_word = second_word.strip()
        if first_word.startswith(second_word) and len(first_word) > len(second_word):
            return ""  # space is before letters
        for c, d in zip(first_word, second_word):
            if c != d:
                if d not in adj_list[c]:
                    adj_list[c].add(d)  # c -> d
                    in_degree[d] += 1
                break
    output = []  # need to repeatedly pick off nodes with an indegree of 0.
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    while queue:
        c = queue.popleft()
        output.append(c)
        for d in adj_list[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0: queue.append(d)
    if len(output) < len(in_degree): return ""  # cyclic ordering
    return "".join(output)

# LC791. Custom Sort String
def customSortString(self, order: str, s: str) -> str:
    count = collections.Counter(s)
    ans = []
    for c in order:
        ans.append(c * count[c])
        count[c] = 0  # remove this char for next for-loop
    for c in count: ans.append(c * count[c])
    return "".join(ans)

# LC451. Sort Characters By Frequency
def frequencySort(self, s: str) -> str:  # O(n)
    if not s: return s
    counts = collections.Counter(s)  # Determine the frequency of each character.
    max_freq = max(counts.values())
    buckets = [[] for _ in range(max_freq + 1)]  # count -> char list
    for c, i in counts.items(): buckets[i].append(c)  # Bucket sort the characters by frequency.
    res = []
    for i in range(len(buckets))[::-1]:
        for c in buckets[i]: res.append(c * i)
    return "".join(res)

# LC1985. Find the Kth Largest Integer in the Array
def kthLargestNumber(self, nums: List[str], k: int) -> str:  # O(nlogk)
    pq = [] # min-heap
    for x in nums:
        heappush(pq, int(x))
        if len(pq) > k: heappop(pq)
    return str(pq[0])
def kthLargestNumber1(self, nums: List[str], k: int) -> str:  # O(n)
    nums = [int(x) for x in nums]
    shuffle(nums)
    def part(lo, hi):  # """Return partition of nums[lo:hi]."""
        i, j = lo+1, hi-1
        while i <= j:
            if nums[i] < nums[lo]: i += 1
            elif nums[lo] < nums[j]: j -= 1
            else:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        nums[lo], nums[j] = nums[j], nums[lo]
        return j
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = part(lo, hi)
        if mid == len(nums)-k: return str(nums[mid])
        elif mid < len(nums)-k: lo = mid + 1
        else: hi = mid
def kthLargestNumber(self, nums: List[str], k: int) -> str:
    nums = [int(x) for x in nums]  # handle 3 and 13
    klarger = heapq.nlargest(k, nums) # O(nlogk)
    return str(sorted(klarger)[0])
