# LC1657. Determine if Two Strings Are Close
def closeStrings(self, word1: str, word2: str) -> bool:
    c1, c2 = Counter(word1), Counter(word2)  # O(m+n)
    # size is 26
    return sorted(c1.values()) == sorted(c2.values()) and sorted(c1.keys()) == sorted(c2.keys())

# LC1061. Lexicographically Smallest Equivalent String - union find
def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
    ##  M union or find sequences on N objects takes O(N + N * log*N), O(N) space
    parents = {c:c for c in string.ascii_lowercase}
    ranks = {c:1 for c in string.ascii_lowercase}

    def find(char):
        res = char
        while res != parents[res]:
            parents[res] = parents[parents[res]]
            res = parents[res]
        return res
    def union(char1, char2):
        p1, p2 = find(char1), find(char2)
        if p1 == p2: return 0
        if ord(p1) <= ord(p2):
            ranks[p1] += ranks[p2]
            parents[p2] = p1
        else:
            ranks[p2] += ranks[p1]
            parents[p1] = p2
    for char1, char2 in zip(s1, s2):
        union(char1, char2)
    res = ""
    for char in baseStr:
        res += find(char)
    return res

# LC451. Sort Characters By Frequency - sort by freq  sort char by freq
def frequencySort(self, s: str) -> str:  # O(n), not nlogn, using bucket sort
    if not s: return s
    counts = collections.Counter(s)  # Determine the frequency of each character.
    max_freq = max(counts.values())
    buckets = [[] for _ in range(max_freq + 1)]  # count -> char list
    for c, i in counts.items(): buckets[i].append(c)  # Bucket sort the characters by frequency.
    res = []
    for i in range(len(buckets))[::-1]:
        for c in buckets[i]: res.append(c * i)
    return "".join(res)
def frequencySort(self, s: str) -> str:  # O(nlogn) time and O(n) space
    counts = collections.Counter(s)  # Count up the occurances.
    string_builder = []
    for letter, freq in counts.most_common():
        # letter * freq makes freq copies of letter.
        # e.g. "a" * 4 -> "aaaa"
        string_builder.append(letter * freq)
    return "".join(string_builder)

# LC791. Custom Sort String
def customSortString(self, order: str, s: str) -> str:  # O(s + t) time and O(t) space
    count = collections.Counter(s)
    ans = []
    for c in order:
        ans.append(c * count[c])
        count[c] = 0  # remove this char for next for-loop
    for c in count: ans.append(c * count[c])  # c not in order
    return "".join(ans)

# LC953. Verifying an Alien Dictionary
from typing import List
def isAlienSorted(self, words: List[str], order: str) -> bool:  # O(total chars in words)
    orderd = {x: idx for idx, x in enumerate(order)}
    for i in range(len(words) - 1):  # O(n)
        for j in range(len(words[i])):  # O(m)
            if j >= len(words[i + 1]): return False  # apple > app, wrong - space is ahead of letters
            if words[i][j] != words[i + 1][j]:
                if orderd[words[i][j]] > orderd[words[i + 1][j]]: return False
                break  # j is sorted correctly, so check j+1
    return True

# LC269. Alien Dictionary, top100
def alienOrder(self, words: List[str]) -> str:  # O(total word lengths)
    adj_list = defaultdict(set)  #  the in_degree of each unique letter to 0. (n^2) space
    in_degree = Counter({c: 0 for word in words for c in word})  # needed below
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
def alienOrder(self, words: List[str]) -> str:  # O(total word lengths)
    graph = defaultdict(list) # {c: [] for c in chars}
    degrees = Counter({c: 0 for word in words for c in word})  # 0 needed below
    for w1, w2 in zip(words, words[1:]):
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                graph[c1].append(c2)  # space is u^2, unique char size
                degrees[c2] += 1
                break
        else:  # won't execute if above break
            if len(w1) > len(w2): return ""  # space is before letters
    queue = [c for c in degrees if degrees[c] == 0]
    ans = ""
    while queue:
        c = queue.pop()
        ans += c
        for n in graph[c]:
            degrees[n] -= 1
            if degrees[n] == 0: queue.append(n)
    return ans if len(ans) == len(degrees) else ""  # cyclic ordering





# LC1985. Find the Kth Largest Integer in the Array - find kth largest
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
