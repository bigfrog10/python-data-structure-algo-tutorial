
# LC502. IPO
def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
    heap = []  ## O(max(N, K) log N)
    projects = sorted(zip(profits, capital), key=lambda l: l[1])
    i = 0
    for _ in range(k):
        while i < len(projects) and projects[i][1] <= w:
            heapq.heappush(heap, -projects[i][0])
            i += 1
        if heap: w -= heapq.heappop(heap)
    return w

# LC871. Minimum Number of Refueling Stops
def minRefuelStops(self, target: int, tank: int, stations: List[List[int]]) -> int:
    pq = []  # A maxheap is simulated using negative values
    stations.append((target, 0))
    ans = prev = 0
    for location, capacity in stations:
        tank -= location - prev
        while pq and tank < 0:  # must refuel in past
            tank += -heapq.heappop(pq)  # fill with largest supply
            ans += 1
        if tank < 0: return -1
        heapq.heappush(pq, -capacity)
        prev = location
    return ans

# https://leetcode.com/discuss/interview-question/124616/Merge-two-interval-lists
# pandas way: https://stackoverflow.com/questions/53371736/quick-algorithm-to-create-union-of-multiple-intervals-data
def mergeIntervals(int1, int2):
    if not int1 and not int2: return []
    ret, i, j = [], 0, 0  # if one of them is empty, we still need to merge intervals in the other.
    if not int1:
        curr = int2[0]
        j = 1
    elif not int2:
        curr = int1[0]
        i = 1
    elif int1[0][0] < int2[0][0]:  # find first interval, smaller start
        curr = int1[0]
        i = 1
    else:
        curr = int2[0]
        j = 1
    while i < len(int1) or j < len(int2):  # find next interval
        if j == len(int2) or (i < len(int1) and int1[i][0] < int2[j][0]):
            nxt = int1[i]
            i += 1
        else:
            nxt = int2[j]
            j += 1
        if curr[1] < nxt[0]:
            ret.append(curr)
            curr = nxt
        else: curr[1] = max(curr[1], nxt[1])
    ret.append(curr)  # leftover
    return ret


A = [[1,5], [10,14], [16,18]]
B = [[2,6], [8,10], [11,20]]
print(mergeIntervals(A, B))  # [1,6], [8, 20]

A = [[1,5], [10,14], [16,18]]
B = [[5,10], [14,16], [18,20]]
print(mergeIntervals(A, B))  # [[1, 20]]
A = [[1,2], [2,3], [3,4]]
B = []
print(mergeIntervals(A, B))  # [[1, 4]]

# full tested with leetcode 616
# 459

# LC616. Add Bold Tag in String, same as LC758.
def addBoldTag(self, s: str, words: List[str]) -> str:
    status = [False] * len(s)
    for word in words:  # O(len(words))
        start, last = s.find(word), len(word)  # O(len(s) * O(max(len(words))))
        while start != -1: # this word appears multiple places
            for i in range(start, last+start): status[i] = True
            start = s.find(word, start+1)
    i, final = 0, ""
    while i < len(s):  # O(len(s))
        if status[i]:
            final += "<b>"
            while i < len(s) and status[i]:
                final += s[i]
                i += 1
            final += "</b>"
        else:
            final += s[i]
            i += 1
    return final
def addBoldTag(self, s: str, words: List[str]) -> str:  # long solution with kmp
    def kmp_pi(s: str):  # the length of the longest *proper* prefix of s which is also a suffix
        m = len(s)
        pi = [0] * m
        fmm_idx = 0  # index of first mismatch
        for i in range(1, m):
            # when there is a mismatch, rollback to previous suffix.
            while fmm_idx and s[i] != s[fmm_idx]: fmm_idx = pi[fmm_idx-1]
            # if fmm_idx = 0, continue with pi[i] = 0
            if s[i] == s[fmm_idx]:  # if matched, move forward
                fmm_idx += 1
                pi[i] = fmm_idx
        return pi
    def kmp_search(text, pattern):  # return first indices of all occurrences of pattern in text
        matches, pi = [], kmp_pi(pattern)
        n, m = len(text), len(pattern)
        k = 0  # pattern index
        for i in range(n):
            while k and text[i] != pattern[k]: k = pi[k - 1]  # if k = 0, continue
            if text[i] == pattern[k]:
                if k == m - 1:
                    matches.append(i - k)
                    k = pi[k]
                else: k += 1
        return matches
    status = [False] * len(s)  # compromise to flags
    for word in words:  # O(len(words) * len(s))
        idxs = kmp_search(s, word)
        for i in idxs: status[i:i+len(word)] = [True] * len(word)
    i, res = 0, ""
    while i < len(s):  # O(len(s))
        if status[i]:
            res += "<b>"
            while i < len(s) and status[i]:
                res += s[i]
                i += 1
            res += "</b>"
        else:
            res += s[i]
            i += 1
    return res
#########################################################################
# This is the best solution, with 2 reusable components
# KMP to find substrings
# merge 2 sorted lists (by start) of intervals - OR operation
def addBoldTag(self, s: str, words: List[str]) -> str:
    def kmp_pi(s: str):  # the length of the longest *proper* prefix of s which is also a suffix
        m = len(s)
        pi = [0] * m
        fmm_idx = 0  # index of first mismatch
        for i in range(1, m):
            # when there is a mismatch, rollback to previous suffix. no need to beginning
            while fmm_idx and s[i] != s[fmm_idx]: fmm_idx = pi[fmm_idx-1]
            # if fmm_idx = 0, continue with pi[i] = 0
            if s[i] == s[fmm_idx]:  # if matched, move forward
                fmm_idx += 1
                pi[i] = fmm_idx
        return pi
    def kmp_search(text, pattern):  # return first indices of all occurrences of pattern in text
        matches, pi = [], kmp_pi(pattern)
        n, m = len(text), len(pattern)
        k = 0  # pattern index
        for i in range(n):
            while k and text[i] != pattern[k]: k = pi[k - 1]  # if k = 0, continue
            if text[i] == pattern[k]:
                if k == m - 1:
                    matches.append(i - k)
                    k = pi[k]
                else: k += 1
        return matches
    def merge(int1, int2):
        if not int1 and not int2: return []
        ret, i, j = [], 0, 0
        if not int1:
            curr = int2[0]
            j = 1
        elif not int2:
            curr = int1[0]
            i = 1
        elif int1[0][0] < int2[0][0]:  # find first interval
            curr = int1[0]
            i = 1
        else:
            curr = int2[0]
            j = 1
        while i < len(int1) or j < len(int2):  # find next interval
            if j == len(int2) or (i < len(int1) and int1[i][0] < int2[j][0]):
                nxt = int1[i]
                i += 1
            else:
                nxt = int2[j]
                j += 1
            if curr[1] < nxt[0]:
                ret.append(curr)
                curr = nxt
            else: curr[1] = max(curr[1], nxt[1])
        ret.append(curr)
        return ret
    intervals = []
    for word in words:
        idxs = kmp_search(s, word)
        ints = [[idx, idx+len(word)] for idx in idxs]
        intervals = merge(intervals, ints)
    res, prev_end = '', 0
    for start, end in intervals:
        res += s[prev_end:start] + '<b>' + s[start:end] + "</b>"
        prev_end = end
    return res + s[prev_end:]
