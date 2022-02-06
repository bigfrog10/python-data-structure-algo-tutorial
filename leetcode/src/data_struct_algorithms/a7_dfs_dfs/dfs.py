
# DFS to explore all possibilities and collect stats, return 1 or all results.
from collections import Counter, defaultdict
from typing import List

# LC1376. Time Needed to Inform All Employees
def numOfMinutes(self, n, headID, manager, informTime):
    reports = defaultdict(list)
    for i, m in enumerate(manager):
        if m >= 0: reports[m].append(i)

    def dfs(i):
        return max([dfs(j) for j in reports[i]] or [0]) + informTime[i]
    return dfs(headID)



# LC377. Combination Sum IV
def combinationSum4(self, nums: List[int], target: int) -> int:  # O(T * N)
    @functools.lru_cache(maxsize = None)
    def combs(remain):
        if remain == 0: return 1
        result = 0
        for num in nums:
            if remain - num >= 0: result += combs(remain - num)
        return result
    return combs(target)

# LC216. Combination Sum III
from itertools import combinations
def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    return [c for c in combinations(range(1, 10), k) if sum(c) == n]

def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    self.ans = []
    def dfs(start, sol, k, n):
        if k == 0 and n == 0:
            self.ans.append(sol)
        if start > 9 or start > n or k <= 0:
            return
        dfs(start+1, sol+[start], k-1, n-start)
        dfs(start+1, sol, k, n)
    dfs(1, [], k, n)
    return self.ans

# LC472. Concatenated Words
def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    word_set = set(words)
    def check(word):
        n = len(word)
        for i in range(1, n):
            if word[i:] not in word_set: continue
            # so now word[i:] in word_set
            if word[:i] in word_set: return True # so both part are words
            if check(word[:i]): return True # recursion check
        return False
    res = []
    for w in words:
        if check(w): res.append(w)
    return res
