
# LC438. Find All Anagrams in a String  all p anagrams in s
def findAnagrams(self, st: str, ana: str): # O(n) time
    d1, d2 = Counter(ana), Counter(st[:len(ana)])  # O(1) space, 26 chars
    res = []
    for start in range(len(ana), len(st)):
        if d1 == d2:  res.append(start - len(ana))
        d2[st[start]] += 1
        k = st[start-len(ana)]
        d2[k] -= 1
        if d2[k] == 0: del d2[k]
    if d1 == d2: res.append(len(st) - len(ana))
    return res

# LC49. Group Anagrams
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:  # O(len(strs)*max(len(strs)))
    ans = collections.defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        ans[tuple(count)].append(s)
    return ans.values()

# LC242. Valid Anagram
def isAnagram(self, s: str, t: str) -> bool:
    return collections.Counter(s) == collections.Counter(t)

# LC1347. Minimum Number of Steps to Make Two Strings Anagram
def minSteps(self, s: str, t: str) -> int:
    cnt1, cnt2 = map(collections.Counter, (s, t))
    return sum((cnt1 - cnt2).values())






