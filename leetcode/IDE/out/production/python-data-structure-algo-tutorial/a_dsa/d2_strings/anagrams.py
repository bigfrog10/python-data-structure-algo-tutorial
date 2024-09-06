
# LC438. Find All Anagrams in a String
def findAnagrams(self, s: str, p: str) -> List[int]:  # O(Ls)
    slen, plen = len(s), len(p)
    if slen < plen: return []  # no enough chars
    pcounts = Counter(p)  # anagram signature
    scounts = Counter()  # accumulate and update counts along the way
    ret = []
    for i in range(slen):
        scounts[s[i]] += 1
        if i >= plen:
            scounts[s[i-plen]] -= 1  # remove from left
            if scounts[s[i-plen]] == 0: del scounts[s[i-plen]]
        if pcounts == scounts: ret.append(i - plen + 1)  # this is whey we need del above
    return ret

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






