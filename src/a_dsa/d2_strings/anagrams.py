
# LC49. Group Anagrams
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    if not strs: return []  # O(nklogk), n = len(strs), k = max(len(str))
    ret = {}
    for s in strs:
        s1 = ''.join(sorted(s))  # hash
        if s1 not in ret: ret[s1] = [s]
        else: ret[s1].append(s)
    return ret.values()

# LC242. Valid Anagram
def isAnagram(self, s: str, t: str) -> bool:
    return collections.Counter(s) == collections.Counter(t)

# LC1347. Minimum Number of Steps to Make Two Strings Anagram
def minSteps(self, s: str, t: str) -> int:
    cnt1, cnt2 = map(collections.Counter, (s, t))
    return sum((cnt1 - cnt2).values())

# LC438. Find All Anagrams in a String
def findAnagrams(self, s: str, p: str) -> List[int]:
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

# LC567. Permutation in String
def checkInclusion(self, s1, s2):
    d1, d2 = Counter(s1), Counter(s2[:len(s1)])
    for start in range(len(s1), len(s2)):
        if d1 == d2: return True
        d2[s2[start]] += 1
        d2[s2[start-len(s1)]] -= 1
        if d2[s2[start-len(s1)]] == 0:
            del d2[s2[start-len(s1)]]
    return d1 == d2


