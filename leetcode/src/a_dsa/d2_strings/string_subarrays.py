
# LC2405. Optimal Partition of String
def partitionString(self, s: str) -> int:
    lastSeen = [-1]*26
    count = 1
    substringStarting = 0
    for i in range(len(s)):
        if lastSeen[ord(s[i]) - ord('a')] >= substringStarting:
            count += 1
            substringStarting = i
        lastSeen[ord(s[i]) - ord('a')] = i
    return count

# LC2947. Count Beautiful Substrings I
def beautifulSubstrings(self, s: str, k: int) -> int:  # O(n) time and space
    prefixSum = [0]
    for i in range(len(s)):
        prefixSum.append(prefixSum[-1] + (1 if s[i] in 'aeiou' else -1))
    ans, cache = 0, {}
    for j in range(len(prefixSum)):
        if prefixSum[j] in cache:
            for i in cache[prefixSum[j]]:
                if ((j-i) // 2)**2 % k == 0:
                    ans += 1
            cache[prefixSum[j]].append(j)
        else:
            cache[prefixSum[j]] = [j]
    return ans

# LC1520. Maximum Number of Non-Overlapping Substrings
def maxNumOfSubstrings(self, s: str) -> List[str]:  # O(n)
    c2intvs = {}
    for i, c in enumerate(s):
        if c in c2intvs: c2intvs[c][1] = i
        else: c2intvs[c] = [i, i]
    for k, v in c2intvs.items():
        left, right = v
        while True:
            t = s[v[0]:v[1]+1]
            for c in t:
                v[0] = min(v[0], c2intvs[c][0])
                v[1] = max(v[1], c2intvs[c][1])
            if [left, right] == v: break
            left, right = v
        c2intvs[k] = (left, right)
    intvs = sorted(list(set(c2intvs.values())), key=lambda x: x[1])
    res = []
    for v in intvs:
        if not res or res[-1][1] < v[0]: res.append(v)
    return [s[v[0]:v[1]+1] for v in res]