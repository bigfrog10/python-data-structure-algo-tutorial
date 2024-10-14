
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
def maxNumOfSubstrings(self, s: str) -> List[str]:
    ranges = {}
    for i, c in enumerate(s):  # O(n)
        if c in ranges: ranges[c][0] = i
        else: ranges[c] = [i, i]
    for c in set(s):  # O(26)
        r, l = ranges[c]
        r_, l_ = -1, -1
        while not (r_ == r and l_ == l):
            r_, l_ = r, l
            r = max(ranges[c][0] for c in set(s[l:r+1]))  # O(n)
            l = min(ranges[c][1] for c in set(s[l:r+1]))
        ranges[c] = (r, l)
    ans, curr = [], 0
    for r, l in sorted(ranges.values()):  # sort by ends
        if l >= curr:  # new start >= previous end
            ans.append(s[l:r+1])
            curr = r
    return ans
# https://leetcode.com/problems/maximum-number-of-non-overlapping-substrings/solutions/744726/python-easy-to-read-solution-with-explanation/?envType=company&envId=amazon&favoriteSlug=amazon-three-months






