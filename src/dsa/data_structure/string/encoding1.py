from functools import lru_cache
# https://leetcode.com/problems/encode-string-with-shortest-length/discuss/985978/Python-The-Fastest-Solution-O(N2)-via-Knuth-Morris-Pratt


@lru_cache(None)
def mincode(s):
    """find the shortest representation of the string and return it"""
    if len(s) < 5:
        return s

    dp = [0]
    for i in range(1, len(s)):
        j = dp[i-1]
        while j > 0 and s[i] != s[j]:
            j = dp[j-1]
        if s[i] == s[j]:
            j += 1
        dp += [j]
    print(dp)

    smallest = s
    for prefix_len in range(len(s), 0, -1):
        repeat_len = prefix_len - dp[prefix_len-1]
        if repeat_len < prefix_len and prefix_len % repeat_len == 0: # we have a repeat
            prefix_compressed = mincode(s[:repeat_len])
            left_compressed = f'{prefix_len // repeat_len}[{prefix_compressed}]'
        else:
            left_compressed = s[:prefix_len]

        right_compressed = mincode(s[prefix_len:])
        if len(left_compressed) + len(right_compressed) < len(smallest):
            smallest = left_compressed + right_compressed

    return smallest


print(mincode('abbbabbbcabbbabbbc'))
