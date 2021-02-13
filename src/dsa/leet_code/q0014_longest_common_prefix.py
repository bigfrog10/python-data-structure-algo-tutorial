################################################################################
# Write a function to find the longest common prefix string amongst an array of
# strings.
# If there is no common prefix, return an empty string "".
# EASY
################################################################################

def longest_common_prefix(strs: list) -> str:
    res = ''
    idx = 0
    while idx <= 200:  # or True, loop idx
        tmp = None
        for s in strs:
            if idx >= len(s):
                return res
            c = s[idx]
            if tmp is None:
                tmp = c
            else:
                if tmp != c:
                    return res

        if tmp:
            res += tmp
        else:
            return res
        idx += 1

    return res


print(longest_common_prefix(["flower", "flow", "flight"]))
print(longest_common_prefix(["dog", "racecar", "car"]))
