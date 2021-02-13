################################################################################
# Longest Substring Without Repeating Characters
# Given a string, find the length of the longest substring without repeating
# characters. For example
# Given "abcabcbb", the answer is "abc", which the length is 3
# Given "bbbbb", the answer is "b", with the length of 1
# Given "pwwkew", the answer is "wke", with the length of 3
# Given "" the answer is "", with the length of 0
# Note that the answer must be a substring, "pwke" is a subsequence and not a
# substring.
# MEDIUM
################################################################################

# https://zxi.mytechroad.com/blog/string/leetcode-3-longest-substring-without-repeating-characters/
# keep track substring starting point, keep track last index of every char.
#    str  abcc
#    idx  0123
#    st   0003
#    ans  1233
#    last 013 <- 3 is changed from 2
#         abc <- 97, 98, 99
def longest_substring_no_char_dupe(s: str):
    last = [-1] * 128  # used to track last index of every char. treat space for speed.
    st = 0  # substring starting point
    ans = 0
    for idx, char in enumerate(s):
        if last[ord(char)] != -1:  # showed before
            st = max(st, last[ord(char)] + 1)  # max across different chars

        ans = max(ans, idx - st + 1)  # if no repeat, idx - st + 1 will increase, otherwise use ans
        last[ord(char)] = idx  # update char's index

    return ans


print(longest_substring_no_char_dupe('abcabcbb') == 3)

print(longest_substring_no_char_dupe('bbbbb') == 1)

print(longest_substring_no_char_dupe('pwwkew') == 3)

print(longest_substring_no_char_dupe('') == 0)

print(longest_substring_no_char_dupe('abcc') == 3)

print(longest_substring_no_char_dupe('aabc') == 3)
