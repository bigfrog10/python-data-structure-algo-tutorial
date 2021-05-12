# You are given two strings s and t. You can select any substring of string s and rearrange the characters of the selected substring. Determine the minimum length of the substring of s such that string t is a substring of the selected substring.
# Signature
# int minLengthSubstring(String s, String t)
# Input
# s and t are non-empty strings that contain less than 1,000,000 characters each
# Output
# Return the minimum length of the substring of s. If it is not possible, return -1
# Example
# s = "dcbefebce"
# t = "fd"'
# output = 5
# Explanation:
# Substring "dcbef" can be rearranged to "cfdeb", "cefdb", and so on. String t is a substring of "cfdeb". Thus, the minimum length required is 5.
# LC76.
from collections import Counter

def min_length_substring(s, t):
    n, cnts = len(t), Counter(t)
    i, ret = 0, float('inf')
    for j, c in enumerate(s, 1): # start index at 1
        n -= cnts[c] > 0  # count down
        cnts[c] -= 1
        if n == 0: # find a window containing t
            while i < j and cnts[s[i]] < 0: # catch upt to make length minimal
                cnts[s[i]] += 1
                i += 1
            ret = min(ret, j-i)
    return -1 if ret == float('inf') else ret

print(min_length_substring("dcbefebce", "fd"))
