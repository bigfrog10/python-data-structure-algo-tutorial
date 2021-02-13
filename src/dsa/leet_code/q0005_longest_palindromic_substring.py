################################################################################
# Given a string s, return the longest palindromic substring in s. You may
# assume that the maximum length of s is 1000. For example, "babad" has bab and
# aba, cbbd has bb.
# MEDIUM
################################################################################
# https://www.cnblogs.com/grandyang/p/4464476.html
# https://goodtecher.com/leetcode-5-longest-palindromic-substring/
#
# palindrome: reading the same backward as forward, such as level, noon.
# DP solution:
#     - if first char and last char in a string are equal
#     - inside substring(i.e., minus first and last chars) is a palindrome.
# Translate these into matrix: fill in upper diagonal matrix vertically.
# Time is O(n^2).
#     ip = bool[len][len]
#     ip[i][j] = s[i] == s[j] if j==i+1 else ip[i+1][j-1] (j > i+1)
#     ip[j][j] = true

def longest_palindrome1(s: str) -> str:
    is_pal = []
    start, end = 0, 0
    for j in range(len(s)):  # loop columns
        col = []
        for i in range(j+1):  # loop rows until diagonal line
            if i == j:
                current = True
                col.append(current)
                is_pal.append(col)
            elif i + 1 == j:
                current = s[i] == s[j]
                col.append(current)  # double chars, like aa
            else:  # i+1 < j
                current = s[i] == s[j] and is_pal[j-1][i+1]
                col.append(current)

            if current:  # track longest
                if j - i + 1 > end - start + 1:  # return the first one
                    start = i
                    end = j

    # print(is_pal)
    return s[start:end+1]  # inclusive


def longest_palindrome(s: str) -> str:
    def find_diameter(s: str, left: int, right: int):
        while left > -1 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1

        return right - left - 1  # really is right - left + 1 - 2, minus 2 to go back the valid case.

    if not s or len(s) == 0:
        return ''

    start = end = 0
    for i in range(len(s)):
        len1 = find_diameter(s, i, i)
        len2 = find_diameter(s, i, i+1)
        d = max(len1, len2)
        if d > end - start:
            start = i - (d - 1) // 2  # to take care cases like aba
            end = i + d // 2  # to take care cases like abba

    return s[start:end + 1]

print(longest_palindrome('aaaa'))
print(longest_palindrome('babad'))
print(longest_palindrome('cbbd'))
print(longest_palindrome('a'))
print(longest_palindrome('ac'))

# Manacher's Algorithm, O(n)
# https://www.cnblogs.com/grandyang/p/4475985.html
