################################################################################
# Return the index of the first occurrence of needle in haystack, or -1 if
# needle is not part of haystack.
# Clarification:
# What should we return when needle is an empty string? This is a great question
# to ask during an interview.
# For the purpose of this problem, we will return 0 when needle is an empty
# string. This is consistent to C's strstr() and Java's indexOf().
# EASY
################################################################################
# rabin karp algorithm

def str_str(haystack: str, needle: str) -> int:
    if not needle:  # '' or None
        return 0

    if not haystack:  # order matters, needle is not empty
        return -1

    h_len = len(haystack)
    n_len = len(needle)
    for i in range(h_len - n_len + 1):  # if they are equal, we need to run once.
        flag = True
        for j in range(n_len):
            if needle[j] != haystack[i+j]:
                flag = False
                break

        if flag:
            return i

    return -1


print(str_str('hello', 'll'))
print(str_str('aaaaa', 'bba'))
print(str_str('', ''))
print(str_str('a', 'a'))
