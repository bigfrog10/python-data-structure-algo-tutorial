################################################################################
# Given a string, determine if it is a palindrome, considering only alphanumeric
# characters and ignoring cases.
#
# Note: For the purpose of this problem, we define empty string as valid
# palindrome.
# EASY
################################################################################

def is_palindrome1(s: str) -> bool:
    char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'

    s = ''.join((i.lower() for i in s if i.lower() in char_set))

    if s is None or s == '':
        return True

    start, end = 0, len(s) - 1
    while start < end and s[start] == s[end]:
        start += 1
        end -= 1

    return start == end or start == end + 1


def is_palindrome(s: str) -> bool:
    if s is None or s == '':
            return True

    char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'

    start, end = 0, len(s) - 1
    while start < end:  # and s[start] == s[end]:
        s1, s2 = s[start].lower(), s[end].lower()
        while s1 not in char_set and start < len(s)-1:
            start += 1
            s1 = s[start].lower()
        s1 = '' if s1 not in char_set else s1

        while s2 not in char_set and end > 0:
            end -= 1
            s2 = s[end].lower()
        s2 = '' if s2 not in char_set else s2

        if s1 == s2:
            start += 1
            end -= 1
        else:
            return False

    return True

print(is_palindrome("A man, a plan, a canal: Panama"))
print(is_palindrome("race a car"))
print(is_palindrome(".,"))
