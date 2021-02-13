################################################################################
# The count-and-say sequence is a sequence of digit strings defined by the
# recursive formula:
#    - countAndSay(1) = "1"
#    - countAndSay(n) is the way you would "say" the digit string from
#      countAndSay(n-1), which is then converted into a different digit string.
# EASY
################################################################################

def count_and_say(n: int) -> str:
    t = str(1)
    for i in range(0, n-1):
        t = say(t)

    return t


def say(s: str) -> str:
    res = ''
    tmp = None
    count = 0
    for c in s:
        if tmp:
            if tmp == c:
                count += 1
            else:
                res += str(count) + tmp
                tmp = c
                count = 1
        else:
            tmp = c
            count += 1

    res += str(count) + tmp

    return res


print(say('3322251'))

print(count_and_say(1))
print(count_and_say(2))
print(count_and_say(3))
print(count_and_say(4))
