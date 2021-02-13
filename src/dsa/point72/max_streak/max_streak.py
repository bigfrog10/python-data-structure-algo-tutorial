

def max_showup(attendences: list) -> int:
    if not attendences:
        return 0

    # number of days and number of employees
    nd, ne = len(attendences), len(attendences[0])
    a = attendences

    start, walker = -1, 0
    all_attn = 'Y' * ne
    maxa = 0
    while walker < nd:
        if a[walker] == all_attn:
            if start == -1:
                start = walker  # mark start of full attn
            walker += 1
        else:
            if start >= 0:
                maxa = max(maxa, walker-start)
                start = -1
            walker += 1
    else:
        if start >= 0:
            maxa = max(maxa, walker-start)

    return maxa


print(max_showup([]))  # 0
attns = ['YYY', 'YYY', 'YNN', 'YYN', 'YYN']
print(max_showup(attns))  # 2

attns = ['YNN', 'YYY', 'YYY', 'YNN']
print(max_showup(attns))  # 2

attns = ['YNN', 'YYY', 'YYY', 'YNN', 'YYY', 'YYY', 'YYY']
print(max_showup(attns))  # 3
