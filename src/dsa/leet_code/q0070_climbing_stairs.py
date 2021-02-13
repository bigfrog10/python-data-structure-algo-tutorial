################################################################################
#
# EASY
################################################################################

def climb_stairs(n: int) -> int:
    if n < 3:
        return n

    n1, n2 = 1, 2
    for i in range(n-2):
        n1, n2 = n2, n1 + n2

    return n2


for n in range(10):
    print(f'i={n}, n={climb_stairs(n)}')
