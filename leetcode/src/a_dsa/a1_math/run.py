
def josephus(n: int, k: int):
    """n people form a circle, whoever counts k is out. Find the index of last person.
        People are 1 based indexed
        math:
            find max m s.t. 2^m < n, then 2m+1 is the solution
            if 2^m == n or n == 1, then 1 is the solution
            https://www.ucd.ie/mathstat/t4media/1.%20The%20Josephus%20problem.pdf
    """
    r = 0
    for i in range(2, n+1):
        r = (r + k) % i
    return r + 1
    # this is the nonrecursive version of below

def josephus2(n: int, k: int):
    if n == 1: return 1
    return (josephus2(n-1, k) + k - 1) % n + 1
    # above is 1-based solution, kind of messy.
    # change it to 0 based, it's just
    # j(n, k) = (j(n-1, k) + k) % n
    # after removing kth, move the 1st k-1 elements to the array end
    # then call j(n-1, k), then we need to add k back.
    # https://cp-algorithms.com/others/josephus_problem.html
    # this link has a klogn algo

if __name__ == "__main__":
    res = josephus2(5, 2)
    print(f'res={res}')  # 3
    res = josephus2(14, 2)
    print(f'res={res}')  # 13