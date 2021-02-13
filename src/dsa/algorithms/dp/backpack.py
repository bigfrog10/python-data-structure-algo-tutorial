# Backpack or Knapsack problem:
# Given n items, values are v[i] and weights are w[i], for i in [1 ... n]. The backpack can
# holds at most weight W when we put items in it. What's the most value we could get?
#
# This is 0-1 problem, meaning for each item we either take it or not. We can't take partials.
# Recursion is on optimal solution for first i items and weight limit u:
#    s[i, u] = s[i-1, w], we don't ith item in the backpack
#              s[i-1, u - w[i]] + v[i], we put the item in the backpack if w[i] <= u
# Base cases/boundary conditions for the recursion are:
#    s[0, u] = 0 for all u, because there is no item available
#    s[i, 0] = 0 for all i, because there is no room to put
# So the key is the decision whether we include ith item or not.

# recursions

def backpack(w: list, v: list, target: int):
    w = [0] + w # we pack w and v to be 1-based values
    v = [0] + v

    def pack(item_idx, weight_limit):
        # base cases
        if item_idx == 0:
            return 0

        if weight_limit == 0:
            return 0

        leave_out = pack(item_idx - 1, weight_limit)
        if w[item_idx] <= weight_limit:
            taken = pack(item_idx - 1, weight_limit - w[item_idx]) + v[item_idx]
            return max(taken, leave_out)
        else:
            return leave_out

    return pack(len(w)-1, target)


# taken from https://en.wikipedia.org/wiki/Knapsack_problem
wl = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
vl = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
wt = 67
result = backpack(wl, vl, wt)
print(result)  # 1270

####################################################################################################
# There are repetitive computations in pack() calls, so we cache them
import functools


def backpack_cache(w: list, v: list, target: int):
    w = [0] + w  # we pack w and v to be 1-based values
    v = [0] + v

    @functools.lru_cache(None)
    def pack(item_idx, weight_limit):  # we don't include w and v here, otherwise caching is awkward
        # base cases
        if item_idx == 0:
            return 0

        if weight_limit == 0:
            return 0

        leave_out = pack(item_idx - 1, weight_limit)
        if w[item_idx] <= weight_limit:
            taken = pack(item_idx - 1, weight_limit - w[item_idx]) + v[item_idx]
            return max(taken, leave_out)
        else:
            return leave_out

    ret = pack(len(w)-1, target)
    print(pack.cache_info())
    return ret


result = backpack_cache(wl, vl, wt)
print(result)  # 1270, we get 23 repetitive computations cached


####################################################################################################
# DP - we take recursion pack() to arrays, including boundaries
# DP on most value for items and weights < limit
def backpack_dp(w: list, v: list, target: int):
    w = [0] + w  # we pack w and v to be 1-based values
    v = [0] + v

    n = len(w)
    dp = [[0] * (target + 1) for _ in range(n)]

    for i in range(1, n):
        for j in range(1, target+1):
            if w[i] <= j:
                taken = dp[i-1][j-w[i]] + v[i]
                dp[i][j] = max(dp[i-1][j], taken)
            else:
                dp[i][j] = dp[i-1][j]

    print(dp)
    return dp[n-1][target]


result = backpack_dp(wl, vl, wt)
print(result)  # 1270
#    0 18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	 47	    48	 49	    50	 51	    52	 53	    54	 55	    56	 57	    58	 59	    60	 61	    62	 63	    64	 65	    66	 67
# 0  0 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	 0	    0	 0	    0	 0	    0	 0	    0	 0	    0	 0	    0	 0	    0	 0	    0	 0	    0	 0	    0	 0
# 1  0 0	0	0	0	0	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	 505	505	 505	505	 505	505	 505	505	 505	505	 505	505	 505	505	 505	505	 505	505	 505	505	 505
# 2  0 0	0	0	0	0	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	 505	505	 857	857	 857	857	 857	857	 857	857	 857	857	 857	857	 857	857	 857	857	 857	857	 857
# 3  0 0	0	458	458	458	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	963	963	963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963
# 4  0 220	220	458	458	458	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	678	678	678	725	725	963	963	963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 1183	1183 1183	1183 1183	1183 1183
# 5  0 220	220	458	458	458	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	678	678	678	725	725	963	963	963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 1183	1183 1183	1183 1183	1183 1183
# 6  0 220	220	458	458	458	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	678	678	678	725	725	963	963	963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 963	963	 1183	1183 1183	1183 1183	1183 1183
# 7  0 220	220	458	458	458	505	505	505	505	505	505	505	505	505	505	505	505	505	505	505	678	678	678	725	725	963	963	963	963	 963	963	 963	963	 963	1003 1003	1003 1003	1003 1003	1003 1003	1003 1183	1183 1183	1183 1183	1183 1183
# 8  0 220	220	458	458	458	505	505	505	545	545	545	545	545	545	545	545	545	545	545	545	678	678	678	725	725	963	963	963	1003 1003	1003 1050	1050 1050	1050 1050	1050 1050	1050 1050	1050 1050	1050 1183	1183 1183	1223 1223	1223 1270
# 9  0 220	220	458	458	458	505	505	505	545	545	545	545	545	545	545	545	545	545	545	545	678	678	678	725	725	963	963	963	1003 1003	1003 1050	1050 1050	1050 1050	1050 1050	1050 1050	1050 1050	1050 1183	1183 1183	1223 1223	1223 1270
# 10 0 220	220	458	458	458	505	505	505	545	545	545	545	545	545	545	545	545	545	545	545	678	678	678	725	725	963	963	963	1003 1003	1003 1050	1050 1050	1050 1088	1088 1088	1088 1088	1088 1088	1088 1183	1183 1183	1223 1223	1223 1270

