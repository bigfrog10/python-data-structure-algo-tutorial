import math
class Solution:
    def consecutiveNumbersSum1(self, N: int) -> int:  # naive working, only for small numbers, exceed max recursion
        def cons_num_sum_start_from(n, start):
            if start < 0 or n < 0:
                return 0
            if start > n:
                return 0
            if start == n:
                return 1

            return cons_num_sum_start_from(n-start, start-1)

        if N < 3:
            return 1

        total = 1  # n itself
        for i in range(1, N // 2 + 2):
            total += cons_num_sum_start_from(N, i)

        return total

    def consecutiveNumbersSum2(self, N: int) -> int:
        dp = [[0] * i for i in range(1, N+1)]  # consecutive number sum of n starting from m
        for i in range(0, N):
            dp[i][i] = 1  #

        for i in range(1, N):
            for j in range(0, i):
                if i >= j+1 and 1 <= j <= len(dp[i-j-1]):
                        dp[i][j] = dp[i-j-1][j-1]

        total = 0
        for i in range(N):
            total += dp[N-1][i]

        return total

    def consecutiveNumbersSum(self, N: int) -> int:
        if N < 3:
            return 1

        count = 1
        # worst case is 1 + 2 + ... + n = N, so n(n+1) = N, solve for n
        upper = (math.sqrt(8 * N + 1) - 1) / 2
        for i in range(2, math.floor(upper) + 1):
            m = N // i
            h = i // 2

            # total = 0
            # for w in range(i):
            #     if m + h - w > 0:
            #         total += m + h - w
            #     else:
            #         total = 0  # to signal that we are count 0s, not good.
            #         break
            if i % 2 == 0:
                a = m - h + 1
                total = (a + m + h) * i / 2
            else:
                total = m * i
            if total == N:
                count += 1

        return count


# print(Solution().consecutiveNumbersSum(1))  # 1
# print(Solution().consecutiveNumbersSum(2))  # 1
print(Solution().consecutiveNumbersSum(3))  # 2
print(Solution().consecutiveNumbersSum(4))  # 1
print(Solution().consecutiveNumbersSum(5))  # 2
print(Solution().consecutiveNumbersSum(9))  # 3
print(Solution().consecutiveNumbersSum(15))  # 4
print(Solution().consecutiveNumbersSum(85895))  # 8
print(Solution().consecutiveNumbersSum(858951))  # 16
print(Solution().consecutiveNumbersSum(1810158))  # 24
