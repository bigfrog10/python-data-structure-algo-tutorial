import functools

class Solution:
    @functools.lru_cache(maxsize=10000)
    def numDecodings1(self, s: str) -> int:  # lru cache version
        if not s:
            return 0

        if s.startswith('0'):
            return 0

        if not s.isnumeric():
            return 0

        if len(s) == 1:
            return 1

        if len(s) == 2:
            if s[1] == '0':
                return 1 if int(s[0]) <= 2 else 0
            else:
                if int(s) <= 26:
                    return 2
                else:
                    return 1

        count = 0
        if s[1] == '0':
            if int(s[0]) <= 2:
                count += self.numDecodings(s[2:])
            else:
                return 0
        elif s[1] >= '7':
            if s[0] == '1':
                count += self.numDecodings(s[2:])  # can't go beyond Z, 26. So 7/8/9 is with 1
                count += self.numDecodings(s[1:])
            else:
                count += self.numDecodings(s[1:])
        else:
            if int(s[0:2]) <= 26:
                count += self.numDecodings(s[2:])
                count += self.numDecodings(s[1:])
            else:
                count += self.numDecodings(s[1:])

        return count

    def numDecodings(self, s: str) -> int:  # dp version
        dp = {0: 0}  # store count for last m chars

        dp[1] = 1 if 0 < int(s[-1]) <= 9 else 0

        if len(s) >= 2:
            if s[-1] == '0':
                dp[2] = 1 if s[-2] == '1' or s[-2] == '2' else 0  # for 10s
            elif s[-2] == '0':
                dp[2] = 0
            else:
                if int(s[-2:]) <= 26:
                    dp[2] = 2  # both or 2 singles
                else:
                    dp[2] = 1  # only 2 singles

        for i in range(3, len(s)+1):
            if s[-i] == '0':
                dp[i] = 0
            elif s[-i+1] == '0':
                dp[i] = dp[i-2] if s[-i] == '1' or s[-i] == '2' else 0
            else:
                if int(s[-i:-i+2]) <= 26:
                    dp[i] = dp[i-1] + dp[i-2]  # both or 2 singles
                else:
                    dp[i] = dp[i-1]  # only 2 singles

        return dp[len(s)]


sol = Solution()
# print(Solution().numDecodings('0'))  # 0
# print(Solution().numDecodings('1'))  # 1
# print(Solution().numDecodings('10'))  # 1
# print(Solution().numDecodings('12'))  # 2
# print(Solution().numDecodings('17'))  # 2
# print(Solution().numDecodings('27'))  # 1
# print(Solution().numDecodings('226'))  # 3
# print(sol.numDecodings('230'))  # 0
# print(sol.numDecodings('301'))  # 0
# print(Solution().numDecodings("1201234"))  # 3
# print(Solution().numDecodings("123"))  # 3
# print(Solution().numDecodings("1231"))  # 3
print(Solution().numDecodings("12312"))  # 6
print(Solution().numDecodings("123123"))  # 9

print(sol.numDecodings("111111111111111111111111111111111111111111111"))  # 1836311903
# print(sol.numDecodings.cache_info())
