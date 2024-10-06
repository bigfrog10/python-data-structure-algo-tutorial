
# LC1140. Stone Game II
def stoneGameII(self, piles):  # O(n^3) time, O(n^2) space
    a = [*itertools.accumulate(piles[::-1])][::-1]  # tail sum
    @functools.cache
    def game(i, m):  # max stones start from pile i with maximal take m
        if i + 2 * m >= len(piles): return a[i]  # take all if we can
        min_score = inf  # rival's score
        for x in range(1, 2 * m + 1):  # find which move can get maximum
            score = game(i + x, max(x, m))  # get rival's score
            if score < min_score: min_score = score  # keep rival's minimal
        return a[i] - min_score  # keep us maximal
    return game(0, 1)

