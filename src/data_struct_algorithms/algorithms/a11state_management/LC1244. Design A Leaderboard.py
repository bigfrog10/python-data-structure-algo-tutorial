# LC1244. Design A Leaderboard
class Leaderboard:
    def __init__(self):
        self.A = collections.Counter()
    def addScore(self, playerId: int, score: int) -> None:
        self.A[playerId] += score
    def top(self, K: int) -> int:
        return sum(v for i,v in self.A.most_common(K))
    def reset(self, playerId: int) -> None:
        self.A[playerId] = 0
