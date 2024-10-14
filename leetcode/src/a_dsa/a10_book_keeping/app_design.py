
# LC355. Design Twitter
class Twitter:
    def __init__(self):
        self._user_followee = defaultdict(set)  # follower -> followee
        self._tweets_by_user = [] # tweet_id, poster_id, timestamp
    def postTweet(self, userId: int, tweetId: int) -> None:
        self._tweets_by_user.append((tweetId, userId))
    def getNewsFeed(self, userId: int) -> List[int]:
        users = set(self._user_followee[userId])
        users.add(userId)
        ret, idx = [], 1
        while idx <= len(self._tweets_by_user):
            tweet = self._tweets_by_user[-idx]  # most recent at end
            idx += 1
            if tweet[1] in users:
                ret.append(tweet[0])
                if len(ret) > 9: break
        return ret
    def follow(self, followerId: int, followeeId: int) -> None:
        self._user_followee[followerId].add(followeeId)
    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self._user_followee:
            self._user_followee[followerId].discard(followeeId)
