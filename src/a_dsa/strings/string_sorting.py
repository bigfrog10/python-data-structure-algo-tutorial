
# LC953. Verifying an Alien Dictionary
from typing import List
def isAlienSorted(self, words: List[str], order: str) -> bool:
    orderd = {x: idx for idx, x in enumerate(order)}
    for w1, w2 in zip(words[:-1], words[1:]):
        idx = 0
        while idx < min(len(w1), len(w2)) and w1[idx] == w2[idx]: idx += 1
        if idx == min(len(w1), len(w2)): # one match other fully
            if len(w1) > len(w2): return False  # space is ahead of letters
        elif orderd[w1[idx]] > orderd[w2[idx]]: return False
    return True

# LC269. Alien Dictionary, top100  # BBG
def alienOrder(self, words: List[str]) -> str:
    adj_list = defaultdict(set)  #  the in_degree of each unique letter to 0.
    in_degree = Counter({c: 0 for word in words for c in word})
    for first_word, second_word in zip(words[:-1], words[1:]):
        first_word = first_word.strip()
        second_word = second_word.strip()
        if first_word.startswith(second_word) and len(first_word) > len(second_word):
            return ""  # This is a strange requirement, we should just skip this

        for c, d in zip(first_word, second_word):
            if c != d:
                if d not in adj_list[c]:
                    adj_list[c].add(d)  # c -> d
                    in_degree[d] += 1
                break

    output = []  # need to repeatedly pick off nodes with an indegree of 0.
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    while queue:
        c = queue.popleft()
        output.append(c)
        for d in adj_list[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0: queue.append(d)
    if len(output) < len(in_degree): return ""  # cyclic ordering
    return "".join(output)

# LC791. Custom Sort String
def customSortString(self, S: str, T: str) -> str:
    count = collections.Counter(T)
    ans = []
    for c in S:
        ans.append(c * count[c])
        count[c] = 0
    for c in count: ans.append(c * count[c])
    return "".join(ans)

# LC451. Sort Characters By Frequency
def frequencySort(self, s: str) -> str:
    counts = Counter(s)
    keys = sorted(counts.keys(), reverse=True, key=lambda x: counts[x])
    ret = ''
    for k in keys: ret += k * counts[k]
    return ret
