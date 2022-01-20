# LC528. Random Pick with Weight
class Solution:  # based on pdf
    def __init__(self, w: List[int]):
        self.samples = [i for i in range(len(w))]
        s = sum(w)
        self.weights = [ x / s for x in w ]
        self.cumusum = list(itertools.accumulate(self.weights))  # another way to do it, faster
    def pickIndex(self) -> int:
        r = random.uniform(0, 1)
        idx = bisect.bisect(self.cumusum, r)
        return self.samples[idx]



# LC398. Random Pick Index
class Solution:
    def __init__(self, nums: List[int]):
        self.num2idx = defaultdict(list)
        for i, n in enumerate(nums): self.num2idx[n].append(i)
    def pick(self, target: int) -> int:
        idxs = self.num2idx[target]
        i = random.randrange(len(idxs))
        return idxs[i]

# LC384. Shuffle an Array
import random
class Solution:
    def __init__(self, nums: List[int]):
        self.n = nums
        self.nc = list(nums)
        self.len = len(nums)
    def reset(self) -> List[int]:
        self.n = list(self.nc)
        return self.n
    def shuffle(self) -> List[int]:
        # return random.shuffle(self.n)  # this works but cheating

        # Fisher-Yates shuffle:
#         aux = list(self.array)

#         for idx in range(len(self.array)):
#             remove_idx = random.randrange(len(aux))
#             self.array[idx] = aux.pop(remove_idx)  # remove from aux, shrink the size, slow.

#         return self.array
        # Knuth, ACP
        for i in range(self.len):
            rand_idx = random.randint(i, self.len-1)
            # swap generated to the front
            self.n[i], self.n[rand_idx] = self.n[rand_idx], self.n[i]
            # be careful here: https://blog.codinghorror.com/the-danger-of-naivete/
        return self.n
