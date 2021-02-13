

def containsDuplicate(self, nums: List[int]) -> bool:
    import collections

    counts = collections.Counter(nums)
    dups = filter(lambda x: x[1] > 1, counts.items())
    return len(list(dups)) > 0

