
# LC912. Sort an Array
def sortArray(self, nums: List[int]) -> List[int]:  # bucket sort, O(n)
    bucket = [0]*100001
    for x in nums: bucket[x + 50000] += 1
    ans = []
    for i, x in enumerate(bucket, -50000):
        ans.extend([i]*x)
    return ans

def sortArray(self, nums: List[int]) -> List[int]:  # merge sort
    n = len(nums)
    if n > 1:
        list1 = self.sortArray(nums[:n//2])
        list2 = self.sortArray(nums[n//2:])
        nums = self.merge(list1, list2)
    return nums
def merge(self, list1, list2):
    sorted_list = []
    while list1 and list2:
        if list1[0] <= list2[0]:
            sorted_list.append(list1.pop(0))
        else:
            sorted_list.append(list2.pop(0))
    if not list1:
        sorted_list.extend(list2)
    if not list2:
        sorted_list.extend(list1)
    return sorted_list

def sortArray(self, N: List[int]) -> List[int]:  # quick sort
    def quicksort(A, I, J):
        if J - I <= 1: return
        p = partition(A, I, J)
        quicksort(A, I, p), quicksort(A, p + 1, J)

    def partition(A, I, J):
        A[J-1], A[(I + J - 1)//2], i = A[(I + J - 1)//2], A[J-1], I
        for j in range(I,J):
            if A[j] < A[J-1]: A[i], A[j], i = A[j], A[i], i + 1
        A[J-1], A[i] = A[i], A[J-1]
        return i

    quicksort(N,0,len(N))
    return N
