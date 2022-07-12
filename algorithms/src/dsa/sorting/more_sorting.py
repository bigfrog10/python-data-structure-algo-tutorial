
# https://www.programiz.com/dsa/counting-sort
# https://stackabuse.com/radix-sort-in-python/

def counting_sort(array):
    """
    sort array from small to large in place.
    Runtime is O(N + m), N = len(array), m = max(array)
    Space is O(m)
    :param array: array of non-negative numbers
    :return: None
    """
    max_v = max(array)
    arr_len = max_v + 1

    count = [0] * arr_len
    for v in array:
        count[v] += 1

    i = 0  # index of array
    for j in range(arr_len):
        for k in range(count[j]):  # there are count[j] of value j to be set
            array[i+k] = j
        i = i + count[j]


# arr = [4, 2, 2, 8, 3, 3, 1]
# counting_sort(arr)
# print(arr)

def radix_sort(array):
    def _counting_sort(arr, nth_digit):
        count = [0] * 10  # 10 buckets, 0, 1, 2, ..., 9
        for v in array:  # bucket sorting
            idx = (v // nth_digit) % 10
            count[idx] += 1
        for i in range(1, 10):  # cumulative counts
            count[i] += count[i - 1]

        arr_len = len(arr)
        output = [0] * arr_len
        i = arr_len - 1
        while i >= 0:
            index = (arr[i] // nth_digit) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1
        for i in range(arr_len):
            arr[i] = output[i]

    max_v = max(array)
    digit_place = 1
    while max_v // digit_place > 0:
        _counting_sort(array, digit_place)
        digit_place *= 10


arr1 = [121, 432, 564, 23, 1, 45, 788]
radix_sort(arr1)
print(arr1)
