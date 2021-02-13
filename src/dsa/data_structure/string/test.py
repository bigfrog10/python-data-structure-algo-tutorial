def largestRectangleArea(self, arr):
    if arr == []:
        return 0
        maxi = float('-inf')
        q = []
        pos = []

        q.append(arr[0])
        pos.append(0)

        for i in range(1, len(arr)):
            if arr[i] == q[-1]:
                pass
            elif arr[i] < q[-1]:
                while q != [] and q[-1] > arr[i]:
                    a = q.pop()
                    b = pos.pop()
                    maxi = max(maxi, a * (i - b))

                if q == [] or q[-1] < arr[i]:
                    q.append(arr[i])
                    pos.append(b)

            elif arr[i] > q[-1]:
                q.append(arr[i])
                pos.append(i)

        for i, j in zip(q, pos):
            l = len(arr)
            maxi = max(maxi, i * (l - j))

        # print (maxi)
        # print (q,pos)
        return maxi
