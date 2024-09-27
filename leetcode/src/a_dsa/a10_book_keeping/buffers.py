# LC158. Read N Characters Given Read4 II - Call multiple times
class Solution:
    def __init__(self):
        self.q = []
    def read(self, buf, n):
        i = 0
        while i < n:
            if self.q:
                buf[i] = self.q.pop(0)
                i += 1
            else:
                buf4 = [''] * 4
                v = read4(buf4)
                if v == 0: break
                self.q += buf4[:v]
        return i

# LC157. Read N Characters Given Read4
def read(self, buf, n):
    size = 0
    while size < n:
        b = [''] * 4
        count = read4(b)
        if count == 0: break
        bs = min(n - size, count)
        buf[size:] = b[:bs]
        size += bs
    return min(size, n)


