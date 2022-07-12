# https://www.codespeedy.com/burrows-wheeler-transform-in-python/

a = 'codespeedy$'
words = list(a)
list = []
for i in range(len(words)):
    word = a[-1] + a[:-1]
    new = ''.join(word)
    a = new
    list.append(new)
    i += 1
print(list)
sort = sorted(list)
print(sort)
for i in range(len(words)):
    element = sort[i]
    last = element[- 1]
    i = i + 1
    print(last)

def bw_transform(s):
    n = len(s)
    m = sorted([s[i:n]+s[0:i] for i in range(n)])
    I = m.index(s)
    L = ''.join([q[-1] for q in m])
    return I, L

# https://gist.github.com/dmckean/9723bc06254809e9068f
print(bw_transform(a))
