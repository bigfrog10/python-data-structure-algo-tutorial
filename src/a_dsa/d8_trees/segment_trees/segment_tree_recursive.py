def st_build(node, start, end, st, vals, nf):
    if start == end:
        st[node] = vals[start]
    else:
        mid = (start + end) // 2
        st_build(2*node, start, mid, st, vals, nf)
        st_build(2*node+1, mid+1, end, st, vals, nf)
        st[node] = nf(st[2*node], st[2*node+1])


def st_query(node, start, end, left, right, st, nf):
    if right < start or left > end:  # out of range
        return None
    if left <= start and end <= right:  # full range
        return st[node]
    # partial range
    mid = (start + end) // 2
    p1 = st_query(2*node, start, mid, left, right, st, nf)
    p2 = st_query(2*node+1, mid+1, end, left, right, st, nf)
    return nf(p1, p2)


def st_update(node, start, end, idx, val, st, vals, nf):
    if start == end:
        vals[idx] += val
        st[node] += val
    else:
        mid = (start + end) // 2
        if start <= idx <= mid:
            st_update(2*node, start, mid, idx, val, st, vals, nf)
        else:  # on second half
            st_update(2*node+1, mid+1, end, idx, val, st, vals, nf)

        st[node] = nf(st[2*node], st[2*node+1])


arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
f = lambda x, y: x + y if x and y else x

segt = [0] * (4*len(arr))
st_build(1, 0, len(arr)-1, segt, arr, f)
print(segt)
print(st_query(1, 0, len(arr)-1, 1, 3, segt, f))
