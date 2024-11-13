

# swap 2 integers without temp
def swap(a, b):
    a = a + b
    b = a - b
    a = a - b
    return a, b

print(swap(2, 3))