def word_machine(string: str) -> int:
    if not string:
        return -1

    lb, ub = 0, 2 ** 20 - 1
    instructions = string.split(' ')
    stack = []
    for ins in instructions:
        if ins in '+-':
            if len(stack) < 2:
                return -1  # no enough operands
            f = stack.pop()
            s = stack.pop()
            if ins == '+':
                r = f + s
                if r > ub:
                    return -1  # overflow
            else:
                r = f - s
                if r < lb:
                    return -1  # underflow
            stack.append(r)
        elif ins == 'DUP':
            if len(stack) < 1:
                return -1  # no enough operand
            stack.append(stack[-1])
        elif ins == 'POP':
            if len(stack) < 1:
                return -1  # no enough operand
            stack.pop()
        elif ins.isnumeric():  # not include negatives
            v = int(ins)
            if v > ub:
                return -1  # overflow

            stack.append(v)
        else:
            return -1  # unknown instruction, including negative numbers.

    if len(stack) == 0:
        return -1  # no result

    return stack.pop()


print(word_machine(''))  # -1
print(word_machine('4 5 6 - 7 +'))  # 6 - 5 + 7 = 8
print(word_machine('13 DUP 4 POP 5 DUP + DUP + -'))  # 7
print(word_machine('5 6 + -'))  # -1
print(word_machine('3 DUP 5 - -'))  # -1
print(word_machine('3 POP'))  # -1
print(word_machine(''))
print(word_machine(''))
print(word_machine(''))
print(word_machine(''))
