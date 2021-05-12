# LC1487. Making File Names Unique
def getFolderNames(self, names: List[str]) -> List[str]:
    used = set()
    counter = defaultdict(int)
    result = []
    for name in names:
        count = counter[name]
        candidate = name
        while candidate in used:
            count += 1
            candidate = f'{name}({count})'
        counter[name] = count
        result.append(candidate)
        used.add(candidate)
    return result
