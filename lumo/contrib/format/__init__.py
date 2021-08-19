from numbers import Number


def to_human(n: Number, lang='zh') -> str:
    if isinstance(n, int):
        if n < 1000:
            return f'{n}'
        elif n < 10000:
            return f'{n / 1000:.2f}'.strip('0').strip('.') + 'k'
        elif n < 1e+9:
            return f'{n / 10000:.2f}'.strip('0').strip('.') + 'w'
        elif n < 1e+12:
            return f'{n / 1e+9:.2f}m'
        # else:
        #     return f'{n / 1e+9:.2f}b'
    elif isinstance(n, float):

        return f'{n:.2e}'


print(to_human(142000000))
