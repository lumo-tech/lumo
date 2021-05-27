from concurrent import futures

executor = futures.ProcessPoolExecutor(3)


def p(*args, **kwargs):
    return args, kwargs


for k in executor.map(p, enumerate([5, 1, 4, 2, 5, 6])):
    print(k)
