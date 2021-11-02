def cmp_scalar(a, b, eps=1e-3):
    residual = a - b
    if residual > eps:
        return 1
    elif residual < -eps:
        return -1
    return 0
