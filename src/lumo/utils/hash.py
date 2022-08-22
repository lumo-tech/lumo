from hashlib import md5


def hash_iter(*object: str):
    hasher = md5()
    for i in object:
        hasher.update(i.encode())
    return hasher.hexdigest()
