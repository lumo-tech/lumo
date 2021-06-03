from lumo.contrib.preprocess.mpqueue import Bucket

q = Bucket('3')

from lumo.contrib.itertools import window

for i in window(range(100), 10, 10):
    res = q.pushk(*i)
    print(i, res)

print(q.bucket(5, 20))
print(q.bucket(1, 20))
