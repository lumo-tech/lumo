from lumo.contrib.preprocess.mpqueue import Bucket

q = Bucket('2')

for i in range(100):
    q.push(i)
    print(i)

print(q.bucket(5, 20))
print(q.bucket(1, 20))
