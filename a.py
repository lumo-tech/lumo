from lumo.contrib.preprocess.mpqueue import Queue

q = Queue('1')

for i in range(100):
    q.push(i)
    print(i)

while q.count > 0:
    print(q.popk(5))
