from lumo import Meter, Record
import numpy as np

record = Record()

for idx in range(500):  # imitate batch
    m = Meter()
    loss = np.random.rand()

    m.mean.loss = loss
    m.last.idx = idx
    m.sum.confusion = np.random.randint(0, 10, (10, 10))
    
    record.record(m)

attr = record.agg() # An Attr (easydict-like) object
print(attr)
print(attr['loss'])
print(attr.confusion)