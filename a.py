from lumo.contrib.itertools import poll
from lumo import DatasetBuilder, DataModule, DataBundler

x = DatasetBuilder().add_input('x', range(100)).add_output('x', 'x').DataLoader()
y = DatasetBuilder().add_input('x', range(200, 300)).add_output('x', 'x').DataLoader()

bundler = DataBundler().add(x).add(y).poll_mode()
# for i in bundler:
#     print(i)

print(len(bundler))
