import os

dist_dir = os.path.join(os.path.dirname(__file__),'dist')
fs = os.listdir(dist_dir)

fs = [os.path.join(i) for i in fs]

fs = sorted(fs, key=lambda x: os.path.getatime(os.path.join(x)), reverse=True)

print(fs)

