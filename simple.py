import numpy as np
from time import time
from statistics import mean

a = [n for n in range(1000)]
b = np.array(a)
d = {n: n for n in range(1000)}

ts = time()
c = mean(a)
print('%.4f' % (1000 * (time() - ts)))

ts = time()
e = b.mean()
print('%.4f' % (1000 * (time() - ts)))

ts = time()
f = mean(d)
print('%.4f' % (1000 * (time() - ts)))