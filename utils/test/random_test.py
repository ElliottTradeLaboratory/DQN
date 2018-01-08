import numpy as np

values = []

"""
np.random.seed(1)
for _ in range(10000000):
    values.append(np.random.randint(0,4))

print('{:.13f}'.format(np.var(values)))
"""

classes=[]
np.random.seed(1)
for _ in range(10000000):
    values.append(np.random.uniform())
    classes.append(int(values[-1] * 10)/10)

print('{:.13f}'.format(np.var(values)))

from collections import Counter
c = Counter(classes)
for n,v in c.most_common():
    print(n,v)
