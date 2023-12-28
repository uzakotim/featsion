import numpy as np
# create 3d matrix
a = np.zeros((800,800,160))
a[4][4][10] = 1
print(a[8][8][10])