"""
Project: TC2
Description: test ComputeError.py
"""

import numpy as np
from TC2.utils import ComputeError as ce

X1 = np.zeros([2,2,2])
X2 = np.ones([2,2,2])
X3 = np.random.uniform(size = 8).reshape([2,2,2])

print ce.L2Error(X1, X2)
print ce.L2Error(X1, X3)