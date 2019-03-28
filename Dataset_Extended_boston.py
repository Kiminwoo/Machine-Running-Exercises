import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_boston

X, y = mglearn.datasets.load_extended_boston()

print("X.shape: {}".format(X.shape))

plt.show()