import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_boston

boston = load_boston()
print("shape: {}".format(boston.data.shape))
print("Boston.keys():\n{}".format(boston.keys()))