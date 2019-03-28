import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import mglearn


# 회귀 알고리즘
# 인위적으로 만든 wave 데이터셋 사용
X, y = mglearn.datasets.make_wave(n_samples=40)

plt.plot(X,y,'o')
plt.ylim(-3, 3)
plt.xlabel("Characteristics")
plt.ylabel("Targets")

plt.show()