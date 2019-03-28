import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# 데이터셋 만들기
# 인위적으로 만든 이진 분류 데이터셋
X, y = mglearn.datasets.make_forge()

# 산점도 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First characteristic")      # 첫 번째 특성
plt.ylabel("Second characteristic")     # 두 번째 특성
print("X.shape: {}".format(X.shape))
plt.show()