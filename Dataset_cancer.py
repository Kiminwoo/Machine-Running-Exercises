# * cancer 데이터셋 : 위스콘신 유방암 데이터셋. 유방암 종양의 임상 데이터가 기록된 실제 데이터셋.
# 569개의 데이터와 30개의 특성을 가진다. 그중 212개는 악성이고 357개는 양성이다.
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
print("데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))