from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

iris_dataset = load_iris()

# 데이터셋에 대한 설명 , iris_dataset.DESCR[:193] 도 가능하다 .

print("iris_dataset의 키 : {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "|n...")

# 붓꽃 품종의 이름

print("타깃의 이름 : {}".format(iris_dataset['target_names']))

# 특성 설명
print("특성의 이름: {}".format(iris_dataset['feature_names']))

# data의 타입
print("data의 타입: {}".format(type(iris_dataset['data'])))

# data의 크기
print("data의 크기: {}".format(iris_dataset['data'].shape))


# data의 처음 다섯 행 :
print("data의 처음 다섯 행 : \n{}".format(iris_dataset['data'][:5]))

# target의 타입
print("target의 타입: {}".format(type(iris_dataset['target'])))

# target의 크기
print("target의 크기: {}".format(iris_dataset['target'].shape))

# target의 데이터
print("타깃: \n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=0
                                                    )
# X , Y train 크기
print("X_train 크기 : {}".format(X_train.shape))
print("y_train 크기 : {}".format(y_train.shape))

# X,Y test 크기
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# X_train 데이터를 사용해서 데이터프레임을 만든다
# 열의 이름은 iris_dataset.feature_name에 있는 문자열을 사용한다
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# 데이터프레임을 사용해서 y_train에 따라 색으로 구분된 산점도 행렬을 만든다
pd.plotting.scatter_matrix(iris_dataframe, c= y_train, figsize=(15,15), marker ='0', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()
