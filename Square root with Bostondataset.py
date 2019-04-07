from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train,y_train)

print("train set score: {:.2f}".format(lr.score(X_train, y_train)))
print("test set score: {:.2f}".format(lr.score(X_test, y_test)))

# 훈련 세트와 테스트 세트의 점수를 비교해보면 훈련 세트에서는 예측이
# 매우 정확한 반면 테스트 세트에서는 R2 값이 매우 낮다 .

# 훈련 데이터와 테스트 데이터 사이의 이런 성능 차이는 모델이 과대적합
# 되었다는 확실한 신호이므로 복잡도를 제어할 수 있는 모델을 사용해야 한다.
