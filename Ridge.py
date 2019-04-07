# 릿지 회귀
# 릿지 ( Ridge ) 도 회귀를 위한 선형 모델이므로 , 최소적합법에서 사용한 것과 같은 예측 함수를 사용한다.
# 하지만, 릿지 회귀에서의 가중치(w) 선택은 훈련 데이터를 잘 예측하기 위해서 뿐만 아니라 추가 제약 조건을
# 만족시키기 위한 목적도 있다.
# 가중치의 절대값을 가능한 한 작게 만드는 것이다.
# 다시 말해서 , w의 모든 원소가 0에 가깝게 되길 원한다.
# 생각을 해보면 , 이는 모든 특성이 출력에 주는 영행을 최소한으로 만든다
# 기울기를 작게 만든다 .
# 이런 제약을 규체 ( regularization ) 라고 한다.
# 규제란 과대적합이 되지 않도록 모델을 강제로 제한한다는 의미이다.
# 릿지 회귀에 사용하는 규제 방식을 L2 규제라고 한다 .
# 릿지 회귀는 linear_model.Ridge에 구현되어 있다. 릿지 회귀가 확장된 보스턴 주택가격
# 데이터셋에 어떻게 적용되는지 살펴 보겠다.

from sklearn.linear_model import Ridge
import mglearn
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train, y_train)
print("training set score (alpha=1): {:.2f}".format(ridge.score(X_train, y_train)))
print("test set score (alpha=1): {:.2f}".format(ridge.score(X_test, y_test)))

# 결과를 보면 훈련 세트에서의 점수는 LinearRegression 보다 낮지만 , 테스트 점수에 대한 점수는 더 높다.
# 선형 회귀는 이 데이터셋에 과대적합되지만 , Ridge 는 덜 자유로운 모델이기 때문에 과대적합이 적어진다.
# 모델의 복잡도가 낮아지면 훈련 세트에서의 성능은 나빠지지만 더 일반화된 모델이 된다 .
# 테스트 세트에 대한 성능이기 때문에 , LinearRegression 보다 Ridge 모델을 선택해야 한다 .

# Ridge는 모델을 단순하게 ( 계수를 0에 가깝게 ) 해주고 훈련 세트에 대한 성능 사이를 절충 할 수 있는 방법을 제공한다.
# 사용자는 alpha 매개변수로 훈련 세트의 성능 대비 모델을 얼마나 단순화할지를 지정할 수 있다.
# 이 alpha의 값이 최적이라고 단정지을 수가 없다 .
# 최적의 alpha 값은 사용하는 데이터셋에 달려있다.
# alpha 값을 높이면 계수를 0에 더 가깝게 만들어서 훈련 세트의 성능은 나빠지지만 , 일반화에는 도움을 줄 수 있다.

# 예를 들면 다음과 같은 것이다.

ridge10 = Ridge(alpha=10).fit(X_train,y_train)
print("training set score (alpha=10): {:.2f}".format(ridge10.score(X_train, y_train)))
print("test set score (alpha=10): {:.2f}".format(ridge10.score(X_test,y_test)))

# alpha 값을 줄이면 계수에 대한 제약이 그만큼 풀리면서 , 오른쪽으로 이동하게 된다.
# 아주 작은 alpha 값은 계수를 거의 제한하지 않으므로 , LinearRegression으로 만든 모델과 거의 같아진다.

ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("training set score (alpha=0.1): {:.2f}".format(ridge01.score(X_train, y_train)))
print("test set score (alpha=0.1): {:.2f}".format(ridge01.score(X_test,y_test)))

# 위 코드 alpha = 0.1 일 때를 보면 , 꽤 좋은 성능을 낸거라고 말할수 있다.
# 테스트 세트에 대한 성능이 높아 질 때까지 alpha값을 줄일 수 있을 것이다.
# 또한 alpha 값에 따라 모델의 coef_ 속성이 어떻게 달라지는지를 알아보자면 ,
# alpha 매개변수가 모델을 어떻게 변경시키는지 이해할 수 있다.
# 높은 alpha 값은 제약이 더 많은 모델이므로 작은 alpha 값일 때보다
# coef_의 절댓값 크기가 작을 것이라고 예상할 수 있다.


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train,y_train)

plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Count List")
plt.ylabel("Coefficient size")
plt.hlines(0,0, len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()

# 위 코드를 실행시켜 보면 , x축은 coef_의 원소를 위치대로 나열한 것입니다.
# 즉, x=0은 첫 번째 특성에 연관된 계수이며 , x=1은 두 번째 특성에 연관된 계수이다.
# 이런 식으로 x=100까지 계속 된다.
# y축은 각 계수의 수치를 나타낸다.
# alpha=10일 때 , 대부분의 계수는 -3과 3사이에 위치한다.

# alpha=1일 때 , Ridge 모델의 계수는 좀더 커진다.
# alpha=0.1일 때 , 계수는 더 커지며 아무런 규제가 없는
# alpha=0 일 때는 선형 회귀의 계수는 값이 더 커져서 그림 밖으로 넘어가게 된다 .

