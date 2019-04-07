from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split
# ordinary least squares 
# 가장 간단하고 오래된 회귀용 선형 알고리즘 
# 예측과 훈련 세트에 있는 타깃 y사이의 평균제곱오차를 최소화하는 파라미터 w와b를 찾음
# 평균제곱오차는 예측값과 타깃값의 차이를 제곱하여 더한 후에 샘플의 개수로 나눈 것이다.
# 선형 회귀는 매개변수가 없는 것이 장점이지만, 그래서 모델의 복잡도를 제어할 방법도 없다.




X,y =mglearn.datasets.make_wave(n_samples=60)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# 훈련 세트와 테스트 세트로 분리

lr = LinearRegression().fit(X_train, y_train)
# 모델 객체 생성 & 학습
# 기울기 파라미터(w)는 가중치 또는 계수라고 하며 lr객체의 coef_ 속성에
# 저장 되어 있고 편향 (offset) 또는 절편 ( intercept ) 파라미터 ( b ) 는 intercept_속성에 저장되어 있다.
# coef_와 intercept_ 뒤에 밑줄이 이상하게 보일 수도 있다.
# scikit_learn은 훈련 데이터에서 유도된 속성은 항상 끝에 밑줄을 붙입니다.
# 그 이유는 사용자가 지정한 매개변수와 구분하기 위해서입니다.

print("lr.coef_:{}".format(lr.coef_))
print("lr.intercept_ : {}".format(lr.intercept_))
# intercept_ 속성은 항상 실수 (float) 값 하나지만 , coef_속성은 각 입력 특성에 하나씩 대응되는
# NumPy 배열이다.
# wave 데이터셋에는 입력 특성이 하나뿐이므로 lr.coef_도 원소를 하나만 가지고 있다.


# 모델 평가 
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))

# R의 제곱 값이 0.66인 것은 그리 좋은 결과는 아니다 . 하지만 훈련 세트와 테스트 세트의 점수가
# 매우 비슷한 것을 알 수 있다.
# 이는 과대적합이 아니라 과소적합인 상태를 의미한다.
# 1차원 데이터셋에서는 모델이 매우 단순하므로 과대적합을 걱정할 필요가 없다.
# 그러나 고차원 ( 특성이 많은 ) 데이터셋에서는 선형 모델의 성능이 매우 높아져서
# 과대적합될 가능성이 높다 .
