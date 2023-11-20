"""
선형회귀

1. 최소 제곱법
	단순 선형 회귀
		단일 독립 변수와 종속 변수 간의 선형 관계를 나타낸다.
		from sklearn.linear_model import LinearRegression
		model = LinearRegression()

	다중 선형 회귀
		여러 개의 독립 변수와 종속 변수 간의 선형 관계를 나타낸다.
		from sklearn.linear_model import LinearRegression
		model = LinearRegression()

- 단순 선형 회귀와 다중 선형 회귀는 입력 데이터로 구분됨

	다항 선형 회귀
		독립 변수와 종속 변수 간의 비선형 관계를 나타낸다.
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression
		from sklearn.pipeline import make_pipeline

		degree = 2  # 다항식 차수
		model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

	릿지 회귀, 라쏘 회귀
		회귀 계수의 크기에 대한 제약을 가한 선형 회귀 방법이다.
		과적합을 방지하고 일반화 성능을 향상시키는 데 사용된다.
		from sklearn.linear_model import Ridge, Lasso
		model = Ridge(alpha=1.0)  # alpha는 규제 강도를 조절하는 매개변수
		model = Lasso(alpha=1.0)



분류에서는 설명력이 아닌, 정확도로 표현

로지스틱 회귀	wx+b/ 이항 분류를 함
	분류 문제에 사용되며, 종속 변수가 범주형인 경우에 적용됩니다.
	최소 제곱법은 아니지만 선형 모델입니다.

odds (오즈 또는 승산 ) - 어떤 사건이 발생할 확률과 발생하지 않을 확률 간의 비율
	p/1-p = 이벤트 발생확률 / 이벤트가 발생하지 않을 확률
	이벤트가 발생할 확률은 이벤트가 발생하지 않을 확률의 오즈 값 배이다.
	두 변수가 독립 변수 라면, 오즈비는 1

odds ratio (오즈비, 승산비) - 오즈비는 두 개의 다른 그룹 간의 오즈를 비교하여 두 그룹 간의 상대적인 이벤트 발생 가능성
ex) 코로나 환자와 만나고 감염될 확률 / 코로나 환자와 만나지않고 감염될 확률
= ( 코로나 환자와 만나고 감염될 확률 / 코로나환자와 만나고 감염되지 않을 확률 ) /
	(코로나 환자와 만나지 않고 감염될 확률 / 코로나 환자와 만나지 않고 감염되지 않을 확률 )
	오즈비가 1보다 크면, 코로나 환자와 만남으로 인해 감염될 확률이 더 높다
    반대로, 1보다 작으면 코로나 환자와 만나지 않고도 감염될 확률이 더 높다

							0		   0.5		1
	오즈 p/1-p				0		    1  	+infinity
	로그 오즈비-log(p/1-p)		-infinity   0 	 +infinity

3항 이상의 회귀 / 다항 회귀
Softmax Regression
로지스틱 회귀는 하나의 입력을 Yes or No 로 분류하는데 반해
소프트 맥스 회귀는 각 입력 데이터에 따라 소수 확률을 부여 한다.
 이때 부여한 소수들의 합은 1이다.

"""