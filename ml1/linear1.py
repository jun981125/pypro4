# 단순 선형 회귀 분석 모델 생성
# 독립변수 ( 연속형 ) , 종속변수 ( 연속형 )
# 두 변수는 상관관계가 있어야하고, 나아가서는 인과관계가 있다는 가정이 필수

import statsmodels.api as sm
import numpy as np
from sklearn.datasets import make_regression

np.random.seed(12)

# 모델 생성 맛보기
# 방법 1 : make_regression, 모델 생성은 안됨

# make_regression : 주어진 매개변수를 기반으로 가상의 회귀 문제 데이터를 생성
# 데이터는 실제로는 어떤 모델에서 생성된 것이 아니라 무작위로 만들어진 것이기 때문에 실제 데이터와 다르다.
# 회귀 분석 알고리즘을 테스트하고 디버깅하는 용도로 사용됨

"""
1. 독립변수 데이터 행렬 X를 무작위로 만든다. -> x 값에 대한 y값 제시
2. 종속변수와 독립변수를 연결하는 가중치 벡터 w를 무작위로 만든다.
3. X와 w를 내적하고 y절편 b 값을 더하여 독립변수와 완전선형인 종속변수 벡터 y_0를 만든다.
4. 기댓값이 0이고 표준편차가 noise인 정규분포를 이용하여 잡음 epsilon를 만든다.
5. 독립변수와 완전선형인 종속변수 벡터 y_0에 잡음 epsilon을 더해서 종속변수 데이터 𝑦를 만든다.
최종적으로 수식이 완성된다 y = wx + b -> 예측값 y =  89.47430739278907 * x + 100
"""
x, y, coef = make_regression(n_samples=50,n_features=1, bias=100, coef=True)
# n_samples 표본수 	n_features 독립변수의 수  coef : True이면 선형 모형의 계수 출력
# coef : weight ( 가중치 )
print(f'x : {x},\ny : {y},\ncoef : {coef}')
pred_y = 89.47430739278907 * -1.70073563 + 100		# 작성된 모델로 x에 대한 예측값 y를 출력
print('y의 실제 값은 : -52.17214291')
print(f'x값 : -1.70073563에 대한 예측값 y는 {pred_y}')
# y의 예측 값과 y의 실제 값은 신뢰도에 의거해 값이 구해짐
# 100퍼센트 정확하진 않음
# 100 퍼센트인 경우 융통성이 없어서, 독립변수가 조금만 어긋나도 엄청난 오류를 가져올 수 있음.

new_pred_y = 89.47430739278907 * 1234.5678 + 100
print(f'궁금한 데이터 (1234.5678) 입력의 예측 값은 : {new_pred_y}')

xx = x
yy = y
# 방법 2 : linear_regression, 모델 생성은 안됨 regression : 회귀
# 입력 변수와 출력 변수 간의 선형 관계를 모델링 / 선형 함수를 찾는 것이 주 목적
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# fit_intercept 인수는 모형에 상수항이 있는가 없는가를 결정하는 인수.
# 디폴트 값이 True는 상수항이 있음으로 지정.

fit_model = model.fit(xx,yy) # fit 함수의 인덱스는 입력 변수와 출력 변수를 의미
# fit 메서드로 가중치 값을 추정한다. 상수항 결합을 자동으로 해줌
# fit 함수는 선형 회귀 모델을 학습시킴.
# 학습이란 y = wx + b 모델을 예로 들었을 때 w와 b값을 최적화된 값으로 도출되도록 학습을 통해
# 찾아주는 행위 / 모델은 데이터들의 어떠한 규칙이나, 패턴을 의미

print(f'slope : {fit_model.coef_}')	# [89.47430739]
print(f'bias : {fit_model.intercept_}')	# 100.0
# 예측 값 확인
y_new = fit_model.predict(xx[[0]])
print(f'y_new : {y_new}')
# y의 실제 값은 : -52.17214291
# 방법 1의 예측값 : -52.17214255248879
# 방법 2의 예측값 : -52.17214291

y_new2 = fit_model.predict(xx[[12]])
# 만들어지는 행렬은 2차원 행렬이기 때문에 [[]] 에 인덱스를 입력
print(f'y_new2 : {y_new2}')

# 방법 3 : ols 사용, 모델 생성 됨
import statsmodels.formula.api as smf
import pandas as pd
# print(xx.shape)		# (50, 1)
x1 = xx.flatten()	# 차원 축소
# print(x1.shape)		# (50, )
y1 = yy
# print(y1.shape)		# (50,)

data = np.array([x1,y1])
df = pd.DataFrame(data.T)
df.columns = ['x1', 'y1']
print(df.head(3), len(df))
model2 = smf.ols(formula='y1 ~ x1', data = df).fit()
print(model2.summary())		# OLS Regression Result

#                    coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept(절편)    100.0000   5.54e-15    1.8e+16      0.000     100.000     100.000
# x1                89.4743    4.9e-15     1.83e+16     0.000     89.474      89.474

# P > |t| : t-통계량에 대한 p-value를 나타냄
# 회귀 계수에 대한 가설 검정에서 해당 계수가 0인지 여부를 확인
# Hø : 회귀 계수는 0이다.
# Hø : 회귀 계수는 0이 아니다.

# p-value  < 0.05
# 귀무가설을 기각 실패라면, 회귀 계수는 통계적으로 유의하며, 독립변수가 종속변수에 영향을 미친다라는 결론이 도출된다.
# 독립변수와 종속변수와의 인과관계를 나타냄 ----
# p-value  > 0.05
# 귀무가설 기각이라면 독립변수와 종속변수의 인과관계가 없음을 나타냄

#  Prob (F-statistic):               0.00 의경우는 변수가 아닌 모델의 인과관계를 나타냄
print(x1[:2])
new_df = pd.DataFrame({'x1':[-1.700736, -0.677945]})
print(new_df)
new_pred = model2.predict(new_df)
print('예측값 new_pred : \n',new_pred)
print('실제값 : \n' , df.y1[:2])

new2_df = pd.DataFrame({'x1':[111, -6.12345]})
new2_pred = model2.predict(new2_df)
print('예측값 new_pred : \n',new2_pred)


"""
선형 회귀 모델에서 종속 변수와 독립 변수간의 선형 관계는 필수적인 요소,
선형 회귀 모델을 만들기 전에 선형 관계를 증명한 후 모델을 만들어야함

선형성 여부 판별 방법
1. 시각적인 방법
	산점도 ( scatter plot ) 
	회귀 직선 ( 곡선인지 직선인지 파악 )
	
2. 통계적인 방법
	잔차 분석
		모델을 통해 예측한 값과 실제 값 간의 잔차 ( 오차 / 차이 ) 를 확인하여 모델이
		데이터를 얼마나 잘 설명하고 있는지 평가한다. 잔차가 무작위로 분포하고 패턴이 없다면
		모델이 데이터를 잘 설명하고 있음을 의미한다. 잔차가 무작위고 패턴이 없다는 것은 
		현재 도출된 잔차가 우연에 의한 것임을 의미하기 때문에 잘 설계된 모델이라 할 수 있다.
	상관 계수
		종속 변수와 독립 변수 간의 선형 관계를 나타내는 피어슨 (Pearson) 상관 계수를 계산하여
		상관 계수가 1이나 -1 즉 |1|에 근사하면 선형 관계가 강한 것이고, 0에 근사하면 선형 관계가
		약함을 의미한다.
"""
