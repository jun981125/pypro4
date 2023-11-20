# 비선형 회귀 분석 ( non linear regression ) -  다항 회귀 분석
# 데이터가 곡선의 형태로 분포되어 있는 경우에 직선의 회귀식을 ( 잔차가 큰 회귀식 )
# 곡선으로 변환해 ( 잔차를 줄임 ) 보다 더 정확하게 데이터 변화를 예측하는 것이 목적이다
# 입력 자료 특징/특성 (독립변수, feature) 변환으로 선형 모델 개선
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

# plt.scatter(x,y)
# plt.show()
# print(np.corrcoef(x,y)[0,1]) 0.4807
# 데이터 분포가 곡선이기 때문에 필요없음

# 선형 회귀 모델을 작성
from sklearn.linear_model import LinearRegression

x = x[:, np.newaxis]	# 	차원 확대
# print(x)
model = LinearRegression().fit(x,y)
y_pred = model.predict(x)
print('y_pred : ' , y_pred)

# plt.scatter(x,y)
# plt.plot(x, y_pred, c='r')
# 잔차가 최소화 된 선 추가
# plt.show()
# from sklearn.metrics import r2_score
# print(f'결정계수 : {r2_score(y,y_pred)}')
# 결정계수 : 0.23113207547169834

# feature에 항 ( 다항식 특징 ) 을 추가
# PolynomialFeatures : 다항 회귀 ( 비선형 )
# 곡선형 그래프가 나타날 때 , non linear하기 때문에 각 특성의 제곱을
# 추가해주어서 특성이 추가된 비선형 데이터를 선형 회귀 모델로 훈련 시키는 방법

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=6, include_bias=True)
# 현재 데이터를 다항식 형태로 변경 ( 각 특성의 제곱 이나 그 이상을 추가 )
# degree : 차수 조절 / include_bias : True면 0차항도 만듬 (  1 + x + x^2 일때 1을 의미)

x2 = poly.fit_transform(x)
# # 새롭게 정의된 numpy 배열은 행별로 각 데이터를 다항 형태로 변형해준다.
# print(poly.get_params(),'\n', x2)

model2 = LinearRegression().fit(x2,y)
ypred2 = model2.predict(x2)

plt.scatter(x,y)
plt.plot(x, ypred2, c='black')
plt.show()

from sklearn.metrics import r2_score
print(f'결정계수 : {r2_score(y,ypred2)}')

