"""
방법 4 :  linregress 사용 모델 생성됨
두 변수 간의 선형 관계를 분석하는 데 사용

IQ에 따른 시험 점수 값 예측

"""

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

score_iq = pd.read_csv("../testdata/score_iq.csv")
# print(score_iq.head(3), score_iq.shape)			# (150, 6)
x = score_iq.iq
y = score_iq.score
# print(x,y)

# 상관 계수 확인
#print(np.corrcoef(x,y))
#print(score_iq.corr())
# plt.scatter(x,y)
# plt.show()

# 인과 관계가 있다는 가정하에 선형회귀분석 모델 생성
model = stats.linregress(x,y)
#print(model)
print(f'x 기울기 : {model.slope}')	# 가중치
print(f'y 절편 : {model.intercept}')	# 바이어스
print(f'상관계수 : {model.rvalue}')
print(f'p_value : {model.pvalue}')
print(f'기울기의 표준 오차 : {model.stderr}')


plt.scatter(x,y)
plt.plot(x, model.slope * x + model.intercept, c='r')
plt.show()
# 회귀모델 수식 : y = model.slope * x + model.intercept
print('점수 예측 : ', model.slope * 80 + model.intercept)
print('점수 예측 : ', model.slope * 120 + model.intercept)
print('점수 예측 : ', model.slope * 140 + model.intercept)
print('점수 예측 : ', model.slope * 150 + model.intercept)

# predict() 지원 X : numpy의 ployval([기울기, 절편], x)을 사용
print(f'실제 점수 : {score_iq.score[:5].values}')
print(f'점수 예측 : {np.polyval([model.slope, model.intercept], np.array(score_iq["iq"][:5]))}')

new_df = pd.DataFrame({'iq': [83, 90, 100, 115, 125, 130, 140]})
# print(new_df)
print('점수 예측', np.polyval([model.slope, model.intercept], new_df))