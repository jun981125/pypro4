"""
*** 선형회귀분석의 기존 가정 충족 조건 ***
. 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
. 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
. 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
. 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
. 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.
"""

import pandas as pd
import matplotlib.pyplot as plt
from django.conf.locale import pl
plt.rc('font', family='AppleGothic')
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np

advdf = pd.read_csv('../testdata/Advertising.csv', usecols=[1,2,3,4])
# print(advdf.info())

print('r : \n', advdf.loc[:,['sales','tv']].corr())

# 인과관계가 있다는 가정하에 회귀분석 모델 작성

lm = smf.ols(formula= 'sales ~ tv', data=advdf).fit()
# print(lm.summary())
# Prob (F-statistic): 1.47e-42  -> 귀무가설 기각 실패 . 유의한 모델
print(lm.pvalues[1])
print(lm.rsquared)
print(lm.params)

# 시각화
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('tv')
plt.ylabel('sales')
y_pred = lm.predict(advdf.tv)
# plt.plot(advdf.tv, y_pred, c='r')
# plt.show()

# 모델 검정
pred = lm.predict(advdf[:10])
print('실제값 : ', advdf.sales[:10])
print('예측값 : ', pred[:10].values)

# 예측 1 : 새로운 TV 광고비 값으로 판매액 예측
x_new = pd.DataFrame({'tv':[200.0, 40.5, 100.0]})
new_pred = lm.predict(x_new)
print('sales 추정값 : ', new_pred.values)

# 다중 선형 회귀 모델
lm_mul = smf.ols(formula= 'sales ~ tv+radio', data=advdf).fit()
# print(lm_mul.summary())

# 예측 2 : 새로운 TV, radio 광고비로 판매액 예측
x_new2 = pd.DataFrame({'tv':[200.0, 40.5, 100.0], 'radio': [37.8 , 45.3, 55.0]})
new_pred2 = lm.predict(x_new2)
print('sales 추정값 : ', new_pred2.values)
print()

print('잔차항 구하기')
fitted = lm_mul.predict(advdf.iloc[:, 0:2])
# print(fitted)
residual = advdf['sales'] - fitted # 잔차
# print(sum(resdiual))  2.020605904817785e-12

# 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
# 예측값과 잔차가 비슷하게 유지

sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})

# lowess = True : 비모수적 : 최적 모델 추정 (로컬 가중 선형 회귀 모델로 )
# plt.plot([fitted.min(), fitted.max()], [0,0] , '--', color='blue')
# plt.show() # 예측값과 잔차가 곡선을 그림 - 선형성을 만족하지 못함 다항회귀(PolynomialFeatures) 분석모델 추천

# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
import scipy.stats

ssz = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(ssz)
sns.scatterplot(x=x, y=y)
plt.plot([-3,3], [-3,3] , '--', color='red')
plt.show()	# 커브를 그리면서 추세선 밖으로 나가고 있는 형태
# 정규성을 만족한다고 볼 수 없음
print('정규성 : ' , scipy.stats.shapiro(residual))
# pvalue=4.190356062139244e-09 < 0.05 -> 정규성 만족 X
# log를 취하는 방법 등을 사용해 데이터 가공 필요

# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다. 잔차가 자기상관(인접 관측치와 독립이어야 함 )이 있는지 확인 필요
# 자기상관은 Durbin-Watson 지수 d를 이용하여 검정한다.
# d 값은 0~4사이에 나오며, 2에 가까울 수록 자기상관이 없이 독립이며, 독립인 경우 회귀분석을 사용할 수 있다.
# DW 값이 0 또는 4에 근사하면 잔차들이 자기 상관이 있고, 계수 ( t, f, R^2 )
# 계수 값을 증가시켜 유의하지 않은 결과를 유의한 결과로 왜곡시킬 수 있다.
print('Durbin-Watson : ', 2.081)



# . 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# 분산은 모든 잔차에 대해 동일해야 한다. 잔차 (y축) 및 예상 값 (x축) 의 산점도를 사용하여 이 가정을 테스트 할 수 있다.
# 결과 산점도는 플롯에서 임의로 플롯된 점의 수평 밴드로 나타나야 한다.
sns.regplot(x=fitted, y=np.sqrt(np.abs(ssz)), lowess=True, line_kws= {'color':'red'})
plt.show() 			# 적색 실선이 수평선을 그리지 않으므로 등분산성 만족 못함.

"""
 . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.
Minitab 회귀 분석을 하게 되면 다음과 같은 VIF 값을 제공합니다. 
결론부터 말씀 드리면 VIF 값이 1 근방에 있으면 다중공선성이 없어 모형을 신뢰할 수 있으며
만약 VIF 값이 10 이상이 되면 매우 높은 다중공선성이 있기 때문에 변수 선택을 신중히 고려해야 합니다.

하나의 독립 변수가 다른 독립 변수들과 상관성이 높아서
회귀 모델에서 이들 변수의 효과를 구별하기 어려워지는 상황
1. 데이터 스케일 차이 : 독립 변수 간 스케일이 차이가 나면 발생할 가능성 증가
2. 높은 상관 관계 : 독립 변수들 간에 높은 상관 관계가 있을 때 가능성 증가
3. 샘플 크기 부족 : 적은 수의 관측치로 다수의 독립 변수를 다루는 경우,
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(advdf.values,1))	 # tv, OLS Regression Result 에서 intercept는 0
# 12.570312383503682
print(variance_inflation_factor(advdf.values,2))	# Radio 3.1534983754953836
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(advdf.values, i) for i in range(1,3)]
print(vifdf)

#    vif_value
# 0  12.570312		tv : 다중 공선성이 있다. 독립변수에서 제거해야 하나, 영향력이 큰 변수라면 고민이 필요하다.
# 1   3.153498		radio


# 독립변수가 더 많은 경우, 예를 들어 남편의 수입과 아내의 수입이 서로 상관성이 높다면,
# 두 개의 변수를 더해 가족 수입이라는 새로운 변수를 선언하거나 또는 주 성분 분석을 이용하여
# 하나의 변수로 만들어 작업할 수 있다.


print("참고 : Cook's distance - 극단 값을 나타 내는 지표 이해")
from statsmodels.stats.outliers_influence import  OLSInfluence
cd, _ = OLSInfluence(lm_mul).cooks_distance	# 극단값을 나타내는 지표로 반환
print(cd.sort_values(ascending=False).head())

statsmodels.api.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()	# 원의 크기가 특별히 큰 데이터는 이상 값 ( outlier ) 이라 볼 수 있다.





# statsmodels.formula.api 모듈의 ols 함수는 선형 회귀 모델을 만들기 위해 사용되는 함수
# ols는 Ordinary Least Squares(최소자승법)를 의미
# 회귀 모델의 파라미터를 추정 / 통계 모델을 쉽게 정의하고 추정
# ols 함수를 사용하여 선형 회귀 모델 생성
# model = smf.ols(formula='종속변수 ~ 독립변수1 + 독립변수2 + ...', data=df)
# data는 무조건 DataFrame형식

# 모델 피팅 (추정)
# result = model.fit()
# 회귀 모델 요약
# print(result.summary())