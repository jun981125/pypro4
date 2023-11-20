"""
회귀분석 문제 3)
kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음)
 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
회귀분석모형의 적절성을 위한 조건도 체크하시오.
완성된 모델로 Sales를 예측.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.conf.locale import pl
plt.rc('font', family='AppleGothic')
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np
import scipy.stats

data = pd.read_csv("../testdata/Carseats.csv").drop(['ShelveLoc', 'Urban', 'US'], axis=1)
# print(data.head(3), data.info())
print()
corr_data = data.corr()['Sales']
corr_sales_sort = corr_data.abs().sort_values(ascending=False)
print(corr_sales_sort)

# Price          0.444951
# Advertising    0.269507
# Age            0.231815
# Income         0.151951

# -CompPrice      0.064079
# -Education      0.051955
# -Population     0.050471

# data['CompPrice'] = (1/data['CompPrice'])
lm = smf.ols(formula= 'Sales ~ Price+Advertising+Age+Income', data=data).fit()
print(lm.summary())
# 유의한 모델임을 낱냄

# Education 0.282	/   Population 0.857  제외



# 잔차항
fitted = lm.predict(data)
# print(fitted)
residual = data['Sales'] - fitted
print('잔차의 평균 : ',np.mean(residual))
# -1.0769163338864018e-14
# 선형성
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.show()
# 선형성 X  -> 다항 회귀 분석


# 정규성
ssz = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(ssz)
sns.scatterplot(x=x, y=y)
plt.plot([-3,3], [-3,3] , '--', color='red')
plt.show()
print('정규성 검사:', scipy.stats.shapiro(residual))
# 0.9949221611022949, 정규성 만족

# 독립성
from statsmodels.stats.stattools import durbin_watson
dw_result = durbin_watson(residual)
print("Durbin-Watson :", dw_result)
# 1.931498127082959  2에 매우 근사하므로 독립


# 등분산성
sr = scipy.stats.zscore(residual)
sns.regplot(x=fitted,y= np.sqrt(np.abs(sr)), lowess = True, line_kws = {'color':'red'})
plt.show()

# 다중공선성

from statsmodels.stats.outliers_influence import variance_inflation_factor

col= data[['Income','Advertising', 'Price' ,'Age']]
vif_df = pd.DataFrame()
vif_df['Variable'] = col.columns
vif_df['vif_value'] = [variance_inflation_factor(data.values, i+2) for i in range(col.shape[1])]
print(vif_df)


new_df = pd.DataFrame({'Price':[105,89,75],'Income':[35,62,24], 'Advertising':[6,3,11],'Age':[35,42,21]})
pred = lm.predict(new_df)
print(f'예측 값 : {pred}')

# 모델 검증이 끝난 경우 모델을 저장
# 방법 1
import pickle
with open('linear6m2.model', 'wb') as obj:
	pickle.dump(lm, obj)

with open('linear6m2.model', 'rb') as obj:
	pickle.load(obj)

# # 방법2
#
import joblib
joblib.dump(lm,'linear.model')

mymodel = joblib.load('linear.model')