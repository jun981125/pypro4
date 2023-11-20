# Logistic Linear Regression
# 선형 회귀 분석 처럼 신뢰 구간 , p 값 등이 제공 되나 회귀 계수의 결과를
# 해석 하는 방법이 선형 회귀 분석과 다르다
# 독립 변수 : 연속형 / 종속 변수 : 범주형
# 이항 분포를 따르며 출력 값은 0 ~ 1 사이의 확률로 제공
# 연속형 결과를 로짓(오즈비에 로그를 취함) 변환 후 시그모이드 함수를 통해 결과를 내 보낸다.

import math
# sigmoid funtion
def sigmoidFunc(x):
	return 1/(1 + math.exp(-x))

# print(sigmoidFunc(3))

print('mtcars dataset')
import statsmodels.api as sm

mtcarData = sm.datasets.get_rdataset('mtcars')
# print(mtcarData.keys())
mtcars = sm.datasets.get_rdataset('mtcars').data
# print(mtcars)
mtcar = mtcars.loc[:,['mpg','hp','am']]


# 연비와 마력수는 변속기에 영향을 주는가?
# 모델 작성 방법 1 : logit()
formula = 'am  ~ hp+mpg'
import statsmodels.formula.api as smf
model1 = smf.logit(formula='am  ~ hp+mpg', data=mtcar).fit()
print(model1)
print(model1.summary())

pred = model1.predict(mtcar[:10])
import numpy as np
print('예측값 : ', pred.values)
print('예측값 : ',np.around(pred.values))
print('실제값 : ', mtcar['am'][:10].values)

conf_tab = model1.pred_table()
print('confustion matrix : \n ', conf_tab)

print('분류 정확도' , (16 + 10) / len(mtcar))
	  # qnsfb wjdghkreh 0.8125 / 91.25%

from sklearn.metrics import accuracy_score
pred2 = model1.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))

# 모델 작성 방법 2 : glm()
model2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()

print(model2)
print(model2.summary())
glmPred = model2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glmPred.values))
print('glm 실제값 : ', mtcar['am'][:10].values)
glmPred2 = model2.predict(mtcar)
print('glm 분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glmPred2)))

print('새로운 값 으로 분류 예측')
newdf = mtcar.iloc[:2].copy()
# print(newdf)
newdf['mpg'] = [10, 30]
newdf['hp'] = [100,130]
# print(newdf)
new_pred = model2.predict(newdf)
print('new_pred : ', np.around(new_pred.values))
print('new_pred : ', np.rint(new_pred.values))

print()
import pandas as pd
new_pred2 = pd.DataFrame({'mpg':[10,35,50,5], 'hp':[80,100,125,50]})
print('new_pred2 : ', np.around(new_pred2.values))

# 머신러닝의 포용성 ( inclusion, tolerance )
# 생성 모델은 최적화와 일반화를 잘 융합
# 분류 정확도가 100%인 경우는 과적합(overfitting)모델이므로 새로운 데이터 에 대해 정확한
# 분류를 할 수 없는 경우가 있다. ( 꼬리없는 동물 )

