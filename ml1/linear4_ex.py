"""
회귀분석 문제 2)
testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다.
이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
  - 국어 점수를 입력하면 수학 점수 예측
  - 국어, 영어 점수를 입력하면 수학 점수 예측
"""
import pandas as pd
from scipy import stats
import numpy as np
import statsmodels.formula.api as smf

data = pd.read_csv("../testdata/student.csv")

model = smf.ols('수학~ 국어', data=data)
model.fit()

# 국어 = int(input('국어점수 : '))
국어 = 55
print('예상 수학 점수 : ', model.predict(data.국어))








"""

model = smf.ols(formula= '수학 ~ 국어', data=data)
result = model.fit()

국어 = int(input('국어점수 : '))
newdf = pd.DataFrame({'국어': [국어]})
new_pred = result.predict(newdf)
print(f'국어 {newdf}에 대한 수학 점수 예측 결과 : \n{new_pred[0]}')



model2 = smf.ols(formula= '수학 ~ 국어+영어', data=data)
result2 = model2.fit()

국어 = int(input('국어점수 : '))
영어 = int(input('영어점수 : '))
newdf2 = pd.DataFrame({'국어': [국어], "영어" : [영어]})
new_pred2 = result.predict(newdf2)
print(f'국어 {newdf2["국어"][0]} 영어 {newdf2["영어"][0]}에 대한 수학 점수 예측 결과 : {new_pred[0]}')
"""
