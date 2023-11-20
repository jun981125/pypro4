# 단순 선형 회귀 : iris dataset , ols()
# 상관관계가 약한 경우와 강한 경우를 나눠 분석 모델을 작성 후 비교

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
# print(iris.head(3))
# print(iris.iloc[:,0:4].corr())

# 상관관계가 약한 경우 (correlation)
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
# print('result1 모델 정보 : ', result1.summary())
print('result1 R_squared : ', result1.rsquared)
print('result1 p-value : ', result1.pvalues[1])		# P-value > 0.05 이므로 모델은 유의하지 않다

# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color='r')		# 회귀선
# plt.show()


# 상관관계가 강한 경우 (correlation) sepla_length, sepal_width :  -0.117570
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
# print('result2 모델 정보 : ', result2.summary())
print('result2 R_squared : ', result2.rsquared)
print('result2 p-value : ', result2.pvalues[1])		# P-value < 0.05 이므로 유의한 모델임

# plt.scatter(iris.petal_length, iris.sepal_length)
# plt.plot(iris.petal_length, result2.predict(), color='r')		# 회귀선
# plt.show()

print('실제 값 : ', iris.sepal_length[:10].values)
print('예측 값 : ', result2.predict()[:10])

# 새로운 petal_length로 sepal_length를 예측 가능
new_data = pd.DataFrame({'petal_length':[1.1, 0.5 , 5.0]})
y_pred = result2.predict(new_data)
print(f'예측 결과 : {y_pred.values}')

print('다중 선형 회귀 : 독립변수가 복수')
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width+sepal_width', data=iris).fit()
column_select = "+".join(iris.columns.difference(['sepal_length', 'species']))
print(column_select)
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
# R 에서는 변수에 .을 처리해 모든 변수들을 입력할 수 있지만 python에선 불가능
# print(result3.summary())