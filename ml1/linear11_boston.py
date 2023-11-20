# Boston Housing Price ( 보스턴 주택 가격 데이터 ) 로 선형 회귀 모델 생성

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')

df = pd.read_csv("../testdata/housing.data", header=None, sep='\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
# print(df.head(2), df.shape)	(506, 14)
print(df.corr())
# LSTAT / MEDV  -0.737663

x = df[['LSTAT']].values
y = df[['MEDV']].values

model = LinearRegression()
model.fit(x,y)

# d=1
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)
model_r2 = r2_score(y, model.predict(x))

# d=2
quad = PolynomialFeatures(degree=2)
x_quad = quad.fit_transform(x)
model.fit(x_quad , y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
quad_r2 = r2_score(y, model.predict(x_quad))

# d = 3
cubic = PolynomialFeatures(degree=1)
x_cubic = cubic.fit_transform(x)
model.fit(x_cubic , y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
cubic_r2 = r2_score(y, model.predict(x_cubic))


# 시각화
plt.scatter(x, y, label='훈련 데이터', color='lightgray')
plt.plot(x_fit, y_lin_fit, label='선형 회귀(d=1), $R^2=%.2f$'%model_r2, lw=3, color='red', linestyle='--')
plt.plot(x_fit, y_quad_fit, label='다항 회귀(d=2), $R^2=%.2f$'%quad_r2, lw=3, color='green', linestyle=':')
plt.plot(x_fit, y_cubic_fit, label='다항 회귀(d=3), $R^2=%.2f$'%cubic_r2, lw=3, color='blue')

plt.xlabel('하위 계층 비율')
plt.ylabel('주택 가격')
plt.legend()
plt.show()
