from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# 공부시간에 따른 시험 점수표
df = pd.DataFrame({'studytime': [3 ,4 ,5 ,8 ,10,5 ,8 ,6 ,3 ,6 ,10,9 ,7 ,0 ,1 ,2],
					  'score' : [76,74,74,89,92,75,84,82,73,81,89,88,83,40,70,69]})

# print(df)

# print(df.head(2), df.shape)  # 16, 2   모집단

# dataset을 분리해서 학습 및 검정을 실시 ( 모델의 과적합 방지 목적 )
train, test = train_test_split(df, test_size=0.3 ,random_state=12)
# 비복원추출
# 0.2 | 8:2    0.4 | 6:4

print(train)
print(test)
x_train = train[['studytime']]
print(x_train.shape)		# (9, 1) -> 2차원 matrix
# matrix인 이유는 : sklean의 분류 및 예측 클래스는
# feature : 2차원  	lable : 1차원

y_trian = train['score']		# 1차원 벡터
# print(y_trian.shape)
x_test = test[['studytime']]
y_test = test['score']
print(x_train.shape, x_test.shape, y_trian.shape, y_test.shape)

# LineartRegression
model = LinearRegression()
model.fit(x_train, y_trian)		#모델학습은 훈련용 데이터로
y_pred =model.predict(x_test)
print(y_test.values)
print(y_pred)

# 결정계수 수식 사용
y_mean = np.mean(y_test)
nomerator = np.sum(np.square(y_test - y_pred)) #SSE ( 오차 제곱합)
denomerator = np.sum(np.square(y_test - y_mean)) # SST ( 편차 제곱합)

r2 = 1 - nomerator /denomerator  # 1- (SSE / SST)
print('결정계수 : ', r2)
print('결정계수 : ', r2_score(y_test, y_pred))
# 결정계수는 분산을 기반으로 측정하므로 중심 극한 정리에 의해 표본 데이터가 많을 수록
# 결정계쑤 수치도 높아진다.
# 무의한 독립변수의 수가 늘면 결정계수 값이 늘어나는 경향이 있으므로, 변수 선택에 주의가 필요하다.
# 결정계수 값은 맹신하면 안됨

print('\n자동차 데이터 ( mtcars .csv )로 선형 회귀 모델 작성')
import statsmodels.api
mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
# print(mtcars.shape)	(32, 11)
print(mtcars.corr(method='pearson'))

# mpg에 영향을 주는 feature로 hp를 선택
feature = mtcars[['hp']].values
label = mtcars['mpg'].values
# feeature는 2차원
# label은 1차원


# plt.scatter(feature, label)
# plt.show()

lmodel = LinearRegression().fit(feature, label)
print('회귀계수 (slope) ', lmodel.coef_)
print('회귀계수 (intercept) ', lmodel.intercept_)

pred = lmodel.predict(feature)
print('예측값 : ', np.round(pred[:5], 1))
print('실제 값:', label[:5] )

# 모델 성능 평가
print('MSE : ', mean_squared_error(label,pred))
print('r2_score : ', r2_score(label, pred))

print()

# 새로운 hp로 mpg를 예측
new_hp = [[110]]
new_pred = lmodel.predict(new_hp)
print(f'{new_hp[0][0]} 마력인 경우 예상 연비는 약 {new_pred[0]}입니다')