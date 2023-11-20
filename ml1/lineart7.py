# LinearRegression 클래스를 사용해 선형회귀모델 작성

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 편차가 큰 표본 데이터를 생성
sample_size = 100

np.random.seed(1)
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:10])
print(y[:10])
print('상관계수 : ', np.corrcoef(x, y)[0,1])  # 0.99939357

# 표준화 : 단위가 달라도 -1~1 사이에, 정규화 : 단위가 달라도 0~1사이로 들어오게

scaler = MinMaxScaler()  # 정규화  StandardScaler는 아웃라이어의 영향이 커 RobustScaler를 써줄 수 있다.
# 모수 추정이 어려우면 MinMaxScaler, 데이터가 정규분포를 따르면 스탠다드나 로보스트 쓰는데 강제는 아니고 분석가의 자유이다
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print('\n',x_scaled[:10].flatten())  # 차원축소하여 출력  15번의 x값과 비교해봐

# 시각화
plt.scatter(x_scaled, y)
# plt.show()

model = LinearRegression().fit(x_scaled, y)  # 독립변수, 종속변수 순. feature, label(=class)라고 얘기하기도 함
y_pred = model.predict(x_scaled)
print('예측값 : ', y_pred[:10])
print('실제값 : ', y[:10])
print()

# 모델성능 확인
# print(model.summary()) 불행히도 써머리는 ols에서만 지원함
def regScoreFunc(y_true, y_pred):
	print('r2_score(결정계수, 설명력) : {}'.format(r2_score(y_true, y_pred)))
	print('explained_variance_score(설명분산점수) : {}'.format(explained_variance_score(y_true, y_pred)))
	print('mean_squared_error(MES, 평균제곱근 오차) : {}'.format(mean_squared_error(y_true, y_pred)))

regScoreFunc(y, y_pred)
# r2_score(결정계수, 설명력) : 0.9987875127274646
# explained_variance_score(설명분산점수) : 0.9987875127274646  // 결정계수와 설명분산점수의 결과의 차이(편향)가 크다면 모델학습이 잘못됐다고 봄
# mean_squared_error(MSE, 평균제곱근 오차) : 86.14795101998743  == SSE 같은 값


print('---------')
# 분산이 크게 다른 표본 데이터를 생성
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print('상관계수 : ', np.corrcoef(x,y)[0,1])
# 0.004011673780558856


x_scaled2 = scaler.fit_transform(x.reshape(-1,1))
print('\n',x_scaled2[:10].flatten())  # 차원축소하여 출력  15번의 x값과 비교해봐


model2 = LinearRegression().fit(x_scaled2, y)  # 독립변수, 종속변수 순. feature, label(=class)라고 얘기하기도 함
y_pred2 = model.predict(x_scaled2)
print('예측값 : ', y_pred2[:10])
print('실제값 : ', y[:10])
regScoreFunc(y, y_pred2)
# r2_score(결정계수, 설명력) : -0.23919537320391226

