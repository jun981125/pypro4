# LogisticRegression 클래스 사용
# 다항분류(활성화 함수 (softmax)) iris dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager, rc
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus']= False
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
# print(iris.DESCR)
print(iris.feature_names)
# print(iris.data)
# print(iris.data[:2])
# 현재 iris 는 numpyarray
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))		# 0.96286543
# 'petal length (cm)', 'petal width (cm)'


x = iris.data[:,[2,3]]	#
y_data = iris.target

x_train,x_test,y_train,y_test= train_test_split(x,y_data,test_size=0.3, random_state=0)
model = LogisticRegression(random_state = 0, solver='liblinear')


# 분류 예측
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print('예측 값 : ', y_pred)
print('실제 값 :	', y_test)
print(f'총 갯수 : {len(y_test)}, 오류 수 : {(y_test!=y_pred).sum()}')
print("정확도", accuracy_score(y_test,y_pred))

print()
"""
# scaling(데이터의 크기를 고르게 만들어줌)
# feature에 대한 표준화 , 정규화 : 최적화 과정에서 안정성 , 수렴 속도를 향상
# 오버피팅 or 언더피팅 방지

# print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])

# 스케일링 값 원복
invers_x_train = sc.inverse_transform(x_train)
invers_x_test = sc.inverse_transform(x_test)
print(invers_x_train[:3])
# 재 복원시는 근사치로 값이 도출될 수 있음


StandardScaler 
데이터의 특성(feature)을 평균을 빼고 표준편차로 나누어서
 평균이 0이고 표준편차(분산)가 1인 값들로 변환해주는 스케일링 방법
각 특성의 값들을 표준 정규 분포로 만들어주어 모델 학습시에 좋은 성능을 낼 수 있도록 도와줌
  
"""

# 모델작성 --------
model = LogisticRegression(C=1.0 , solver='lbfgs', multi_class='auto',random_state=0, verbose=0)
# C : ( )  -> 분류 정확도 조장 - L2 규제 ( 정규화 ) : 값이 작을 수록 규제가 강함
# default는 1
# 가중치의 제곱의 합에 비례하는 페널티를 부여하여 모든 가중치를 작게 만든다
# mult_class = 'auto' 또는 'multinomial', 'ovr'
# 'ovr' (One-vs-Rest)
# 기본값으로, 이진 분류 방식
# 각 클래스에 대해 하나의 이진 분류 모델을 만들어 해당 클래스에 속하는지 여부를 판별
# 'multinomial'
# 다중 클래스 분류 방식을 사용
# 소프트맥스(softmax) 함수를 사용하여 각 클래스에 속할 확률을 계산하고, 가장 확률이 높은 클래스로 분류
# 'auto'
# 자동으로 'ovr' 또는 'multinomial' 중에서 선택
# 이는 주어진 문제에 대해 알고리즘이 자동으로 선택
"""
 solver = 'lbfgs' : softmax 지원
 
'lbfgs' (Limited-memory Broyden–Fletcher–Goldfarb–Shanno):
연속된 함수의 값을 최소화하기 위한 반복 알고리즘 중 하나입니다.
작은 규모의 데이터셋에 적합하며, 다양한 다변수 함수에서 잘 작동합니다.

'liblinear':
작은 데이터셋에서 이진 및 다중 클래스 문제에 적합합니다.
L1 규제와 L2 규제를 모두 지원하며, 이진 분류에 특히 적합합니다.

'newton-cg':
Newton's Conjugate Gradient 알고리즘을 사용합니다.
작은 규모의 데이터셋에 적합하며, L2 규제를 지원합니다.

'sag' (Stochastic Average Gradient Descent):
확률적 경사하강법을 기반으로 하는 알고리즘입니다.
대규모 데이터셋에서 효과적이며, L2 규제를 지원합니다.

'saga':
'sag'의 변종으로, L1 규제와 L2 규제를 모두 지원합니다.
대규모 데이터셋에 적합합니다.
"""

# verbose
# 0 , 출력 없음 (기본값)
# 1, 학습 진행 상황을 간단하게 출력
# 2, 학습 진행 상황을 상세하게 출력
# 추가적인 학습을 해도 큰 변화가 없다고 생각하면 스스로 학습을 중단함


# print(model)
# -------------
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('예측 값 : ', y_pred)
print('실제 값 :	', y_test)
print(f'총 갯수 : {len(y_test)}, 오류 수 : {(y_test!=y_pred).sum()}')
print("정확도", accuracy_score(y_test,y_pred))

print('분류 정확도 확인 2')
con_mat = pd.crosstab(y_test, y_pred , rownames=['예측값'], colnames=['관측값'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat [2][2])/len(y_test) )

print('분류 정확도 확인 3')
print('test로 정확도는 ',model.score(x_test, y_test))
print('train으로 정확도는 ',model.score(x_train, y_train))
# iris의 정확도 최댓값 0.9777777777777777

# 모델 성능이 만족스러운 경우 모델 저장
import joblib
# joblib.dump(model, 'mymodel.sav')


del model			# 모델 삭제

mymodel = joblib.load('mymodel.sav')
print('새로운 값으로 분류 예측 - petal length (cm), petal width (cm) - 스케일링해서 학습했다면, '
	  '예측 데이터도 스케일링 해야함 ')

print(x_test[:2])
new_data = np.array([[5.1,2.4], [0.1,0.1], [5.6,5.6], [8.1, 0.5]])
new_pred = mymodel.predict(new_data)
# softmax 함수가 제공한 결과에 대해 가장 큰 인덱스를 반환

print(f'예측 결과 : {new_pred}')
print(f'softmax 결과  : {mymodel.predict_proba(new_data)}')

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
	markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
	colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	# print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

	# decision surface 그리기
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

	# xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의
	# predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
	Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
	Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

	# X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
	plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())

	X_test = X[test_idx, :]
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

	if test_idx:
		X_test = X[test_idx, :]
		plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

	plt.xlabel('꽃잎 길이')
	plt.ylabel('꽃잎 너비')
	plt.legend(loc=2)
	plt.title(title)
	plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, test_idx=range(105, 150), title='scikit-learn제공')
