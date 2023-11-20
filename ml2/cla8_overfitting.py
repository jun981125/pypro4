"""
과적합 방지 방법 : train/test split , KFold, GridSearch ...
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data
train_label = iris.target

print(train_data[:2])
print(train_label[:2])

# 분류 모델
dt_clf = DecisionTreeClassifier()	# 다른 분류 모델도 가능
dt_clf.fit(train_data,train_label)
pred = dt_clf.predict(train_data)
print('예측값 : ',pred[:10])
print('실제값 : ',train_label[:10])
print('분류 정확도 : ', accuracy_score(train_label, pred))
# 과접합 의심 / 정확도 1.0

print('\n과적합 방지 방법1: train/test split')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=121)
dt_clf.fit(x_train, y_train)	# 	train으로 학습
pred2 = dt_clf.predict(x_test)	# 	test로 검정

print('예측값 : ',pred2[:10])
print('실제값 : ', y_test[:10])
print('분류 정확도:' ,accuracy_score(y_test, pred2))

# 분류 정확도: 0.9555555555555556 포용성이 있는 모델 생성



from sklearn.model_selection import KFold
print('과접합 방지 방법 2 : cross validation, KFold')
# ttest와는 다르게 동적으로 검정을 진행함 ttest의 경우는 학습용 데이터와
# 테스트용 데이터를 사용해서 메소드를 실행해 편중된 결과가 도출될 수 있지만,
# 교차검증을 사용하면 모든 데이터를 훈련에 교차적으로 참여시킴으로써
# 모델의 성능과 정확도를 더 향상 시킬 수 있고 좀 더 일반화된 모델을 만들 수 있다.
# 또한 데이터 부족으로 발생하는 undefitting을 방지 할 수있고, 가장 큰 이유인 test set의
# 편중을 방지할 수 있다.
import numpy as np
feature = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=123)
kfold = KFold(n_splits=5)
cv_acc = []
print('iris shape : ', feature.shape)	#  (150, 4)
# 전체 행 수가 150, 학습 데이터 0.8 * 150 = 120, 검증데이터 30

n_iter =0
for train_index, test_index in kfold.split(feature):
	print('n_iter : ',n_iter)
	print(train_index, ' ', len(train_index))
	print(test_index, ' ', len(test_index))
	xtrain, xtest = feature[train_index], feature[test_index]
	ytrain, ytest = label[train_index], label[test_index]
	# 모델 생성 중 학습 및 검증
	dt_clf.fit(xtrain, ytrain)		# 학습
	pred = dt_clf.predict(xtest)	# 검증
	n_iter+=1

	# 반복할 때마다 정확도 측정
	acc = np.round(accuracy_score(ytest, pred), 3)
	train_size = xtrain.shape[0]
	test_size = xtest.shape[0]
	print(f'반복수 : {n_iter}, 교차 검증 정확도 : {acc}, 학습 데이터 크기 : {train_size}, 검증 데이터 크기 : {test_size}')
	print(f'반복수 : {n_iter}, 검증 자료 인덱스 : {test_index}')
	cv_acc.append(acc)

print('모델 생성 도중 학습 평균 검증 정확도', np.mean(cv_acc))
# 0.9199999999999999
# 교차 검정을 끝낸 후 도출되는 정확도 ( 교차 검정의 정확도들의 평균으로 정해짐 )

print('실제 교차검증( cross validation , KFold ) 시 cross_val_score 사용')
from sklearn.model_selection import cross_val_score
# sklearn이 kfold를 내부적으로 지원
data = iris.data
label = iris.target

score = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=5)
print(f'교차 검증별 정확도 {np.round(score,3)}')
print('검증 정확도 : ', np.round(np.mean(score),3))


# 참고
# StratifiedKFold : 불균형한 분포를 가진 레이블 데이터의 경우 사용
# 편향이나, 왜곡이 되어있는 데이터를 다룰 때 사용

from sklearn.model_selection import StratifiedKFold



print('과적합 방지 방법 3 : GridSearchCV')
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[1,2,3], 'min_samples_split': [2,3]}

grid_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=3, refit=True)
# refit = True 일 때, 최적의 파라미터를 위해 재학습을 시도
#  최적의 Estimator(추정기) 모델을 찾기 위해 학습을 반복

grid_dtree.fit(x_train, y_train)	# 자동으로 복수의 내부 모형을 생성하고 실행시켜 파라미터를 찾아줌

import pandas as pd
score_df = pd.DataFrame(grid_dtree.cv_results_)
print(score_df)

print('GridSearchCV가 추천한 최적 파라미터 : ', grid_dtree.best_params_)
print('GridSearchCV가 추천한 최적 정확도 : ', grid_dtree.best_score_)

print()
estimator = grid_dtree.best_estimator_
print(estimator)
pred = estimator.predict(x_test)
print(pred)
print('estimator(추천된 DecisionTreeClassifier) 정확도 : ', accuracy_score(y_test, pred))
# 0.9555555555555556

