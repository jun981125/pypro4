"""
AdaBoost (Adaptive Boosting)
가장 기본적인 부스팅 알고리즘이며, 잘못 분류된 샘플에 더 큰 가중치를 주어 다음 모델이 더 잘 학습하도록 하는 방식이야.

Gradient Boosting Machines (GBM)
Residual에 대한 모델을 학습해 나가는 방식으로, 기존 예측과 실제 값 간의 차이에 대해 새로운 모델을 만들어 나가는 방식이야.

XGBoost (eXtreme Gradient Boosting)
뛰어난 성능과 효율성을 갖고 있음. Regularization term과 두 번째 도함수를 이용하여 과적합을 줄이고, 복잡한 데이터 패턴을 학습할 수 있도록 설계돼 있어.

LightGBM
대용량 데이터와 고속으로 학습이 가능함

CatBoost
 범주형 변수를 자동으로 처리하고, 고속으로 학습할 수 있음

"""
# breast cancer dataset으로 실습
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost
from xgboost import plot_importance
import lightgbm
from lightgbm import LGBMClassifier
# xgboost 보다 연산량이 적어 손실을 줄일 수 있다. 대용량 처리가 효유렂ㄱ

dataset = load_breast_cancer()

x_feature = dataset.data
y_lable = dataset.target

cancerDf = pd.DataFrame(data=x_feature, columns=dataset.feature_names)
# print(cancerDf.head(3))
# print(dataset.target_names)
# print(np.sum(y_lable == 0))		# 양성
# print(np.sum(y_lable == 1))		# 악성

x_train, x_test, y_train, y_test = train_test_split(x_feature, y_lable, test_size=0.2, random_state=12)
# model = LGBMClassifier(boosting_type='gbdt').fit(x_train, y_train)
model = xgboost.XGBClassifier(booster='gbtree', max_depth=6, n_estimators=500).fit(x_train, y_train)
# print(model)

pred = model.predict(x_test)
print('예측 값 : ', pred[:10])
print('실제 값 : ', y_test[:10])
acc = metrics.accuracy_score(y_test, pred)
print('acc : ', acc)
print(metrics.classification_report(y_test, pred))
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(model, ax=ax)
plt.show()

