# Support Vector Machine
# 데이터 분류를 및 예측을 위한 가장 큰 폭의 경계선을 찾는 알고리즘 사용
# 커널트릭이라는 기술을 통해 선형은 물론 비선형, 이미지 분류 까지도 처리 가능

# SVM으로 Xor
x_data = [
	[0,0,0],
	[0,1,1],
	[1,0,1],
	[1,1,1],
]

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm,metrics

df = pd.DataFrame(x_data)
# print(df)

feature = np.array(df.iloc[:, 0:2])
label = np.array(df.iloc[:,2])
print(feature)
print(label)

model1 = LogisticRegression().fit(feature, label)
pred = model1.predict(feature)
print(f'Logistic 예측 값  : {pred}')
print(f'Logistic 정확도 : {metrics.accuracy_score(label, pred)}')


model2 = svm.SVC(C=0.001).fit(feature, label)
# model2 = svm.LinearSVC().fit(feature,label)
pred2 = model2.predict(feature)
print(f'SVM 예측 값  : {pred2}')
print(f'SVM 정확도 : {metrics.accuracy_score(label, pred2)}')