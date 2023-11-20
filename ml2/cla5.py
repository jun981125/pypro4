# 분류 모델 성능 파악을 위해 Confusion Matrix를 활용
# Accuracy, Precision, Recall 등의 지표를 사용
# ROC curve, AUC도 사용
import pandas as pd
# 연구 보고서 주제를 설정 -> 데이터 수집 및 가공 -> 모델 생성 및 학습
# -> 모델 평가 -> 유의하다면 인사이트를 얻어 의사 결정 자로로 활용
# (유의하지 않다면 다시 학습)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np

x, y = make_classification(n_samples= 100, n_features=2, n_redundant=0, random_state=123)
# 표본 100 독립벼수 2개 , n_redundant는 독립 변수 중 다른 독립변수의 선형 조합으로 나타낸 성분 수
print(x[:3],x.shape)
print(y[:3],y.shape)

import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,0])
# plt.show()
model = LogisticRegression().fit(x,y)
y_hat = model.predict(x)

print('예측값 : ',y_hat[:3])
print()

# 판별 경계썬 설정
f_value = model.decision_function(x)
# 결정함수 (판별함수) : 불확실성 추정함수, 판별 경계썬 설정을 위한
# 샘플 데이터 지원
print('f_value : ',f_value[:10])
# [ 0 0 0 1 1 1 0 0 0 1 ]
df=pd.DataFrame(np.stack([f_value, y_hat, y]).T, columns=['f_value', 'y_hat', 'y'])
print(df.head(3))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat))

acc = (44 + 44) / 100		# TP + TN / 전체수

recall = 44 / (44 + 4)		# TP / TP + FN

precision = 44 / (44+8)		# TP / TP + FP

specificity = 44 / (8+44)	# TN / FP + TN

fallout = 8 / (8+44)		# FP / (FP + TN )


print(f'acc(정확도) : {acc}')
print(f'recall(재현율,민감도, TPR) : {recall}')
print(f'precision(정밀도) : {precision}')			# 전체 양성 자료 중에 양성으로 예측된 비율 1에 가까울 수록 좋음
print(f'specificity(특이도) : {specificity}')
print(f'fallout(위양성율, FPR) : {fallout}')		# 전체 음성 자료 중에 양성으로 잘못 예측된 비율 0에 가까울 수록 좋음
print(f'fallout(위양성율) : {1 - specificity}')
print()

from sklearn import metrics
ac_sco = metrics.accuracy_score(y, y_hat)
print('ac_sco : ', ac_sco)
cl_rep = metrics.classification_report(y,y_hat)
print(cl_rep)
"""
정밀도 (Precision): 양성으로 예측한 것 중에서 실제로 양성인 비율을 나타냅니다
재현율 (Recall 또는 Sensitivity): 실제 양성 중에서 모델이 양성으로 예측한 비율을 나타냅니다.
F1 점수 (F1-score): 정밀도와 재현율의 조화 평균입니다. 이는 두 메트릭 사이의 균형을 나타냅니다.
지원 (Support): 각 클래스에 대한 실제 샘플 수를 나타냅니다.
"""

print()
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류결정 임계값 : ', thresholds)


# ROC(Receiver Operator Characteristic, 수신자 판단) Curve
# 클래스 판별 기준값의 변화에 따른 Fall-out과 Recall의 변화를 시각화 함.
# FPR이 변할 때 TPR이 어떻게 변화하는지를 나타내는 곡선이다.

# ROC는 위양성률(1-특이도)을 x축으로, 그에 대한 실제 양성률(민감도)을 y축 으로 놓고
# 그 좌푯값들을 이어 그래프로 표현한 것이다. 일반적으로 0.7~0.8 수준이 보통의 성능을 의미한다.
# 0.8~0.9는 좋음, 0.9~1.0은 매우 좋은 성능을 보이는 모델이라 평가할 수 있다.

# import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1],[0,1], 'k--', label='Landom classifier Line(Auc:0.5)')
plt.plot([fallout], [recall], 'ro', ms=10)
plt.xlabel('fpr', fontdict={'fontsize': 16})
plt.ylabel('tpr')
plt.legend()
plt.show()


# Auc 모델의 성능을 정량적으로 측정하는 지표
# 0과 1사이의 값을 갖고 높을 수록 모델이 좋은 성능을 가진다고 평가함
print('Auc : ', metrics.auc(fpr,tpr))


# Accuracy Score: 예측이 전체 데이터 중에서 얼마나 정확한지를 나타내는 지표로,
# 정확히 분류된 샘플의 비율을 나타냅니다. 다음과 같이 정의됩니다.

# AUC (Area Under the Curve):
# ROC 커브는 True Positive Rate (Sensitivity)와 False Positive Rate 사이의 관계를 시각화한 것입니다.
# AUC는 ROC 커브의 아래 영역을 나타내며, 이 값이 1에 가까울수록 모델의 성능이 좋다고 할 수 있습니다.
# AUC가 0.5에 가까우면 모델이 무작위로 분류한 것과 비슷하며
# , 1에 가까울수록 모델의 분류 성능이 좋습니다.