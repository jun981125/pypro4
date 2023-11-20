import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np

data = pd.read_csv('../testdata/advertisement.csv')
"""
[로지스틱 분류분석 문제3]
참여 칼럼 : 
  Daily Time Spent on Site : 사이트 이용 시간 (분)
  Age : 나이,
  Area Income : 지역 소득,
  Daily Internet Usage:일별 인터넷 사용량(분),
  Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
데이터 간의 단위가 큰 경우 표준화 작업을 시도한다.
ROC 커브와 AUC 출력"""

x = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = data['Clicked on Ad']
# model = LogisticRegression().fit(x,y)
# y_hat = model.predict(x)
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
x_standardized = scaler.fit_transform(x)


model = LogisticRegression().fit(x_standardized, y)
y_hat = model.predict(x_standardized)

fpr, tpr, _ = metrics.roc_curve(y, model.decision_function(x_standardized))
print('Auc:', metrics.auc(fpr, tpr))

# 정확도 및 분류 보고서 계산
ac_sco = metrics.accuracy_score(y, y_hat)
print('Accuracy Score:', ac_sco)

cl_rep = metrics.classification_report(y, y_hat)
print('Classification Report:\n', cl_rep)

# 표준화된 데이터를 사용하여 ROC 커브 그리기
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier Line (AUC: 0.5)')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend()
plt.show()
