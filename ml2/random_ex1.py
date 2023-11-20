import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager, rc

plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus']= False

data = pd.read_csv('winequality-red.csv')
# print(data.columns)

x = data.drop('quality', axis=1)
y = data['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = RandomForestClassifier(n_estimators=100, random_state=12, criterion='entropy',)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print('예측값 :', pred)
print('실제값 :', y_test)
print('총 개수 : %d, 오류 수 : %d'%(len(y_test), (y_test != pred).sum()))
# 총 개수 : 320, 오류 수 : 95
# accuracy : 0.703125

acc = accuracy_score(y_test, pred)
print(f'accuracy : {acc}')

print(f'결과 : {classification_report(y_test, pred)}')


print('분류 정확도 확인3')
print('test로 정확도는', model.score(x_test, y_test))
print('train으로 정확도는', model.score(x_train, y_train))
# 시각화
