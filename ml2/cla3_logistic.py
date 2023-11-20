# 인디언들의 당뇨병 관련 데이터를 이용한 이항분류 : LogistricRegression 클래스 사용

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# LogisticRegression soft max function을 사용하기 때문에 , 다항 분류가 가능
# sigmoid function이 아님

names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("../testdata/pima-indians-diabetes.data.csv", header=None, names=names)
# print(df.columns)  #(767, 9)

array = df.values
# print(array, array.shape)	# (768, 9)

# sklearn feature은 행렬  label은 벡터

x = array[:, 0:8]
y = array[:, 8]
print(x[:2], x.shape) # (768, 8)
print(y[:2], y.shape) # (768,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=7)
# print(x_train , x_test)
# print(y_train , y_test)

model = LogisticRegression()
model.fit(x_train, y_train)
print('예측 값 : ', model.predict(x_test[:10]))
print('실제 값 : ', y_test[:10])
print(' 예측에 실패한 값 : ',(model.predict(x_test) != y_test).sum())

# LogisticRegression를 사용하면 score로 정확도를 바로 확인할 수 있음

print('ttest로 검정한 분류 정확도 : ', model.score(x_test,y_test))
print('train으로 검정한 분류 정확도 : ', model.score(x_train,y_train))
# train과 ttest 각각을 학습한 모델의 정확도는 비슷하게 도출되어야 한다.
pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
print('분류 정확도 : ', accuracy_score(y_test, pred))


# 모델 저장
import joblib
# joblib.dump(model,'cla4model.sav')
print("\n\n")

# 학습이 끝난 후 모델 파일 로딩 후 사용
mymodel = joblib.load('cla4model.sav')
# print(x_test[:1])
print('분류 예측 : ', mymodel.predict(x_test[:1]))

