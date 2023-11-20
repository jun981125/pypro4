# Ensemble 학습 : 개별적으로 동작하는 여러 모델들을 종합하여
# 예측한 결과를 투표에 가장 좋은 결과를 취하는 방법

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

pd.set_option('display.max_columns', None) # 항상 모두 출력
pd.set_option('display.max_rows', None) # 항상 모두 출력
cancerData = load_breast_cancer()
dataDf = pd.DataFrame(cancerData.data, columns=cancerData.feature_names)
print(dataDf.head(100000)) # [569 rows x 30 columns]
print(set(cancerData.target))	# {0, 1}
print(cancerData.target_names)	# ['malignant' 'benign']

x_train, x_test, y_train, y_test = train_test_split(cancerData.data, cancerData.target, test_size=0.2, random_state=12)

logiModel = LogisticRegression()
knnModel = KNeighborsClassifier(n_neighbors=3)
decModel = DecisionTreeClassifier()


classifires = [logiModel, knnModel, decModel]

for cl in classifires:
	cl.fit(x_train, y_train)
	pred=cl.predict(x_test)
	class_name= cl.__class__.__name__
	print(f'{class_name} 정확도 : {accuracy_score(y_test,pred)}')

votingModel = VotingClassifier(estimators=[('LR', logiModel),('KNN', knnModel),('Decision',decModel)], voting='soft')
votingModel.fit(x_train, y_train)

vpred= votingModel.predict(x_test)
print(f'보팅 분류기의 정확도 {accuracy_score(y_test, vpred)}')
# 보팅 분류기의 정확도 0.9385964912280702

