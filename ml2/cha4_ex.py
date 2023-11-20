"""
[로지스틱 분류분석 문제2]

게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.

안경 : 값0(착용X), 값1(착용O)

예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv

새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
"""

import pandas as pd

data = pd.read_csv("../testdata/bodycheck.csv")
print(data)


"""
Accuracy  TP+TN / 전체 개수
recall(재현율, 민감도) : TP/TP+FN 

precision : TP/TP+FP
Specifictiy 특이도 : TN/TN+FN


예측값 \ 실제값			Y					N
		Y		True Positive(TP)	False Positive(FP)
		N		False Negative(FN)	True Negative(TN)
		
머신 러닝의 라이프 사이클
init model -> train model -> score model -> evaluate model
Initialize (Init) Model:
이 단계에서는 모델을 초기화하고 필요한 매개변수를 설정합니다. 
모델의 아키텍처를 정의하고 초기 가중치를 설정하는 등의 작업이 이루어집니다.

Train Model:
초기화된 모델에 학습 데이터를 제공하여 모델을 학습시킵니다. 
학습은 주어진 입력과 실제 출력(혹은 레이블) 간의 차이를 최소화하기 위해 모델의 가중치를 조정하는 과정입니다. 
주로 경사 하강법과 같은 최적화 알고리즘이 사용됩니다.

Score Model:
학습된 모델은 새로운 입력 데이터에 대한 예측을 수행할 수 있습니다. 
"Score model" 단계에서는 학습된 모델을 사용하여 입력 데이터에 대한 예측을 생성합니다.

Evaluate Model:
생성된 예측 결과를 실제 결과와 비교하여 모델의 성능을 평가합니다. 
분류 모델의 경우 정확도, 정밀도, 재현율 등이 사용되며, 회귀 모델의 경우 평균 제곱 오차(MSE), 결정 계수(R-squared) 
등이 일반적인 평가 메트릭으로 활용됩니다. ROC curve를 그려서 AUC (밑 면적 평균)을 구함

Auc 모델의 성능을 정량적으로 측정하는 지표
0과 1사이의 값을 갖고 높을 수록 모델이 좋은 성능을 가진다고 평가함

		
"""