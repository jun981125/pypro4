from io import StringIO
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

data = StringIO("""
요일,외식유무,소득수준
토,0,57
토,0,39
토,0,28
화,1,60
토,0,31
월,1,42
토,1,54
토,1,65
토,0,45
토,0,37
토,1,98
토,1,60
토,0,41
토,1,52
일,1,75
월,1,45
화,0,46
수,0,39
목,1,70
금,1,44
토,1,74
토,1,65
토,0,46
토,0,39
일,1,60
토,1,44
일,0,30
토,0,34""")
df = pd.read_csv(data)
df = df[(df['요일'] == '토') | (df['요일']=='일')]
train, test = train_test_split(df, test_size=0.3, random_state=5)

model = smf.logit(formula='외식유무 ~ 소득수준', data = df).fit()
# print(model.summary())
pred = model.predict(df)
print('glm 분류 정확도 : ', accuracy_score(df['외식유무'], np.around(pred)))
param = input('소득 수준을 입력하세요 : ')
param_df = pd.DataFrame({'소득수준':[float(param)]})
param_pred = model.predict(param_df)
# print(np.around(param_pred.values[0]))
if param_pred.values > 0.5:
	param_pred.values = 1
else:
	param_pred.values = 0

print(f'소득수준 {param}에 의한 의식 여부는 {param_pred.values}')