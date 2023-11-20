import pandas as pd
import joblib

mymodel = joblib.load('linear6m.model')

new_df = pd.DataFrame({'Price':[105,89,75],'Income':[35,62,24], 'Advertising':[6,3,11],'Age':[35,42,21]})
pred = mymodel.predict(new_df)
print(f'예측 값 : {pred}')