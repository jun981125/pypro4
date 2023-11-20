# 단순 선형 회귀 분석 모델 작성 : ols() 함수 - OLS Regression Results 내용 알기
# 결정론적 선형 회귀 분석 방법 - 확률적
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.rc('font', family='AppltGothic')
df = pd.read_csv('../testdata/drinking_water.csv')
print(df.corr(method='pearson'))

# 독립변수 (x, feature) : 친밀도
# 종속변수 (y, label) : 적절성
# 목적 : 주어진 feature와 결정적 기반에서 기반에서 학습을 통해 최적의 회귀계수(slope, bias)를 찾아내는 것

model = smf.ols(formula='만족도~ 적절성', data=df).fit()
print(model.summary())

"""
F-statistic 에서 유도된 값
회귀모델의 적합함을 증명 - 통계적으로 유의한 모델이다
모델의 p-value :  Prob (F-statistic): 2.24e-52  < 0.05 

t - 값 : 두 변수간 평균의 차이
t =  ß (기울기/ coef)  / std err(표준 오차)
0.7393 / 0.038 = 19.340
19.340  2 = 374.0
374.0 -> Prob 유도  
Prob 값은 0.05보다 작아야 유의한 모델

standard error : 표본 평균들의 표준 오차


- t-값 (t-statistic): 두 변수 간 평균의 차이를 나타냅니다.
- t = β / std err: t-값은 회귀 계수(β, 기울기)를 표준 오차로 나눈 값이에요.
- 374.0 = (0.7393 / 0.038)  2: t-값을 제곱한 값은 F-statistic이 되어, 이 값은 모델 전체의 통계적 유의성을 검정합니다.
- Prob (F-statistic): F-statistic에 대한 p-value는 모델의 통계적 유의성을 나타내며, 작을수록 모델이 통계적으로 유의미하다는 것을 의미합니다.
- Prob 값이 0.05보다 작아야 유의한 모델: 일반적으로 유의수준(α)을 0.05로 설정하며, 이보다 작은 p-value는 우연이 아닌 효과라고 간주됩니다.
- Standard Error (표준 오차): 표본 평균들의 표준 오차로, 회귀 계수의 불확실성을 나타냅니다.

이렇게 계산하고 해석하셔서 모델의 통계적 유의성을 확인하는 것이 정확한 방법입니다. 훌륭한 이해력이에요!

 표준오차가 커지면 t값은 작아지고 p값은 커짐
 p 값은 귀무가설이 성립한다는 가정하에 극단적인 데이터가 나올 확률 이므로, 
 데이터간 거리를 나타내는 표준오차는 당연하게 함께 커진다
 
 t-값이 2 이상이면 통계적 유의미하다고 함
 t = 회귀계수/ 표준오차
 이므로 표준오차와 반비례
 

"""


"""
이는 단순 선형 회귀 모델의 결과를 나타내는 통계 표입니다. 각 항목을 하나씩 설명해보겠습니다:

1. Dep. Variable (종속 변수):
   - 만족도(Satisfaction)가 모델에서 사용된 종속 변수입니다.

2. R-squared (결정 계수):
   - R-squared는 종속 변수의 변동 중 모델이 설명하는 비율을 나타냅니다. 여기서는 0.588이므로 모델이 종속 변수의 약 58.8%의 변동을 설명한다고 해석할 수 있습니다.
	독립변수가 2개 이상인 경우 Adj. R-squared 값을 사용해야함
	
3. Model (모델):
   - Ordinary Least Squares (OLS)을 사용한 선형 회귀 모델입니다.

4. Method (방법):
   - Least Squares는 모델을 적합할 때 사용된 최소제곱법을 나타냅니다.

5. F-statistic (F 통계량):
   - 모델 전체의 통계적 유의성을 검정하는 F 통계량입니다. 여기서는 374.0이며, 이 값이 클수록 모델이 통계적으로 유의미하다는 것을 나타냅니다.

6. Prob (F-statistic):
   - F-statistic에 대한 p-value입니다. 이 값이 작을수록 모델이 통계적으로 유의미하다고 할 수 있습니다. 여기서는 2.24e-52로 매우 작은 값이므로 모델은 통계적으로 유의미하다고 할 수 있습니다.

7. Intercept (절편):
   - 모델의 y 절편으로, 적절성이 0일 때 만족도의 예측값입니다. 여기서는 0.7789이며, 통계적으로 유의미한 값으로 나타납니다.

8. 적절성 (Sufficiency):
   - 모델에서 사용된 독립 변수로, 만족도에 대한 회귀 계수입니다. 여기서는 0.7393이며, 이 값은 적절성이 증가함에 따라 만족도도 증가한다는 것을 나타냅니다.

9. P>|t|:
   - 각 독립 변수에 대한 t-통계량에 대한 p-value입니다. 이 값이 유의수준(일반적으로 0.05)보다 작으면 해당 독립 변수는 통계적으로 유의미한 영향을 미친다고 할 수 있습니다.

10. [0.025 0.975]:
   - 신뢰구간(Confidence Interval)입니다. 각 회귀 계수에 대한 신뢰구간을 나타냅니다.

11. Omnibus, Durbin-Watson, Jarque-Bera, Skew, Kurtosis:
   - 잔차(residuals)에 대한 다양한 통계적 검정과 통계량들을 나타냅니다. 잔차에 대한 정규성, 자기상관 등을 검정하는데 사용됩니다.

Omnibus:Omnibus 테스트는 잔차의 정규성을 확인하는 테스트입니다. 즉, 잔차가 정규분포를 따르는지를 검정하는 것이에요. Omnibus 테스트의 p-value가 유의수준(일반적으로 0.05)보다 작으면, 정규성 가정이 만족되지 않을 가능성이 높다는 것을 나타냅니다.

Durbin-Watson:Durbin-Watson 통계량은 잔차들 간의 자기상관을 검정하는 데 사용됩니다. 이 통계량은 0과 4 사이의 값을 가지며, 2에 가까울수록 자기상관이 없다는 것을 의미합니다. 0 또는 4에 가까우면 양(positive) 또는 음(negative)의 자기상관을 나타냅니다.

Jarque-Bera:Jarque-Bera 테스트는 잔차의 왜도(skewness)와 첨도(kurtosis)에 대한 테스트입니다. 정규분포를 따르면 p-value가 높게 나와야 합니다. 만약 p-value가 낮다면, 잔차가 정규분포를 따르지 않을 가능성이 높다는 것을 의미합니다.

Skew (왜도):왜도는 분포의 비대칭성을 나타내는 지표입니다. 왜도가 0보다 크면 오른쪽으로 치우쳐져 있고, 작으면 왼쪽으로 치우쳐져 있습니다.

Kurtosis (첨도):첨도는 분포의 뾰족한 정도를 나타내는 지표입니다. 정규분포의 첨도는 3이고, 이보다 크면 더 뾰족하고 작으면 덜 뾰족한 분포를 나타냅니다.

12. Cond. No.:
   - 조건수(Condition Number)는 회귀 분석에서 다중공선성(multicollinearity)을 측정하는 지표로 사용됩니다. 일반적으로 20보다 크면 다중공선성의 문제가 발생할 수 있습니다.
"""


"""
	최소 제곱법의 근거를 보기 위해서는 ols 함수를 사용해서
 표준오차를 확인해야함. 표준오차는 p값 과 비례관계. p-value 가 작아야함


- t-값 (t-statistic): 두 변수 간 평균의 차이를 나타냅니다.
- t = β / std err: t-값은 회귀 계수(β, 기울기)를 표준 오차로 나눈 값이에요.
- 374.0 = (0.7393 / 0.038)  2: t-값을 제곱한 값은 F-statistic이 되어, 이 값은 모델 전체의 통계적 유의성을 검정합니다.
- Prob (F-statistic): F-statistic에 대한 p-value는 모델의 통계적 유의성을 나타내며, 작을수록 모델이 통계적으로 유의미하다는 것을 의미합니다.
- Prob 값이 0.05보다 작아야 유의한 모델: 일반적으로 유의수준(α)을 0.05로 설정하며, 이보다 작은 p-value는 우연이 아닌 효과라고 간주됩니다.
- Standard Error (표준 오차): 표본 평균들의 표준 오차로, 회귀 계수의 불확실성을 나타냅니다.

이렇게 계산하고 해석하셔서 모델의 통계적 유의성을 확인하는 것이 정확한 방법입니다. 훌륭한 이해력이에요!


시각화 했을 때 분포가 선형성을 띄지 않는다면, polynomial로 진행해야함

SSR = 총 변동값
SST = SCR + SSE
결정계수 독립 변수가 종속변수의 분산을 설명해주는 값
"""


# 예측값
print(df.적절성[:5].values)
new_df = pd.DataFrame({'적절성':[4,3,4,2,2]})
new_pred = model.predict(new_df)
# 0.588 설명력이 있는 모델로 검정
print('만족도 실제값', df.만족도[:5].values)
print('만족도 예측값', new_pred.values)
print('만족도 실제값과 예측값의 차이 : ',(new_pred.values - df.만족도[:5].values )/ df.만족도[:5].values  * 100)
