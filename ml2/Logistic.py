"""
로지 스틱 회귀 분석
	종속 변수의 특이성
	종속 변수가 이항 변수 (Binomial) 일 때


odds (오즈 또는 승산 ) - 어떤 사건이 발생할 확률과 발생하지 않을 확률 간의 비율
p/1-p = 이벤트 발생확률 / 이벤트가 발생하지 않을 확률

odds ratio (오즈비, 승산비) - 오즈비는 두 개의 다른 그룹 간의 오즈를 비교하여 두 그룹 간의 상대적인 이벤트 발생 가능성

ex)
 독립 변수 : 코로나 환자와의 접촉 여부  / 종속 변수 : 코로나 감염 유무
 코로나 환자와 접촉이 없던 상황 대비 직접 접촉한 사람의 코로나 감염 오즈비
=  코로나 환자와 만나고 감염될 확률 / 코로나 환자와 만나지않고 감염될 확률
= ( 코로나 환자와 만나고 감염될 확률 / 코로나환자와 만나고 감염되지 않을 확률 ) /
	(코로나 환자와 만나지 않고 감염될 확률 / 코로나 환자와 만나지 않고 감염되지 않을 확률 )


오즈비가 1보다 크면, 코로나 환자와 만남으로 인해 감염될 확률이 더 높다
반대로, 1보다 작으면 코로나 환자와 만나지 않고도 감염될 확률이 더 높다

오즈와 단순 확률 p는 거의 동일하고 오즈를 오즈로 나누면 비교가 가능해진다 - 오즈비
추가적으로 오즈비에 로그를 취하면 p가 0에 가까울 수록 0으로 수렴
p가 1에 가까울 수록 무한대로 수렴

0 < odds ratio < infinity
-infinity < log(odds ratio) : 로짓(logit) < +infinity
오즈비에 로그를 취하는 이유는 종속 변수는 이항 변수이므로 데이터가 0과 1이다. 이 값에 로그를 취하면
도출되는 값들은 연속 변수가 되는데 이 때의 범위가 0 부터 무한대 에서 -무한대 부터 +무한대까지로 범위가
바뀌므로 분류를 좀 더 용이하게 변형해준다.

"""
import collections
import graphviz
import pydotplus
from sklearn import tree

# 키, 머리카락 길이로 남여 구분
x = [[3000000000000000000,3000000000000000000],[3000000000000000000,3000000000000000000],[156,35],[174,5],[3000000000000000000,3000000000000000000],[192,5],[197,3000000000000000000],[190,25],[184,10],[166,13]]
y = ['남', '여', '여', '남', '여', '남', '남', '남', '남', '여']
label_names = ['키', '머리길이']

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(x, y)
# entropy = 데이터 집합에서 혼잡도(불순도)를 말하며, 클수록 불순도가 높음
pred = model.predict(x)
print('훈련 정확도 ', model.score(x, y))

# 시각화
dot_data = tree.export_graphviz(model, feature_names=label_names, filled=True, rounded=True, impurity=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('red', 'orange')
edges = collections.defaultdict(list)

for e in graph.get_edge_list():
	edges[e.get_source()].append(int(e.get_destination()))

for e in edges:
	edges[e].sort()
	for i in range(2):
		dest = graph.get_node(str(edges[e][i]))[0]
		dest.set_fillcolor(colors[i])

# 이미지 저장
image_path = 'tree.png'
graph.write_png(image_path)
print(f'이미지가 {image_path}에 저장되었습니다.')
