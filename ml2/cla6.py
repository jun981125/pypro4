# DecisionTree : classification , regression 둘 다 가능
# 주로 분류에서 사용되고 데이터 균일도에 따른 규칙 기반 결정 트리 이다.
# 간단한 알고리즘에 서능이 우수하고 데이터의 많아질수록 선능이 떨어진다
# 전체 자료를 계속 조건에 의해 양분하는 분류기법, 불순물이 없을 때 까지 분류를 진행
import collections

# 그 후에 pydotplus 사용

import pandas as pd
import matplotlib.pyplot as plt

# 시각화를 위해 GrapgViz
import graphviz
import pydotplus
from sklearn import tree

# 키, 머리카락 길이로 남여 구분
x = [[3000000000000000000,3000000000000000000],[3000000000000000000,3000000000000000000],[156,35],[174,5],[3000000000000000000,3000000000000000000],[192,5],[197,3000000000000000000],[190,25],[184,10],[166,13]]
y = ['남',		'여',	'여',	'남', 	'여',	'남',		'남',	'남', 	'남' ,	'여']
label_names = ['키','머리길이']

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(x,y)
# entropy = 데이터 집합에서 혼잡도(불순도)를 말하며, 클수록 불순도가 높음
pred = model.predict(x)
print()
print('훈련 정확도 ', model.score(x,y))

# 시각화
dot_data = tree.export_graphviz(model, feature_names=label_names, out_file='tree.jpg', filled=True, rounded=True, impurity=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# colors = ('red','orange')
# edges = collections.defaultdict(list)
#
# for e in graph.get_edge_list():
# 	edges[e.get_source()].append(int(e.get_destination()))
#
# for e in edges:
# 	edges[e].sort()
# 	for i in range(2):
# 		dest = graph.get_node(str(edges[e][i]))[0]
# 		dest.set_fillcolor(colors[i])
#
# graph.write_png('cla6tree.png')



