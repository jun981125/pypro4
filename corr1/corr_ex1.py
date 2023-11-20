# 공분산/ 상관계수 연습
import numpy as np
import fileinput

print(np.cov(np.arange(1,6), np.arange(2,7)))	# 2.5

print(np.cov(np.arange(1,6), (3,3,3,3,3)))	# 0

print(np.cov(np.arange(1,6), np.arange(6,1,-1))) # -2.5

x = [8,3,6,6,9,4,3,9,4,3]
print('평균 : ', np.mean(x), ' 분산 : ',np.var(x))
y = [6,2,4,6,9,5,1,8,4,5]
print('평균 : ', np.mean(y), ' 분산 : ',np.var(y))

import matplotlib.pyplot as plt

#plt.scatter(x,y)
#plt.show()

#print('x, y의 공분산 : ',np.cov(x,y))
print('x, y의 공분산 : ',np.cov(x,y)[0,1])
#print('x, y의 상관계수 : ',np.corrcoef(x,y))
print('x, y의 상관계수 : ',np.corrcoef(x,y)[0,1])

# 공분산과는 다르게 상관계수는 패턴만 일치하다면 값은 일정
# 피어슨 상관계수 :
# x, y의 공분산 :  51.11111111111111
# x, y의 상관계수 :  0.847935270867641

# x, y의 공분산 :  5.111111111111111
# x, y의 상관계수 :  0.847935270867641

print('0-------------0')
m = [-3, -2,-1, 0 , 1 , 2 , 3]
n= [9,4,1,0,1,4,9]
plt.plot(m,n)
plt.show()


print('m, n의 공분산 : ',np.cov(m,n)[0,1])
print('m, n의 상관계수 : ',np.corrcoef(m,n)[0,1])

# 곡선의 경우에는 상관계수는 의미 없음. 1차 함수에서만 의미가 있음
