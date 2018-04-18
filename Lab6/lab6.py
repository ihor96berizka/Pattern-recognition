import numpy as np
from random import gauss
import matplotlib.pyplot as plt

#к-кість ознак
k = 2
#кількість класів
M = 3
#центри та розсіяння класів
mu = []
sigma = []
#кількість образів в одному кластері
n_points = 100
#к-кісь образів в кожному класі
N = []

#вектори з ознаками. поки що 2 - x ta y
data = []


#x1 = []
#x2 = []
#x3 = []

for i in range(M):
    mu.append([])
    sigma.append([])
    for j in range(k):
        mu[i].append(np.random.uniform(0, 10))
        sigma[i].append(np.random.uniform(0, 1))
    N.append(n_points)
    data.append(list())
        
#пройти по всіх кластерах
for i in range(M):
#пройти по всіх ознаках
    for j in range(k):
        data[i].append(np.random.normal(mu[i][j], sigma[i][j], N[i]))

    if k == 2:
        plt.scatter(data[i][0], data[i][1])
    print("Cluster # ", i)
    for j in range(k):
        print("M[x", j, "] = ", np.mean(data[i][j]))
        print("D[x", j, "] = ", np.std(data[i][j])) 

plt.show()
