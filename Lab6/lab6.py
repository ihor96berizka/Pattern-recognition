import numpy as np
from random import gauss
import matplotlib.pyplot as plt

#к-кість ознак
k = 4
#кількість класів
M = 3
#центри та розсіяння класів
mu = []
sigma = []
#кількість образів в одному кластері
n_points = 10000
#к-кісь образів в кожному класі
N = []

#3D матриця з всіма даними. 
data = []

#вектор із обчисленими крайніми точками класу по заданій ознаці
vidr = []

for i in range(M):
    mu.append([])
    sigma.append([])
    for j in range(k):
        mu[i].append(np.random.uniform(0, 10))
        sigma[i].append(np.random.uniform(0, 1))
    N.append(n_points)
    data.append(list())
    vidr.append(list())
        
#пройти по всіх кластерах
for i in range(M):
#пройти по всіх ознаках
    for j in range(k):
        #згенерувати вектор розміром N[i] випадкових чисел із заданим mu i sigma
        data[i].append(np.random.normal(mu[i][j], sigma[i][j], N[i]))

    if k == 2:
        plt.scatter(data[i][0], data[i][1])
    print("Cluster # ", i)
    for j in range(k):
        curr_mean = np.mean(data[i][j])
        curr_std = np.std(data[i][j])
        print("M[x", j, "] = ", curr_mean)
        print("D[x", j, "] = ", curr_std)
        x1 = curr_mean - 3 * curr_std
        x2 = curr_mean + 3 * curr_std
        vidr[i].append([x1, x2])
        print("vidr[",i, "|", j, "] =", vidr[i][j])

#пройти по всіх кластерах. попарно порахувати чи перетинаються класи по заданій ознаці
for i in range(M):
    for j in range(i+1, M):   
    #по всіх ознаках
        for g in range(k):
            print("Перевірка: ")
            #print("i = ", i, "j = ", j, "g = ", g)
            #print(vidr[i][g][1]," >=", vidr[j][g][0], "and ", vidr[j][g][1], ">= ",vidr[i][g][0])
            
            if vidr[i][g][1] >= vidr[j][g][0] and vidr[j][g][1] >= vidr[i][g][0]:
                print("Cluster # ", i, "and Cluster # ", j,
                      "Перетинаються по ознаці № ", g)
            else:
                print("Cluster # ", i, "and Cluster # ", j,
                      "НЕ Перетинаються по ознаці № ", g)
        
plt.show()
