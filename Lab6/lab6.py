import numpy as np
from random import gauss
import matplotlib.pyplot as plt

mu = [1, 3]
sigma = [0.1, 0.2]
N = [100, 100]
#кількість класів
M = 2
#вектори з ознаками. поки що 2
x = []
y = []
for i in range(M):
    x.append(np.random.normal(mu[i], sigma[i], N[i]))
    y.append(np.random.normal(mu[i], sigma[i], N[i]))
    plt.scatter(x[i], y[i])
    print("M1[x] = %f, M2[x] = %f\n",
          np.statistics.mean(x[i]), np.statistics.mean(y[i]))

plt.show()
