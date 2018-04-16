import numpy as np
from random import gauss
import matplotlib.pyplot as plt

mu = [1, 3]
sigma = [0.1, 0.2]
N = [100, 100]
M = 2
x = []
y = []
for i in range(M):
    x.append(np.random.normal(mu[i], sigma[i], N[i]))
    y.append(np.random.normal(mu[i], sigma[i], N[i]))
    plt.scatter(x[i], y[i])

plt.show()
