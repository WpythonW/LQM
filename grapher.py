import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

file = pd.read_excel("Table.xlsx")
X = np.array(file.loc[:, "1-cosθ\n-"])
np.insert(X, 0, 0)
Y = np.array(file.loc[:, "Δλ = λ' - λ\nпм"])
np.insert(Y, 0, 0)


m = 1
X2 = np.transpose(np.array([np.ones(len(X)), X, X**2, X**3, X**4]))
X2 = np.transpose(np.transpose(X2)[:m+1])
TH = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X2), X2)), np.transpose(X2)), Y)


alphas = np.linalg.inv(np.dot(np.transpose(X2), X2))
al = alphas[-1][-1]

E = Y - np.dot(X2, TH)

norm_of_E = np.sqrt(sum(E**2))

print(norm_of_E)
t = TH[-1] * np.sqrt(40 - m - 1) / (np.sqrt(al) * norm_of_E)

print(f"{stats.t(df = len(X) - m - 1).ppf(0.025/2)} < {t} < {-stats.t(df = len(X) - m - 1).ppf(0.025/2)}")


X_continuous = np.linspace(min(X), max(X), 500)
Y_continuous = []
for i in X_continuous:
    summ = 0
    for j in range(m+1):
        summ += i**m * TH[j]
    Y_continuous.append(summ)



plt.xlabel('1-cosθ')
plt.ylabel("Δλ = λ' - λ")
real = plt.plot(X, Y, label='real')
mlq = plt.plot(X_continuous, Y_continuous, 'r', label='MLQ')
plt.legend(['real', 'MLQ'])
plt.show()
