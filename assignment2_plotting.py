import numpy as np
from matplotlib import pyplot as plt

analytics_results = np.loadtxt("results.csv")

plt.semilogx(analytics_results[0,:], analytics_results[1,:])
plt.show()
