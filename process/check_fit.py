import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

name = 'Time_series_prueba'
# data = loadmat(name + '.mat')
# fit = np.loadtxt('Time_series_prueba-M.aju', delimiter=' ', skiprows=3)
# print(fit)
fit = pd.read_csv('Time_series_prueba-M.aju', sep='   ', header=1, skiprows=3, engine='python')
print(fit)
