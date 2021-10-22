import pandas as pd
from scipy.io import loadmat, savemat
import glob
import os
import numpy as np

# python3 process/DE-time-series-manu.py
# data/thermistor_chain/AGL_Abril_2019/Time_series/Time_series_Abril.mat


n = 10

data_path = '../data/thermistor_chain/AGL_Abril_2019/SBE56'
output_path = '../data/thermistor_chain/AGL_Abril_2019/Time_series'
output_fn = 'Time_Series_10.mat'

# os.chdir(path)
mat_files = glob.glob('{}/*.mat'.format(data_path))


order = ['0218', '5894', '0221', '5895', '0222', '0225', '0226',
         '5897', '5899', '0235', '5900', '5901', '5902', '5903']

mat_files = [file for x in order for file in mat_files if x in file]

data = [loadmat(file) for file in mat_files]

def extract_variable(variable, data):
    # print([thermistor[variable[:n]]] for thermistor in data)
    return [np.array(thermistor[variable][:n]) for thermistor in data]

tems = extract_variable('tem', data)
print(tems)
dates = extract_variable('dates', data)


def crop_thermistor_data(tems, dates):
    '''Crop all the thermistor data so that they all contain the same period.'''

    size_dates = [thermistor.size for thermistor in dates]
    min_length = min(size_dates)
    dates = np.array([thermistor[:min_length].reshape(-1) for thermistor in dates])
    size_dates = [thermistor.size for thermistor in dates]
    tems = np.array([thermistor[:min_length].reshape(-1) for thermistor in tems])
    return tems, dates

tems, dates = crop_thermistor_data(tems, dates)


pres = np.vstack(np.array([1, 8, 23, 28, 33, 43, 53, 63, 78, 96, 108, 126, 151, 176]))

lat = [43.789 for _ in range(len(dates[0]))]
lon = [-3.782 for _ in range(len(dates[0]))]

series = {'pres': pres, 'tems': tems, 'dates': dates[0], 'lat': lat, 'lon': lon}
savemat(os.path.join(output_path, output_fn), series)


a = loadmat(os.path.join(output_path, output_fn))
print(a)
