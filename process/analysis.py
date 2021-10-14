import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os
import sys

def import_data(file_name):
    file_name = 'Time_series_Abril_R.csv'
    # path = '../Results/' + file_name
    # df_fit = pd.read_csv(path)
    data = loadmat(
        '../data/thermistor_chain/AGL_Abril_2019/Time_series/Time_series_Abril.mat')

    tems = data['tems']
    pres = data['pres']
    dates = data['dates']

    return tems, pres, dates

def fit_fun(z, df):
    D1, b2, c2 = df['D1m'], df['c2m'], df['a2m']
    b3, a2, a1 = df['a3m'], df['b2m'], df['a1m']
    print(D1, b2, c2, b3, a2, a1)

    pos = np.where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) *(b2 + (z - D1) *c2)
    return a1 + pos *(b3 *(z - D1) + a2 *(np.exp(zaux) - 1.0))


def plot_fit_variable(variable, interval=None, save=False):
    var = df_fit[variable][::interval]
    dates = df_fit['Dates'][::interval]

    fig, ax = plt.subplots()
    ax.scatter(dates, var, s=8)
    ax.set_xlabel('Date')
    ax.set_ylabel(variable)
    fig.tight_layout()
    fig.savefig('oli.pdf')
    plt.show()


def plot_profile_fit(number):
    temp = tems[:, number]

    zz = np.linspace(0, 200, 300)
    fig, ax = plt.subplots()
    ax.scatter(temp, pres, marker='o', c='tab:red')
    ax.axhline(df_fit.iloc[number, 3], c='grey', ls='--')
    ax.set_ylim(pres[-1] + 10, 0)

    ax.plot(fit_fun(zz, df_fit.iloc[number]), zz)
    ax.set_xlabel('Temperatura (ºC)')
    ax.set_ylabel('Profundidad (mb)')

    plt.show()


def compute_physical_parameters(df):
    alfa = 0.5
    b2 = df['b2m']
    c2 = df['c2m']
    D = df['D1m']
    l = 2 *c2 /b2**2
    Delta = -  b2 /2 /c2 *(1 -(1 -2 *l *np.log(alfa)**0.5))
    temp_gradient = 1 /Delta *(fit_fun(D, df) - fit_fun(D + Delta, df))


# plot_fit_variable('a2m', 1000)

def plot_worst_fit_profiles(df, number):
    em = df['em']
    profiles = df.index[df['em'] > 2]
    print(profiles)
    for profile in profiles:
        plot_profile_fit(profile)

# plot_worst_fit_profiles(df_fit, 4)

if '__name__' == '__main__':
    profiles = (np.linspace(0, np.size(dates) - 1, 10, dtype='int'))
    print(profiles)
    for profile in profiles:
        print(profile)
        plot_profile_fit(profile)
