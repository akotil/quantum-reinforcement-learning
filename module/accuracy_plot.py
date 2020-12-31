import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from scipy import linalg

import module.qhll as qhhl


def plot_accuracy_graph(n):
    # First T-range to be tested: {10, ..., 1000}
    T_arr_1 = [100 * i for i in range(1, 11)]
    # Second T-range: to be tested: {1000, ..., 10000}
    T_arr_2 = [1000 * i for i in range(1, 11)]

    # First k-range to be tested: {2, ... , 32}
    k_arr_1 = [2 ** i for i in range(1, 6)]
    # Second k-range to be tested: {64, ... , 1024}
    k_arr_2 = [2 ** i for i in range(6, 11)]

    k_arr = k_arr_1 + k_arr_2
    T_arr = T_arr_1 + T_arr_2
    df = {"T": T_arr}
    for k in k_arr_1:
        residue_arr = []
        for t in T_arr_1:
            mean_residue = get_mean_residue(k, n, t)
            residue_arr.append(mean_residue)
        df[str(k)] = residue_arr

    for k in k_arr_2:
        residue_arr = []
        for t in T_arr_2:
            mean_residue = get_mean_residue(k, n, t)
            residue_arr.append(mean_residue)
        df[str(k)] = residue_arr

    create_plot(df)


def create_plot(dataframe):
    data = pd.DataFrame(data=dataframe)
    data = pd.melt(data, ['T'], var_name="Condition Number", value_name="Residue")
    subdata1 = data[~data["Condition Number"].isin(["64","128", "256", "512", "1024"])]
    subdata2 = data[data["Condition Number"].isin(["64","128", "256", "512", "1024"])]

    sns.set()
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    ax[0].set_xlim([0, 1000])
    ax[1].set_xlim([1000, 10000])

    splot1 = sns.lineplot(x='T', y='Residue', hue="Condition Number",
                          data=subdata1, palette="mako", ax=ax[0])
    splot1.set(yscale="log")
    splot2 = sns.lineplot(x='T', y='Residue', hue="Condition Number",
                          data=subdata2, palette="mako", ax=ax[1])
    splot2.set(yscale="log")

    plt.savefig("./accuracy_plot.png", bbox_inches='tight')
    fig.show()


def get_mean_residue(condition_number, n, T):
    # Repeat the hhl algorithm 10 times with 10 random matrices to get a mean residue
    residue_arr = []
    for i in range(0, 2):
        A = produce_matrix(condition_number, n)
        b = np.random.random(n)
        quantum_res = qhhl.hhl(A, b, 0.01, T)
        residue = np.linalg.norm(np.dot(A, quantum_res) - b)
        residue_arr.append(residue)

    return statistics.mean(residue_arr)


def produce_matrix(condition_number, n):
    A = np.random.random((n, n))
    u, s, v = sp.linalg.svd(A)
    s = np.linspace(1, condition_number, endpoint=True, num=n)
    s = np.diag(s)

    A = u @ s @ v
    return A


if __name__ == '__main__':
    plot_accuracy_graph(10)
