import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from scipy import linalg

import module.qhll as qhhl


def plot_accuracy_graph(n):
    # Range of the parameter T to be tested: {512, ... , 8192}
    T_arr = [2 ** i for i in range(9, 14)]

    # Range of the condition number to be tested: {2, ... , 512}
    k_arr = [2 ** i for i in range(1, 10)]

    df = {"T": T_arr}
    for k in k_arr:
        residue_arr = []
        for t in T_arr:
            mean_residue = get_mean_residue(k, n, t)
            residue_arr.append(mean_residue)
        df[str(k)] = residue_arr

    create_plot(df)


def create_plot(dataframe):
    data = pd.DataFrame(data=dataframe)
    data = pd.melt(data, ['T'], var_name="Condition Number", value_name="Residue")
    subdata1 = data[~data["Condition Number"].isin(["64","128", "256", "512"])]
    subdata2 = data[data["Condition Number"].isin(["64","128", "256", "512"])]

    sns.set()
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    ax[0].set_xlim([250, 8500])
    ax[1].set_xlim([250, 8500])

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
    for i in range(0, 10):
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
    plot_accuracy_graph(4)
