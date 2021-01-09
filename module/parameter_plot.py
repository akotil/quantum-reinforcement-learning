import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import linalg
import itertools
from module.maze import *
import random
import math

import module.qhll as qhhl
import module.reinforcement_learning as rl


def plot_gamma():
    for gamma in [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        rl_system = rl.RL(n=10, m=10, gamma=gamma)
        rl_system.learn()
        cond_numbers = rl_system.condition_numbers
        iterations = list(range(len(cond_numbers)))
        plt.plot(iterations, cond_numbers, label=r'$\gamma$: ' + str(gamma))

    plt.xlabel('Iteration instances')
    plt.ylabel('Condition number')

    plt.yscale('log')
    #   labelLines(plt.gca().get_lines(), zorder=2)
    plt.title("n=10")
    plt.legend(loc="center right", bbox_to_anchor=(1, 0.65))
    plt.savefig("./plots/parameter_plot2.png")
    plt.show()


def plot_gamma2():
    gamma_arr = [0.75, 0.8, 0.85, 0.9, 0.95]
    cond_arr_first = []
    cond_arr_last = []

    for gamma in gamma_arr:
        rl_system = rl.RL(n=10, m=10, gamma=gamma)
        rl_system.learn()
        cond_numbers = rl_system.condition_numbers

        cond_arr_first.append(cond_numbers[0])
        cond_arr_last.append(cond_numbers[-1])

    res1 = np.polyfit(gamma_arr, np.log(cond_arr_first), 1, w=np.sqrt(cond_arr_first))
    x = np.arange(0.75, 0.98, 0.01)
    plt.plot(x, np.exp(res1[1] + res1[0] * x))

    res2 = np.polyfit(gamma_arr, np.log(cond_arr_last), 1, w=np.sqrt(cond_arr_last))
    plt.plot(x, np.exp(res2[1] + res2[0] * x))

    print("f: ", cond_arr_first)
    print("l:", cond_arr_last)

    plt.scatter(gamma_arr, cond_arr_first, label="Initial iteration")
    plt.scatter(gamma_arr, cond_arr_last, label="Last iteration")

    plt.xlabel('Gamma values')
    plt.ylabel('Condition number')

    plt.xticks(gamma_arr)

    plt.title("n=10")
    plt.legend(loc="upper left")
    plt.savefig("./plots/parameter_plot2.png")
    plt.show()


def plot_t(epsilon=0.01, n=10):
    # First T-range to be tested: {10, ..., 600}
    T_arr_1 = [100 * i for i in range(1, 7)]
    # Second T-range: to be tested: {1000, ..., 10000}
    T_arr_2 = [1000 * i for i in range(1, 11)]

    # First k-range to be tested: {2, ... , 32}
    k_arr_1 = [2 ** i for i in range(1, 6)]
    # Second k-range to be tested: {64, ... , 1024}
    k_arr_2 = [2 ** i for i in range(6, 11)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xscale("log")

    fig.set_size_inches(12, 4)

    fig.suptitle("n=10, " r'$\epsilon$=' + str(epsilon))

    colors = plt.rcParams['axes.prop_cycle']()
    for k in k_arr_1:
        residue_arr = []
        for t in T_arr_1:
            mean_residue = get_mean_residue(k, n, t, epsilon)
            residue_arr.append(mean_residue)
        ax1.plot(T_arr_1, residue_arr, label=r'$\kappa$: ' + str(k), **next(colors))

    for k in k_arr_2:
        residue_arr = []
        for t in T_arr_2:
            mean_residue = get_mean_residue(k, n, t, epsilon)
            residue_arr.append(mean_residue)
        ax2.plot(T_arr_2, residue_arr, label=r'$\kappa$: ' + str(k), **next(colors))

    ax1.legend(loc="upper right")
    ax1.set_xlabel('T')
    ax1.set_ylabel('Residue')

    ax2.legend(loc="lower left")
    ax2.set_xlabel('T')
    ax2.set_ylabel('Residue')

    plt.savefig("./plots/t_plot_" + str(epsilon) + ".png")
    plt.show()


def get_mean_residue(condition_number, n, T, epsilon):
    # Repeat the hhl algorithm 10 times with 10 random matrices to get a mean residue
    residue_arr = []
    for i in range(0, 5):
        A = produce_matrix(condition_number, n)
        b = np.random.random(n)
        quantum_res = qhhl.hhl(A, b, epsilon, T)
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


def plot_cumulated_error():
    rl_system = rl.RL(n=10, m=10, gamma=0.95, quantum=True)
    rl_system.learn()
    q_results = rl_system.quantum_results

    rl_system = rl.RL(n=10, m=10, gamma=0.95, quantum=False)
    rl_system.learn()
    c_results = rl_system.classical_results

    iterations = list(range(len(q_results)))
    diff_arr = []
    for i in iterations:
        diff = list(np.array(c_results[i]) - np.array(q_results[i]))
        diff_arr.append(np.linalg.norm(diff))

    plt.plot(iterations, diff_arr)
    plt.xlabel('Iteration instances')
    plt.ylabel('Error')
    plt.title("n=10, " r'$\epsilon$=0.01')
    plt.savefig("./plots/error.png")
    plt.show()


def plot_k_n(randomized):
    conds = []
    dim = []
    for i in range(4, 15):
        for j in range(4, 15):
            if randomized:
                maze = Maze(i, j)
                states = list(range(i*j))
                maze.set_blocked_states(random.sample(states, math.ceil(i*j*0.1)))
                non_blocked_states = [x for x in states if x not in maze.blocked_states]
                maze.set_exit_states(random.sample(non_blocked_states, 2))
            else:
                maze = Maze(i, j)
                maze.set_blocked_states([0, i*j-1])
                maze.set_exit_states([math.floor(i*j/2), math.floor(i*j/3)])

            rl_system = rl.RL(n=i, m=j, maze=maze, gamma=0.95)
            rl_system.learn()
            cond = rl_system.condition_numbers[-1]
            conds.append(cond)
            dim.append(i * j + 1)
            print(i,j)
    log_func = np.polyfit(np.log(dim), conds, 1)
    x = np.arange(0, 200, 0.01)
    plt.plot(x, log_func[0]*np.log(x) + log_func[1])
    plt.ylim(100,)
    plt.ylabel("Condition number")
    plt.xlabel("Dimension")
    plt.scatter(dim, conds)
    plt.show()

if __name__ == '__main__':
    plot_k_n(True)
