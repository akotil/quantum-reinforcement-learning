import math
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colors

import module.qhll as qhhl
from module.maze import *


class RL:
    """
    Class representing the policy learning procedure of an agent
    """

    def __init__(self, n, m, maze=None, gamma=0.85, quantum=False, plot_enabled=False):
        """
        Parameters
        ----------
        n : int
            The vertical dimension of the grid
        m : int
            The horizontal dimension of the grid
        maze: Maze object, optional
            A custom maze
        gamma : float, optional
            The discounting factor
        quantum: bool, optional
            If quantum is set to True, the HHL Algorithm will be used for policy evaluation
        plot_enabled: bool, optional
            If plot_enabled is set to True, policies will be plotted as animation

        """

        self.n = n
        self.m = m
        self.gamma = gamma
        self.quantum = quantum
        self._plot_enabled = plot_enabled

        if maze == None:
            maze = Maze(n, m)

        self._probs = maze.construct()
        self._state_dic = maze.state_dic

        self._exit_states = maze.exit_states
        self._rewards = np.zeros(n * m + 1)
        self._rewards[range(n * m + 1)] = -0.05
        self._rewards[self._exit_states[0]] = 1
        self._rewards[self._exit_states[1]] = -1
        self._rewards[self._exit_states[2]] = 1
        self._rewards[n * m] = 0

        self._curr_utility = np.zeros(n * m + 1)
        self._utilities = []

        self.gamma_cond_dic = {}
        self.condition_numbers = []

        self.classical_results = []
        self.quantum_results = []

        self._converged = False

    def learn(self):

        # Randomly chose a policy at the beginning by checking
        # the allowed actions per state using the state dictionary.
        policy_0 = np.zeros(self.n * self.m + 1)
        for state in self._state_dic:
            policy_0[state] = random.choice(self._state_dic[state])

        # for testing purposes
        #      policy_0 = [1, 2, 1, 2, 1, 1, 3, 3, 3, 3, 1, 2, 0, 0, 2, 0, 1, 0, 1, 0, 1, 3, 3, 1,
        #                  1, 1, 0, 3, 1, 3, 3, 1, 0, 3, 0, 1, 0, 1, 3, 2, 1, 3, 3, 0, 0, 1, 2, 1,
        #                  0, 2, 0, 1, 2, 3, 0, 3, 0, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 2, 0, 3, 1, 3,
        #                  1, 0, 1, 3, 1, 3, 2, 2, 2, 0, 0, 2, 2, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 1,
        #                  3, 0, 3, 3, 0]

        learnt_policies = []
        learnt_policies.append(policy_0)
        curr_policy = policy_0
        while not self._converged:
            print(curr_policy)
            learnt_policies.append(curr_policy)
            curr_policy = self.__policy_iter_step(np.array(curr_policy))

        if self._plot_enabled:
            learnt_policies = np.array(learnt_policies)
            learnt_policies = learnt_policies[:, :-1]
            self._plot_animation(learnt_policies)
            self._plot_utilities()

    def _plot_animation(self, learnt_policies):
        # Plot an animation containing all episodes
        mat, fig = self._setup_grid_animation(learnt_policies)
        ani = animation.FuncAnimation(fig, self._animate, frames=learnt_policies.shape[0],
                                      fargs=(mat, learnt_policies),
                                      interval=500)
        ani.save("./module/plots/animation_policies.gif")

    def _setup_grid_animation(self, pol):
        """
        Sets up the animation grid.

        Parameters
        ----------
        pol : ndarray
                Array of learned policies per episode.

        Returns
        -------
        mat: `~matplotlib.image.AxesImage`
                The first image corresponding to the initial random policy.
        fig: `~matplotlib.figure.Figure`

        """

        fig, ax = plt.subplots()
        extent = (0, self.n, self.n, 0)
        cmap = colors.ListedColormap(["firebrick", "midnightblue", "moccasin", "teal"])
        mat = ax.matshow(pol[0].reshape((self.n, self.n)), extent=extent, cmap=cmap, vmin=-0.5, vmax=3.5)

        cbar = plt.colorbar(mat, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(["up", "right", "down", "left"])
        cbar.outline.set_edgecolor('white')

        ax.yaxis.grid(color='w', linewidth=2)
        ax.xaxis.grid(color='w', linewidth=2)
        ax.set_xticks(range(0, self.n), minor=False)
        ax.set_yticks(range(0, self.n), minor=False)
        ax.set_frame_on(False)

        plt.axis([0, self.n, self.n, 0])

        return mat, fig

    def _animate(self, i, mat, policies):
        """
        This is the animation function which is called in every episode to plot the corresponding policy grid.
        The animation shows the learned state-action mapping per episode.

        Parameters
        ----------
        i : int
                Iteration index.
        mat : `~matplotlib.image.AxesImage`
                The animation image.
        policies: ndarray
                Learned policies from policy iterations. The corresponding policy is displayed per iteration.

        Returns
        -------
        `~matplotlib.image.AxesImage`

        """
        if i == 0:
            return mat
        else:
            mat.set_data(policies[i].reshape((self.n, self.n)))
            return mat

    def _plot_utilities(self):
        fig, ax = plt.subplots()
        utility_matrix = np.reshape(self._utilities[-1][:-1], (10, 10))
        extent = (0, self.n, self.m, 0)
        mat = ax.matshow(utility_matrix, cmap=plt.cm.get_cmap("Blues"), extent=extent)
        cbar = plt.colorbar(mat)
        cbar.outline.set_edgecolor('white')
        plt.savefig("./module/plots/utilities.png")
        plt.show()

    # (a) Policy Evaluation
    def _utility_from_policy(self, pol):
        """
        Parameters
        ----------
        pol : ndarray
                 The current policy.

        Returns
        -------
        The utility obtained by the current policy.

        """
        tprob_pol = np.array([self._probs[:, s, int(pol[s])] for s in range(self.n * self.m + 1)])
        self.condition_numbers.append(np.linalg.cond(np.identity(self.n * self.m + 1) - self.gamma * tprob_pol))

        if self.quantum:
            quantum_result = qhhl.hhl(np.identity(self.n * self.m + 1) - self.gamma * tprob_pol, self._rewards, 0.01,
                                      5000)
            self.quantum_results.append(quantum_result)
            result = quantum_result
        else:
            classical_result = np.linalg.solve(np.identity(self.n * self.m + 1) - self.gamma * tprob_pol, self._rewards)
            self._utilities.append(classical_result)
            self.classical_results.append(classical_result)
            result = classical_result

        self._converged = np.allclose(self._curr_utility, result)
        self._curr_utility = result
        return result

    # (b) Policy Improvement Step
    def _policy_from_utility(self, utility):
        """
        Parameters
        ----------
        utility : ndarray
                 The current utility.

        Returns
        -------
        list: The learned policy from the iteration.

        """
        policy = []
        # Iterate over the current states
        for scurr in range(self._probs.shape[1]):
            state_sum = []
            # Iterate over the actions
            for act in range(self._probs.shape[2]):
                action_sum = 0
                # Iterate over the next states
                for snext in range(self._probs.shape[0]):
                    action_sum += self._probs[snext][scurr][act] * utility[snext]
                state_sum.append(action_sum)
            max_action = np.argmax(state_sum)
            policy.append(max_action)

        return policy

    # (a) + (b)
    def __policy_iter_step(self, policy):
        """
        Parameters
        ----------
        policy : ndarray
                 The current policy.

        Returns
        -------
        list : The learned policy from the iteration.

        """
        return self._policy_from_utility(self._utility_from_policy(policy))


if __name__ == '__main__':
    rl_system = RL(n=10, m=10, gamma=0.95)
    rl_system.learn()
