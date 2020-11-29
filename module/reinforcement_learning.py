import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colors

from module.maze import *


class RL:
    """
    Class representing the policy learning procedure of an agent
    """

    def __init__(self, n=10, m=10, gamma=0.85):
        """
        Parameters
        ----------
        n : int, optional
            The dimension of the grid
        gamma : float, optional
            The discounting factor
        """

        self.n = n
        self.m = m
        self.gamma = gamma

        maze = Maze(n, m)
        self._probs = maze.construct()
        self._state_dic = maze.state_dic
        self._rand_states = maze.corner_states + maze.edge_states

        self._exit_states = maze.exit_states
        self._rewards = np.zeros(n * m + 1)
        self._rewards[range(n * m + 1)] = -0.04
        self._rewards[self._exit_states[0]] = 1
        self._rewards[self._exit_states[1]] = -1
        self._rewards[n*m] = 0

        self._curr_policy = np.zeros(n * m + 1)

    def learn(self):
        utility_arr = np.array(self.value_iteration())

        # Plot an animation containing all episodes, disregarding the game-over state
        utility_arr = utility_arr[:,:-1]
        mat, fig = self._setup_grid_animation(utility_arr)
        ani = animation.FuncAnimation(fig, self._animate, frames=utility_arr.shape[0],
                                      fargs=(mat, utility_arr),
                                      interval=500)
        ani.save('animation_utilities.gif')

    def _setup_grid_animation(self, utility):

        fig, ax = plt.subplots()
        extent = (0, self.m, self.n, 0)
        mat = ax.matshow(utility[0].reshape((self.n, self.m)), extent=extent, cmap='GnBu')

        cbar = plt.colorbar(mat)
        cbar.outline.set_edgecolor('white')

        ax.yaxis.grid(color='w', linewidth=2)
        ax.xaxis.grid(color='w', linewidth=2)
        ax.set_xticks(range(0, self.m), minor=False)
        ax.set_yticks(range(0, self.n), minor=False)
        ax.set_frame_on(False)

        plt.axis([0, self.m, self.n, 0])

        return mat, fig

    def _animate(self, i, mat, utilities):
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
            mat.set_data(utilities[i].reshape((self.n, self.m)))
            return mat

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
        tprob_pol = np.array([self._probs[:, s, int(pol[s])] for s in range(self.n ** 2)])
        classical_result = np.linalg.solve(np.identity(self.n ** 2) - self.gamma * tprob_pol, self._rewards)
        return classical_result

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

        # For every action in the new policy, check if the action is allowed
        # When not, repeat the old action
        for a, i in zip(policy, range(self.n ** 2)):
            if a not in self._state_dic[i] and i in self._rand_states:
                policy[i] = self._curr_policy[i]

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

    def value_iter_step(self, u):

        # The sum over probabilities
        value_vector = []
        # Iterate over the current states
        for k in range(self._probs.shape[1]):
            iterations = []
            # Iterate over the actions
            for l in range(self._probs.shape[2]):
                sum = 0
                # Iterate over the next states
                for m in range(self._probs.shape[0]):
                    sum += self._probs[m][k][l] * u[m]
                iterations.append(sum)
            max_value = np.amax(iterations)
            value_vector.append(self.gamma * max_value)

        return np.add(value_vector, self._rewards)

    def value_iteration(self, epsilon=1e-14, maxsteps=5000):
        """Value iteration algorithm."""
        u = np.zeros(self._probs.shape[1])
        utility_iterations = []
        for i in range(maxsteps):
            unext = self.value_iter_step(u)
            utility_iterations.append(unext)
            diff = np.linalg.norm(unext - u)
            u = unext
            if diff <= epsilon * (1 - self.gamma) / self.gamma:
                print('value iteration with epsilon={} completed after {} iterations'.format(epsilon, i))
                print(u)
                return utility_iterations
        return u


if __name__ == '__main__':
    rl_system = RL(n=4, m=4, gamma=1)
    rl_system.learn()
