import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colors

from module.maze import *


class RL:
    """
    Class representing the policy learning procedure of an agent
    """

    def __init__(self, n=10, gamma=0.85):
        """
        Parameters
        ----------
        n : int, optional
            The dimension of the grid
        gamma : float, optional
            The discounting factor
        """

        self.n = n
        self.gamma = gamma

        maze = Maze(n)
        self._probs = maze.construct()
        self._state_dic = maze.state_dic
        self._rand_states = maze.corner_states + maze.edge_states

        exit_state = 20
        self._rewards = np.zeros(n ** 2)
        self._rewards[range(n ** 2)] = -1
        self._rewards[exit_state] = 100

        self._curr_policy = np.zeros(n ** 2)

    def learn(self):

        # Contains learned policies per episode
        learnt_policies = []

        # Randomly chose a policy at the beginning by checking
        # the allowed actions per state using the state dictionary.
        policy_0 = np.zeros(self.n ** 2)
        for state in self._state_dic:
            policy_0[state] = random.choice(self._state_dic[state])
        print("Starting policy:\n", policy_0)
        learnt_policies.append(policy_0)

        self._curr_policy = policy_0

        policy_new = self.__policy_iter_step(policy_0)
        print("Policy iter:\n", policy_new)
        while not np.array_equal(policy_0, policy_new):
            policy_0 = policy_new
            learnt_policies.append(policy_0)
            policy_new = self.__policy_iter_step(np.array(policy_0))
            print("Policy iter:\n", policy_new)

        learnt_policies = np.array(learnt_policies)

        # Plot an animation containing all episodes
        mat, fig = self._setup_grid_animation(learnt_policies)
        ani = animation.FuncAnimation(fig, self._animate, frames=learnt_policies.shape[0],
                                      fargs=(mat, learnt_policies),
                                      interval=500)
        ani.save('animation.gif')

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


if __name__ == '__main__':
    rl_system = RL(n=10)
    rl_system.learn()
