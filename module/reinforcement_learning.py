import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy import linalg as LA

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
            The vertical dimension of the grid
        m : int, optional
            The horizontal dimension of the grid
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

        # a dictionary from states to available states
        self._state_to_states_dic = self._state_dic
        for i in self._state_dic.keys():
            self._state_to_states_dic[i] = [self.get_next_state_from_action(i,j)  for j in self._state_dic[i]]

        # all states except the blocked and the exit states
        self._available_states = list(set(range(0, n * m)) - set(maze.blocked_states))
        self._available_states = list(set(self._available_states) - set(maze.exit_states))

        self._curr_utility = []
        self._start = True

        self._exit_states = maze.exit_states
        self._rewards = np.zeros(n * m + 1)
        self._rewards[range(n * m + 1)] = -0.05
        self._rewards[self._exit_states[0]] = 1
        self._rewards[self._exit_states[1]] = -1
        self._rewards[n * m] = 0

        self._curr_policy = np.zeros(n * m + 1)

    def get_next_state_from_action(self, curr_state:int, action:int):
        if action == 0:
            return curr_state-self.m
        elif action == 1:
            return curr_state+1
        elif action == 2:
            return curr_state+self.m
        else:
            return curr_state-1

    def learn(self):
        # Contains learned policies per episode
        learnt_policies = []

        # Randomly chose a policy at the beginning by checking
        # the allowed actions per state using the state dictionary.
        policy_0 = np.zeros(self.n * self.m + 1)
        for state in self._state_dic:
            policy_0[state] = random.choice(self._state_dic[state])
        print("Starting policy:\n", policy_0)
        learnt_policies.append(policy_0)

        self._curr_policy = policy_0

        updated_state_index = 1
        policy_new = self.__policy_iter_step(policy_0, updated_state_index)
        print("Policy iter:\n", policy_new)
        while not np.array_equal(policy_0, policy_new):
            updated_state_index += 1
            policy_0 = policy_new
            learnt_policies.append(policy_0)
            policy_new = self.__policy_iter_step(np.array(policy_0), updated_state_index%len(self._available_states))
            print("Policy iter:\n", policy_new)

        learnt_policies = np.array(learnt_policies)
        #disregard the game-over state
        learnt_policies = learnt_policies[:,:-1]

        # Plot an animation containing all episodes
        mat, fig = self._setup_grid_animation(learnt_policies)
        ani = animation.FuncAnimation(fig, self._animate, frames=learnt_policies.shape[0],
                                      fargs=(mat, learnt_policies),
                                      interval=500)
        ani.save('animation_policies.gif')

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
    def _utility_from_policy(self, pol, updated_state_index):
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
        classical_result = np.linalg.solve(np.identity(self.n * self.m + 1) - self.gamma * tprob_pol, self._rewards)

        if self._start:
            self._curr_utility = classical_result
            self._start = False
        else:
            self._curr_utility[updated_state_index] = classical_result[updated_state_index]
            for neighbour in self._state_to_states_dic[updated_state_index]:
                if neighbour in self._available_states:
                    self._curr_utility[neighbour] = classical_result[neighbour]

        return self._curr_utility

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
    def __policy_iter_step(self, policy, updated_state_index):
        """
        Parameters
        ----------
        policy : ndarray
                 The current policy.

        Returns
        -------
        list : The learned policy from the iteration.

        """
        return self._policy_from_utility(self._utility_from_policy(policy, updated_state_index))


if __name__ == '__main__':
    rl_system = RL(n=4, m=4, gamma=0.90)
    rl_system.learn()
