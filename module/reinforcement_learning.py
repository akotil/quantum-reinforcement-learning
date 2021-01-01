import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colors

from module.maze import *
import module.qhll as qhhl

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

        self._exit_states = maze.exit_states
        self._rewards = np.zeros(n * m + 1)
        self._rewards[range(n * m + 1)] = -0.05
        self._rewards[self._exit_states[0]] = 1
        self._rewards[self._exit_states[1]] = -1
        self._rewards[self._exit_states[2]] = 1
        self._rewards[n * m] = 0

        self._curr_policy = np.zeros(n * m + 1)

        self.gamma_cond_dic = {}
        self.condition_numbers = []

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

        policy_new = self.__policy_iter_step(policy_0)
        print("Policy iter:\n", policy_new)
        while not np.array_equal(policy_0, policy_new):
            policy_0 = policy_new
            learnt_policies.append(policy_0)
            policy_new = self.__policy_iter_step(np.array(policy_0))
            print("Policy iter:\n", policy_new)

        learnt_policies = np.array(learnt_policies)
        #disregard the game-over state
        learnt_policies = learnt_policies[:,:-1]

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
     #   classical_result = np.linalg.solve(np.identity(self.n * self.m + 1) - self.gamma * tprob_pol, self._rewards)
        quantum_result = qhhl.hhl(np.identity(self.n * self.m + 1) - self.gamma * tprob_pol, self._rewards, 0.01, 5000)
      #  print("classical: ", classical_result)
      #  print("quantum: ", quantum_result)
        return quantum_result

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