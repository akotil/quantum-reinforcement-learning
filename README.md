# quantum-reinforcement-learning

Where the policy iteration of reinforcement learning meets HLL.

**Requirements**

Python 3.6, numpy, scipy.

**How it works**

- maze.py builds an environment including probability transitions and actions.
- reinforcement_learning.py produces a learning process with a Maze object; no further configuration is needed apart from the user's specification on the grid dimension and learning rate. When the iterations finish, a grid animation including the action mapping from every state (i.e. grid field) is produced as 'animation.gif'.
- qhll.py is an implementation of the quantum version of solving linear systems of equations based on the following paper: https://arxiv.org/abs/0811.3171 . 

**Notes**

We are currently testing the HHL algorithm with different inputs. Till then, the policy iteration uses the classical algorithm which will soon be replaced by the HHL algorithm.
