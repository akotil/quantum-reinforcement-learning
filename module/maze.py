import numpy as np


class Maze:

    # n : number of states
    # a : number of actions
    def __init__(self, n: int):
        self.probs = np.zeros((n ** 2, n ** 2, 4))
        self.n = n
        self.constructed_states = []
        # state dictionary is a mapping from every state to the state's allowed actions
        self.state_dic = {}

        self.edge_states = []
        self.corner_states = []

    def construct(self):
        n = self.n
        # states on the edges can have 3 different actions

        # upper edge
        dir_uppe = ["down", "left", "right"]
        self._construct_edge(dir_uppe, 1, n - 1)

        # lower edge
        dir_lowe = ["up", "left", "right"]
        self._construct_edge(dir_lowe, n ** 2 - n + 1, n ** 2 - 1)

        # left edge
        dir_le = ["up", "down", "right"]
        self._construct_edge(dir_le, n, n ** 2 - n, step=n)

        # right edge
        dir_re = ["up", "down", "left"]
        self._construct_edge(dir_re, 2 * n - 1, n ** 2 - 1, n)

        # states on the corners can have 2 different actions
        self._construct_corners()

        # states in the middle can have 4 different actions
        self._construct_middle()

        return self.probs

    def _construct_edge(self, directions: list, start_state: int, end_state: int, step=1):
        n = self.n
        for j in range(start_state, end_state, step):
            for d1 in directions:
                for d2 in directions:
                    self._assign_probability(j, d2, d1, edge=True)
            self.constructed_states.append(j)
            self.state_dic[j] = [self._get_identifier(dir) for dir in directions]
            self.edge_states.append(j)

    def _construct_corners(self):
        n = self.n
        directions = [["right", "down"], ["left", "down"], ["right", "up"], ["left", "up"]]
        corners = [0, n - 1, n ** 2 - n, n ** 2 - 1]

        for dir, state in zip(directions, corners):
            for d1 in dir:
                for d2 in dir:
                    self._assign_probability(state, d2, d1, corner=True)
            self.constructed_states.append(state)
            self.state_dic[state] = [self._get_identifier(d) for d in dir]

        self.corner_states = corners

    def _construct_middle(self):
        n = self.n
        directions = ["left", "right", "down", "up"]
        all_states = range(0, n ** 2)
        remaining_states = list(set(all_states) - set(self.constructed_states))
        for j in remaining_states:
            for d1 in directions:
                for d2 in directions:
                    self._assign_probability(j, d2, d1)

            self.state_dic[j] = [self._get_identifier(dir) for dir in directions]

    def _assign_probability(self, curr_state: int, direction: str, action: str, corner=False, edge=False):
        n = self.n
        if (corner):
            probability = 0.2
        elif (edge):
            probability = 0.1
        else:
            probability = 0.2 / 3

        dir = self._get_identifier(direction)
        act = self._get_identifier(action)

        # direction is a possible but not necessarily an intended action
        if dir == act:
            probability = 0.8

        if direction == "up":
            self.probs[curr_state - n][curr_state][act] = probability
        elif direction == "down":
            self.probs[curr_state + n][curr_state][act] = probability
        elif direction == "right":
            self.probs[curr_state + 1][curr_state][act] = probability
        else:
            self.probs[curr_state - 1][curr_state][act] = probability

    @staticmethod
    def _get_identifier(direction: str) -> int:
        if direction == "up":
            return 0
        elif direction == "right":
            return 1
        elif direction == "down":
            return 2
        else:
            return 3
