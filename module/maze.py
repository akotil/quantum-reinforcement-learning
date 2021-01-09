import numpy as np


class Maze:

    def __init__(self, n: int, m: int):
        # number of states = number of fields + game over state = n * m + 1
        # number of actions = |{up, right, down, left}| = 4
        self.probs = np.zeros((n * m + 1, n * m + 1, 4))
        self.n = n
        self.m = m
        self.constructed_states = []
        # state dictionary is a mapping from every state to the state's allowed actions
        self.state_dic = {}

        self.blocked_states = [12, 26, 43, 48, 56, 60, 82, 85]
        self.exit_states = [30, 40, 47]

        self.edge_states = []
        self.corner_states = []

    def set_blocked_states(self, blocked_indices):
        self.blocked_states = blocked_indices

    def set_exit_states(self, exit_indices):
        self.exit_states = exit_indices

    def construct(self):
        n = self.n
        m = self.m

        # states on the edges can have 3 different actions

        # upper edge
        dir_up = ["down", "left", "right"]
        self._construct_edge(dir_up, 1, m - 1)

        # lower edge
        dir_low = ["up", "left", "right"]
        self._construct_edge(dir_low, n * m - m + 1, n * m - 1)

        # left edge
        dir_le = ["up", "down", "right"]
        self._construct_edge(dir_le, m, n * m - m, step=m)

        # right edge
        dir_re = ["up", "down", "left"]
        self._construct_edge(dir_re, 2 * m - 1, n * m - 1, step=m)

        # states on the corners can have 2 different actions
        self._construct_corners()

        # states in the middle can have 4 different actions
        self._construct_middle()

        # the game over state is an absorbent state, has only the action 'stop'
        self._construct_game_over()

        return self.probs

    def _construct_edge(self, directions: list, start_state: int, end_state: int, step=1):
        for j in range(start_state, end_state, step):
            if j in self.exit_states:
                self.probs[self.n * self.m, j, :] = 1
                self.constructed_states.append(j)
                self.edge_states.append(j)
                self.state_dic[j] = [0, 1, 2, 3]
                continue
            elif j in self.blocked_states:
                continue
            for d1 in directions:
                for d2 in directions:
                    self._assign_probability(j, d2, d1, edge=True)
            self.constructed_states.append(j)
            self.state_dic[j] = [self._get_identifier(dir) for dir in directions]
            self.edge_states.append(j)

    def _construct_corners(self):
        n = self.n
        m = self.m
        directions = [["right", "down"], ["left", "down"], ["right", "up"], ["left", "up"]]
        corners = [0, m - 1, n * m - m, n * m - 1]

        for dir, state in zip(directions, corners):
            if state in self.exit_states:
                self.probs[n * m, state, :] = 1
                self.constructed_states.append(state)
                self.state_dic[state] = [0, 1, 2, 3]
                continue
            elif state in self.blocked_states:
                continue
            for d1 in dir:
                for d2 in dir:
                    self._assign_probability(state, d2, d1, corner=True)
            self.constructed_states.append(state)
            self.state_dic[state] = [self._get_identifier(d) for d in dir]

        self.corner_states = corners

    def _construct_middle(self):
        n = self.n
        m = self.m
        directions = ["left", "right", "down", "up"]
        all_states = range(0, n * m)
        remaining_states = [x for x in all_states if x not in self.blocked_states and x not in self.constructed_states]
        for j in remaining_states:
            if j in self.exit_states:
                self.probs[n * m, j, :] = 1
                self.state_dic[j] = [0, 1, 2, 3]
                continue
            for d1 in directions:
                for d2 in directions:
                    self._assign_probability(j, d2, d1)

            self.state_dic[j] = [self._get_identifier(dir) for dir in directions]

    def _construct_game_over(self):
        n = self.n
        m = self.m
        self.probs[n * m, n * m, :] = 1

    def _assign_probability(self, curr_state: int, direction: str, action: str, corner=False, edge=False):
        n = self.n
        if corner:
            probability = 0.2
        elif edge:
            probability = 0.1
        else:
            probability = 0.2 / 3

        dir = self._get_identifier(direction)
        act = self._get_identifier(action)

        # direction is a possible but not necessarily an intended action
        if dir == act:
            probability = 0.8

        self.probs[self._get_nstate_from_dir(curr_state, direction)][curr_state][act] = probability

    def _get_nstate_from_dir(self, curr_state, direction):
        m = self.m
        if direction == "up" and not curr_state - m in self.blocked_states:
            return curr_state - m
        elif direction == "right" and not curr_state + 1 in self.blocked_states:
            return curr_state + 1
        elif direction == "down" and not curr_state + m in self.blocked_states:
            return curr_state + m
        elif direction == "left" and not curr_state - 1 in self.blocked_states:
            return curr_state - 1
        else:
            return curr_state

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
