from Project2.SimWorld.Board import Board
import numpy as np

class StateManager:

    def __init__(self, size):
        self.board = Board(size)

    # TODO: GET_ACTIONS: do we use probabilities to find action? Or do we find them on our own?

    def perform_action(self, player, action):
        self.board[action[0]][action[1]].add_peg(player)

    def check_if_final_state(self):
        if self.check_player1_win(): return 1
        if self.check_player2_win(): return 2
        return 0

    def check_player1_win(self):  # player1 = black, owns northwest and southeast (the sides in table)
        visited_nodes = []
        furthest = 0
        nw_col = [peghole for peghole in self.get_north_west_col() if peghole.filled == 1]
        for peghole in nw_col:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_col(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == int(np.sqrt(self.board.size)): break

            return furthest == int(np.sqrt(self.board.size))

    def get_north_west_col(self):
        nw = []
        for i in range(len(self.board)):
            nw.append(self.board[i][0])
        return nw

    def check_player2_win(self):  # player2 = red, owns northeast and southwest (top and bottom in table)
        visited_nodes = []
        furthest = 0
        ne_row = [peghole for peghole in self.board[0] if peghole.filled == 2]
        for peghole in ne_row:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == int(np.sqrt(self.board.size)): break

        return furthest == int(np.sqrt(self.board.size))

    def check_furthest_neighbour_col(self, peghole, visited_nodes):
        furthest = peghole.col
        if peghole.col == int(np.sqrt(self.board.size)): return peghole.col

        for neighbour in peghole.get_players_neighbours(1):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_col(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes

    def check_furthest_neighbour_row(self, peghole, visited_nodes):
        furthest = peghole.row
        if peghole.row == int(np.sqrt(self.board.size)): return peghole.row, visited_nodes

        for neighbour in peghole.get_players_neighbours(2):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes
