from Project2.SimWorld.Board import Board
from Project2.SimWorld.BoardVisualization import BoardVisualization
import numpy as np

class StateManager:

    def __init__(self, size, visualization=False):
        self.board = Board(size)
        self.visualization = visualization
        self.graph = BoardVisualization if visualization else None

    def get_board_state(self):
        state = ""
        for row in self.board.table:
            for peghole in row:
                state += str(peghole.filled)
        return state

    def perform_action(self, player, action):
        peghole = self.board[action[0]][action[1]]
        peghole.add_peg(player)

        if self.visualization: self.graph.change_node_color(peghole)

    def check_if_final_state(self):
        if self.check_player1_win(): return 1
        if self.check_player2_win(): return 2
        return 0

    def check_player1_win(self):  # player1 = red, owns northeast and southwest (top and bottom in table)
        visited_nodes = []
        furthest = 0
        ne_row = [peghole for peghole in self.board[0] if peghole.filled == 1]
        for peghole in ne_row:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == int(np.sqrt(self.board.size))-1: break

        return furthest == int(np.sqrt(self.board.size))-1

    def check_player2_win(self):  # player2 = black, owns northwest and southeast (the sides in table)
        visited_nodes = []
        furthest = 0
        nw_col = [peghole for peghole in self.get_north_west_col() if peghole.filled == 2]
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

    def check_furthest_neighbour_row(self, peghole, visited_nodes):
        furthest = peghole.row
        if peghole.row == int(np.sqrt(self.board.size))-1: return peghole.row, visited_nodes

        for neighbour in peghole.get_players_neighbours(1):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes

    def check_furthest_neighbour_col(self, peghole, visited_nodes):
        furthest = peghole.col
        if peghole.col == int(np.sqrt(self.board.size))-1: return peghole.col

        for neighbour in peghole.get_players_neighbours(2):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_col(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes

    def show_animation(self):
        if self.visualization: self.graph.show_graph_animation()

    def reset_board(self):
        self.board.reset_board()

        if self.visualization: self.graph.reset_board()
