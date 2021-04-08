from Project2.SimWorld.Board import Board
from Project2.SimWorld.BoardVisualization import BoardVisualization
import numpy as np
import copy


class StateManager:

    def __init__(self, size, visualization=False, gif_name='testing'):
        self.board = Board(size)
        self.visualization = visualization
        self.graph = BoardVisualization(self.board, 1000, gif_name+'.gif') if visualization else None
        self.current_player = 1

        self.sim_board = None
        self.current_sim_player = None

    # ---- REAL GAME ----
    def get_board_state(self):
        return self.board.get_board_state()

    # ---- Methods for both REAL GAME and ROLL OUT GAMES ----

    def get_current_player(self, rollout=False):
        return self.current_player if not rollout else self.current_sim_player

    def change_player(self, rollout=False):
        if not rollout:
            self.current_player = 1 if self.current_player == 2 else 2
        else:
            self.current_sim_player = 1 if self.current_sim_player == 2 else 2

    def perform_action(self, player, action, rollout=False):
        board = self.board if not rollout else self.sim_board
        peghole = board.table[action[0]][action[1]]
        peghole.add_peg(player)
        self.change_player(rollout=rollout)

        if self.visualization and not rollout: self.graph.change_node_color(peghole)

    def check_if_final_state(self, rollout=False):
        board = self.board if not rollout else self.sim_board
        if self.check_player1_win(board): return 1
        if self.check_player2_win(board): return 2
        return 0

    def check_player1_win(self, board):  # player1 = red, owns northeast and southwest (top and bottom in table)
        visited_nodes = []
        furthest = 0
        ne_row = [peghole for peghole in board.table[0] if peghole.filled == 1]
        for peghole in ne_row:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == board.size-1: break

        return furthest == board.size-1

    def check_player2_win(self, board):  # player2 = black, owns northwest and southeast (the sides in table)
        visited_nodes = []
        furthest = 0
        nw_col = [peghole for peghole in self.get_north_west_col(board) if peghole.filled == 2]
        for peghole in nw_col:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_col(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == board.size-1: break

        return furthest == board.size-1

    def get_north_west_col(self, board):
        nw = []
        for i in range(len(board.table)):
            nw.append(board.table[i][0])
        return nw

    def check_furthest_neighbour_row(self, peghole, visited_nodes):
        furthest = peghole.row
        if peghole.row == self.board.size-1: return peghole.row, visited_nodes

        for neighbour in peghole.get_players_neighbours(1):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes

    def check_furthest_neighbour_col(self, peghole, visited_nodes):
        furthest = peghole.column
        if peghole.column == self.board.size-1: return peghole.column, visited_nodes

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

    # ------ ROLL OUT GAMES -------
    def init_roll_out_game(self):
        self.sim_board = copy.deepcopy(self.board)
        self.current_sim_player = self.get_current_player()

    def get_roll_out_state(self):
        self.sim_state.get_board_state()




