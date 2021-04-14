from SimWorld.Peghole import Peghole
from SimWorld.BoardVisualization import BoardVisualization

class Board:

    def __init__(self, size, visualization=False, board_gif_name='board.gif'):
        self.table = []
        self.size = size
        self.create_board(size)
        self.player = 1

        self.visualize = visualization
        self.gif_name = board_gif_name
        self.graph = BoardVisualization(self.table, 1000, board_gif_name) if visualization else None

    def create_board(self, size):
        for row in range(size):
            self.table.append([])
            for column in range(size):
                peghole = Peghole(row, column, filled=0)  # Filled = 0/1/2
                self.connect_neighbours(peghole)
                self.table[row].append(peghole)

    def connect_neighbours(self, peghole):
        if peghole.column-1 >= 0:  # Checks for left neighbour
            neighbour = self.table[peghole.row][peghole.column-1]
            self.add_neighbours(peghole, neighbour)

        if peghole.row - 1 >= 0:  # Checks for up neighbour
            neighbour = self.table[peghole.row-1][peghole.column]
            self.add_neighbours(peghole, neighbour)

            if peghole.column+1 < self.size:  # Checks for rightup neighbour
                neighbour = self.table[peghole.row-1][peghole.column+1]
                self.add_neighbours(peghole, neighbour)

    def add_neighbours(self, peghole_1, peghole_2):
        peghole_1.add_neighbour(peghole_1.find_neighbour_direction(peghole_2.row, peghole_2.column), peghole_2)
        peghole_2.add_neighbour(peghole_2.find_neighbour_direction(peghole_1.row, peghole_1.column), peghole_1)

    def get_board_state(self):
        state = ""
        for row in self.table:
            for peghole in row:
                state += str(peghole.filled)
        return str(self.player)+state

    def set_player(self, player):
        self.player = int(player)

    def update_board(self, state):
        for i in range(len(state)):
            row = i // self.size
            col = i % self.size
            self.table[row][col].filled = int(state[i])

    def reset_board(self):
        for row in self.table:
            for peghole in row:
                peghole.filled = 0
        self.set_player(1)

    def start_visualisation(self, name='board.gif'):
        self.visualize = True
        self.graph = BoardVisualization(self.table, 1000, name)

    def stop_visualization(self):
        self.visualize = False
        self.graph = None
