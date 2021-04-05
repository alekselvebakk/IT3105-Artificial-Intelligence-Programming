from Project2.SimWorld.Peghole import Peghole

class Board:
    def __init__(self, size):
        self.table = []
        self.size = size
        self.num_pegs = 0
        self.create_board(size)

    def create_board(self, size):
        for row in range(size):
            self.table.append([])
            for column in range(size):
                peghole = Peghole(row, column, None)  # Filled = none/player1/player2
                self.connect_diamond_neighbours(peghole)
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

    def reset_board(self):
        self.num_pegs = 0
        for row in self.table:
            for peghole in row:
                peghole.filled = None
