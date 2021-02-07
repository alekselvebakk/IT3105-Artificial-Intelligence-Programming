from Project1.SimWorld.Board import Board
from Project1.SimWorld.Peghole import Peghole


class DiamondBoard(Board):
    def __init__(self, size, empty):
        super().__init__(size, empty)

    def create_board(self, size, empty):
        for row in range(size):
            self.table.append([])
            for column in range(size):
                if [row, column] in empty:
                    filled = False
                else:
                    filled = True
                    self.num_pegs += 1
                peghole = Peghole(row, column, filled)
                self.connect_diamond_neighbours(peghole)
                self.table[row].append(peghole)

    def connect_diamond_neighbours(self, peghole):
        if peghole.column-1 >= 0:  # Checks for left neighbour
            neighbour = self.table[peghole.row][peghole.column-1]
            self.connect_neighbours(peghole, neighbour)

        if peghole.row - 1 >= 0:  # Checks for up neighbour
            neighbour = self.table[peghole.row-1][peghole.column]
            self.connect_neighbours(peghole, neighbour)

            if peghole.column+1 < self.size:  # Checks for rightup neighbour
                neighbour = self.table[peghole.row-1][peghole.column+1]
                self.connect_neighbours(peghole, neighbour)

        # Checks only for these neighbours as the last three neighbours have not been added yet
        # and will be updated later if they are added
