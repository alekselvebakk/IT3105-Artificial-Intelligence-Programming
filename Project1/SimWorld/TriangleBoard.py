from Project1.SimWorld.Board import Board
from Project1.SimWorld.Peghole import Peghole


class TriangleBoard(Board):
    def __init__(self, size, empty):
        super().__init__(size, empty)

    def create_board(self, size, empty):
        column_len = 1
        for row in range(size):
            self.table.append([])
            for column in range(column_len):
                filled = False if [row, column] in empty else True
                peghole = Peghole(row, column, filled)
                self.connect_triangle_neighbours(peghole)
                self.table[row].append(peghole)
            column_len += 1

    def connect_triangle_neighbours(self, peghole):
        if peghole.column-1 >= 0:  # Checks for left neighbour
            neighbour = self.table[peghole.row][peghole.column-1]
            self.connect_neighbours(peghole, neighbour)

            if peghole.row - 1 >= 0:  # Checks for upleft neighbour
                neighbour = self.table[peghole.row-1][peghole.column-1]
                self.connect_neighbours(peghole, neighbour)

        if peghole.row - 1 >= 0 and len(self.table[peghole.row-1]) > peghole.column:  # Checks for up neighbour
            neighbour = self.table[peghole.row - 1][peghole.column]
            self.connect_neighbours(peghole, neighbour)

        # Checks only for these neighbours as the last three neighbours have not been added yet
        # and will be updated later if they are added
