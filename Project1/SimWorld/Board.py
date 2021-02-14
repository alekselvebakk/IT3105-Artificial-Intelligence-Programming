from SimWorld.Peghole import Peghole

class Board:
    def __init__(self, size, empty):
        self.table = []
        self.size = size
        self.empty = empty
        self.num_pegs = 0
        self.create_board(size, empty)

    def create_board(self, size, empty):
        return []

    def reset_board(self):
        self.num_pegs = 0
        for row in self.table:
            for peghole in row:
                peghole.filled = True if [peghole.row, peghole.column] not in self.empty else False
                if peghole.filled: self.num_pegs += 1

    def connect_neighbours(self, peghole_1, peghole_2):
        peghole_1.add_neighbour(peghole_1.find_neighbour_direction(peghole_2.row, peghole_2.column), peghole_2)
        peghole_2.add_neighbour(peghole_2.find_neighbour_direction(peghole_1.row, peghole_1.column), peghole_1)

