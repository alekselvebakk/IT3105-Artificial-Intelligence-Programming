from Project1.SimWorld.Peghole import Peghole

class Board:
    def __init__(self, size, empty, ):
        self.size = size
        self.empty = empty
        self.board = self.create_board(size, empty)

    def create_board(self, size, empty):
        return []

    def connect_neighbours(self, peghole_1, peghole_2):
        peghole_1.add_neighbour(peghole_1.find_neighbour_direction(peghole_2.row, peghole_2.column), peghole_2)
        peghole_2.add_neighbour(peghole_2.find_neighbour_direction(peghole_1.row, peghole_1.column), peghole_1)