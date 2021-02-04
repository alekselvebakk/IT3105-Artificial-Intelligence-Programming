from Project1.SimWorld.Board import Board


class TriangleBoard(Board):
    def __init__(self, size, empty):
        super().__init__(size, empty)

    def create_board(self, size, empty):
        return None

    def connect_triangle_neighbours(self, peghole):
        if peghole.column-1 >= 0: #Checks for left neighbour
            neighbour = self.board[peghole.row][peghole.column-1]
            self.connect_neighbours(peghole, neighbour)

            if peghole.row - 1 >= 0: #Checks for upleft neighbour
                neighbour = self.board[peghole.row-1][peghole.column-1]
                self.connect_neighbours(peghole, neighbour)

        if peghole.row - 1 >= 0 and len(self[peghole.row-1])-1 >= peghole.column-1: # Checks for up neighbour
            neighbour = self.board[peghole.row-1][peghole.column]
            self.connect_neighbours(peghole, neighbour)

        # Checks only for these neighbours as the last three neighbours have not been added yet
        # and will be updated later if they are added

