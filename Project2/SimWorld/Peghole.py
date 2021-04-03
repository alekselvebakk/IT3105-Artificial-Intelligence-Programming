class Peghole:
    def __init__(self, row, column, filled):
        self.row = row
        self.column = column
        self.filled = filled
        self.neighbours = {}

    def add_neighbour(self, direction, peghole):
        self.neighbours[direction] = peghole

    def find_neighbour_direction(self, neighbour_row, neighbour_column):
        direction = ""
        if neighbour_column - self.column <= -1:
            direction += "left"
        elif neighbour_column - self.column >= 1:
            direction += "right"

        if neighbour_row - self.row <= -1:
            direction += "up"
        elif neighbour_row - self.row >= 1:
            direction += "down"
        return direction

    def remove_peg(self):
        self.filled = None

    def add_peg(self, player):
        self.filled = player

    def get_not_up_neighbours(self):
        nu_neighbours = []
        for peghole in self.neighbours:
            if 'up' not in self.neighbours[peghole]:
                nu_neighbours.append(peghole)
        return nu_neighbours
