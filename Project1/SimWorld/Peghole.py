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
        if neighbour_column - self.column == -1:
            direction += "left"
        elif neighbour_column - self.column == 1:
            direction += "right"

        if neighbour_row - self.row == -1:
            direction += "up"
        elif neighbour_row - self.row == 1:
            direction += "down"

        return direction

    def remove_peg(self):
        self.filled = False

    def add_peg(self):
        self.filled = True

    def get_actions(self):
        actions = []
        for direction, peghole in self.neighbours.items():
            if peghole.filled and not peghole.neighbours[direction].filled:
                actions.append([[self.row, self.column], [peghole.row, peghole.column]])
        return actions

