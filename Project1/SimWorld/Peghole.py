class Peghole:
    def __init__(self, position, filled):
        self.row = position[0]
        self.column = position[1]
        self.filled = filled
        self.neighbours = {}

    def add_neighbour(self, direction, peghole):
        self.neighbours[direction] = peghole

    def find_neighbour_direction(self, neighbour_row, neighbour_column):
        direction = ""
        if neighbour_column - self.column == -1:
            direction += "right"
        elif neighbour_column[1] - self.column[1] == 1:
            direction += "left"

        if neighbour_row[0] - self.row[0] == -1:
            direction += "up"
        elif neighbour_row[0] - self.row[0] == 1:
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

