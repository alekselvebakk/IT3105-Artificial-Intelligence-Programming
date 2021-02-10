from Project1.SimWorld.DiamondBoard import DiamondBoard
from Project1.SimWorld.TriangleBoard import TriangleBoard


class GameHandler:
    def __init__(self, board_type, size, empty):
        self.board = DiamondBoard(size, empty) if board_type == "diamond" else TriangleBoard(size, empty)

    def get_actions(self):
        actions = []
        for peghole in self.board.table:
            if peghole.filled:
                actions += peghole.get_actions()
        return actions

    def perform_action(self, action):
        self.move_peg(action)
        return self.calculate_reward(), self.get_board_state(), self.get_actions()

    def move_peg(self, action):
        start = self.board.table[int(action[0])][int(action[1])]
        end = self.board.table[int(action[2])][int(action[3])]

        start.remove_peg()
        end.add_peg()
        self.remove_middle_peg(start, end)

    def remove_middle_peg(self, start, end):
        self.board.num_pegs -= 1
        direction = start.find_neighbour_direction(end.row, end.column)
        middle = start.neighbours[direction]
        middle.remove_peg()

    def get_board_state(self):
        bn = ''
        for row in range(self.board.size):
            for peghole in self.board.table[row]:
                bn += '1' if peghole.filled else '0'
        return bn

    def calculate_reward(self):
        state_status = self.check_if_final_state()
        if state_status == "Win":
            return 10
        elif state_status == "Lose":
            return -10
        else:
            return 0

    def check_if_final_state(self):
        if not self.get_actions():
            return True, "Win" if self.board.num_pegs == 1 else True, "Lose"
        return False, None
