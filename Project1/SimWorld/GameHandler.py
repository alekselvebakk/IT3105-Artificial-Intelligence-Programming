from Project1.SimWorld.Board import Board
from Project1.SimWorld.DiamondBoard import DiamondBoard
from Project1.SimWorld.TriangleBoard import TriangleBoard

class GameHandler:
    def __init__(self, board_type, size, empty):
        self.board = DiamondBoard(size, empty) if board_type == "diamond" else TriangleBoard(size, empty)

        # TODO: find out how A wants to ML to get states
        self.current_state = ""
        self.new_state = ""
        self.current_actions = []
        self.new_actions = []

    def get_actions(self):
        actions = []
        for peghole in self.board.board:
            if peghole.filled:
                actions += peghole.get_actions()
        return actions

    def move_peg(self, action):  # Action can be sent as string instead?
        number = str(action) if len(action) == 4 else "0"+str(action)
        self.board.table[int(number[0])][int(number[1])].remove_peg()
        self.board.table[int(number[2])][int(number[3])].add_peg()
        self.remove_middle_peg(number)

    def remove_middle_peg(self, action):
        self.board.num_pegs -= 1
        start = self.board.table[int(action[0])][int(action[1])]
        end = self.board.table[int(action[2])][int(action[3])]
        direction = start.find_neighbour_direction(end.row, end.column)
        middle = start.neighbours[direction]
        middle.remove_peg()


    def get_board_state(self):
        bin = ''
        for row in range(self.board.size):
            for peghole in self.board.table[row]:
                print(peghole.filled)
                bin += '1' if peghole.filled else '0'
        print(bin)
        return int(bin, 2)


    def check_if_final_state(self):
        if self.get_actions() == [] or self.board.num_pegs == 1:  # or only one peg filled
            return True
        return False
