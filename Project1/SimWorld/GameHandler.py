from Project1.SimWorld.Board import Board
from Project1.SimWorld.DiamondBoard import DiamondBoard
from Project1.SimWorld.TriangleBoard import TriangleBoard

class GameHandler():
    def __init__(self, board_type, size, empty):
        self.board = DiamondBoard(size, empty) if board_type == "diamond" else TriangleBoard(size, empty)

        # TODO: find out how A wants to ML to get states
        self.current_state = ""
        self.new_state = ""
        self.current_actions = []
        self.new_actions = []

    def get_actions(self):
        actions = []
        for peghole in self.board:
            if peghole.filled:
                actions += peghole.get_actions()
        return actions

    def move_peg(self, start_pos, end_pos): # TODO: How will the actions be represented?
        self.board[start_pos[0]][start_pos[1]].remove_peg()
        self.board[end_pos[0]][end_pos[1]].add_peg()
        self.remove_middle_peg(start_pos, end_pos)

    # TODO: How to remove the middle peg?
    def remove_middle_peg(self, start_post, end_pos):
        return None


    def check_if_final_state(self):
        if self.get_actions() == []:  # or only one peg filled
            return True
        return False
