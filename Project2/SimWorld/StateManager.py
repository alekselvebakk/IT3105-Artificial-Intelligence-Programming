from Project2.SimWorld.Board import Board

class StateManager():

    def __init__(self, size):
        self.board = Board(size)

    # TODO: GET_ACTIONS: do we use probabilities to find action? Or do we find them on our own?

    def perform_action(self, player, action):
        self.board[action[0]][action[1]].add_peg(player)

    def check_if_final_state(self):
        if self.check_player1_win(): return True, 'player1'
        if self.check_player2_win(): return True, 'player2'
        return False

    def check_player1_win(self):  # player1 = black, owns northwest and southeast (the sides in table)
        return False

    def check_player2_win(self):  # player2 = red, owns northeast and southwest (top and bottom in table)
        win = False
        nw_row = self.board[0]
        for peghole in nw_row:
            peghole = peghole.check_notup_neighbours('player2')

        return win
