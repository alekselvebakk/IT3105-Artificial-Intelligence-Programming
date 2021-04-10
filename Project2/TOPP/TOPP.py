

class TOPP:
    def __init__(self, number_of_nets, games_between_nets, state_manager, board):
        self.number_of_nets = number_of_nets
        self.number_of_wins = [0]*number_of_nets
        self.games_between_nets = games_between_nets
        self.state_manager = state_manager
        self.board = board
        

    def single_game_between_2_nets(self, actor0, actor1):
        self.state_manager.reset_board(self.board)
        turn = 0
        while not self.state_manager.state_is_final(self.board):
            state = self.state_manager.get_state(self.board)
            if turn == 0:
                action = actor0.get_action(state)
                turn = 1
            else:
                action = actor1.get_action(state)
                turn = 0
            self.state_manager.perform_action(self.board,action)
        winner = self.state_manager.get_result(self.board)-1
        return winner #0 eller 1

    def series_between_2_nets(self, actor0, actor1):
        player0_wins = 0
        player1_wins = 0
        for i in range(self.games_between_nets):
            if i % 2 == 0:
                winner = self.single_game_between_2_nets(actor0, actor1)
            else:
                winner = self.single_game_between_2_nets(actor1, actor0)^1
            
            if winner == 0:
                player0_wins += 1
            else:
                player1_wins += 1
        return [player0_wins, player1_wins]

    def run_tournament(self, actors):
        self.number_of_wins = [0]*self.number_of_nets

        for i in range(len(actors)):
            for j in range(i+1,len(actors)):
                winner_array = self.series_between_2_nets(actors[i], actors[j])
                self.number_of_wins[i] += winner_array[0]
                self.number_of_wins[j] += winner_array[1]

        standings = self.number_of_wins 
        return standings
