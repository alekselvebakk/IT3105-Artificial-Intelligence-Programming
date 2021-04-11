import pathlib
from shutil import copyfile

class TOPP:

    def __init__(self, number_of_nets, games_between_nets, state_manager, board, tournament_id, save_actors, savefolder):
        self.number_of_nets = number_of_nets
        self.number_of_wins = [0]*number_of_nets
        self.games_between_nets = games_between_nets
        self.state_manager = state_manager
        self.board = board
        self.tournament_id = tournament_id
        self.save_actors = save_actors
        self.folder_name = savefolder + str(self.tournament_id)

    def single_game_between_2_nets(self, actor0, actor1, vis=False):
        self.state_manager.reset_board(self.board)
        if vis: self.board.start_visualisation()
        turn = 0
        while not self.state_manager.state_is_final(self.board):
            state = self.state_manager.get_state(self.board)
            if turn == 0:
                action = actor0.get_action(state)
                turn = 1
            else:
                action = actor1.get_action(state)
                turn = 0
            self.state_manager.perform_action(self.board, action)
        winner = self.state_manager.get_result(self.board)-1
        if vis:
            self.board.graph.show_graph_animation()
            self.board.stop_visualization()
        return winner #0 eller 1

    def series_between_2_nets(self, actor0, actor1, vis=False):
        player0_wins = 0
        player1_wins = 0
        for i in range(self.games_between_nets):
            ja = True if vis and i == 5 else False
            if i % 2 == 0:
                winner = self.single_game_between_2_nets(actor0, actor1, ja)
            else:
                winner = self.single_game_between_2_nets(actor1, actor0, ja)^1
            
            if winner == 0:
                player0_wins += 1
            else:
                player1_wins += 1
        return [player0_wins, player1_wins]

    def run_tournament(self, actors):
        self.number_of_wins = [0]*self.number_of_nets

        for i in range(len(actors)):
            for j in range(i+1,len(actors)):
                vis = True if i == 1 and j == 4 else False
                winner_array = self.series_between_2_nets(actors[i], actors[j], vis)
                self.number_of_wins[i] += winner_array[0]
                self.number_of_wins[j] += winner_array[1]

        standings = self.number_of_wins 
        return standings

    def save_net(self, actor_critic, game_number, number_actual_games):
        if self.save_actors:
            games_between_saving = int(number_actual_games / (self.number_of_nets - 1))
            final_modulo = game_number / games_between_saving == (self.number_of_nets - 1)
            saving_time = game_number % games_between_saving == 0

            finished_training = game_number == number_actual_games

            if (saving_time and not final_modulo) or finished_training:
                # Save net
                neural_net_name = self.folder_name + "/ANET" + str(game_number)
                actor_critic.save_net(neural_net_name)

    def save_config(self, config_path):
        if self.save_actors:
            copyfile(config_path, self.folder_name + "/config.ini")