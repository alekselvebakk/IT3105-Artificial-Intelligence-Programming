import pathlib
from shutil import copyfile
import numpy as np

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

    def single_game_between_2_nets(self, actor0, actor1, name, animate=False):
        self.state_manager.reset_board(self.board)
        if animate: self.board.start_visualisation(name+".gif")
        turn = 0
        while not self.state_manager.state_is_final(self.board):
            state = self.state_manager.get_state(self.board)
            if turn == 0:
                action = actor0.get_action(state, epsilon=0.2)
                turn = 1
            else:
                action = actor1.get_action(state, epsilon=0.2)
                turn = 0
            self.state_manager.perform_action(self.board, action)
        winner = self.state_manager.get_result(self.board)-1
        if animate:
            self.board.graph.show_graph_animation()
            self.board.stop_visualization()
        return winner #0 eller 1

    def series_between_2_nets(self, actor0, actor1, name, animate=False):
        player0_wins = 0
        player1_wins = 0
        for i in range(self.games_between_nets):
            animate_game = True if animate and i == 0 else False
            if i % 2 == 0:
                winner = self.single_game_between_2_nets(actor0, actor1, name, animate_game)
            else:
                winner = self.single_game_between_2_nets(actor1, actor0, name, animate_game)^1
            
            if winner == 0:
                player0_wins += 1
            else:
                player1_wins += 1
        return [player0_wins, player1_wins]

    def run_tournament(self, actors, show_games_between):
        self.number_of_wins = np.zeros((self.number_of_nets, self.number_of_nets))
        for i in range(len(actors)):
            for j in range(i+1, len(actors)):
                animate = True if str(i)+str(j) in show_games_between else False
                winner_array = self.series_between_2_nets(actors[i], actors[j], str(i)+str(j), animate)
                self.number_of_wins[i][j] += winner_array[0]
                self.number_of_wins[i][i] += winner_array[0]
                self.number_of_wins[j][i] += winner_array[1]
                self.number_of_wins[j][j] += winner_array[1]

    def calculate_win_rate(self):
        win_rate = self.number_of_wins/self.games_between_nets
        for i in range(self.number_of_nets):
            win_rate[i][i] = self.number_of_wins[i][i]/((self.number_of_nets-1)*self.games_between_nets)
        return win_rate

    def print_standings(self):
        print_table = self.matrix_for_print()
        
        s = [[str(e) for e in row] for row in print_table]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))

    
    def matrix_for_print(self):
        p = []
        win_rate = self.calculate_win_rate()
        for i in range(self.number_of_nets+1):
            line = []
            if i == 0:
                line = ['', 'Total win and win rate\t\t']
                for j in range(self.number_of_nets):
                    line.append(str(j)+'\t\t')
            else:
                space = '\t' #if len(str(int(self.number_of_wins[i-1][i-1]))) == 1 else '\t'

                line = [str(i-1), 'W: '+str(int(self.number_of_wins[i-1][i-1]))+space+'WR: '+str(int(win_rate[i-1][i-1]*100))+' %\t']
                for j in range(self.number_of_nets):
                    if i-1 == j:
                        line.append('-\t\t')
                    else:
                        space = '\t' #if len(str(int(self.number_of_wins[i-1][j]))) == 1 else '\t'
                        line.append('W: '+str(int(self.number_of_wins[i-1][j]))+space+'WR: '+str(int(win_rate[i-1][j]*100))+' %\t')
            p.append(line)
        return p


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