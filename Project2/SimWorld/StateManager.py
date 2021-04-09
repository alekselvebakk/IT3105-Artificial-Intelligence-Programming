class StateManager:

    def get_state(self, board):
        return board.get_board_state()

    def set_state(self, board, state):
        board.set_player(state[0])
        board.update_board(state[1:])

    def change_player(self, board):  # may not be needed no more
        player = 1 if board.player == 2 else 2
        board.set_player(player)

    def get_actions(self, board):
        actions = [0]*(board.size*board.size)
        for row in board.table:
            for peghole in row:
                if peghole.filled == 0:
                    action = str(peghole.row)+str(peghole.column)
                else: action = False
                actions[peghole.row*board.size + peghole.column] = action
        return actions

    def perform_action(self, board, action):
        peghole = board.table[action[1]][action[2]]
        peghole.add_peg(action[0])
        self.change_player(board)

        if board.visualize: board.graph.change_node_color(peghole)

    def state_is_final(self, board):
        return self.find_winner != 0

    def find_winner(self, board):
        if self.check_player1_win(board): return 1
        if self.check_player2_win(board): return 2
        return 0

    def check_player1_win(self, board):  # player1 = red, owns northeast and southwest (top and bottom in table)
        visited_nodes = []
        furthest = 0
        ne_row = [peghole for peghole in board.table[0] if peghole.filled == 1]
        for peghole in ne_row:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == board.size-1: break

        return furthest == board.size-1

    def check_player2_win(self, board):  # player2 = black, owns northwest and southeast (the sides in table)
        visited_nodes = []
        furthest = 0
        nw_col = [peghole for peghole in self.get_north_west_col(board) if peghole.filled == 2]
        for peghole in nw_col:
            if peghole not in visited_nodes:
                visited_nodes.append(peghole)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_col(peghole, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest
            if furthest == board.size-1: break

        return furthest == board.size-1

    def get_north_west_col(self, board):
        nw = []
        for i in range(len(board.table)):
            nw.append(board.table[i][0])
        return nw

    def check_furthest_neighbour_row(self, peghole, visited_nodes):
        furthest = peghole.row
        for neighbour in peghole.get_players_neighbours(1):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_row(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes

    def check_furthest_neighbour_col(self, peghole, visited_nodes):
        furthest = peghole.column
        for neighbour in peghole.get_players_neighbours(2):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                possible_furthest, visited_nodes = self.check_furthest_neighbour_col(neighbour, visited_nodes)
                furthest = possible_furthest if possible_furthest > furthest else furthest

        return furthest, visited_nodes

    def show_animation(self, board):
        if board.visualize: board.graph.show_graph_animation()
        else: print("This board do not have an animated gif of the game.")

    def reset_board(self, board):
        board.reset_board()

        if board.visualize: board.graph.reset_board()




