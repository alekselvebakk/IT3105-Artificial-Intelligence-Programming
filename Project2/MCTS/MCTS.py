class MCTS():
    def __init__(GameHandler, number_search_games):
        self.GameHandler = GameHandler
        state = GameHandler.get_state()
        self.tree = MCTS_Tree(state)
        self.number_search_games = number_search_games

    def perform_search_game():
        possible_actions = GameHandler.get_actions()
        action = self.tree.get_action()
