from SimWorld.DiamondBoard import DiamondBoard
from SimWorld.TriangleBoard import TriangleBoard
from SimWorld.BoardVisualization import BoardVisualization
import time

class GameHandler:
    def __init__(   self, 
                    board_type, 
                    size, 
                    empty, 
                    fps, 
                    visualization=False, 
                    board_gif_name=None):
        self.board = DiamondBoard(size, empty) if board_type == "diamond" else TriangleBoard(size, empty)
        self.visualization = visualization
        self.vis_graph = BoardVisualization(self.board, 
                                            fps, 
                                            board_gif_name) if visualization else None


    def get_actions(self):
        actions = []
        for row in self.board.table:
            for peghole in row:
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
        middle = self.remove_middle_peg(start, end)

        if self.visualization: self.update_node_colors([start, middle, end])

    def remove_middle_peg(self, start, end):
        self.board.num_pegs -= 1
        direction = start.find_neighbour_direction(end.row, end.column)
        middle = start.neighbours[direction]
        middle.remove_peg()
        return middle

    def get_board_state(self):
        bn = ''
        for row in range(self.board.size):
            for peghole in self.board.table[row]:
                bn += '1' if peghole.filled else '0'
        return bn

    def calculate_reward(self):
        winning_reward = 15
        losing_reward = -15
        state_status = self.check_if_final_state()
        if state_status == "Win":
            return winning_reward
        elif state_status == "Lose":
            return losing_reward
        
        """
        ## Begynt å tenkte på reward-opplegg
        num_pegs = self.board.num_pegs
        empty_holes = 
        total_peg_holes 
        """

        return 1

    def check_if_final_state(self):
        if not self.get_actions():
            return True, "Win" if self.board.num_pegs == 1 else True, "Lose"
        return False, None

    def visualize_board(self):
        if self.visualization: self.vis_graph.show_graph_animation()

    def update_node_colors(self, peghole_list):
        color = ['yellow', 'red', 'yellow']
        for i in range(len(peghole_list)):
            self.vis_graph.change_node_color(peghole_list[i], color[i])
        self.vis_graph.update_color_combo()

    def reset_board(self):
        self.board.reset_board()
        if self.visualization: self.vis_graph.reset_board()
