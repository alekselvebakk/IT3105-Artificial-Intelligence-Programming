from Project1.SimWorld.DiamondBoard import DiamondBoard
from Project1.SimWorld.TriangleBoard import TriangleBoard
from Project1.SimWorld.BoardVisualization import BoardVisualization

class GameHandler:
    def __init__(self, board_type, size, empty, visualization=False):
        self.board = DiamondBoard(size, empty) if board_type == "diamond" else TriangleBoard(size, empty)
        self.visualization = visualization
        self.vis_graph = BoardVisualization(self.board) if visualization else None



    def get_actions(self):
        actions = []
        for row in self.board.table:
            for peghole in row:
                if peghole.filled:
                    actions += peghole.get_actions()
        return actions

    def perform_action(self, action):
        self.move_peg(action)
        if self.visualization: self.vis_graph.draw_graph()
        return self.calculate_reward(), self.get_board_state(), self.get_actions()

    def move_peg(self, action):
        start = self.board.table[int(action[0])][int(action[1])]
        end = self.board.table[int(action[2])][int(action[3])]

        start.remove_peg()
        end.add_peg()
        middle = self.remove_middle_peg(start, end)

        if self.visualization: self.update_node_colors([start, end, middle])

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

    def update_node_colors(self, peghole_list):
        for peghole in peghole_list:
            self.vis_graph.change_node_color(peghole)
