from Project2.SimWorld.Board import Board
from Project2.SimWorld.BoardVisualization import Board
from Project2.SimWorld.StateManager import StateManager

def main():
    sm = StateManager(4, True)
    sm.perform_action(1, [1, 2])
    sm.perform_action(1, [0, 0])
    sm.perform_action(1, [2, 2])
    sm.perform_action(2, [0, 1])
    sm.perform_action(1, [3, 1])
    sm.perform_action(2, [0, 2])
    sm.perform_action(2, [0, 3])
    sm.perform_action(2, [3, 2])
    sm.perform_action(2, [1, 0])
    sm.graph.draw_graph()
    print(sm.check_if_final_state())


if __name__ == '__main__':
    main()
