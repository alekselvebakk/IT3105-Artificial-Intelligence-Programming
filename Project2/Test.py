from SimWorld.Board import Board
from SimWorld.StateManager import StateManager

def main():
    sm = StateManager()
    b = Board(size=5)
    test = '12221211111222111211222202'
    sm.set_state(b, test)
    b.start_visualisation()
    print(sm.state_is_final(b))
    b.graph.draw_graph()


if __name__ == '__main__':
    main()
