from Project2.SimWorld.Board import Board
from Project2.SimWorld.StateManager import StateManager

def main():
    sm = StateManager()
    b = Board(size=5)
    test = '21211221212221121211112222'
    sm.set_state(b, test)
    print(sm.state_is_final(b))


if __name__ == '__main__':
    main()
