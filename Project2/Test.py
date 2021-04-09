from SimWorld.Board import Board
from SimWorld.StateManager import StateManager

def main():
    sm = StateManager()
    b = Board(size=5)
    test = '21211211122212121221221211'
    sm.set_state(b, test)
    print(sm.state_is_final(b))


if __name__ == '__main__':
    main()
