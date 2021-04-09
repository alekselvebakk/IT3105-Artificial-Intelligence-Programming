from Project2.SimWorld.Board import Board
from Project2.SimWorld.StateManager import StateManager

def main():
    sm = StateManager()
    b = Board(size=5)
    test = '22122211120212112211212121'
    sm.set_state(b, test)
    print(sm.get_result(b))


if __name__ == '__main__':
    main()
