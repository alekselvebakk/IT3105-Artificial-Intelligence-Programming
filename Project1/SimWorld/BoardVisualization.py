import networkx as nx
import matplotlib.pyplot as plt
from Project1.SimWorld.DiamondBoard import DiamondBoard
from Project1.SimWorld.TriangleBoard import TriangleBoard


class BoardVisualization:

    def __init__(self, board):
        self.board = board
        self.G = nx.Graph()

    def create_board_graph(self):
        for row in self.board.table:
            for peghole in row:
                self.G.add_node(peghole)
                self.add_neighbours(peghole)

    def add_neighbours(self, peghole):
        for neighbour in peghole.neighbours.values():
            if neighbour in self.G.nodes:
                self.G.add_edge(peghole, neighbour)


def main():
    db = DiamondBoard(6, [[2, 2]])
    tb = TriangleBoard(4, [[2, 2]])
    bv = BoardVisualization(tb)
    bv.create_board_graph()
    print(bv.G.number_of_nodes())
    print(bv.G.number_of_edges())
    print(bv.G.nodes)
    nx.draw(bv.G)
    plt.show()


if __name__ == '__main__':
    main()