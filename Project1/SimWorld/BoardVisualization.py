import networkx as nx
import matplotlib.pyplot as plt
from Project1.SimWorld.DiamondBoard import DiamondBoard
from Project1.SimWorld.TriangleBoard import TriangleBoard


class BoardVisualization:

    def __init__(self, board):
        self.board = board
        self.G = nx.Graph()
        self.node_pos = []
        self.node_color = []
        self.board_type = 'triangle' if board.__class__.__name__ == 'TriangleBoard' else 'diamond'

    def create_board_graph(self):
        # Information for the top node
        row_pos = self.board.size
        col_pos = self.board.size
        top_node = self.board.table[0][0]

        self.add_node_with_properties(top_node, (col_pos, row_pos))  # Adds the first/top noe
        self.add_down_neighbour_nodes(top_node, col_pos, row_pos)  # Function to recursively adds the other nodes



    def add_neighbour_edges(self, peghole):
        for neighbour in peghole.neighbours.values():
            if neighbour in self.G.nodes:
                self.G.add_edge(peghole, neighbour)

    # adds the node with right position, color and edges to each of its existing neighbours
    def add_node_with_properties(self, peghole, pos):
        self.G.add_node(peghole, pos=pos)
        self.node_color.append('blue') if peghole.filled else self.node_color.append('black')
        self.add_neighbour_edges(peghole)

    def add_down_neighbour_nodes(self, peghole, row_pos, col_pos):
        down_neighbours = ["right", "down"] if self.board_type == 'diamond' else ["down", "rightdown"]
        row = row_pos - 1
        print("PEGHOLE", peghole)
        for neighbour in peghole.neighbours:
            neighbour_peghole = peghole.neighbours[neighbour]
            print(neighbour)
            if neighbour in down_neighbours and neighbour_peghole not in self.G.nodes:
                col = col_pos-1 if neighbour == "down" else col_pos+1
                self.add_node_with_properties(neighbour_peghole, (col, row))
                self.add_down_neighbour_nodes(neighbour_peghole, row, col)

def main():
    db = DiamondBoard(5, [[0, 0], [0, 2]])
    tb = TriangleBoard(8, [[0, 0], [1,1]])
    bv = BoardVisualization(db)
    bv.create_board_graph()
    pos = nx.get_node_attributes(bv.G, 'pos')
    nx.draw(bv.G, node_color=bv.node_color, pos=pos)
    plt.show()


if __name__ == '__main__':
    main()