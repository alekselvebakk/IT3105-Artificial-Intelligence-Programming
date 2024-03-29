import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation


class BoardVisualization:

    def __init__(self, table, interval, board_gif_name):
        self.table = table
        self.G = nx.Graph()
        self.peghole_colors = {0: '#0000ff', 1: '#ff0000', 2: '#000000', 3: '#FFCCCB', 4: '#606060'}  # 3 and 4 are used for enhancing when player 1 and 2 makes a move
        self.node_pos = []
        self.node_color = {}
        self.color_combo = []
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.create_board_graph()
        self.color_combo.append(dict(self.node_color).values())
        self.board_gif_name = board_gif_name

    def create_board_graph(self):
        # Information for the top node
        row_pos = 0
        col_pos = 0
        top_node = self.table[0][0]

        # Adds the first/top node
        self.add_node_with_properties(top_node, (col_pos, row_pos))

        # Function to recursively adds the other nodes
        self.add_down_neighbour_nodes(top_node, col_pos, row_pos)

    def add_neighbour_edges(self, peghole):
        for neighbour in peghole.neighbours.values():
            if neighbour in self.G.nodes:
                self.G.add_edge(peghole, neighbour)

    # adds the node with right position, color and edges to each of its existing neighbours
    def add_node_with_properties(self, peghole, pos):
        self.G.add_node(peghole, pos=pos)
        self.node_color[peghole] = self.peghole_colors.get(peghole.filled)
        self.add_neighbour_edges(peghole)

    def add_down_neighbour_nodes(self, peghole, row_pos, col_pos):
        down_neighbours = ["right", "down"]
        row = row_pos - 1
        for neighbour in peghole.neighbours:
            neighbour_peghole = peghole.neighbours[neighbour]
            if neighbour in down_neighbours and neighbour_peghole not in self.G.nodes:
                col = col_pos - 1 if neighbour == "down" else col_pos + 1
                self.add_node_with_properties(neighbour_peghole, (col, row))
                self.add_down_neighbour_nodes(neighbour_peghole, row, col)

    def draw_graph(self):
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, node_color=self.node_color.values(), pos=pos)
        plt.show()

    def show_graph_animation(self):
        self.ax.clear()
        anim = matplotlib.animation.FuncAnimation(self.fig,
                                                  self.animate,
                                                  frames=len(self.color_combo),
                                                  interval=self.interval,
                                                  repeat=True)
        anim.save(self.board_gif_name, writer='pillow')
        plt.show()

    def change_node_color(self, peghole):
        self.node_color[peghole] = self.peghole_colors.get(peghole.filled+2)
        self.update_color_combo()
        self.node_color[peghole] = self.peghole_colors.get(peghole.filled)
        self.update_color_combo()

    def update_color_combo(self):
        copy_dict = self.node_color.copy()
        colors = copy_dict.values()
        self.color_combo.append(colors)

    def animate(self, i):
        colors = self.color_combo[i]
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, node_color=colors, pos=pos, ax=self.ax)

    def reset_board(self):
        for peghole in self.node_color:
            self.node_color[peghole] = self.peghole_colors.get(peghole.filled)
        self.color_combo.clear()
        self.update_color_combo()
