import numpy as np
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0

        self.neighbor_nodes = []

        def add_neighbor(self, grid):
            columns, rows = grid.shape
            if self.x < columns - 1:
                self.neighbor_nodes.append(grid[self.x + 1][self.y])
            if self.x > 0:
                self.neighbor_nodes.append(grid[self.x - 1][self.y])
            if self.y < rows - 1:
                self.neighbor_nodes.append(grid[self.x][self.y + 1])
            if self.y > 0:
                self.neighbor_nodes.append(grid[self.x][self.y - 1])
class Astar:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape

    @staticmethod
    def h_dist(current_node, goal):
        return abs(current_node.x - goal.x) + abs(current_node.y - goal.y)
    
    def get_neighbors(self, node):
        for i in range(self.rows):
            for j in range(self.cols):
                self.grid[i][j].add_neighbor(self.grid)

    
def create_grid():
    rows = 7
    cols = 7
    obstacles = [(3,3), (3,4), (4,3), (4,4)]
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if (i, j) in obstacles:
                grid[i][j] = 1
            else:
                grid[i][j] = Node(i, j)
    return grid
