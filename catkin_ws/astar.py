import numpy as np
import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.position = (x, y)
        self.parent = parent
        self.g = 0 # Start ~ Current cost
        self.h = 0 # Current ~ Goal cost
        self.f = 0 # Total cost

    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.position == other.position


def h_dist(current_node, goal_node):
    return abs(current_node.x - goal_node.x) + abs(current_node.y - goal_node.y)


def astar(grid, start, end):
    start_node = Node(start[0], start[1])
    goal_node = Node(end[0], end[1])

    open_list = []
    closed_set = set()
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent if hasattr(current_node, 'parent') else None

            return path[::-1]
        for move in moves:
            child_pos = (current_node.x + move[0], current_node.y + move[1])
            if (child_pos[0] < 0 or child_pos[0] >= grid.shape[0] or
                child_pos[1] < 0 or child_pos[1] >= grid.shape[1]):
                continue
            if grid[child_pos[0]][child_pos[1]] == 1:
                continue
            if child_pos in closed_set:
                continue

            child_node = Node(child_pos[0], child_pos[1], current_node)
            child_node.g = current_node.g + 1
            child_node.h = h_dist(child_node, goal_node)
            child_node.f = child_node.g + child_node.h

            skip = False
            for open_node in open_list:
                if child_node.position == open_node.position and child_node.g >= open_node.g:
                    skip = True
                    break
            if not skip:
                heapq.heappush(open_list, child_node)
    return None


    
def create_grid(obstacles=None):
    rows = 7
    cols = 7
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if (i, j) in obstacles:
                grid[i][j] = 1
            else:
                grid[i][j] = 0
    return np.array(grid)

def discretize(position, grid_size=1.0):
    """Convert continuous position to discrete grid coordinates."""
    x, y = position
    return (int(x // grid_size), int(y // grid_size))

if __name__ == "__main__":
    obstacles = [(3,3), (3,4), (4,3), (4,4)]
    grid = create_grid(obstacles)
    print(grid)
    start = (0, 0)
    end = (6, 6)

    path = astar(grid, start, end)
    print(path)

    for (x, y) in path:
        grid[x][y] = 3

    print(grid)
    

    