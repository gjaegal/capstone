import numpy as np
import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.position = (x, y)
        self.parent = parent
        self.g = 0  # 시작점부터 현재 노드까지의 실제 비용
        self.h = 0  # 현재 노드에서 목표점까지의 휴리스틱(추정) 비용
        self.f = 0  # 총 비용 f = g + h

    def __lt__(self, other):
        return self.f < other.f  # f값이 더 작은 노드를 우선 선택하도록 설정
    
    def __eq__(self, other):
        return self.position == other.position  # 좌표가 같으면 동일 노드로 간주


def h_dist(current_node, goal_node):
    # 맨해튼 거리(Manhattan distance)를 휴리스틱으로 사용
    return abs(current_node.x - goal_node.x) + abs(current_node.y - goal_node.y)


def astar(grid, start, end):
    # 시작점과 목표점을 노드로 초기화
    start_node = Node(start[0], start[1])
    goal_node = Node(end[0], end[1])

    open_list = []  # 탐색할 노드 목록 (우선순위 큐)
    closed_set = set()  # 이미 방문한 노드 저장
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 상하좌우 이동 방향 정의

    heapq.heappush(open_list, start_node)  # 시작 노드를 open_list에 추가

    while open_list:
        current_node = heapq.heappop(open_list)  # f값이 가장 낮은 노드 선택
        closed_set.add(current_node.position)  # 방문 처리

        # 목표 지점 도착 시 경로를 추적하여 반환
        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent if hasattr(current_node, 'parent') else None
            return path[::-1]  # 역순이므로 반전하여 반환

        # 인접 노드 탐색
        for move in moves:
            child_pos = (current_node.x + move[0], current_node.y + move[1])
            
            # grid 범위를 벗어나면 무시
            if (child_pos[0] < 0 or child_pos[0] >= grid.shape[0] or
                child_pos[1] < 0 or child_pos[1] >= grid.shape[1]):
                continue

            # 장애물(1)은 통과 불가
            if grid[child_pos[0]][child_pos[1]] == 1:
                continue

            # 이미 방문한 노드면 무시
            if child_pos in closed_set:
                continue

            # 자식 노드 생성 및 비용 계산
            child_node = Node(child_pos[0], child_pos[1], current_node)
            child_node.g = current_node.g + 1
            child_node.h = h_dist(child_node, goal_node)
            child_node.f = child_node.g + child_node.h

            # 기존 open_list에 동일한 노드가 더 효율적일 경우 skip
            skip = False
            for open_node in open_list:
                if child_node.position == open_node.position and child_node.g >= open_node.g:
                    skip = True
                    break

            # 새로운 노드 추가
            if not skip:
                heapq.heappush(open_list, child_node)

    return None  # 경로가 없는 경우 None 반환


def create_grid(obstacles=None, grid_size=(7, 7)):
    # 주어진 장애물 좌표를 기반으로 2D 격자(grid) 생성      >>> 이게 현재 임시 그리드인 상황 
    rows = grid_size[0]
    cols = grid_size[1]
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if (i, j) in obstacles:
                grid[i][j] = 1  # 장애물
            else:
                grid[i][j] = 0  # 이동 가능
    return np.array(grid)

def discretize(position, grid_x=0.025, grid_y=0.03):
    """연속 좌표를 격자 좌표로 변환"""
    x, y = position
    return (int(x // grid_x), int(y // grid_y))

if __name__ == "__main__":
    # 테스트용 코드: 7x7 grid에서 장애물 설정     >> 이것도 테스트용으로 장애물 설정
    obstacles = [(3,3), (3,4), (4,3), (4,4)]
    grid = create_grid(obstacles, grid_size=(20,10))
    print(grid)

    # 시작점과 목표점 설정     >> 이거도 시작점은 현재 카메라 좌표로, 목표점은 타겟 좌료로 변경
    start = (0, 0)
    end = (6, 6)


    # ROS 토픽(/current_xyz, /target_xyz)에서 받은 실시간 좌표 사용 >> 실제 할 때 사용
    start = (0, 0)
    end = (6, 6)

    # A* 알고리즘 실행
    path = astar(grid, start, end)
    print(path)

    # 경로를 grid에 표시 (3으로)
    for (x, y) in path:
        grid[x][y] = 3

    print(grid)
